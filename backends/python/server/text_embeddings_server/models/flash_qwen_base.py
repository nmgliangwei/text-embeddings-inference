import torch
import torch_npu
from torch import nn
from typing import Optional
from safetensors import safe_open
from transformers.modeling_outputs import BaseModelOutputWithPast


def load_weight(model_path, weight_map, prefix, name, dtype, device):
    prefix = prefix + "." if prefix else ""
    name = prefix + name

    if weight_map is None:
        with safe_open(f"{model_path}/model.safetensors", framework="pt") as f:
            return f.get_tensor(name).to(dtype).to(device)
    else:
        target_file = weight_map[name]
        with safe_open(f"{model_path}/{target_file}", framework="pt") as f:
            return f.get_tensor(name).to(dtype).to(device)


def apply_rotary_pos_emb_npu(q, k, cos, sin):
    q_is_tnd = q.dim() == 3
    k_is_tnd = k.dim() == 3
    if q_is_tnd:
        q = q.unsqueeze(0)
    if k_is_tnd:
        k = k.unsqueeze(0)

    head_dim = q.shape[-1]
    cos = cos.reshape(1, -1, 1, head_dim).to(device=q.device, dtype=q.dtype)
    sin = sin.reshape(1, -1, 1, head_dim).to(device=q.device, dtype=q.dtype)

    q = torch_npu.npu_rotary_mul(q, cos, sin)
    k = torch_npu.npu_rotary_mul(k, cos, sin)

    if q_is_tnd:
        q = q.squeeze(0)
    if k_is_tnd:
        k = k.squeeze(0)
    return q, k


def compute_default_rope_parameters(config, device):
    base = config.rope_theta
    partial_rotary_factor = (
        config.partial_rotary_factor
        if hasattr(config, "partial_rotary_factor")
        else 1.0
    )
    head_dim = (
        getattr(config, "head_dim", None)
        or config.hidden_size // config.num_attention_heads
    )
    dim = int(head_dim * partial_rotary_factor)
    attention_factor = 1.0

    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, dim, 2, dtype=torch.int64).to(
                device=device, dtype=torch.float
            )
            / dim
        )
    )
    return inv_freq, attention_factor


class QwenRMSNorm:
    def __init__(
        self,
        model_path,
        weight_map,
        prefix,
        name,
        device,
        dtype,
        eps=1e-6,
    ):
        self.weight = load_weight(model_path, weight_map, prefix, name, dtype, device)
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        if hidden_states.device.type == "hpu":
            from habana_frameworks.torch.hpex.normalization import (
                FusedRMSNorm as FusedRMSNorm,
            )

            hidden_states = FusedRMSNorm.apply(
                hidden_states, self.weight, self.variance_epsilon
            )
            return hidden_states
        elif hidden_states.device.type == "npu":
            return torch_npu.npu_rms_norm(hidden_states,
                                          self.weight, epsilon=self.variance_epsilon)[0]
        else:
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(
                variance + self.variance_epsilon
            )
            return self.weight * hidden_states.to(input_dtype)


class QwenMLP:
    def __init__(
        self,
        model_path,
        weight_map,
        prefix,
        device,
        dtype,
        layer_idx: Optional[int] = None,
    ):
        self.gate_proj_weight = load_weight(
            model_path,
            weight_map,
            prefix,
            f"layers.{layer_idx}.mlp.gate_proj.weight",
            dtype,
            device,
        )
        self.up_proj_weight = load_weight(
            model_path,
            weight_map,
            prefix,
            f"layers.{layer_idx}.mlp.up_proj.weight",
            dtype,
            device,
        )
        self.gate_up_proj_weight = torch.cat([self.gate_proj_weight, self.up_proj_weight], dim=0)
        self.down_proj_weight = load_weight(
            model_path,
            weight_map,
            prefix,
            f"layers.{layer_idx}.mlp.down_proj.weight",
            dtype,
            device,
        )

    def forward(self, hidden_state):
        gate_up_states = torch_npu.npu_linear(
            hidden_state,
            self.gate_up_proj_weight,
        )
        hidden_states = torch_npu.npu_swiglu(gate_up_states, dim=-1)

        return torch_npu.npu_linear(
            hidden_states,
            self.down_proj_weight,
        )


class QwenDecoderLayer:
    def __init__(
        self,
        model_path,
        weight_map,
        prefix,
        device,
        dtype,
        config,
        layer_idx: Optional[int] = None,
        attention_cls=None,
    ):
        self.config = config
        self.attention = attention_cls(
            model_path, weight_map, prefix, device, dtype, config, layer_idx
        )
        self.mlp = QwenMLP(model_path, weight_map, prefix, device, dtype, layer_idx)
        self.input_layernorm = QwenRMSNorm(
            model_path,
            weight_map,
            prefix,
            f"layers.{layer_idx}.input_layernorm.weight",
            device,
            dtype,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = QwenRMSNorm(
            model_path,
            weight_map,
            prefix,
            f"layers.{layer_idx}.post_attention_layernorm.weight",
            device,
            dtype,
            eps=config.rms_norm_eps,
        )

    def forward(
        self, hidden_states, residual, position_embeddings, cu_seqlens, max_s, attn_mask=None
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm.forward(hidden_states)
        else:
            hidden_states, _, residual = torch_npu.npu_add_rms_norm(hidden_states, residual, self.input_layernorm.weight, epsilon=self.config.rms_norm_eps)
        hidden_states = self.attention.forward(
            hidden_states, position_embeddings, cu_seqlens, max_s, attn_mask
        )

        hidden_states, _, residual = torch_npu.npu_add_rms_norm(hidden_states, residual, self.post_attention_layernorm.weight, epsilon=self.config.rms_norm_eps)
        hidden_states = self.mlp.forward(hidden_states)
        return hidden_states, residual


class QwenRotaryEmbedding(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        inv_freq, self.attention_scaling = compute_default_rope_parameters(
            config, device
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, position_ids):
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class FlashQwenModel:
    def __init__(self, model_path, weight_map, prefix, device, dtype, config, attention_cls):
        self.word_embeddings_weight = load_weight(
            model_path,
            weight_map,
            prefix,
            "embed_tokens.weight",
            dtype,
            device,
        )
        self.config = config
        self.layers = [
            QwenDecoderLayer(
                model_path,
                weight_map,
                prefix,
                device,
                dtype,
                config,
                layer_idx,
                attention_cls=attention_cls,
            )
            for layer_idx in range(config.num_hidden_layers)
        ]
        self.rotary_emb = QwenRotaryEmbedding(config=config, device=device)
        self.norm = QwenRMSNorm(
            model_path,
            weight_map,
            prefix,
            f"norm.weight",
            device,
            dtype,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        input_ids,
        position_ids,
        cu_seqlens,
        max_s,
        mask=None,
        attn_mask=None,
    ):
        inputs_embeds = nn.functional.embedding(input_ids, self.word_embeddings_weight)
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer.forward(
                hidden_states, residual, position_embeddings, cu_seqlens, max_s, attn_mask
            )
        hidden_states, _, _ = torch_npu.npu_add_rms_norm(hidden_states, residual, self.norm.weight, epsilon=self.config.rms_norm_eps)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)
