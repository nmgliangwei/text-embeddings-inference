import torch
import torch_npu
import json
from pathlib import Path
from torch import nn
from typing import List, Union, Optional
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from opentelemetry import trace
from text_embeddings_server.models import Model
from text_embeddings_server.models.types import (
    FlashBatch,
    PaddedBatch,
    Embedding,
    Score,
)
from text_embeddings_server.models.flash_qwen_base import (
    load_weight,
    apply_rotary_pos_emb_npu,
    FlashQwenModel,
)
from text_embeddings_server.utils.flash_attn import attention


tracer = trace.get_tracer(__name__)


class Qwen2Attention:
    def __init__(
        self,
        model_path,
        weight_map,
        prefix,
        device,
        dtype,
        config: Qwen2Config,
        layer_idx: Optional[int] = None,
    ):
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.softmax_scale = self.head_dim**-0.5
        self.q_proj_weight = load_weight(
            model_path,
            weight_map,
            prefix,
            f"layers.{layer_idx}.self_attn.q_proj.weight",
            dtype,
            device,
        )
        self.q_proj_bias = load_weight(
            model_path,
            weight_map,
            prefix,
            f"layers.{layer_idx}.self_attn.q_proj.bias",
            dtype,
            device,
        )
        self.k_proj_weight = load_weight(
            model_path,
            weight_map,
            prefix,
            f"layers.{layer_idx}.self_attn.k_proj.weight",
            dtype,
            device,
        )
        self.k_proj_bias = load_weight(
            model_path,
            weight_map,
            prefix,
            f"layers.{layer_idx}.self_attn.k_proj.bias",
            dtype,
            device,
        )
        self.v_proj_weight = load_weight(
            model_path,
            weight_map,
            prefix,
            f"layers.{layer_idx}.self_attn.v_proj.weight",
            dtype,
            device,
        )
        self.v_proj_bias = load_weight(
            model_path,
            weight_map,
            prefix,
            f"layers.{layer_idx}.self_attn.v_proj.bias",
            dtype,
            device,
        )
        self.o_proj_weight = load_weight(
            model_path,
            weight_map,
            prefix,
            f"layers.{layer_idx}.self_attn.o_proj.weight",
            dtype,
            device,
        )
        # Q/K/V projections are forced into grouped matmul for peak throughput.
        self._grouped_qkv_weights = [
            self.q_proj_weight.transpose(0, 1).contiguous(),
            self.k_proj_weight.transpose(0, 1).contiguous(),
            self.v_proj_weight.transpose(0, 1).contiguous(),
        ]
        self._grouped_qkv_bias = [
            self.q_proj_bias,
            self.k_proj_bias,
            self.v_proj_bias,
        ]
        self.is_310p = torch.npu.get_device_name().lower().find("310p") != -1

    def _project_qkv(self, hidden_states):
        hidden_states = hidden_states.contiguous()
        qkv_outputs = torch_npu.npu_grouped_matmul(
            [hidden_states, hidden_states, hidden_states],
            self._grouped_qkv_weights,
            bias=self._grouped_qkv_bias,
            group_type=-1,
            split_item=0,
        )
        return qkv_outputs[0], qkv_outputs[1], qkv_outputs[2]

    def forward(
        self, hidden_states, position_embeddings, cu_seqlens, max_s, attn_mask=None
    ):
        input_shape = hidden_states.shape[:-1]
        if self.is_310p:
            q = torch_npu.npu_linear(
                hidden_states, self.q_proj_weight, self.q_proj_bias
            ).view(*input_shape, self.num_heads, self.head_dim)
            k = torch_npu.npu_linear(
                hidden_states, self.k_proj_weight, self.k_proj_bias
            ).view(*input_shape, self.num_key_value_heads, self.head_dim)
            v = torch_npu.npu_linear(
                hidden_states, self.v_proj_weight, self.v_proj_bias
            ).view(*input_shape, self.num_key_value_heads, self.head_dim)
        else:
            q_proj, k_proj, v_proj = self._project_qkv(hidden_states)
            q = q_proj.reshape(*input_shape, self.num_heads, self.head_dim)
            k = k_proj.reshape(*input_shape, self.num_key_value_heads, self.head_dim)
            v = v_proj.reshape(*input_shape, self.num_key_value_heads, self.head_dim)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_npu(q, k, cos, sin)

        attn_output = attention(
            q,
            k,
            v,
            self.num_heads,
            None,
            cu_seqlens,
            max_s,
            self.softmax_scale,
            is_causal=False,
            attn_mask=attn_mask,
            num_key_value_heads=self.num_key_value_heads,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = torch_npu.npu_linear(attn_output, self.o_proj_weight)

        return attn_output


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, model_path, weight_map, prefix, device, dtype):
        super().__init__()
        self.score_weight = load_weight(
            model_path,
            weight_map,
            "",
            "score.weight",
            dtype,
            device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CLSPool has already been applied in `pooling`
        x = torch_npu.npu_linear(x, self.score_weight)
        return x


class ForSequenceClassification(nn.Module):
    def __init__(
        self, model_path, weight_map, prefix, device, dtype, config: Qwen2Config
    ):
        super().__init__()
        self.config = config

        self.model = FlashQwenModel(
            model_path,
            weight_map,
            prefix,
            device,
            dtype,
            config,
            attention_cls=Qwen2Attention,
        )
        self.classifier = ClassificationHead(
            model_path, weight_map, prefix, device, dtype
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

        sequence_output = self.model.forward(
            input_ids, position_ids, cu_seqlens, max_s, mask, attn_mask
        )
        last_token_indices = cu_seqlens[1:] - 1
        hidden_states = torch.index_select(
            sequence_output.last_hidden_state,
            0,
            last_token_indices.to(sequence_output.last_hidden_state.device),
        )
        logits = self.classifier(hidden_states)
        return logits


class FlashQwen2(Model):
    def __init__(
        self,
        model_path: Path,
        device: torch.device,
        dtype: torch.dtype,
        pool: str = "lasttoken",
        trust_remote: bool = False,
    ):
        config = Qwen2Config.from_pretrained(model_path, trust_remote_code=trust_remote)
        if hasattr(config, "max_seq_length"):
            self.max_input_length = config.max_seq_length
        else:
            self.max_input_length = config.max_position_embeddings

        index_file = model_path / "model.safetensors.index.json"
        if index_file.exists():
            with open(index_file, "r") as f:
                index_data = json.load(f)
            weight_map = index_data["weight_map"]
        else:
            weight_map = None

        if config.architectures[0].endswith("Classification"):
            model = ForSequenceClassification(
                model_path, weight_map, "model", device, dtype, config
            )
        else:
            model = FlashQwenModel(
                model_path,
                weight_map,
                "model",
                device,
                dtype,
                config,
                attention_cls=Qwen2Attention,
            )

        self.hidden_size = config.hidden_size
        self.device = device
        self.dtype = dtype

        super(FlashQwen2, self).__init__(model=model, dtype=dtype, device=device)

    @property
    def batch_type(self) -> Union[FlashBatch, PaddedBatch]:
        # for hpu devices, we use PaddedBatch as we do not have real varlen fwd yet
        return FlashBatch if self.device.type != "hpu" else PaddedBatch

    @tracer.start_as_current_span("embed")
    def embed(self, batch: Union[FlashBatch, PaddedBatch]) -> List[Embedding]:
        if not isinstance(batch, FlashBatch):
            raise ValueError("only FlashBatch is supported")

        cu_seqlens = batch.cu_seqlens
        mask = None
        attn_mask = None
        max_input_lens = batch.max_s

        output = self.model.forward(
            input_ids=batch.input_ids,
            position_ids=batch.position_ids,
            cu_seqlens=cu_seqlens,
            max_s=max_input_lens,
            mask=mask,
            attn_mask=attn_mask,
        )

        last_token_indices = cu_seqlens[1:] - 1
        hidden_states = torch.index_select(
            output.last_hidden_state,
            0,
            last_token_indices.to(output.last_hidden_state.device),
        )
        cpu_results = hidden_states.view(-1).tolist()

        return [
            Embedding(
                values=cpu_results[i * self.hidden_size : (i + 1) * self.hidden_size]
            )
            for i in range(len(batch))
        ]

    @tracer.start_as_current_span("predict")
    def predict(self, batch: Union[FlashBatch, PaddedBatch]) -> List[Score]:
        if not isinstance(batch, FlashBatch):
            raise ValueError("only FlashBatch is supported")

        cu_seqlens = batch.cu_seqlens
        mask = None
        attn_mask = None
        max_input_lens = batch.max_s

        logits = self.model.forward(
            input_ids=batch.input_ids,
            position_ids=batch.position_ids,
            cu_seqlens=cu_seqlens,
            max_s=max_input_lens,
            mask=mask,
            attn_mask=attn_mask,
        )

        all_scores = logits.tolist()
        return [Score(values=scores) for scores in all_scores]
