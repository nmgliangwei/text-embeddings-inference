import os
import torch
import torch_npu
import json
import itertools
from pathlib import Path
from torch import nn
from typing import List, Union, Optional
from transformers.models.qwen3 import Qwen3Config
from transformers import AutoTokenizer
from opentelemetry import trace
from text_embeddings_server.models import Model
from text_embeddings_server.models.pooling import DefaultPooling
from text_embeddings_server.models.types import (
    FlashBatch,
    PaddedBatch,
    Embedding,
    Score,
)
from text_embeddings_server.models.flash_qwen_base import (
    load_weight,
    apply_rotary_pos_emb_npu,
    QwenRMSNorm,
    FlashQwenModel,
)
from text_embeddings_server.utils.flash_attn import attention


tracer = trace.get_tracer(__name__)

IS_RERANK = os.getenv("IS_RERANK", "false").lower() in ["true", "1"]


class Qwen3Attention:
    def __init__(
        self,
        model_path,
        weight_map,
        prefix,
        device,
        dtype,
        config: Qwen3Config,
        layer_idx: Optional[int] = None,
    ):
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
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
        self.k_proj_weight = load_weight(
            model_path,
            weight_map,
            prefix,
            f"layers.{layer_idx}.self_attn.k_proj.weight",
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
        self.o_proj_weight = load_weight(
            model_path,
            weight_map,
            prefix,
            f"layers.{layer_idx}.self_attn.o_proj.weight",
            dtype,
            device,
        )
        self.q_norm = QwenRMSNorm(
            model_path,
            weight_map,
            prefix,
            f"layers.{layer_idx}.self_attn.q_norm.weight",
            device,
            dtype,
            eps=config.rms_norm_eps,
        )
        self.k_norm = QwenRMSNorm(
            model_path,
            weight_map,
            prefix,
            f"layers.{layer_idx}.self_attn.k_norm.weight",
            device,
            dtype,
            eps=config.rms_norm_eps,
        )
        # Q/K/V projections are forced into grouped matmul for peak throughput.
        self._grouped_qkv_weights = [
            self.q_proj_weight.transpose(0, 1).contiguous(),
            self.k_proj_weight.transpose(0, 1).contiguous(),
            self.v_proj_weight.transpose(0, 1).contiguous(),
        ]
        self.is_310p = torch.npu.get_device_name().lower().find("310p") != -1

    def _project_qkv(self, hidden_states):
        hidden_states = hidden_states.contiguous()
        qkv_outputs = torch_npu.npu_grouped_matmul(
            [hidden_states, hidden_states, hidden_states],
            self._grouped_qkv_weights,
            group_type=-1,
            split_item=0,
        )
        return qkv_outputs[0], qkv_outputs[1], qkv_outputs[2]

    def forward(
        self, hidden_states, position_embeddings, cu_seqlens, max_s, attn_mask=None
    ):
        input_shape = hidden_states.shape[:-1]
        if self.is_310p:
            q = self.q_norm.forward(
                torch_npu.npu_linear(hidden_states, self.q_proj_weight).view(
                    *input_shape, self.num_heads, self.head_dim
                )
            )
            k = self.k_norm.forward(
                torch_npu.npu_linear(hidden_states, self.k_proj_weight).view(
                    *input_shape, self.num_key_value_heads, self.head_dim
                )
            )
            v = torch_npu.npu_linear(hidden_states, self.v_proj_weight).view(
                *input_shape, self.num_key_value_heads, self.head_dim
            )
        else:
            q_proj, k_proj, _ = self._project_qkv(hidden_states)

            q = self.q_norm.forward(
                q_proj.reshape(*input_shape, self.num_heads, self.head_dim)
            )
            k = self.k_norm.forward(
                k_proj.reshape(*input_shape, self.num_key_value_heads, self.head_dim)
            )
            v = torch_npu.npu_linear(hidden_states, self.v_proj_weight).view(
                *input_shape, self.num_key_value_heads, self.head_dim
            )

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
            is_causal=True,
            attn_mask=attn_mask,
            num_key_value_heads=self.num_key_value_heads,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = torch_npu.npu_linear(attn_output, self.o_proj_weight)

        return attn_output


class ForSequenceClassification(nn.Module):
    def __init__(
        self,
        model_path,
        weight_map,
        prefix,
        device,
        dtype,
        config: Qwen3Config,
        tokenizer,
    ):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.model = FlashQwenModel(
            model_path,
            weight_map,
            prefix,
            device,
            dtype,
            config,
            attention_cls=Qwen3Attention,
        )

        self.word_embeddings_weight = load_weight(
            model_path,
            weight_map,
            prefix,
            "embed_tokens.weight",
            dtype,
            device,
        )

        if hasattr(self.config, "max_seq_length"):
            self.max_input_length = self.config.max_seq_length
        else:
            self.max_input_length = self.config.max_position_embeddings

        prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)

        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")

    def forward(
        self,
        input_ids,
        position_ids,
        cu_seqlens,
        max_s,
        mask=None,
        attn_mask=None,
    ):
        input_ids_list = input_ids.tolist()
        cu_seqlens_list = cu_seqlens.tolist()
        input_ids_list_new = []
        seqlens_list_new = [0]
        position_ids_list = []

        prev_end = 0
        for end in cu_seqlens_list[1:]:
            sequence = input_ids_list[prev_end:end]
            new_sequence = self.prefix_tokens + sequence + self.suffix_tokens
            input_ids_list_new.extend(new_sequence)

            seqlens_list_new.append(len(new_sequence))
            position_ids_list.extend(range(len(new_sequence)))
            prev_end = end
        cu_seqlens_list_new = list(itertools.accumulate(seqlens_list_new))

        input_ids_list_new_tensor = torch.tensor(
            input_ids_list_new, device=input_ids.device, dtype=torch.int32
        )
        cu_seqlens_list_new_tensor = torch.tensor(
            cu_seqlens_list_new, device=cu_seqlens.device, dtype=torch.int32
        )
        position_ids_list_tensor = torch.tensor(
            position_ids_list, device=input_ids.device, dtype=torch.int32
        )

        # 将总长度向上补齐到能被 16 整除
        total_length = cu_seqlens_list_new[-1]
        padding_length = (16 - (total_length % 16)) % 16
        max_length = max(seqlens_list_new)
        size = len(cu_seqlens_list_new) - 1

        need_padding = (
            torch.npu.get_device_name().lower().find("310p") != -1 and IS_RERANK
        )
        if padding_length > 0 and need_padding:
            # 补齐 input_ids
            padding_input_ids = torch.zeros(
                padding_length, dtype=torch.int32, device=input_ids.device
            )
            input_ids_list_new_tensor = torch.cat(
                [input_ids_list_new_tensor, padding_input_ids], dim=0
            )

            # 补齐 position_ids
            padding_position_ids = torch.zeros(
                padding_length, dtype=torch.int32, device=input_ids.device
            )
            position_ids_list_tensor = torch.cat(
                [position_ids_list_tensor, padding_position_ids], dim=0
            )

            # 补齐 cu_seqlens
            padding_cu_seqlens = torch.tensor(
                [total_length + padding_length], dtype=torch.int32, device="cpu"
            )
            cu_seqlens_list_new_tensor = torch.cat(
                [cu_seqlens_list_new_tensor, padding_cu_seqlens], dim=0
            )

            max_length = max(max_length, padding_length)
        output = self.model.forward(
            input_ids_list_new_tensor,
            position_ids_list_tensor,
            cu_seqlens_list_new_tensor,
            max_length,
            mask,
            attn_mask,
        )

        last_token_indices = cu_seqlens_list_new_tensor[1 : size + 1] - 1
        hidden_states = output.last_hidden_state

        logits = torch_npu.npu_linear(
            hidden_states[last_token_indices], self.word_embeddings_weight
        ).cpu()
        true_logits = logits[:, self.token_true_id]
        false_logits = logits[:, self.token_false_id]
        logit_diff = true_logits - false_logits

        return logit_diff


class FlashQwen3(Model):
    def __init__(
        self,
        model_path: Path,
        device: torch.device,
        dtype: torch.dtype,
        pool: str = "lasttoken",
        trust_remote: bool = False,
    ):
        config = Qwen3Config.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
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

        if IS_RERANK:
            model = ForSequenceClassification(
                model_path, weight_map, "model", device, dtype, config, tokenizer
            )
        else:
            model = FlashQwenModel(
                model_path,
                weight_map,
                "",
                device,
                dtype,
                config,
                attention_cls=Qwen3Attention,
            )

        self.hidden_size = config.hidden_size
        self.pooling = DefaultPooling(self.hidden_size, pooling_mode=pool)
        self.device = device
        self.dtype = dtype

        super(FlashQwen3, self).__init__(model=model, dtype=dtype, device=device)

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

        return [Score(values=[p.item()]) for p in logits]
