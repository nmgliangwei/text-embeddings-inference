import torch
import torch_npu


class AttentionMaskBuilder:
    def __init__(self):
        self.max_seq_len = 0
        self.attn_mask_cache = None

    def _round_up(self, x: int, align: int):
        # round up x to align, for example, if align is 16, x will be rounded up to 16, 32, 48, etc.
        # input: 15, 16 -> output: 16
        # input: 17, 16 -> output: 32
        # input: 30, 16 -> output: 32
        # input: 33, 16 -> output: 48
        # ...
        return (x + align - 1) // align * align

    def _custom_pad(self, x, pad_dims):
        # pad the input tensor to the shape of pad_dims
        # input: (13, 30), pad_dims: [0, 2, 0, 3]
        # output: (16, 32)
        return torch.nn.functional.pad(x, pad_dims)

    def _custom_reshape(self, x, target_shape):
        # reshape the input tensor to the shape of target_shape
        # input: (16, 32), target_shape: [1, 16, 2, 16]
        # output: (1, 16, 2, 16)
        return x.reshape(target_shape)

    def _custom_transpose(self, x, dim1, dim2):
        # transpose the input tensor
        # input: (1, 16, 2, 16), dim1: 1, dim2: 2
        # output: (1, 2, 16, 16)
        return x.transpose(dim1, dim2)

    def _nd_to_nz_2d(self, in_tensor: torch.Tensor) -> torch.Tensor:
        # in_tensor: (13, 30)
        aux_dims = [1, 0, 0, 16]
        # aux_dims[1]: 16
        aux_dims[1] = self._round_up(in_tensor.size(0), 16)
        # aux_dims[2]: 2
        aux_dims[2] = self._round_up(in_tensor.size(1), 16) // 16

        # after: aux_dims: [1, 16, 2, 16]

        pad_dims = [0, 0, 0, 0]
        # pad_dims[1]: 2
        pad_dims[1] = self._round_up(in_tensor.size(1), 16) - in_tensor.size(1)
        # pad_dims[3]: 3
        pad_dims[3] = self._round_up(in_tensor.size(0), 16) - in_tensor.size(0)

        # after: pad_dims: [0, 2, 0, 3]

        # return: (1, 2, 16, 16)
        return self._custom_transpose(
            self._custom_reshape(self._custom_pad(in_tensor, pad_dims), aux_dims), 1, 2
        ).contiguous()

    def get_npu_attn_mask_no_causal(
        self, max_seq_len: int, dtype: torch.dtype, device: torch.device
    ):
        if self.attn_mask_cache is not None and self.max_seq_len >= max_seq_len:
            return self.attn_mask_cache

        self.max_seq_len = max_seq_len
        attn_mask_npu_cache = torch.zeros(
            size=(max_seq_len, max_seq_len), dtype=dtype, device=device
        )
        attn_mask_npu_cache = self._nd_to_nz_2d(attn_mask_npu_cache)
        attn_mask_npu_cache = torch_npu.npu_format_cast(
            attn_mask_npu_cache.contiguous(), 29
        )

        self.max_seq_len = max_seq_len
        self.attn_mask_cache = attn_mask_npu_cache

        return self.attn_mask_cache

    def get_npu_attn_mask_causal(
        self, max_seq_len: int, dtype: torch.dtype, device: torch.device
    ):
        if self.attn_mask_cache is not None and self.max_seq_len >= max_seq_len:
            return self.attn_mask_cache

        tril = torch.ones(
            (max_seq_len, max_seq_len), dtype=torch.bool, device=device
        ).tril_()
        upper = ~tril
        attn_mask_npu = torch.zeros(
            (max_seq_len, max_seq_len), dtype=dtype, device=device
        )
        attn_mask_npu.masked_fill_(upper, float("-inf"))

        attn_mask_npu = self._nd_to_nz_2d(attn_mask_npu)
        attn_mask_npu = torch_npu.npu_format_cast(attn_mask_npu.contiguous(), 29)

        self.max_seq_len = max_seq_len
        self.attn_mask_cache = attn_mask_npu

        return self.attn_mask_cache

    def get_npu_attn_mask_310p(
        self,
        max_seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
        causal: bool = False,
    ):
        if causal:
            return self.get_npu_attn_mask_causal(max_seq_len, dtype, device)
        else:
            return self.get_npu_attn_mask_no_causal(max_seq_len, dtype, device)

    def get_npu_attn_mask_910b(
        self,
        max_seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
        causal: bool = False,
    ):
        if causal:
            if self.attn_mask_cache is not None:
                return self.attn_mask_cache
            else:
                self.attn_mask_cache = torch.triu(
                    torch.ones((2048, 2048), dtype=torch.bool, device=device),
                    diagonal=1,
                )
        else:
            if self.attn_mask_cache is not None and self.max_seq_len >= max_seq_len:
                return self.attn_mask_cache

            self.max_seq_len = max_seq_len
            self.attn_mask_cache = torch.zeros(
                (1, max_seq_len, max_seq_len), device=device, dtype=torch.bool
            )

        return self.attn_mask_cache
