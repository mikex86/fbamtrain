from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class FlashAttentionVariant:
    direction: str  # "fwd" or "bwd"
    dtype: str  # "bf16" or "fp16"
    head_dim: int
    write_lse: bool
    accumulate_in_fp16: bool
    even_mn: bool

    @property
    def kernel_name(self) -> str:
        if self.direction == "fwd":
            if self.dtype == "fp16" and self.accumulate_in_fp16:
                base = f"mha_full_attn_fa_fwd_hs{self.head_dim}_fp16_acc_fp16"
            else:
                base = f"mha_full_attn_fa_fwd_hs{self.head_dim}_{self.dtype}"
            if self.even_mn:
                base += "_even"
            if not self.write_lse:
                base += "_nolse"
            return base
        if self.direction == "bwd":
            base = f"mha_full_attn_fa_bwd_hs{self.head_dim}_{self.dtype}"
            if self.even_mn:
                base += "_even"
            return base
        raise ValueError(f"Unsupported flash attention direction '{self.direction}'")


def required_flash_attention_variants(head_dim: int = 128) -> Iterable[FlashAttentionVariant]:
    for even_mn in (False, True):
        for dtype in ("bf16", "fp16"):
            for write_lse in (True, False):
                yield FlashAttentionVariant(
                    direction="fwd",
                    dtype=dtype,
                    head_dim=head_dim,
                    write_lse=write_lse,
                    accumulate_in_fp16=False,
                    even_mn=even_mn,
                )
        for write_lse in (True, False):
            yield FlashAttentionVariant(
                direction="fwd",
                dtype="fp16",
                head_dim=head_dim,
                write_lse=write_lse,
                accumulate_in_fp16=True,
                even_mn=even_mn,
            )
        for dtype in ("bf16", "fp16"):
            yield FlashAttentionVariant(
                direction="bwd",
                dtype=dtype,
                head_dim=head_dim,
                write_lse=True,
                accumulate_in_fp16=False,
                even_mn=even_mn,
            )
