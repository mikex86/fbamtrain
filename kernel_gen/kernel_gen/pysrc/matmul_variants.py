from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class MatmulVariant:
    name: str
    kernel: str
    kind: str
    dtype_ab: str
    dtype_cd: str
    dtype_e: str | None
    aligned: bool
    transpose: str
    accum_fp16: bool
    activation: int
    has_bias: bool
    cuda_only: bool


def _append(variants: list[MatmulVariant], variant: MatmulVariant) -> None:
    variants.append(variant)


def _variant(
    *,
    name: str,
    kernel: str,
    kind: str,
    dtype_ab: str,
    dtype_cd: str,
    aligned: bool,
    transpose: str,
    accum_fp16: bool,
    activation: int,
    has_bias: bool,
    cuda_only: bool,
    dtype_e: str | None = None,
) -> MatmulVariant:
    return MatmulVariant(
        name=name,
        kernel=kernel,
        kind=kind,
        dtype_ab=dtype_ab,
        dtype_cd=dtype_cd,
        dtype_e=dtype_e,
        aligned=aligned,
        transpose=transpose,
        accum_fp16=accum_fp16,
        activation=activation,
        has_bias=has_bias,
        cuda_only=cuda_only,
    )


def _transpose_suffix(transpose: str) -> str:
    if transpose == "none":
        return ""
    return f"_{transpose}"


def _alignment_suffix(aligned: bool) -> str:
    return "" if aligned else "_unaligned"


def _base_variants() -> list[MatmulVariant]:
    variants: list[MatmulVariant] = []

    for aligned in (True, False):
        suffix = _alignment_suffix(aligned)
        _append(
            variants,
            _variant(
                name=f"addmm_bf16{suffix}",
                kernel="addmm_act",
                kind="addmm",
                dtype_ab="bf16",
                dtype_cd="bf16",
                aligned=aligned,
                transpose="none",
                accum_fp16=False,
                activation=0,
                has_bias=True,
                cuda_only=False,
            ),
        )
        _append(
            variants,
            _variant(
                name=f"addmm_fp16{suffix}",
                kernel="addmm_act",
                kind="addmm",
                dtype_ab="fp16",
                dtype_cd="fp16",
                aligned=aligned,
                transpose="none",
                accum_fp16=False,
                activation=0,
                has_bias=True,
                cuda_only=False,
            ),
        )
        _append(
            variants,
            _variant(
                name=f"addmm_fp16_acc_fp16{suffix}",
                kernel="addmm_act",
                kind="addmm",
                dtype_ab="fp16",
                dtype_cd="fp16",
                aligned=aligned,
                transpose="none",
                accum_fp16=True,
                activation=0,
                has_bias=True,
                cuda_only=True,
            ),
        )
        _append(
            variants,
            _variant(
                name=f"addmm_fp16_out_fp32{suffix}",
                kernel="addmm_act",
                kind="addmm",
                dtype_ab="fp16",
                dtype_cd="fp32",
                aligned=aligned,
                transpose="none",
                accum_fp16=False,
                activation=0,
                has_bias=True,
                cuda_only=False,
            ),
        )
        _append(
            variants,
            _variant(
                name=f"addmm_fp16_acc_out_fp32{suffix}",
                kernel="addmm_act",
                kind="addmm",
                dtype_ab="fp16",
                dtype_cd="fp32",
                aligned=aligned,
                transpose="none",
                accum_fp16=True,
                activation=0,
                has_bias=True,
                cuda_only=True,
            ),
        )

    for aligned in (True, False):
        for transpose in ("none", "ta", "tb", "tab"):
            suffix = f"{_transpose_suffix(transpose)}{_alignment_suffix(aligned)}"
            _append(
                variants,
                _variant(
                    name=f"matmul_bf16{suffix}",
                    kernel="addmm_act",
                    kind="matmul",
                    dtype_ab="bf16",
                    dtype_cd="bf16",
                    aligned=aligned,
                    transpose=transpose,
                    accum_fp16=False,
                    activation=0,
                    has_bias=False,
                    cuda_only=False,
                ),
            )
            _append(
                variants,
                _variant(
                    name=f"matmul_bf16_out_fp32{suffix}",
                    kernel="addmm_act",
                    kind="matmul",
                    dtype_ab="bf16",
                    dtype_cd="fp32",
                    aligned=aligned,
                    transpose=transpose,
                    accum_fp16=False,
                    activation=0,
                    has_bias=False,
                    cuda_only=False,
                ),
            )
            _append(
                variants,
                _variant(
                    name=f"matmul_fp16{suffix}",
                    kernel="addmm_act",
                    kind="matmul",
                    dtype_ab="fp16",
                    dtype_cd="fp16",
                    aligned=aligned,
                    transpose=transpose,
                    accum_fp16=False,
                    activation=0,
                    has_bias=False,
                    cuda_only=False,
                ),
            )
            _append(
                variants,
                _variant(
                    name=f"matmul_fp16_out_fp32{suffix}",
                    kernel="addmm_act",
                    kind="matmul",
                    dtype_ab="fp16",
                    dtype_cd="fp32",
                    aligned=aligned,
                    transpose=transpose,
                    accum_fp16=False,
                    activation=0,
                    has_bias=False,
                    cuda_only=False,
                ),
            )
            _append(
                variants,
                _variant(
                    name=f"matmul_fp16_acc_out_fp32{suffix}",
                    kernel="addmm_act",
                    kind="matmul",
                    dtype_ab="fp16",
                    dtype_cd="fp32",
                    aligned=aligned,
                    transpose=transpose,
                    accum_fp16=True,
                    activation=0,
                    has_bias=False,
                    cuda_only=True,
                ),
            )

    for aligned in (True, False):
        suffix = _alignment_suffix(aligned)
        _append(
            variants,
            _variant(
                name=f"matmul_fp16_acc_fp16{suffix}",
                kernel="addmm_act",
                kind="matmul",
                dtype_ab="fp16",
                dtype_cd="fp16",
                aligned=aligned,
                transpose="none",
                accum_fp16=True,
                activation=0,
                has_bias=False,
                cuda_only=True,
            ),
        )

    return variants


def _gelu_variants() -> list[MatmulVariant]:
    variants: list[MatmulVariant] = []

    for aligned in (True, False):
        suffix = _alignment_suffix(aligned)
        _append(
            variants,
            _variant(
                name=f"addmm_gelu_bf16{suffix}",
                kernel="addmm_act",
                kind="addmm",
                dtype_ab="bf16",
                dtype_cd="bf16",
                aligned=aligned,
                transpose="none",
                accum_fp16=False,
                activation=2,
                has_bias=True,
                cuda_only=False,
            ),
        )
        _append(
            variants,
            _variant(
                name=f"addmm_gelu_fp16{suffix}",
                kernel="addmm_act",
                kind="addmm",
                dtype_ab="fp16",
                dtype_cd="fp16",
                aligned=aligned,
                transpose="none",
                accum_fp16=False,
                activation=2,
                has_bias=True,
                cuda_only=False,
            ),
        )
        _append(
            variants,
            _variant(
                name=f"addmm_gelu_fp16_acc_fp16{suffix}",
                kernel="addmm_act",
                kind="addmm",
                dtype_ab="fp16",
                dtype_cd="fp16",
                aligned=aligned,
                transpose="none",
                accum_fp16=True,
                activation=2,
                has_bias=True,
                cuda_only=True,
            ),
        )
        _append(
            variants,
            _variant(
                name=f"addmm_gelu_preact_bf16{suffix}",
                kernel="addmm_act_preact",
                kind="addmm_preact",
                dtype_ab="bf16",
                dtype_cd="bf16",
                dtype_e="bf16",
                aligned=aligned,
                transpose="none",
                accum_fp16=False,
                activation=2,
                has_bias=True,
                cuda_only=False,
            ),
        )
        _append(
            variants,
            _variant(
                name=f"addmm_gelu_preact_fp16{suffix}",
                kernel="addmm_act_preact",
                kind="addmm_preact",
                dtype_ab="fp16",
                dtype_cd="fp16",
                dtype_e="fp16",
                aligned=aligned,
                transpose="none",
                accum_fp16=False,
                activation=2,
                has_bias=True,
                cuda_only=False,
            ),
        )
        _append(
            variants,
            _variant(
                name=f"addmm_gelu_preact_fp16_acc_fp16{suffix}",
                kernel="addmm_act_preact",
                kind="addmm_preact",
                dtype_ab="fp16",
                dtype_cd="fp16",
                dtype_e="fp16",
                aligned=aligned,
                transpose="none",
                accum_fp16=True,
                activation=2,
                has_bias=True,
                cuda_only=True,
            ),
        )
        _append(
            variants,
            _variant(
                name=f"matmul_gelu_bf16{suffix}",
                kernel="addmm_act",
                kind="matmul",
                dtype_ab="bf16",
                dtype_cd="bf16",
                aligned=aligned,
                transpose="none",
                accum_fp16=False,
                activation=2,
                has_bias=False,
                cuda_only=False,
            ),
        )
        _append(
            variants,
            _variant(
                name=f"matmul_gelu_fp16{suffix}",
                kernel="addmm_act",
                kind="matmul",
                dtype_ab="fp16",
                dtype_cd="fp16",
                aligned=aligned,
                transpose="none",
                accum_fp16=False,
                activation=2,
                has_bias=False,
                cuda_only=False,
            ),
        )
        _append(
            variants,
            _variant(
                name=f"matmul_gelu_fp16_acc_fp16{suffix}",
                kernel="addmm_act",
                kind="matmul",
                dtype_ab="fp16",
                dtype_cd="fp16",
                aligned=aligned,
                transpose="none",
                accum_fp16=True,
                activation=2,
                has_bias=False,
                cuda_only=True,
            ),
        )

    return variants


def get_variants(variant_set: str) -> list[MatmulVariant]:
    if variant_set == "base":
        return _base_variants()
    if variant_set == "gelu":
        return _gelu_variants()
    if variant_set in {"all", "base+gelu", "gelu+base"}:
        return _base_variants() + _gelu_variants()
    raise ValueError(f"Unknown variant set '{variant_set}'. Expected 'base', 'gelu', or 'all'.")


def get_variant_sets(variant_sets: Iterable[str]) -> list[MatmulVariant]:
    variants: list[MatmulVariant] = []
    seen = set()
    for name in variant_sets:
        for variant in get_variants(name):
            if variant.name in seen:
                continue
            seen.add(variant.name)
            variants.append(variant)
    return variants
