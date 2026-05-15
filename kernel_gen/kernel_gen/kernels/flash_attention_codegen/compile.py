from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

from ..cutlass_codegen.fetch_cutlass_sources import fetch_cutlass_sources
from .ensure_flash_attention import ensure_flash_attention_sources
from .variants import required_flash_attention_variants
from pysrc import kinfo_asm_gen


_ROOT_DIR = Path(__file__).resolve().parent
_OUTPUT_ROOT = _ROOT_DIR.parents[1] / "output"
_GENERATED_ROOT = _ROOT_DIR / "generated"
_BUILD_ROOT = _ROOT_DIR / "build"

_LOG2E = 1.4426950408889634

_DTYPE_MAP = {
    "fp16": "cutlass::half_t",
    "bf16": "cutlass::bfloat16_t",
}


def _load_autotune_payload(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text())
    except OSError as exc:
        raise SystemExit(f"Failed to read autotune config '{path}': {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Autotune config '{path}' is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"Autotune config '{path}' must contain a JSON object at the top level.")
    return payload


def _parse_autotune_entry(entry: object, name: str, path: Path) -> dict[str, int]:
    if not isinstance(entry, dict):
        raise SystemExit(f"Autotune entry '{name}' in '{path}' must be a JSON object.")
    payload = entry.get("config", entry)
    if not isinstance(payload, dict):
        raise SystemExit(f"Autotune entry '{name}' in '{path}' has invalid 'config' payload.")
    try:
        block_m = int(payload["block_m"])
        block_n = int(payload["block_n"])
        num_warps = int(payload["num_warps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"Autotune entry '{name}' in '{path}' is missing required fields.") from exc
    return {"block_m": block_m, "block_n": block_n, "num_warps": num_warps}


def _extract_flash_autotune_section(section: Mapping[str, object], path: Path) -> dict[str, dict[str, dict[str, int]]]:
    result: dict[str, dict[str, dict[str, int]]] = {"fwd": {}, "bwd": {}}
    for direction in ("fwd", "bwd"):
        entries = section.get(direction, {})
        if entries is None:
            continue
        if not isinstance(entries, dict):
            raise SystemExit(f"Autotune section '{direction}' in '{path}' must be a JSON object.")
        for name, entry in entries.items():
            if not isinstance(name, str):
                raise SystemExit(f"Autotune entry name in '{path}' must be a string (got {name!r}).")
            result[direction][name] = _parse_autotune_entry(entry, name, path)
    return result


def _load_flash_autotune_overlays(paths: list[Path]) -> dict[str, dict[str, dict[str, dict[str, int]]]]:
    overlays: dict[str, dict[str, dict[str, dict[str, int]]]] = {}
    for path in paths:
        payload = _load_autotune_payload(path)
        section = payload.get("flash_attention")
        if section is None:
            raise SystemExit(f"Autotune config '{path}' missing 'flash_attention' section.")
        if not isinstance(section, dict):
            raise SystemExit(f"Autotune config '{path}' 'flash_attention' section must be a JSON object.")
        has_sm_scope = any(isinstance(key, str) and key.startswith("sm_") for key in section.keys())
        if has_sm_scope:
            for sm_key, sm_section in section.items():
                if not (isinstance(sm_key, str) and sm_key.startswith("sm_")):
                    raise SystemExit(f"Autotune config '{path}' contains invalid SM key '{sm_key}'.")
                if not isinstance(sm_section, dict):
                    raise SystemExit(f"Autotune config '{path}' section '{sm_key}' must be a JSON object.")
                overlays.setdefault(sm_key, {"fwd": {}, "bwd": {}})
                extracted = _extract_flash_autotune_section(sm_section, path)
                overlays[sm_key]["fwd"].update(extracted["fwd"])
                overlays[sm_key]["bwd"].update(extracted["bwd"])
        else:
            overlays.setdefault("default", {"fwd": {}, "bwd": {}})
            extracted = _extract_flash_autotune_section(section, path)
            overlays["default"]["fwd"].update(extracted["fwd"])
            overlays["default"]["bwd"].update(extracted["bwd"])
    return overlays


def _select_autotune_override(
    overlays: Mapping[str, Mapping[str, Mapping[str, dict[str, int]]]],
    sm: str,
    direction: str,
    dtype: str,
    even_mn: bool,
) -> Optional[dict[str, int]]:
    key = f"{dtype}_even" if even_mn else dtype
    fallback_key = dtype
    for scope in (sm, "default"):
        scoped = overlays.get(scope)
        if not scoped:
            continue
        entries = scoped.get(direction, {})
        if key in entries:
            return entries[key]
        if even_mn and fallback_key in entries:
            return entries[fallback_key]
    return None


@dataclass(frozen=True)
class KernelConfig:
    direction: str
    dtype: str
    head_dim: int
    block_m: int
    block_n: int
    num_warps: int
    arg_count: int
    shared_mem_bytes: int
    function_name: str
    is_even_mn: bool
    is_even_k: bool
    accumulate_in_fp16: bool


_FWD_BLOCK_M = 128
_FWD_BLOCK_N = 32
_FWD_NUM_WARPS = 4
_BWD_BLOCK_M = 64
_BWD_BLOCK_N = 64
_BWD_NUM_WARPS = 8

_BWD_ATOM_LAYOUT_MSDP = 4
_BWD_ATOM_LAYOUT_NDKV = 2
_BWD_ATOM_LAYOUT_MDQ = 2
_BWD_IS_V_IN_REGS = True
_BWD_NO_DOUBLE_BUFFER = False


def _normalize_sm(sm: str) -> str:
    sm = sm.lower().strip()
    if sm.startswith("sm_"):
        return f"sm{sm[3:]}"
    return sm


def _force_uneven_bwd(sm: str) -> bool:
    # FlashAttention bwd even-MN path can fail to compile on Ampere (sm80).
    return _normalize_sm(sm) == "sm80"


def _emit_fwd_source(config: KernelConfig) -> str:
    element = _DTYPE_MAP[config.dtype]
    return f"""// Auto-generated by flash_attention_codegen/compile.py. Do not edit.
#include <cstdint>
#include <type_traits>

#include \"namespace_config.h\"
#include \"flash.h\"
#include \"flash_fwd_kernel.h\"
#include <cutlass/cutlass.h>

extern \"C\" __global__ void mha_attn_fwd(
    float softmax_scale,
    void *scratch_m,
    uint32_t batch_size,
    uint32_t n_head,
    void *q_ptr,
    void *k_ptr,
    void *v_ptr,
    void *o_ptr,
    uint32_t q_row_stride_in,
    uint32_t n_ctx,
    void *global_scratch_ptr) {{
    (void)global_scratch_ptr;
    using Element = {element};
    using KernelTraits = Flash_fwd_kernel_traits<{config.head_dim}, {config.block_m}, {config.block_n}, {config.num_warps}, false, false, Element, {str(config.accumulate_in_fp16).lower()}>;
    FLASH_NAMESPACE::Flash_fwd_params params{{}};

    const int head_dim = {config.head_dim};
    const int64_t q_row_stride = static_cast<int64_t>(q_row_stride_in);
    const int64_t q_batch_stride = q_row_stride * n_ctx;
    const int64_t q_head_stride = head_dim;

    const int64_t k_batch_stride = q_batch_stride;
    const int64_t k_head_stride = q_head_stride;
    const int64_t k_row_stride = q_row_stride;

    const int64_t v_batch_stride = q_batch_stride;
    const int64_t v_head_stride = q_head_stride;
    const int64_t v_row_stride = q_row_stride;

    const int64_t o_batch_stride = static_cast<int64_t>(n_ctx) * n_head * head_dim;
    const int64_t o_head_stride = head_dim;
    const int64_t o_row_stride = static_cast<int64_t>(n_head) * head_dim;

    params.q_ptr = q_ptr;
    params.k_ptr = k_ptr;
    params.v_ptr = v_ptr;
    params.o_ptr = o_ptr;
    params.oaccum_ptr = nullptr;
    params.p_ptr = nullptr;
    params.softmax_lse_ptr = scratch_m;
    params.softmax_lseaccum_ptr = scratch_m;

    params.q_batch_stride = q_batch_stride;
    params.k_batch_stride = k_batch_stride;
    params.v_batch_stride = v_batch_stride;
    params.q_row_stride = q_row_stride;
    params.k_row_stride = k_row_stride;
    params.v_row_stride = v_row_stride;
    params.q_head_stride = q_head_stride;
    params.k_head_stride = k_head_stride;
    params.v_head_stride = v_head_stride;

    params.o_batch_stride = o_batch_stride;
    params.o_row_stride = o_row_stride;
    params.o_head_stride = o_head_stride;

    params.b = static_cast<int>(batch_size);
    params.h = static_cast<int>(n_head);
    params.h_k = static_cast<int>(n_head);
    params.h_h_k_ratio = 1;

    params.seqlen_q = static_cast<int>(n_ctx);
    params.seqlen_k = static_cast<int>(n_ctx);
    params.seqlen_knew = 0;
    params.d = head_dim;
    const int seqlen_q_rounded = static_cast<int>(((n_ctx + 127u) / 128u) * 128u);
    const int seqlen_k_rounded = static_cast<int>(((n_ctx + 127u) / 128u) * 128u);
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d_rounded = head_dim;
    params.rotary_dim = 0;
    params.total_q = static_cast<int>(n_ctx);

    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * { _LOG2E }f;
    params.p_dropout = 1.0f;
    params.p_dropout_in_uint8_t = 255;
    params.rp_dropout = 1.0f;
    params.scale_softmax_rp_dropout = softmax_scale;

    params.window_size_left = -1;
    params.window_size_right = -1;
    params.softcap = 0.0f;

    params.cu_seqlens_q = nullptr;
    params.cu_seqlens_k = nullptr;
    params.leftpad_k = nullptr;
    params.seqused_k = nullptr;
    params.blockmask = nullptr;

    params.knew_ptr = nullptr;
    params.vnew_ptr = nullptr;
    params.knew_batch_stride = 0;
    params.vnew_batch_stride = 0;
    params.knew_row_stride = 0;
    params.vnew_row_stride = 0;
    params.knew_head_stride = 0;
    params.vnew_head_stride = 0;

    params.rotary_cos_ptr = nullptr;
    params.rotary_sin_ptr = nullptr;
    params.cache_batch_idx = nullptr;

    params.block_table = nullptr;
    params.block_table_batch_stride = 0;
    params.page_block_size = 0;

    params.philox_args = at::PhiloxCudaState{{0, 0}};
    params.rng_state = nullptr;

    params.is_bf16 = std::is_same_v<Element, cutlass::bfloat16_t>;
    params.is_causal = false;
    params.is_seqlens_k_cumulative = false;
    params.is_rotary_interleaved = false;
    params.num_splits = 0;

    params.alibi_slopes_ptr = nullptr;
    params.alibi_slopes_batch_stride = 0;

    params.unpadded_lse = false;
    params.seqlenq_ngroups_swapped = false;

    FLASH_NAMESPACE::compute_attn<KernelTraits, false, false, false, false, {str(config.is_even_mn).lower()},
                                  {str(config.is_even_k).lower()}, false, false>(params);
}}
"""


def _emit_bwd_source(config: KernelConfig, kind: str) -> str:
    if kind == "main":
        kernel_call = (
            "FLASH_NAMESPACE::compute_dq_dk_dv_seqk_parallel<KernelTraits, false, false, false, false, "
            f"{str(config.is_even_mn).lower()}, {str(config.is_even_k).lower()}, false>(params);"
        )
    elif kind == "dot":
        kernel_call = "FLASH_NAMESPACE::compute_dot_do_o<true, KernelTraits>(params);"
    elif kind == "convert":
        kernel_call = "FLASH_NAMESPACE::convert_dQ<KernelTraits>(params, 1);"
    else:
        raise ValueError(f"Unsupported FlashAttention backward kind '{kind}'")
    element = _DTYPE_MAP[config.dtype]
    return f"""// Auto-generated by flash_attention_codegen/compile.py. Do not edit.
#include <cstdint>
#include <type_traits>

#include \"namespace_config.h\"
#include \"flash.h\"
#include \"flash_bwd_preprocess_kernel.h\"
#include \"flash_bwd_kernel.h\"
#include <cutlass/cutlass.h>

extern \"C\" __global__ void {config.function_name}(
    void *q_ptr,
    void *k_ptr,
    void *v_ptr,
    void *o_ptr,
    float softmax_scale,
    void *do_ptr,
    void *dq_ptr,
    void *dk_ptr,
    void *dv_ptr,
    void *m_ptr,
    void *delta_ptr,
    uint32_t batch_size,
    uint32_t n_head,
    uint32_t n_ctx,
    uint32_t q_row_stride_in,
    uint32_t do_row_stride_in,
    void *global_scratch_ptr) {{
    using Element = {element};
    using KernelTraits = Flash_bwd_kernel_traits<{config.head_dim}, {config.block_m}, {config.block_n}, {config.num_warps},
        { _BWD_ATOM_LAYOUT_MSDP }, { _BWD_ATOM_LAYOUT_NDKV }, { _BWD_ATOM_LAYOUT_MDQ }, { str(_BWD_IS_V_IN_REGS).lower() }, { str(_BWD_NO_DOUBLE_BUFFER).lower() }, Element>;

    FLASH_NAMESPACE::Flash_bwd_params params{{}};

    const int head_dim = {config.head_dim};
    const int64_t q_row_stride = static_cast<int64_t>(q_row_stride_in);
    const int64_t q_batch_stride = q_row_stride * n_ctx;
    const int64_t q_head_stride = head_dim;

    const int64_t k_batch_stride = q_batch_stride;
    const int64_t k_head_stride = q_head_stride;
    const int64_t k_row_stride = q_row_stride;

    const int64_t v_batch_stride = q_batch_stride;
    const int64_t v_head_stride = q_head_stride;
    const int64_t v_row_stride = q_row_stride;

    const int64_t o_row_stride = static_cast<int64_t>(do_row_stride_in);
    const int64_t o_batch_stride = o_row_stride * n_ctx;
    const int64_t o_head_stride = head_dim;

    params.q_ptr = q_ptr;
    params.k_ptr = k_ptr;
    params.v_ptr = v_ptr;
    params.o_ptr = o_ptr;
    params.oaccum_ptr = nullptr;

    params.do_ptr = do_ptr;
    params.dq_ptr = dq_ptr;
    params.dk_ptr = dk_ptr;
    params.dv_ptr = dv_ptr;

    params.dq_accum_ptr = global_scratch_ptr;
    params.dk_accum_ptr = nullptr;
    params.dv_accum_ptr = nullptr;

    params.dsoftmax_sum = delta_ptr;
    params.softmax_lse_ptr = m_ptr;
    params.softmax_lseaccum_ptr = m_ptr;

    params.q_batch_stride = q_batch_stride;
    params.k_batch_stride = k_batch_stride;
    params.v_batch_stride = v_batch_stride;
    params.q_row_stride = q_row_stride;
    params.k_row_stride = k_row_stride;
    params.v_row_stride = v_row_stride;
    params.q_head_stride = q_head_stride;
    params.k_head_stride = k_head_stride;
    params.v_head_stride = v_head_stride;

    params.o_batch_stride = o_batch_stride;
    params.o_row_stride = o_row_stride;
    params.o_head_stride = o_head_stride;

    params.do_batch_stride = o_batch_stride;
    params.do_row_stride = o_row_stride;
    params.do_head_stride = o_head_stride;

    params.dq_batch_stride = q_batch_stride;
    params.dk_batch_stride = k_batch_stride;
    params.dv_batch_stride = v_batch_stride;
    params.dq_row_stride = q_row_stride;
    params.dk_row_stride = k_row_stride;
    params.dv_row_stride = v_row_stride;
    params.dq_head_stride = q_head_stride;
    params.dk_head_stride = k_head_stride;
    params.dv_head_stride = v_head_stride;

    params.b = static_cast<int>(batch_size);
    params.h = static_cast<int>(n_head);
    params.h_k = static_cast<int>(n_head);
    params.h_h_k_ratio = 1;

    params.seqlen_q = static_cast<int>(n_ctx);
    params.seqlen_k = static_cast<int>(n_ctx);
    params.seqlen_knew = 0;
    params.d = head_dim;
    const int seqlen_q_rounded = static_cast<int>(((n_ctx + 127u) / 128u) * 128u);
    const int seqlen_k_rounded = static_cast<int>(((n_ctx + 127u) / 128u) * 128u);
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d_rounded = head_dim;
    params.rotary_dim = 0;
    params.total_q = static_cast<int>(n_ctx);

    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * { _LOG2E }f;
    params.p_dropout = 1.0f;
    params.p_dropout_in_uint8_t = 255;
    params.rp_dropout = 1.0f;
    params.scale_softmax_rp_dropout = softmax_scale;

    params.window_size_left = -1;
    params.window_size_right = -1;
    params.softcap = 0.0f;

    params.cu_seqlens_q = nullptr;
    params.cu_seqlens_k = nullptr;
    params.leftpad_k = nullptr;
    params.seqused_k = nullptr;
    params.blockmask = nullptr;

    params.knew_ptr = nullptr;
    params.vnew_ptr = nullptr;
    params.knew_batch_stride = 0;
    params.vnew_batch_stride = 0;
    params.knew_row_stride = 0;
    params.vnew_row_stride = 0;
    params.knew_head_stride = 0;
    params.vnew_head_stride = 0;

    params.rotary_cos_ptr = nullptr;
    params.rotary_sin_ptr = nullptr;
    params.cache_batch_idx = nullptr;

    params.block_table = nullptr;
    params.block_table_batch_stride = 0;
    params.page_block_size = 0;

    params.philox_args = at::PhiloxCudaState{{0, 0}};
    params.rng_state = nullptr;

    params.is_bf16 = std::is_same_v<Element, cutlass::bfloat16_t>;
    params.is_causal = false;
    params.is_seqlens_k_cumulative = false;
    params.is_rotary_interleaved = false;
    params.num_splits = 0;

    params.alibi_slopes_ptr = nullptr;
    params.alibi_slopes_batch_stride = 0;

    params.unpadded_lse = false;
    params.seqlenq_ngroups_swapped = false;

    params.deterministic = false;
    params.dq_accum_split_stride = 0;

    {kernel_call}
}}
"""


def _parse_sm_list(raw: str) -> list[str]:
    items = [item.strip().lower() for item in raw.split(",") if item.strip()]
    return items or ["sm_89", "sm_90", "sm_100"]


def _resolve_nvcc() -> Path:
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        nvcc = Path(cuda_home) / "bin" / "nvcc"
        if nvcc.exists():
            return nvcc
    nvcc = shutil.which("nvcc")
    if nvcc:
        return Path(nvcc)
    raise FileNotFoundError("nvcc not found. Set CUDA_HOME or ensure nvcc is on PATH.")


def _smem_bytes_fwd(head_dim: int, block_m: int, block_n: int, elem_bytes: int = 2) -> int:
    q_size = block_m * head_dim * elem_bytes
    kv_size = block_n * head_dim * 2 * elem_bytes
    return q_size + kv_size


def _smem_bytes_bwd(
    head_dim: int,
    block_m: int,
    block_n: int,
    elem_bytes: int = 2,
    no_double_buffer: bool = False,
    is_v_in_regs: bool = True,
) -> int:
    qdo_mult = 2 if no_double_buffer else 3
    qdo_size = block_m * head_dim * qdo_mult * elem_bytes
    kv_size = block_n * head_dim * 2 * elem_bytes
    ds_size = block_m * block_n * elem_bytes
    p_size = ds_size
    dq_size = block_m * head_dim * elem_bytes

    if is_v_in_regs:
        tail = max(kv_size, kv_size // 2 + ds_size + max(p_size, dq_size))
    else:
        tail = kv_size + ds_size + max(p_size, dq_size)
    return qdo_size + tail


def _smem_bytes_bwd_seqk(
    head_dim: int,
    block_m: int,
    block_n: int,
    elem_bytes: int = 2,
    no_double_buffer: bool = False,
    is_v_in_regs: bool = True,
) -> int:
    qdo_mult = 2 if no_double_buffer else 3
    qdo_size = block_m * head_dim * qdo_mult * elem_bytes
    kv_size = block_n * head_dim * 2 * elem_bytes
    ds_size = block_m * block_n * elem_bytes
    p_size = ds_size

    if is_v_in_regs:
        tail = max(kv_size, kv_size // 2 + ds_size + p_size)
    else:
        tail = kv_size + ds_size + p_size
    return qdo_size + tail


def _smem_bytes_bwd_convert(head_dim: int, block_m: int, elem_bytes: int = 2) -> int:
    return block_m * head_dim * elem_bytes


def _write_kinfo(out_base: Path, config: KernelConfig) -> None:
    kv = {
        "function_name": config.function_name,
        "smem_bytes": config.shared_mem_bytes,
        "num_warps": config.num_warps,
        "arg_count": config.arg_count,
        "global_scratch_size": 0,
        "block_size_x": config.block_m,
        "block_size_y": config.block_n,
    }
    kinfo_asm_gen.write_kinfo_and_macro(out_base, config.function_name, kv)


def _compile_cuda(
    src: Path,
    out_cubin: Path,
    sm: str,
    include_dirs: list[Path],
) -> None:
    nvcc = _resolve_nvcc()
    sm = sm.lower().replace("sm_", "")
    cmd = [
        str(nvcc),
        "-std=c++17",
        "-O3",
        "--use_fast_math",
        "--expt-relaxed-constexpr",
        "-cubin",
        "-o",
        str(out_cubin),
        str(src),
        "-gencode",
        f"arch=compute_{sm},code=sm_{sm}",
    ]
    for include_dir in include_dirs:
        cmd.extend(["-I", str(include_dir)])
    subprocess.run(cmd, check=True)


def _compile_task(src: Path, out_cubin: Path, sm: str, include_dirs: list[Path]) -> Path:
    _compile_cuda(src, out_cubin, sm, include_dirs)
    return out_cubin


def _parallel_compile(tasks: list[tuple[Path, Path, str, list[Path]]], jobs: int) -> None:
    if jobs <= 1 or len(tasks) <= 1:
        for src, out_cubin, sm, include_dirs in tasks:
            _compile_cuda(src, out_cubin, sm, include_dirs)
        return
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as executor:
        futures = [executor.submit(_compile_task, src, out_cubin, sm, include_dirs) for src, out_cubin, sm, include_dirs in tasks]
        for future in as_completed(futures):
            future.result()


def _make_fwd_config(
    head_dim: int,
    dtype: str,
    even_mn: bool,
    accumulate_in_fp16: bool,
    sm: str,
    overlays: Mapping[str, Mapping[str, Mapping[str, dict[str, int]]]],
) -> KernelConfig:
    block_m = _FWD_BLOCK_M
    block_n = _FWD_BLOCK_N
    num_warps = _FWD_NUM_WARPS
    override = _select_autotune_override(overlays, sm, "fwd", dtype, even_mn)
    if override:
        block_m = override["block_m"]
        block_n = override["block_n"]
        num_warps = override["num_warps"]
    return KernelConfig(
        direction="fwd",
        dtype=dtype,
        head_dim=head_dim,
        block_m=block_m,
        block_n=block_n,
        num_warps=num_warps,
        arg_count=11,
        shared_mem_bytes=_smem_bytes_fwd(head_dim, block_m, block_n),
        function_name="mha_attn_fwd",
        is_even_mn=even_mn,
        is_even_k=True,
        accumulate_in_fp16=accumulate_in_fp16,
    )


def _make_bwd_config(
    head_dim: int,
    dtype: str,
    even_mn: bool,
    sm: str,
    overlays: Mapping[str, Mapping[str, Mapping[str, dict[str, int]]]],
) -> KernelConfig:
    effective_even_mn = False if _force_uneven_bwd(sm) else even_mn
    block_m = _BWD_BLOCK_M
    block_n = _BWD_BLOCK_N
    num_warps = _BWD_NUM_WARPS
    override = _select_autotune_override(overlays, sm, "bwd", dtype, effective_even_mn)
    if override:
        block_m = override["block_m"]
        block_n = override["block_n"]
        num_warps = override["num_warps"]
    return KernelConfig(
        direction="bwd",
        dtype=dtype,
        head_dim=head_dim,
        block_m=block_m,
        block_n=block_n,
        num_warps=num_warps,
        arg_count=17,
        shared_mem_bytes=_smem_bytes_bwd_seqk(
            head_dim,
            block_m,
            block_n,
            no_double_buffer=_BWD_NO_DOUBLE_BUFFER,
            is_v_in_regs=_BWD_IS_V_IN_REGS,
        ),
        function_name="mha_attn_bwd",
        is_even_mn=effective_even_mn,
        is_even_k=True,
        accumulate_in_fp16=False,
    )


def _build_kernel_configs(
    head_dim: int,
    sm: str,
    overlays: Mapping[str, Mapping[str, Mapping[str, dict[str, int]]]],
) -> tuple[dict[tuple[str, str, bool, bool], KernelConfig], dict[tuple[str, str], KernelConfig]]:
    configs: dict[tuple[str, str, bool, bool], KernelConfig] = {}
    for even_mn in (False, True):
        for dtype in ("fp16", "bf16"):
            if dtype == "fp16":
                for accumulate_in_fp16 in (False, True):
                    configs[("fwd", dtype, even_mn, accumulate_in_fp16)] = _make_fwd_config(
                        head_dim,
                        dtype,
                        even_mn,
                        accumulate_in_fp16,
                        sm,
                        overlays,
                    )
            else:
                configs[("fwd", dtype, even_mn, False)] = _make_fwd_config(head_dim, dtype, even_mn, False, sm, overlays)
            configs[("bwd", dtype, even_mn, False)] = _make_bwd_config(head_dim, dtype, even_mn, sm, overlays)

    aux_configs: dict[tuple[str, str], KernelConfig] = {}
    for dtype in ("fp16", "bf16"):
        base = configs[("bwd", dtype, False, False)]
        aux_configs[("bwd_dot", dtype)] = KernelConfig(
            direction="bwd",
            dtype=dtype,
            head_dim=head_dim,
            block_m=base.block_m,
            block_n=base.block_n,
            num_warps=base.num_warps,
            arg_count=17,
            shared_mem_bytes=0,
            function_name="mha_attn_bwd_dot_do_o",
            is_even_mn=False,
            is_even_k=True,
            accumulate_in_fp16=False,
        )
        aux_configs[("bwd_convert", dtype)] = KernelConfig(
            direction="bwd",
            dtype=dtype,
            head_dim=head_dim,
            block_m=base.block_m,
            block_n=base.block_n,
            num_warps=base.num_warps,
            arg_count=17,
            shared_mem_bytes=_smem_bytes_bwd_convert(head_dim, base.block_m),
            function_name="mha_attn_bwd_convert_dq",
            is_even_mn=False,
            is_even_k=True,
            accumulate_in_fp16=False,
        )
    return configs, aux_configs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate FlashAttention kernel binaries required by the runtime contract."
    )
    parser.add_argument(
        "--sm",
        default="sm_89,sm_90,sm_100",
        help="Comma-separated list of SM targets to include (default: sm_89,sm_90,sm_100).",
    )
    parser.add_argument(
        "--head-dim",
        type=int,
        default=128,
        help="Head dimension to generate variants for (default: 128).",
    )
    parser.add_argument(
        "--skip-fetch-sources",
        action="store_true",
        help="Skip downloading the flash-attention sources into third_party.",
    )
    parser.add_argument(
        "--autotune-config",
        action="append",
        dest="autotune_configs",
        default=[],
        help="Path to a JSON file with auto-tuned kernel constants to overlay.",
    )
    args = parser.parse_args()

    if not args.skip_fetch_sources:
        flash_root = ensure_flash_attention_sources(_ROOT_DIR)
        cutlass_root = fetch_cutlass_sources(_ROOT_DIR)
    else:
        flash_root = _ROOT_DIR / "third_party" / "flash-attention"
        cutlass_root = _ROOT_DIR / "third_party" / "cutlass"
        flash_marker = flash_root / "flash_attn" / "__init__.py"
        cutlass_marker = cutlass_root / "include" / "cutlass" / "cutlass.h"
        if not flash_marker.exists():
            raise SystemExit(
                "FlashAttention sources not found under flash_attention_codegen/third_party. "
                "Run without --skip-fetch-sources or run kernels.flash_attention_codegen.fetch_flash_attention."
            )
        if not cutlass_marker.exists():
            raise SystemExit(
                "CUTLASS sources not found under flash_attention_codegen/third_party. "
                "Run without --skip-fetch-sources to fetch a local CUTLASS copy."
            )

    _OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    _GENERATED_ROOT.mkdir(parents=True, exist_ok=True)
    _BUILD_ROOT.mkdir(parents=True, exist_ok=True)

    variants = list(required_flash_attention_variants(head_dim=args.head_dim))
    include_dirs = [
        flash_root / "csrc" / "flash_attn",
        flash_root / "csrc" / "flash_attn" / "src",
        cutlass_root / "include",
        _ROOT_DIR / "third_party" / "aten_stub",
    ]
    autotune_paths = [Path(path) for path in args.autotune_configs]
    autotune_overlays = _load_flash_autotune_overlays(autotune_paths) if autotune_paths else {}
    jobs = int(os.getenv("TRITON_JOBS", "")) if os.getenv("TRITON_JOBS") else (os.cpu_count() or 1)

    for sm in _parse_sm_list(args.sm):
        configs, aux_configs = _build_kernel_configs(args.head_dim, sm, autotune_overlays)

        built_cubins: dict[tuple[str, str, bool, bool], Path] = {}
        built_aux_cubins: dict[tuple[str, str], Path] = {}
        compile_tasks: list[tuple[Path, Path, str, list[Path]]] = []
        for direction in ("fwd", "bwd"):
            for dtype in ("fp16", "bf16"):
                for even_mn in (False, True):
                    accumulate_variants = (False, True) if (direction == "fwd" and dtype == "fp16") else (False,)
                    for accumulate_in_fp16 in accumulate_variants:
                        config = configs[(direction, dtype, even_mn, accumulate_in_fp16)]
                        even_tag = "_even" if even_mn else ""
                        acc_tag = "_acc_fp16" if accumulate_in_fp16 else ""
                        src = _GENERATED_ROOT / f"flash_fa_{direction}_{dtype}{acc_tag}{even_tag}.cu"
                        if direction == "fwd":
                            src.write_text(_emit_fwd_source(config))
                        else:
                            src.write_text(_emit_bwd_source(config, "main"))

                        cubin_name = f"flash_fa_{direction}_{dtype}{acc_tag}{even_tag}_{sm}.cubin"
                        cubin_path = _BUILD_ROOT / cubin_name
                        compile_tasks.append((src, cubin_path, sm, include_dirs))
                        built_cubins[(direction, dtype, even_mn, accumulate_in_fp16)] = cubin_path

        for direction in ("bwd_dot", "bwd_convert"):
            for dtype in ("fp16", "bf16"):
                config = aux_configs[(direction, dtype)]
                src = _GENERATED_ROOT / f"flash_fa_{direction}_{dtype}.cu"
                if direction == "bwd_dot":
                    src.write_text(_emit_bwd_source(config, "dot"))
                else:
                    src.write_text(_emit_bwd_source(config, "convert"))

                cubin_name = f"flash_fa_{direction}_{dtype}_{sm}.cubin"
                cubin_path = _BUILD_ROOT / cubin_name
                compile_tasks.append((src, cubin_path, sm, include_dirs))
                built_aux_cubins[(direction, dtype)] = cubin_path

        _parallel_compile(compile_tasks, jobs)

        sm_suffix = sm.replace("sm_", "sm")
        for variant in variants:
            config = configs[(variant.direction, variant.dtype, variant.even_mn, variant.accumulate_in_fp16)]
            cubin_src = built_cubins[(variant.direction, variant.dtype, variant.even_mn, variant.accumulate_in_fp16)]
            out_name = f"{variant.kernel_name}_{sm_suffix}"
            out_cubin = _OUTPUT_ROOT / f"{out_name}.cubin"
            shutil.copyfile(cubin_src, out_cubin)
            _write_kinfo(_OUTPUT_ROOT / out_name, config)
        for dtype in ("fp16", "bf16"):
            for suffix, kind in (("bwd_dot_do_o", "bwd_dot"), ("bwd_convert_dq", "bwd_convert")):
                config = aux_configs[(kind, dtype)]
                cubin_src = built_aux_cubins[(kind, dtype)]
                base_name = f"mha_full_attn_fa_{suffix}_hs{args.head_dim}_{dtype}"
                out_name = f"{base_name}_{sm_suffix}"
                out_cubin = _OUTPUT_ROOT / f"{out_name}.cubin"
                shutil.copyfile(cubin_src, out_cubin)
                _write_kinfo(_OUTPUT_ROOT / out_name, config)


if __name__ == "__main__":
    main()
