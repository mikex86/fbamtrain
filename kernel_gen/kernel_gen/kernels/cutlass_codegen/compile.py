from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import shutil
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


_ROOT_DIR = Path(__file__).resolve().parent
_OUTPUT_ROOT = _ROOT_DIR.parents[1] / "output"
_BUILD_BASE = _ROOT_DIR / "build"
_CONV_SOURCE_FILE = _ROOT_DIR / "conv2d_cutlass_kernel.cu"
_CONV_DGRAD_SOURCE_FILE = _ROOT_DIR / "conv2d_cutlass_dgrad_kernel.cu"
_CONV_WGRAD_SOURCE_FILE = _ROOT_DIR / "conv2d_cutlass_wgrad_kernel.cu"
_GEMM_SOURCE_FILE = _ROOT_DIR / "gemm_cutlass_kernel.cu"
_MHA_SOURCE_FILE = _ROOT_DIR / "mha_cutlass_kernel.cu"
_MHA_BWD_SOURCE_FILE = _ROOT_DIR / "mha_cutlass_bwd_kernel.cu"
_CUTLASS_ROOT = _ROOT_DIR / "third_party" / "cutlass"
_CUTLASS_HEADER = _CUTLASS_ROOT / "include" / "cutlass" / "cutlass.h"

_CONV_REQUIRED_DEFAULT_DEFINES: dict[str, str] = {
    "CUTLASS_CONV_ENABLE_CONFIG_CHECK": "0",
    "CUTLASS_CONV_OPERATOR_CLASS": "cutlass::arch::OpClassTensorOp",
    "CUTLASS_CONV_ITERATOR_ALGO": "cutlass::conv::IteratorAlgorithm::kOptimized",
    "CUTLASS_CONV_STRIDE_SUPPORT": "cutlass::conv::StrideSupport::kUnity",
    "CUTLASS_CONV_ALIGNMENT_A": "8",
    "CUTLASS_CONV_ALIGNMENT_B": "8",
    "CUTLASS_CONV_THREADBLOCK_M": "128",
    "CUTLASS_CONV_THREADBLOCK_N": "128",
    "CUTLASS_CONV_THREADBLOCK_K": "64",
    "CUTLASS_CONV_WARP_M": "64",
    "CUTLASS_CONV_WARP_N": "64",
    "CUTLASS_CONV_WARP_K": "64",
    "CUTLASS_CONV_INSTRUCTION_M": "16",
    "CUTLASS_CONV_INSTRUCTION_N": "8",
    "CUTLASS_CONV_INSTRUCTION_K": "16",
    "CUTLASS_CONV_NUM_STAGES": "3",
    "CUTLASS_CONV_EXPECT_STRIDE_H": "1",
    "CUTLASS_CONV_EXPECT_STRIDE_W": "1",
    "CUTLASS_CONV_EXPECT_PADDING_H": "1",
    "CUTLASS_CONV_EXPECT_PADDING_W": "1",
    "CUTLASS_CONV_EXPECT_DILATION_H": "1",
    "CUTLASS_CONV_EXPECT_DILATION_W": "1",
    "CUTLASS_CONV_EXPECT_GROUPS": "1",
    "CUTLASS_CONV_EXPECT_KERNEL_H": "3",
    "CUTLASS_CONV_EXPECT_KERNEL_W": "3",
    "CUTLASS_CONV_META_IN_CHANNELS": "0",
    "CUTLASS_CONV_FP16": "0",
    "CUTLASS_CONV_FP16_ACCUM": "0",
}

_GEMM_REQUIRED_DEFAULT_DEFINES: dict[str, str] = {
    "CUTLASS_GEMM_ENABLE_CONFIG_CHECK": "0",
    "CUTLASS_GEMM_OPERATOR_CLASS": "cutlass::arch::OpClassTensorOp",
    "CUTLASS_GEMM_ALIGNMENT_A": "4",
    "CUTLASS_GEMM_ALIGNMENT_B": "4",
    "CUTLASS_GEMM_THREADBLOCK_M": "128",
    "CUTLASS_GEMM_THREADBLOCK_N": "128",
    "CUTLASS_GEMM_THREADBLOCK_K": "64",
    "CUTLASS_GEMM_WARP_M": "64",
    "CUTLASS_GEMM_WARP_N": "64",
    "CUTLASS_GEMM_WARP_K": "64",
    "CUTLASS_GEMM_INSTRUCTION_M": "16",
    "CUTLASS_GEMM_INSTRUCTION_N": "8",
    "CUTLASS_GEMM_INSTRUCTION_K": "16",
    "CUTLASS_GEMM_NUM_STAGES": "3",
    "CUTLASS_GEMM_ACTIVATION": "0",
    "CUTLASS_GEMM_BIAS_OP": "0",
    "CUTLASS_GEMM_SPLIT_K_SLICES": "1",
    "CUTLASS_GEMM_SWIZZLE_SIZE": "1",
    "CUTLASS_GEMM_FP16": "0",
    "CUTLASS_GEMM_FP16_ACCUM": "0",
    "CUTLASS_GEMM_FP32_OUTPUT": "0",
    "CUTLASS_GEMM_WITH_BIAS": "1",
    "CUTLASS_GEMM_TRANSPOSE_A": "0",
    "CUTLASS_GEMM_TRANSPOSE_B": "0",
    "CUTLASS_GEMM_WRITE_OUT_PREACT": "0",
}

_MHA_REQUIRED_DEFAULT_DEFINES: dict[str, str] = {
    "CUTLASS_MHA_FP16": "0",
    "CUTLASS_MHA_HEAD_DIM": "128",
    "CUTLASS_MHA_KEYS_PER_BLOCK": "128",
    "CUTLASS_MHA_IS_ALIGNED": "1",
    "CUTLASS_MHA_SUPPORTS_DROPOUT": "0",
    "CUTLASS_MHA_SUPPORTS_BIAS": "0",
    "CUTLASS_MHA_WRITE_LSE": "0",
}

_MHA_BWD_REQUIRED_DEFAULT_DEFINES: dict[str, str] = {
    "CUTLASS_MHA_BWD_FP16": "0",
    "CUTLASS_MHA_BWD_HEAD_DIM": "128",
    "CUTLASS_MHA_BWD_BLOCK_I": "128",
    "CUTLASS_MHA_BWD_BLOCK_J": "128",
}


def _require_cutlass_sources() -> Path:
    if not _CUTLASS_HEADER.exists():
        raise RuntimeError(
            "CUTLASS sources not found. Run kernels/cutlass_codegen/fetch_cutlass_sources.py "
            "or the top-level compile script to fetch them."
        )
    return _CUTLASS_ROOT


@dataclass(frozen=True)
class CutlassArtifacts:
    build_dir: Path
    ptx: Path
    cubin: Path
    metadata: Path
    sm_suffix: str


@dataclass(frozen=True)
class CutlassKernelMeta:
    num_warps: int
    shared_mem_bytes: int
    block_pixels: int
    block_oc: int
    block_k: int
    in_channels: int
    kernel_h: int
    kernel_w: int
    stride_h: int
    stride_w: int
    padding_h: int
    padding_w: int
    dilation_h: int
    dilation_w: int
    groups: int


@dataclass(frozen=True)
class CutlassGemmKernelMeta:
    num_warps: int
    shared_mem_bytes: int
    block_m: int
    block_n: int
    block_k: int
    swizzle_size: int


@dataclass(frozen=True)
class CutlassMhaKernelMeta:
    num_warps: int
    shared_mem_bytes: int
    block_size_x: int
    block_size_y: int
    head_dim: int


@dataclass(frozen=True)
class CutlassMhaBwdKernelMeta:
    num_warps: int
    shared_mem_bytes: int
    block_size_i: int
    block_size_j: int
    head_dim: int
    gradq_tile_elements: int
    gradq_temp_bytes: int



def _normalise_sm(sm: str | None) -> tuple[str, str, str]:
    sm_str = (sm or "sm_89").strip().lower()
    match = re.fullmatch(r"(?:sm_?|compute_)?(\d+)", sm_str)
    if not match:
        raise ValueError(f"Unrecognised SM version '{sm}'. Expected formats like 'sm_89' or '89'.")
    code = match.group(1)
    return f"sm_{code}", f"compute_{code}", f"sm{code}"


def _select_codegen_arch_for_source(
    *,
    source_file: Path,
    requested_sm: str,
    requested_compute: str,
) -> tuple[str, str]:
    """Select CUDA codegen arch for the source file.

    CUTLASS FMHA example kernels for Hopper/Blackwell gate kernel bodies behind
    architecture feature macros enabled by `sm_90a` / `sm_100a` codegen.
    Compiling those files as plain `sm_90` / `sm_100` can produce stub kernels.
    """
    source_name = source_file.name
    is_mha_source = source_name in {_MHA_SOURCE_FILE.name, _MHA_BWD_SOURCE_FILE.name}
    is_cutlass3_conv_source = source_name in {"conv2d_cutlass3_kernel.cu", "conv2d_cutlass3_wgrad_kernel.cu"}
    if not is_mha_source and not is_cutlass3_conv_source:
        return requested_sm, requested_compute

    sm_num = int(requested_sm.split("_", 1)[1])
    if sm_num == 90:
        return "sm_90a", "compute_90a"
    if sm_num >= 100:
        return "sm_100a", "compute_100a"
    return requested_sm, requested_compute


def _inject_sm_arch_define(
    defines: Mapping[str, str],
    sm: str | None,
    key: str,
    *,
    sm_version_key: str | None = None,
) -> dict[str, str]:
    updated = dict(defines)
    sm_norm, _, _ = _normalise_sm(sm)
    sm_num = int(sm_norm.split("_", 1)[1])
    arch = f"cutlass::arch::Sm{sm_num}"
    existing = updated.get(key)
    if existing is not None and existing != arch:
        raise RuntimeError(f"{key}={existing} does not match target architecture {arch}.")
    updated[key] = arch
    if sm_version_key is not None:
        existing_version = updated.get(sm_version_key)
        if existing_version is not None and int(existing_version) != sm_num:
            raise RuntimeError(
                f"{sm_version_key}={existing_version} does not match target architecture sm_{sm_num}."
            )
        updated[sm_version_key] = str(sm_num)
    return updated


def _with_sm_suffix(basename: str, suffix: str | None) -> str:
    if not suffix:
        return basename
    match = re.match(r"(.+)_sm\d+$", basename)
    if match:
        return f"{match.group(1)}_{suffix}"
    return basename


def _sm_override_keys(sm: str | None) -> tuple[str, ...]:
    sm_norm, _, sm_suffix = _normalise_sm(sm)
    return sm_norm, sm_norm.replace("_", ""), sm_suffix


def _apply_sm_specific_defines(spec: Mapping[str, object], defines: dict[str, str], sm: str | None) -> dict[str, str]:
    overrides = spec.get("defines_by_sm")
    if not isinstance(overrides, Mapping):
        return defines

    updated = dict(defines)
    for key in _sm_override_keys(sm):
        payload = overrides.get(key)
        if payload is None:
            continue
        if not isinstance(payload, Mapping):
            raise SystemExit(f"defines_by_sm.{key} must be a JSON object.")
        updated.update(_stringify_defines(payload))
    return updated


def _source_path_for_spec(spec: Mapping[str, object], default_source: Path, sm: str | None) -> Path:
    source_file_hint = spec.get("source_file")
    source_by_sm = spec.get("source_file_by_sm")
    if isinstance(source_by_sm, Mapping):
        for key in _sm_override_keys(sm):
            if key in source_by_sm:
                source_file_hint = source_by_sm[key]
                break

    if source_file_hint:
        source_path = Path(str(source_file_hint))
        if not source_path.is_absolute():
            source_path = (_ROOT_DIR / source_path).resolve()
        return source_path
    return default_source


def _merge_config_file_entry(existing: Mapping[str, object] | None, spec: Mapping[str, object]) -> dict[str, object]:
    """Merge generated bin metadata into an autotuned config entry.

    Autotune files can already contain a config with the same stable name as a
    generated model-shape bin.  The generated bin entry is still the authority
    for target-specific source/define overrides, because those are packaging
    decisions rather than measured kernel timing constants.
    """
    if existing is None:
        return dict(spec)

    merged = dict(existing)
    for key in ("source_file_by_sm", "defines_by_sm"):
        if key in spec:
            merged[key] = spec[key]
    return merged


def ensure_workdir() -> Path:
    os.chdir(_ROOT_DIR)
    return _ROOT_DIR


def _stringify_defines(defines: Mapping[str, object]) -> dict[str, str]:
    return {str(key): str(value) for key, value in defines.items()}


def _apply_required_cutlass_defines(
    defines: Mapping[str, object],
    *,
    category: str,
    sm: str | None,
) -> dict[str, str]:
    updated = _stringify_defines(defines)
    if category in {"conv2d", "conv2d_dgrad", "conv2d_wgrad"}:
        for key, value in _CONV_REQUIRED_DEFAULT_DEFINES.items():
            updated.setdefault(key, value)
        if category == "conv2d":
            updated.setdefault("CUTLASS_CONV_OP", "cutlass::conv::Operator::kFprop")
        elif category == "conv2d_dgrad":
            updated.setdefault("CUTLASS_CONV_OP", "cutlass::conv::Operator::kDgrad")
        else:
            updated.setdefault("CUTLASS_CONV_OP", "cutlass::conv::Operator::kWgrad")
        return updated

    if category == "gemm":
        for key, value in _GEMM_REQUIRED_DEFAULT_DEFINES.items():
            updated.setdefault(key, value)
        if "CUTLASS_GEMM_ELEMENTS_PER_THREAD" not in updated:
            updated["CUTLASS_GEMM_ELEMENTS_PER_THREAD"] = (
                "4" if updated.get("CUTLASS_GEMM_FP32_OUTPUT") == "1" else "8"
            )
        return updated

    if category == "mha":
        for key, value in _MHA_REQUIRED_DEFAULT_DEFINES.items():
            updated.setdefault(key, value)
        _ = _normalise_sm(sm)
        updated.setdefault("CUTLASS_MHA_QUERIES_PER_BLOCK", "32")
        return updated

    if category == "mha_bwd":
        for key, value in _MHA_BWD_REQUIRED_DEFAULT_DEFINES.items():
            updated.setdefault(key, value)
        return updated

    raise ValueError(f"Unknown CUTLASS category '{category}'.")


def _load_autotune_payload(path: Path) -> dict[str, object]:
    try:
        text = path.read_text()
    except OSError as exc:
        raise SystemExit(f"Failed to read autotune config '{path}': {exc}") from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Autotune config '{path}' is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"Autotune config '{path}' must contain a JSON object at the top level.")
    return payload


def _load_autotune_entries(paths: Sequence[Path], category: str) -> dict[str, dict[str, object]]:
    entries: dict[str, dict[str, object]] = {}
    sources: dict[str, Path] = {}
    for path in paths:
        payload = _load_autotune_payload(path)
        if category not in payload:
            continue
        section = payload[category]
        if not isinstance(section, Mapping):
            raise SystemExit(f"Autotune config section '{category}' must be a JSON object.")
        for name, entry in section.items():
            if not isinstance(name, str):
                raise SystemExit(f"Autotune config keys must be strings (got {name!r}).")
            if name in entries:
                raise SystemExit(
                    f"Duplicate autotune entry '{name}' in '{category}' "
                    f"(found in {sources[name]} and {path})."
                )
            if not isinstance(entry, Mapping):
                raise SystemExit(f"Autotune entry for '{name}' must be a JSON object.")
            entry_dict = dict(entry)
            defines_payload = entry_dict.pop("defines", None)
            if defines_payload is None:
                raise SystemExit(f"Autotune entry '{name}' in '{path}' is missing a 'defines' mapping.")
            if not isinstance(defines_payload, Mapping):
                raise SystemExit(f"Autotune entry '{name}' defines must be a JSON object.")
            entry_dict["defines"] = _stringify_defines(defines_payload)
            entry_dict.setdefault("basename", name)
            entry_dict.setdefault("enabled", True)
            entry_dict["autotuned"] = True
            entry_dict["autotune_source"] = str(path)
            entries[name] = entry_dict
            sources[name] = path
    return entries


def _load_conv2d_config_entries(path: Path, category: str = "conv2d") -> dict[str, dict[str, object]]:
    try:
        text = path.read_text()
    except OSError as exc:
        raise SystemExit(f"Failed to read conv2d config file '{path}': {exc}") from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Conv2d config '{path}' is not valid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise SystemExit(f"Conv2d config '{path}' must contain a JSON object at the top level.")
    conv2d_section = payload.get(category, {})
    if not isinstance(conv2d_section, Mapping):
        raise SystemExit(f"Conv2d config '{path}' expects '{category}' to be a JSON object.")
    entries: dict[str, dict[str, object]] = {}
    for name, spec in conv2d_section.items():
        if not isinstance(spec, Mapping):
            raise SystemExit(f"Conv2d config '{path}' entry '{category}.{name}' must be a JSON object.")
        entries[str(name)] = dict(spec)
    return entries


def _run(cmd: list[str], *, verbose: bool = False, cwd: Path | None = None) -> None:
    if verbose:
        print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


def _nvcc_version() -> str:
    try:
        result = subprocess.run(["nvcc", "--version"], check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:  # pragma: no cover - developer environment issue
        raise RuntimeError("nvcc compiler not found in PATH") from exc
    return result.stdout.strip()


def _source_digest(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(data).hexdigest()


def _compute_env_hash(
    *,
    cutlass_root: Path,
    source_file: Path,
    defines: Optional[dict[str, str]] = None,
    sm: str = "sm_89",
) -> str:
    nvcc_ver = _nvcc_version()
    source_hash = _source_digest(source_file)
    env_parts = [
        nvcc_ver,
        sm,
        str(cutlass_root.resolve()),
        source_file.name,
        source_hash,
    ]
    if defines:
        for key in sorted(defines):
            env_parts.append(f"{key}={defines[key]}")

    env_key = "|".join(env_parts)
    return hashlib.sha1(env_key.encode("utf-8")).hexdigest()[:10]


def _require_define(defines: dict[str, str], key: str) -> str:
    raw = defines.get(key)
    if raw is None or raw == "":
        raise RuntimeError(f"CUTLASS define '{key}' is required but missing.")
    return raw


def _int_define(defines: dict[str, str], key: str) -> int:
    raw = _require_define(defines, key)
    try:
        return int(raw)
    except ValueError as exc:  # pragma: no cover - configuration error
        raise RuntimeError(f"CUTLASS define '{key}' must be an integer (got {raw!r})") from exc


def _conv_kernel_meta_from_defines(
    defines: dict[str, str], *, shared_mem_bytes: int, num_warps: Optional[int] = None
) -> CutlassKernelMeta:
    stride_h = _int_define(defines, "CUTLASS_CONV_EXPECT_STRIDE_H")
    stride_w = _int_define(defines, "CUTLASS_CONV_EXPECT_STRIDE_W")
    padding_h = _int_define(defines, "CUTLASS_CONV_EXPECT_PADDING_H")
    padding_w = _int_define(defines, "CUTLASS_CONV_EXPECT_PADDING_W")
    dilation_h = _int_define(defines, "CUTLASS_CONV_EXPECT_DILATION_H")
    dilation_w = _int_define(defines, "CUTLASS_CONV_EXPECT_DILATION_W")
    kernel_h = _int_define(defines, "CUTLASS_CONV_EXPECT_KERNEL_H")
    kernel_w = _int_define(defines, "CUTLASS_CONV_EXPECT_KERNEL_W")
    in_channels_raw = _int_define(defines, "CUTLASS_CONV_META_IN_CHANNELS")
    in_channels = in_channels_raw if in_channels_raw > 0 else 0
    groups = max(1, _int_define(defines, "CUTLASS_CONV_EXPECT_GROUPS"))

    block_pixels = _int_define(defines, "CUTLASS_CONV_THREADBLOCK_M")
    block_oc = _int_define(defines, "CUTLASS_CONV_THREADBLOCK_N")
    block_k = _int_define(defines, "CUTLASS_CONV_THREADBLOCK_K")

    warp_m = _int_define(defines, "CUTLASS_CONV_WARP_M")
    warp_n = _int_define(defines, "CUTLASS_CONV_WARP_N")
    warp_k = _int_define(defines, "CUTLASS_CONV_WARP_K")

    threadblock_volume = block_pixels * block_oc * block_k
    warp_volume = warp_m * warp_n * warp_k
    if warp_volume <= 0 or threadblock_volume % warp_volume != 0:
        raise RuntimeError("Invalid CUTLASS threadblock/warp configuration.")
    derived_warps = threadblock_volume // warp_volume
    if num_warps is None or num_warps <= 0:
        num_warps = derived_warps

    return CutlassKernelMeta(
        num_warps=num_warps,
        shared_mem_bytes=shared_mem_bytes,
        block_pixels=block_pixels,
        block_oc=block_oc,
        block_k=block_k,
        in_channels=in_channels,
        kernel_h=kernel_h,
        kernel_w=kernel_w,
        stride_h=stride_h,
        stride_w=stride_w,
        padding_h=padding_h,
        padding_w=padding_w,
        dilation_h=dilation_h,
        dilation_w=dilation_w,
        groups=groups,
    )

def _gemm_kernel_meta_from_defines(
    defines: dict[str, str], *, shared_mem_bytes: int, num_warps: Optional[int]
) -> CutlassGemmKernelMeta:
    block_m = _int_define(defines, "CUTLASS_GEMM_THREADBLOCK_M")
    block_n = _int_define(defines, "CUTLASS_GEMM_THREADBLOCK_N")
    block_k = _int_define(defines, "CUTLASS_GEMM_THREADBLOCK_K")
    swizzle_size = _int_define(defines, "CUTLASS_GEMM_SWIZZLE_SIZE")

    if num_warps is None or num_warps <= 0:
        warp_m = _int_define(defines, "CUTLASS_GEMM_WARP_M")
        warp_n = _int_define(defines, "CUTLASS_GEMM_WARP_N")
        warp_k = _int_define(defines, "CUTLASS_GEMM_WARP_K")
        threadblock_volume = block_m * block_n * block_k
        warp_volume = warp_m * warp_n * warp_k
        if warp_volume <= 0 or threadblock_volume % warp_volume != 0:
            raise RuntimeError("Invalid CUTLASS GEMM threadblock configuration.")
        num_warps = threadblock_volume // warp_volume
        if num_warps <= 0:
            raise RuntimeError("Derived CUTLASS GEMM warp count must be positive.")

    return CutlassGemmKernelMeta(
        num_warps=num_warps,
        shared_mem_bytes=shared_mem_bytes,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        swizzle_size=swizzle_size,
    )


def _mha_kernel_meta_from_defines(
    defines: dict[str, str], *, shared_mem_bytes: int, num_warps: Optional[int]
) -> CutlassMhaKernelMeta:
    block_size_x = _int_define(defines, "CUTLASS_MHA_QUERIES_PER_BLOCK")
    block_size_y = _int_define(defines, "CUTLASS_MHA_KEYS_PER_BLOCK")
    head_dim = _int_define(defines, "CUTLASS_MHA_HEAD_DIM")

    if block_size_x <= 0 or block_size_y <= 0 or head_dim <= 0:
        raise RuntimeError("Invalid CUTLASS MHA kernel configuration derived from defines.")

    if num_warps is None or num_warps <= 0:
        raise RuntimeError("CUTLASS MHA metadata must provide a positive warp count.")

    return CutlassMhaKernelMeta(
        num_warps=num_warps,
        shared_mem_bytes=shared_mem_bytes,
        block_size_x=block_size_x,
        block_size_y=block_size_y,
        head_dim=head_dim,
    )


def _mha_bwd_kernel_meta_from_defines(
    defines: dict[str, str],
    *,
    shared_mem_bytes: int,
    num_warps: Optional[int],
    gradq_tile_elements: int,
    gradq_temp_bytes: int,
) -> CutlassMhaBwdKernelMeta:
    block_size_i = _int_define(defines, "CUTLASS_MHA_BWD_BLOCK_I")
    block_size_j = _int_define(defines, "CUTLASS_MHA_BWD_BLOCK_J")
    head_dim = _int_define(defines, "CUTLASS_MHA_BWD_HEAD_DIM")

    if block_size_i <= 0 or block_size_j <= 0 or head_dim <= 0:
        raise RuntimeError("Invalid CUTLASS MHA bwd kernel configuration derived from defines.")

    if num_warps is None or num_warps <= 0:
        raise RuntimeError("CUTLASS MHA bwd metadata must provide a positive warp count.")

    return CutlassMhaBwdKernelMeta(
        num_warps=num_warps,
        shared_mem_bytes=shared_mem_bytes,
        block_size_i=block_size_i,
        block_size_j=block_size_j,
        head_dim=head_dim,
        gradq_tile_elements=gradq_tile_elements,
        gradq_temp_bytes=gradq_temp_bytes,
    )




def _build_needed(targets: list[Path], source_file: Path) -> bool:
    if any(not t.exists() for t in targets):
        return True
    source_mtime = source_file.stat().st_mtime
    return any(t.stat().st_mtime < source_mtime for t in targets)


def _extract_cubin_metadata(cubin_path: Path) -> tuple[int, int]:
    result = subprocess.run(
        ["cuobjdump", "--dump-elf", str(cubin_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    lines = result.stdout.splitlines()
    for idx, line in enumerate(lines):
        if line.strip() != ".nv.constant3":
            continue
        for j in range(idx + 1, len(lines)):
            tokens = lines[j].strip().split()
            if not tokens:
                continue
            values = [int(tok, 16) for tok in tokens if tok.startswith("0x")]
            if len(values) >= 2:
                return values[0], values[1]
            break
    raise RuntimeError(f"Failed to extract constant metadata from {cubin_path}")


def build_cutlass_artifacts(
    *,
    verbose: bool = False,
    defines: Optional[dict[str, str]] = None,
    sm: str = "sm_89",
    source_file: Path = _CONV_SOURCE_FILE,
) -> CutlassArtifacts:
    ensure_workdir()
    cutlass_root = _require_cutlass_sources()

    defines = {k: str(v) for k, v in (defines or {}).items()}

    arch_sm, arch_compute, arch_suffix = _normalise_sm(sm)
    codegen_sm, codegen_compute = _select_codegen_arch_for_source(
        source_file=source_file,
        requested_sm=arch_sm,
        requested_compute=arch_compute,
    )

    env_hash = _compute_env_hash(
        cutlass_root=cutlass_root,
        source_file=source_file,
        defines=defines,
        sm=f"{arch_sm}|{codegen_sm}|{codegen_compute}",
    )
    build_dir = _BUILD_BASE / env_hash
    build_dir.mkdir(parents=True, exist_ok=True)

    stem = source_file.stem
    ptx_path = build_dir / f"{stem}.ptx"
    cubin_path = build_dir / f"{stem}.cubin"
    metadata_path = build_dir / "metadata.json"

    sm_num = int(arch_sm.split("_", 1)[1])
    include_flags = [
        f"-I{cutlass_root / 'include'}",
        f"-I{cutlass_root / 'tools' / 'util' / 'include'}",
    ]
    if sm_num >= 100:
        include_flags.append(f"-I{cutlass_root / 'examples' / '77_blackwell_fmha'}")
    elif sm_num >= 90:
        include_flags.append(f"-I{cutlass_root / 'examples' / '88_hopper_fmha'}")
    common_flags = ["-std=c++17", "-O3", "-DNDEBUG", "--expt-relaxed-constexpr", "--expt-extended-lambda", "--use_fast_math"]
    define_flags = [f"-D{key}={value}" if value else f"-D{key}" for key, value in sorted(defines.items())]

    # PTX generation via nvcc
    if _build_needed([ptx_path], source_file):
        _run(
            [
                "nvcc",
                *common_flags,
                *define_flags,
                # Use real target arch (not virtual compute arch) so CUTLASS
                # feature macros like __CUDA_ARCH_FEAT_SM90_ALL are available
                # during device compilation of FMHA example kernels.
                f"-arch={codegen_sm}",
                *include_flags,
                "-ptx",
                str(source_file),
                "-o",
                str(ptx_path),
            ],
            verbose=verbose,
            cwd=_ROOT_DIR,
        )

    # ptxas to cubin
    metadata_values: tuple[int, int] | None = None

    if _build_needed([cubin_path], source_file) or not metadata_path.exists():
        _run(
            [
                "ptxas",
                str(ptx_path),
                "-o",
                str(cubin_path),
                f"-arch={codegen_sm}",
            ],
            verbose=verbose,
            cwd=_ROOT_DIR,
        )
        metadata_values = _extract_cubin_metadata(cubin_path)

    # Ensure metadata exists even when cubin is up to date
    if metadata_values is None and not metadata_path.exists():
        metadata_values = _extract_cubin_metadata(cubin_path)

    if metadata_values is not None:
        shared_mem_bytes, num_warps = metadata_values
        metadata = {"shared_mem_bytes": int(shared_mem_bytes), "num_warps": int(num_warps)}
        metadata_path.write_text(json.dumps(metadata))
    elif metadata_path.exists():
        # Ensure metadata is valid JSON; raise if corrupt
        try:
            _ = json.loads(metadata_path.read_text())
        except json.JSONDecodeError as exc:  # pragma: no cover - corrupted metadata
            raise RuntimeError(f"Invalid metadata file: {metadata_path}") from exc
    else:  # pragma: no cover - should not happen
        raise RuntimeError("Missing CUTLASS metadata; please rebuild artifacts")

    legacy_helper_path = build_dir / f"lib{stem}.so"
    if legacy_helper_path.exists():
        try:
            legacy_helper_path.unlink()
        except OSError:
            pass

    return CutlassArtifacts(
        build_dir=build_dir,
        ptx=ptx_path,
        cubin=cubin_path,
        metadata=metadata_path,
        sm_suffix=arch_suffix,
    )


def _write_conv_kinfo(path: Path, *, function_name: str, meta: CutlassKernelMeta, arg_count: int) -> None:
    prefix_token = "\\prefix\\()"

    def sym(name: str) -> str:
        return f"{prefix_token}_{name}"

    def emit_numeric(name: str, value: int) -> list[str]:
        return [
            "",
            "\t.p2align 3",
            "#if defined(__ELF__)",
            f"\t.type {sym(name)}, @object",
            "#endif",
            f"{sym(name)}:",
            f"\t.quad {value}",
            "#if defined(__ELF__)",
            f"\t.size {sym(name)}, .-{sym(name)}",
            "#endif",
        ]

    lines: list[str] = [
        "/* auto-generated: DO NOT EDIT */",
        "#ifndef DECLARE_KINFO_SEEN",
        "#define DECLARE_KINFO_SEEN 1",
        "",
        "#else",
        ".purgem DECLARE_KINFO",
        "#endif",
        "",
        ".macro DECLARE_KINFO prefix, align=3",
        "",
        "#if defined(__ELF__)",
        "\t.section .rodata.kernels,\"a\",@progbits",
        "#elif defined(__MACH__)",
        "\t.section __TEXT,__const",
        "#else",
        "\t.section .rdata",
        "#endif",
        "",
        "\t.p2align \\align",
        "",
    ]

    exported_symbols = [
        "function_name",
        "num_warps",
        "smem_bytes",
        "arg_count",
        "global_scratch_size",
        "block_pixels",
        "block_oc",
        "block_k",
        "in_channels",
        "kernel_h",
        "kernel_w",
        "stride_h",
        "stride_w",
        "padding_h",
        "padding_w",
        "dilation_h",
        "dilation_w",
        "groups",
    ]
    lines.extend(f"\t.globl {sym(name)}" for name in exported_symbols)
    lines.extend(
        [
            "",
            "\t.p2align 0",
            "#if defined(__ELF__)",
            f"\t.type {sym('function_name')}, @object",
            "#endif",
            f"{sym('function_name')}:",
            f'\t.asciz "{function_name}"',
            "#if defined(__ELF__)",
            f"\t.size {sym('function_name')}, .-{sym('function_name')}",
            "#endif",
        ]
    )

    numeric_fields = [
        ("num_warps", meta.num_warps),
        ("smem_bytes", meta.shared_mem_bytes),
        ("arg_count", arg_count),
        ("global_scratch_size", 0),
        ("block_pixels", meta.block_pixels),
        ("block_oc", meta.block_oc),
        ("block_k", meta.block_k),
        ("in_channels", meta.in_channels),
        ("kernel_h", meta.kernel_h),
        ("kernel_w", meta.kernel_w),
        ("stride_h", meta.stride_h),
        ("stride_w", meta.stride_w),
        ("padding_h", meta.padding_h),
        ("padding_w", meta.padding_w),
        ("dilation_h", meta.dilation_h),
        ("dilation_w", meta.dilation_w),
        ("groups", meta.groups),
    ]
    for name, value in numeric_fields:
        lines.extend(emit_numeric(name, value))

    lines.extend(
        [
            "",
            ".endm",
            "",
        ]
    )

    path.write_text("\n".join(lines))


def _write_gemm_kinfo(path: Path, *, function_name: str, meta: CutlassGemmKernelMeta, arg_count: int) -> None:
    lines = [
        "/* auto-generated: DO NOT EDIT */",
        "#ifndef DECLARE_KINFO_SEEN",
        "#define DECLARE_KINFO_SEEN 1",
        "",
        "#else",
        ".purgem DECLARE_KINFO",
        "#endif",
        "",
        ".macro DECLARE_KINFO prefix, align=3",
        "",
        "#if defined(__ELF__)",
        "	.section .rodata.kernels,\"a\",@progbits",
        "#elif defined(__MACH__)",
        "	.section __TEXT,__const",
        "#else",
        "	.section .rdata",
        "#endif",
        "",
        "	.p2align " r"\align",
        "",
        "	" r".globl \prefix\()_function_name",
        "	" r".globl \prefix\()_num_warps",
        "	" r".globl \prefix\()_smem_bytes",
        "	" r".globl \prefix\()_arg_count",
        "	" r".globl \prefix\()_global_scratch_size",
        "	" r".globl \prefix\()_block_size_m",
        "	" r".globl \prefix\()_block_size_n",
        "	" r".globl \prefix\()_block_size_k",
        "	" r".globl \prefix\()_swizzle_size",
        "",
        "	.p2align 0",
        "#if defined(__ELF__)",
        "	" r".type \prefix\()_function_name, @object",
        "#endif",
        r"\prefix\()_function_name:",
        f"	.asciz \"{function_name}\"",
        "#if defined(__ELF__)",
        "	" r".size \prefix\()_function_name, .-\prefix\()_function_name",
        "#endif",
        "",
        "	.p2align 3",
        "#if defined(__ELF__)",
        "	" r".type \prefix\()_num_warps, @object",
        "#endif",
        r"\prefix\()_num_warps:",
        f"	.quad {meta.num_warps}",
        "#if defined(__ELF__)",
        "	" r".size \prefix\()_num_warps, .-\prefix\()_num_warps",
        "#endif",
        "",
        "	.p2align 3",
        "#if defined(__ELF__)",
        "	" r".type \prefix\()_smem_bytes, @object",
        "#endif",
        r"\prefix\()_smem_bytes:",
        f"	.quad {meta.shared_mem_bytes}",
        "#if defined(__ELF__)",
        "	" r".size \prefix\()_smem_bytes, .-\prefix\()_smem_bytes",
        "#endif",
        "",
        "	.p2align 3",
        "#if defined(__ELF__)",
        "	" r".type \prefix\()_arg_count, @object",
        "#endif",
        r"\prefix\()_arg_count:",
        f"	.quad {arg_count}",
        "#if defined(__ELF__)",
        "	" r".size \prefix\()_arg_count, .-\prefix\()_arg_count",
        "#endif",
        "",
        "	.p2align 3",
        "#if defined(__ELF__)",
        "	" r".type \prefix\()_global_scratch_size, @object",
        "#endif",
        r"\prefix\()_global_scratch_size:",
        "	.quad 0",
        "#if defined(__ELF__)",
        "	" r".size \prefix\()_global_scratch_size, .-\prefix\()_global_scratch_size",
        "#endif",
        "",
        "	.p2align 3",
        "#if defined(__ELF__)",
        "	" r".type \prefix\()_block_size_m, @object",
        "#endif",
        r"\prefix\()_block_size_m:",
        f"	.quad {meta.block_m}",
        "#if defined(__ELF__)",
        "	" r".size \prefix\()_block_size_m, .-\prefix\()_block_size_m",
        "#endif",
        "",
        "	.p2align 3",
        "#if defined(__ELF__)",
        "	" r".type \prefix\()_block_size_n, @object",
        "#endif",
        r"\prefix\()_block_size_n:",
        f"	.quad {meta.block_n}",
        "#if defined(__ELF__)",
        "	" r".size \prefix\()_block_size_n, .-\prefix\()_block_size_n",
        "#endif",
        "",
        "	.p2align 3",
        "#if defined(__ELF__)",
        "	" r".type \prefix\()_block_size_k, @object",
        "#endif",
        r"\prefix\()_block_size_k:",
        f"	.quad {meta.block_k}",
        "#if defined(__ELF__)",
        "	" r".size \prefix\()_block_size_k, .-\prefix\()_block_size_k",
        "#endif",
        "",
        "	.p2align 3",
        "#if defined(__ELF__)",
        "	" r".type \prefix\()_swizzle_size, @object",
        "#endif",
        r"\prefix\()_swizzle_size:",
        f"	.quad {meta.swizzle_size}",
        "#if defined(__ELF__)",
        "	" r".size \prefix\()_swizzle_size, .-\prefix\()_swizzle_size",
        "#endif",
        "",
        ".endm",
        "",
    ]
    path.write_text("\n".join(lines))


def _write_mha_kinfo(path: Path, *, function_name: str, meta: CutlassMhaKernelMeta, arg_count: int) -> None:
    lines = [
        "/* auto-generated: DO NOT EDIT */",
        "#ifndef DECLARE_KINFO_SEEN",
        "#define DECLARE_KINFO_SEEN 1",
        "",
        "#else",
        ".purgem DECLARE_KINFO",
        "#endif",
        "",
        ".macro DECLARE_KINFO prefix, align=3",
        "",
        "#if defined(__ELF__)",
        "\t.section .rodata.kernels,\"a\",@progbits",
        "#elif defined(__MACH__)",
        "\t.section __TEXT,__const",
        "#else",
        "\t.section .rdata",
        "#endif",
        "",
        "\t.p2align \\align",
        "",
        "\t.globl \\prefix\\()_function_name",
        "\t.globl \\prefix\\()_num_warps",
        "\t.globl \\prefix\\()_smem_bytes",
        "\t.globl \\prefix\\()_arg_count",
        "\t.globl \\prefix\\()_global_scratch_size",
        "\t.globl \\prefix\\()_block_size_x",
        "\t.globl \\prefix\\()_block_size_y",
        "\t.globl \\prefix\\()_head_dim",
        "",
        "\t.p2align 0",
        "#if defined(__ELF__)",
        "\t.type \\prefix\\()_function_name, @object",
        "#endif",
        "\\prefix\\()_function_name:",
        f"\t.asciz \"{function_name}\"",
        "#if defined(__ELF__)",
        "\t.size \\prefix\\()_function_name, .-\\prefix\\()_function_name",
        "#endif",
        "",
        "\t.p2align 3",
        "#if defined(__ELF__)",
        "\t.type \\prefix\\()_num_warps, @object",
        "#endif",
        "\\prefix\\()_num_warps:",
        f"\t.quad {meta.num_warps}",
        "#if defined(__ELF__)",
        "\t.size \\prefix\\()_num_warps, .-\\prefix\\()_num_warps",
        "#endif",
        "",
        "\t.p2align 3",
        "#if defined(__ELF__)",
        "\t.type \\prefix\\()_smem_bytes, @object",
        "#endif",
        "\\prefix\\()_smem_bytes:",
        f"\t.quad {meta.shared_mem_bytes}",
        "#if defined(__ELF__)",
        "\t.size \\prefix\\()_smem_bytes, .-\\prefix\\()_smem_bytes",
        "#endif",
        "",
        "\t.p2align 3",
        "#if defined(__ELF__)",
        "\t.type \\prefix\\()_arg_count, @object",
        "#endif",
        "\\prefix\\()_arg_count:",
        f"\t.quad {arg_count}",
        "#if defined(__ELF__)",
        "\t.size \\prefix\\()_arg_count, .-\\prefix\\()_arg_count",
        "#endif",
        "",
        "\t.p2align 3",
        "#if defined(__ELF__)",
        "\t.type \\prefix\\()_global_scratch_size, @object",
        "#endif",
        "\\prefix\\()_global_scratch_size:",
        "\t.quad 0",
        "#if defined(__ELF__)",
        "\t.size \\prefix\\()_global_scratch_size, .-\\prefix\\()_global_scratch_size",
        "#endif",
        "",
        "\t.p2align 3",
        "#if defined(__ELF__)",
        "\t.type \\prefix\\()_block_size_x, @object",
        "#endif",
        "\\prefix\\()_block_size_x:",
        f"\t.quad {meta.block_size_x}",
        "#if defined(__ELF__)",
        "\t.size \\prefix\\()_block_size_x, .-\\prefix\\()_block_size_x",
        "#endif",
        "",
        "\t.p2align 3",
        "#if defined(__ELF__)",
        "\t.type \\prefix\\()_block_size_y, @object",
        "#endif",
        "\\prefix\\()_block_size_y:",
        f"\t.quad {meta.block_size_y}",
        "#if defined(__ELF__)",
        "\t.size \\prefix\\()_block_size_y, .-\\prefix\\()_block_size_y",
        "#endif",
        "",
        "\t.p2align 3",
        "#if defined(__ELF__)",
        "\t.type \\prefix\\()_head_dim, @object",
        "#endif",
        "\\prefix\\()_head_dim:",
        f"\t.quad {meta.head_dim}",
        "#if defined(__ELF__)",
        "\t.size \\prefix\\()_head_dim, .-\\prefix\\()_head_dim",
        "#endif",
        "",
        ".endm",
        "",
    ]
    path.write_text("\n".join(lines))


def _write_mha_bwd_kinfo(path: Path, *, function_name: str, meta: CutlassMhaBwdKernelMeta, arg_count: int) -> None:
    lines = [
        "/* auto-generated: DO NOT EDIT */",
        "#ifndef DECLARE_KINFO_SEEN",
        "#define DECLARE_KINFO_SEEN 1",
        "",
        "#else",
        ".purgem DECLARE_KINFO",
        "#endif",
        "",
        ".macro DECLARE_KINFO prefix, align=3",
        "",
        "#if defined(__ELF__)",
        "\t.section .rodata.kernels,\"a\",@progbits",
        "#elif defined(__MACH__)",
        "\t.section __TEXT,__const",
        "#else",
        "\t.section .rdata",
        "#endif",
        "",
        "\t.p2align \\align",
        "",
        "\t.globl \\prefix\\()_function_name",
        "\t.globl \\prefix\\()_num_warps",
        "\t.globl \\prefix\\()_smem_bytes",
        "\t.globl \\prefix\\()_arg_count",
        "\t.globl \\prefix\\()_global_scratch_size",
        "\t.globl \\prefix\\()_block_size_i",
        "\t.globl \\prefix\\()_block_size_j",
        "\t.globl \\prefix\\()_head_dim",
        "\t.globl \\prefix\\()_gradq_tile_elements",
        "\t.globl \\prefix\\()_gradq_temp_bytes",
        "",
        "\t.p2align 0",
        "#if defined(__ELF__)",
        "\t.type \\prefix\\()_function_name, @object",
        "#endif",
        "\\prefix\\()_function_name:",
        f"\t.asciz \"{function_name}\"",
        "#if defined(__ELF__)",
        "\t.size \\prefix\\()_function_name, .-\\prefix\\()_function_name",
        "#endif",
        "",
        "\t.p2align 3",
        "#if defined(__ELF__)",
        "\t.type \\prefix\\()_num_warps, @object",
        "#endif",
        "\\prefix\\()_num_warps:",
        f"\t.quad {meta.num_warps}",
        "#if defined(__ELF__)",
        "\t.size \\prefix\\()_num_warps, .-\\prefix\\()_num_warps",
        "#endif",
        "",
        "\t.p2align 3",
        "#if defined(__ELF__)",
        "\t.type \\prefix\\()_smem_bytes, @object",
        "#endif",
        "\\prefix\\()_smem_bytes:",
        f"\t.quad {meta.shared_mem_bytes}",
        "#if defined(__ELF__)",
        "\t.size \\prefix\\()_smem_bytes, .-\\prefix\\()_smem_bytes",
        "#endif",
        "",
        "\t.p2align 3",
        "#if defined(__ELF__)",
        "\t.type \\prefix\\()_arg_count, @object",
        "#endif",
        "\\prefix\\()_arg_count:",
        f"\t.quad {arg_count}",
        "#if defined(__ELF__)",
        "\t.size \\prefix\\()_arg_count, .-\\prefix\\()_arg_count",
        "#endif",
        "",
        "\t.p2align 3",
        "#if defined(__ELF__)",
        "\t.type \\prefix\\()_global_scratch_size, @object",
        "#endif",
        "\\prefix\\()_global_scratch_size:",
        "\t.quad 0",
        "#if defined(__ELF__)",
        "\t.size \\prefix\\()_global_scratch_size, .-\\prefix\\()_global_scratch_size",
        "#endif",
        "",
        "\t.p2align 3",
        "#if defined(__ELF__)",
        "\t.type \\prefix\\()_block_size_i, @object",
        "#endif",
        "\\prefix\\()_block_size_i:",
        f"\t.quad {meta.block_size_i}",
        "#if defined(__ELF__)",
        "\t.size \\prefix\\()_block_size_i, .-\\prefix\\()_block_size_i",
        "#endif",
        "",
        "\t.p2align 3",
        "#if defined(__ELF__)",
        "\t.type \\prefix\\()_block_size_j, @object",
        "#endif",
        "\\prefix\\()_block_size_j:",
        f"\t.quad {meta.block_size_j}",
        "#if defined(__ELF__)",
        "\t.size \\prefix\\()_block_size_j, .-\\prefix\\()_block_size_j",
        "#endif",
        "",
        "\t.p2align 3",
        "#if defined(__ELF__)",
        "\t.type \\prefix\\()_head_dim, @object",
        "#endif",
        "\\prefix\\()_head_dim:",
        f"\t.quad {meta.head_dim}",
        "#if defined(__ELF__)",
        "\t.size \\prefix\\()_head_dim, .-\\prefix\\()_head_dim",
        "#endif",
        "",
        "\t.p2align 3",
        "#if defined(__ELF__)",
        "\t.type \\prefix\\()_gradq_tile_elements, @object",
        "#endif",
        "\\prefix\\()_gradq_tile_elements:",
        f"\t.quad {meta.gradq_tile_elements}",
        "#if defined(__ELF__)",
        "\t.size \\prefix\\()_gradq_tile_elements, .-\\prefix\\()_gradq_tile_elements",
        "#endif",
        "",
        "\t.p2align 3",
        "#if defined(__ELF__)",
        "\t.type \\prefix\\()_gradq_temp_bytes, @object",
        "#endif",
        "\\prefix\\()_gradq_temp_bytes:",
        f"\t.quad {meta.gradq_temp_bytes}",
        "#if defined(__ELF__)",
        "\t.size \\prefix\\()_gradq_temp_bytes, .-\\prefix\\()_gradq_temp_bytes",
        "#endif",
        "",
        ".endm",
        "",
    ]
    path.write_text("\n".join(lines))


def _parse_constant_from_ptx(ptx_path: Path, symbol: str) -> int:
    pattern = f"{symbol} ="
    for line in ptx_path.read_text().splitlines():
        line = line.strip()
        if not line or pattern not in line:
            continue
        tokens = line.replace(";", "").split()
        try:
            idx = tokens.index("=")
        except ValueError:
            continue
        if idx + 1 >= len(tokens):
            continue
        value_token = tokens[idx + 1]
        try:
            return int(value_token, 0)
        except ValueError:
            continue
    raise RuntimeError(f"Failed to parse constant '{symbol}' from {ptx_path}")


def _parse_entry_param_count(ptx_path: Path, function_name: str) -> int:
    text = ptx_path.read_text()
    match = re.search(rf"\.entry\s+{re.escape(function_name)}\s*\((.*?)\)", text, re.S)
    if not match:
        return 0
    return len(re.findall(r"\.param", match.group(1)))



def package_cutlass_conv_artifacts(
    *,
    artifacts: CutlassArtifacts,
    basename: str,
    defines: Optional[dict[str, str]] = None,
    sm_suffix: Optional[str] = None,
) -> str:
    defines = {k: str(v) for k, v in (defines or {}).items()}
    metadata = json.loads(artifacts.metadata.read_text())
    shared_mem_bytes = int(metadata.get("shared_mem_bytes", 0))
    if shared_mem_bytes <= 0:
        raise RuntimeError("CUTLASS metadata missing shared memory usage")
    num_warps_override = int(metadata.get("num_warps", 0))

    if sm_suffix is None:
        sm_suffix = artifacts.sm_suffix
    basename = _with_sm_suffix(basename, sm_suffix)

    meta = _conv_kernel_meta_from_defines(
        defines,
        shared_mem_bytes=shared_mem_bytes,
        num_warps=num_warps_override if num_warps_override > 0 else None,
    )
    kernel_name = _require_define(defines, "CUTLASS_CONV_KERNEL_NAME")

    output_dir = _OUTPUT_ROOT.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(artifacts.ptx, output_dir / f"{basename}.ptx")
    shutil.copyfile(artifacts.cubin, output_dir / f"{basename}.cubin")

    arg_count = _parse_entry_param_count(artifacts.ptx, kernel_name)
    kinfo_path = output_dir / f"{basename}.kinfo.inc.asm"
    _write_conv_kinfo(kinfo_path, function_name=kernel_name, meta=meta, arg_count=arg_count)

    print("Packaged artifacts written to:")
    print(f"  Kernel: {kernel_name}")
    print(f"  PTX : {output_dir / (basename + '.ptx')}")
    print(f"  CUBIN: {output_dir / (basename + '.cubin')}")
    print(f"  KINFO: {output_dir / (basename + '.kinfo.inc.asm')}")
    return basename


def package_cutlass_gemm_artifacts(
    *,
    artifacts: CutlassArtifacts,
    basename: str,
    defines: Optional[dict[str, str]] = None,
    sm_suffix: Optional[str] = None,
) -> str:
    defines = {k: str(v) for k, v in (defines or {}).items()}
    metadata = json.loads(artifacts.metadata.read_text())
    shared_mem_bytes = int(metadata.get("shared_mem_bytes", 0))
    if shared_mem_bytes <= 0:
        raise RuntimeError("CUTLASS metadata missing shared memory usage")
    num_warps_override = int(metadata.get("num_warps", 0)) or None

    if sm_suffix is None:
        sm_suffix = artifacts.sm_suffix
    basename = _with_sm_suffix(basename, sm_suffix)

    meta = _gemm_kernel_meta_from_defines(
        defines,
        shared_mem_bytes=shared_mem_bytes,
        num_warps=num_warps_override,
    )
    kernel_name = _require_define(defines, "CUTLASS_GEMM_KERNEL_NAME")

    output_dir = _OUTPUT_ROOT.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(artifacts.ptx, output_dir / f"{basename}.ptx")
    shutil.copyfile(artifacts.cubin, output_dir / f"{basename}.cubin")

    arg_count = _parse_entry_param_count(artifacts.ptx, kernel_name)
    kinfo_path = output_dir / f"{basename}.kinfo.inc.asm"
    _write_gemm_kinfo(kinfo_path, function_name=kernel_name, meta=meta, arg_count=arg_count)

    print("Packaged artifacts written to:")
    print(f"  Kernel: {kernel_name}")
    print(f"  PTX : {output_dir / (basename + '.ptx')}")
    print(f"  CUBIN: {output_dir / (basename + '.cubin')}")
    print(f"  KINFO: {output_dir / (basename + '.kinfo.inc.asm')}")
    return basename


def package_cutlass_mha_artifacts(
    *,
    artifacts: CutlassArtifacts,
    basename: str,
    defines: Optional[dict[str, str]] = None,
    sm_suffix: Optional[str] = None,
) -> str:
    defines = {k: str(v) for k, v in (defines or {}).items()}
    metadata = json.loads(artifacts.metadata.read_text())
    shared_mem_bytes = int(metadata.get("shared_mem_bytes", 0))
    if shared_mem_bytes <= 0:
        shared_mem_bytes = _parse_constant_from_ptx(artifacts.ptx, "cutlass_mha_shared_mem_bytes")
    num_warps = int(metadata.get("num_warps", 0))
    if num_warps <= 0:
        num_warps = _parse_constant_from_ptx(artifacts.ptx, "cutlass_mha_num_warps")

    if shared_mem_bytes <= 0 or num_warps <= 0:
        raise RuntimeError("CUTLASS metadata missing warp/shared memory information")

    if sm_suffix is None:
        sm_suffix = artifacts.sm_suffix
    basename = _with_sm_suffix(basename, sm_suffix)

    meta = _mha_kernel_meta_from_defines(defines, shared_mem_bytes=shared_mem_bytes, num_warps=num_warps)
    kernel_name = _require_define(defines, "CUTLASS_MHA_KERNEL_NAME")

    output_dir = _OUTPUT_ROOT.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(artifacts.ptx, output_dir / f"{basename}.ptx")
    shutil.copyfile(artifacts.cubin, output_dir / f"{basename}.cubin")

    arg_count = _parse_entry_param_count(artifacts.ptx, kernel_name)
    kinfo_path = output_dir / f"{basename}.kinfo.inc.asm"
    _write_mha_kinfo(kinfo_path, function_name=kernel_name, meta=meta, arg_count=arg_count)

    print("Packaged artifacts written to:")
    print(f"  Kernel: {kernel_name}")
    print(f"  PTX : {output_dir / (basename + '.ptx')}")
    print(f"  CUBIN: {output_dir / (basename + '.cubin')}")
    print(f"  KINFO: {output_dir / (basename + '.kinfo.inc.asm')}")
    return basename


def package_cutlass_mha_bwd_artifacts(
    *,
    artifacts: CutlassArtifacts,
    basename: str,
    defines: Optional[dict[str, str]] = None,
    sm_suffix: Optional[str] = None,
) -> str:
    defines = {k: str(v) for k, v in (defines or {}).items()}
    metadata = json.loads(artifacts.metadata.read_text())
    shared_mem_bytes = int(metadata.get("shared_mem_bytes", 0))
    if shared_mem_bytes <= 0:
        shared_mem_bytes = _parse_constant_from_ptx(artifacts.ptx, "cutlass_mha_bwd_shared_mem_bytes")
    num_warps = int(metadata.get("num_warps", 0))
    if num_warps <= 0:
        num_warps = _parse_constant_from_ptx(artifacts.ptx, "cutlass_mha_bwd_num_warps")

    if shared_mem_bytes <= 0 or num_warps <= 0:
        raise RuntimeError("CUTLASS MHA bwd metadata missing warp/shared memory information")

    block_size_i = _parse_constant_from_ptx(artifacts.ptx, "cutlass_mha_bwd_block_size_i")
    block_size_j = _parse_constant_from_ptx(artifacts.ptx, "cutlass_mha_bwd_block_size_j")
    head_dim = _parse_constant_from_ptx(artifacts.ptx, "cutlass_mha_bwd_head_dim")
    gradq_tile_elements = _parse_constant_from_ptx(artifacts.ptx, "cutlass_mha_bwd_gradq_tile_elements")
    gradq_temp_bytes = _parse_constant_from_ptx(artifacts.ptx, "cutlass_mha_bwd_gradq_temp_bytes")

    if sm_suffix is None:
        sm_suffix = artifacts.sm_suffix
    basename = _with_sm_suffix(basename, sm_suffix)

    defines["CUTLASS_MHA_BWD_BLOCK_I"] = _require_define(defines, "CUTLASS_MHA_BWD_BLOCK_I")
    defines["CUTLASS_MHA_BWD_BLOCK_J"] = _require_define(defines, "CUTLASS_MHA_BWD_BLOCK_J")
    defines["CUTLASS_MHA_BWD_HEAD_DIM"] = _require_define(defines, "CUTLASS_MHA_BWD_HEAD_DIM")

    if int(defines["CUTLASS_MHA_BWD_BLOCK_I"]) != int(block_size_i):
        raise RuntimeError("CUTLASS MHA bwd block size I does not match compiled kernel.")
    if int(defines["CUTLASS_MHA_BWD_BLOCK_J"]) != int(block_size_j):
        raise RuntimeError("CUTLASS MHA bwd block size J does not match compiled kernel.")
    if int(defines["CUTLASS_MHA_BWD_HEAD_DIM"]) != int(head_dim):
        raise RuntimeError("CUTLASS MHA bwd head dim does not match compiled kernel.")

    meta = _mha_bwd_kernel_meta_from_defines(
        defines,
        shared_mem_bytes=shared_mem_bytes,
        num_warps=num_warps,
        gradq_tile_elements=gradq_tile_elements,
        gradq_temp_bytes=gradq_temp_bytes,
    )
    kernel_name = _require_define(defines, "CUTLASS_MHA_BWD_KERNEL_NAME")

    output_dir = _OUTPUT_ROOT.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(artifacts.ptx, output_dir / f"{basename}.ptx")
    shutil.copyfile(artifacts.cubin, output_dir / f"{basename}.cubin")

    arg_count = _parse_entry_param_count(artifacts.ptx, kernel_name)
    kinfo_path = output_dir / f"{basename}.kinfo.inc.asm"
    _write_mha_bwd_kinfo(kinfo_path, function_name=kernel_name, meta=meta, arg_count=arg_count)

    print("Packaged artifacts written to:")
    print(f"  Kernel: {kernel_name}")
    print(f"  PTX : {output_dir / (basename + '.ptx')}")
    print(f"  CUBIN: {output_dir / (basename + '.cubin')}")
    print(f"  KINFO: {output_dir / (basename + '.kinfo.inc.asm')}")
    return basename


def main(args: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Build the CUTLASS Conv2d artifacts.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose nvcc output.")
    parser.add_argument(
        "--package",
        action="store_true",
        help="Copy artifacts and kinfo metadata into the local output directory.",
    )
    parser.add_argument(
        "--sm",
        default="sm_89",
        help="Target SM architecture (e.g. sm_89) used for PTX/cubin compilation.",
    )
    parser.add_argument(
        "--basename",
        default=None,
        help="Base filename to use for packaged artifacts.",
    )
    parser.add_argument(
        "--autotune-config",
        action="append",
        dest="autotune_configs",
        default=[],
        help="Path to a JSON file with auto-tuned kernel constants for CUTLASS kernels.",
    )
    parser.add_argument(
        "--conv2d-configs-file",
        action="append",
        dest="conv2d_config_files",
        default=[],
        help="Path to a JSON file containing additional conv2d kernel configs to register.",
    )
    parser.add_argument(
        "--config",
        action="append",
        dest="configs",
        help="Name of a conv2d CUTLASS kernel configuration to build and package.",
    )
    parser.add_argument(
        "--all-configs",
        action="store_true",
        help="Build and package all conv2d CUTLASS kernel configurations.",
    )
    parser.add_argument(
        "--all-configs-from-file",
        action="store_true",
        help="Build and package all conv2d CUTLASS kernel configurations loaded from config files.",
    )
    parser.add_argument(
        "--conv2d-dgrad-config",
        action="append",
        dest="conv2d_dgrad_configs",
        help="Name of a conv2d dgrad CUTLASS kernel configuration to build and package.",
    )
    parser.add_argument(
        "--all-conv2d-dgrad-configs",
        action="store_true",
        help="Build and package all conv2d dgrad CUTLASS kernel configurations.",
    )
    parser.add_argument(
        "--list-conv2d-dgrad-configs",
        action="store_true",
        help="List available conv2d dgrad CUTLASS configurations and exit.",
    )
    parser.add_argument(
        "--conv2d-wgrad-config",
        action="append",
        dest="conv2d_wgrad_configs",
        help="Name of a conv2d wgrad CUTLASS kernel configuration to build and package.",
    )
    parser.add_argument(
        "--all-conv2d-wgrad-configs",
        action="store_true",
        help="Build and package all conv2d wgrad CUTLASS kernel configurations.",
    )
    parser.add_argument(
        "--list-conv2d-wgrad-configs",
        action="store_true",
        help="List available conv2d wgrad CUTLASS configurations and exit.",
    )
    parser.add_argument(
        "--gemm-config",
        action="append",
        dest="gemm_configs",
        help="Name of a GEMM CUTLASS kernel configuration to build and package.",
    )
    parser.add_argument(
        "--all-gemm-configs",
        action="store_true",
        help="Build and package all GEMM CUTLASS kernel configurations.",
    )
    parser.add_argument(
        "--list-gemm-configs",
        action="store_true",
        help="List available GEMM CUTLASS configurations and exit.",
    )
    parser.add_argument(
        "--mha-config",
        action="append",
        dest="mha_configs",
        help="Name of an MHA CUTLASS kernel configuration to build and package.",
    )
    parser.add_argument(
        "--all-mha-configs",
        action="store_true",
        help="Build and package all MHA CUTLASS kernel configurations.",
    )
    parser.add_argument(
        "--list-mha-configs",
        action="store_true",
        help="List available MHA CUTLASS configurations and exit.",
    )
    parser.add_argument(
        "--mha-bwd-config",
        action="append",
        dest="mha_bwd_configs",
        help="Name of an MHA backward CUTLASS kernel configuration to build and package.",
    )
    parser.add_argument(
        "--all-mha-bwd-configs",
        action="store_true",
        help="Build and package all MHA backward CUTLASS kernel configurations.",
    )
    parser.add_argument(
        "--list-mha-bwd-configs",
        action="store_true",
        help="List available MHA backward CUTLASS configurations and exit.",
    )
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List available conv2d CUTLASS configurations and exit.",
    )
    parsed = parser.parse_args(args=args)

    try:
        _, _, sm_suffix = _normalise_sm(parsed.sm)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if not parsed.basename:
        parsed.basename = f"conv2d_cutlass_bf16_stride1_pad2_dil2_{sm_suffix}"

    autotune_paths = [Path(p) for p in parsed.autotune_configs]
    conv_configs = _load_autotune_entries(autotune_paths, "conv2d")
    conv_dgrad_configs = _load_autotune_entries(autotune_paths, "conv2d_dgrad")
    conv_wgrad_configs = _load_autotune_entries(autotune_paths, "conv2d_wgrad")
    gemm_configs = _load_autotune_entries(autotune_paths, "gemm")
    mha_configs = _load_autotune_entries(autotune_paths, "mha")
    mha_bwd_configs = _load_autotune_entries(autotune_paths, "mha_bwd")
    file_conv_configs: set[str] = set()
    for cfg_path in parsed.conv2d_config_files:
        entries = _load_conv2d_config_entries(Path(cfg_path), "conv2d")
        for name, spec in entries.items():
            file_conv_configs.add(name)
            conv_configs[name] = _merge_config_file_entry(conv_configs.get(name), spec)
        dgrad_entries = _load_conv2d_config_entries(Path(cfg_path), "conv2d_dgrad")
        for name, spec in dgrad_entries.items():
            conv_dgrad_configs[name] = _merge_config_file_entry(conv_dgrad_configs.get(name), spec)
        wgrad_entries = _load_conv2d_config_entries(Path(cfg_path), "conv2d_wgrad")
        for name, spec in wgrad_entries.items():
            conv_wgrad_configs[name] = _merge_config_file_entry(conv_wgrad_configs.get(name), spec)
    if parsed.all_configs_from_file and not file_conv_configs:
        raise SystemExit("No conv2d configs loaded from file. Provide --conv2d-configs-file.")

    if (parsed.list_configs or parsed.all_configs or parsed.configs) and not conv_configs:
        if autotune_paths or parsed.conv2d_config_files:
            raise SystemExit("No conv2d configs found in the provided autotune/config files.")
        raise SystemExit("No conv2d autotune configs provided. Use --autotune-config.")
    if (
        parsed.list_conv2d_dgrad_configs
        or parsed.all_conv2d_dgrad_configs
        or parsed.conv2d_dgrad_configs
    ) and not conv_dgrad_configs:
        if autotune_paths:
            raise SystemExit("No conv2d dgrad configs found in the provided autotune config(s).")
        raise SystemExit("No conv2d dgrad autotune configs provided. Use --autotune-config.")
    if (
        parsed.list_conv2d_wgrad_configs
        or parsed.all_conv2d_wgrad_configs
        or parsed.conv2d_wgrad_configs
    ) and not conv_wgrad_configs:
        if autotune_paths:
            raise SystemExit("No conv2d wgrad configs found in the provided autotune config(s).")
        raise SystemExit("No conv2d wgrad autotune configs provided. Use --autotune-config.")
    if (parsed.list_gemm_configs or parsed.all_gemm_configs or parsed.gemm_configs) and not gemm_configs:
        if autotune_paths:
            raise SystemExit("No GEMM configs found in the provided autotune config(s).")
        raise SystemExit("No GEMM autotune configs provided. Use --autotune-config.")
    if (parsed.list_mha_configs or parsed.all_mha_configs or parsed.mha_configs) and not mha_configs:
        if autotune_paths:
            raise SystemExit("No MHA configs found in the provided autotune config(s).")
        raise SystemExit("No MHA autotune configs provided. Use --autotune-config.")
    if (parsed.list_mha_bwd_configs or parsed.all_mha_bwd_configs or parsed.mha_bwd_configs) and not mha_bwd_configs:
        if autotune_paths:
            raise SystemExit("No MHA bwd configs found in the provided autotune config(s).")
        raise SystemExit("No MHA bwd autotune configs provided. Use --autotune-config.")

    if (
        parsed.list_configs
        or parsed.list_conv2d_dgrad_configs
        or parsed.list_conv2d_wgrad_configs
        or parsed.list_gemm_configs
        or parsed.list_mha_configs
        or parsed.list_mha_bwd_configs
    ):
        if parsed.list_configs:
            print("Available CUTLASS conv2d configurations:")
            for name, cfg in conv_configs.items():
                print(f"  {name}: {cfg.get('basename', name)}")
            if (
                parsed.list_conv2d_dgrad_configs
                or parsed.list_conv2d_wgrad_configs
                or parsed.list_gemm_configs
                or parsed.list_mha_configs
                or parsed.list_mha_bwd_configs
            ):
                print("")
        if parsed.list_conv2d_dgrad_configs:
            print("Available CUTLASS conv2d dgrad configurations:")
            for name, cfg in conv_dgrad_configs.items():
                print(f"  {name}: {cfg.get('basename', name)}")
            if (
                parsed.list_conv2d_wgrad_configs
                or parsed.list_gemm_configs
                or parsed.list_mha_configs
                or parsed.list_mha_bwd_configs
            ):
                print("")
        if parsed.list_conv2d_wgrad_configs:
            print("Available CUTLASS conv2d wgrad configurations:")
            for name, cfg in conv_wgrad_configs.items():
                print(f"  {name}: {cfg.get('basename', name)}")
            if parsed.list_gemm_configs or parsed.list_mha_configs or parsed.list_mha_bwd_configs:
                print("")
        if parsed.list_gemm_configs:
            print("Available CUTLASS GEMM configurations:")
            for name, cfg in gemm_configs.items():
                print(f"  {name}: {cfg.get('basename', name)}")
            if parsed.list_mha_configs or parsed.list_mha_bwd_configs:
                print("")
        if parsed.list_mha_configs:
            print("Available CUTLASS MHA configurations:")
            for name, cfg in mha_configs.items():
                print(f"  {name}: {cfg.get('basename', name)}")
            if parsed.list_mha_bwd_configs:
                print("")
        if parsed.list_mha_bwd_configs:
            print("Available CUTLASS MHA backward configurations:")
            for name, cfg in mha_bwd_configs.items():
                print(f"  {name}: {cfg.get('basename', name)}")
        return

    selected_configs: list[str] = []
    if parsed.all_configs:
        selected_configs.extend(conv_configs.keys())
    if parsed.all_configs_from_file:
        selected_configs.extend(sorted(file_conv_configs))
    if parsed.configs:
        for cfg_name in parsed.configs:
            if cfg_name not in conv_configs:
                raise SystemExit(
                    f"Unknown conv2d configuration '{cfg_name}'. "
                    "Ensure it exists in the autotune config or conv2d config files."
                )
            selected_configs.append(cfg_name)

    selected_conv_dgrad_configs: list[str] = []
    if parsed.all_conv2d_dgrad_configs:
        selected_conv_dgrad_configs.extend(conv_dgrad_configs.keys())
    if parsed.conv2d_dgrad_configs:
        for cfg_name in parsed.conv2d_dgrad_configs:
            if cfg_name not in conv_dgrad_configs:
                raise SystemExit(
                    f"Unknown conv2d dgrad configuration '{cfg_name}'. "
                    "Ensure it exists in the autotune config or conv2d config files."
                )
            selected_conv_dgrad_configs.append(cfg_name)

    selected_conv_wgrad_configs: list[str] = []
    if parsed.all_conv2d_wgrad_configs:
        selected_conv_wgrad_configs.extend(conv_wgrad_configs.keys())
    if parsed.conv2d_wgrad_configs:
        for cfg_name in parsed.conv2d_wgrad_configs:
            if cfg_name not in conv_wgrad_configs:
                raise SystemExit(
                    f"Unknown conv2d wgrad configuration '{cfg_name}'. "
                    "Ensure it exists in the autotune config or conv2d config files."
                )
            selected_conv_wgrad_configs.append(cfg_name)

    selected_gemm_configs: list[str] = []
    if parsed.all_gemm_configs:
        selected_gemm_configs.extend(gemm_configs.keys())
    if parsed.gemm_configs:
        for cfg_name in parsed.gemm_configs:
            if cfg_name not in gemm_configs:
                raise SystemExit(
                    f"Unknown GEMM configuration '{cfg_name}'. Ensure it exists in the autotune config."
                )
            selected_gemm_configs.append(cfg_name)

    selected_mha_bwd_configs: list[str] = []
    if parsed.all_mha_bwd_configs:
        selected_mha_bwd_configs.extend(mha_bwd_configs.keys())
    if parsed.mha_bwd_configs:
        for cfg_name in parsed.mha_bwd_configs:
            if cfg_name not in mha_bwd_configs:
                raise SystemExit(
                    f"Unknown MHA bwd configuration '{cfg_name}'. Ensure it exists in the autotune config."
                )
            selected_mha_bwd_configs.append(cfg_name)

    selected_mha_configs: list[str] = []
    if parsed.all_mha_configs:
        selected_mha_configs.extend(mha_configs.keys())
    if parsed.mha_configs:
        for cfg_name in parsed.mha_configs:
            if cfg_name not in mha_configs:
                raise SystemExit(
                    f"Unknown MHA configuration '{cfg_name}'. Ensure it exists in the autotune config."
                )
            selected_mha_configs.append(cfg_name)

    processed = False

    if selected_configs:
        unique_configs: list[str] = []
        seen = set()
        for name in selected_configs:
            if name in seen:
                continue
            seen.add(name)
            unique_configs.append(name)

        for name in unique_configs:
            spec = conv_configs[name]
            if not spec.get("enabled", True):
                if parsed.configs and name in parsed.configs:
                    raise SystemExit(
                        f"Configuration '{name}' is not yet supported by the CUTLASS generator."
                    )
                print(f"Skipping disabled configuration '{name}'.")
                continue
            defines = spec.get("defines")
            if not isinstance(defines, Mapping):
                raise SystemExit(f"Configuration '{name}' is missing a 'defines' mapping.")
            defines_map = _apply_required_cutlass_defines(defines, category="conv2d", sm=parsed.sm)
            defines_map = _apply_sm_specific_defines(spec, defines_map, parsed.sm)
            defines_map = _inject_sm_arch_define(defines_map, parsed.sm, "CUTLASS_CONV_SM_ARCH")
            source_path = _source_path_for_spec(spec, _CONV_SOURCE_FILE, parsed.sm)
            artifacts = build_cutlass_artifacts(
                verbose=parsed.verbose,
                defines=defines_map,
                sm=parsed.sm,
                source_file=source_path,
            )
            packaged_basename = package_cutlass_conv_artifacts(
                artifacts=artifacts,
                basename=str(spec.get("basename") or name),
                defines=defines_map,
                sm_suffix=artifacts.sm_suffix,
            )
            print(f"Built {name} -> {packaged_basename}")
        processed = True

    if selected_conv_dgrad_configs:
        unique_dgrad_configs: list[str] = []
        seen_dgrad = set()
        for name in selected_conv_dgrad_configs:
            if name in seen_dgrad:
                continue
            seen_dgrad.add(name)
            unique_dgrad_configs.append(name)

        for name in unique_dgrad_configs:
            spec = conv_dgrad_configs[name]
            if not spec.get("enabled", True):
                if parsed.conv2d_dgrad_configs and name in parsed.conv2d_dgrad_configs:
                    raise SystemExit(
                        f"Configuration '{name}' is not yet supported by the CUTLASS dgrad generator."
                    )
                print(f"Skipping disabled configuration '{name}'.")
                continue
            defines = spec.get("defines")
            if not isinstance(defines, Mapping):
                raise SystemExit(f"Configuration '{name}' is missing a 'defines' mapping.")
            defines_map = _apply_required_cutlass_defines(defines, category="conv2d_dgrad", sm=parsed.sm)
            defines_map = _apply_sm_specific_defines(spec, defines_map, parsed.sm)
            defines_map = _inject_sm_arch_define(defines_map, parsed.sm, "CUTLASS_CONV_SM_ARCH")
            source_path = _source_path_for_spec(spec, _CONV_DGRAD_SOURCE_FILE, parsed.sm)
            artifacts = build_cutlass_artifacts(
                verbose=parsed.verbose,
                defines=defines_map,
                sm=parsed.sm,
                source_file=source_path,
            )
            packaged_basename = package_cutlass_conv_artifacts(
                artifacts=artifacts,
                basename=str(spec.get("basename") or name),
                defines=defines_map,
                sm_suffix=artifacts.sm_suffix,
            )
            print(f"Built {name} -> {packaged_basename}")
        processed = True

    if selected_conv_wgrad_configs:
        unique_wgrad_configs: list[str] = []
        seen_wgrad = set()
        for name in selected_conv_wgrad_configs:
            if name in seen_wgrad:
                continue
            seen_wgrad.add(name)
            unique_wgrad_configs.append(name)

        for name in unique_wgrad_configs:
            spec = conv_wgrad_configs[name]
            if not spec.get("enabled", True):
                if parsed.conv2d_wgrad_configs and name in parsed.conv2d_wgrad_configs:
                    raise SystemExit(
                        f"Configuration '{name}' is not yet supported by the CUTLASS wgrad generator."
                    )
                print(f"Skipping disabled configuration '{name}'.")
                continue
            defines = spec.get("defines")
            if not isinstance(defines, Mapping):
                raise SystemExit(f"Configuration '{name}' is missing a 'defines' mapping.")
            defines_map = _apply_required_cutlass_defines(defines, category="conv2d_wgrad", sm=parsed.sm)
            defines_map = _apply_sm_specific_defines(spec, defines_map, parsed.sm)
            defines_map = _inject_sm_arch_define(defines_map, parsed.sm, "CUTLASS_CONV_SM_ARCH")
            source_path = _source_path_for_spec(spec, _CONV_WGRAD_SOURCE_FILE, parsed.sm)
            artifacts = build_cutlass_artifacts(
                verbose=parsed.verbose,
                defines=defines_map,
                sm=parsed.sm,
                source_file=source_path,
            )
            packaged_basename = package_cutlass_conv_artifacts(
                artifacts=artifacts,
                basename=str(spec.get("basename") or name),
                defines=defines_map,
                sm_suffix=artifacts.sm_suffix,
            )
            print(f"Built {name} -> {packaged_basename}")
        processed = True

    if selected_gemm_configs:
        unique_gemm_configs: list[str] = []
        seen_gemm = set()
        for name in selected_gemm_configs:
            if name in seen_gemm:
                continue
            seen_gemm.add(name)
            unique_gemm_configs.append(name)

        for name in unique_gemm_configs:
            spec = gemm_configs[name]
            if not spec.get("enabled", True):
                if parsed.gemm_configs and name in parsed.gemm_configs:
                    raise SystemExit(
                        f"Configuration '{name}' is not yet supported by the CUTLASS GEMM generator."
                    )
                print(f"Skipping disabled GEMM configuration '{name}'.")
                continue
            defines = spec.get("defines")
            if not isinstance(defines, Mapping):
                raise SystemExit(f"GEMM configuration '{name}' is missing a 'defines' mapping.")
            defines_map = _apply_required_cutlass_defines(defines, category="gemm", sm=parsed.sm)
            defines_map = _inject_sm_arch_define(defines_map, parsed.sm, "CUTLASS_GEMM_SM_ARCH")
            source_file_hint = spec.get("source_file")
            if source_file_hint:
                source_path = Path(source_file_hint)
                if not source_path.is_absolute():
                    source_path = (_ROOT_DIR / source_path).resolve()
            else:
                source_path = _GEMM_SOURCE_FILE
            artifacts = build_cutlass_artifacts(
                verbose=parsed.verbose,
                defines=defines_map,
                sm=parsed.sm,
                source_file=source_path,
            )
            packaged_basename = package_cutlass_gemm_artifacts(
                artifacts=artifacts,
                basename=str(spec.get("basename") or name),
                defines=defines_map,
                sm_suffix=artifacts.sm_suffix,
            )
            print(f"Built {name} -> {packaged_basename}")
        processed = True

    if selected_mha_configs:
        unique_mha_configs: list[str] = []
        seen_mha = set()
        for name in selected_mha_configs:
            if name in seen_mha:
                continue
            seen_mha.add(name)
            unique_mha_configs.append(name)

        for name in unique_mha_configs:
            spec = mha_configs[name]
            if not spec.get("enabled", True):
                if parsed.mha_configs and name in parsed.mha_configs:
                    raise SystemExit(
                        f"Configuration '{name}' is not yet supported by the CUTLASS MHA generator."
                    )
                print(f"Skipping disabled configuration '{name}'.")
                continue
            defines = spec.get("defines")
            if not isinstance(defines, Mapping):
                raise SystemExit(f"MHA configuration '{name}' is missing a 'defines' mapping.")
            defines_map = _apply_required_cutlass_defines(defines, category="mha", sm=parsed.sm)
            defines_map = _inject_sm_arch_define(
                defines_map,
                parsed.sm,
                "CUTLASS_MHA_SM_ARCH",
                sm_version_key="CUTLASS_MHA_SM_VERSION",
            )
            source_file_hint = spec.get("source_file")
            if source_file_hint:
                source_path = Path(source_file_hint)
                if not source_path.is_absolute():
                    source_path = (_ROOT_DIR / source_path).resolve()
            else:
                source_path = _MHA_SOURCE_FILE
            artifacts = build_cutlass_artifacts(
                verbose=parsed.verbose,
                defines=defines_map,
                sm=parsed.sm,
                source_file=source_path,
            )
            packaged_basename = package_cutlass_mha_artifacts(
                artifacts=artifacts,
                basename=str(spec.get("basename") or name),
                defines=defines_map,
                sm_suffix=artifacts.sm_suffix,
            )
            print(f"Built {name} -> {packaged_basename}")
        processed = True

    if selected_mha_bwd_configs:
        unique_mha_bwd_configs: list[str] = []
        seen_mha_bwd = set()
        for name in selected_mha_bwd_configs:
            if name in seen_mha_bwd:
                continue
            seen_mha_bwd.add(name)
            unique_mha_bwd_configs.append(name)

        for name in unique_mha_bwd_configs:
            spec = mha_bwd_configs[name]
            if not spec.get("enabled", True):
                if parsed.mha_bwd_configs and name in parsed.mha_bwd_configs:
                    raise SystemExit(
                        f"Configuration '{name}' is not yet supported by the CUTLASS MHA bwd generator."
                    )
                print(f"Skipping disabled configuration '{name}'.")
                continue
            defines = spec.get("defines")
            if not isinstance(defines, Mapping):
                raise SystemExit(f"MHA bwd configuration '{name}' is missing a 'defines' mapping.")
            defines_map = _apply_required_cutlass_defines(defines, category="mha_bwd", sm=parsed.sm)
            defines_map = _inject_sm_arch_define(
                defines_map,
                parsed.sm,
                "CUTLASS_MHA_BWD_SM_ARCH",
                sm_version_key="CUTLASS_MHA_BWD_SM_VERSION",
            )
            source_file_hint = spec.get("source_file")
            if source_file_hint:
                source_path = Path(source_file_hint)
                if not source_path.is_absolute():
                    source_path = (_ROOT_DIR / source_path).resolve()
            else:
                source_path = _MHA_BWD_SOURCE_FILE
            artifacts = build_cutlass_artifacts(
                verbose=parsed.verbose,
                defines=defines_map,
                sm=parsed.sm,
                source_file=source_path,
            )
            packaged_basename = package_cutlass_mha_bwd_artifacts(
                artifacts=artifacts,
                basename=str(spec.get("basename") or name),
                defines=defines_map,
                sm_suffix=artifacts.sm_suffix,
            )
            print(f"Built {name} -> {packaged_basename}")
        processed = True

    if processed:
        return

    artifacts = build_cutlass_artifacts(verbose=parsed.verbose, sm=parsed.sm)
    print("CUTLASS conv2d artifacts built:")
    print(f"  PTX : {artifacts.ptx}")
    print(f"  CUBIN: {artifacts.cubin}")

    if parsed.package:
        package_cutlass_conv_artifacts(
            artifacts=artifacts,
            basename=parsed.basename,
            sm_suffix=artifacts.sm_suffix,
        )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
