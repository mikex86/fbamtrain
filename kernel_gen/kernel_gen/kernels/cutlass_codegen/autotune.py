from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from cutlass_codegen.compile import (
    _CONV_SOURCE_FILE,
    _CONV_DGRAD_SOURCE_FILE,
    _CONV_WGRAD_SOURCE_FILE,
    _GEMM_SOURCE_FILE,
    _MHA_BWD_SOURCE_FILE,
    _MHA_SOURCE_FILE,
    _conv_kernel_meta_from_defines,
    _gemm_kernel_meta_from_defines,
    _mha_bwd_kernel_meta_from_defines,
    _mha_kernel_meta_from_defines,
    _load_conv2d_config_entries,
    _parse_constant_from_ptx,
    _apply_required_cutlass_defines,
    _stringify_defines,
    build_cutlass_artifacts,
)

_CUmodule = ctypes.c_void_p
_CUfunction = ctypes.c_void_p
_CUstream = ctypes.c_void_p
_CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8


def _ceil_div(a: int, b: int) -> int:
    if b <= 0:
        raise ValueError("Cannot divide by non-positive value.")
    return (a + b - 1) // b


def _align_up(value: int, alignment: int) -> int:
    if alignment <= 0:
        raise ValueError("Alignment must be positive.")
    return (value + alignment - 1) // alignment * alignment


class _CudaDriver:
    def __init__(self) -> None:
        self._lib = ctypes.CDLL("libcuda.so")
        self._lib.cuInit.argtypes = [ctypes.c_uint]
        self._lib.cuInit.restype = ctypes.c_int
        self._lib.cuModuleLoad.argtypes = [ctypes.POINTER(_CUmodule), ctypes.c_char_p]
        self._lib.cuModuleLoad.restype = ctypes.c_int
        self._lib.cuModuleUnload.argtypes = [_CUmodule]
        self._lib.cuModuleUnload.restype = ctypes.c_int
        self._lib.cuModuleGetFunction.argtypes = [ctypes.POINTER(_CUfunction), _CUmodule, ctypes.c_char_p]
        self._lib.cuModuleGetFunction.restype = ctypes.c_int
        self._lib.cuFuncSetAttribute.argtypes = [_CUfunction, ctypes.c_int, ctypes.c_int]
        self._lib.cuFuncSetAttribute.restype = ctypes.c_int
        self._lib.cuLaunchKernel.argtypes = [
            _CUfunction,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            _CUstream,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
        ]
        self._lib.cuLaunchKernel.restype = ctypes.c_int
        self._lib.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        self._lib.cuGetErrorString.restype = ctypes.c_int

        status = self._lib.cuInit(0)
        if status != 0:
            raise RuntimeError(self._format_error("cuInit", status))

    def _format_error(self, api: str, status: int) -> str:
        msg = ctypes.c_char_p()
        if self._lib.cuGetErrorString(status, ctypes.byref(msg)) == 0 and msg.value:
            return f"{api} failed with error {status}: {msg.value.decode('utf-8')}"
        return f"{api} failed with error {status}"

    def _check(self, api: str, status: int) -> None:
        if status == 0:
            return
        raise RuntimeError(self._format_error(api, status))

    def load_module(self, path: Path) -> _CUmodule:
        module = _CUmodule()
        status = self._lib.cuModuleLoad(ctypes.byref(module), str(path).encode("utf-8"))
        self._check("cuModuleLoad", status)
        return module

    def unload_module(self, module: _CUmodule) -> None:
        if module:
            self._check("cuModuleUnload", self._lib.cuModuleUnload(module))

    def get_function(self, module: _CUmodule, name: str) -> _CUfunction:
        function = _CUfunction()
        status = self._lib.cuModuleGetFunction(ctypes.byref(function), module, name.encode("utf-8"))
        self._check("cuModuleGetFunction", status)
        return function

    def set_dynamic_smem(self, function: _CUfunction, shared_mem_bytes: int) -> None:
        self._check(
            "cuFuncSetAttribute",
            self._lib.cuFuncSetAttribute(function, _CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, int(shared_mem_bytes)),
        )

    def launch_kernel(
        self,
        function: _CUfunction,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        shared_mem_bytes: int,
        stream_ptr: int,
        args_array: ctypes.Array,
    ) -> None:
        status = self._lib.cuLaunchKernel(
            function,
            ctypes.c_uint(grid[0]),
            ctypes.c_uint(grid[1]),
            ctypes.c_uint(grid[2]),
            ctypes.c_uint(block[0]),
            ctypes.c_uint(block[1]),
            ctypes.c_uint(block[2]),
            ctypes.c_uint(shared_mem_bytes),
            _CUstream(stream_ptr),
            args_array,
            None,
        )
        self._check("cuLaunchKernel", status)


@dataclass(frozen=True)
class _TimingConfig:
    iters: int
    warmup: int


@dataclass
class _KernelTiming:
    time_ms: float
    max_abs_diff: float
    mean_abs_diff: float
    valid: bool


def _conv_candidate(
    tb_m: int,
    tb_n: int,
    tb_k: int,
    warp_m: int,
    warp_n: int,
    warp_k: int,
    stages: int,
) -> dict[str, int]:
    return {
        "CUTLASS_CONV_THREADBLOCK_M": tb_m,
        "CUTLASS_CONV_THREADBLOCK_N": tb_n,
        "CUTLASS_CONV_THREADBLOCK_K": tb_k,
        "CUTLASS_CONV_WARP_M": warp_m,
        "CUTLASS_CONV_WARP_N": warp_n,
        "CUTLASS_CONV_WARP_K": warp_k,
        "CUTLASS_CONV_NUM_STAGES": stages,
    }


# Ordered hierarchically: first the previous/core tiles, then smaller/asymmetric
# tiles that may reduce register pressure, then a short stage-depth refinement.
DEFAULT_CONV_SEARCH: Sequence[Mapping[str, Any]] = (
    _conv_candidate(128, 128, 64, 64, 64, 64, 3),
    _conv_candidate(128, 64, 64, 64, 32, 64, 3),
    _conv_candidate(128, 128, 32, 64, 64, 32, 3),
    _conv_candidate(64, 128, 32, 32, 64, 32, 3),
    _conv_candidate(64, 64, 64, 32, 32, 64, 3),
    _conv_candidate(64, 64, 32, 32, 32, 32, 3),
    _conv_candidate(64, 128, 64, 32, 64, 64, 3),
    _conv_candidate(128, 64, 32, 64, 32, 32, 3),
    _conv_candidate(128, 256, 32, 64, 64, 32, 3),
    _conv_candidate(256, 64, 32, 64, 32, 32, 3),
    _conv_candidate(256, 128, 32, 64, 64, 32, 3),
    _conv_candidate(128, 128, 64, 64, 64, 64, 2),
    _conv_candidate(128, 128, 64, 64, 64, 64, 4),
    _conv_candidate(128, 128, 32, 64, 64, 32, 2),
    _conv_candidate(128, 128, 32, 64, 64, 32, 4),
    _conv_candidate(64, 128, 32, 32, 64, 32, 2),
    _conv_candidate(64, 128, 32, 32, 64, 32, 4),
    _conv_candidate(128, 64, 32, 64, 32, 32, 2),
    _conv_candidate(128, 64, 32, 64, 32, 32, 4),
)

DEFAULT_GEMM_SEARCH: Sequence[Mapping[str, Any]] = (
    {
        "CUTLASS_GEMM_THREADBLOCK_M": 256,
        "CUTLASS_GEMM_THREADBLOCK_N": 128,
        "CUTLASS_GEMM_THREADBLOCK_K": 32,
        "CUTLASS_GEMM_WARP_M": 64,
        "CUTLASS_GEMM_WARP_N": 64,
        "CUTLASS_GEMM_WARP_K": 32,
        "CUTLASS_GEMM_SWIZZLE_SIZE": 1,
    },
    {
        "CUTLASS_GEMM_THREADBLOCK_M": 256,
        "CUTLASS_GEMM_THREADBLOCK_N": 128,
        "CUTLASS_GEMM_THREADBLOCK_K": 32,
        "CUTLASS_GEMM_WARP_M": 64,
        "CUTLASS_GEMM_WARP_N": 64,
        "CUTLASS_GEMM_WARP_K": 32,
        "CUTLASS_GEMM_SWIZZLE_SIZE": 8,
    },
)

DEFAULT_MHA_SEARCH_LEGACY: Sequence[Mapping[str, Any]] = (
    {"CUTLASS_MHA_QUERIES_PER_BLOCK": 32, "CUTLASS_MHA_KEYS_PER_BLOCK": 128},
    {"CUTLASS_MHA_QUERIES_PER_BLOCK": 64, "CUTLASS_MHA_KEYS_PER_BLOCK": 64},
    {"CUTLASS_MHA_QUERIES_PER_BLOCK": 128, "CUTLASS_MHA_KEYS_PER_BLOCK": 128},
)

DEFAULT_MHA_BWD_SEARCH: Sequence[Mapping[str, Any]] = (
    {"CUTLASS_MHA_BWD_BLOCK_I": 128, "CUTLASS_MHA_BWD_BLOCK_J": 64},
    {"CUTLASS_MHA_BWD_BLOCK_I": 64, "CUTLASS_MHA_BWD_BLOCK_J": 64},
)

ENABLE_MHA_TUNING = True

_TUNABLE_KEYS: dict[str, set[str]] = {
    "conv2d": {
        "CUTLASS_CONV_THREADBLOCK_M",
        "CUTLASS_CONV_THREADBLOCK_N",
        "CUTLASS_CONV_THREADBLOCK_K",
        "CUTLASS_CONV_WARP_M",
        "CUTLASS_CONV_WARP_N",
        "CUTLASS_CONV_WARP_K",
        "CUTLASS_CONV_NUM_STAGES",
    },
    "conv2d_dgrad": {
        "CUTLASS_CONV_THREADBLOCK_M",
        "CUTLASS_CONV_THREADBLOCK_N",
        "CUTLASS_CONV_THREADBLOCK_K",
        "CUTLASS_CONV_WARP_M",
        "CUTLASS_CONV_WARP_N",
        "CUTLASS_CONV_WARP_K",
        "CUTLASS_CONV_NUM_STAGES",
    },
    "conv2d_wgrad": {
        "CUTLASS_CONV_THREADBLOCK_M",
        "CUTLASS_CONV_THREADBLOCK_N",
        "CUTLASS_CONV_THREADBLOCK_K",
        "CUTLASS_CONV_WARP_M",
        "CUTLASS_CONV_WARP_N",
        "CUTLASS_CONV_WARP_K",
        "CUTLASS_CONV_NUM_STAGES",
    },
    "gemm": {
        "CUTLASS_GEMM_THREADBLOCK_M",
        "CUTLASS_GEMM_THREADBLOCK_N",
        "CUTLASS_GEMM_THREADBLOCK_K",
        "CUTLASS_GEMM_WARP_M",
        "CUTLASS_GEMM_WARP_N",
        "CUTLASS_GEMM_WARP_K",
        "CUTLASS_GEMM_SWIZZLE_SIZE",
    },
    "mha": {
        "CUTLASS_MHA_QUERIES_PER_BLOCK",
        "CUTLASS_MHA_KEYS_PER_BLOCK",
    },
    "mha_bwd": {
        "CUTLASS_MHA_BWD_BLOCK_I",
        "CUTLASS_MHA_BWD_BLOCK_J",
    },
}


def _resolve_source_path(value: Optional[Any], default_path: Path) -> Path:
    if value is None:
        return default_path
    path = Path(value)
    if not path.is_absolute():
        path = (default_path.parent / path).resolve()
    return path


def _normalise_sm(sm: str) -> str:
    sm_str = sm.strip().lower()
    match = re.fullmatch(r"(?:sm_?|compute_)?(\d+)", sm_str)
    if not match:
        raise ValueError(f"Unrecognised SM version '{sm}'. Expected formats like 'sm_90' or '90'.")
    return f"sm_{match.group(1)}"


def _default_mha_search(sm: str) -> Sequence[Mapping[str, Any]]:
    _ = _normalise_sm(sm)
    return DEFAULT_MHA_SEARCH_LEGACY


def _inject_sm_arch_define(
    defines: Mapping[str, str],
    sm: str,
    key: str,
    *,
    sm_version_key: str | None = None,
) -> dict[str, str]:
    updated = dict(defines)
    sm_norm = _normalise_sm(sm)
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


def _load_search_space(path: Optional[Path]) -> dict[str, dict[str, dict[str, Any]]]:
    if path is None:
        return {}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Search space file '{path}' is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError("Search space JSON must contain an object at the top level.")
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for category, entries in data.items():
        if not isinstance(entries, dict):
            raise RuntimeError(f"Search space category '{category}' must be an object.")
        result[category] = {}
        for name, payload in entries.items():
            if not isinstance(payload, dict):
                raise RuntimeError(f"Search space entry '{name}' in '{category}' must be an object.")
            result[category][name] = payload
    return result


def _load_base_payload(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except OSError as exc:
        raise RuntimeError(f"Failed to read base config file '{path}': {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Base config file '{path}' is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"Base config file '{path}' must contain a JSON object at the top level.")
    return data


def _load_base_entries(paths: Sequence[Path], category: str) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    sources: dict[str, Path] = {}
    for path in paths:
        payload = _load_base_payload(path)
        if category not in payload:
            continue
        section = payload[category]
        if not isinstance(section, Mapping):
            raise RuntimeError(f"Base config section '{category}' in '{path}' must be a JSON object.")
        for name, entry in section.items():
            if not isinstance(name, str):
                raise RuntimeError(f"Base config keys must be strings (got {name!r}).")
            if name in entries:
                raise RuntimeError(
                    f"Duplicate base config '{name}' in '{category}' "
                    f"(found in {sources[name]} and {path})."
                )
            if not isinstance(entry, Mapping):
                raise RuntimeError(f"Base config entry for '{name}' must be a JSON object.")
            entry_dict = dict(entry)
            defines_payload = entry_dict.get("defines")
            if defines_payload is None:
                raise RuntimeError(f"Base config '{name}' in '{path}' is missing a 'defines' mapping.")
            if not isinstance(defines_payload, Mapping):
                raise RuntimeError(f"Base config '{name}' defines must be a JSON object.")
            entry_dict["defines"] = _stringify_defines(defines_payload)
            entries[name] = entry_dict
            sources[name] = path
    return entries


def _require_define(defines: Mapping[str, str], key: str) -> str:
    value = defines.get(key)
    if value is None or value == "":
        raise RuntimeError(f"CUTLASS define '{key}' is required but missing.")
    return value


def _ensure_conv_bwd_kernel_names(
    configs: Mapping[str, dict[str, Any]],
    *,
    op: str,
    source_file: Path,
) -> dict[str, dict[str, Any]]:
    if op not in ("dgrad", "wgrad"):
        raise ValueError(f"Unsupported conv2d backward op '{op}'.")
    updated: dict[str, dict[str, Any]] = {}
    default_kernel = f"cutlass_conv2d_{op}_stride1_pad2_dil2_kernel"
    for name, spec in configs.items():
        entry = dict(spec)
        defines = _stringify_defines(entry.get("defines", {}))
        if "CUTLASS_CONV_KERNEL_NAME" not in defines:
            defines["CUTLASS_CONV_KERNEL_NAME"] = entry.get("kernel_name", default_kernel)
        entry.setdefault("kernel_name", defines["CUTLASS_CONV_KERNEL_NAME"])
        entry["defines"] = defines
        entry.setdefault("source_file", str(source_file))
        updated[name] = entry
    return updated


def _candidate_overrides(
    category: str,
    name: str,
    search_space: Mapping[str, dict[str, Any]],
    defaults: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], Optional[dict[str, Any]]]:
    entries = search_space.get(category, {})
    problem: Optional[dict[str, Any]] = None
    candidates: list[Mapping[str, Any]] = []

    default_entry = entries.get("default")
    if isinstance(default_entry, dict):
        default_problem = default_entry.get("problem")
        if isinstance(default_problem, dict):
            problem = dict(default_problem)
        default_candidates = default_entry.get("candidates")
        if isinstance(default_candidates, list):
            candidates.extend(item for item in default_candidates if isinstance(item, Mapping))

    specific_entry = entries.get(name)
    if isinstance(specific_entry, dict):
        override_problem = specific_entry.get("problem")
        if isinstance(override_problem, dict):
            problem = dict(override_problem)
        specific_candidates = specific_entry.get("candidates")
        if isinstance(specific_candidates, list):
            candidates.extend(item for item in specific_candidates if isinstance(item, Mapping))

    candidates.extend(defaults)
    unique: list[Mapping[str, Any]] = []
    seen = set()
    for cand in candidates:
        frozen = tuple(sorted((str(k), str(v)) for k, v in cand.items()))
        if frozen in seen:
            continue
        seen.add(frozen)
        unique.append(dict(cand))
    return unique, problem


def _merge_defines(base: Mapping[str, str], override: Mapping[str, Any]) -> dict[str, str]:
    merged = dict(base)
    for key, value in override.items():
        merged[str(key)] = str(value)
    return merged


def _require_problem_keys(
    category: str,
    name: str,
    spec: Mapping[str, Any],
    override: Optional[Mapping[str, Any]],
    keys: Sequence[str],
) -> None:
    spec_problem = spec.get("problem")
    if not isinstance(spec_problem, Mapping):
        spec_problem = {}
    override_map = override or {}
    missing = [
        key
        for key in keys
        if key not in spec and key not in spec_problem and key not in override_map
    ]
    if missing:
        joined = ", ".join(sorted(missing))
        raise RuntimeError(f"[{category}] {name}: missing required problem keys: {joined}")


def _assert_no_tunable_in_base(category: str, name: str, base_defines: Mapping[str, str]) -> None:
    tunables = _TUNABLE_KEYS.get(category, set())
    if category in {"conv2d", "conv2d_dgrad", "conv2d_wgrad"}:
        tunables = tunables - {"CUTLASS_CONV_NUM_STAGES"}
    present = sorted(key for key in tunables if key in base_defines)
    if present:
        joined = ", ".join(present)
        raise RuntimeError(f"[{category}] {name}: base defines must not include tunables: {joined}")


def _assert_candidate_has_all_tunables(category: str, name: str, candidate: Mapping[str, Any]) -> None:
    tunables = _TUNABLE_KEYS.get(category, set())
    missing = sorted(key for key in tunables if key not in candidate)
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"[{category}] {name}: candidate missing tunables: {joined}")


def _validate_search_space(
    category: str,
    name: str,
    base_defines: Mapping[str, str],
    candidates: Sequence[Mapping[str, Any]],
) -> None:
    _assert_no_tunable_in_base(category, name, base_defines)
    for candidate in candidates:
        _assert_candidate_has_all_tunables(category, name, candidate)


def _detect_conv_dtype(defines: Mapping[str, str]) -> torch.dtype:
    return torch.float16 if defines.get("CUTLASS_CONV_FP16") == "1" else torch.bfloat16


def _detect_gemm_dtype(defines: Mapping[str, str]) -> torch.dtype:
    return torch.float16 if defines.get("CUTLASS_GEMM_FP16") == "1" else torch.bfloat16


def _detect_mha_dtype(defines: Mapping[str, str]) -> torch.dtype:
    return torch.float16 if defines.get("CUTLASS_MHA_FP16") == "1" else torch.bfloat16


def _detect_mha_bwd_dtype(defines: Mapping[str, str]) -> torch.dtype:
    return torch.float16 if defines.get("CUTLASS_MHA_BWD_FP16") == "1" else torch.bfloat16


def _resolve_conv_problem(spec: Mapping[str, Any], override: Optional[Mapping[str, Any]]) -> dict[str, int]:
    _require_problem_keys(
        "conv2d",
        str(spec.get("kernel_name", "")) or "config",
        spec,
        override,
        (
            "batch",
            "height",
            "width",
            "in_channels",
            "out_channels",
            "stride",
            "padding",
            "dilation",
            "kernel_h",
            "kernel_w",
            "groups",
        ),
    )
    spec_problem = spec.get("problem")
    if not isinstance(spec_problem, Mapping):
        spec_problem = {}
    problem = {
        "batch": int(spec_problem.get("batch", spec.get("batch"))),
        "height": int(spec_problem.get("height", spec.get("height"))),
        "width": int(spec_problem.get("width", spec.get("width"))),
        "in_channels": int(
            spec_problem.get("in_channels", spec.get("in_channels", spec.get("CUTLASS_CONV_META_IN_CHANNELS")))
        ),
        "out_channels": int(spec_problem.get("out_channels", spec.get("out_channels"))),
        "stride": int(spec_problem.get("stride", spec.get("stride", spec.get("stride_h")))),
        "padding": int(spec_problem.get("padding", spec.get("padding", spec.get("padding_h")))),
        "dilation": int(spec_problem.get("dilation", spec.get("dilation", spec.get("dilation_h")))),
        "kernel_h": int(spec_problem.get("kernel_h", spec.get("kernel_h"))),
        "kernel_w": int(spec_problem.get("kernel_w", spec.get("kernel_w"))),
        "groups": int(spec_problem.get("groups", spec.get("groups"))),
    }
    if override:
        for key, value in override.items():
            if key in problem:
                problem[key] = int(value)
    return problem


def _resolve_gemm_problem(spec: Mapping[str, Any], override: Optional[Mapping[str, Any]]) -> dict[str, int]:
    _require_problem_keys(
        "gemm",
        str(spec.get("kernel_name", "")) or "config",
        spec,
        override,
        ("m", "n", "k", "lda", "ldb", "ldd", "bias_stride", "split_k_slices"),
    )
    spec_problem = spec.get("problem")
    if not isinstance(spec_problem, Mapping):
        spec_problem = {}
    problem = {
        "m": int(spec_problem.get("m", spec.get("m"))),
        "n": int(spec_problem.get("n", spec.get("n"))),
        "k": int(spec_problem.get("k", spec.get("k"))),
        "lda": int(spec_problem.get("lda", spec.get("lda"))),
        "ldb": int(spec_problem.get("ldb", spec.get("ldb"))),
        "ldd": int(spec_problem.get("ldd", spec.get("ldd"))),
        "bias_stride": int(spec_problem.get("bias_stride", spec.get("bias_stride"))),
        "split_k_slices": int(spec_problem.get("split_k_slices", spec.get("split_k_slices"))),
    }
    if override:
        for key, value in override.items():
            if key in problem:
                problem[key] = int(value)
    return problem


def _resolve_mha_problem(
    defines: Mapping[str, str],
    spec: Mapping[str, Any],
    override: Optional[Mapping[str, Any]],
) -> dict[str, int]:
    if "CUTLASS_MHA_HEAD_DIM" not in defines:
        raise RuntimeError("[mha] CUTLASS_MHA_HEAD_DIM must be defined in base defines.")
    head_dim = int(defines["CUTLASS_MHA_HEAD_DIM"])
    _require_problem_keys(
        "mha",
        str(spec.get("kernel_name", "")) or "config",
        spec,
        override,
        ("batch_size", "num_heads", "sequence_length"),
    )
    spec_problem = spec.get("problem")
    if not isinstance(spec_problem, Mapping):
        spec_problem = {}
    problem = {
        "batch_size": int(spec_problem.get("batch_size", spec.get("batch_size"))),
        "num_heads": int(spec_problem.get("num_heads", spec.get("num_heads"))),
        "sequence_length": int(spec_problem.get("sequence_length", spec.get("sequence_length"))),
        "head_dim": head_dim,
    }
    if override:
        for key, value in override.items():
            if key in problem:
                problem[key] = int(value)
    problem["head_dim"] = head_dim
    return problem


def _resolve_mha_bwd_problem(
    defines: Mapping[str, str],
    spec: Mapping[str, Any],
    override: Optional[Mapping[str, Any]],
) -> dict[str, int]:
    if "CUTLASS_MHA_BWD_HEAD_DIM" not in defines:
        raise RuntimeError("[mha_bwd] CUTLASS_MHA_BWD_HEAD_DIM must be defined in base defines.")
    head_dim = int(defines["CUTLASS_MHA_BWD_HEAD_DIM"])
    _require_problem_keys(
        "mha_bwd",
        str(spec.get("kernel_name", "")) or "config",
        spec,
        override,
        ("batch_size", "num_heads", "sequence_length"),
    )
    spec_problem = spec.get("problem")
    if not isinstance(spec_problem, Mapping):
        spec_problem = {}
    problem = {
        "batch_size": int(spec_problem.get("batch_size", spec.get("batch_size"))),
        "num_heads": int(spec_problem.get("num_heads", spec.get("num_heads"))),
        "sequence_length": int(spec_problem.get("sequence_length", spec.get("sequence_length"))),
        "head_dim": head_dim,
    }
    if override:
        for key, value in override.items():
            if key in problem:
                problem[key] = int(value)
    problem["head_dim"] = head_dim
    return problem


def _generate_conv_inputs(problem: Mapping[str, int], dtype: torch.dtype, device: torch.device):
    batch = int(problem["batch"])
    in_channels = int(problem["in_channels"])
    out_channels = int(problem["out_channels"])
    height = int(problem["height"])
    width = int(problem["width"])
    kernel_h = int(problem.get("kernel_h", 3))
    kernel_w = int(problem.get("kernel_w", 3))

    input_scale = 1.0 / max(1.0, math.sqrt(float(in_channels)))
    weight_scale = 1.0 / max(1.0, math.sqrt(float(in_channels * kernel_h * kernel_w)))
    if dtype == torch.float16:
        input_scale *= 0.25
        weight_scale *= 0.25

    x = (torch.randn((batch, in_channels, height, width), device=device, dtype=torch.float32) * input_scale).to(dtype)
    w = (torch.randn((out_channels, in_channels, kernel_h, kernel_w), device=device, dtype=torch.float32) * weight_scale).to(dtype)
    bias = (torch.randn((out_channels,), device=device, dtype=torch.float32) * weight_scale).to(dtype)

    x = x.contiguous(memory_format=torch.channels_last)
    w = w.contiguous(memory_format=torch.channels_last)
    return x, w, bias


def _generate_gemm_inputs(problem: Mapping[str, int], dtype: torch.dtype, device: torch.device):
    m = int(problem["m"])
    n = int(problem["n"])
    k = int(problem["k"])

    scale = 1.0 / max(1.0, math.sqrt(float(k)))

    a = (torch.randn((m, k), device=device, dtype=torch.float32) * scale).to(dtype).contiguous()
    b = (torch.randn((k, n), device=device, dtype=torch.float32) * scale).to(dtype).contiguous()
    bias = (torch.randn((n,), device=device, dtype=torch.float32) * scale).to(dtype).contiguous()
    return a, b, bias


def _generate_mha_inputs(problem: Mapping[str, int], dtype: torch.dtype, device: torch.device):
    batch = int(problem["batch_size"])
    heads = int(problem["num_heads"])
    seq = int(problem["sequence_length"])
    dim = int(problem["head_dim"])
    shape = (batch, seq, heads, dim)
    scale = 1.0 / max(1.0, math.sqrt(float(dim)))
    q = (torch.randn(shape, device=device, dtype=torch.float32) * scale).to(dtype).contiguous()
    k = (torch.randn(shape, device=device, dtype=torch.float32) * scale).to(dtype).contiguous()
    v = (torch.randn(shape, device=device, dtype=torch.float32) * scale).to(dtype).contiguous()
    return q, k, v


def _generate_mha_bwd_inputs(problem: Mapping[str, int], dtype: torch.dtype, device: torch.device):
    batch = int(problem["batch_size"])
    heads = int(problem["num_heads"])
    seq = int(problem["sequence_length"])
    dim = int(problem["head_dim"])
    scale = 1.0 / max(1.0, math.sqrt(float(dim)))

    q = (torch.randn((batch, seq, heads, dim), device=device, dtype=torch.float32) * scale).to(dtype).contiguous()
    k = (torch.randn((batch, seq, heads, dim), device=device, dtype=torch.float32) * scale).to(dtype).contiguous()
    v = (torch.randn((batch, seq, heads, dim), device=device, dtype=torch.float32) * scale).to(dtype).contiguous()

    with torch.no_grad():
        q_bhtd = q.float().permute(0, 2, 1, 3).contiguous()
        k_bhtd = k.float().permute(0, 2, 1, 3).contiguous()
        v_bhtd = v.float().permute(0, 2, 1, 3).contiguous()
        scores = torch.matmul(q_bhtd, k_bhtd.transpose(-1, -2)) * scale
        probs = torch.softmax(scores, dim=-1)
        out_bhtd = torch.matmul(probs, v_bhtd)
        o = out_bhtd.permute(0, 2, 1, 3).contiguous().to(dtype)

    do = (torch.randn((batch, seq, heads, dim), device=device, dtype=torch.float32) * scale).to(dtype).contiguous()
    do_bhtd = do.float().permute(0, 2, 1, 3).contiguous()
    delta = (do_bhtd * out_bhtd).sum(-1).contiguous()
    lse = torch.logsumexp(scores, dim=-1).contiguous()
    return q, k, v, o, do, lse, delta


def _time_function(fn, timing: _TimingConfig) -> float:
    for _ in range(timing.warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(max(timing.iters, 1)):
        fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end) / max(timing.iters, 1))




def _normalise_metric(value: float) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(numeric):
        return numeric
    return None


def _tolerances(op: str, dtype: torch.dtype) -> Tuple[float, float]:
    if op == "conv":
        if dtype == torch.float16:
            return 0.75, 5e-2
        return 2.5, 5e-3
    if op == "gemm":
        if dtype == torch.float16:
            return 1.0, 2e-2
        return 2.0, 2e-2
    if op == "mha":
        if dtype == torch.float16:
            return 1.0, 1e-3
        return 1.5, 2e-3
    return 1.0, 1e-3


def _conv_tolerances(defines: Mapping[str, str], dtype: torch.dtype) -> Tuple[float, float]:
    max_tol, mean_tol = _tolerances("conv", dtype)
    if defines.get("CUTLASS_CONV_FP16_ACCUM") == "1":
        max_tol = max(max_tol, 3.0)
        mean_tol = max(mean_tol, 0.4)
    return max_tol, mean_tol


def _gemm_tolerances(defines: Mapping[str, str], dtype: torch.dtype) -> Tuple[float, float]:
    max_tol, mean_tol = _tolerances("gemm", dtype)
    if defines.get("CUTLASS_GEMM_FP32_OUTPUT") == "1":
        max_tol = max(max_tol, 5.0)
        mean_tol = max(mean_tol, 0.5)
    return max_tol, mean_tol


def _conv_output_dims(problem: Mapping[str, int]) -> tuple[int, int]:
    stride = int(problem["stride"])
    padding = int(problem["padding"])
    dilation = int(problem["dilation"])
    kernel_h = int(problem.get("kernel_h", 3))
    kernel_w = int(problem.get("kernel_w", 3))
    height = int(problem["height"])
    width = int(problem["width"])
    eff_kernel_h = dilation * (kernel_h - 1) + 1
    eff_kernel_w = dilation * (kernel_w - 1) + 1
    out_h = (height + 2 * padding - eff_kernel_h) // stride + 1
    out_w = (width + 2 * padding - eff_kernel_w) // stride + 1
    return out_h, out_w


def _compute_gemm_grid(meta, problem: Mapping[str, int]) -> tuple[int, int]:
    grid_tiles_m = _ceil_div(problem["m"], meta.block_m)
    grid_tiles_n = _ceil_div(problem["n"], meta.block_n)
    swizzle = max(meta.swizzle_size, 1)
    if grid_tiles_m == 0 or grid_tiles_n == 0:
        return grid_tiles_m, grid_tiles_n
    log_tile = 0
    if swizzle >= 8 and grid_tiles_n >= 6:
        log_tile = 3
    elif swizzle >= 4 and grid_tiles_n >= 3:
        log_tile = 2
    elif swizzle >= 2 and grid_tiles_n >= 2:
        log_tile = 1
    tile = 1 << log_tile
    grid_dim_x = grid_tiles_m * tile
    grid_dim_y = grid_tiles_n
    return grid_dim_x, max(grid_dim_y, 1)


def _launch_conv_kernel(
    driver: _CudaDriver,
    artifacts,
    defines: Mapping[str, str],
    meta,
    problem: Mapping[str, int],
    x_cutlass: torch.Tensor,
    w_cutlass: torch.Tensor,
    bias: torch.Tensor,
    timing: _TimingConfig,
) -> tuple[torch.Tensor, _KernelTiming]:
    kernel_name = _require_define(defines, "CUTLASS_CONV_KERNEL_NAME")
    module = driver.load_module(artifacts.cubin)
    try:
        function = driver.get_function(module, kernel_name)
        driver.set_dynamic_smem(function, meta.shared_mem_bytes)

        dtype = x_cutlass.dtype

        out_h, out_w = _conv_output_dims(problem)
        batch = int(problem["batch"])
        out_channels = int(problem["out_channels"])
        grid_x = max(1, _ceil_div(batch * out_h * out_w, meta.block_pixels))
        grid_y = max(1, _ceil_div(out_channels, meta.block_oc))
        split_k_slices = max(
            1,
            int(defines.get("CUTLASS_CONV_SPLIT_K_SLICES", problem.get("split_k_slices", 1))),
        )
        grid_z = split_k_slices
        block_x = meta.num_warps * 32
        block = (block_x, 1, 1)

        class CutlassConv2dConfig(ctypes.Structure):
            _fields_ = [
                ("N", ctypes.c_int),
                ("H", ctypes.c_int),
                ("W", ctypes.c_int),
                ("C", ctypes.c_int),
                ("K", ctypes.c_int),
                ("R", ctypes.c_int),
                ("S", ctypes.c_int),
                ("pad_h", ctypes.c_int),
                ("pad_w", ctypes.c_int),
                ("stride_h", ctypes.c_int),
                ("stride_w", ctypes.c_int),
                ("dilation_h", ctypes.c_int),
                ("dilation_w", ctypes.c_int),
                ("groups", ctypes.c_int),
                ("split_k_slices", ctypes.c_int),
            ]

        cfg = CutlassConv2dConfig(
            batch,
            int(problem["height"]),
            int(problem["width"]),
            int(problem["in_channels"]),
            out_channels,
            int(problem.get("kernel_h", 3)),
            int(problem.get("kernel_w", 3)),
            int(problem["padding"]),
            int(problem["padding"]),
            int(problem["stride"]),
            int(problem["stride"]),
            int(problem["dilation"]),
            int(problem["dilation"]),
            int(problem.get("groups", 1)),
            split_k_slices,
        )

        workspace_elems = grid_x * grid_y
        workspace = torch.zeros(workspace_elems, device=x_cutlass.device, dtype=torch.int32)

        cfg_ptr = ctypes.cast(ctypes.byref(cfg), ctypes.c_void_p)
        x_ptr = ctypes.c_void_p(x_cutlass.data_ptr())
        w_ptr = ctypes.c_void_p(w_cutlass.data_ptr())
        out = torch.empty(
            (batch, out_channels, out_h, out_w),
            device=x_cutlass.device,
            dtype=x_cutlass.dtype,
        ).contiguous(memory_format=torch.channels_last)
        out_ptr = ctypes.c_void_p(out.data_ptr())
        alpha = ctypes.c_float(1.0)
        beta = ctypes.c_float(0.0)
        workspace_ptr = ctypes.c_void_p(workspace.data_ptr())

        kernel_args = [
            cfg_ptr,
            ctypes.cast(ctypes.byref(x_ptr), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(w_ptr), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(out_ptr), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(alpha), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(beta), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(workspace_ptr), ctypes.c_void_p),
        ]
        args_array = (ctypes.c_void_p * len(kernel_args))(*[arg.value for arg in kernel_args])
        stream_ptr = torch.cuda.current_stream(x_cutlass.device).cuda_stream

        def _launch() -> None:
            driver.launch_kernel(function, (grid_x, grid_y, grid_z), block, meta.shared_mem_bytes, stream_ptr, args_array)

        time_ms = _time_function(_launch, timing)
        out.add_(bias.view(1, -1, 1, 1))

        with torch.no_grad():
            ref = F.conv2d(
                x_cutlass.to(dtype).contiguous(memory_format=torch.channels_last),
                w_cutlass.to(dtype).contiguous(memory_format=torch.channels_last),
                bias=bias.to(dtype),
                stride=int(problem["stride"]),
                padding=int(problem["padding"]),
                dilation=int(problem["dilation"]),
                groups=int(problem.get("groups", 1)),
            )
            cutlass_out = out.to(memory_format=torch.contiguous_format)
            ref_out = ref.to(memory_format=torch.contiguous_format)
            diff = (cutlass_out - ref_out).abs().float()
            max_diff = float(diff.max().item())
            mean_diff = float(diff.mean().item())
            max_tol, mean_tol = _conv_tolerances(defines, dtype)
            timing_info = _KernelTiming(
                time_ms=time_ms,
                max_abs_diff=max_diff,
                mean_abs_diff=mean_diff,
                valid=max_diff <= max_tol and mean_diff <= mean_tol,
            )
        return cutlass_out, timing_info
    finally:
        driver.unload_module(module)


def _launch_conv_dgrad_kernel(
    driver: _CudaDriver,
    artifacts,
    defines: Mapping[str, str],
    meta,
    problem: Mapping[str, int],
    dout: torch.Tensor,
    weight: torch.Tensor,
    timing: _TimingConfig,
) -> tuple[torch.Tensor, _KernelTiming]:
    kernel_name = _require_define(defines, "CUTLASS_CONV_KERNEL_NAME")
    module = driver.load_module(artifacts.cubin)
    try:
        function = driver.get_function(module, kernel_name)
        driver.set_dynamic_smem(function, meta.shared_mem_bytes)

        batch = int(problem["batch"])
        height = int(problem["height"])
        width = int(problem["width"])
        in_channels = int(problem["in_channels"])
        out_channels = int(problem["out_channels"])
        kernel_h = int(problem.get("kernel_h", 3))
        kernel_w = int(problem.get("kernel_w", 3))
        stride = int(problem["stride"])
        padding = int(problem["padding"])
        dilation = int(problem["dilation"])
        groups = int(problem.get("groups", 1))

        total_pixels = batch * height * width
        grid_x = max(1, _ceil_div(total_pixels, meta.block_pixels))
        grid_y = max(1, _ceil_div(in_channels, meta.block_oc))
        grid_z = 1
        block_x = meta.num_warps * 32
        block = (block_x, 1, 1)

        class CutlassConv2dConfig(ctypes.Structure):
            _fields_ = [
                ("N", ctypes.c_int),
                ("H", ctypes.c_int),
                ("W", ctypes.c_int),
                ("C", ctypes.c_int),
                ("K", ctypes.c_int),
                ("R", ctypes.c_int),
                ("S", ctypes.c_int),
                ("pad_h", ctypes.c_int),
                ("pad_w", ctypes.c_int),
                ("stride_h", ctypes.c_int),
                ("stride_w", ctypes.c_int),
                ("dilation_h", ctypes.c_int),
                ("dilation_w", ctypes.c_int),
                ("groups", ctypes.c_int),
                ("split_k_slices", ctypes.c_int),
            ]

        cfg = CutlassConv2dConfig(
            batch,
            height,
            width,
            in_channels,
            out_channels,
            kernel_h,
            kernel_w,
            padding,
            padding,
            stride,
            stride,
            dilation,
            dilation,
            groups,
            1,
        )

        workspace_elems = grid_x * grid_y
        workspace = torch.zeros(workspace_elems, device=dout.device, dtype=torch.int32)

        dinput = torch.empty(
            (batch, in_channels, height, width),
            device=dout.device,
            dtype=dout.dtype,
        ).contiguous(memory_format=torch.channels_last)

        cfg_ptr = ctypes.cast(ctypes.byref(cfg), ctypes.c_void_p)
        dout_ptr = ctypes.c_void_p(dout.data_ptr())
        weight_ptr = ctypes.c_void_p(weight.data_ptr())
        dinput_ptr = ctypes.c_void_p(dinput.data_ptr())
        alpha = ctypes.c_float(1.0)
        beta = ctypes.c_float(0.0)
        workspace_ptr = ctypes.c_void_p(workspace.data_ptr())

        kernel_args = [
            cfg_ptr,
            ctypes.cast(ctypes.byref(dout_ptr), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(weight_ptr), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(dinput_ptr), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(alpha), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(beta), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(workspace_ptr), ctypes.c_void_p),
        ]
        args_array = (ctypes.c_void_p * len(kernel_args))(*[arg.value for arg in kernel_args])
        stream_ptr = torch.cuda.current_stream(dout.device).cuda_stream

        def _launch() -> None:
            driver.launch_kernel(function, (grid_x, grid_y, grid_z), block, meta.shared_mem_bytes, stream_ptr, args_array)

        time_ms = _time_function(_launch, timing)

        with torch.no_grad():
            ref = torch.nn.grad.conv2d_input(
                (batch, in_channels, height, width),
                weight.contiguous(),
                dout.contiguous(),
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
            ref = ref.to(dinput.dtype)

        out_nhwc = dinput.permute(0, 2, 3, 1).contiguous()
        ref_nhwc = ref.permute(0, 2, 3, 1).contiguous()
        diff = (out_nhwc - ref_nhwc).abs().float()
        max_diff = float(diff.max().item())
        mean_diff = float(diff.mean().item())
        max_tol, mean_tol = _conv_tolerances(defines, dinput.dtype)
        timing_info = _KernelTiming(
            time_ms=time_ms,
            max_abs_diff=max_diff,
            mean_abs_diff=mean_diff,
            valid=max_diff <= max_tol and mean_diff <= mean_tol,
        )
        return dinput, timing_info
    finally:
        driver.unload_module(module)


def _launch_conv_wgrad_kernel(
    driver: _CudaDriver,
    artifacts,
    defines: Mapping[str, str],
    meta,
    problem: Mapping[str, int],
    x: torch.Tensor,
    dout: torch.Tensor,
    timing: _TimingConfig,
) -> tuple[torch.Tensor, _KernelTiming]:
    kernel_name = _require_define(defines, "CUTLASS_CONV_KERNEL_NAME")
    module = driver.load_module(artifacts.cubin)
    try:
        function = driver.get_function(module, kernel_name)
        driver.set_dynamic_smem(function, meta.shared_mem_bytes)

        batch = int(problem["batch"])
        height = int(problem["height"])
        width = int(problem["width"])
        in_channels = int(problem["in_channels"])
        out_channels = int(problem["out_channels"])
        kernel_h = int(problem.get("kernel_h", 3))
        kernel_w = int(problem.get("kernel_w", 3))
        stride = int(problem["stride"])
        padding = int(problem["padding"])
        dilation = int(problem["dilation"])
        groups = int(problem.get("groups", 1))

        gemm_m = out_channels
        gemm_n = kernel_h * kernel_w * in_channels
        grid_x = max(1, _ceil_div(gemm_m, meta.block_pixels))
        grid_y = max(1, _ceil_div(gemm_n, meta.block_oc))
        grid_z = 1
        block_x = meta.num_warps * 32
        block = (block_x, 1, 1)

        class CutlassConv2dConfig(ctypes.Structure):
            _fields_ = [
                ("N", ctypes.c_int),
                ("H", ctypes.c_int),
                ("W", ctypes.c_int),
                ("C", ctypes.c_int),
                ("K", ctypes.c_int),
                ("R", ctypes.c_int),
                ("S", ctypes.c_int),
                ("pad_h", ctypes.c_int),
                ("pad_w", ctypes.c_int),
                ("stride_h", ctypes.c_int),
                ("stride_w", ctypes.c_int),
                ("dilation_h", ctypes.c_int),
                ("dilation_w", ctypes.c_int),
                ("groups", ctypes.c_int),
                ("split_k_slices", ctypes.c_int),
            ]

        cfg = CutlassConv2dConfig(
            batch,
            height,
            width,
            in_channels,
            out_channels,
            kernel_h,
            kernel_w,
            padding,
            padding,
            stride,
            stride,
            dilation,
            dilation,
            groups,
            1,
        )

        workspace_elems = grid_x * grid_y
        workspace = torch.zeros(workspace_elems, device=x.device, dtype=torch.int32)

        dweight = torch.empty(
            (out_channels, kernel_h, kernel_w, in_channels),
            device=x.device,
            dtype=x.dtype,
        ).contiguous()

        cfg_ptr = ctypes.cast(ctypes.byref(cfg), ctypes.c_void_p)
        dout_ptr = ctypes.c_void_p(dout.data_ptr())
        x_ptr = ctypes.c_void_p(x.data_ptr())
        dweight_ptr = ctypes.c_void_p(dweight.data_ptr())
        alpha = ctypes.c_float(1.0)
        beta = ctypes.c_float(0.0)
        workspace_ptr = ctypes.c_void_p(workspace.data_ptr())

        kernel_args = [
            cfg_ptr,
            ctypes.cast(ctypes.byref(dout_ptr), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(x_ptr), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(dweight_ptr), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(alpha), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(beta), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(workspace_ptr), ctypes.c_void_p),
        ]
        args_array = (ctypes.c_void_p * len(kernel_args))(*[arg.value for arg in kernel_args])
        stream_ptr = torch.cuda.current_stream(x.device).cuda_stream

        def _launch() -> None:
            driver.launch_kernel(function, (grid_x, grid_y, grid_z), block, meta.shared_mem_bytes, stream_ptr, args_array)

        time_ms = _time_function(_launch, timing)

        with torch.no_grad():
            ref = torch.nn.grad.conv2d_weight(
                x.contiguous(),
                (out_channels, in_channels, kernel_h, kernel_w),
                dout.contiguous(),
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
            ref = ref.to(dweight.dtype)

        ref_ohwi = ref.permute(0, 2, 3, 1).contiguous()
        diff = (dweight - ref_ohwi).abs().float()
        max_diff = float(diff.max().item())
        mean_diff = float(diff.mean().item())
        max_tol, mean_tol = _conv_tolerances(defines, dweight.dtype)
        timing_info = _KernelTiming(
            time_ms=time_ms,
            max_abs_diff=max_diff,
            mean_abs_diff=mean_diff,
            valid=max_diff <= max_tol and mean_diff <= mean_tol,
        )
        return dweight, timing_info
    finally:
        driver.unload_module(module)


def _launch_gemm_kernel(
    driver: _CudaDriver,
    artifacts,
    defines: Mapping[str, str],
    meta,
    problem: Mapping[str, int],
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor,
    timing: _TimingConfig,
) -> tuple[torch.Tensor, _KernelTiming]:
    kernel_name = _require_define(defines, "CUTLASS_GEMM_KERNEL_NAME")
    module = driver.load_module(artifacts.cubin)
    try:
        function = driver.get_function(module, kernel_name)
        driver.set_dynamic_smem(function, meta.shared_mem_bytes)

        grid_x, grid_y = _compute_gemm_grid(meta, problem)
        grid_z = max(1, int(problem.get("split_k_slices", 1)))
        block = (meta.num_warps * 32, 1, 1)

        class CutlassGemmConfig(ctypes.Structure):
            _fields_ = [
                ("m", ctypes.c_int),
                ("n", ctypes.c_int),
                ("k", ctypes.c_int),
                ("lda", ctypes.c_int),
                ("ldb", ctypes.c_int),
                ("ldd", ctypes.c_int),
                ("bias_stride", ctypes.c_int),
                ("split_k_slices", ctypes.c_int),
            ]

        cfg = CutlassGemmConfig(
            int(problem["m"]),
            int(problem["n"]),
            int(problem["k"]),
            int(problem["lda"]),
            int(problem["ldb"]),
            int(problem["ldd"]),
            int(problem["bias_stride"]),
            grid_z,
        )

        with_bias = defines.get("CUTLASS_GEMM_WITH_BIAS", "1") != "0"
        write_out_preact = defines.get("CUTLASS_GEMM_WRITE_OUT_PREACT", "0") != "0"
        activation_kind = defines.get("CUTLASS_GEMM_ACTIVATION", "0")
        out_dtype = torch.float32 if defines.get("CUTLASS_GEMM_FP32_OUTPUT", "0") == "1" else a.dtype

        cfg_ptr = ctypes.cast(ctypes.byref(cfg), ctypes.c_void_p)
        a_ptr = ctypes.c_void_p(a.data_ptr())
        b_ptr = ctypes.c_void_p(b.data_ptr())
        bias_ptr = ctypes.c_void_p(bias.data_ptr()) if with_bias else ctypes.c_void_p(0)
        out = torch.empty((problem["m"], problem["n"]), device=a.device, dtype=out_dtype).contiguous()
        out_ptr = ctypes.c_void_p(out.data_ptr())
        preact = None
        preact_ptr = ctypes.c_void_p(0)
        if write_out_preact:
            preact = torch.empty_like(out)
            preact_ptr = ctypes.c_void_p(preact.data_ptr())
        alpha = ctypes.c_float(1.0)
        beta = ctypes.c_float(0.0)
        workspace_ptr = ctypes.c_void_p(0)

        kernel_args = [
            cfg_ptr,
            ctypes.cast(ctypes.byref(a_ptr), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(b_ptr), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(bias_ptr), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(out_ptr), ctypes.c_void_p),
        ]
        if write_out_preact:
            kernel_args.append(ctypes.cast(ctypes.byref(preact_ptr), ctypes.c_void_p))
        kernel_args += [
            ctypes.cast(ctypes.byref(alpha), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(beta), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(workspace_ptr), ctypes.c_void_p),
        ]
        args_array = (ctypes.c_void_p * len(kernel_args))(*[arg.value for arg in kernel_args])
        stream_ptr = torch.cuda.current_stream(a.device).cuda_stream

        def _launch() -> None:
            driver.launch_kernel(function, (grid_x, grid_y, grid_z), block, meta.shared_mem_bytes, stream_ptr, args_array)

        time_ms = _time_function(_launch, timing)

        with torch.no_grad():
            a_ref = a.to(a.dtype)
            b_ref = b.to(b.dtype)
            ref = torch.matmul(a_ref, b_ref)
            if with_bias:
                ref.add_(bias.to(out_dtype))
            if activation_kind == "2":
                ref = F.gelu(ref)
            ref = ref.to(out.dtype)
            diff = (out - ref).abs().float()
            max_diff = float(diff.max().item())
            mean_diff = float(diff.mean().item())
            max_tol, mean_tol = _gemm_tolerances(defines, a.dtype)
            timing_info = _KernelTiming(
                time_ms=time_ms,
                max_abs_diff=max_diff,
                mean_abs_diff=mean_diff,
                valid=max_diff <= max_tol and mean_diff <= mean_tol,
            )
        return out, timing_info
    finally:
        driver.unload_module(module)


def _launch_mha_kernel(
    driver: _CudaDriver,
    artifacts,
    defines: Mapping[str, str],
    meta,
    problem: Mapping[str, int],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    timing: _TimingConfig,
) -> tuple[torch.Tensor, _KernelTiming]:
    kernel_name = _require_define(defines, "CUTLASS_MHA_KERNEL_NAME")
    module = driver.load_module(artifacts.cubin)
    try:
        function = driver.get_function(module, kernel_name)
        driver.set_dynamic_smem(function, meta.shared_mem_bytes)

        batch = int(problem["batch_size"])
        heads = int(problem["num_heads"])
        seq = int(problem["sequence_length"])
        head_dim = int(problem["head_dim"])

        grid_x = max(1, _ceil_div(seq, meta.block_size_x))
        grid_y = max(1, heads)
        grid_z = max(1, batch)
        # Example-41 MHA forward kernels expect a 2D block layout:
        # block.x = warp size, block.y = warps per CTA.
        block = (32, meta.num_warps, 1)

        q_stride_z = int(q.stride(0))
        q_stride_t = int(q.stride(1))
        q_stride_h = int(q.stride(2))
        k_stride_z = int(k.stride(0))
        k_stride_t = int(k.stride(1))
        k_stride_h = int(k.stride(2))
        v_stride_z = int(v.stride(0))
        v_stride_t = int(v.stride(1))
        v_stride_h = int(v.stride(2))

        out = torch.empty_like(q).contiguous()

        scratch = torch.empty((batch, heads, seq), device=q.device, dtype=torch.float32).contiguous()
        # Example-41 accumulates in fp32 for bf16/fp16 outputs.
        workspace = torch.empty((batch, seq, heads, head_dim), device=q.device, dtype=torch.float32).contiguous()
        softmax_scale = ctypes.c_float(1.0 / math.sqrt(head_dim))
        scratch_ptr = ctypes.c_void_p(scratch.data_ptr())
        batch_arg = ctypes.c_uint32(batch)
        heads_arg = ctypes.c_uint32(heads)
        q_ptr = ctypes.c_void_p(q.data_ptr())
        q_stride_z_arg = ctypes.c_uint32(q_stride_z)
        q_stride_h_arg = ctypes.c_uint32(q_stride_h)
        q_stride_t_arg = ctypes.c_uint32(q_stride_t)
        k_ptr = ctypes.c_void_p(k.data_ptr())
        k_stride_z_arg = ctypes.c_uint32(k_stride_z)
        k_stride_h_arg = ctypes.c_uint32(k_stride_h)
        k_stride_t_arg = ctypes.c_uint32(k_stride_t)
        v_ptr = ctypes.c_void_p(v.data_ptr())
        v_stride_z_arg = ctypes.c_uint32(v_stride_z)
        v_stride_h_arg = ctypes.c_uint32(v_stride_h)
        v_stride_t_arg = ctypes.c_uint32(v_stride_t)
        o_ptr = ctypes.c_void_p(out.data_ptr())
        o_stride_z_arg = ctypes.c_uint32(out.stride(0))
        o_stride_h_arg = ctypes.c_uint32(out.stride(2))
        o_stride_t_arg = ctypes.c_uint32(out.stride(1))
        seq_arg = ctypes.c_uint32(seq)
        zero_u32_a = ctypes.c_uint32(0)
        zero_u32_b = ctypes.c_uint32(0)
        zero_u32_c = ctypes.c_uint32(0)
        workspace_ptr = ctypes.c_void_p(workspace.data_ptr())

        kernel_args = [
            ctypes.cast(ctypes.byref(softmax_scale), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(scratch_ptr), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(batch_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(heads_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(q_ptr), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(q_stride_z_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(q_stride_h_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(q_stride_t_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(k_ptr), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(k_stride_z_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(k_stride_h_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(k_stride_t_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(v_ptr), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(v_stride_z_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(v_stride_h_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(v_stride_t_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(o_ptr), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(o_stride_z_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(o_stride_h_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(o_stride_t_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(seq_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(zero_u32_a), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(zero_u32_b), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(zero_u32_c), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(workspace_ptr), ctypes.c_void_p),
        ]
        args_array = (ctypes.c_void_p * len(kernel_args))(*[arg.value for arg in kernel_args])
        stream_ptr = torch.cuda.current_stream(q.device).cuda_stream

        def _launch() -> None:
            driver.launch_kernel(function, (grid_x, grid_y, grid_z), block, meta.shared_mem_bytes, stream_ptr, args_array)

        time_ms = _time_function(_launch, timing)

        with torch.no_grad():
            q_ref = q.to(torch.float32).permute(0, 2, 1, 3).contiguous()
            k_ref = k.to(torch.float32).permute(0, 2, 1, 3).contiguous()
            v_ref = v.to(torch.float32).permute(0, 2, 1, 3).contiguous()
            scale = 1.0 / math.sqrt(head_dim)
            scores = torch.matmul(q_ref, k_ref.transpose(-1, -2)) * scale
            probs = torch.softmax(scores, dim=-1)
            ref = torch.matmul(probs, v_ref)
            ref = ref.permute(0, 2, 1, 3).contiguous().to(out.dtype)
            diff = (out - ref).abs().float()
            max_diff = float(diff.max().item())
            mean_diff = float(diff.mean().item())
            max_tol, mean_tol = _tolerances("mha", q.dtype)
            timing_info = _KernelTiming(
                time_ms=time_ms,
                max_abs_diff=max_diff,
                mean_abs_diff=mean_diff,
                valid=max_diff <= max_tol and mean_diff <= mean_tol,
            )
        return out, timing_info
    finally:
        driver.unload_module(module)


def _launch_mha_bwd_kernel(
    driver: _CudaDriver,
    artifacts,
    defines: Mapping[str, str],
    meta,
    problem: Mapping[str, int],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    do: torch.Tensor,
    lse: torch.Tensor,
    delta: torch.Tensor,
    timing: _TimingConfig,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], _KernelTiming]:
    kernel_name = _require_define(defines, "CUTLASS_MHA_BWD_KERNEL_NAME")
    module = driver.load_module(artifacts.cubin)
    try:
        function = driver.get_function(module, kernel_name)
        driver.set_dynamic_smem(function, meta.shared_mem_bytes)

        batch = int(problem["batch_size"])
        heads = int(problem["num_heads"])
        seq = int(problem["sequence_length"])
        head_dim = int(problem["head_dim"])

        grid_x = 1
        grid_y = max(1, heads)
        grid_z = max(1, batch)
        block = (meta.num_warps * 32, 1, 1)

        dq = torch.empty((batch, seq, heads, head_dim), device=q.device, dtype=q.dtype).contiguous()
        dk = torch.empty_like(dq)
        dv = torch.empty_like(dq)

        block_i = int(meta.block_size_i)
        block_j = int(meta.block_size_j)
        gradq_temp_bytes = int(meta.gradq_temp_bytes)
        num_blocks = _ceil_div(seq, block_i)
        num_cols = _ceil_div(head_dim, block_j)
        num_key_blocks = _ceil_div(seq, block_j)
        num_splits_key = max(1, min(num_key_blocks, 8))
        grid_x = num_splits_key
        gradq_temp_elems = _ceil_div(gradq_temp_bytes, 4) if gradq_temp_bytes > 0 else 0

        output_in_rf = head_dim <= block_i
        workspace_elements_gk = 0
        workspace_elements_gv = 0
        if not output_in_rf:
            aligned_keys = _align_up(seq, block_j)
            aligned_head = _align_up(head_dim, block_i)
            workspace_elements_gk = num_splits_key * aligned_keys * aligned_head
            workspace_elements_gv = num_splits_key * aligned_keys * aligned_head

        workspace_elements_gq = num_blocks * num_cols * gradq_temp_elems
        workspace_stride_bh = _align_up(workspace_elements_gk + workspace_elements_gv + workspace_elements_gq, 4)
        workspace_bytes = workspace_stride_bh * batch * heads * 4
        workspace = None

        if workspace_bytes > 0:
            total_elems = int(workspace_stride_bh * batch * heads)
            workspace = torch.zeros((total_elems,), device=q.device, dtype=torch.int32)

        q_stride_z = int(q.stride(0))
        q_stride_t = int(q.stride(1))
        q_stride_h = int(q.stride(2))
        k_stride_z = int(k.stride(0))
        k_stride_t = int(k.stride(1))
        k_stride_h = int(k.stride(2))
        v_stride_z = int(v.stride(0))
        v_stride_t = int(v.stride(1))
        v_stride_h = int(v.stride(2))
        o_stride_z = int(o.stride(0))
        o_stride_t = int(o.stride(1))
        o_stride_h = int(o.stride(2))
        do_stride_z = int(do.stride(0))
        do_stride_t = int(do.stride(1))
        do_stride_h = int(do.stride(2))

        softmax_scale = ctypes.c_float(1.0 / math.sqrt(head_dim))
        batch_arg = ctypes.c_uint32(batch)
        heads_arg = ctypes.c_uint32(heads)
        seq_arg = ctypes.c_uint32(seq)
        workspace_ptr = ctypes.c_void_p(workspace.data_ptr() if workspace is not None else 0)

        q_ptr = ctypes.c_void_p(q.data_ptr())
        k_ptr = ctypes.c_void_p(k.data_ptr())
        v_ptr = ctypes.c_void_p(v.data_ptr())
        o_ptr = ctypes.c_void_p(o.data_ptr())
        do_ptr = ctypes.c_void_p(do.data_ptr())
        dq_ptr = ctypes.c_void_p(dq.data_ptr())
        dk_ptr = ctypes.c_void_p(dk.data_ptr())
        dv_ptr = ctypes.c_void_p(dv.data_ptr())
        lse_ptr = ctypes.c_void_p(lse.data_ptr())
        delta_ptr = ctypes.c_void_p(delta.data_ptr())

        q_stride_z_arg = ctypes.c_uint32(q_stride_z)
        q_stride_h_arg = ctypes.c_uint32(q_stride_h)
        q_stride_t_arg = ctypes.c_uint32(q_stride_t)
        k_stride_z_arg = ctypes.c_uint32(k_stride_z)
        k_stride_h_arg = ctypes.c_uint32(k_stride_h)
        k_stride_t_arg = ctypes.c_uint32(k_stride_t)
        v_stride_z_arg = ctypes.c_uint32(v_stride_z)
        v_stride_h_arg = ctypes.c_uint32(v_stride_h)
        v_stride_t_arg = ctypes.c_uint32(v_stride_t)
        o_stride_z_arg = ctypes.c_uint32(o_stride_z)
        o_stride_h_arg = ctypes.c_uint32(o_stride_h)
        o_stride_t_arg = ctypes.c_uint32(o_stride_t)
        do_stride_z_arg = ctypes.c_uint32(do_stride_z)
        do_stride_h_arg = ctypes.c_uint32(do_stride_h)
        do_stride_t_arg = ctypes.c_uint32(do_stride_t)

        kernel_args = [
            ctypes.cast(ctypes.byref(q_ptr), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(q_stride_z_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(q_stride_h_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(q_stride_t_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(k_ptr), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(k_stride_z_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(k_stride_h_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(k_stride_t_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(v_ptr), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(v_stride_z_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(v_stride_h_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(v_stride_t_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(o_ptr), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(o_stride_z_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(o_stride_h_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(o_stride_t_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(softmax_scale), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(do_ptr), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(do_stride_z_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(do_stride_h_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(do_stride_t_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(dq_ptr), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(dk_ptr), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(dv_ptr), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(lse_ptr), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(delta_ptr), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(batch_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(heads_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(seq_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(workspace_ptr), ctypes.c_void_p),
        ]
        args_array = (ctypes.c_void_p * len(kernel_args))(*[arg.value for arg in kernel_args])
        stream_ptr = torch.cuda.current_stream(q.device).cuda_stream

        def _launch() -> None:
            if workspace is not None and num_splits_key > 1:
                workspace.zero_()
            driver.launch_kernel(function, (grid_x, grid_y, grid_z), block, meta.shared_mem_bytes, stream_ptr, args_array)

        time_ms = _time_function(_launch, timing)

        q_ref = q.float().permute(0, 2, 1, 3).contiguous().detach().requires_grad_(True)
        k_ref = k.float().permute(0, 2, 1, 3).contiguous().detach().requires_grad_(True)
        v_ref = v.float().permute(0, 2, 1, 3).contiguous().detach().requires_grad_(True)
        do_ref = do.float().permute(0, 2, 1, 3).contiguous()

        scores = torch.matmul(q_ref, k_ref.transpose(-1, -2)) * (1.0 / math.sqrt(head_dim))
        probs = torch.softmax(scores, dim=-1)
        out_ref = torch.matmul(probs, v_ref)
        out_ref.backward(do_ref)

        dq_ref = q_ref.grad.permute(0, 2, 1, 3).contiguous().to(dq.dtype)
        dk_ref = k_ref.grad.permute(0, 2, 1, 3).contiguous().to(dk.dtype)
        dv_ref = v_ref.grad.permute(0, 2, 1, 3).contiguous().to(dv.dtype)

        diff_q = (dq - dq_ref).abs().float()
        diff_k = (dk - dk_ref).abs().float()
        diff_v = (dv - dv_ref).abs().float()
        max_diff = float(torch.max(torch.stack([diff_q.max(), diff_k.max(), diff_v.max()])).item())
        mean_diff = float(torch.mean(torch.stack([diff_q.mean(), diff_k.mean(), diff_v.mean()])).item())
        max_tol, mean_tol = _tolerances("mha", q.dtype)
        timing_info = _KernelTiming(
            time_ms=time_ms,
            max_abs_diff=max_diff,
            mean_abs_diff=mean_diff,
            valid=max_diff <= max_tol and mean_diff <= mean_tol,
        )
        return (dq, dk, dv), timing_info
    finally:
        driver.unload_module(module)


def _tune_conv_config(
    name: str,
    spec: Mapping[str, Any],
    base_defines: Mapping[str, str],
    candidates: Sequence[Mapping[str, Any]],
    problem: Mapping[str, int],
    sm: str,
    timing: _TimingConfig,
    driver: _CudaDriver,
    device: torch.device,
) -> tuple[dict[str, Any], _KernelTiming]:
    _validate_search_space("conv2d", name, base_defines, candidates)
    dtype = _detect_conv_dtype(base_defines)
    x, w, bias = _generate_conv_inputs(problem, dtype, device)
    source_path = _resolve_source_path(spec.get("source_file"), _CONV_SOURCE_FILE)
    best: Optional[tuple[dict[str, str], _KernelTiming]] = None

    for override in candidates:
        defines = _merge_defines(base_defines, override)
        defines = _apply_required_cutlass_defines(defines, category="conv2d", sm=sm)
        defines = _inject_sm_arch_define(defines, sm, "CUTLASS_CONV_SM_ARCH")
        try:
            artifacts = build_cutlass_artifacts(verbose=False, defines=defines, sm=sm, source_file=source_path)
        except Exception as exc:
            print(f"[conv2d] {name}: skipping candidate {override} (build failed: {exc})")
            continue
        metadata = json.loads(artifacts.metadata.read_text())
        try:
            meta = _conv_kernel_meta_from_defines(
                defines,
                shared_mem_bytes=int(metadata.get("shared_mem_bytes", 0)),
                num_warps=int(metadata.get("num_warps", 0)) or None,
            )
        except Exception as exc:
            print(f"[conv2d] {name}: skipping candidate {override} (metadata failed: {exc})")
            continue
        try:
            _, metrics = _launch_conv_kernel(driver, artifacts, defines, meta, problem, x, w, bias, timing)
        except Exception as exc:
            print(f"[conv2d] {name}: skipping candidate {override} (launch failed: {exc})")
            continue
        if not metrics.valid:
            print(
                f"[conv2d] {name}: candidate {override} failed validation "
                f"(max diff {metrics.max_abs_diff:.3e}, mean diff {metrics.mean_abs_diff:.3e})"
            )
            continue
        if best is None or metrics.time_ms < best[1].time_ms:
            best = (defines, metrics)

    if best is None:
        raise RuntimeError(f"[conv2d] {name}: no valid candidate found.")
    defines, metrics = best
    entry = {
        "defines": dict(defines),
        "metrics": {
            "kernel_ms": _normalise_metric(metrics.time_ms),
            "max_abs_diff": _normalise_metric(metrics.max_abs_diff),
            "mean_abs_diff": _normalise_metric(metrics.mean_abs_diff),
            "validation_passed": metrics.valid,
        },
        "problem": {k: int(v) for k, v in problem.items()},
        "autotuned": True,
    }
    if "source_file" in spec:
        src = Path(spec["source_file"])
        if src.is_absolute():
            src = src.relative_to(_ROOT_DIR)
        entry["source_file"] = str(src)
    if "basename" in spec:
        entry["basename"] = spec["basename"]
    return entry, metrics


def _tune_conv_dgrad_config(
    name: str,
    spec: Mapping[str, Any],
    base_defines: Mapping[str, str],
    candidates: Sequence[Mapping[str, Any]],
    problem: Mapping[str, int],
    sm: str,
    timing: _TimingConfig,
    driver: _CudaDriver,
    device: torch.device,
) -> tuple[dict[str, Any], _KernelTiming]:
    _validate_search_space("conv2d_dgrad", name, base_defines, candidates)
    dtype = _detect_conv_dtype(base_defines)
    x, w, _bias = _generate_conv_inputs(problem, dtype, device)
    out_h, out_w = _conv_output_dims(problem)
    dout = torch.randn(
        (int(problem["batch"]), int(problem["out_channels"]), out_h, out_w),
        device=device,
        dtype=torch.float32,
    ).to(dtype).contiguous(memory_format=torch.channels_last)
    source_path = _resolve_source_path(spec.get("source_file"), _CONV_DGRAD_SOURCE_FILE)
    best: Optional[tuple[dict[str, str], _KernelTiming]] = None

    for override in candidates:
        defines = _merge_defines(base_defines, override)
        defines = _apply_required_cutlass_defines(defines, category="conv2d_dgrad", sm=sm)
        defines = _inject_sm_arch_define(defines, sm, "CUTLASS_CONV_SM_ARCH")
        try:
            artifacts = build_cutlass_artifacts(verbose=False, defines=defines, sm=sm, source_file=source_path)
        except Exception as exc:
            print(f"[conv2d_dgrad] {name}: skipping candidate {override} (build failed: {exc})")
            continue
        metadata = json.loads(artifacts.metadata.read_text())
        try:
            meta = _conv_kernel_meta_from_defines(
                defines,
                shared_mem_bytes=int(metadata.get("shared_mem_bytes", 0)),
                num_warps=int(metadata.get("num_warps", 0)) or None,
            )
        except Exception as exc:
            print(f"[conv2d_dgrad] {name}: skipping candidate {override} (metadata failed: {exc})")
            continue
        try:
            _, metrics = _launch_conv_dgrad_kernel(driver, artifacts, defines, meta, problem, dout, w, timing)
        except Exception as exc:
            print(f"[conv2d_dgrad] {name}: skipping candidate {override} (launch failed: {exc})")
            continue
        if not metrics.valid:
            print(
                f"[conv2d_dgrad] {name}: candidate {override} failed validation "
                f"(max diff {metrics.max_abs_diff:.3e}, mean diff {metrics.mean_abs_diff:.3e})"
            )
            continue
        if best is None or metrics.time_ms < best[1].time_ms:
            best = (defines, metrics)

    if best is None:
        raise RuntimeError(f"[conv2d_dgrad] {name}: no valid candidate found.")
    defines, metrics = best
    entry = {
        "defines": dict(defines),
        "metrics": {
            "kernel_ms": _normalise_metric(metrics.time_ms),
            "max_abs_diff": _normalise_metric(metrics.max_abs_diff),
            "mean_abs_diff": _normalise_metric(metrics.mean_abs_diff),
            "validation_passed": metrics.valid,
        },
        "problem": {k: int(v) for k, v in problem.items()},
        "autotuned": True,
    }
    if "source_file" in spec:
        src = Path(spec["source_file"])
        if src.is_absolute():
            src = src.relative_to(_ROOT_DIR)
        entry["source_file"] = str(src)
    if "basename" in spec:
        entry["basename"] = spec["basename"]
    return entry, metrics


def _tune_conv_wgrad_config(
    name: str,
    spec: Mapping[str, Any],
    base_defines: Mapping[str, str],
    candidates: Sequence[Mapping[str, Any]],
    problem: Mapping[str, int],
    sm: str,
    timing: _TimingConfig,
    driver: _CudaDriver,
    device: torch.device,
) -> tuple[dict[str, Any], _KernelTiming]:
    _validate_search_space("conv2d_wgrad", name, base_defines, candidates)
    dtype = _detect_conv_dtype(base_defines)
    x, _w, _bias = _generate_conv_inputs(problem, dtype, device)
    out_h, out_w = _conv_output_dims(problem)
    dout = torch.randn(
        (int(problem["batch"]), int(problem["out_channels"]), out_h, out_w),
        device=device,
        dtype=torch.float32,
    ).to(dtype).contiguous(memory_format=torch.channels_last)
    source_path = _resolve_source_path(spec.get("source_file"), _CONV_WGRAD_SOURCE_FILE)
    best: Optional[tuple[dict[str, str], _KernelTiming]] = None

    for override in candidates:
        defines = _merge_defines(base_defines, override)
        defines = _apply_required_cutlass_defines(defines, category="conv2d_wgrad", sm=sm)
        defines = _inject_sm_arch_define(defines, sm, "CUTLASS_CONV_SM_ARCH")
        try:
            artifacts = build_cutlass_artifacts(verbose=False, defines=defines, sm=sm, source_file=source_path)
        except Exception as exc:
            print(f"[conv2d_wgrad] {name}: skipping candidate {override} (build failed: {exc})")
            continue
        metadata = json.loads(artifacts.metadata.read_text())
        try:
            meta = _conv_kernel_meta_from_defines(
                defines,
                shared_mem_bytes=int(metadata.get("shared_mem_bytes", 0)),
                num_warps=int(metadata.get("num_warps", 0)) or None,
            )
        except Exception as exc:
            print(f"[conv2d_wgrad] {name}: skipping candidate {override} (metadata failed: {exc})")
            continue
        try:
            _, metrics = _launch_conv_wgrad_kernel(driver, artifacts, defines, meta, problem, x, dout, timing)
        except Exception as exc:
            print(f"[conv2d_wgrad] {name}: skipping candidate {override} (launch failed: {exc})")
            continue
        if not metrics.valid:
            print(
                f"[conv2d_wgrad] {name}: candidate {override} failed validation "
                f"(max diff {metrics.max_abs_diff:.3e}, mean diff {metrics.mean_abs_diff:.3e})"
            )
            continue
        if best is None or metrics.time_ms < best[1].time_ms:
            best = (defines, metrics)

    if best is None:
        raise RuntimeError(f"[conv2d_wgrad] {name}: no valid candidate found.")
    defines, metrics = best
    entry = {
        "defines": dict(defines),
        "metrics": {
            "kernel_ms": _normalise_metric(metrics.time_ms),
            "max_abs_diff": _normalise_metric(metrics.max_abs_diff),
            "mean_abs_diff": _normalise_metric(metrics.mean_abs_diff),
            "validation_passed": metrics.valid,
        },
        "problem": {k: int(v) for k, v in problem.items()},
        "autotuned": True,
    }
    if "source_file" in spec:
        src = Path(spec["source_file"])
        if src.is_absolute():
            src = src.relative_to(_ROOT_DIR)
        entry["source_file"] = str(src)
    if "basename" in spec:
        entry["basename"] = spec["basename"]
    return entry, metrics


def _tune_gemm_config(
    name: str,
    spec: Mapping[str, Any],
    base_defines: Mapping[str, str],
    candidates: Sequence[Mapping[str, Any]],
    problem: Mapping[str, int],
    sm: str,
    timing: _TimingConfig,
    driver: _CudaDriver,
    device: torch.device,
) -> tuple[dict[str, Any], _KernelTiming]:
    _validate_search_space("gemm", name, base_defines, candidates)
    dtype = _detect_gemm_dtype(base_defines)
    a, b, bias = _generate_gemm_inputs(problem, dtype, device)
    source_path = _resolve_source_path(spec.get("source_file"), _GEMM_SOURCE_FILE)
    best: Optional[tuple[dict[str, str], _KernelTiming]] = None

    for override in candidates:
        defines = _merge_defines(base_defines, override)
        defines = _apply_required_cutlass_defines(defines, category="gemm", sm=sm)
        defines = _inject_sm_arch_define(defines, sm, "CUTLASS_GEMM_SM_ARCH")
        try:
            artifacts = build_cutlass_artifacts(verbose=False, defines=defines, sm=sm, source_file=source_path)
        except Exception as exc:
            print(f"[gemm] {name}: skipping candidate {override} (build failed: {exc})")
            continue
        metadata = json.loads(artifacts.metadata.read_text())
        try:
            meta = _gemm_kernel_meta_from_defines(
                defines,
                shared_mem_bytes=int(metadata.get("shared_mem_bytes", 0)),
                num_warps=int(metadata.get("num_warps", 0)) or None,
            )
        except Exception as exc:
            print(f"[gemm] {name}: skipping candidate {override} (metadata failed: {exc})")
            continue
        try:
            _, metrics = _launch_gemm_kernel(driver, artifacts, defines, meta, problem, a, b, bias, timing)
        except Exception as exc:
            print(f"[gemm] {name}: skipping candidate {override} (launch failed: {exc})")
            continue
        if not metrics.valid:
            print(
                f"[gemm] {name}: candidate {override} failed validation "
                f"(max diff {metrics.max_abs_diff:.3e}, mean diff {metrics.mean_abs_diff:.3e})"
            )
            continue
        if best is None or metrics.time_ms < best[1].time_ms:
            best = (defines, metrics)

    if best is None:
        raise RuntimeError(f"[gemm] {name}: no valid candidate found.")
    defines, metrics = best
    entry = {
        "defines": dict(defines),
        "metrics": {
            "kernel_ms": _normalise_metric(metrics.time_ms),
            "max_abs_diff": _normalise_metric(metrics.max_abs_diff),
            "mean_abs_diff": _normalise_metric(metrics.mean_abs_diff),
            "validation_passed": metrics.valid,
        },
        "problem": {k: int(v) for k, v in problem.items()},
        "autotuned": True,
    }
    if "source_file" in spec:
        src = Path(spec["source_file"])
        if src.is_absolute():
            src = src.relative_to(_ROOT_DIR)
        entry["source_file"] = str(src)
    if "basename" in spec:
        entry["basename"] = spec["basename"]
    return entry, metrics


def _tune_mha_config(
    name: str,
    spec: Mapping[str, Any],
    base_defines: Mapping[str, str],
    candidates: Sequence[Mapping[str, Any]],
    problem: Mapping[str, int],
    sm: str,
    timing: _TimingConfig,
    driver: _CudaDriver,
    device: torch.device,
) -> tuple[dict[str, Any], _KernelTiming]:
    _validate_search_space("mha", name, base_defines, candidates)
    dtype = _detect_mha_dtype(base_defines)
    source_path = _resolve_source_path(spec.get("source_file"), _MHA_SOURCE_FILE)
    best: Optional[tuple[dict[str, str], _KernelTiming]] = None

    if not ENABLE_MHA_TUNING:
        raise RuntimeError(f"[mha] {name}: autotuning disabled.")

    q, k, v = _generate_mha_inputs(problem, dtype, device)

    for override in candidates:
        defines = _merge_defines(base_defines, override)
        defines = _apply_required_cutlass_defines(defines, category="mha", sm=sm)
        defines = _inject_sm_arch_define(
            defines,
            sm,
            "CUTLASS_MHA_SM_ARCH",
            sm_version_key="CUTLASS_MHA_SM_VERSION",
        )
        try:
            artifacts = build_cutlass_artifacts(verbose=False, defines=defines, sm=sm, source_file=source_path)
        except Exception as exc:
            print(f"[mha] {name}: skipping candidate {override} (build failed: {exc})")
            continue
        metadata = json.loads(artifacts.metadata.read_text())
        shared_mem_bytes = int(metadata.get("shared_mem_bytes", 0))
        num_warps_override = int(metadata.get("num_warps", 0))
        if num_warps_override <= 0:
            try:
                num_warps_override = int(_parse_constant_from_ptx(artifacts.ptx, "cutlass_mha_num_warps"))
            except Exception:
                num_warps_override = 0
        if shared_mem_bytes <= 0:
            try:
                shared_mem_bytes = int(_parse_constant_from_ptx(artifacts.ptx, "cutlass_mha_shared_mem_bytes"))
            except Exception:
                shared_mem_bytes = 0
        try:
            meta = _mha_kernel_meta_from_defines(
                defines,
                shared_mem_bytes=shared_mem_bytes,
                num_warps=num_warps_override or None,
            )
        except Exception as exc:
            print(f"[mha] {name}: skipping candidate {override} (metadata failed: {exc})")
            continue
        try:
            _, metrics = _launch_mha_kernel(driver, artifacts, defines, meta, problem, q, k, v, timing)
        except Exception as exc:
            print(f"[mha] {name}: skipping candidate {override} (launch failed: {exc})")
            continue
        if not metrics.valid:
            print(
                f"[mha] {name}: candidate {override} failed validation "
                f"(max diff {metrics.max_abs_diff:.3e}, mean diff {metrics.mean_abs_diff:.3e})"
            )
            continue
        if best is None or metrics.time_ms < best[1].time_ms:
            best = (defines, metrics)

    if best is None:
        raise RuntimeError(f"[mha] {name}: no valid candidate found.")
    defines, metrics = best
    entry = {
        "defines": dict(defines),
        "metrics": {
            "kernel_ms": _normalise_metric(metrics.time_ms),
            "max_abs_diff": _normalise_metric(metrics.max_abs_diff),
            "mean_abs_diff": _normalise_metric(metrics.mean_abs_diff),
            "validation_passed": metrics.valid,
        },
        "problem": {k: int(v) for k, v in problem.items()},
        "autotuned": True,
    }
    if "source_file" in spec:
        src = Path(spec["source_file"])
        if src.is_absolute():
            src = src.relative_to(_ROOT_DIR)
        entry["source_file"] = str(src)
    if "basename" in spec:
        entry["basename"] = spec["basename"]
    return entry, metrics


def _tune_mha_bwd_config(
    name: str,
    spec: Mapping[str, Any],
    base_defines: Mapping[str, str],
    candidates: Sequence[Mapping[str, Any]],
    problem: Mapping[str, int],
    sm: str,
    timing: _TimingConfig,
    driver: _CudaDriver,
    device: torch.device,
) -> tuple[dict[str, Any], _KernelTiming]:
    _validate_search_space("mha_bwd", name, base_defines, candidates)
    dtype = _detect_mha_bwd_dtype(base_defines)
    source_path = _resolve_source_path(spec.get("source_file"), _MHA_BWD_SOURCE_FILE)
    best: Optional[tuple[dict[str, str], _KernelTiming]] = None
    q, k, v, o, do, lse, delta = _generate_mha_bwd_inputs(problem, dtype, device)
    for override in candidates:
        defines = _merge_defines(base_defines, override)
        defines = _apply_required_cutlass_defines(defines, category="mha_bwd", sm=sm)
        defines = _inject_sm_arch_define(
            defines,
            sm,
            "CUTLASS_MHA_BWD_SM_ARCH",
            sm_version_key="CUTLASS_MHA_BWD_SM_VERSION",
        )
        try:
            artifacts = build_cutlass_artifacts(verbose=False, defines=defines, sm=sm, source_file=source_path)
        except Exception as exc:
            print(f"[mha_bwd] {name}: skipping candidate {override} (build failed: {exc})")
            continue
        metadata = json.loads(artifacts.metadata.read_text())
        shared_mem_bytes = int(metadata.get("shared_mem_bytes", 0))
        num_warps_override = int(metadata.get("num_warps", 0))
        if num_warps_override <= 0:
            try:
                num_warps_override = int(_parse_constant_from_ptx(artifacts.ptx, "cutlass_mha_bwd_num_warps"))
            except Exception:
                num_warps_override = 0
        if shared_mem_bytes <= 0:
            try:
                shared_mem_bytes = int(_parse_constant_from_ptx(artifacts.ptx, "cutlass_mha_bwd_shared_mem_bytes"))
            except Exception:
                shared_mem_bytes = 0
        try:
            gradq_tile_elements = int(_parse_constant_from_ptx(artifacts.ptx, "cutlass_mha_bwd_gradq_tile_elements"))
            gradq_temp_bytes = int(_parse_constant_from_ptx(artifacts.ptx, "cutlass_mha_bwd_gradq_temp_bytes"))
            meta = _mha_bwd_kernel_meta_from_defines(
                defines,
                shared_mem_bytes=shared_mem_bytes,
                num_warps=num_warps_override or None,
                gradq_tile_elements=gradq_tile_elements,
                gradq_temp_bytes=gradq_temp_bytes,
            )
        except Exception as exc:
            print(f"[mha_bwd] {name}: skipping candidate {override} (metadata failed: {exc})")
            continue
        try:
            _, metrics = _launch_mha_bwd_kernel(
                driver, artifacts, defines, meta, problem, q, k, v, o, do, lse, delta, timing
            )
        except Exception as exc:
            print(f"[mha_bwd] {name}: skipping candidate {override} (launch failed: {exc})")
            continue
        if not metrics.valid:
            print(
                f"[mha_bwd] {name}: candidate {override} failed validation "
                f"(max diff {metrics.max_abs_diff:.3e}, mean diff {metrics.mean_abs_diff:.3e})"
            )
            continue
        if best is None or metrics.time_ms < best[1].time_ms:
            best = (defines, metrics)

    if best is None:
        raise RuntimeError(f"[mha_bwd] {name}: no valid candidate found.")
    defines, metrics = best
    entry = {
        "defines": dict(defines),
        "metrics": {
            "kernel_ms": _normalise_metric(metrics.time_ms),
            "max_abs_diff": _normalise_metric(metrics.max_abs_diff),
            "mean_abs_diff": _normalise_metric(metrics.mean_abs_diff),
            "validation_passed": metrics.valid,
        },
        "problem": {k: int(v) for k, v in problem.items()},
        "autotuned": True,
    }
    if "source_file" in spec:
        entry["source_file"] = spec["source_file"]
    if "basename" in spec:
        entry["basename"] = spec["basename"]
    return entry, metrics


def _new_output_payload() -> dict[str, dict[str, Any]]:
    return {
        "conv2d": {},
        "conv2d_dgrad": {},
        "conv2d_wgrad": {},
        "gemm": {},
        "mha": {},
        "mha_bwd": {},
    }


def _run_autotune_subprocess(args: list[str]) -> None:
    env = dict(os.environ)
    subprocess.run(args, check=True, env=env)


def _autotune_mha_only_isolated(
    *,
    script_path: Path,
    sm: str,
    iters: int,
    warmup: int,
    output: Path,
    base_configs: Sequence[Path],
    search_space: Optional[Path],
    mha_targets: Sequence[str],
    mha_bwd_targets: Sequence[str],
) -> None:
    output_payload = _new_output_payload()
    total_tuned = 0
    failures: list[str] = []

    with tempfile.TemporaryDirectory(prefix="cutlass_mha_autotune_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        common_args = [
            sys.executable,
            str(script_path),
            "--sm",
            sm,
            "--iters",
            str(iters),
            "--warmup",
            str(warmup),
        ]
        for base in base_configs:
            common_args.extend(["--base-config", str(base)])
        if search_space is not None:
            common_args.extend(["--search-space", str(search_space)])

        for name in mha_targets:
            partial_path = tmp_dir / f"mha_{name}.json"
            cmd = [*common_args, "--output", str(partial_path), "--mha-config", name]
            try:
                _run_autotune_subprocess(cmd)
            except subprocess.CalledProcessError:
                failures.append(f"mha:{name}")
                continue
            payload = json.loads(partial_path.read_text())
            entry = payload.get("mha", {}).get(name)
            if not isinstance(entry, Mapping):
                failures.append(f"mha:{name}")
                continue
            output_payload["mha"][name] = dict(entry)
            total_tuned += 1

        for name in mha_bwd_targets:
            partial_path = tmp_dir / f"mha_bwd_{name}.json"
            cmd = [*common_args, "--output", str(partial_path), "--mha-bwd-config", name]
            try:
                _run_autotune_subprocess(cmd)
            except subprocess.CalledProcessError:
                failures.append(f"mha_bwd:{name}")
                continue
            payload = json.loads(partial_path.read_text())
            entry = payload.get("mha_bwd", {}).get(name)
            if not isinstance(entry, Mapping):
                failures.append(f"mha_bwd:{name}")
                continue
            output_payload["mha_bwd"][name] = dict(entry)
            total_tuned += 1

    if total_tuned == 0:
        raise RuntimeError("MHA-only autotune did not produce any successful configuration.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(output_payload, indent=2, sort_keys=True, allow_nan=False))
    if failures:
        print(f"Skipped {len(failures)} failing MHA config(s): {', '.join(failures)}")
    print(f"Wrote {total_tuned} tuned CUTLASS MHA entries to {output}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Auto-tune CUTLASS kernel configurations.")
    parser.add_argument("--sm", default="sm_89", help="Target SM architecture (e.g. sm_89).")
    parser.add_argument("--iters", type=int, default=50, help="Timed iterations per candidate.")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations per candidate.")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSON for tuned constants.")
    parser.add_argument("--search-space", type=Path, help="Optional JSON describing candidate overrides and problems.")
    parser.add_argument(
        "--base-config",
        action="append",
        dest="base_configs",
        default=[],
        help="Path to a base config JSON defining kernel configs to tune.",
    )
    parser.add_argument(
        "--conv2d-configs-file",
        action="append",
        dest="conv2d_config_files",
        default=[],
        help="Path to a JSON file containing additional conv2d kernel configs to tune.",
    )
    parser.add_argument("--config", action="append", dest="configs", help="Conv2d configuration to tune.")
    parser.add_argument("--all-configs", action="store_true", help="Tune all available conv2d configurations.")
    parser.add_argument(
        "--conv2d-dgrad-config",
        action="append",
        dest="conv2d_dgrad_configs",
        help="Conv2d dgrad configuration to tune.",
    )
    parser.add_argument(
        "--all-conv2d-dgrad-configs",
        action="store_true",
        help="Tune all available conv2d dgrad configurations.",
    )
    parser.add_argument(
        "--conv2d-wgrad-config",
        action="append",
        dest="conv2d_wgrad_configs",
        help="Conv2d wgrad configuration to tune.",
    )
    parser.add_argument(
        "--all-conv2d-wgrad-configs",
        action="store_true",
        help="Tune all available conv2d wgrad configurations.",
    )
    parser.add_argument("--gemm-config", action="append", dest="gemm_configs", help="GEMM configuration to tune.")
    parser.add_argument("--all-gemm-configs", action="store_true", help="Tune all available GEMM configurations.")
    parser.add_argument("--mha-config", action="append", dest="mha_configs", help="MHA configuration to tune.")
    parser.add_argument("--all-mha-configs", action="store_true", help="Tune all available MHA configurations.")
    parser.add_argument("--mha-bwd-config", action="append", dest="mha_bwd_configs", help="MHA bwd configuration to tune.")
    parser.add_argument("--all-mha-bwd-configs", action="store_true", help="Tune all available MHA bwd configurations.")
    parser.add_argument(
        "--mha-only",
        action="store_true",
        help="Tune only MHA and MHA bwd configs in isolated subprocesses (one process per config).",
    )
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required for autotuning.")
    major, minor = torch.cuda.get_device_capability()
    actual_sm = _normalise_sm(f"sm_{major}{minor}")
    requested_sm = _normalise_sm(args.sm)
    if actual_sm != requested_sm:
        raise RuntimeError(f"Requested SM {requested_sm} does not match GPU SM {actual_sm}.")

    base_paths = [Path(p) for p in args.base_configs]
    conv_configs = _load_base_entries(base_paths, "conv2d")
    for cfg_path in args.conv2d_config_files:
        entries = _load_conv2d_config_entries(Path(cfg_path))
        for name, spec in entries.items():
            if name in conv_configs:
                raise RuntimeError(f"Conv2d configuration '{name}' is already defined by base configs.")
            conv_configs[name] = spec
    conv_dgrad_configs = _load_base_entries(base_paths, "conv2d_dgrad")
    conv_wgrad_configs = _load_base_entries(base_paths, "conv2d_wgrad")
    conv_dgrad_configs = _ensure_conv_bwd_kernel_names(
        conv_dgrad_configs,
        op="dgrad",
        source_file=_CONV_DGRAD_SOURCE_FILE,
    )
    conv_wgrad_configs = _ensure_conv_bwd_kernel_names(
        conv_wgrad_configs,
        op="wgrad",
        source_file=_CONV_WGRAD_SOURCE_FILE,
    )
    gemm_configs = _load_base_entries(base_paths, "gemm")
    mha_configs = _load_base_entries(base_paths, "mha")
    mha_bwd_configs = _load_base_entries(base_paths, "mha_bwd")

    if (args.all_configs or args.configs) and not conv_configs:
        raise RuntimeError("No conv2d base configs loaded. Provide --base-config or --conv2d-configs-file.")
    if (args.all_conv2d_dgrad_configs or args.conv2d_dgrad_configs) and not conv_dgrad_configs:
        raise RuntimeError("No conv2d dgrad base configs loaded. Provide --base-config.")
    if (args.all_conv2d_wgrad_configs or args.conv2d_wgrad_configs) and not conv_wgrad_configs:
        raise RuntimeError("No conv2d wgrad base configs loaded. Provide --base-config.")
    if (args.all_gemm_configs or args.gemm_configs) and not gemm_configs:
        raise RuntimeError("No GEMM base configs loaded. Provide --base-config.")
    if (args.all_mha_configs or args.mha_configs) and not mha_configs:
        raise RuntimeError("No MHA base configs loaded. Provide --base-config.")
    if (args.all_mha_bwd_configs or args.mha_bwd_configs) and not mha_bwd_configs:
        raise RuntimeError("No MHA bwd base configs loaded. Provide --base-config.")

    if args.all_configs:
        conv_targets = sorted(conv_configs.keys())
    else:
        conv_targets = args.configs or []
    if args.all_conv2d_dgrad_configs:
        conv_dgrad_targets = sorted(conv_dgrad_configs.keys())
    else:
        conv_dgrad_targets = args.conv2d_dgrad_configs or []
    if args.all_conv2d_wgrad_configs:
        conv_wgrad_targets = sorted(conv_wgrad_configs.keys())
    else:
        conv_wgrad_targets = args.conv2d_wgrad_configs or []
    if args.all_gemm_configs:
        gemm_targets = sorted(gemm_configs.keys())
    else:
        gemm_targets = args.gemm_configs or []
    if args.all_mha_configs:
        mha_targets = sorted(mha_configs.keys())
    else:
        mha_targets = args.mha_configs or []
    if args.all_mha_bwd_configs:
        mha_bwd_targets = sorted(mha_bwd_configs.keys())
    else:
        mha_bwd_targets = args.mha_bwd_configs or []

    if args.mha_only:
        if args.all_configs or args.configs or args.all_conv2d_dgrad_configs or args.conv2d_dgrad_configs:
            raise RuntimeError("--mha-only cannot be combined with conv2d tuning selections.")
        if args.all_conv2d_wgrad_configs or args.conv2d_wgrad_configs or args.all_gemm_configs or args.gemm_configs:
            raise RuntimeError("--mha-only cannot be combined with wgrad/gemm tuning selections.")
        if not mha_targets:
            mha_targets = sorted(mha_configs.keys())
        if not mha_bwd_targets:
            mha_bwd_targets = sorted(mha_bwd_configs.keys())
        if not mha_targets and not mha_bwd_targets:
            raise RuntimeError("--mha-only requested but no mha or mha_bwd configurations are available.")
        _autotune_mha_only_isolated(
            script_path=Path(__file__).resolve(),
            sm=args.sm,
            iters=max(args.iters, 1),
            warmup=max(args.warmup, 0),
            output=args.output,
            base_configs=base_paths,
            search_space=args.search_space,
            mha_targets=mha_targets,
            mha_bwd_targets=mha_bwd_targets,
        )
        return

    search_space = _load_search_space(args.search_space)
    device = torch.device("cuda", torch.cuda.current_device())
    driver = _CudaDriver()
    timing = _TimingConfig(iters=max(args.iters, 1), warmup=max(args.warmup, 0))

    output_payload: dict[str, dict[str, Any]] = {
        "conv2d": {},
        "conv2d_dgrad": {},
        "conv2d_wgrad": {},
        "gemm": {},
        "mha": {},
        "mha_bwd": {},
    }
    total_tuned = 0

    for name in conv_targets:
        if name not in conv_configs:
            raise RuntimeError(f"Unknown conv2d configuration '{name}'.")
        spec = conv_configs[name]
        if not spec.get("enabled", True):
            print(f"[conv2d] {name}: skipping (configuration disabled).")
            continue
        base_defines = _stringify_defines(spec.get("defines", {}))
        candidates, problem_override = _candidate_overrides("conv2d", name, search_space, DEFAULT_CONV_SEARCH)
        problem = _resolve_conv_problem(spec, problem_override)
        entry, metrics = _tune_conv_config(name, spec, base_defines, candidates, problem, args.sm, timing, driver, device)
        output_payload["conv2d"][name] = entry
        total_tuned += 1
        print(
            f"[conv2d] {name}: {metrics.time_ms:.3f} ms "
            f"(max diff {metrics.max_abs_diff:.3e}, mean diff {metrics.mean_abs_diff:.3e})"
        )

    for name in conv_dgrad_targets:
        if name not in conv_dgrad_configs:
            raise RuntimeError(f"Unknown conv2d dgrad configuration '{name}'.")
        spec = conv_dgrad_configs[name]
        if not spec.get("enabled", True):
            print(f"[conv2d_dgrad] {name}: skipping (configuration disabled).")
            continue
        base_defines = _stringify_defines(spec.get("defines", {}))
        candidates, problem_override = _candidate_overrides("conv2d", name, search_space, DEFAULT_CONV_SEARCH)
        problem = _resolve_conv_problem(spec, problem_override)
        entry, metrics = _tune_conv_dgrad_config(
            name,
            spec,
            base_defines,
            candidates,
            problem,
            args.sm,
            timing,
            driver,
            device,
        )
        output_payload["conv2d_dgrad"][name] = entry
        total_tuned += 1
        print(
            f"[conv2d_dgrad] {name}: {metrics.time_ms:.3f} ms "
            f"(max diff {metrics.max_abs_diff:.3e}, mean diff {metrics.mean_abs_diff:.3e})"
        )

    for name in conv_wgrad_targets:
        if name not in conv_wgrad_configs:
            raise RuntimeError(f"Unknown conv2d wgrad configuration '{name}'.")
        spec = conv_wgrad_configs[name]
        if not spec.get("enabled", True):
            print(f"[conv2d_wgrad] {name}: skipping (configuration disabled).")
            continue
        base_defines = _stringify_defines(spec.get("defines", {}))
        candidates, problem_override = _candidate_overrides("conv2d", name, search_space, DEFAULT_CONV_SEARCH)
        problem = _resolve_conv_problem(spec, problem_override)
        entry, metrics = _tune_conv_wgrad_config(
            name,
            spec,
            base_defines,
            candidates,
            problem,
            args.sm,
            timing,
            driver,
            device,
        )
        output_payload["conv2d_wgrad"][name] = entry
        total_tuned += 1
        print(
            f"[conv2d_wgrad] {name}: {metrics.time_ms:.3f} ms "
            f"(max diff {metrics.max_abs_diff:.3e}, mean diff {metrics.mean_abs_diff:.3e})"
        )

    for name in gemm_targets:
        if name not in gemm_configs:
            raise RuntimeError(f"Unknown GEMM configuration '{name}'.")
        spec = gemm_configs[name]
        if not spec.get("enabled", True):
            print(f"[gemm] {name}: skipping (configuration disabled).")
            continue
        base_defines = _stringify_defines(spec.get("defines", {}))
        candidates, problem_override = _candidate_overrides("gemm", name, search_space, DEFAULT_GEMM_SEARCH)
        problem = _resolve_gemm_problem(spec, problem_override)
        entry, metrics = _tune_gemm_config(name, spec, base_defines, candidates, problem, args.sm, timing, driver, device)
        output_payload["gemm"][name] = entry
        total_tuned += 1
        print(
            f"[gemm] {name}: {metrics.time_ms:.3f} ms "
            f"(max diff {metrics.max_abs_diff:.3e}, mean diff {metrics.mean_abs_diff:.3e})"
        )

    mha_default_search = _default_mha_search(args.sm)
    for name in mha_targets:
        if name not in mha_configs:
            raise RuntimeError(f"Unknown MHA configuration '{name}'.")
        spec = mha_configs[name]
        if not spec.get("enabled", True):
            print(f"[mha] {name}: skipping (configuration disabled).")
            continue
        base_defines = _stringify_defines(spec.get("defines", {}))
        candidates, problem_override = _candidate_overrides("mha", name, search_space, mha_default_search)
        problem = _resolve_mha_problem(base_defines, spec, problem_override)
        entry, metrics = _tune_mha_config(name, spec, base_defines, candidates, problem, args.sm, timing, driver, device)
        output_payload["mha"][name] = entry
        total_tuned += 1
        print(
            f"[mha] {name}: {metrics.time_ms:.3f} ms "
            f"(max diff {metrics.max_abs_diff:.3e}, mean diff {metrics.mean_abs_diff:.3e})"
        )

    for name in mha_bwd_targets:
        if name not in mha_bwd_configs:
            raise RuntimeError(f"Unknown MHA bwd configuration '{name}'.")
        spec = mha_bwd_configs[name]
        if not spec.get("enabled", True):
            print(f"[mha_bwd] {name}: skipping (configuration disabled).")
            continue
        base_defines = _stringify_defines(spec.get("defines", {}))
        candidates, problem_override = _candidate_overrides("mha_bwd", name, search_space, DEFAULT_MHA_BWD_SEARCH)
        problem = _resolve_mha_bwd_problem(base_defines, spec, problem_override)
        entry, metrics = _tune_mha_bwd_config(
            name, spec, base_defines, candidates, problem, args.sm, timing, driver, device
        )
        output_payload["mha_bwd"][name] = entry
        total_tuned += 1
        print(
            f"[mha_bwd] {name}: {metrics.time_ms:.3f} ms "
            f"(max diff {metrics.max_abs_diff:.3e}, mean diff {metrics.mean_abs_diff:.3e})"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output_payload, indent=2, sort_keys=True, allow_nan=False))
    print(f"Autotune completed for {total_tuned} configuration(s). Results written to {args.output}")


if __name__ == "__main__":  # pragma: no cover
    main()
