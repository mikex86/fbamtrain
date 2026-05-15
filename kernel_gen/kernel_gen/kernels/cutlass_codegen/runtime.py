from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
from dataclasses import dataclass, replace
from typing import Optional, Tuple

import torch

from .compile import CutlassArtifacts, build_cutlass_artifacts, ensure_workdir


class _CutlassConv2dConfig(ctypes.Structure):
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


_CUmodule = ctypes.c_void_p
_CUfunction = ctypes.c_void_p
_CUstream = ctypes.c_void_p

_CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8


@dataclass(frozen=True)
class _KernelMeta:
    kernel_name: str
    stride_h: int
    stride_w: int
    padding_h: int
    padding_w: int
    dilation_h: int
    dilation_w: int
    kernel_h: int
    kernel_w: int
    in_channels: Optional[int]
    groups: int
    block_pixels: int
    block_oc: int
    block_k: int
    num_warps: int
    split_k_slices: int
    shared_mem_bytes: int

    @property
    def block_threads(self) -> int:
        return self.num_warps * 32


class _RuntimeState:
    __slots__ = ("artifacts", "module_cache", "libcuda", "defines", "kernel_name", "meta")

    def __init__(self) -> None:
        self.artifacts: Optional[CutlassArtifacts] = None
        self.module_cache: dict[int, tuple[_CUmodule, _CUfunction]] = {}
        self.libcuda: Optional[ctypes.CDLL] = None
        self.defines: Optional[tuple[tuple[str, str], ...]] = None
        self.kernel_name: Optional[bytes] = None
        self.meta: Optional[_KernelMeta] = None


_STATE = _RuntimeState()
_CONV_CONFIGS: Optional[dict[str, dict[str, object]]] = None


def _load_conv2d_configs() -> dict[str, dict[str, object]]:
    global _CONV_CONFIGS
    if _CONV_CONFIGS is not None:
        return _CONV_CONFIGS
    config_env = os.getenv("FBAMTRAIN_CUTLASS_AUTOTUNE_CONFIG")
    if not config_env:
        raise RuntimeError(
            "FBAMTRAIN_CUTLASS_AUTOTUNE_CONFIG must point to a tuned CUTLASS config JSON "
            "(e.g. autotune_constants_sm90.json)."
        )
    config_path = Path(config_env)
    if not config_path.exists():
        raise RuntimeError(f"CUTLASS autotune config file not found at {config_path}")
    payload = json.loads(config_path.read_text())
    if not isinstance(payload, dict) or "conv2d" not in payload or not isinstance(payload["conv2d"], dict):
        raise RuntimeError(f"Invalid CUTLASS conv2d config file at {config_path}")
    configs: dict[str, dict[str, object]] = {}
    for name, spec in payload["conv2d"].items():
        if not isinstance(spec, dict):
            continue
        configs[str(name)] = spec
    _CONV_CONFIGS = configs
    return configs


def _sorted_defines(defines: Optional[dict[str, str]]) -> tuple[tuple[str, str], ...]:
    if not defines:
        return tuple()
    return tuple(sorted((str(k), str(v)) for k, v in defines.items()))


def _defines_tuple_to_dict(defines: Optional[tuple[tuple[str, str], ...]]) -> Optional[dict[str, str]]:
    if not defines:
        return None
    return {k: v for k, v in defines}


def _require_define(defines: dict[str, str], key: str) -> str:
    raw = defines.get(key)
    if raw is None or raw == "":
        raise RuntimeError(f"CUTLASS define '{key}' is required but missing.")
    return raw


def _int_define(defines: dict[str, str], key: str) -> int:
    raw = _require_define(defines, key)
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:  # pragma: no cover - configuration error
        raise ValueError(f"CUTLASS define '{key}' must be an integer (got {raw!r})") from exc


def _parse_kernel_meta(defines: dict[str, str]) -> _KernelMeta:
    kernel_name = _require_define(defines, "CUTLASS_CONV_KERNEL_NAME")

    stride_h = _int_define(defines, "CUTLASS_CONV_EXPECT_STRIDE_H")
    stride_w = _int_define(defines, "CUTLASS_CONV_EXPECT_STRIDE_W")
    padding_h = _int_define(defines, "CUTLASS_CONV_EXPECT_PADDING_H")
    padding_w = _int_define(defines, "CUTLASS_CONV_EXPECT_PADDING_W")
    dilation_h = _int_define(defines, "CUTLASS_CONV_EXPECT_DILATION_H")
    dilation_w = _int_define(defines, "CUTLASS_CONV_EXPECT_DILATION_W")
    kernel_h = _int_define(defines, "CUTLASS_CONV_EXPECT_KERNEL_H")
    kernel_w = _int_define(defines, "CUTLASS_CONV_EXPECT_KERNEL_W")
    in_channels_raw = _int_define(defines, "CUTLASS_CONV_META_IN_CHANNELS")
    in_channels = None if in_channels_raw <= 0 else in_channels_raw
    groups = max(1, _int_define(defines, "CUTLASS_CONV_EXPECT_GROUPS"))

    block_pixels = _int_define(defines, "CUTLASS_CONV_THREADBLOCK_M")
    block_oc = _int_define(defines, "CUTLASS_CONV_THREADBLOCK_N")
    block_k = _int_define(defines, "CUTLASS_CONV_THREADBLOCK_K")

    warp_m = _int_define(defines, "CUTLASS_CONV_WARP_M")
    warp_n = _int_define(defines, "CUTLASS_CONV_WARP_N")
    warp_k = _int_define(defines, "CUTLASS_CONV_WARP_K")

    total_threadblock_volume = block_pixels * block_oc * block_k
    total_warp_volume = warp_m * warp_n * warp_k
    if total_warp_volume <= 0 or total_threadblock_volume % total_warp_volume != 0:
        raise RuntimeError("Invalid CUTLASS threadblock/warp shape configuration.")
    num_warps = total_threadblock_volume // total_warp_volume
    if num_warps <= 0:
        raise RuntimeError("Derived CUTLASS warp count must be positive.")

    split_k_candidates = (
        "CUTLASS_CONV_EXPECT_SPLIT_K_SLICES",
        "CUTLASS_CONV_EXPECT_SPLIT_K",
        "CUTLASS_CONV_SPLIT_K_SLICES",
    )
    split_k_slices = 1
    for key in split_k_candidates:
        if key in defines:
            split_k_slices = max(1, _int_define(defines, key))
            break

    return _KernelMeta(
        kernel_name=kernel_name,
        stride_h=stride_h,
        stride_w=stride_w,
        padding_h=padding_h,
        padding_w=padding_w,
        dilation_h=dilation_h,
        dilation_w=dilation_w,
        kernel_h=kernel_h,
        kernel_w=kernel_w,
        in_channels=in_channels,
        groups=groups,
        block_pixels=block_pixels,
        block_oc=block_oc,
        block_k=block_k,
        num_warps=num_warps,
        split_k_slices=split_k_slices,
        shared_mem_bytes=0,
    )


def _reset_state() -> None:
    if _STATE.libcuda is not None:
        for module, _ in _STATE.module_cache.values():
            if module:
                try:
                    _STATE.libcuda.cuModuleUnload(module)
                except Exception:  # pragma: no cover - cleanup failure
                    pass
    _STATE.module_cache.clear()
    _STATE.artifacts = None
    _STATE.kernel_name = None
    _STATE.meta = None
    _STATE.defines = None


def _ensure_initialized(defines: Optional[dict[str, str]] = None) -> None:
    desired_defines = _sorted_defines(defines)
    if _STATE.artifacts is not None and _STATE.defines == desired_defines:
        return

    _reset_state()
    ensure_workdir()

    build_defines = dict(defines) if defines else {}
    artifacts = build_cutlass_artifacts(verbose=False, defines=build_defines or None)
    meta = _parse_kernel_meta(build_defines)

    metadata = json.loads(artifacts.metadata.read_text())
    shared_mem_bytes = int(metadata.get("shared_mem_bytes", 0))
    if shared_mem_bytes <= 0:
        raise RuntimeError("CUTLASS metadata missing shared memory requirement.")
    num_warps = int(metadata.get("num_warps", meta.num_warps))
    if num_warps <= 0:
        num_warps = meta.num_warps
    meta = replace(meta, shared_mem_bytes=shared_mem_bytes, num_warps=num_warps)

    _STATE.artifacts = artifacts
    _STATE.kernel_name = meta.kernel_name.encode("utf-8")
    _STATE.meta = meta
    _STATE.defines = desired_defines


def _load_cuda_driver() -> ctypes.CDLL:
    if _STATE.libcuda is not None:
        return _STATE.libcuda

    libcuda = ctypes.CDLL("libcuda.so")
    libcuda.cuInit.argtypes = [ctypes.c_uint]
    libcuda.cuInit.restype = ctypes.c_int
    libcuda.cuModuleLoad.argtypes = [ctypes.POINTER(_CUmodule), ctypes.c_char_p]
    libcuda.cuModuleLoad.restype = ctypes.c_int
    libcuda.cuModuleUnload.argtypes = [_CUmodule]
    libcuda.cuModuleUnload.restype = ctypes.c_int
    libcuda.cuModuleGetFunction.argtypes = [ctypes.POINTER(_CUfunction), _CUmodule, ctypes.c_char_p]
    libcuda.cuModuleGetFunction.restype = ctypes.c_int
    libcuda.cuFuncSetAttribute.argtypes = [_CUfunction, ctypes.c_int, ctypes.c_int]
    libcuda.cuFuncSetAttribute.restype = ctypes.c_int
    libcuda.cuLaunchKernel.argtypes = [
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
    libcuda.cuLaunchKernel.restype = ctypes.c_int
    libcuda.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
    libcuda.cuGetErrorString.restype = ctypes.c_int

    status = libcuda.cuInit(0)
    if status != 0:  # pragma: no cover - driver init failure
        raise RuntimeError(f"cuInit failed with error code {status}")

    _STATE.libcuda = libcuda
    return libcuda


def _check_cuda(status: int) -> None:
    if status == 0:
        return
    libcuda = _load_cuda_driver()
    msg_ptr = ctypes.c_char_p()
    if libcuda.cuGetErrorString(status, ctypes.byref(msg_ptr)) == 0 and msg_ptr.value:
        raise RuntimeError(f"CUDA driver error {status}: {msg_ptr.value.decode('utf-8')}")
    raise RuntimeError(f"CUDA driver error {status}")


def _ceil_div(a: int, b: int) -> int:
    if b <= 0:
        raise ValueError("Division by non-positive value when computing launch configuration.")
    return (a + b - 1) // b


def _compute_launch_shape(
    *,
    batch: int,
    out_h: int,
    out_w: int,
    out_channels: int,
    meta: _KernelMeta,
) -> tuple[int, int, int, int, int, int]:
    total_pixels = batch * out_h * out_w
    grid_x = max(1, _ceil_div(total_pixels, meta.block_pixels))
    grid_y = max(1, _ceil_div(out_channels, meta.block_oc))
    grid_z = max(1, meta.split_k_slices)
    block_x = meta.block_threads
    block_y = 1
    block_z = 1
    return grid_x, grid_y, grid_z, block_x, block_y, block_z


def _get_kernel(device_index: int) -> tuple[_CUmodule, _CUfunction]:
    if device_index in _STATE.module_cache:
        return _STATE.module_cache[device_index]

    current_defines = _defines_tuple_to_dict(_STATE.defines)
    _ensure_initialized(current_defines)

    if _STATE.artifacts is None or _STATE.kernel_name is None:
        raise RuntimeError("CUTLASS runtime is not initialised.")

    libcuda = _load_cuda_driver()

    module = _CUmodule()
    status = libcuda.cuModuleLoad(ctypes.byref(module), str(_STATE.artifacts.cubin).encode("utf-8"))
    _check_cuda(status)

    function = _CUfunction()
    status = libcuda.cuModuleGetFunction(ctypes.byref(function), module, _STATE.kernel_name)
    try:
        _check_cuda(status)
    except Exception:
        libcuda.cuModuleUnload(module)
        raise

    meta = _STATE.meta
    if meta is None or meta.shared_mem_bytes <= 0:
        raise RuntimeError("CUTLASS kernel shared memory metadata unavailable.")

    _check_cuda(
        libcuda.cuFuncSetAttribute(
            function,
            _CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            int(meta.shared_mem_bytes),
        )
    )

    _STATE.module_cache[device_index] = (module, function)
    return module, function


def load_cutlass_config(config_name: str) -> dict[str, object]:
    configs = _load_conv2d_configs()
    if config_name not in configs:
        available = ", ".join(sorted(configs.keys()))
        raise ValueError(f"Unknown CUTLASS conv2d configuration '{config_name}'. Available: {available}")

    spec = configs[config_name]
    if not spec.get("enabled", True):
        raise ValueError(f"CUTLASS configuration '{config_name}' is marked as disabled.")

    defines = dict(spec.get("defines", {}))
    _ensure_initialized(defines)
    return spec


def run_cutlass_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
) -> torch.Tensor:
    if not x.is_cuda or not weight.is_cuda:
        raise RuntimeError("Cutlass Conv2d expects CUDA tensors")
    if x.dtype not in (torch.bfloat16, torch.float16) or weight.dtype != x.dtype:
        raise RuntimeError("Cutlass Conv2d expects BF16 or FP16 tensors with matching weight dtype")
    if bias is not None and bias.dtype != x.dtype:
        raise RuntimeError("Cutlass Conv2d bias must match input dtype when provided")
    if x.ndim != 4 or weight.ndim != 4:
        raise RuntimeError("Cutlass Conv2d expects NCHW input and OIHW weights")

    defines_dict = _defines_tuple_to_dict(_STATE.defines)
    _ensure_initialized(defines_dict)
    meta = _STATE.meta
    if meta is None:
        raise RuntimeError("CUTLASS runtime metadata unavailable.")

    expected_dtype = torch.float16 if defines_dict and defines_dict.get("CUTLASS_CONV_FP16") == "1" else torch.bfloat16
    if x.dtype != expected_dtype or weight.dtype != expected_dtype:
        raise RuntimeError(f"Loaded CUTLASS kernel expects tensors of dtype {expected_dtype}")
    if bias is not None and bias.dtype != expected_dtype:
        raise RuntimeError(f"Loaded CUTLASS kernel expects bias of dtype {expected_dtype}")

    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dilation_h, dilation_w = dilation

    if stride_h != meta.stride_h or stride_w != meta.stride_w:
        raise ValueError(f"Generated Cutlass kernel expects stride=({meta.stride_h}, {meta.stride_w})")
    if dilation_h != meta.dilation_h or dilation_w != meta.dilation_w:
        raise ValueError(f"Generated Cutlass kernel expects dilation=({meta.dilation_h}, {meta.dilation_w})")
    if pad_h != meta.padding_h or pad_w != meta.padding_w:
        raise ValueError(f"Generated Cutlass kernel expects padding=({meta.padding_h}, {meta.padding_w})")

    batch, in_channels, in_h, in_w = x.shape
    out_channels, kernel_in_channels, kernel_h, kernel_w = weight.shape

    if kernel_h != meta.kernel_h or kernel_w != meta.kernel_w:
        raise ValueError(f"Generated Cutlass kernel expects kernel=({meta.kernel_h}, {meta.kernel_w})")
    if in_channels != kernel_in_channels:
        raise ValueError("Weight in_channels must match input in_channels")
    if meta.in_channels is not None and in_channels != meta.in_channels:
        raise ValueError(f"Generated Cutlass kernel expects input channels={meta.in_channels}")
    if bias is not None and bias.shape != (out_channels,):
        raise ValueError("Bias shape must equal (out_channels,)")
    if meta.groups not in (0, 1):
        raise ValueError("Current runtime only supports non-grouped CUTLASS conv kernels.")

    if not x.is_contiguous(memory_format=torch.channels_last):
        raise ValueError("Cutlass Conv2d expects input tensors in channels_last memory format")

    if not weight.is_contiguous(memory_format=torch.channels_last):
        raise ValueError("Cutlass Conv2d expects weight tensors in channels_last memory format")

    out_h = (in_h + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
    out_w = (in_w + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1

    output_cl = torch.empty(
        (batch, out_channels, out_h, out_w),
        device=x.device,
        dtype=x.dtype,
        memory_format=torch.channels_last,
    )

    cfg = _CutlassConv2dConfig(
        batch,
        in_h,
        in_w,
        in_channels,
        out_channels,
        kernel_h,
        kernel_w,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        max(1, meta.groups),
        max(1, meta.split_k_slices),
    )

    device_index = x.device.index if x.device.index is not None else torch.cuda.current_device()
    torch.cuda.set_device(device_index)
    torch.cuda.current_stream(x.device)
    module, function = _get_kernel(device_index)

    meta = _STATE.meta
    if meta is None:
        raise RuntimeError("CUTLASS runtime metadata unavailable after kernel load.")
    if meta.shared_mem_bytes <= 0:
        raise RuntimeError("CUTLASS kernel did not report shared memory requirements.")

    grid_x, grid_y, grid_z, block_x, block_y, block_z = _compute_launch_shape(
        batch=batch,
        out_h=out_h,
        out_w=out_w,
        out_channels=out_channels,
        meta=meta,
    )

    workspace_ptr = ctypes.c_void_p(0)
    if meta.split_k_slices > 1:
        workspace_elems = grid_x * grid_y
        if workspace_elems <= 0:
            raise RuntimeError("Invalid workspace size computed for CUTLASS conv2d kernel.")
        workspace_tensor = torch.zeros(workspace_elems, device=x.device, dtype=torch.int32)
        workspace_ptr = ctypes.c_void_p(workspace_tensor.data_ptr())

    libcuda = _load_cuda_driver()

    cfg_arg = ctypes.byref(cfg)
    input_arg = ctypes.c_void_p(x.data_ptr())
    weight_arg = ctypes.c_void_p(weight.data_ptr())
    output_arg = ctypes.c_void_p(output_cl.data_ptr())
    alpha_arg = ctypes.c_float(1.0)
    beta_arg = ctypes.c_float(0.0)

    kernel_args = [
        ctypes.cast(cfg_arg, ctypes.c_void_p),
        ctypes.cast(ctypes.byref(input_arg), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(weight_arg), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(output_arg), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(alpha_arg), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(beta_arg), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(workspace_ptr), ctypes.c_void_p),
    ]
    args_array = (ctypes.c_void_p * len(kernel_args))(*[arg.value for arg in kernel_args])

    stream = torch.cuda.current_stream(x.device)
    stream_ptr = _CUstream(stream.cuda_stream)

    status = libcuda.cuLaunchKernel(
        function,
        ctypes.c_uint(grid_x),
        ctypes.c_uint(grid_y),
        ctypes.c_uint(grid_z),
        ctypes.c_uint(block_x),
        ctypes.c_uint(block_y),
        ctypes.c_uint(block_z),
        ctypes.c_uint(meta.shared_mem_bytes),
        stream_ptr,
        args_array,
        None,
    )
    _check_cuda(status)

    if bias is not None:
        output_cl.add_(bias.view(1, -1, 1, 1))

    return output_cl


__all__ = ["load_cutlass_config", "run_cutlass_conv2d"]
