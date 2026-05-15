from __future__ import annotations

import argparse
import ctypes
import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import torch
import torch.nn.functional as F

from ..cutlass_codegen.fetch_cutlass_sources import fetch_cutlass_sources
from .compile import (
    KernelConfig,
    _compile_cuda,
    _emit_bwd_source,
    _emit_fwd_source,
    _smem_bytes_bwd_convert,
    _smem_bytes_bwd_seqk,
    _smem_bytes_fwd,
)
from .ensure_flash_attention import ensure_flash_attention_sources


_ROOT_DIR = Path(__file__).resolve().parent


_BWD_IS_V_IN_REGS = True
_BWD_NO_DOUBLE_BUFFER = False


_DEFAULT_FWD_SPACE = {
    "block_m": [64, 128],
    "block_n": [32, 64, 128],
    "num_warps": [4, 8],
}
_DEFAULT_BWD_SPACE = {
    "block_m": [64, 128],
    "block_n": [32, 64],
    "num_warps": [4, 8],
}


def _normalize_sm(sm: str) -> str:
    sm = sm.lower().strip()
    if sm.startswith("sm_"):
        return f"sm{sm[3:]}"
    return sm


def _force_uneven_bwd(sm: str) -> bool:
    # FlashAttention bwd even-MN path can fail to compile on Ampere (sm80).
    return _normalize_sm(sm) == "sm80"


@dataclass(frozen=True)
class _Candidate:
    block_m: int
    block_n: int
    num_warps: int


@dataclass
class _Timing:
    time_ms: float
    max_abs_diff: float
    mean_abs_diff: float
    valid: bool
    skip_reason: Optional[str] = None


_CUmodule = ctypes.c_void_p
_CUfunction = ctypes.c_void_p
_CUstream = ctypes.c_void_p
_CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8


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


def _load_search_space(path: Optional[Path]) -> dict[str, dict[str, list[int]]]:
    if path is None:
        return {"fwd": dict(_DEFAULT_FWD_SPACE), "bwd": dict(_DEFAULT_BWD_SPACE)}
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise RuntimeError("Search space JSON must contain an object.")
    result: dict[str, dict[str, list[int]]] = {"fwd": dict(_DEFAULT_FWD_SPACE), "bwd": dict(_DEFAULT_BWD_SPACE)}
    for direction in ("fwd", "bwd"):
        section = data.get(direction, {})
        if not isinstance(section, dict):
            raise RuntimeError(f"Search space section '{direction}' must be a JSON object.")
        for key in ("block_m", "block_n", "num_warps"):
            values = section.get(key)
            if values is None:
                continue
            if not isinstance(values, list) or not all(isinstance(v, int) for v in values):
                raise RuntimeError(f"Search space '{direction}.{key}' must be a list of integers.")
            result[direction][key] = values
    return result


def _build_candidates(space: Mapping[str, list[int]]) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    for block_m in space["block_m"]:
        for block_n in space["block_n"]:
            for num_warps in space["num_warps"]:
                candidates.append(_Candidate(block_m=block_m, block_n=block_n, num_warps=num_warps))
    return candidates


def _candidate_fits(candidate: _Candidate, seq: int) -> bool:
    return candidate.block_m > 0 and candidate.block_n > 0 and candidate.num_warps > 0 and \
        candidate.block_m <= seq and candidate.block_n <= seq


def _max_smem_bytes() -> int:
    props = torch.cuda.get_device_properties(torch.device("cuda"))
    for attr in ("shared_memory_per_block_optin", "shared_memory_per_block"):
        if hasattr(props, attr):
            try:
                value = int(getattr(props, attr))
            except (TypeError, ValueError):
                value = 0
            if value > 0:
                return value

    # Fallback to CUDA driver query when torch does not expose shared memory limits.
    try:
        lib = ctypes.CDLL("libcuda.so")
        lib.cuInit.argtypes = [ctypes.c_uint]
        lib.cuInit.restype = ctypes.c_int
        lib.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        lib.cuDeviceGet.restype = ctypes.c_int
        lib.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
        lib.cuDeviceGetAttribute.restype = ctypes.c_int

        if lib.cuInit(0) == 0:
            device = ctypes.c_int()
            if lib.cuDeviceGet(ctypes.byref(device), 0) == 0:
                value = ctypes.c_int()
                # CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97
                if lib.cuDeviceGetAttribute(ctypes.byref(value), 97, device.value) == 0 and value.value > 0:
                    return int(value.value)
                # CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8
                if lib.cuDeviceGetAttribute(ctypes.byref(value), 8, device.value) == 0 and value.value > 0:
                    return int(value.value)
    except Exception:
        pass

    raise RuntimeError("Failed to determine maximum shared memory size.")


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float16:
        return 1.0, 1e-3
    return 1.5, 2e-3


def _make_fwd_config(candidate: _Candidate, dtype: str, head_dim: int, even_mn: bool) -> KernelConfig:
    return KernelConfig(
        direction="fwd",
        dtype=dtype,
        head_dim=head_dim,
        block_m=candidate.block_m,
        block_n=candidate.block_n,
        num_warps=candidate.num_warps,
        arg_count=10,
        shared_mem_bytes=_smem_bytes_fwd(head_dim, candidate.block_m, candidate.block_n),
        function_name="mha_attn_fwd",
        is_even_mn=even_mn,
        is_even_k=True,
        accumulate_in_fp16=False,
    )


def _make_bwd_config(candidate: _Candidate, dtype: str, head_dim: int, even_mn: bool) -> KernelConfig:
    return KernelConfig(
        direction="bwd",
        dtype=dtype,
        head_dim=head_dim,
        block_m=candidate.block_m,
        block_n=candidate.block_n,
        num_warps=candidate.num_warps,
        arg_count=15,
        shared_mem_bytes=_smem_bytes_bwd_seqk(
            head_dim,
            candidate.block_m,
            candidate.block_n,
            no_double_buffer=_BWD_NO_DOUBLE_BUFFER,
            is_v_in_regs=_BWD_IS_V_IN_REGS,
        ),
        function_name="mha_attn_bwd",
        is_even_mn=even_mn,
        is_even_k=True,
        accumulate_in_fp16=False,
    )


def _compile_kernel(src: str, name: str, sm: str, include_dirs: list[Path], workdir: Path) -> Path:
    src_path = workdir / f"{name}.cu"
    src_path.write_text(src)
    cubin_path = workdir / f"{name}_{sm}.cubin"
    _compile_cuda(src_path, cubin_path, sm, include_dirs)
    return cubin_path


def _prepare_fwd_args(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    m: torch.Tensor,
    softmax_scale: float,
    batch: int,
    heads: int,
    seq: int,
) -> tuple[list[ctypes._SimpleCData], ctypes.Array]:
    softmax_scale_c = ctypes.c_float(softmax_scale)
    scratch_ptr = ctypes.c_void_p(m.data_ptr())
    batch_arg = ctypes.c_uint32(batch)
    heads_arg = ctypes.c_uint32(heads)
    q_ptr = ctypes.c_void_p(q.data_ptr())
    k_ptr = ctypes.c_void_p(k.data_ptr())
    v_ptr = ctypes.c_void_p(v.data_ptr())
    o_ptr = ctypes.c_void_p(o.data_ptr())
    q_row_stride_arg = ctypes.c_uint32(q.stride(1))
    seq_arg = ctypes.c_uint32(seq)
    global_scratch_ptr = ctypes.c_void_p(0)

    arg_values = [
        softmax_scale_c,
        scratch_ptr,
        batch_arg,
        heads_arg,
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        q_row_stride_arg,
        seq_arg,
        global_scratch_ptr,
    ]
    kernel_args = [ctypes.cast(ctypes.byref(arg), ctypes.c_void_p) for arg in arg_values]
    args_array = (ctypes.c_void_p * len(kernel_args))(*[arg.value for arg in kernel_args])
    return arg_values, args_array


def _prepare_bwd_args(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    do: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    m: torch.Tensor,
    delta: torch.Tensor,
    softmax_scale: float,
    batch: int,
    heads: int,
    seq: int,
    dq_accum: torch.Tensor,
) -> tuple[list[ctypes._SimpleCData], ctypes.Array]:
    softmax_scale_c = ctypes.c_float(softmax_scale)
    batch_arg = ctypes.c_uint32(batch)
    heads_arg = ctypes.c_uint32(heads)
    seq_arg = ctypes.c_uint32(seq)

    q_ptr = ctypes.c_void_p(q.data_ptr())
    k_ptr = ctypes.c_void_p(k.data_ptr())
    v_ptr = ctypes.c_void_p(v.data_ptr())
    o_ptr = ctypes.c_void_p(o.data_ptr())
    do_ptr = ctypes.c_void_p(do.data_ptr())
    dq_ptr = ctypes.c_void_p(dq.data_ptr())
    dk_ptr = ctypes.c_void_p(dk.data_ptr())
    dv_ptr = ctypes.c_void_p(dv.data_ptr())
    m_ptr = ctypes.c_void_p(m.data_ptr())
    delta_ptr = ctypes.c_void_p(delta.data_ptr())
    dq_accum_ptr = ctypes.c_void_p(dq_accum.data_ptr())
    q_row_stride_arg = ctypes.c_uint32(q.stride(1))
    do_row_stride_arg = ctypes.c_uint32(do.stride(1))

    arg_values = [
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        softmax_scale_c,
        do_ptr,
        dq_ptr,
        dk_ptr,
        dv_ptr,
        m_ptr,
        delta_ptr,
        batch_arg,
        heads_arg,
        seq_arg,
        q_row_stride_arg,
        do_row_stride_arg,
        dq_accum_ptr,
    ]
    kernel_args = [ctypes.cast(ctypes.byref(arg), ctypes.c_void_p) for arg in arg_values]
    args_array = (ctypes.c_void_p * len(kernel_args))(*[arg.value for arg in kernel_args])
    return arg_values, args_array


def _math_sdp_only() -> torch.autograd.profiler.record_function:
    if hasattr(torch.backends.cuda, "sdp_kernel"):
        return torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
    return torch.autograd.profiler.record_function("sdp_math_fallback")


def _reference_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    with _math_sdp_only():
        return F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            dropout_p=0.0,
            is_causal=False,
        )


def _reference_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q_ref = q.transpose(1, 2).contiguous().detach().clone().requires_grad_(True)
    k_ref = k.transpose(1, 2).contiguous().detach().clone().requires_grad_(True)
    v_ref = v.transpose(1, 2).contiguous().detach().clone().requires_grad_(True)
    do_ref = do.transpose(1, 2).contiguous()
    with _math_sdp_only():
        out_ref = F.scaled_dot_product_attention(q_ref, k_ref, v_ref, dropout_p=0.0, is_causal=False)
    out_ref.backward(do_ref)
    return out_ref.detach(), q_ref.grad.detach(), k_ref.grad.detach(), v_ref.grad.detach()


def _compute_m_tensor(q: torch.Tensor, k: torch.Tensor, softmax_scale: float) -> torch.Tensor:
    q_fp32 = q.transpose(1, 2).float()
    k_fp32 = k.transpose(1, 2).float()
    scores = torch.matmul(q_fp32, k_fp32.transpose(-1, -2)) * softmax_scale
    m = torch.logsumexp(scores, dim=-1)
    return m.to(dtype=torch.float32).contiguous()


def _bench_kernel(iters: int, warmup: int, run_once: callable) -> float:
    for _ in range(warmup):
        run_once()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        run_once()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / max(iters, 1)


def _measure_output_diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual - expected).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    return float(max_abs), float(mean_abs)


def _benchmark_fwd_candidate(
    driver: _CudaDriver,
    candidate: _Candidate,
    dtype: str,
    head_dim: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out_ref: torch.Tensor,
    softmax_scale: float,
    iters: int,
    warmup: int,
    max_smem_bytes: int,
    include_dirs: list[Path],
    sm: str,
) -> _Timing:
    even_mn = (q.shape[1] % candidate.block_m) == 0 and (q.shape[1] % candidate.block_n) == 0
    config = _make_fwd_config(candidate, dtype, head_dim, even_mn)
    if config.shared_mem_bytes > max_smem_bytes:
        return _Timing(
            time_ms=0.0,
            max_abs_diff=0.0,
            mean_abs_diff=0.0,
            valid=False,
            skip_reason="shared_mem_exceeds_limit",
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)
        try:
            src = _emit_fwd_source(config)
            cubin_path = _compile_kernel(src, f"flash_fa_fwd_{dtype}", sm, include_dirs, workdir)
        except Exception:
            return _Timing(
                time_ms=0.0,
                max_abs_diff=0.0,
                mean_abs_diff=0.0,
                valid=False,
                skip_reason="compile_failed",
            )

        module = driver.load_module(cubin_path)
        try:
            function = driver.get_function(module, "mha_attn_fwd")
            if config.shared_mem_bytes > 0:
                driver.set_dynamic_smem(function, config.shared_mem_bytes)

            o = torch.empty((q.shape[0], q.shape[1], q.shape[2], head_dim), device=q.device, dtype=q.dtype)
            m = torch.empty((q.shape[0], q.shape[2], q.shape[1]), device=q.device, dtype=torch.float32)
            kernel_args, args_array = _prepare_fwd_args(
                q, k, v, o, m, softmax_scale, q.shape[0], q.shape[2], q.shape[1]
            )
            grid = (
                math.ceil(q.shape[1] / config.block_m),
                q.shape[0],
                q.shape[2],
            )
            block = (32 * config.num_warps, 1, 1)
            stream_ptr = torch.cuda.current_stream(q.device).cuda_stream

            def _run() -> None:
                _ = kernel_args
                driver.launch_kernel(function, grid, block, config.shared_mem_bytes, stream_ptr, args_array)

            time_ms = _bench_kernel(iters, warmup, _run)
            _run()
            torch.cuda.synchronize()
        except Exception:
            return _Timing(
                time_ms=0.0,
                max_abs_diff=0.0,
                mean_abs_diff=0.0,
                valid=False,
                skip_reason="launch_or_runtime_failed",
            )
        finally:
            try:
                driver.unload_module(module)
            except Exception:
                pass

    expected = out_ref.transpose(1, 2).contiguous()
    max_abs, mean_abs = _measure_output_diff(o, expected)
    max_tol, mean_tol = _tolerances(q.dtype)
    valid = math.isfinite(max_abs) and max_abs <= max_tol and mean_abs <= mean_tol
    return _Timing(
        time_ms=time_ms,
        max_abs_diff=max_abs,
        mean_abs_diff=mean_abs,
        valid=valid,
        skip_reason=None if valid else "validation_failed",
    )


def _benchmark_bwd_candidate(
    driver: _CudaDriver,
    candidate: _Candidate,
    dtype: str,
    head_dim: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o_ref: torch.Tensor,
    do_ref: torch.Tensor,
    dq_ref: torch.Tensor,
    dk_ref: torch.Tensor,
    dv_ref: torch.Tensor,
    m: torch.Tensor,
    softmax_scale: float,
    iters: int,
    warmup: int,
    max_smem_bytes: int,
    include_dirs: list[Path],
    sm: str,
) -> _Timing:
    even_mn = (q.shape[1] % candidate.block_m) == 0 and (q.shape[1] % candidate.block_n) == 0
    if _force_uneven_bwd(sm):
        even_mn = False
    config = _make_bwd_config(candidate, dtype, head_dim, even_mn)
    if config.shared_mem_bytes > max_smem_bytes:
        return _Timing(
            time_ms=0.0,
            max_abs_diff=0.0,
            mean_abs_diff=0.0,
            valid=False,
            skip_reason="shared_mem_exceeds_limit",
        )

    dot_config = KernelConfig(
        direction="bwd",
        dtype=dtype,
        head_dim=head_dim,
        block_m=candidate.block_m,
        block_n=candidate.block_n,
        num_warps=candidate.num_warps,
        arg_count=15,
        shared_mem_bytes=0,
        function_name="mha_attn_bwd_dot_do_o",
        is_even_mn=False,
        is_even_k=True,
        accumulate_in_fp16=False,
    )
    convert_config = KernelConfig(
        direction="bwd",
        dtype=dtype,
        head_dim=head_dim,
        block_m=candidate.block_m,
        block_n=candidate.block_n,
        num_warps=candidate.num_warps,
        arg_count=15,
        shared_mem_bytes=_smem_bytes_bwd_convert(head_dim, candidate.block_m),
        function_name="mha_attn_bwd_convert_dq",
        is_even_mn=False,
        is_even_k=True,
        accumulate_in_fp16=False,
    )
    if convert_config.shared_mem_bytes > max_smem_bytes:
        return _Timing(
            time_ms=0.0,
            max_abs_diff=0.0,
            mean_abs_diff=0.0,
            valid=False,
            skip_reason="shared_mem_exceeds_limit",
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)
        try:
            bwd_src = _emit_bwd_source(config, "main")
            dot_src = _emit_bwd_source(dot_config, "dot")
            convert_src = _emit_bwd_source(convert_config, "convert")
            bwd_cubin = _compile_kernel(bwd_src, f"flash_fa_bwd_{dtype}", sm, include_dirs, workdir)
            dot_cubin = _compile_kernel(dot_src, f"flash_fa_bwd_dot_{dtype}", sm, include_dirs, workdir)
            convert_cubin = _compile_kernel(convert_src, f"flash_fa_bwd_convert_{dtype}", sm, include_dirs, workdir)
        except Exception:
            return _Timing(
                time_ms=0.0,
                max_abs_diff=0.0,
                mean_abs_diff=0.0,
                valid=False,
                skip_reason="compile_failed",
            )

        bwd_module = driver.load_module(bwd_cubin)
        dot_module = driver.load_module(dot_cubin)
        convert_module = driver.load_module(convert_cubin)
        try:
            bwd_fn = driver.get_function(bwd_module, "mha_attn_bwd")
            dot_fn = driver.get_function(dot_module, "mha_attn_bwd_dot_do_o")
            convert_fn = driver.get_function(convert_module, "mha_attn_bwd_convert_dq")
            if config.shared_mem_bytes > 0:
                driver.set_dynamic_smem(bwd_fn, config.shared_mem_bytes)
            if convert_config.shared_mem_bytes > 0:
                driver.set_dynamic_smem(convert_fn, convert_config.shared_mem_bytes)

            o = o_ref.transpose(1, 2).contiguous()
            do = do_ref.contiguous()
            delta = torch.empty((q.shape[0], q.shape[2], q.shape[1]), device=q.device, dtype=torch.float32)
            dq = torch.empty_like(do)
            dk = torch.empty_like(do)
            dv = torch.empty_like(do)
            dq_accum = torch.empty_like(do, dtype=torch.float32)
            dot_kernel_args, dot_args = _prepare_bwd_args(
                q,
                k,
                v,
                o,
                do,
                dq,
                dk,
                dv,
                m,
                delta,
                softmax_scale,
                q.shape[0],
                q.shape[2],
                q.shape[1],
                dq_accum,
            )
            main_kernel_args, main_args = _prepare_bwd_args(
                q,
                k,
                v,
                o,
                do,
                dq,
                dk,
                dv,
                m,
                delta,
                softmax_scale,
                q.shape[0],
                q.shape[2],
                q.shape[1],
                dq_accum,
            )
            convert_kernel_args, convert_args = _prepare_bwd_args(
                q,
                k,
                v,
                o,
                do,
                dq,
                dk,
                dv,
                m,
                delta,
                softmax_scale,
                q.shape[0],
                q.shape[2],
                q.shape[1],
                dq_accum,
            )

            grid_main = (
                math.ceil(q.shape[1] / config.block_n),
                q.shape[0],
                q.shape[2],
            )
            grid_dot = (
                math.ceil(q.shape[1] / config.block_m),
                q.shape[0],
                q.shape[2],
            )
            block = (32 * config.num_warps, 1, 1)
            stream_ptr = torch.cuda.current_stream(q.device).cuda_stream

            def _run() -> None:
                _ = dot_kernel_args
                _ = main_kernel_args
                _ = convert_kernel_args
                driver.launch_kernel(dot_fn, grid_dot, block, dot_config.shared_mem_bytes, stream_ptr, dot_args)
                driver.launch_kernel(bwd_fn, grid_main, block, config.shared_mem_bytes, stream_ptr, main_args)
                driver.launch_kernel(
                    convert_fn, grid_dot, block, convert_config.shared_mem_bytes, stream_ptr, convert_args
                )

            time_ms = _bench_kernel(iters, warmup, _run)
            _run()
            torch.cuda.synchronize()
        except Exception:
            return _Timing(
                time_ms=0.0,
                max_abs_diff=0.0,
                mean_abs_diff=0.0,
                valid=False,
                skip_reason="launch_or_runtime_failed",
            )
        finally:
            for mod in (bwd_module, dot_module, convert_module):
                try:
                    driver.unload_module(mod)
                except Exception:
                    pass

    dq_actual = dq.transpose(1, 2).contiguous()
    dk_actual = dk.transpose(1, 2).contiguous()
    dv_actual = dv.transpose(1, 2).contiguous()
    max_abs_dq, mean_abs_dq = _measure_output_diff(dq_actual, dq_ref)
    max_abs_dk, mean_abs_dk = _measure_output_diff(dk_actual, dk_ref)
    max_abs_dv, mean_abs_dv = _measure_output_diff(dv_actual, dv_ref)
    max_abs = max(max_abs_dq, max_abs_dk, max_abs_dv)
    mean_abs = max(mean_abs_dq, mean_abs_dk, mean_abs_dv)
    max_tol, mean_tol = _tolerances(q.dtype)
    valid = math.isfinite(max_abs) and max_abs <= max_tol and mean_abs <= mean_tol
    return _Timing(
        time_ms=time_ms,
        max_abs_diff=max_abs,
        mean_abs_diff=mean_abs,
        valid=valid,
        skip_reason=None if valid else "validation_failed",
    )


def _pick_best(results: list[tuple[_Candidate, _Timing]]) -> tuple[_Candidate, _Timing]:
    valid = [(cand, timing) for cand, timing in results if timing.valid]
    if not valid:
        return results[0]
    return min(valid, key=lambda item: item[1].time_ms)


def _tune_direction(
    direction: str,
    dtype: str,
    candidates: Iterable[_Candidate],
    device: torch.device,
    head_dim: int,
    batch: int,
    heads: int,
    seq: int,
    iters: int,
    warmup: int,
    max_smem_bytes: int,
    include_dirs: list[Path],
    sm: str,
    driver: _CudaDriver,
) -> dict[str, Any]:
    torch_dtype = torch.float16 if dtype == "fp16" else torch.bfloat16
    q = torch.randn((batch, seq, heads, head_dim), device=device, dtype=torch_dtype)
    k = torch.randn((batch, seq, heads, head_dim), device=device, dtype=torch_dtype)
    v = torch.randn((batch, seq, heads, head_dim), device=device, dtype=torch_dtype)
    softmax_scale = 1.0 / math.sqrt(head_dim)

    if direction == "fwd":
        out_ref = _reference_forward(q, k, v)
        results: list[tuple[_Candidate, _Timing]] = []
        skipped = 0
        for candidate in candidates:
            if not _candidate_fits(candidate, seq):
                skipped += 1
                print(
                    f"[autotune][{sm}][{direction}][{dtype}] skipped candidate "
                    f"(block_m={candidate.block_m}, block_n={candidate.block_n}, num_warps={candidate.num_warps}): "
                    f"out_of_problem_bounds"
                )
                continue
            timing = _benchmark_fwd_candidate(
                driver,
                candidate,
                dtype,
                head_dim,
                q,
                k,
                v,
                out_ref,
                softmax_scale,
                iters,
                warmup,
                max_smem_bytes,
                include_dirs,
                sm,
            )
            results.append((candidate, timing))
            if not timing.valid:
                skipped += 1
                reason = timing.skip_reason or "invalid_candidate"
                print(
                    f"[autotune][{sm}][{direction}][{dtype}] skipped candidate "
                    f"(block_m={candidate.block_m}, block_n={candidate.block_n}, num_warps={candidate.num_warps}): "
                    f"{reason}"
                )
        if not results:
            raise RuntimeError("No valid forward candidates available for tuning.")
        print(f"[autotune][{sm}][{direction}][{dtype}] attempted={len(results)} skipped={skipped}")
        best_candidate, best_timing = _pick_best(results)
        entry = {
            "autotuned": best_timing.valid,
            "config": {
                "block_m": best_candidate.block_m,
                "block_n": best_candidate.block_n,
                "num_warps": best_candidate.num_warps,
            },
            "metrics": {
                "kernel_ms": best_timing.time_ms,
                "max_abs_diff": best_timing.max_abs_diff,
                "mean_abs_diff": best_timing.mean_abs_diff,
                "validation_passed": best_timing.valid,
            },
            "problem": {"batch": batch, "heads": heads, "seqlen": seq, "head_dim": head_dim},
        }
        return entry

    do = torch.randn_like(q)
    out_ref, dq_ref, dk_ref, dv_ref = _reference_backward(q, k, v, do)
    m = _compute_m_tensor(q, k, softmax_scale)
    results = []
    skipped = 0
    for candidate in candidates:
        if not _candidate_fits(candidate, seq):
            skipped += 1
            print(
                f"[autotune][{sm}][{direction}][{dtype}] skipped candidate "
                f"(block_m={candidate.block_m}, block_n={candidate.block_n}, num_warps={candidate.num_warps}): "
                f"out_of_problem_bounds"
            )
            continue
        timing = _benchmark_bwd_candidate(
            driver,
            candidate,
            dtype,
            head_dim,
            q,
            k,
            v,
            out_ref,
            do,
            dq_ref,
            dk_ref,
            dv_ref,
            m,
            softmax_scale,
            iters,
            warmup,
            max_smem_bytes,
            include_dirs,
            sm,
        )
        results.append((candidate, timing))
        if not timing.valid:
            skipped += 1
            reason = timing.skip_reason or "invalid_candidate"
            print(
                f"[autotune][{sm}][{direction}][{dtype}] skipped candidate "
                f"(block_m={candidate.block_m}, block_n={candidate.block_n}, num_warps={candidate.num_warps}): "
                f"{reason}"
            )
    if not results:
        raise RuntimeError("No valid backward candidates available for tuning.")
    print(f"[autotune][{sm}][{direction}][{dtype}] attempted={len(results)} skipped={skipped}")
    best_candidate, best_timing = _pick_best(results)
    entry = {
        "autotuned": best_timing.valid,
        "config": {
            "block_m": best_candidate.block_m,
            "block_n": best_candidate.block_n,
            "num_warps": best_candidate.num_warps,
        },
        "metrics": {
            "kernel_ms": best_timing.time_ms,
            "max_abs_diff": best_timing.max_abs_diff,
            "mean_abs_diff": best_timing.mean_abs_diff,
            "validation_passed": best_timing.valid,
        },
        "problem": {"batch": batch, "heads": heads, "seqlen": seq, "head_dim": head_dim},
    }
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-tune FlashAttention kernel configurations.")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSON for tuned constants.")
    parser.add_argument(
        "--sm",
        default=None,
        help="Target SM to tune (default: current device SM).",
    )
    parser.add_argument("--head-dim", type=int, default=128, help="Head dimension to tune (default: 128).")
    parser.add_argument("--batch", type=int, default=8, help="Batch size for tuning (default: 8).")
    parser.add_argument("--heads", type=int, default=8, help="Num heads for tuning (default: 8).")
    parser.add_argument("--seqlen", type=int, default=1024, help="Sequence length for tuning (default: 1024).")
    parser.add_argument("--iters", type=int, default=30, help="Timing iterations (default: 30).")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations (default: 5).")
    parser.add_argument("--search-space", type=Path, default=None, help="Optional search space JSON.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required to autotune FlashAttention kernels.")

    device = torch.device("cuda")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    if args.sm is None:
        major, minor = torch.cuda.get_device_capability(device)
        sm = f"sm_{major}{minor}"
    else:
        sm = args.sm

    flash_root = ensure_flash_attention_sources(_ROOT_DIR)
    cutlass_root = fetch_cutlass_sources(_ROOT_DIR)
    include_dirs = [
        flash_root / "csrc" / "flash_attn",
        flash_root / "csrc" / "flash_attn" / "src",
        cutlass_root / "include",
        _ROOT_DIR / "third_party" / "aten_stub",
    ]

    search_space = _load_search_space(args.search_space)
    fwd_candidates = _build_candidates(search_space["fwd"])
    bwd_candidates = _build_candidates(search_space["bwd"])
    max_smem_bytes = _max_smem_bytes()
    driver = _CudaDriver()

    payload = {"flash_attention": {sm: {"fwd": {}, "bwd": {}}}}
    for dtype in ("bf16", "fp16"):
        payload["flash_attention"][sm]["fwd"][dtype] = _tune_direction(
            "fwd",
            dtype,
            fwd_candidates,
            device,
            args.head_dim,
            args.batch,
            args.heads,
            args.seqlen,
            args.iters,
            args.warmup,
            max_smem_bytes,
            include_dirs,
            sm,
            driver,
        )
        payload["flash_attention"][sm]["bwd"][dtype] = _tune_direction(
            "bwd",
            dtype,
            bwd_candidates,
            device,
            args.head_dim,
            args.batch,
            args.heads,
            args.seqlen,
            args.iters,
            args.warmup,
            max_smem_bytes,
            include_dirs,
            sm,
            driver,
        )

    args.output.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    try:
        main()
        print("[autotune] STATUS: SUCCESS (no fatal error)")
    except Exception as exc:
        print(f"[autotune] FATAL: {exc}")
        print("[autotune] STATUS: FATAL_ERROR")
        raise
