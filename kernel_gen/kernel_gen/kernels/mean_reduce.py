import math
import os
from typing import Optional

import torch
import triton
import triton.language as tl
import triton.testing

_PRINTED_AUTOTUNE = False

_MEAN_CONTIG_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 64}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 256}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=2),
]


_MEAN_COLUMN_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 32}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_SIZE": 64}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=2),
]


@triton.autotune(configs=_MEAN_CONTIG_CONFIGS, key=["cols"])
@triton.jit
def mean_reduce_contiguous(
    out_ptr,
    in_ptr,
    outer,
    inner,
    cols,
    BLOCK_SIZE: tl.constexpr,
):
    pid_outer = tl.program_id(axis=0)
    pid_inner = tl.program_id(axis=1)

    if pid_outer >= outer or pid_inner >= inner:
        return

    base_offset = pid_outer * cols * inner + pid_inner
    offsets = tl.arange(0, BLOCK_SIZE)

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for col_start in tl.range(0, cols, BLOCK_SIZE):
        col_idx = col_start + offsets
        mask = col_idx < cols
        ptrs = base_offset + col_idx * inner
        vals = tl.load(in_ptr + ptrs, mask=mask, other=0.0)
        vals = vals.to(tl.float32)
        acc += tl.where(mask, vals, 0.0)

    total = tl.sum(acc, axis=0)
    mean = total / cols
    out_dtype = out_ptr.dtype.element_ty
    tl.store(out_ptr + pid_outer * inner + pid_inner, mean.to(out_dtype))


@triton.autotune(configs=_MEAN_COLUMN_CONFIGS, key=["inner"])
@triton.jit
def mean_reduce_column_tiled(
    out_ptr,
    in_ptr,
    outer,
    inner,
    cols,
    BLOCK_SIZE: tl.constexpr,
):
    pid_outer = tl.program_id(axis=0)
    pid_inner = tl.program_id(axis=1)

    if pid_outer >= outer:
        return

    inner_start = pid_inner * BLOCK_SIZE
    inner_offsets = inner_start + tl.arange(0, BLOCK_SIZE)
    inner_mask = inner_offsets < inner

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for col_start in tl.range(0, cols, 32):
        col_offsets = col_start + tl.arange(0, 32)
        col_mask = col_offsets < cols
        ptrs = pid_outer * cols * inner + col_offsets[:, None] * inner + inner_offsets[None, :]
        mask = col_mask[:, None] & inner_mask[None, :]
        vals = tl.load(in_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(vals, axis=0)

    mean = acc / cols
    out_dtype = out_ptr.dtype.element_ty
    tl.store(out_ptr + pid_outer * inner + inner_offsets, mean.to(out_dtype), mask=inner_mask)


def _canonical_dim(dim: int, ndim: int) -> int:
    if dim < 0:
        dim += ndim
    if dim < 0 or dim >= ndim:
        raise ValueError(f"dimension out of range (expected in [-{ndim}, {ndim - 1}], got {dim - ndim if dim < 0 else dim})")
    return dim


def _resolve_kernel(kernel, meta: Optional[dict]):
    if meta:
        return getattr(kernel, "fn", kernel)
    return kernel


def launch_mean(
    x: torch.Tensor,
    dim: int = -1,
    keepdim: bool = False,
    *,
    block_size: Optional[int] = None,
) -> torch.Tensor:
    if not x.is_contiguous():
        raise ValueError("launch_mean expects contiguous tensor")
    ndim = x.ndim
    dim = _canonical_dim(dim, ndim)

    cols = x.shape[dim]
    if cols == 0:
        raise ValueError("launch_mean input must have non-zero size along the reduced dimension")
    outer = math.prod(x.shape[:dim]) if dim > 0 else 1
    inner = math.prod(x.shape[dim + 1 :]) if dim + 1 < ndim else 1
    rows = outer * inner
    if rows == 0:
        raise ValueError("launch_mean input has zero-sized outer/inner dimensions")

    output_shape = list(x.shape)
    if keepdim:
        output_shape[dim] = 1
    else:
        output_shape.pop(dim)
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)

    out_flat = out.view(outer, inner)

    meta = {"BLOCK_SIZE": block_size} if block_size is not None else None
    use_column_tiled = inner != 1
    kernel = _resolve_kernel(mean_reduce_column_tiled if use_column_tiled else mean_reduce_contiguous, meta)

    if use_column_tiled:
        grid = lambda META: (outer, triton.cdiv(inner, META["BLOCK_SIZE"]))
    else:
        grid = lambda META: (outer, inner)
    launcher = kernel[grid](
        out_flat,
        x,
        outer,
        inner,
        cols,
        **(meta or {}),
    )

    global _PRINTED_AUTOTUNE
    if (
        meta is None
        and os.environ.get("TRITON_PRINT_AUTOTUNING") == "1"
        and not _PRINTED_AUTOTUNE
    ):
        best_config = getattr(launcher, "best_config", None)
        if best_config is None:
            best_config = getattr(kernel, "best_config", None)
        if best_config is not None:
            block = None
            meta_cfg = getattr(best_config, "meta", None)
            if isinstance(meta_cfg, dict):
                block = meta_cfg.get("BLOCK_SIZE")
            if block is None:
                kwargs = getattr(best_config, "kwargs", {})
                block = kwargs.get("BLOCK_SIZE")
            print(
                "Best autotune config: "
                f"BLOCK_SIZE={block}, "
                f"NUM_WARPS={best_config.num_warps}, "
                f"NUM_STAGES={best_config.num_stages}"
            )
            _PRINTED_AUTOTUNE = True

    return out


def _reference_mean(x: torch.Tensor, dim: int, keepdim: bool) -> torch.Tensor:
    return x.to(torch.float32).mean(dim=dim, keepdim=keepdim).to(x.dtype)


def benchmark_mean(
    shape=(128, 1024, 256),
    dim: int = -1,
    keepdim: bool = False,
    iters: int = 100,
    block_size: Optional[int] = None,
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required to run the benchmark")

    device = triton.runtime.driver.active.get_active_torch_device()
    dtype = torch.float16

    torch.manual_seed(0)
    x = torch.randn(shape, device=device, dtype=dtype)

    def run_triton():
        return launch_mean(x, dim=dim, keepdim=keepdim, block_size=block_size)

    def run_torch():
        return _reference_mean(x, dim=dim, keepdim=keepdim)

    triton.testing.assert_close(run_triton(), run_torch(), atol=2e-3, rtol=2e-3)

    triton_ms = triton.testing.do_bench(run_triton, warmup=25, rep=iters)
    torch_ms = triton.testing.do_bench(run_torch, warmup=25, rep=iters)

    ndim = len(shape)
    dim_index = _canonical_dim(dim, ndim)
    cols = shape[dim_index]
    outer = math.prod(shape[:dim_index]) if dim_index > 0 else 1
    inner = math.prod(shape[dim_index + 1 :]) if dim_index + 1 < ndim else 1
    rows = outer * inner
    bytes_read = rows * cols * torch.finfo(dtype).bits // 8
    bytes_written = rows * torch.finfo(dtype).bits // 8
    gbs_triton = (bytes_read + bytes_written) / (triton_ms * 1e-3) / 1e9
    gbs_torch = (bytes_read + bytes_written) / (torch_ms * 1e-3) / 1e9

    print(f"Mean FP16 shape={tuple(shape)}, dim={dim}, keepdim={keepdim}, block={block_size or 'auto'}")
    print(f" Triton: {triton_ms:.3f} ms  ({gbs_triton:.2f} GB/s)")
    print(f" PyTorch: {torch_ms:.3f} ms  ({gbs_torch:.2f} GB/s)")

    return {
        "triton_ms": triton_ms,
        "torch_ms": torch_ms,
        "triton_gbs": gbs_triton,
        "torch_gbs": gbs_torch,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark Triton Mean reduction against PyTorch")
    parser.add_argument("--shape", type=int, nargs="+", default=[128, 1024, 256], help="Input tensor shape")
    parser.add_argument("--dim", type=int, default=-1, help="Dimension to reduce (supports negative indices)")
    parser.add_argument("--keepdim", action="store_true", help="Keep reduced dimension")
    parser.add_argument("--iters", type=int, default=100, help="Benchmark repetitions")
    parser.add_argument("--block-size", type=int, default=None, help="Override BLOCK_SIZE meta parameter")

    args = parser.parse_args()

    benchmark_mean(
        shape=tuple(args.shape),
        dim=args.dim,
        keepdim=args.keepdim,
        iters=args.iters,
        block_size=args.block_size,
    )


if __name__ == "__main__":
    main()
