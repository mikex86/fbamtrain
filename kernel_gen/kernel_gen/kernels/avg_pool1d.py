import math

import torch
import triton
import triton.language as tl
import triton.testing


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 32, "MAX_KERNEL_SIZE": 32}, num_warps=1, num_stages=2),
        triton.Config({"BLOCK_SIZE": 64, "MAX_KERNEL_SIZE": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_SIZE": 64, "MAX_KERNEL_SIZE": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_SIZE": 128, "MAX_KERNEL_SIZE": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 256, "MAX_KERNEL_SIZE": 64}, num_warps=8, num_stages=2),
    ],
    key=["pool_out", "kernel_size"],
)
@triton.jit
def avg_pool1d_kernel(
    input_ptr,
    output_ptr,
    outer_stride_in0,
    outer_stride_in1,
    pool_stride_in,
    outer_stride_out0,
    outer_stride_out1,
    pool_stride_out,
    outer_size0,
    outer_size1,
    pool_in,
    pool_out,
    kernel_size,
    stride,
    inv_kernel_size,
    BLOCK_SIZE: tl.constexpr,
    MAX_KERNEL_SIZE: tl.constexpr,
):
    pid_pool = tl.program_id(axis=0)
    pid_outer = tl.program_id(axis=1)

    outer_total = outer_size0 * outer_size1
    if pid_outer >= outer_total:
        return

    offs_out = pid_pool * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_out = offs_out < pool_out

    tl.device_assert(kernel_size > 0)
    tl.device_assert(kernel_size <= MAX_KERNEL_SIZE)

    idx_outer0 = pid_outer // outer_size1
    idx_outer1 = pid_outer % outer_size1

    base_in = input_ptr + idx_outer0 * outer_stride_in0 + idx_outer1 * outer_stride_in1
    base_out = output_ptr + idx_outer0 * outer_stride_out0 + idx_outer1 * outer_stride_out1

    in_offsets = offs_out * stride

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for k in tl.static_range(MAX_KERNEL_SIZE):
        mask_k = k < kernel_size
        sample_idx = in_offsets + k
        mask = mask_out & mask_k & (sample_idx < pool_in)
        vals = tl.load(base_in + sample_idx * pool_stride_in, mask=mask, other=0.0)
        acc += vals.to(tl.float32)

    avg = acc * inv_kernel_size
    # Store in the output tensor's dtype (was incorrectly forced to BF16).
    out_dtype = output_ptr.dtype.element_ty
    tl.store(base_out + offs_out * pool_stride_out, avg.to(out_dtype), mask=mask_out)


def _avg_pool1d_triton(x: torch.Tensor, kernel_size: int, stride: int, pool_dim: int = -1):
    assert x.dtype == torch.bfloat16
    assert x.ndim == 3

    pool_dim = pool_dim if pool_dim >= 0 else x.ndim + pool_dim
    if not (0 <= pool_dim < x.ndim):
        raise ValueError("pool_dim must refer to a valid dimension of the input tensor")

    shape = list(x.shape)
    strides_in = list(x.stride())

    pool_in = shape[pool_dim]
    pool_out = (pool_in - kernel_size) // stride + 1
    if pool_out <= 0:
        raise ValueError("Invalid kernel/stride combination leading to empty output")

    shape_out = shape.copy()
    shape_out[pool_dim] = pool_out
    y = torch.empty(shape_out, dtype=x.dtype, device=x.device)
    strides_out = list(y.stride())

    outer_dims = [dim for dim in range(x.ndim) if dim != pool_dim]
    outer_size0, outer_size1 = shape[outer_dims[0]], shape[outer_dims[1]]
    outer_stride_in0, outer_stride_in1 = strides_in[outer_dims[0]], strides_in[outer_dims[1]]
    outer_stride_out0, outer_stride_out1 = strides_out[outer_dims[0]], strides_out[outer_dims[1]]

    pool_stride_in = strides_in[pool_dim]
    pool_stride_out = strides_out[pool_dim]

    inv_kernel = 1.0 / float(kernel_size)

    grid = lambda meta: (triton.cdiv(pool_out, meta["BLOCK_SIZE"]), outer_size0 * outer_size1, 1)

    avg_pool1d_kernel[grid](
        x,
        y,
        outer_stride_in0,
        outer_stride_in1,
        pool_stride_in,
        outer_stride_out0,
        outer_stride_out1,
        pool_stride_out,
        outer_size0,
        outer_size1,
        pool_in,
        pool_out,
        kernel_size,
        stride,
        inv_kernel,
    )
    return y


def bench_avg_pool1d(
    batch=16,
    dim1=32,
    dim2=1024,
    kernel_size=3,
    stride=2,
    pool_dim=2,
    iters=100,
):
    device = triton.runtime.driver.active.get_active_torch_device()
    dtype = torch.bfloat16

    torch.manual_seed(0)
    shape = [batch, dim1, dim2]
    x = torch.randn(tuple(shape), device=device, dtype=dtype)

    def run_triton() -> torch.Tensor:
        return _avg_pool1d_triton(x, kernel_size, stride, pool_dim=pool_dim)

    def run_torch() -> torch.Tensor:
        perm = list(range(x.ndim))
        perm[pool_dim], perm[-1] = perm[-1], perm[pool_dim]
        inv_perm = [perm.index(i) for i in range(x.ndim)]
        x_perm = x.permute(perm).to(torch.float32)
        y_perm = torch.nn.functional.avg_pool1d(x_perm, kernel_size, stride=stride)
        return y_perm.to(dtype).permute(inv_perm)

    triton.testing.assert_close(run_triton(), run_torch(), atol=2e-3, rtol=2e-3)

    triton_ms = triton.testing.do_bench(run_triton, warmup=25, rep=iters)
    torch_ms = triton.testing.do_bench(run_torch, warmup=25, rep=iters)

    out_shape = list(shape)
    pool_len_in = shape[pool_dim]
    pool_len_out = (pool_len_in - kernel_size) // stride + 1
    out_shape[pool_dim] = pool_len_out
    total_elems = batch * dim1 * dim2 * kernel_size
    bytes_moved = total_elems * torch.finfo(dtype).bits / 8

    print(
        f"AvgPool1d BF16 shape={shape}, pool_dim={pool_dim}, K={kernel_size}, S={stride}"
    )
    print(f" Triton: {triton_ms:.3f} ms")
    print(f" PyTorch: {torch_ms:.3f} ms")

    return {"triton_ms": triton_ms, "torch_ms": torch_ms}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark Triton AvgPool1d against PyTorch")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--dim1", type=int, default=32)
    parser.add_argument("--dim2", type=int, default=1024)
    parser.add_argument("--kernel", type=int, default=3)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--pool-dim", type=int, default=2, help="Dimension to pool over (0-based)")
    parser.add_argument("--iters", type=int, default=100)

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required to run the benchmark")

    bench_avg_pool1d(
        batch=args.batch,
        dim1=args.dim1,
        dim2=args.dim2,
        kernel_size=args.kernel,
        stride=args.stride,
        pool_dim=args.pool_dim,
        iters=args.iters,
    )


if __name__ == "__main__":
    main()
