import torch
import triton
import triton.language as tl
import triton.testing

_TRAILING_BROADCAST_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=2),
]

_ELEMENTWISE_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=2),
]


def _resolve_kernel(kernel, overrides: dict | None):
    if overrides:
        kernel = getattr(kernel, "fn", kernel)
    return kernel


@triton.autotune(configs=_TRAILING_BROADCAST_CONFIGS, key=["rows", "cols"])
@triton.jit
def mul_trailing_broadcast(
    out_ptr,
    in_ptr,
    scale_ptr,
    rows,
    cols,
    stride_row,
    stride_col,
    stride_scale,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    col_block = tl.program_id(1)

    if row >= rows:
        return

    tile_cols = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = tile_cols < cols

    row_offset = row * stride_row
    offsets = row_offset + tile_cols * stride_col

    scale = tl.load(scale_ptr + tile_cols * stride_scale, mask=mask, other=0)
    vals = tl.load(in_ptr + offsets, mask=mask, other=0)
    result = vals * scale
    tl.store(out_ptr + offsets, result, mask=mask)


@triton.autotune(configs=_ELEMENTWISE_CONFIGS, key=["n_elements"])
@triton.jit
def mul_elementwise(
    out_ptr,
    lhs_ptr,
    rhs_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    lhs = tl.load(lhs_ptr + offsets, mask=mask, other=0)
    rhs = tl.load(rhs_ptr + offsets, mask=mask, other=0)
    result = lhs * rhs
    tl.store(out_ptr + offsets, result, mask=mask)


@triton.autotune(configs=_ELEMENTWISE_CONFIGS, key=["n_elements"])
@triton.jit
def mul_scalar(
    out_ptr,
    lhs_ptr,
    rhs_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    lhs = tl.load(lhs_ptr + offsets, mask=mask, other=0)
    rhs = tl.load(rhs_ptr)
    result = lhs * rhs
    tl.store(out_ptr + offsets, result, mask=mask)


def launch_mul_trailing_broadcast(
    x: torch.Tensor,
    scale: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    block_size: int | None = None,
):
    """Utility to run the trailing-dimension broadcast kernel in eager mode for debugging."""
    if out is None:
        out = torch.empty_like(x)
    assert x.ndim == 2, "expected 2D activation"
    assert scale.ndim == 1, "expected 1D scale"
    if x.dtype != scale.dtype or x.dtype != out.dtype:
        raise ValueError("launch_mul_trailing_broadcast expects matching dtypes")

    rows, cols = x.shape
    assert scale.numel() == cols, "scale must match trailing dimension"
    stride_row, stride_col = x.stride()
    stride_scale = scale.stride()[0]

    meta = {"BLOCK_SIZE": block_size} if block_size is not None else None
    kernel = _resolve_kernel(mul_trailing_broadcast, meta)

    grid = lambda META: (rows, triton.cdiv(cols, META["BLOCK_SIZE"]))
    kernel[grid](
        out_ptr=out,
        in_ptr=x,
        scale_ptr=scale,
        rows=rows,
        cols=cols,
        stride_row=stride_row,
        stride_col=stride_col,
        stride_scale=stride_scale,
        **(meta or {}),
    )
    return out


def launch_mul_elementwise(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    block_size: int | None = None,
):
    """Utility to run the element-wise mul kernel in eager mode for debugging."""
    if out is None:
        out = torch.empty_like(lhs)
    assert lhs.shape == rhs.shape == out.shape, "all tensors must share shape"
    assert lhs.is_contiguous() and rhs.is_contiguous() and out.is_contiguous(), "tensors must be contiguous"
    if lhs.dtype != rhs.dtype or lhs.dtype != out.dtype:
        raise ValueError("launch_mul_elementwise expects matching dtypes")

    n_elements = lhs.numel()

    meta = {"BLOCK_SIZE": block_size} if block_size is not None else None
    kernel = _resolve_kernel(mul_elementwise, meta)

    grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
    kernel[grid](
        out_ptr=out,
        lhs_ptr=lhs,
        rhs_ptr=rhs,
        n_elements=n_elements,
        **(meta or {}),
    )
    return out


def benchmark_mul_trailing_broadcast(rows=2048, cols=512, iters=100, block_size: int | None = None):
    device = torch.device("cuda")
    dtype = torch.float16

    torch.manual_seed(0)
    activation = torch.randn((rows, cols), device=device, dtype=dtype)
    scale = torch.randn((rows,), device=device, dtype=dtype)

    out_triton = torch.empty_like(activation)

    def run_triton():
        launch_mul_trailing_broadcast(activation, scale, out_triton, block_size=block_size)
        return out_triton

    def run_torch():
        return activation * scale.unsqueeze(1)

    triton_out = run_triton()
    torch_out = run_torch()
    triton.testing.assert_close(triton_out, torch_out, atol=1e-2, rtol=1e-2)

    triton_ms = triton.testing.do_bench(run_triton, warmup=25, rep=iters)
    torch_ms = triton.testing.do_bench(run_torch, warmup=25, rep=iters)

    elems = rows * cols
    bytes_moved = elems * dtype.itemsize * 3  # read activation + scale (broadcast) + write output (approx)
    gbps_triton = bytes_moved / (triton_ms * 1e-3) / 1e9
    gbps_torch = bytes_moved / (torch_ms * 1e-3) / 1e9

    block_desc = block_size if block_size is not None else "auto"
    print(f"Mul Trailing Broadcast rows={rows}, cols={cols}, block={block_desc}, dtype={dtype}")
    print(f" Triton: {triton_ms:.3f} ms  ({gbps_triton:.2f} GB/s)")
    print(f" Torch : {torch_ms:.3f} ms  ({gbps_torch:.2f} GB/s)")

    return {
        "triton_ms": triton_ms,
        "torch_ms": torch_ms,
        "triton_gbps": gbps_triton,
        "torch_gbps": gbps_torch,
    }


def benchmark_mul_elementwise(numel=2048 * 512, iters=100, block_size: int | None = None):
    device = torch.device("cuda")
    dtype = torch.float16

    torch.manual_seed(0)
    lhs = torch.randn((numel,), device=device, dtype=dtype)
    rhs = torch.randn((numel,), device=device, dtype=dtype)
    out_triton = torch.empty_like(lhs)

    def run_triton():
        launch_mul_elementwise(lhs, rhs, out_triton, block_size=block_size)
        return out_triton

    def run_torch():
        return lhs * rhs

    triton_out = run_triton()
    torch_out = run_torch()
    triton.testing.assert_close(triton_out, torch_out, atol=1e-2, rtol=1e-2)

    triton_ms = triton.testing.do_bench(run_triton, warmup=25, rep=iters)
    torch_ms = triton.testing.do_bench(run_torch, warmup=25, rep=iters)

    bytes_moved = numel * dtype.itemsize * 3  # read + read + write
    gbps_triton = bytes_moved / (triton_ms * 1e-3) / 1e9
    gbps_torch = bytes_moved / (torch_ms * 1e-3) / 1e9

    block_desc = block_size if block_size is not None else "auto"
    print(f"Mul Elementwise numel={numel}, block={block_desc}, dtype={dtype}")
    print(f" Triton: {triton_ms:.3f} ms  ({gbps_triton:.2f} GB/s)")
    print(f" Torch : {torch_ms:.3f} ms  ({gbps_torch:.2f} GB/s)")

    return {
        "triton_ms": triton_ms,
        "torch_ms": torch_ms,
        "triton_gbps": gbps_triton,
        "torch_gbps": gbps_torch,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark Triton mul kernels against PyTorch")
    parser.add_argument(
        "--mode",
        choices=["trailing-broadcast", "elementwise"],
        default="trailing-broadcast",
        help="Select which kernel to benchmark/tune",
    )
    parser.add_argument("--rows", type=int, default=2048, help="Number of rows in the activation tensor")
    parser.add_argument("--cols", type=int, default=512, help="Number of columns in the activation tensor")
    parser.add_argument("--numel", type=int, default=None, help="Number of elements for elementwise mul benchmarking")
    parser.add_argument("--iters", type=int, default=100, help="Number of benchmark repetitions")
    parser.add_argument("--block-size", type=int, default=None, help="Override kernel BLOCK_SIZE meta parameter")

    args = parser.parse_args()

    if args.mode == "trailing-broadcast":
        benchmark_mul_trailing_broadcast(
            rows=args.rows,
            cols=args.cols,
            iters=args.iters,
            block_size=args.block_size,
        )
    else:
        numel = args.numel if args.numel is not None else args.rows * args.cols
        benchmark_mul_elementwise(
            numel=numel,
            iters=args.iters,
            block_size=args.block_size,
        )


if __name__ == "__main__":
    main()
