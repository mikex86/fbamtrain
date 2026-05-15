import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.testing
from triton.language import erf  # pylint: disable=unused-import

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


@triton.autotune(configs=_ELEMENTWISE_CONFIGS, key=["n_elements"])
@triton.jit
def act_gelu_elementwise(
    out_ptr,
    in_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    n_elements_i64 = n_elements.to(tl.int64)
    mask = offsets < n_elements_i64

    values = tl.load(in_ptr + offsets, mask=mask, other=0).to(tl.float32)
    activated = gelu(values)
    out_dtype = out_ptr.dtype.element_ty
    tl.store(out_ptr + offsets, activated.to(out_dtype), mask=mask)


@triton.autotune(configs=_ELEMENTWISE_CONFIGS, key=["n_elements"])
@triton.jit
def act_gelu_bwd_elementwise(
    out_ptr,
    in_ptr,
    upstream_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    n_elements_i64 = n_elements.to(tl.int64)
    mask = offsets < n_elements_i64

    x = tl.load(in_ptr + offsets, mask=mask, other=0).to(tl.float32)
    upstream = tl.load(upstream_ptr + offsets, mask=mask, other=0).to(tl.float32)
    erf_term = erf(x / tl.sqrt(2.0))
    exp_term = tl.exp(-0.5 * x * x)
    grad = 0.5 * (1.0 + erf_term) + 0.5 * x * 0.7978845608028654 * exp_term
    out_dtype = out_ptr.dtype.element_ty
    tl.store(out_ptr + offsets, (upstream * grad).to(out_dtype), mask=mask)


@triton.autotune(configs=_ELEMENTWISE_CONFIGS, key=["n_elements"])
@triton.jit
def act_relu_elementwise(
    out_ptr,
    in_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    n_elements_i64 = n_elements.to(tl.int64)
    mask = offsets < n_elements_i64

    values = tl.load(in_ptr + offsets, mask=mask, other=0).to(tl.float32)
    activated = tl.maximum(values, 0.0)
    out_dtype = out_ptr.dtype.element_ty
    tl.store(out_ptr + offsets, activated.to(out_dtype), mask=mask)


@triton.jit
def gelu(x):
    """GELU activation function"""
    return 0.5 * x * (1.0 + erf(x / tl.sqrt(2.0)))


def launch_act_gelu(
    x: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    block_size: int | None = None,
):
    """Utility to run the GELU activation kernel in eager mode for debugging."""
    if out is None:
        out = torch.empty_like(x)

    assert x.is_cuda and out.is_cuda, "expected CUDA tensors"
    assert x.dtype == out.dtype, "unexpected tensor dtype"
    assert x.is_contiguous() and out.is_contiguous(), "expected contiguous tensors"
    assert x.numel() == out.numel() > 0, "expected matching non-empty tensors"

    n_elements = x.numel()

    meta = {"BLOCK_SIZE": block_size} if block_size is not None else None
    kernel = _resolve_kernel(act_gelu_elementwise, meta)

    grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
    kernel[grid](
        out_ptr=out,
        in_ptr=x,
        n_elements=n_elements,
        **(meta or {}),
    )
    return out


def launch_act_relu(
    x: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    block_size: int | None = None,
):
    """Utility to run the ReLU activation kernel in eager mode for debugging."""
    if out is None:
        out = torch.empty_like(x)

    assert x.is_cuda and out.is_cuda, "expected CUDA tensors"
    assert x.dtype == out.dtype, "unexpected tensor dtype"
    assert x.is_contiguous() and out.is_contiguous(), "expected contiguous tensors"
    assert x.numel() == out.numel() > 0, "expected matching non-empty tensors"

    n_elements = x.numel()

    meta = {"BLOCK_SIZE": block_size} if block_size is not None else None
    kernel = _resolve_kernel(act_relu_elementwise, meta)

    grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
    kernel[grid](
        out_ptr=out,
        in_ptr=x,
        n_elements=n_elements,
        **(meta or {}),
    )
    return out


def _benchmark(activation: str, numel=2048 * 512, iters=100, block_size: int | None = None):
    device = torch.device("cuda")
    dtype = torch.float16

    torch.manual_seed(0)
    x = torch.randn((numel,), device=device, dtype=dtype)
    out_triton = torch.empty_like(x)

    if activation == "gelu":
        run_triton = lambda: launch_act_gelu(x, out_triton, block_size=block_size)
        run_torch = lambda: F.gelu(x, approximate="none")
    elif activation == "relu":
        run_triton = lambda: launch_act_relu(x, out_triton, block_size=block_size)
        run_torch = lambda: F.relu(x)
    else:
        raise ValueError(f"Unsupported activation: {activation}")

    def run_triton_wrapper():
        run_triton()
        return out_triton

    triton_out = run_triton_wrapper()
    torch_out = run_torch()
    triton.testing.assert_close(triton_out, torch_out, atol=1e-2, rtol=1e-2)

    triton_ms = triton.testing.do_bench(run_triton_wrapper, warmup=25, rep=iters)
    torch_ms = triton.testing.do_bench(run_torch, warmup=25, rep=iters)

    bytes_moved = numel * dtype.itemsize * 2  # read + write
    gbps_triton = bytes_moved / (triton_ms * 1e-3) / 1e9
    gbps_torch = bytes_moved / (torch_ms * 1e-3) / 1e9

    block_desc = block_size if block_size is not None else "auto"
    print(f"ACT {activation.upper()} FP16 numel={numel}, block={block_desc}")
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

    parser = argparse.ArgumentParser(description="Benchmark Triton activation kernels against PyTorch")
    parser.add_argument(
        "--activation",
        choices=["gelu", "relu"],
        default="gelu",
        help="Activation to benchmark",
    )
    parser.add_argument("--numel", type=int, default=2048 * 512, help="Number of elements to activate")
    parser.add_argument("--iters", type=int, default=100, help="Number of benchmark repetitions")
    parser.add_argument("--block-size", type=int, default=None, help="Override kernel BLOCK_SIZE meta parameter")

    args = parser.parse_args()

    _benchmark(args.activation, numel=args.numel, iters=args.iters, block_size=args.block_size)


if __name__ == "__main__":
    main()
