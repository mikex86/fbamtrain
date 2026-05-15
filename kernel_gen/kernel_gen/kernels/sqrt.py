import torch
import triton
import triton.language as tl
import triton.testing

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
def sqrt_elementwise(
    out_ptr,
    in_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    vals = tl.load(in_ptr + offsets, mask=mask, other=0)
    vals_fp32 = vals.to(tl.float32)
    result = tl.sqrt(vals_fp32)
    out_dtype = out_ptr.dtype.element_ty
    tl.store(out_ptr + offsets, result.to(out_dtype), mask=mask)


def launch_sqrt_elementwise(
    x: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    block_size: int | None = None,
):
    """Utility to run the element-wise sqrt kernel in eager mode for debugging."""
    if out is None:
        out = torch.empty_like(x)
    assert x.shape == out.shape, "input/output must share shape"
    assert x.is_contiguous() and out.is_contiguous(), "tensors must be contiguous"
    if x.dtype != out.dtype:
        raise ValueError("launch_sqrt_elementwise expects matching dtypes")

    n_elements = x.numel()

    meta = {"BLOCK_SIZE": block_size} if block_size is not None else None
    kernel = _resolve_kernel(sqrt_elementwise, meta)

    grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
    kernel[grid](
        out_ptr=out,
        in_ptr=x,
        n_elements=n_elements,
        **(meta or {}),
    )
    return out


def benchmark_sqrt_elementwise(numel=2048 * 512, iters=100, block_size: int | None = None):
    device = torch.device("cuda")
    dtype = torch.float32

    torch.manual_seed(0)
    x = torch.rand((numel,), device=device, dtype=dtype) + 1e-3
    out_triton = torch.empty_like(x)

    def run_triton():
        launch_sqrt_elementwise(x, out_triton, block_size=block_size)
        return out_triton

    def run_torch():
        return torch.sqrt(x)

    triton_out = run_triton()
    torch_out = run_torch()
    triton.testing.assert_close(triton_out, torch_out, atol=1e-5, rtol=1e-5)

    triton_ms = triton.testing.do_bench(run_triton, warmup=25, rep=iters)
    torch_ms = triton.testing.do_bench(run_torch, warmup=25, rep=iters)

    elems = numel
    bytes_moved = elems * dtype.itemsize * 2  # read input + write output
    gbps_triton = bytes_moved / (triton_ms * 1e-3) / 1e9
    gbps_torch = bytes_moved / (torch_ms * 1e-3) / 1e9

    block_desc = block_size if block_size is not None else "auto"
    print(f"Sqrt Elementwise numel={numel}, block={block_desc}, dtype={dtype}")
    print(f" Triton: {triton_ms:.3f} ms  ({gbps_triton:.2f} GB/s)")
    print(f" Torch : {torch_ms:.3f} ms  ({gbps_torch:.2f} GB/s)")

    return {
        "triton_ms": triton_ms,
        "torch_ms": torch_ms,
        "triton_gbps": gbps_triton,
        "torch_gbps": gbps_torch,
    }


if __name__ == "__main__":
    benchmark_sqrt_elementwise()
