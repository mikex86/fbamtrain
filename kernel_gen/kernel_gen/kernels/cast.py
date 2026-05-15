import argparse
import torch
import triton
import triton.language as tl
import triton.testing


def _device() -> torch.device:
    return triton.runtime.driver.active.get_active_torch_device()


_configs = [
    triton.Config({"BLOCK_SIZE": 64}, num_warps=1, num_stages=2),
    triton.Config({"BLOCK_SIZE": 128}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=2),
]


@triton.autotune(configs=_configs, key=["n_elements"])
@triton.jit
def cast_bf16_to_fp32_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    vals = tl.load(in_ptr + offsets, mask=mask, other=0).to(tl.float32)
    tl.store(out_ptr + offsets, vals, mask=mask)


@triton.autotune(configs=_configs, key=["n_elements"])
@triton.jit
def cast_fp32_to_bf16_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    vals = tl.load(in_ptr + offsets, mask=mask, other=0).to(tl.bfloat16)
    tl.store(out_ptr + offsets, vals, mask=mask)


@triton.autotune(configs=_configs, key=["n_elements"])
@triton.jit
def cast_fp16_to_fp32_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    vals = tl.load(in_ptr + offsets, mask=mask, other=0).to(tl.float32)
    tl.store(out_ptr + offsets, vals, mask=mask)


@triton.autotune(configs=_configs, key=["n_elements"])
@triton.jit
def cast_fp32_to_fp16_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    vals = tl.load(in_ptr + offsets, mask=mask, other=0).to(tl.float16)
    tl.store(out_ptr + offsets, vals, mask=mask)


@triton.autotune(configs=_configs, key=["n_elements"])
@triton.jit
def cast_bf16_to_fp16_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    vals = tl.load(in_ptr + offsets, mask=mask, other=0).to(tl.float16)
    tl.store(out_ptr + offsets, vals, mask=mask)


@triton.autotune(configs=_configs, key=["n_elements"])
@triton.jit
def cast_fp16_to_bf16_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    vals = tl.load(in_ptr + offsets, mask=mask, other=0).to(tl.bfloat16)
    tl.store(out_ptr + offsets, vals, mask=mask)


def _launch_cast(kernel, inp: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    n_elements = inp.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    kernel[grid](inp, out, n_elements=n_elements)
    return out


def cast_bf16_to_fp32(inp: torch.Tensor) -> torch.Tensor:
    if inp.dtype != torch.bfloat16:
        raise ValueError("Expected bf16 input for bf16->fp32 cast")
    if not inp.is_cuda:
        raise ValueError("Input tensor must be on CUDA device")
    out = torch.empty_like(inp, dtype=torch.float32)
    return _launch_cast(cast_bf16_to_fp32_kernel, inp, out)


def cast_fp32_to_bf16(inp: torch.Tensor) -> torch.Tensor:
    if inp.dtype != torch.float32:
        raise ValueError("Expected fp32 input for fp32->bf16 cast")
    if not inp.is_cuda:
        raise ValueError("Input tensor must be on CUDA device")
    out = torch.empty_like(inp, dtype=torch.bfloat16)
    return _launch_cast(cast_fp32_to_bf16_kernel, inp, out)


def cast_bf16_to_fp16(inp: torch.Tensor) -> torch.Tensor:
    if inp.dtype != torch.bfloat16:
        raise ValueError("Expected bf16 input for bf16->fp16 cast")
    if not inp.is_cuda:
        raise ValueError("Input tensor must be on CUDA device")
    out = torch.empty_like(inp, dtype=torch.float16)
    return _launch_cast(cast_bf16_to_fp16_kernel, inp, out)


def cast_fp16_to_bf16(inp: torch.Tensor) -> torch.Tensor:
    if inp.dtype != torch.float16:
        raise ValueError("Expected fp16 input for fp16->bf16 cast")
    if not inp.is_cuda:
        raise ValueError("Input tensor must be on CUDA device")
    out = torch.empty_like(inp, dtype=torch.bfloat16)
    return _launch_cast(cast_fp16_to_bf16_kernel, inp, out)


def _bench(direction: str, num_elements: int, iters: int = 100):
    device = _device()
    torch.manual_seed(0)

    if direction == "bf16_to_fp32":
        src = torch.randn(num_elements, device=device, dtype=torch.bfloat16)
        ref = src.to(torch.float32)
        out = cast_bf16_to_fp32(src)
    elif direction == "fp32_to_bf16":
        src = torch.randn(num_elements, device=device, dtype=torch.float32)
        ref = src.to(torch.bfloat16)
        out = cast_fp32_to_bf16(src)
    else:
        raise ValueError(f"Unknown direction: {direction}")

    triton.testing.assert_close(out, ref, atol=0.0, rtol=0.0)

    grid = lambda meta: (triton.cdiv(num_elements, meta["BLOCK_SIZE"]),)

    if direction == "bf16_to_fp32":
        def run_triton():
            cast_bf16_to_fp32_kernel[grid](src, out, n_elements=num_elements)
            return out
    else:
        def run_triton():
            cast_fp32_to_bf16_kernel[grid](src, out, n_elements=num_elements)
            return out

    bytes_moved = src.element_size() * num_elements + out.element_size() * num_elements

    triton_ms = triton.testing.do_bench(run_triton, warmup=25, rep=iters)
    gbs = bytes_moved / (triton_ms * 1e-3) / 1e9

    print(f"Cast {direction} N={num_elements}")
    print(f" Triton: {triton_ms:.4f} ms, {gbs:.2f} GB/s")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Triton cast kernels")
    parser.add_argument("direction", choices=["bf16_to_fp32", "fp32_to_bf16"], help="Cast direction")
    parser.add_argument("--num-elements", type=int, default=1 << 20, help="Number of elements")
    parser.add_argument("--iters", type=int, default=100, help="Benchmark iterations")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required to run the benchmark")

    _bench(args.direction, args.num_elements, args.iters)


if __name__ == "__main__":
    main()
