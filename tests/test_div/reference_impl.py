from pathlib import Path

import torch
import triton
import triton.language as tl
from safetensors.torch import save_file


@triton.jit
def fill_uniform(
        out_ptr,
        n_elements: tl.uint32,
        low: tl.float32,
        high: tl.float32,
        seed: tl.uint32,
        BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    rnd = tl.rand(seed, offsets.to(tl.uint32))
    vals = low + (high - low) * rnd
    tl.store(out_ptr + offsets, vals, mask=mask)


BLOCK_SIZE = 1024


def fill_uniform_(tensor, low=0.0, high=1.0, seed=0):
    n_elements = tensor.numel()
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    fill_uniform[grid](tensor, n_elements, low, high, seed, BLOCK_SIZE=BLOCK_SIZE)


ROWS = 2048
COLS = 512
NUMEL = ROWS * COLS


def generate_reference(dtype: torch.dtype, suffix: str):
    lhs = torch.zeros(NUMEL, dtype=dtype, device="cuda")
    rhs = torch.zeros(NUMEL, dtype=dtype, device="cuda")

    seed = 321
    fill_uniform_(lhs, -0.5, 0.5, seed=seed)
    seed += 1
    fill_uniform_(rhs, 0.5, 1.5, seed=seed)

    elementwise_output = lhs / rhs

    save_file(
        {
            "elementwise_output": elementwise_output.cpu(),
        },
        Path(__file__).parent / f"reference_{suffix}.safetensors",
    )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required")

    for dtype, suffix in ((torch.bfloat16, "bf16"), (torch.float16, "fp16")):
        generate_reference(dtype, suffix)


if __name__ == "__main__":
    main()
