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
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    u = tl.rand(seed, offs.to(tl.uint32))
    vals = low + (high - low) * u
    tl.store(out_ptr + offs, vals, mask=mask)


BLOCK_SIZE = 1024


def fill_uniform_(tensor, low=0.0, high=1.0, seed=0):
    n_elements = tensor.numel()
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    fill_uniform[grid](tensor, n_elements, low, high, seed, BLOCK_SIZE=BLOCK_SIZE)


BATCH = 16
SEQ = 256
HIDDEN = 256
EPS = 1e-5


def generate_reference(dtype: torch.dtype, suffix: str):
    x = torch.zeros(BATCH, SEQ, HIDDEN, dtype=dtype, device="cuda")
    weight = torch.zeros(HIDDEN, dtype=dtype, device="cuda")
    bias = torch.zeros(HIDDEN, dtype=dtype, device="cuda")

    seed = 42
    fill_uniform_(x, -0.5, 0.5, seed=seed)
    seed += 1
    fill_uniform_(weight, -0.5, 0.5, seed=seed)
    seed += 1
    fill_uniform_(bias, -0.5, 0.5, seed=seed)

    y = torch.nn.functional.layer_norm(x, (HIDDEN,), weight, bias, eps=EPS)

    save_file({
        "output": y.cpu(),
    }, Path(__file__).parent / f"reference_{suffix}.safetensors")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required")

    for dtype, suffix in ((torch.bfloat16, "bf16"), (torch.float16, "fp16")):
        generate_reference(dtype, suffix)


if __name__ == "__main__":
    main()
