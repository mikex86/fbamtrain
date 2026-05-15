from pathlib import Path

import torch
import triton
import triton.language as tl
from safetensors.torch import save_file


@triton.jit
def fill_uniform(out_ptr, n_elements: tl.uint32, low: tl.float32, high: tl.float32, seed: tl.uint32,
                 BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    random = tl.rand(seed, offsets.to(tl.uint32))
    values = low + (high - low) * random
    tl.store(out_ptr + offsets, values, mask=mask)


BLOCK_SIZE = 1024


def fill_uniform_(tensor, low=0.0, high=1.0, seed=0):
    n_elements = tensor.numel()
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    fill_uniform[grid](tensor, n_elements, low, high, seed, BLOCK_SIZE=BLOCK_SIZE)


BATCH = 16
SEQ = 256
HIDDEN = 256
EPS = 1e-5
SEED = 2024


def generate_reference(dtype: torch.dtype, suffix: str):
    device = torch.device("cuda")
    x = torch.empty((BATCH, SEQ, HIDDEN), dtype=dtype, device=device)
    weight = torch.empty((HIDDEN,), dtype=dtype, device=device)

    fill_uniform_(x, -0.5, 0.5, seed=SEED)
    fill_uniform_(weight, 0.9, 1.1, seed=SEED + 1)

    x32 = x.to(torch.float32)
    weight32 = weight.to(torch.float32)
    rms = torch.sqrt((x32 * x32).mean(dim=-1, keepdim=True) + EPS)
    y = (x32 / rms) * weight32

    save_file({"output": y.to(dtype).cpu()}, Path(__file__).parent / f"reference_{suffix}.safetensors")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required to generate reference tensors")

    for dtype, suffix in ((torch.bfloat16, "bf16"), (torch.float16, "fp16")):
        generate_reference(dtype, suffix)


if __name__ == "__main__":
    main()
