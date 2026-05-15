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


NUMEL = 2048 * 512
INPUT_LOW = -0.5
INPUT_HIGH = 0.5
RNG_SEED = 123


def generate_reference(dtype: torch.dtype, suffix: str):
    gelu_input = torch.empty(NUMEL, dtype=dtype, device="cuda")
    relu_input = torch.empty(NUMEL, dtype=dtype, device="cuda")
    gelu_inplace = torch.empty(NUMEL, dtype=dtype, device="cuda")
    relu_inplace = torch.empty(NUMEL, dtype=dtype, device="cuda")

    seed = RNG_SEED
    fill_uniform_(gelu_input, INPUT_LOW, INPUT_HIGH, seed=seed)
    seed += 1
    fill_uniform_(relu_input, INPUT_LOW, INPUT_HIGH, seed=seed)
    seed += 1
    fill_uniform_(gelu_inplace, INPUT_LOW, INPUT_HIGH, seed=seed)
    seed += 1
    fill_uniform_(relu_inplace, INPUT_LOW, INPUT_HIGH, seed=seed)

    gelu_output = torch.nn.functional.gelu(gelu_input, approximate="none")
    relu_output = torch.nn.functional.relu(relu_input)
    gelu_inplace_output = torch.nn.functional.gelu(gelu_inplace, approximate="none")
    relu_inplace_output = torch.nn.functional.relu(relu_inplace)

    save_file(
        {
            "gelu_output": gelu_output.cpu(),
            "relu_output": relu_output.cpu(),
            "gelu_inplace_output": gelu_inplace_output.cpu(),
            "relu_inplace_output": relu_inplace_output.cpu(),
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
