import math

import torch
import triton
import triton.language as tl
from safetensors.torch import save_file

BATCH_SIZE = 32
N_EMBD = 16
HIDDEN_DIM = 4 * N_EMBD
MODEL_INIT_SEED = 42
INPUT_LOW = -0.5
INPUT_HIGH = 0.5
BLOCK_SIZE = 1024


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


def fill_uniform_(tensor: torch.Tensor, low: float, high: float, seed: int):
    n_elements = tensor.numel()
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    fill_uniform[grid](tensor, n_elements, low, high, seed, BLOCK_SIZE=BLOCK_SIZE)


def kaiming_uniform_(tensor: torch.Tensor, fan_in: int, seed: int):
    bound = math.sqrt(1.0 / fan_in)
    fill_uniform_(tensor, -bound, bound, seed)


def main():
    device = torch.device("cuda")
    dtype = torch.bfloat16

    w1 = torch.empty((N_EMBD, HIDDEN_DIM), dtype=dtype, device=device)
    b1 = torch.empty((HIDDEN_DIM,), dtype=dtype, device=device)
    w2 = torch.empty((HIDDEN_DIM, N_EMBD), dtype=dtype, device=device)
    b2 = torch.empty((N_EMBD,), dtype=dtype, device=device)

    seed = MODEL_INIT_SEED
    kaiming_uniform_(w1, N_EMBD, seed)
    seed += 1
    kaiming_uniform_(b1, N_EMBD, seed)
    seed += 1
    kaiming_uniform_(w2, HIDDEN_DIM, seed)
    seed += 1
    kaiming_uniform_(b2, HIDDEN_DIM, seed)
    seed += 1

    x = torch.empty((BATCH_SIZE, N_EMBD), dtype=dtype, device=device)
    fill_uniform_(x, INPUT_LOW, INPUT_HIGH, seed)

    x_fp32 = x.to(torch.float32)
    w1_fp32 = w1.to(torch.float32)
    b1_fp32 = b1.to(torch.float32)
    hidden = torch.matmul(x_fp32, w1_fp32) + b1_fp32
    hidden = torch.nn.functional.gelu(hidden)

    w2_fp32 = w2.to(torch.float32)
    b2_fp32 = b2.to(torch.float32)
    out = torch.matmul(hidden, w2_fp32) + b2_fp32
    out = out.to(dtype)

    save_file({"output": out.cpu()}, "reference.safetensors")


if __name__ == "__main__":
    main()
