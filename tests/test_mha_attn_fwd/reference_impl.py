import math
import os
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


DEFAULT_B = 32
DEFAULT_H = 8
DEFAULT_T = 4096
HS = 128


def parse_positive_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if not value:
        return default
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def reference_suffix(dtype_suffix: str, batch: int, heads: int, seq_len: int) -> str:
    suffix = dtype_suffix
    if batch != DEFAULT_B:
        suffix = f"{suffix}_b{batch}"
    if heads != DEFAULT_H:
        suffix = f"{suffix}_h{heads}"
    if seq_len != DEFAULT_T:
        suffix = f"{suffix}_t{seq_len}"
    return suffix


def generate_reference(dtype: torch.dtype, suffix: str, batch: int, heads: int, seq_len: int):
    q = torch.zeros(batch, seq_len, heads, HS, dtype=dtype, device='cuda')
    k = torch.zeros(batch, seq_len, heads, HS, dtype=dtype, device='cuda')
    v = torch.zeros(batch, seq_len, heads, HS, dtype=dtype, device='cuda')

    seed = 42
    fill_uniform_(q, low=-0.5, high=0.5, seed=seed)
    seed += 1
    fill_uniform_(k, low=-0.5, high=0.5, seed=seed)
    seed += 1
    fill_uniform_(v, low=-0.5, high=0.5, seed=seed)

    y = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        attn_mask=None, dropout_p=0.0, is_causal=False, scale=1 / math.sqrt(HS)
    )
    y = y.transpose(1, 2).contiguous()

    save_file({
        'output': y.cpu(),
    }, Path(__file__).parent / f'reference_{suffix}.safetensors')


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required")

    batch = parse_positive_int("MHA_TEST_BATCH", DEFAULT_B)
    heads = parse_positive_int("MHA_TEST_HEADS", DEFAULT_H)
    seq_len = parse_positive_int("MHA_TEST_SEQLEN", DEFAULT_T)

    for dtype, suffix in ((torch.bfloat16, 'bf16'), (torch.float16, 'fp16')):
        file_suffix = reference_suffix(suffix, batch, heads, seq_len)
        generate_reference(dtype, file_suffix, batch, heads, seq_len)


if __name__ == '__main__':
    main()
