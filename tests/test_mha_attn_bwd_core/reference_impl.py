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


B = 2
H = 2
DEFAULT_T = 128
HS = 128


def parse_seqlen_list():
    value = os.environ.get("MHA_TEST_SEQLEN_LIST")
    if not value:
        return [DEFAULT_T, 127]
    seqlens = []
    for entry in value.split(","):
        entry = entry.strip()
        if not entry:
            continue
        seqlens.append(int(entry))
    return seqlens


def generate_reference(dtype: torch.dtype, suffix: str, t: int):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required")

    scale = 1.0 / math.sqrt(HS)
    device = "cuda"

    q = torch.zeros((B, t, H, HS), dtype=dtype, device=device, requires_grad=True)
    k = torch.zeros_like(q, requires_grad=True)
    v = torch.zeros_like(q, requires_grad=True)
    seed_base = 42 + (0 if t == DEFAULT_T else 100)
    fill_uniform_(q, low=-0.5, high=0.5, seed=seed_base)
    fill_uniform_(k, low=-0.5, high=0.5, seed=seed_base + 1)
    fill_uniform_(v, low=-0.5, high=0.5, seed=seed_base + 2)

    upstream = torch.zeros_like(q)
    fill_uniform_(upstream, low=-0.5, high=0.5, seed=seed_base + 3)

    output = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
    )
    output = output.contiguous()
    output_bths = output.transpose(1, 2).contiguous()

    output.backward(upstream.transpose(1, 2))

    q_bhts = q.transpose(1, 2)
    k_bhts = k.transpose(1, 2)
    qk = torch.matmul(q_bhts.float(), k_bhts.float().transpose(-1, -2)) * scale
    scratch = (torch.logsumexp(qk, dim=-1) / math.log(2.0)).contiguous()
    grad_q_bths = q.grad.contiguous()
    grad_k_bths = k.grad.contiguous()
    grad_v_bths = v.grad.contiguous()

    file_suffix = suffix if t == DEFAULT_T else f"{suffix}_t{t}"
    save_file({
        "q": q.detach().cpu(),
        "k": k.detach().cpu(),
        "v": v.detach().cpu(),
        "output": output_bths.detach().cpu(),
        "scratch": scratch.detach().cpu(),
        "upstream": upstream.detach().cpu(),
        "grad_q": grad_q_bths.detach().cpu(),
        "grad_k": grad_k_bths.detach().cpu(),
        "grad_v": grad_v_bths.detach().cpu(),
    }, Path(__file__).parent / f"reference_{file_suffix}.safetensors")


def main():
    for t in parse_seqlen_list():
        generate_reference(torch.bfloat16, "bf16", t)
        generate_reference(torch.float16, "fp16", t)


if __name__ == "__main__":
    main()
