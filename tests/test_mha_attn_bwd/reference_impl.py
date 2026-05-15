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


def kaiming_uniform_(tensor, input_features: int, seed: int):
    k = 1.0 / input_features
    bound = math.sqrt(k)
    fill_uniform_(tensor, -bound, bound, seed=seed)


class FullMhaAttention:

    def __init__(self, embed_dim: int, num_heads: int, dtype: torch.dtype):
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        device = 'cuda'
        self.dtype = dtype

        self.w_qkv = torch.zeros(embed_dim, embed_dim * 3, dtype=dtype, device=device, requires_grad=True)
        self.b_qkv = torch.zeros(embed_dim * 3, dtype=dtype, device=device, requires_grad=True)

        self.w_proj = torch.zeros(embed_dim, embed_dim, dtype=dtype, device=device, requires_grad=True)
        self.b_proj = torch.zeros(embed_dim, dtype=dtype, device=device, requires_grad=True)

    def init_parameters(self, seed: torch.Tensor):
        kaiming_uniform_(self.w_qkv, self.embed_dim, seed.item())
        seed += 1
        kaiming_uniform_(self.b_qkv, self.embed_dim, seed.item())
        seed += 1

        kaiming_uniform_(self.w_proj, self.embed_dim, seed.item())
        seed += 1
        kaiming_uniform_(self.b_proj, self.embed_dim, seed.item())
        seed += 1

    def forward(self, x: torch.Tensor):
        bsz, seqlen, channels = x.shape
        hs = self.head_dim
        nh = self.num_heads

        qkv = torch.matmul(x, self.w_qkv) + self.b_qkv
        q, k, v = qkv.split(channels, dim=-1)

        q = q.view(bsz, seqlen, nh, hs).transpose(1, 2)
        k = k.view(bsz, seqlen, nh, hs).transpose(1, 2)
        v = v.view(bsz, seqlen, nh, hs).transpose(1, 2)

        scale = 1.0 / math.sqrt(hs)
        attn = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
        )

        attn = attn.transpose(1, 2).contiguous().view(bsz * seqlen, channels)
        out = torch.matmul(attn, self.w_proj) + self.b_proj
        return out.view(bsz, seqlen, channels)


B = 2
DEFAULT_HEADS = 2
DEFAULT_T = 128
HS = 128


def parse_heads_list():
    value = os.environ.get("MHA_TEST_HEADS_LIST")
    if not value:
        return [DEFAULT_HEADS]
    heads = []
    for entry in value.split(","):
        entry = entry.strip()
        if not entry:
            continue
        heads.append(int(entry))
    return heads


def parse_seqlen_list():
    value = os.environ.get("MHA_TEST_SEQLEN_LIST")
    if not value:
        return [DEFAULT_T]
    seqlens = []
    for entry in value.split(","):
        entry = entry.strip()
        if not entry:
            continue
        seqlens.append(int(entry))
    return seqlens


def generate_reference(dtype: torch.dtype, suffix: str, heads: int, t: int):
    embed_dim = heads * HS
    attn = FullMhaAttention(embed_dim, heads, dtype)

    seed = torch.tensor(42, dtype=torch.int32)
    attn.init_parameters(seed)

    x = torch.zeros(B, t, embed_dim, dtype=dtype, device='cuda', requires_grad=True)
    fill_uniform_(x, low=-0.5, high=0.5, seed=seed.item())
    seed += 1
    upstream = torch.zeros(B, t, embed_dim, dtype=dtype, device='cuda')
    fill_uniform_(upstream, low=-0.5, high=0.5, seed=seed.item())

    y = attn.forward(x)
    torch.autograd.backward(y, upstream)

    file_suffix = suffix if heads == DEFAULT_HEADS else f"{suffix}_h{heads}"
    if t != DEFAULT_T:
        file_suffix = f"{file_suffix}_t{t}"
    save_file({
        'x': x.detach().cpu(),
        'w_qkv': attn.w_qkv.detach().cpu(),
        'b_qkv': attn.b_qkv.detach().cpu(),
        'w_proj': attn.w_proj.detach().cpu(),
        'b_proj': attn.b_proj.detach().cpu(),
        'upstream': upstream.detach().cpu(),
        'grad_x': x.grad.detach().cpu(),
        'grad_w_qkv': attn.w_qkv.grad.detach().cpu(),
        'grad_b_qkv': attn.b_qkv.grad.detach().cpu(),
        'grad_w_proj': attn.w_proj.grad.detach().cpu(),
        'grad_b_proj': attn.b_proj.grad.detach().cpu(),
    }, Path(__file__).parent / f'reference_{file_suffix}.safetensors')


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required")

    for t in parse_seqlen_list():
        for heads in parse_heads_list():
            for dtype, suffix in ((torch.bfloat16, 'bf16'), (torch.float16, 'fp16')):
                generate_reference(dtype, suffix, heads, t)


if __name__ == '__main__':
    main()
