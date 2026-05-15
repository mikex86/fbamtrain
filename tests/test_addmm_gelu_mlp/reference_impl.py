import math
import time
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.language import erf

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


# Computes D = act(A @ B + C)
@triton.jit
def addmm_act(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr, d_ptr,

        # Matrix dimensions
        M, N, K,

        # Strides
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        stride_dm, stride_dn,

        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        ACTIVATION: tl.constexpr,
):
    """Kernel for computing D = activation(A @ B + C).
    A: (M, K), B: (K, N), C/D: (M, N)
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm >= 0)
    tl.assume(stride_cn >= 0)
    tl.assume(stride_dm > 0)
    tl.assume(stride_dn > 0)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    offs_am = offs_m % M
    offs_bn = offs_n % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    tile_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c_tile = tl.load(c_ptrs, mask=tile_mask, other=0.0).to(tl.float32)

    accumulator = c_tile
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if ACTIVATION == 1:  # ReLU
        accumulator = tl.maximum(accumulator, 0.0)
    elif ACTIVATION == 2:  # GELU
        accumulator = gelu(accumulator)

    d = accumulator.to(a_ptr.dtype.element_ty)  # restore original precision in accumulator
    d_ptrs = d_ptr + stride_dm * offs_m[:, None] + stride_dn * offs_n[None, :]
    tl.store(d_ptrs, d, mask=tile_mask)


@triton.jit
def gelu(x):
    """GELU activation function"""
    return 0.5 * x * (1.0 + erf(x / tl.sqrt(2.0)))


def addmm_gelu(a, b, bias, out):
    M, K = a.shape
    K, N = b.shape
    assert bias.shape == (N,)
    assert out.shape == (M, N)
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 6
    ACTIVATION = 2  # GELU

    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N) + GROUP_SIZE_M - 1) // GROUP_SIZE_M,
    addmm_act[grid](
        a, b, bias, out,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        0, bias.stride(0),
        out.stride(0), out.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        ACTIVATION=ACTIVATION,
        num_warps=4,
        num_stages=4,
    )


def torch_gelu(x):
    """GELU activation function"""
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


def kaiming_uniform_(tensor, input_features: int, seed=0):
    k = 1.0 / input_features
    bound = math.sqrt(k)
    fill_uniform_(tensor, -bound, bound, seed)


class Linear:

    def __init__(self, in_features, out_features, dtype: torch.dtype):
        self.in_features = in_features
        self.dtype = dtype
        self.weight = torch.zeros(in_features, out_features, dtype=dtype, device='cuda')
        self.bias = torch.zeros(out_features, dtype=dtype, device='cuda')

    def init_parameters(self, seed: torch.Tensor):
        kaiming_uniform_(self.weight, self.in_features, seed=seed.item())
        seed += 1
        kaiming_uniform_(self.bias, self.in_features, seed=seed.item())
        seed += 1

    def forward(self, x):
        out = torch.zeros(x.shape[0], self.weight.shape[1], dtype=self.dtype, device='cuda')
        addmm_gelu(x, self.weight, self.bias, out)
        return out


def generate_reference(dtype: torch.dtype, suffix: str):
    l1 = Linear(10, 5, dtype)
    l2 = Linear(5, 2, dtype)

    seed = torch.tensor(42, dtype=torch.int32)
    l1.init_parameters(seed)
    l2.init_parameters(seed)

    x = torch.zeros(32, 10, dtype=dtype, device='cuda')
    fill_uniform_(x, -0.5, 0.5, seed=seed.item())

    start_time = time.perf_counter_ns()
    y = l1.forward(x)
    y = l2.forward(y)
    end_time = time.perf_counter_ns()

    time_us = (end_time - start_time) / 1e3
    print(f'[{suffix}] Inference time: {time_us:.2f} µs')

    save_file({
        'output': y.cpu(),
    }, Path(__file__).parent / f'reference_{suffix}.safetensors')


if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required")

    for dtype, suffix in ((torch.bfloat16, 'bf16'), (torch.float16, 'fp16')):
        generate_reference(dtype, suffix)
