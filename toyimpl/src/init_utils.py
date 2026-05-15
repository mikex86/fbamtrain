import math

import torch
import triton
import triton.language as tl


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


@triton.jit
def fill_normal(
        out_ptr,  # *float32
        n_elements: tl.uint32,  # number of outputs
        mean: tl.float32,  # mean of the normal
        std: tl.float32,  # std  of the normal
        seed: tl.uint32,  # RNG seed
        BLOCK_SIZE: tl.constexpr,  # block size
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    nonce = 0x9E3779B9

    # Two independent uniform(0,1) streams; xor the seed for decorrelation
    u = tl.rand(seed, offs)
    v = tl.rand(seed ^ nonce.to(tl.uint32), offs)

    # Numerical safety: avoid log(0) and ignore out-of-bounds lanes
    u = tl.where(mask, tl.maximum(u, 1e-7), 0.5)
    v = tl.where(mask, v, 0.0)

    # Box–Muller transform -> N(0,1)
    r = tl.sqrt(-2.0 * tl.log(u))
    theta = 6.283185307179586 * v
    z = r * tl.cos(theta)

    # Affine transform to N(mean, std^2) and store
    out = mean + std * z
    tl.store(out_ptr + offs, out, mask=mask)


def fill_normal_(
        out_tensor: torch.Tensor,
        mean: float = 0.0,
        std: float = 1.0,
        seed: int = 0
):
    assert out_tensor.is_cuda, "out_tensor must live on CUDA device"
    n = out_tensor.numel()

    def grid(meta):
        return (triton.cdiv(n, meta["BLOCK_SIZE"]),)

    fill_normal[grid](
        out_tensor,
        n,
        float(mean),
        float(std),
        int(seed),
        BLOCK_SIZE=1024,
        num_warps=2,
        num_stages=2,
    )


def fill_uniform_(
        out_tensor: torch.Tensor,
        low: float,
        high: float,
        seed: int
):
    assert out_tensor.is_cuda, "out_tensor must live on CUDA device"
    n = out_tensor.numel()

    def grid(meta):
        return (triton.cdiv(n, meta["BLOCK_SIZE"]),)

    fill_uniform[grid](
        out_tensor,
        n,
        float(low),
        float(high),
        int(seed),
        BLOCK_SIZE=1024,
        num_warps=2,
        num_stages=2,
    )


def init_normal(tensor, mean: float, std: float, seed: int):
    fill_normal_(tensor, mean, std, seed)


def init_uniform(tensor, low: float, high: float, seed: int):
    fill_uniform_(tensor, low, high, seed)


def init_kaiming_uniform(tensor, fan_in: int, seed: int):
    bound = math.sqrt(1.0 / float(fan_in))
    init_uniform(tensor, -bound, bound, seed)


def init_lstm_streaming_weights(input_size: int, hidden_size: int, seed: int, device, dtype):
    """
    Initialize streaming-layout LSTM parameters and return (w_ih, w_hh, b_ih, b_hh).
    Weights are shaped (I, 4H) in streaming layout; biases are (4H).
    """
    w_ih = torch.empty((input_size, 4 * hidden_size), device=device, dtype=dtype)
    init_kaiming_uniform(w_ih, hidden_size, seed)
    seed += 1

    w_hh = torch.empty((hidden_size, 4 * hidden_size), device=device, dtype=dtype)
    init_kaiming_uniform(w_hh, hidden_size, seed)
    seed += 1

    b_ih = torch.empty((4 * hidden_size,), device=device, dtype=dtype)
    init_kaiming_uniform(b_ih, hidden_size, seed)
    seed += 1

    b_hh = torch.empty((4 * hidden_size,), device=device, dtype=dtype)
    init_kaiming_uniform(b_hh, hidden_size, seed)

    return w_ih, w_hh, b_ih, b_hh
