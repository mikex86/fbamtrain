import math
import os
from pathlib import Path

import torch
import triton
import triton.language as tl
from safetensors.torch import save_file

TORCH_DTYPE = torch.bfloat16
TL_DTYPE = tl.bfloat16


def set_active_dtypes(torch_dtype: torch.dtype) -> None:
    global TORCH_DTYPE, TL_DTYPE
    TORCH_DTYPE = torch_dtype
    TL_DTYPE = tl.bfloat16 if torch_dtype == torch.bfloat16 else tl.float16


DEVICE = torch.device("cuda")


def is_hip():
    return False


def is_cuda():
    return torch.cuda.is_available()


def is_blackwell():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 10


def is_hopper():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 9


@triton.jit
def _attn_fwd_inner(
        acc, l_i, m_i, q,
        k_ptr, v_ptr,
        off_z, off_h,
        stride_k_z, stride_k_h, stride_k_t,
        stride_v_z, stride_v_h, stride_v_t,
        dtype: tl.constexpr, start_m, qk_scale,
        BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
        STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,
        N_CTX: tl.constexpr, warp_specialize: tl.constexpr, IS_HOPPER: tl.constexpr
):
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    else:
        lo, hi = 0, N_CTX

    stride_k_z = stride_k_z.to(tl.int64)
    stride_k_h = stride_k_h.to(tl.int64)
    stride_k_t = stride_k_t.to(tl.int64)

    stride_v_z = stride_v_z.to(tl.int64)
    stride_v_h = stride_v_h.to(tl.int64)
    stride_v_t = stride_v_t.to(tl.int64)

    off_z = off_z.to(tl.int64)
    off_h = off_h.to(tl.int64)

    base_k = k_ptr + off_z * stride_k_z + off_h * stride_k_h
    base_v = v_ptr + off_z * stride_v_z + off_h * stride_v_h

    offs_d = tl.arange(0, HEAD_DIM)

    for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=warp_specialize):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n_curr = start_n + offs_n
        k_offsets = base_k + offs_n_curr[:, None] * stride_k_t + offs_d[None, :]
        mask_k = (offs_n_curr[:, None] < N_CTX)
        k = tl.load(k_offsets, mask=mask_k, other=0.0).T
        qk = tl.dot(q, k)
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]

        p = tl.math.exp2(qk)
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)

        if not IS_HOPPER and warp_specialize and BLOCK_M == 128 and HEAD_DIM == 128:
            BM: tl.constexpr = acc.shape[0]
            BN: tl.constexpr = acc.shape[1]
            acc0, acc1 = acc.reshape([BM, 2, BN // 2]).permute(0, 2, 1).split()
            acc0 = acc0 * alpha[:, None]
            acc1 = acc1 * alpha[:, None]
            acc = tl.join(acc0, acc1).permute(0, 2, 1).reshape([BM, BN])
        else:
            acc = acc * alpha[:, None]

        v_offsets = base_v + offs_n_curr[:, None] * stride_v_t + offs_d[None, :]
        mask_v = (offs_n_curr[:, None] < N_CTX)
        v = tl.load(v_offsets, mask=mask_v, other=0.0)
        p = p.to(dtype)
        acc = tl.dot(p, v, acc)

        l_i = l_i * alpha + l_ij
        m_i = m_ij
    return acc, l_i, m_i


NUM_STAGES_OPTIONS = [1] if is_hip() else [2, 3, 4]

configs = [
    triton.Config({'BLOCK_SIZE_X': 128, 'BLOCK_SIZE_Y': 64}, num_stages=2, num_warps=4)
]


def keep(conf):
    BLOCK_SIZE_X = conf.kwargs["BLOCK_SIZE_X"]
    BLOCK_SIZE_Y = conf.kwargs["BLOCK_SIZE_Y"]
    return not (is_cuda() and torch.cuda.get_device_capability()[0] == 9 and BLOCK_SIZE_X * BLOCK_SIZE_Y < 128 * 128
                and conf.num_warps == 8)


def prune_invalid_configs(configs, named_args, **kwargs):
    N_CTX = kwargs["N_CTX"]
    return [conf for conf in configs if conf.kwargs.get("BLOCK_SIZE_X", 0) <= N_CTX]


@triton.autotune(
    configs=list(filter(keep, configs)),
    key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT", "warp_specialize"],
    prune_configs_by={'early_config_prune': prune_invalid_configs}
)
@triton.jit
def mha_attn_fwd(
        sm_scale, M_ptr,
        Z, H,
        q_ptr, stride_q_z, stride_q_h, stride_q_t,
        k_ptr, stride_k_z, stride_k_h, stride_k_t,
        v_ptr, stride_v_z, stride_v_h, stride_v_t,
        o_ptr, stride_o_z, stride_o_h, stride_o_t,
        N_CTX,
        stride_m_z, stride_m_h, stride_m_t,
        HEAD_DIM: tl.constexpr,
        BLOCK_SIZE_X: tl.constexpr,
        BLOCK_SIZE_Y: tl.constexpr,
        FP8_OUTPUT: tl.constexpr,
        STAGE: tl.constexpr,
        warp_specialize: tl.constexpr,
        IS_HOPPER: tl.constexpr,
):
    dtype = tl.float8e5 if FP8_OUTPUT else TL_DTYPE
    tl.static_assert(BLOCK_SIZE_Y <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    offs_m = start_m * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    offs_n = tl.arange(0, BLOCK_SIZE_Y)
    offs_d = tl.arange(0, HEAD_DIM)

    q_base = q_ptr + off_z * stride_q_z + off_h * stride_q_h
    q_offsets = q_base + offs_m[:, None] * stride_q_t + offs_d[None, :]
    mask_q = (offs_m[:, None] < N_CTX)
    q = tl.load(q_offsets, mask=mask_q, other=0.0)

    m_i = tl.zeros([BLOCK_SIZE_X], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_SIZE_X], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_SIZE_X, HEAD_DIM], dtype=tl.float32)

    qk_scale = sm_scale * 1.44269504  # 1/log(2)

    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q,
            k_ptr, v_ptr,
            off_z, off_h,
            stride_k_z, stride_k_h, stride_k_t,
            stride_v_z, stride_v_h, stride_v_t,
            dtype, start_m, qk_scale,
            BLOCK_SIZE_X, HEAD_DIM, BLOCK_SIZE_Y,
            4 - STAGE, offs_m, offs_n, N_CTX,
            warp_specialize, IS_HOPPER
        )
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q,
            k_ptr, v_ptr,
            off_z, off_h,
            stride_k_z, stride_k_h, stride_k_t,
            stride_v_z, stride_v_h, stride_v_t,
            dtype, start_m, qk_scale,
            BLOCK_SIZE_X, HEAD_DIM, BLOCK_SIZE_Y,
            2, offs_m, offs_n, N_CTX,
            warp_specialize, IS_HOPPER
        )

    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]

    m_offsets = M_ptr + off_z * stride_m_z + off_h * stride_m_h + offs_m * stride_m_t
    tl.store(m_offsets, m_i)

    o_base = o_ptr + off_z * stride_o_z + off_h * stride_o_h
    o_offsets = o_base + offs_m[:, None] * stride_o_t + offs_d[None, :]
    tl.store(o_offsets, acc.to(dtype), mask=mask_q)


def flash_attn_fwd(q, k, v, causal=True, sm_scale=1.0, warp_specialize=False):
    HEAD_DIM = q.shape[-1]
    assert HEAD_DIM in {16, 32, 64, 128, 256}
    assert q.dtype == TORCH_DTYPE and k.dtype == TORCH_DTYPE and v.dtype == TORCH_DTYPE
    assert q.shape == k.shape == v.shape

    o = torch.empty_like(q)
    stage = 3 if causal else 1
    extra_kern_args = {}

    if is_hip():
        waves_per_eu = 3 if HEAD_DIM <= 64 else 2
        extra_kern_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}

    M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)

    stride_q_z, stride_q_h, stride_q_t, stride_q_d = q.stride()
    stride_k_z, stride_k_h, stride_k_t, _ = k.stride()
    stride_v_z, stride_v_h, stride_v_t, _ = v.stride()
    stride_o_z, stride_o_h, stride_o_t, _ = o.stride()

    stride_m_z, stride_m_h, stride_m_t = M.stride()

    def alloc_fn(size: int, align: int, _):
        return torch.empty(size, dtype=torch.int8, device=q.device)

    triton.set_allocator(alloc_fn)

    def grid(META):
        return (triton.cdiv(q.shape[2], META["BLOCK_SIZE_X"]), q.shape[0] * q.shape[1], 1)

    if is_blackwell() and warp_specialize:
        extra_kern_args["maxnreg"] = 168 if HEAD_DIM == 128 else 80

    mha_attn_fwd[grid](
        sm_scale, M,
        q.shape[0], q.shape[1],
        q, stride_q_z, stride_q_h, stride_q_t,
        k, stride_k_z, stride_k_h, stride_k_t,
        v, stride_v_z, stride_v_h, stride_v_t,
        o, stride_o_z, stride_o_h, stride_o_t,
        N_CTX=q.shape[2],
        stride_m_z=stride_m_z,
        stride_m_h=stride_m_h,
        stride_m_t=stride_m_t,
        HEAD_DIM=HEAD_DIM,
        FP8_OUTPUT=False,
        STAGE=stage,
        warp_specialize=warp_specialize,
        IS_HOPPER=is_hopper(),
        **extra_kern_args
    )
    return o


BATCH_SIZE = 2
SEQ_LEN = 128
NUM_HEADS = 8
HEAD_DIM = 128
N_EMBD = NUM_HEADS * HEAD_DIM
HIDDEN_DIM = 4 * N_EMBD
RMS_EPS = 1e-5
INPUT_LOW = -0.5
INPUT_HIGH = 0.5
MODEL_INIT_SEED = 1337
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


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + eps)
    return x * inv_rms * weight


def generate_reference(torch_dtype: torch.dtype, suffix: str) -> None:
    set_active_dtypes(torch_dtype)

    device = torch.device("cuda")
    dtype = torch_dtype

    w_qkv = torch.empty((N_EMBD, 3 * N_EMBD), dtype=dtype, device=device)
    b_qkv = torch.empty((3 * N_EMBD,), dtype=dtype, device=device)
    w_proj = torch.empty((N_EMBD, N_EMBD), dtype=dtype, device=device)
    b_proj = torch.empty((N_EMBD,), dtype=dtype, device=device)

    w_fc1 = torch.empty((N_EMBD, HIDDEN_DIM), dtype=dtype, device=device)
    b_fc1 = torch.empty((HIDDEN_DIM,), dtype=dtype, device=device)
    w_fc2 = torch.empty((HIDDEN_DIM, N_EMBD), dtype=dtype, device=device)
    b_fc2 = torch.empty((N_EMBD,), dtype=dtype, device=device)

    ln1_weight = torch.full((N_EMBD,), 1.0, dtype=dtype, device=device)
    ln2_weight = torch.full((N_EMBD,), 1.0, dtype=dtype, device=device)

    seed = MODEL_INIT_SEED

    kaiming_uniform_(w_qkv, N_EMBD, seed)
    seed += 1
    kaiming_uniform_(b_qkv, N_EMBD, seed)
    seed += 1
    kaiming_uniform_(w_proj, N_EMBD, seed)
    seed += 1
    kaiming_uniform_(b_proj, N_EMBD, seed)
    seed += 1

    kaiming_uniform_(w_fc1, N_EMBD, seed)
    seed += 1
    kaiming_uniform_(b_fc1, N_EMBD, seed)
    seed += 1
    kaiming_uniform_(w_fc2, HIDDEN_DIM, seed)
    seed += 1
    kaiming_uniform_(b_fc2, HIDDEN_DIM, seed)
    seed += 1

    x = torch.empty((BATCH_SIZE, SEQ_LEN, N_EMBD), dtype=dtype, device=device)
    fill_uniform_(x, INPUT_LOW, INPUT_HIGH, seed)

    x_fp32 = x.to(torch.float32)
    ln1_weight_fp32 = ln1_weight.to(torch.float32)
    ln2_weight_fp32 = ln2_weight.to(torch.float32)

    # Attention block
    norm1 = rms_norm(x_fp32, ln1_weight_fp32, RMS_EPS)

    norm1_flat = norm1.view(BATCH_SIZE * SEQ_LEN, N_EMBD)
    w_qkv_fp32 = w_qkv.to(torch.float32)
    b_qkv_fp32 = b_qkv.to(torch.float32)

    qkv = norm1_flat @ w_qkv_fp32 + b_qkv_fp32
    qkv = qkv.view(BATCH_SIZE, SEQ_LEN, 3, N_EMBD)
    q, k, v = qkv.unbind(dim=2)

    q = q.view(BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM).permute(0, 2, 1, 3).contiguous().to(TORCH_DTYPE)
    k = k.view(BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM).permute(0, 2, 1, 3).contiguous().to(TORCH_DTYPE)
    v = v.view(BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM).permute(0, 2, 1, 3).contiguous().to(TORCH_DTYPE)

    attn = flash_attn_fwd(q, k, v, causal=False, sm_scale=1.0 / math.sqrt(HEAD_DIM))

    attn_out = attn.permute(0, 2, 1, 3).contiguous().view(BATCH_SIZE * SEQ_LEN, N_EMBD).to(torch.float32)
    w_proj_fp32 = w_proj.to(torch.float32)
    b_proj_fp32 = b_proj.to(torch.float32)

    attn_proj = attn_out @ w_proj_fp32 + b_proj_fp32
    attn_proj = attn_proj.view(BATCH_SIZE, SEQ_LEN, N_EMBD)

    residual1 = x_fp32 + attn_proj

    norm2 = rms_norm(residual1, ln2_weight_fp32, RMS_EPS)
    norm2_flat = norm2.view(BATCH_SIZE * SEQ_LEN, N_EMBD)

    w_fc1_fp32 = w_fc1.to(torch.float32)
    b_fc1_fp32 = b_fc1.to(torch.float32)
    hidden = norm2_flat @ w_fc1_fp32 + b_fc1_fp32
    hidden = torch.nn.functional.gelu(hidden)

    w_fc2_fp32 = w_fc2.to(torch.float32)
    b_fc2_fp32 = b_fc2.to(torch.float32)
    mlp_out = hidden @ w_fc2_fp32 + b_fc2_fp32
    mlp_out = mlp_out.view(BATCH_SIZE, SEQ_LEN, N_EMBD)

    output = residual1 + mlp_out
    output = output.to(dtype)

    save_file({
        "input": x.to(dtype).cpu(),
        "output": output.cpu(),
        "norm1": norm1.to(dtype).cpu(),
        "residual_after_attn": residual1.to(dtype).cpu(),
        "attn_out": attn_proj.to(dtype).cpu(),
        "norm2": norm2.to(dtype).cpu(),
        "mlp_out": mlp_out.to(dtype).cpu(),
    }, Path(__file__).parent / f"reference_{suffix}.safetensors")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required")

    dtype_env = os.environ.get("FBAMTRAIN_REF_DTYPE", "bf16").lower()
    if dtype_env in {"bf16", "bfloat16"}:
        torch_dtype, suffix = torch.bfloat16, "bf16"
    elif dtype_env in {"fp16", "float16"}:
        torch_dtype, suffix = torch.float16, "fp16"
    else:
        raise RuntimeError(f"Unsupported FBAMTRAIN_REF_DTYPE value: {dtype_env}")

    generate_reference(torch_dtype, suffix)


if __name__ == "__main__":
    main()
