import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.math import erf
from safetensors.torch import save_file

BATCH_SIZE = 32
ROWS = 48
COLS = 128
SEQUENCE_LENGTH = ROWS * COLS
MAX_CODE_POINT = 128
NUM_CHANNELS_PER_CELL = 3
CELL_CODEPOINT_CHANNEL_IDX = 0
CELL_FG_COLOR_CHANNEL_IDX = 1
CELL_BG_COLOR_CHANNEL_IDX = 2

NUM_DOWNSAMPLE_BLOCKS = 2
NUM_LAYERS = 1
NUM_HEADS = 8
N_EMBD = 1024
HIDDEN_DIM = 4 * N_EMBD
RMS_EPS = 1e-5

DOWNSAMPLE_CONV_MODE = "dilated"
DOWNSAMPLE_CONV_DILATION = 2

MODEL_INIT_SEED = 1337
DEVICE = torch.device("cuda")


def compute_codepoint(index: torch.Tensor) -> torch.Tensor:
    return (index * 13 + 7) % MAX_CODE_POINT


def compute_color(index: torch.Tensor, r_mul: int, r_bias: int, g_mul: int, g_bias: int, b_mul: int,
                  b_bias: int) -> torch.Tensor:
    r = (index * r_mul + r_bias) % 256
    g = (index * g_mul + g_bias) % 256
    b = (index * b_mul + b_bias) % 256
    return (r << 16) | (g << 8) | b


def compute_cell_states(device: torch.device) -> torch.Tensor:
    n_cells = BATCH_SIZE * ROWS * COLS
    idx = torch.arange(n_cells, device=device, dtype=torch.int64)

    cp = compute_codepoint(idx).to(torch.int32)
    fg = compute_color(idx, 17, 5, 19, 11, 23, 13).to(torch.int32)
    bg = compute_color(idx, 29, 17, 31, 19, 37, 23).to(torch.int32)

    states = torch.stack([cp, fg, bg], dim=-1)
    return states.view(BATCH_SIZE, ROWS, COLS, NUM_CHANNELS_PER_CELL).contiguous()


def build_cell_embeds(
        input_cell_states: torch.Tensor,
        cp_embed: torch.Tensor,
        position_embed: torch.Tensor,
        fg_r_embed: torch.Tensor,
        fg_g_embed: torch.Tensor,
        fg_b_embed: torch.Tensor,
        bg_r_embed: torch.Tensor,
        bg_g_embed: torch.Tensor,
        bg_b_embed: torch.Tensor,
        out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    states = input_cell_states.view(-1, NUM_CHANNELS_PER_CELL).to(torch.int32)
    cp_raw = states[:, CELL_CODEPOINT_CHANNEL_IDX]
    fg_rgb = states[:, CELL_FG_COLOR_CHANNEL_IDX]
    bg_rgb = states[:, CELL_BG_COLOR_CHANNEL_IDX]

    vocab_size = cp_embed.size(0)
    cp_clamped = torch.minimum(cp_raw, torch.tensor(vocab_size - 1, dtype=torch.int32, device=states.device))
    cp_valid = (cp_raw >= 0) & (cp_raw < vocab_size)
    cp_vec = cp_embed[cp_clamped.to(torch.long)]
    cp_vec = torch.where(cp_valid.view(-1, 1), cp_vec, torch.zeros_like(cp_vec))

    def rgb_to_unit(rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rgb = rgb.to(torch.int32)
        r = ((rgb >> 16) & 0xFF).to(torch.float32)
        g = ((rgb >> 8) & 0xFF).to(torch.float32)
        b = (rgb & 0xFF).to(torch.float32)
        inv255 = 1.0 / 255.0
        return r * inv255, g * inv255, b * inv255

    fg_r, fg_g, fg_b = rgb_to_unit(fg_rgb)
    bg_r, bg_g, bg_b = rgb_to_unit(bg_rgb)

    acc = cp_vec.to(torch.float32)
    pos_dim = position_embed.size(0)
    pos_indices = torch.arange(states.size(0), device=states.device, dtype=torch.int64) % pos_dim
    pos_vec = position_embed.index_select(0, pos_indices).to(torch.float32)
    acc = acc + pos_vec
    acc = acc + fg_r.view(-1, 1) * fg_r_embed.view(1, -1)
    acc = acc + fg_g.view(-1, 1) * fg_g_embed.view(1, -1)
    acc = acc + fg_b.view(-1, 1) * fg_b_embed.view(1, -1)
    acc = acc + bg_r.view(-1, 1) * bg_r_embed.view(1, -1)
    acc = acc + bg_g.view(-1, 1) * bg_g_embed.view(1, -1)
    acc = acc + bg_b.view(-1, 1) * bg_b_embed.view(1, -1)

    return acc.to(dtype=out_dtype, copy=True, memory_format=torch.contiguous_format)


def is_hip():
    return torch.version.hip is not None


def is_cuda():
    return torch.cuda.is_available()


def supports_host_descriptor():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def is_blackwell():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 10


def is_hopper():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 9


@triton.jit
def _attn_fwd_inner(
        acc,
        l_i,
        m_i,
        q,
        k_ptr,
        v_ptr,
        off_z,
        off_h,
        stride_k_z,
        stride_k_h,
        stride_k_t,
        stride_v_z,
        stride_v_h,
        stride_v_t,
        dtype: tl.constexpr,
        start_m,
        qk_scale,
        BLOCK_M: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_N: tl.constexpr,
        STAGE: tl.constexpr,
        offs_m: tl.constexpr,
        offs_n: tl.constexpr,
        N_CTX: tl.constexpr,
        warp_specialize: tl.constexpr,
        IS_HOPPER: tl.constexpr,
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
        mask_k = offs_n_curr[:, None] < N_CTX
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
        mask_v = offs_n_curr[:, None] < N_CTX
        v = tl.load(v_offsets, mask=mask_v, other=0.0)
        p = p.to(dtype)
        acc = tl.dot(p, v, acc)

        l_i = l_i * alpha + l_ij
        m_i = m_ij
    return acc, l_i, m_i


NUM_STAGES_OPTIONS = [1] if is_hip() else [2, 3, 4]

configs = [
    triton.Config({"BLOCK_SIZE_X": 128, "BLOCK_SIZE_Y": 64}, num_stages=2, num_warps=4)
]


def keep(conf):
    block_x = conf.kwargs["BLOCK_SIZE_X"]
    block_y = conf.kwargs["BLOCK_SIZE_Y"]
    return not (
            is_cuda()
            and torch.cuda.get_device_capability()[0] == 9
            and block_x * block_y < 128 * 128
            and conf.num_warps == 8
    )


def prune_invalid_configs(configs, named_args, **kwargs):
    n_ctx = kwargs["N_CTX"]
    return [conf for conf in configs if conf.kwargs.get("BLOCK_SIZE_X", 0) <= n_ctx]


@triton.autotune(
    configs=list(filter(keep, configs)),
    key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT", "warp_specialize"],
    prune_configs_by={"early_config_prune": prune_invalid_configs},
)
@triton.jit
def mha_attn_fwd(
        sm_scale,
        M_ptr,
        Z,
        H,
        q_ptr,
        stride_q_z,
        stride_q_h,
        stride_q_t,
        k_ptr,
        stride_k_z,
        stride_k_h,
        stride_k_t,
        v_ptr,
        stride_v_z,
        stride_v_h,
        stride_v_t,
        o_ptr,
        stride_o_z,
        stride_o_h,
        stride_o_t,
        N_CTX,
        stride_m_z,
        stride_m_h,
        stride_m_t,
        HEAD_DIM: tl.constexpr,
        BLOCK_SIZE_X: tl.constexpr,
        BLOCK_SIZE_Y: tl.constexpr,
        FP8_OUTPUT: tl.constexpr,
        STAGE: tl.constexpr,
        warp_specialize: tl.constexpr,
        IS_HOPPER: tl.constexpr,
):
    dtype = tl.float8e5 if FP8_OUTPUT else tl.bfloat16
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
    mask_q = offs_m[:, None] < N_CTX
    q = tl.load(q_offsets, mask=mask_q, other=0.0)

    m_i = tl.zeros([BLOCK_SIZE_X], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_SIZE_X], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_SIZE_X, HEAD_DIM], dtype=tl.float32)

    qk_scale = sm_scale * 1.44269504

    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            k_ptr,
            v_ptr,
            off_z,
            off_h,
            stride_k_z,
            stride_k_h,
            stride_k_t,
            stride_v_z,
            stride_v_h,
            stride_v_t,
            dtype,
            start_m,
            qk_scale,
            BLOCK_SIZE_X,
            HEAD_DIM,
            BLOCK_SIZE_Y,
            4 - STAGE,
            offs_m,
            offs_n,
            N_CTX,
            warp_specialize,
            IS_HOPPER,
        )
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            k_ptr,
            v_ptr,
            off_z,
            off_h,
            stride_k_z,
            stride_k_h,
            stride_k_t,
            stride_v_z,
            stride_v_h,
            stride_v_t,
            dtype,
            start_m,
            qk_scale,
            BLOCK_SIZE_X,
            HEAD_DIM,
            BLOCK_SIZE_Y,
            2,
            offs_m,
            offs_n,
            N_CTX,
            warp_specialize,
            IS_HOPPER,
        )

    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]

    m_offsets = M_ptr + off_z * stride_m_z + off_h * stride_m_h + offs_m * stride_m_t
    tl.store(m_offsets, m_i)

    o_base = o_ptr + off_z * stride_o_z + off_h * stride_o_h
    o_offsets = o_base + offs_m[:, None] * stride_o_t + offs_d[None, :]
    tl.store(o_offsets, acc.to(dtype), mask=mask_q)


def flash_attn_fwd(q, k, v, causal=True, sm_scale=1.0, warp_specialize=False):
    head_dim = q.shape[-1]
    assert head_dim in {16, 32, 64, 128, 256}
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("flash_attn_fwd expects bf16 or fp16 inputs")
    assert q.shape == k.shape == v.shape

    kernel_dtype = torch.bfloat16
    cast_inputs = q.dtype != kernel_dtype
    if cast_inputs:
        q_kernel = q.to(kernel_dtype)
        k_kernel = k.to(kernel_dtype)
        v_kernel = v.to(kernel_dtype)
    else:
        q_kernel = q
        k_kernel = k
        v_kernel = v

    o_kernel = torch.empty_like(q_kernel)
    stage = 3 if causal else 1
    extra_kern_args = {}

    if is_hip():
        waves_per_eu = 3 if head_dim <= 64 else 2
        extra_kern_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}

    M = torch.empty((q_kernel.shape[0], q_kernel.shape[1], q_kernel.shape[2]), device=q_kernel.device,
                    dtype=torch.float32)

    stride_q_z, stride_q_h, stride_q_t, stride_q_d = q_kernel.stride()
    stride_k_z, stride_k_h, stride_k_t, _ = k_kernel.stride()
    stride_v_z, stride_v_h, stride_v_t, _ = v_kernel.stride()
    stride_o_z, stride_o_h, stride_o_t, _ = o_kernel.stride()

    stride_m_z, stride_m_h, stride_m_t = M.stride()

    def alloc_fn(size: int, align: int, _):
        return torch.empty(size, dtype=torch.int8, device=q_kernel.device)

    triton.set_allocator(alloc_fn)

    def grid(meta):
        return (triton.cdiv(q_kernel.shape[2], meta["BLOCK_SIZE_X"]), q_kernel.shape[0] * q_kernel.shape[1], 1)

    if is_blackwell() and warp_specialize:
        extra_kern_args["maxnreg"] = 168 if head_dim == 128 else 80

    mha_attn_fwd[grid](
        sm_scale,
        M,
        q_kernel.shape[0],
        q_kernel.shape[1],
        q_kernel,
        stride_q_z,
        stride_q_h,
        stride_q_t,
        k_kernel,
        stride_k_z,
        stride_k_h,
        stride_k_t,
        v_kernel,
        stride_v_z,
        stride_v_h,
        stride_v_t,
        o_kernel,
        stride_o_z,
        stride_o_h,
        stride_o_t,
        N_CTX=q_kernel.shape[2],
        stride_m_z=stride_m_z,
        stride_m_h=stride_m_h,
        stride_m_t=stride_m_t,
        HEAD_DIM=head_dim,
        FP8_OUTPUT=False,
        STAGE=stage,
        warp_specialize=warp_specialize,
        IS_HOPPER=is_hopper(),
        **extra_kern_args,
    )

    if cast_inputs:
        return o_kernel.to(q.dtype)
    return o_kernel
    return o


@triton.jit
def addmm_act(
        a_ptr,
        b_ptr,
        c_ptr,
        d_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        stride_dm,
        stride_dn,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        ACTIVATION: tl.constexpr,
):
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
    for k_iter in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k_iter * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k_iter * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if ACTIVATION == 1:
        accumulator = tl.maximum(accumulator, 0.0)
    elif ACTIVATION == 2:
        accumulator = gelu(accumulator)

    d = accumulator.to(a_ptr.dtype.element_ty)
    d_ptrs = d_ptr + stride_dm * offs_m[:, None] + stride_dn * offs_n[None, :]
    tl.store(d_ptrs, d, mask=tile_mask)


@triton.jit
def gelu(x):
    return 0.5 * x * (1.0 + erf(x / tl.sqrt(2.0)))


ADDMM_NO_ACT_BLOCK_SIZE_M = 128
ADDMM_NO_ACT_BLOCK_SIZE_N = 64
ADDMM_NO_ACT_BLOCK_SIZE_K = 64
ADDMM_NO_ACT_GROUP_SIZE_M = 8
ADDMM_NO_ACT_NUM_WARPS = 4
ADDMM_NO_ACT_NUM_STAGES = 4

ADDMM_GELU_BLOCK_SIZE_M = 128
ADDMM_GELU_BLOCK_SIZE_N = 128
ADDMM_GELU_BLOCK_SIZE_K = 32
ADDMM_GELU_GROUP_SIZE_M = 6
ADDMM_GELU_NUM_WARPS = 4
ADDMM_GELU_NUM_STAGES = 4


def _launch_addmm(a: torch.Tensor, b: torch.Tensor, bias: Optional[torch.Tensor], *, activation: int,
                  block_size_m: int, block_size_n: int, block_size_k: int, group_size_m: int,
                  num_warps: int, num_stages: int) -> torch.Tensor:
    if a.device != b.device:
        raise ValueError("Tensors must share device")
    if bias is not None and bias.device != a.device:
        raise ValueError("Bias must reside on same device as inputs")

    common_dtype = a.dtype
    if b.dtype != common_dtype:
        raise ValueError("Input matrices must share dtype")
    if bias is not None and bias.dtype != common_dtype:
        bias = bias.to(common_dtype)

    M, K = a.shape
    if b.shape[0] != K:
        raise ValueError(f"Incompatible shapes: a={a.shape}, b={b.shape}")
    N = b.shape[1]

    if common_dtype != torch.bfloat16:
        a_fp32 = a.to(torch.float32)
        b_fp32 = b.to(torch.float32)
        out_fp32 = torch.matmul(a_fp32, b_fp32)
        if bias is not None:
            if bias.ndim == 1 and bias.shape[0] == N:
                out_fp32 = out_fp32 + bias.to(torch.float32)
            elif bias.ndim == 2 and bias.shape[1] == N:
                out_fp32 = out_fp32 + bias.to(torch.float32)
            else:
                raise ValueError("Bias must be 1D of size N or 2D with shape (1, N)")
        if activation == 1:
            out_fp32 = torch.relu(out_fp32)
        elif activation == 2:
            out_fp32 = F.gelu(out_fp32, approximate="none")
        return out_fp32.to(dtype=common_dtype)

    if bias is None:
        out = torch.matmul(a.to(torch.float32), b.to(torch.float32))
        if activation == 1:
            out = torch.relu(out)
        elif activation == 2:
            out = F.gelu(out, approximate="none")
        return out.to(dtype=a.dtype)

    if not a.is_contiguous():
        raise ValueError("addmm kernels expect contiguous 'a' tensor")

    if not b.is_contiguous():
        raise ValueError("addmm kernels expect contiguous 'b' tensor")

    if bias.ndim == 1 and bias.shape[0] == N:
        bias_matrix = bias.unsqueeze(0)
    elif bias.ndim == 2 and bias.shape[1] == N:
        bias_matrix = bias
    else:
        raise ValueError("Bias must be 1D of size N or 2D with shape (1, N)")

    bias_mat = bias_matrix.expand(M, -1)
    out = torch.empty((M, N), dtype=a.dtype, device=a.device)

    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),)

    addmm_act[grid](
        a, b, bias_mat, out,
        M, N, K,
        *a.stride(),
        *b.stride(),
        *bias_mat.stride(),
        *out.stride(),
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        GROUP_SIZE_M=group_size_m,
        ACTIVATION=activation,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


def addmm_linear(a: torch.Tensor, b: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
    return _launch_addmm(
        a, b, bias,
        activation=0,
        block_size_m=ADDMM_NO_ACT_BLOCK_SIZE_M,
        block_size_n=ADDMM_NO_ACT_BLOCK_SIZE_N,
        block_size_k=ADDMM_NO_ACT_BLOCK_SIZE_K,
        group_size_m=ADDMM_NO_ACT_GROUP_SIZE_M,
        num_warps=ADDMM_NO_ACT_NUM_WARPS,
        num_stages=ADDMM_NO_ACT_NUM_STAGES,
    )


def addmm_linear_gelu(a: torch.Tensor, b: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
    return _launch_addmm(
        a, b, bias,
        activation=2,
        block_size_m=ADDMM_GELU_BLOCK_SIZE_M,
        block_size_n=ADDMM_GELU_BLOCK_SIZE_N,
        block_size_k=ADDMM_GELU_BLOCK_SIZE_K,
        group_size_m=ADDMM_GELU_GROUP_SIZE_M,
        num_warps=ADDMM_GELU_NUM_WARPS,
        num_stages=ADDMM_GELU_NUM_STAGES,
    )


@triton.jit
def avg_pool1d_kernel(
        input_ptr,
        output_ptr,
        stride_in_n,
        stride_in_c,
        stride_in_l,
        stride_out_n,
        stride_out_c,
        stride_out_l,
        N,
        C,
        L_IN,
        L_OUT,
        kernel_size,
        stride,
        inv_kernel_size,
        BLOCK_SIZE: tl.constexpr,
        MAX_KERNEL_SIZE: tl.constexpr,
):
    pid_out = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)

    if pid_n >= N or pid_c >= C:
        return

    offs_out = pid_out * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_out = offs_out < L_OUT

    tl.device_assert(kernel_size > 0)
    tl.device_assert(kernel_size <= MAX_KERNEL_SIZE)

    base_in = input_ptr + pid_n * stride_in_n + pid_c * stride_in_c
    base_out = output_ptr + pid_n * stride_out_n + pid_c * stride_out_c

    in_offsets = offs_out * stride

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for k in tl.static_range(MAX_KERNEL_SIZE):
        mask_k = k < kernel_size
        sample_idx = in_offsets + k
        mask = mask_out & mask_k & (sample_idx < L_IN)
        vals = tl.load(base_in + sample_idx * stride_in_l, mask=mask, other=0.0)
        acc += vals.to(tl.float32)

    avg = acc * inv_kernel_size
    tl.store(base_out + offs_out * stride_out_l, avg.to(tl.bfloat16), mask=mask_out)


AVG_POOL_BLOCK_SIZE = 64
AVG_POOL_MAX_KERNEL_SIZE = 32
AVG_POOL_NUM_WARPS = 2
AVG_POOL_NUM_STAGES = 2


def avg_pool1d_triton(x: torch.Tensor, kernel_size: int, stride: int) -> torch.Tensor:
    if x.dtype != torch.bfloat16:
        raise ValueError("avg_pool1d_triton expects bf16 input")
    if not x.is_contiguous():
        raise ValueError("avg_pool1d_triton expects contiguous input")

    N, C, L_in = x.shape
    L_out = (L_in - kernel_size) // stride + 1
    if L_out <= 0:
        raise ValueError("avg_pool1d_triton produced empty output")
    y = torch.empty((N, C, L_out), dtype=x.dtype, device=x.device)

    stride_in_n, stride_in_c, stride_in_l = x.stride()
    stride_out_n, stride_out_c, stride_out_l = y.stride()

    inv_kernel = 1.0 / float(kernel_size)

    def grid(meta):
        return (triton.cdiv(L_out, meta["BLOCK_SIZE"]), C, N)

    avg_pool1d_kernel[grid](
        x,
        y,
        stride_in_n,
        stride_in_c,
        stride_in_l,
        stride_out_n,
        stride_out_c,
        stride_out_l,
        N,
        C,
        L_in,
        L_out,
        kernel_size,
        stride,
        inv_kernel,
        BLOCK_SIZE=AVG_POOL_BLOCK_SIZE,
        MAX_KERNEL_SIZE=AVG_POOL_MAX_KERNEL_SIZE,
        num_warps=AVG_POOL_NUM_WARPS,
        num_stages=AVG_POOL_NUM_STAGES,
    )
    return y


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
    block_size = 1024
    grid = (triton.cdiv(n_elements, block_size),)
    fill_uniform[grid](tensor, n_elements, low, high, seed, BLOCK_SIZE=block_size)


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


def fill_normal_(tensor: torch.Tensor, mean: float, std: float, seed: int):
    n_elements = tensor.numel()
    block_size = 1024
    grid = (triton.cdiv(n_elements, block_size),)
    fill_normal[grid](tensor, n_elements, mean, std, seed, BLOCK_SIZE=block_size)


def kaiming_uniform_(tensor: torch.Tensor, fan_in: int, seed: int):
    bound = math.sqrt(1.0 / fan_in)
    fill_uniform_(tensor, -bound, bound, seed)


@triton.autotune(
    configs=[triton.Config({"BLOCK_SIZE": 64}, num_warps=2, num_stages=2)],
    key=["num_rows", "num_cols"],
)
@triton.jit
def rms_norm_fwd(
        input_ptr,
        weight_ptr,
        output_ptr,
        num_rows,
        num_cols,
        eps,
        BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= num_rows:
        return

    row_start = row_idx * num_cols
    offsets = tl.arange(0, BLOCK_SIZE)
    sum_squares = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for col_start in tl.range(0, num_cols, BLOCK_SIZE):
        cols = col_start + offsets
        mask = cols < num_cols
        x = tl.load(input_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
        sum_squares += tl.where(mask, x * x, 0.0)

    total_squares = tl.sum(sum_squares, axis=0)
    mean_square = total_squares / num_cols
    inv_rms = tl.rsqrt(mean_square + eps)

    for col_start in tl.range(0, num_cols, BLOCK_SIZE):
        cols = col_start + offsets
        mask = cols < num_cols
        x = tl.load(input_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
        gamma = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        y = x * inv_rms * gamma
        tl.store(output_ptr + row_start + cols, y.to(tl.bfloat16), mask=mask)


def _rms_norm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    if not x.is_cuda or not weight.is_cuda:
        raise RuntimeError("rms_norm expects CUDA tensors for Triton execution")

    if not x.is_contiguous():
        raise RuntimeError("rms_norm expects contiguous input tensor")

    if not weight.is_contiguous():
        raise RuntimeError("rms_norm expects contiguous weight tensor")

    x_contig = x
    weight_contig = weight

    x_bf16 = x_contig.to(torch.bfloat16)
    weight_bf16 = weight_contig.to(torch.bfloat16)
    output_bf16 = torch.empty_like(x_bf16)

    num_rows = math.prod(x_bf16.shape[:-1])
    num_cols = x_bf16.shape[-1]

    if num_rows == 0 or num_cols == 0:
        return output_bf16.view_as(x_bf16).to(x_contig.dtype)

    rms_norm_fwd[(num_rows,)](
        x_bf16.view(-1),
        weight_bf16.view(-1),
        output_bf16.view(-1),
        num_rows,
        num_cols,
        eps,
    )

    return output_bf16.view_as(x_bf16).to(x_contig.dtype)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return _rms_norm_triton(x, weight, eps)


def init_conv_block(seed: int, device: torch.device, dtype: torch.dtype, mode: str) -> Tuple[
    Dict[str, torch.Tensor], int]:
    params: Dict[str, torch.Tensor] = {}

    params["ln_weight"] = torch.full((N_EMBD,), 1.0, dtype=dtype, device=device)

    out_channels = N_EMBD if mode == "dilated" else 2 * N_EMBD
    params["conv_weight"] = torch.empty((out_channels, 3, 3, N_EMBD), dtype=dtype, device=device)
    kaiming_uniform_(params["conv_weight"], N_EMBD * 9, seed)
    seed += 1

    params["conv_weight_oihw"] = params["conv_weight"].permute(0, 3, 1, 2).contiguous()

    return params, seed


def init_attention_block(seed: int, device: torch.device, dtype: torch.dtype, use_bias: bool):
    params = {}

    params["ln1_weight"] = torch.full((N_EMBD,), 1.0, dtype=dtype, device=device)

    params["attn_w_qkv"] = torch.empty((N_EMBD, 3 * N_EMBD), dtype=dtype, device=device)
    kaiming_uniform_(params["attn_w_qkv"], N_EMBD, seed)
    seed += 1

    if use_bias:
        params["attn_b_qkv"] = torch.empty((3 * N_EMBD,), dtype=dtype, device=device)
        kaiming_uniform_(params["attn_b_qkv"], N_EMBD, seed)
        seed += 1
    else:
        params["attn_b_qkv"] = None

    params["attn_w_proj"] = torch.empty((N_EMBD, N_EMBD), dtype=dtype, device=device)
    kaiming_uniform_(params["attn_w_proj"], N_EMBD, seed)
    seed += 1

    if use_bias:
        params["attn_b_proj"] = torch.empty((N_EMBD,), dtype=dtype, device=device)
        kaiming_uniform_(params["attn_b_proj"], N_EMBD, seed)
        seed += 1
    else:
        params["attn_b_proj"] = None

    params["ln2_weight"] = torch.full((N_EMBD,), 1.0, dtype=dtype, device=device)

    params["fc1_weight"] = torch.empty((N_EMBD, HIDDEN_DIM), dtype=dtype, device=device)
    kaiming_uniform_(params["fc1_weight"], N_EMBD, seed)
    seed += 1

    params["fc1_bias"] = torch.empty((HIDDEN_DIM,), dtype=dtype, device=device)
    kaiming_uniform_(params["fc1_bias"], N_EMBD, seed)
    seed += 1

    params["fc2_weight"] = torch.empty((HIDDEN_DIM, N_EMBD), dtype=dtype, device=device)
    kaiming_uniform_(params["fc2_weight"], HIDDEN_DIM, seed)
    seed += 1

    params["fc2_bias"] = torch.empty((N_EMBD,), dtype=dtype, device=device)
    kaiming_uniform_(params["fc2_bias"], HIDDEN_DIM, seed)
    seed += 1

    return params, seed


def attention_block_forward(x: torch.Tensor, params: dict, dtype: torch.dtype) -> torch.Tensor:
    batch, seq, _ = x.shape
    head_dim = N_EMBD // NUM_HEADS

    ln1_weight = params["ln1_weight"]
    norm1_fp32 = rms_norm(x, ln1_weight, RMS_EPS)
    norm1 = norm1_fp32.to(dtype)

    w_qkv = params["attn_w_qkv"]
    b_qkv = params["attn_b_qkv"]
    qkv = addmm_linear(norm1.view(batch * seq, N_EMBD), w_qkv, b_qkv).view(batch, seq, 3, N_EMBD)
    q, k, v = qkv.unbind(dim=2)

    q = q.view(batch, seq, NUM_HEADS, head_dim).permute(0, 2, 1, 3).to(dtype)
    k = k.view(batch, seq, NUM_HEADS, head_dim).permute(0, 2, 1, 3).to(dtype)
    v = v.view(batch, seq, NUM_HEADS, head_dim).permute(0, 2, 1, 3).to(dtype)

    attn = flash_attn_fwd(q, k, v, causal=False, sm_scale=1.0 / math.sqrt(head_dim))

    attn_out = attn.permute(0, 2, 1, 3).contiguous().view(batch * seq, N_EMBD)
    w_proj = params["attn_w_proj"]
    b_proj = params["attn_b_proj"]

    attn_proj = addmm_linear(attn_out, w_proj, b_proj).view(batch, seq, N_EMBD)

    residual_after_attn = x + attn_proj.to(dtype)

    ln2_weight = params["ln2_weight"]
    norm2 = rms_norm(residual_after_attn, ln2_weight, RMS_EPS)

    w_fc1 = params["fc1_weight"]
    b_fc1 = params["fc1_bias"]
    hidden = addmm_linear_gelu(norm2.view(batch * seq, N_EMBD), w_fc1, b_fc1)

    w_fc2 = params["fc2_weight"]
    b_fc2 = params["fc2_bias"]
    mlp_out = addmm_linear(hidden, w_fc2, b_fc2).view(batch, seq, N_EMBD)

    output = residual_after_attn + mlp_out
    return output.to(x.dtype)


def conv_block_forward_nchw_cl(image_nchw_cl: torch.Tensor, params: dict, mode: str, dilation: int,
                               dtype: torch.dtype) -> torch.Tensor:
    ln_weight = params["ln_weight"]
    N, C, H, W = image_nchw_cl.shape

    # Original math does RMSNorm on (B, T, C) before conv; do the same via a view.
    x_bt_c = image_nchw_cl.permute(0, 2, 3, 1).reshape(N, H * W, C)
    norm_bt_c = rms_norm(x_bt_c, ln_weight, RMS_EPS).to(dtype)
    norm_nchw = norm_bt_c.reshape(N, H, W, C).permute(0, 3, 1, 2)

    weight = params["conv_weight_oihw"]  # precomputed once

    if mode == "dilated":
        conv_out = torch.nn.functional.conv2d(norm_nchw, weight, bias=None, stride=1, padding=dilation,
                                              dilation=dilation)
        activated = torch.nn.functional.gelu(conv_out.to(torch.float32), approximate="none").to(dtype)
    else:
        conv_out = torch.nn.functional.conv2d(norm_nchw, weight, bias=None, stride=1, padding=1, dilation=1)
        val, gate = conv_out.split(N_EMBD, dim=1)
        gate_act = torch.nn.functional.gelu(gate.to(torch.float32), approximate="none").to(dtype)
        activated = val * gate_act

    return activated  # (N, C, H, W), channels_last-friendly by inheritance


def generate_variant_output(use_bias: bool, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    param_seed = MODEL_INIT_SEED

    position_embedding = torch.empty((ROWS * COLS, N_EMBD), dtype=dtype, device=device)
    fill_normal_(position_embedding, 0.0, 1.0, param_seed)

    codepoint_embedding = torch.empty((MAX_CODE_POINT, N_EMBD), dtype=dtype, device=device)
    fill_normal_(codepoint_embedding, 0.0, 1.0, param_seed)
    param_seed += 1

    fg_r_embed = torch.empty((N_EMBD,), dtype=dtype, device=device)
    fill_normal_(fg_r_embed, 0.0, 1.0, param_seed)
    param_seed += 1
    fg_g_embed = torch.empty((N_EMBD,), dtype=dtype, device=device)
    fill_normal_(fg_g_embed, 0.0, 1.0, param_seed)
    param_seed += 1
    fg_b_embed = torch.empty((N_EMBD,), dtype=dtype, device=device)
    fill_normal_(fg_b_embed, 0.0, 1.0, param_seed)
    param_seed += 1

    bg_r_embed = torch.empty((N_EMBD,), dtype=dtype, device=device)
    fill_normal_(bg_r_embed, 0.0, 1.0, param_seed)
    param_seed += 1
    bg_g_embed = torch.empty((N_EMBD,), dtype=dtype, device=device)
    fill_normal_(bg_g_embed, 0.0, 1.0, param_seed)
    param_seed += 1
    bg_b_embed = torch.empty((N_EMBD,), dtype=dtype, device=device)
    fill_normal_(bg_b_embed, 0.0, 1.0, param_seed)
    param_seed += 1

    downsample_blocks = []
    for _ in range(NUM_DOWNSAMPLE_BLOCKS):
        params, param_seed = init_conv_block(param_seed, device, dtype, DOWNSAMPLE_CONV_MODE)
        downsample_blocks.append(params)

    blocks = []
    for _ in range(NUM_LAYERS):
        params, param_seed = init_attention_block(param_seed, device, dtype, use_bias)
        blocks.append(params)

    cell_states = compute_cell_states(device)
    cell_embeddings = build_cell_embeds(
        cell_states,
        codepoint_embedding,
        position_embedding,
        fg_r_embed,
        fg_g_embed,
        fg_b_embed,
        bg_r_embed,
        bg_g_embed,
        bg_b_embed,
        out_dtype=dtype,
    )

    x = cell_embeddings.view(BATCH_SIZE, SEQUENCE_LENGTH, N_EMBD)

    rows = ROWS
    cols = COLS
    B = x.shape[0]

    image = (
        x.view(B, rows, cols, N_EMBD)
        .permute(0, 3, 1, 2)
        .to(memory_format=torch.channels_last)
    )

    for params in downsample_blocks:
        image = conv_block_forward_nchw_cl(image, params, DOWNSAMPLE_CONV_MODE, DOWNSAMPLE_CONV_DILATION, dtype)
        image = torch.nn.functional.avg_pool2d(image.to(torch.float32), kernel_size=2, stride=2).to(dtype)

        rows //= 2
        cols //= 2

    x = image.permute(0, 2, 3, 1).reshape(B, rows * cols, N_EMBD)

    for params in blocks:
        x = attention_block_forward(x, params, dtype)

    x = x.mean(dim=1).to(dtype)
    return x.cpu().contiguous()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required to generate reference tensors")

    device = torch.device("cuda")
    variants = (
        (torch.bfloat16, "reference_bf16.safetensors"),
        (torch.float16, "reference_fp16.safetensors"),
    )

    for dtype, filename in variants:
        outputs = {
            "output_bias_true": generate_variant_output(True, device, dtype),
            "output_bias_false": generate_variant_output(False, device, dtype),
        }
        save_file(outputs, filename)


if __name__ == "__main__":
    main()
