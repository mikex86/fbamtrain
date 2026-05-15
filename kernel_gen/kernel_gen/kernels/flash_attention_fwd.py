import math
import torch
import triton
import triton.language as tl

# --------------------------------------------------------------------------------
# Device / arch helpers
# --------------------------------------------------------------------------------
DEVICE = triton.runtime.driver.active.get_active_torch_device()


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def supports_host_descriptor():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def is_blackwell():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 10


def is_hopper():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 9


# --------------------------------------------------------------------------------
# FlashAttention v2 forward (Triton, descriptor-style layout)
# --------------------------------------------------------------------------------
@triton.jit
def _attn_fwd_inner(
        acc, l_i, m_i, q,
        desc_k, desc_v,
        k_ptr, v_ptr, stride_q,
        offset_y, dtype: tl.constexpr, start_m, qk_scale,
        BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
        STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,
        N_CTX: tl.constexpr, warp_specialize: tl.constexpr, IS_HOPPER: tl.constexpr,
        ACCUMULATE_IN_FP16: tl.constexpr,
        EVEN_N_CTX: tl.constexpr
):
    acc_dtype = tl.float16 if ACCUMULATE_IN_FP16 else tl.float32
    use_warp_specialize: tl.constexpr = warp_specialize and EVEN_N_CTX
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    else:
        lo, hi = 0, N_CTX

    offsetk_y = (offset_y + lo).to(tl.int32)
    offsetv_y = (offset_y + lo).to(tl.int32)
    if not EVEN_N_CTX:
        offs_d = tl.arange(0, HEAD_DIM)
    for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=use_warp_specialize):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        if EVEN_N_CTX:
            k = desc_k.load([offsetk_y, 0]).T
        else:
            offs_n_curr = start_n + offs_n
            mask_n = offs_n_curr < N_CTX
            k_ptrs = k_ptr + (offset_y + offs_n_curr)[:, None] * stride_q + offs_d[None, :]
            k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0).T
        qk = tl.dot(q, k)
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            if not EVEN_N_CTX:
                mask = mask & mask_n[None, :]
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            if not EVEN_N_CTX:
                qk = tl.where(mask_n[None, :], qk, -float("inf"))
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]

        p = tl.math.exp2(qk)
        if not EVEN_N_CTX:
            p = tl.where(mask_n[None, :], p, 0.0)
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)

        if not IS_HOPPER and use_warp_specialize and BLOCK_M == 128 and HEAD_DIM == 128:
            BM: tl.constexpr = acc.shape[0]
            BN: tl.constexpr = acc.shape[1]
            acc0, acc1 = acc.reshape([BM, 2, BN // 2]).permute(0, 2, 1).split()
            acc0 = acc0 * alpha[:, None].to(acc_dtype)
            acc1 = acc1 * alpha[:, None].to(acc_dtype)
            acc = tl.join(acc0, acc1).permute(0, 2, 1).reshape([BM, BN])
        else:
            acc = acc * alpha[:, None].to(acc_dtype)

        if dtype == tl.float8e5:
            if EVEN_N_CTX:
                v = desc_v.load([0, offsetv_y]).T
            else:
                v_ptrs = v_ptr + offs_d[:, None] + (offset_y + offs_n_curr)[None, :] * stride_q
                v = tl.load(v_ptrs, mask=mask_n[None, :], other=0.0).T
        else:
            if EVEN_N_CTX:
                v = desc_v.load([offsetv_y, 0])
            else:
                v_ptrs = v_ptr + (offset_y + offs_n_curr)[:, None] * stride_q + offs_d[None, :]
                v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        p = p.to(dtype)
        acc = tl.dot(p, v, acc, out_dtype=acc_dtype)

        l_i = l_i * alpha + l_ij
        m_i = m_ij
        if EVEN_N_CTX:
            offsetk_y += BLOCK_N
            offsetv_y += BLOCK_N
    return acc, l_i, m_i


if is_hip():
    NUM_STAGES_OPTIONS = [1]
else:
    NUM_STAGES_OPTIONS = [2, 3, 4]

configs = [
    triton.Config({'BLOCK_SIZE_X': BM, 'BLOCK_SIZE_Y': BN}, num_stages=s, num_warps=w)
    for BM in [64, 128]
    for BN in [32, 64, 128]
    for s in NUM_STAGES_OPTIONS
    for w in [4, 8]
]


def keep(conf):
    BLOCK_SIZE_X = conf.kwargs["BLOCK_SIZE_X"]
    BLOCK_SIZE_Y = conf.kwargs["BLOCK_SIZE_Y"]
    return not (is_cuda() and torch.cuda.get_device_capability()[0] == 9 and BLOCK_SIZE_X * BLOCK_SIZE_Y < 128 * 128
                and conf.num_warps == 8)


def prune_invalid_configs(configs, named_args, **kwargs):
    N_CTX = kwargs["N_CTX"]
    return [conf for conf in configs if conf.kwargs.get("BLOCK_SIZE_X", 0) <= N_CTX]


@triton.jit
def _maybe_make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
    if isinstance(desc_or_ptr, tl.tensor_descriptor):
        return desc_or_ptr
    else:
        return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)


@triton.autotune(
    configs=list(filter(keep, configs)),
    key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT", "warp_specialize"],
    prune_configs_by={'early_config_prune': prune_invalid_configs}
)
@triton.jit
def mha_attn_fwd(
        sm_scale, M_ptr,
        Z, H,
        q_ptr, k_ptr, v_ptr, o_ptr,
        stride_q,
        N_CTX,
        HEAD_DIM: tl.constexpr,
        BLOCK_SIZE_X: tl.constexpr,
        BLOCK_SIZE_Y: tl.constexpr,
        FP8_OUTPUT: tl.constexpr,
        STAGE: tl.constexpr,
        warp_specialize: tl.constexpr,
        IS_HOPPER: tl.constexpr,
        ACCUMULATE_IN_FP16: tl.constexpr,
        WRITE_M: tl.constexpr,
        EVEN_N_CTX: tl.constexpr,
):
    tl.static_assert(BLOCK_SIZE_Y <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    y_dim = Z * N_CTX
    row_stride = stride_q
    head_offset = off_h * HEAD_DIM
    q_ptr += head_offset
    k_ptr += head_offset
    v_ptr += head_offset

    desc_q = _maybe_make_tensor_desc(q_ptr, shape=[y_dim, HEAD_DIM],
                                     strides=[row_stride, 1], block_shape=[BLOCK_SIZE_X, HEAD_DIM])
    desc_k = _maybe_make_tensor_desc(k_ptr, shape=[y_dim, HEAD_DIM],
                                     strides=[row_stride, 1], block_shape=[BLOCK_SIZE_Y, HEAD_DIM])
    if FP8_OUTPUT:
        desc_v = _maybe_make_tensor_desc(v_ptr, shape=[HEAD_DIM, y_dim],
                                         strides=[1, row_stride], block_shape=[HEAD_DIM, BLOCK_SIZE_Y])
    else:
        desc_v = _maybe_make_tensor_desc(v_ptr, shape=[y_dim, HEAD_DIM],
                                         strides=[row_stride, 1], block_shape=[BLOCK_SIZE_Y, HEAD_DIM])
    offset_y = off_z * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_SIZE_X

    offs_m = start_m * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    offs_n = tl.arange(0, BLOCK_SIZE_Y)
    offs_d = tl.arange(0, HEAD_DIM)

    if EVEN_N_CTX:
        q = desc_q.load([qo_offset_y, 0])
    else:
        mask_m = offs_m < N_CTX
        q_ptrs = q_ptr + (offset_y + offs_m)[:, None] * stride_q + offs_d[None, :]
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    input_dtype = q.dtype
    dtype = tl.float8e5 if FP8_OUTPUT else input_dtype

    m_i = tl.zeros([BLOCK_SIZE_X], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_SIZE_X], dtype=tl.float32) + 1.0
    acc_dtype = tl.float16 if ACCUMULATE_IN_FP16 else tl.float32
    acc = tl.zeros([BLOCK_SIZE_X, HEAD_DIM], dtype=acc_dtype)

    qk_scale = sm_scale * 1.44269504  # 1/log(2)

    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q,
            desc_k, desc_v, k_ptr, v_ptr, stride_q,
            offset_y, dtype, start_m, qk_scale,
            BLOCK_SIZE_X, HEAD_DIM, BLOCK_SIZE_Y,
            4 - STAGE, offs_m, offs_n, N_CTX,
            warp_specialize, IS_HOPPER, ACCUMULATE_IN_FP16,
            EVEN_N_CTX
        )
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q,
            desc_k, desc_v, k_ptr, v_ptr, stride_q,
            offset_y, dtype, start_m, qk_scale,
            BLOCK_SIZE_X, HEAD_DIM, BLOCK_SIZE_Y,
            2, offs_m, offs_n, N_CTX,
            warp_specialize, IS_HOPPER, ACCUMULATE_IN_FP16,
            EVEN_N_CTX
        )

    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None].to(acc_dtype)

    if WRITE_M:
        m_offsets = M_ptr + off_hz * N_CTX + offs_m
        if EVEN_N_CTX:
            tl.store(m_offsets, m_i)
        else:
            tl.store(m_offsets, m_i, mask=mask_m)

    out_base = (off_z * N_CTX + offs_m) * H + off_h
    o_ptrs = o_ptr + out_base[:, None] * HEAD_DIM + offs_d[None, :]
    if EVEN_N_CTX:
        tl.store(o_ptrs, acc.to(dtype))
    else:
        tl.store(o_ptrs, acc.to(dtype), mask=mask_m[:, None])


# --------------------------------------------------------------------------------
# Python wrapper (forward-only) + benchmark
# --------------------------------------------------------------------------------
def flash_attn_fwd(q, k, v, causal=True, sm_scale=1.0, warp_specialize=True, accumulate_in_fp16=False):
    """Forward-only wrapper around the Triton kernel. q/k/v: (B,T,H,D) BF16/FP16."""
    HEAD_DIM = q.shape[-1]
    assert HEAD_DIM in {16, 32, 64, 128, 256}
    if q.dtype not in {torch.bfloat16, torch.float16}:
        raise ValueError("flash_attn_fwd expects BF16 or FP16 tensors")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("flash_attn_fwd requires q, k, v tensors with matching dtype")
    assert q.shape == k.shape == v.shape

    o = torch.empty_like(q)
    stage = 3 if causal else 1
    extra_kern_args = {}

    if is_hip():
        waves_per_eu = 3 if HEAD_DIM <= 64 else 2
        extra_kern_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}

    # scratch for m_i
    M = torch.empty((q.shape[0], q.shape[2], q.shape[1]), device=q.device, dtype=torch.float32)

    def alloc_fn(size: int, align: int, _):
        # Triton scratch allocator
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)

    def grid(META):
        # program_id(0) sweeps tokens in BLOCK_SIZE_X tiles, program_id(1) sweeps B*H heads
        return (triton.cdiv(q.shape[1], META["BLOCK_SIZE_X"]), q.shape[0] * q.shape[2], 1)

    # register budget tuning for Blackwell
    if is_blackwell() and warp_specialize:
        extra_kern_args["maxnreg"] = 168 if HEAD_DIM == 128 else 80

    mha_attn_fwd[grid](
        sm_scale, M,
        q.shape[0], q.shape[2],
        q, k, v, o,
        q.stride(1),
        N_CTX=q.shape[1],
        HEAD_DIM=HEAD_DIM,
        FP8_OUTPUT=False,
        STAGE=stage,
        warp_specialize=warp_specialize,
        IS_HOPPER=is_hopper(),
        ACCUMULATE_IN_FP16=accumulate_in_fp16,
        WRITE_M=True,
        EVEN_N_CTX=False,
        **extra_kern_args
    )
    return o


def bench_once(B=32, T=4096, C=1024, HEAD_DIM=128, causal=False, use_triton=True):
    assert C % HEAD_DIM == 0, "C must be divisible by HEAD_DIM"
    H = C // HEAD_DIM
    dtype = torch.bfloat16
    ws = is_blackwell() or (is_hopper() and not causal)  # follow safe defaults

    torch.manual_seed(0)
    q = torch.randn((B, T, H * HEAD_DIM), dtype=dtype, device=DEVICE)
    k = torch.randn((B, T, H * HEAD_DIM), dtype=dtype, device=DEVICE)
    v = torch.randn((B, T, H * HEAD_DIM), dtype=dtype, device=DEVICE)

    q = q.view(B, T, H, HEAD_DIM)
    k = k.view(B, T, H, HEAD_DIM)
    v = v.view(B, T, H, HEAD_DIM)

    sm_scale = 1.0 / math.sqrt(HEAD_DIM)
    if use_triton:
        fn = lambda: flash_attn_fwd(q, k, v, causal=causal, sm_scale=sm_scale, warp_specialize=ws)
    else:
        fn = lambda: torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None if not causal else torch.tril(torch.ones((T, T), device=q.device)).bool(),
            dropout_p=0.0, is_causal=causal, scale=sm_scale
        )

    ms = triton.testing.do_bench(fn)  # latency in milliseconds

    # FLOPs accounting: two matmuls (QK^T and P@V). If causal, ~half the work on average.
    flops_per_matmul = 2.0 * B * H * T * T * HEAD_DIM
    total_flops = 2.0 * flops_per_matmul
    if causal:
        total_flops *= 0.5

    tflops = total_flops * 1e-12 / (ms * 1e-3)
    return ms, tflops, dict(B=B, H=H, T=T, D=HEAD_DIM, C=C, causal=causal, ws=ws)


if __name__ == "__main__":
    ms, tflops, meta = bench_once(B=32, T=4096, C=1024, HEAD_DIM=128, causal=True, use_triton=True)
    print(
        f"FlashAttention FWD (causal={meta['causal']}, warp_specialize={meta['ws']}) "
        f"— B={meta['B']}, H={meta['H']}, T={meta['T']}, D={meta['D']} (C={meta['C']}) on {DEVICE}"
    )
    print(f"Latency: {ms:.2f} ms | Throughput: {tflops:.2f} TFLOPS")

    ms, tflops, meta = bench_once(B=32, T=4096, C=1024, HEAD_DIM=128, causal=False, use_triton=False)
    print(
        f"PyTorch Scaled Dot-Product Attention (causal={meta['causal']}) "
        f"— B={meta['B']}, H={meta['H']}, T={meta['T']}, D={meta['D']} (C={meta['C']}) on {DEVICE}"
    )
    print(f"Latency: {ms:.2f} ms | Throughput: {tflops:.2f} TFLOPS")
