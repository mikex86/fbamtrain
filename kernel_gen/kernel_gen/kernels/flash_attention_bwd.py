import triton
import triton.language as tl


@triton.jit
def mha_attn_bwd_pre(
        o_ptr, do_ptr, delta_ptr,
        Z, H, N_CTX,
        stride_z, stride_h, stride_tok, stride_d,
        stride_delta_z, stride_delta_h, stride_delta_t,
        BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr,
        BLOCK_SIZE_X: tl.constexpr, BLOCK_SIZE_Y: tl.constexpr
):
    tl.static_assert(BLOCK_M == BLOCK_SIZE_X)
    tl.static_assert(HEAD_DIM == BLOCK_SIZE_Y)
    pid_m = tl.program_id(0)
    pid_hz = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    # Flatten (B, H) into a single dimension for contiguous layout.
    off_hz = pid_hz
    off_hz = off_hz.to(tl.int64)
    off_z = off_hz // H
    off_h = off_hz % H

    base = off_z * stride_z + off_h * stride_h
    mask_m = offs_m < N_CTX

    o_ptrs = o_ptr + base + offs_m[:, None] * stride_tok + offs_d[None, :] * stride_d
    do_ptrs = do_ptr + base + offs_m[:, None] * stride_tok + offs_d[None, :] * stride_d

    o = tl.load(o_ptrs, mask=mask_m[:, None], other=0.0)
    do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
    delta = tl.sum(o * do, axis=1)

    delta_ptrs = delta_ptr + off_hz * N_CTX + offs_m
    tl.store(delta_ptrs, delta, mask=mask_m)


@triton.jit
def _attn_bwd_dkdv(dk, dv,
                   Q, k, v, sm_scale,
                   DO,
                   M, D,
                    BLOCK_M1: tl.constexpr,
                    BLOCK_N1: tl.constexpr,
                    HEAD_DIM: tl.constexpr,
                   start_n, start_m, num_steps,
                   stride_do, stride_q,
                   N_CTX,
                   MASK: tl.constexpr,
                   EVEN_N_CTX: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, HEAD_DIM)
    qT_ptrs = Q + offs_m[None, :] * stride_q + offs_k[:, None]
    do_ptrs = DO + offs_m[:, None] * stride_do + offs_k[None, :]
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    if not EVEN_N_CTX:
        mask_n = offs_n < N_CTX
    curr_m = start_m
    step_m = BLOCK_M1
    qk_scale = sm_scale * 1.44269504
    for _ in range(num_steps):
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        if EVEN_N_CTX:
            qT = tl.load(qT_ptrs)
            m = tl.load(M + offs_m)
            do = tl.load(do_ptrs)
            Di = tl.load(D + offs_m)
        else:
            mask_m = offs_m < N_CTX
            qT = tl.load(qT_ptrs, mask=mask_m[None, :], other=0.0)
            m = tl.load(M + offs_m, mask=mask_m, other=0.0)
            do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)
            Di = tl.load(D + offs_m, mask=mask_m, other=0.0)
        qkT = tl.dot(k, qT) * qk_scale
        pT = tl.math.exp2(qkT - m[None, :])
        if MASK:
            mask = (offs_m[None, :] >= offs_n[:, None])
            pT = tl.where(mask, pT, 0.0)
        if not EVEN_N_CTX:
            pT = tl.where(mask_m[None, :], pT, 0.0)
            pT = tl.where(mask_n[:, None], pT, 0.0)
        ppT = pT
        ppT = ppT.to(qT.dtype)
        dv += tl.dot(ppT, do)
        dpT = tl.dot(v, tl.trans(do), out_dtype=tl.float32)
        dsT = pT * (dpT - Di[None, :])
        dsT = dsT.to(qT.dtype)
        dk += tl.dot(dsT, tl.trans(qT))
        curr_m += step_m
        qT_ptrs += step_m * stride_q
        do_ptrs += step_m * stride_do
    return dk, dv


@triton.jit
def _attn_bwd_dq(dq, q, K, V,
                 sm_scale, do, m, D,
                 BLOCK_M2: tl.constexpr,
                 BLOCK_N2: tl.constexpr,
                 HEAD_DIM: tl.constexpr,
                 start_m, start_n, num_steps, stride_kv,
                 N_CTX,
                 MASK: tl.constexpr,
                 EVEN_N_CTX: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, HEAD_DIM)
    kT_ptrs = K + offs_n[None, :] * stride_kv + offs_k[:, None]
    vT_ptrs = V + offs_n[None, :] * stride_kv + offs_k[:, None]
    if EVEN_N_CTX:
        Di = tl.load(D + offs_m)
    else:
        mask_m = offs_m < N_CTX
        Di = tl.load(D + offs_m, mask=mask_m, other=0.0)
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2
    qk_scale = sm_scale * 1.44269504
    for _ in range(num_steps):
        if EVEN_N_CTX:
            kT = tl.load(kT_ptrs)
            vT = tl.load(vT_ptrs)
        else:
            offs_n = curr_n + tl.arange(0, BLOCK_N2)
            mask_n = offs_n < N_CTX
            kT = tl.load(kT_ptrs, mask=mask_n[None, :], other=0.0)
            vT = tl.load(vT_ptrs, mask=mask_n[None, :], other=0.0)
        qk = tl.dot(q, kT) * qk_scale
        p = tl.math.exp2(qk - m)
        if not EVEN_N_CTX:
            p = tl.where(mask_n[None, :], p, 0.0)
        if MASK:
            offs_n = curr_n + tl.arange(0, BLOCK_N2)
            mask = (offs_m[:, None] >= offs_n[None, :])
            p = tl.where(mask, p, 0.0)
        dp = tl.dot(do, vT).to(tl.float32)
        ds = p * (dp - Di[:, None])
        ds = ds.to(q.dtype)
        dq += tl.dot(ds, tl.trans(kT))
        curr_n += step_n
        kT_ptrs += step_n * stride_kv
        vT_ptrs += step_n * stride_kv
    return dq


@triton.jit
def mha_attn_bwd(
        Q, K, V, O, sm_scale,
        DO,
        DQ, DK, DV,
        M, D,
        stride_q, stride_do,
        B, H, N_CTX,
        BLOCK_M1: tl.constexpr,
        BLOCK_N1: tl.constexpr,
        BLOCK_M2: tl.constexpr,
        BLOCK_N2: tl.constexpr,
        BLK_SLICE_FACTOR: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        CAUSAL: tl.constexpr,
        BLOCK_SIZE_X: tl.constexpr,
        BLOCK_SIZE_Y: tl.constexpr,
        EVEN_N_CTX: tl.constexpr,
):
    tl.static_assert(BLOCK_SIZE_X == BLOCK_N1)
    tl.static_assert(BLOCK_SIZE_Y == BLOCK_M2)
    bhid = tl.program_id(2)
    pid = tl.program_id(0)

    off_hz = bhid.to(tl.int64)
    off_z = off_hz // H
    off_h = off_hz % H
    base_q = off_z * (N_CTX * stride_q) + off_h * HEAD_DIM
    base_do = off_z * (N_CTX * stride_do) + off_h * HEAD_DIM

    Q += base_q
    K += base_q
    V += base_q
    DO += base_do
    DQ += base_q
    DK += base_q
    DV += base_q
    M += off_hz * N_CTX
    D += off_hz * N_CTX

    offs_k = tl.arange(0, HEAD_DIM)

    start_n = pid * BLOCK_N1
    start_m = 0

    MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
    offs_n = start_n + tl.arange(0, BLOCK_N1)

    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

    if EVEN_N_CTX:
        k = tl.load(K + offs_n[:, None] * stride_q + offs_k[None, :])
        v = tl.load(V + offs_n[:, None] * stride_q + offs_k[None, :])
    else:
        mask_n = offs_n < N_CTX
        k = tl.load(K + offs_n[:, None] * stride_q + offs_k[None, :], mask=mask_n[:, None], other=0.0)
        v = tl.load(V + offs_n[:, None] * stride_q + offs_k[None, :], mask=mask_n[:, None], other=0.0)

    if CAUSAL:
        start_m = start_n
        num_steps = BLOCK_N1 // MASK_BLOCK_M1
        dk, dv = _attn_bwd_dkdv(
            dk, dv,
            Q, k, v, sm_scale,
            DO,
            M, D,
            MASK_BLOCK_M1, BLOCK_N1, HEAD_DIM,
            start_n, start_m, num_steps,
            stride_do, stride_q,
            N_CTX,
            MASK=True,
            EVEN_N_CTX=EVEN_N_CTX,
        )
        start_m += num_steps * MASK_BLOCK_M1

    if EVEN_N_CTX:
        num_steps = (N_CTX - start_m) // BLOCK_M1
    else:
        num_steps = tl.cdiv(N_CTX - start_m, BLOCK_M1)
    dk, dv = _attn_bwd_dkdv(
        dk, dv,
        Q, k, v, sm_scale,
        DO,
        M, D,
        BLOCK_M1, BLOCK_N1, HEAD_DIM,
        start_n, start_m, num_steps,
        stride_do, stride_q,
        N_CTX,
        MASK=False,
        EVEN_N_CTX=EVEN_N_CTX,
    )

    dv_ptrs = DV + offs_n[:, None] * stride_q + offs_k[None, :]
    if EVEN_N_CTX:
        tl.store(dv_ptrs, dv)
    else:
        mask_n = offs_n < N_CTX
        tl.store(dv_ptrs, dv, mask=mask_n[:, None])

    dk *= sm_scale
    dk_ptrs = DK + offs_n[:, None] * stride_q + offs_k[None, :]
    if EVEN_N_CTX:
        tl.store(dk_ptrs, dk)
    else:
        mask_n = offs_n < N_CTX
        tl.store(dk_ptrs, dk, mask=mask_n[:, None])

    start_m = pid * BLOCK_M2
    start_n = 0
    if EVEN_N_CTX:
        num_steps = N_CTX // BLOCK_N2
    else:
        num_steps = tl.cdiv(N_CTX, BLOCK_N2)

    MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
    offs_m = start_m + tl.arange(0, BLOCK_M2)

    if EVEN_N_CTX:
        q = tl.load(Q + offs_m[:, None] * stride_q + offs_k[None, :])
        do = tl.load(DO + offs_m[:, None] * stride_do + offs_k[None, :])
        m = tl.load(M + offs_m)
    else:
        mask_m = offs_m < N_CTX
        q = tl.load(Q + offs_m[:, None] * stride_q + offs_k[None, :], mask=mask_m[:, None], other=0.0)
        do = tl.load(DO + offs_m[:, None] * stride_do + offs_k[None, :], mask=mask_m[:, None], other=0.0)
        m = tl.load(M + offs_m, mask=mask_m, other=0.0)
    dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)

    m = m[:, None]

    if CAUSAL:
        end_n = start_m + BLOCK_M2
        num_steps = BLOCK_M2 // MASK_BLOCK_N2
        dq = _attn_bwd_dq(
            dq, q, K, V,
            sm_scale, do, m, D,
            BLOCK_M2, MASK_BLOCK_N2, HEAD_DIM,
            start_m, end_n - num_steps * MASK_BLOCK_N2, num_steps, stride_q,
            N_CTX,
            MASK=True,
            EVEN_N_CTX=EVEN_N_CTX,
        )
        end_n -= num_steps * MASK_BLOCK_N2
        num_steps = end_n // BLOCK_N2
        start_n = end_n - num_steps * BLOCK_N2

    dq = _attn_bwd_dq(
        dq, q, K, V,
        sm_scale, do, m, D,
        BLOCK_M2, BLOCK_N2, HEAD_DIM,
        start_m, start_n, num_steps, stride_q,
        N_CTX,
        MASK=False,
        EVEN_N_CTX=EVEN_N_CTX,
    )
    dq_ptrs = DQ + offs_m[:, None] * stride_q + offs_k[None, :]
    dq *= sm_scale
    if EVEN_N_CTX:
        tl.store(dq_ptrs, dq)
    else:
        mask_m = offs_m < N_CTX
        tl.store(dq_ptrs, dq, mask=mask_m[:, None])
