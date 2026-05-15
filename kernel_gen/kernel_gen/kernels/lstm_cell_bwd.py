import triton
import triton.language as tl


@triton.jit
def lstm_cell_recompute_out_fp16(
    gate_pre_ptr,  # fp32 [B, 4H] or fp16
    bias_ptr,  # fp32 [4H]
    c_prev_ptr,  # fp32 [B, H]
    gate_out_ptr,  # fp32 [B, 4H]
    h_out_ptr,  # fp16 [B, H]
    c_out_ptr,  # fp32 [B, H]
    batch_size: tl.constexpr,
    hidden_size: tl.constexpr,
    gate_stride,
    bias_stride,
    c_prev_stride,
    gate_out_stride,
    h_out_stride,
    c_out_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = batch_size * hidden_size
    mask = offs < total

    batch = offs // hidden_size
    hid = offs % hidden_size
    gate_dim = 4 * hidden_size
    gate_base = batch * gate_stride + hid

    gate_i = tl.load(gate_pre_ptr + gate_base + 0 * hidden_size, mask=mask, other=0.0).to(tl.float32)
    gate_f = tl.load(gate_pre_ptr + gate_base + 1 * hidden_size, mask=mask, other=0.0).to(tl.float32)
    gate_g = tl.load(gate_pre_ptr + gate_base + 2 * hidden_size, mask=mask, other=0.0).to(tl.float32)
    gate_o = tl.load(gate_pre_ptr + gate_base + 3 * hidden_size, mask=mask, other=0.0).to(tl.float32)

    # fuse bias
    gate_i += tl.load(bias_ptr + 0 * hidden_size + hid, mask=mask, other=0.0)
    gate_f += tl.load(bias_ptr + 1 * hidden_size + hid, mask=mask, other=0.0)
    gate_g += tl.load(bias_ptr + 2 * hidden_size + hid, mask=mask, other=0.0)
    gate_o += tl.load(bias_ptr + 3 * hidden_size + hid, mask=mask, other=0.0)

    c_prev = tl.load(c_prev_ptr + batch * c_prev_stride + hid, mask=mask, other=0.0).to(tl.float32)

    i = tl.sigmoid(gate_i)
    f = tl.sigmoid(gate_f)
    g = 2.0 * tl.sigmoid(gate_g * 2.0) - 1.0
    o = tl.sigmoid(gate_o)

    c_new = f * c_prev + i * g
    tanh_c = 2.0 * tl.sigmoid(c_new * 2.0) - 1.0
    h_new = o * tanh_c

    # stash gates for backward
    tl.store(gate_out_ptr + batch * gate_out_stride + hid + 0 * hidden_size, i, mask=mask)
    tl.store(gate_out_ptr + batch * gate_out_stride + hid + 1 * hidden_size, f, mask=mask)
    tl.store(gate_out_ptr + batch * gate_out_stride + hid + 2 * hidden_size, g, mask=mask)
    tl.store(gate_out_ptr + batch * gate_out_stride + hid + 3 * hidden_size, o, mask=mask)

    tl.store(c_out_ptr + batch * c_out_stride + hid, c_new, mask=mask)
    tl.store(h_out_ptr + batch * h_out_stride + hid, h_new.to(tl.float16), mask=mask)


@triton.jit
def lstm_cell_bwd_pointwise_out_fp16(
    dY_ptr,  # fp16 [B, H]
    dh_next_ptr,  # fp32 [B, H]
    dc_next_ptr,  # fp32 [B, H]
    gate_out_ptr,  # fp32 [B, 4H] (i,f,g,o)
    c_prev_ptr,  # fp32 [B, H]
    c_out_ptr,  # fp32 [B, H]
    dGates_ptr,  # fp32 [B, 4H]
    dGates_half_ptr,  # fp16 [B, 4H]
    dc_prev_ptr,  # fp32 [B, H]
    batch_size: tl.constexpr,
    hidden_size: tl.constexpr,
    gate_stride,
    gate_out_stride,
    state_stride,
    c_prev_stride,
    c_out_stride,
    dGates_stride,
    dc_prev_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = batch_size * hidden_size
    mask = offs < total

    batch = offs // hidden_size
    hid = offs % hidden_size
    gate_dim = 4 * hidden_size

    dh_total = tl.load(dY_ptr + batch * state_stride + hid, mask=mask, other=0.0).to(tl.float32)
    dh_total += tl.load(dh_next_ptr + batch * state_stride + hid, mask=mask, other=0.0).to(tl.float32)
    dc_next = tl.load(dc_next_ptr + batch * dc_prev_stride + hid, mask=mask, other=0.0).to(tl.float32)

    gate_base = batch * gate_out_stride + hid
    i_gate = tl.load(gate_out_ptr + gate_base + 0 * hidden_size, mask=mask, other=0.0)
    f_gate = tl.load(gate_out_ptr + gate_base + 1 * hidden_size, mask=mask, other=0.0)
    g_gate = tl.load(gate_out_ptr + gate_base + 2 * hidden_size, mask=mask, other=0.0)
    o_gate = tl.load(gate_out_ptr + gate_base + 3 * hidden_size, mask=mask, other=0.0)

    c_prev = tl.load(c_prev_ptr + batch * c_prev_stride + hid, mask=mask, other=0.0).to(tl.float32)
    c_t = tl.load(c_out_ptr + batch * c_out_stride + hid, mask=mask, other=0.0).to(tl.float32)
    tanh_c = 2.0 * tl.sigmoid(c_t * 2.0) - 1.0

    do_gate = dh_total * tanh_c
    one_minus_tanh_sq = 1.0 - tanh_c * tanh_c
    dc_total = dc_next + dh_total * o_gate * one_minus_tanh_sq

    di = dc_total * g_gate
    df = dc_total * c_prev
    dg = dc_total * i_gate
    dc_prev = dc_total * f_gate

    dai = di * i_gate * (1.0 - i_gate)
    daf = df * f_gate * (1.0 - f_gate)
    dag = dg * (1.0 - g_gate * g_gate)
    dao = do_gate * o_gate * (1.0 - o_gate)

    # write dGates (column-major by gate then hidden)
    base = batch * dGates_stride + hid
    tl.store(dGates_ptr + base + 0 * hidden_size, dai, mask=mask)
    tl.store(dGates_ptr + base + 1 * hidden_size, daf, mask=mask)
    tl.store(dGates_ptr + base + 2 * hidden_size, dag, mask=mask)
    tl.store(dGates_ptr + base + 3 * hidden_size, dao, mask=mask)

    if dGates_half_ptr is not None:
        tl.store(dGates_half_ptr + base + 0 * hidden_size, dai.to(tl.float16), mask=mask)
        tl.store(dGates_half_ptr + base + 1 * hidden_size, daf.to(tl.float16), mask=mask)
        tl.store(dGates_half_ptr + base + 2 * hidden_size, dag.to(tl.float16), mask=mask)
        tl.store(dGates_half_ptr + base + 3 * hidden_size, dao.to(tl.float16), mask=mask)

    tl.store(dc_prev_ptr + batch * dc_prev_stride + hid, dc_prev, mask=mask)


@triton.jit
def lstm_cell_recompute_out_bf16(
    gate_pre_ptr,  # fp32 [B, 4H] or fp16
    bias_ptr,  # fp32 [4H]
    c_prev_ptr,  # fp32 [B, H]
    gate_out_ptr,  # fp32 [B, 4H]
    h_out_ptr,  # bf16 [B, H]
    c_out_ptr,  # fp32 [B, H]
    batch_size: tl.constexpr,
    hidden_size: tl.constexpr,
    gate_stride,
    bias_stride,
    c_prev_stride,
    gate_out_stride,
    h_out_stride,
    c_out_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = batch_size * hidden_size
    mask = offs < total

    batch = offs // hidden_size
    hid = offs % hidden_size
    gate_dim = 4 * hidden_size
    gate_base = batch * gate_stride + hid

    gate_i = tl.load(gate_pre_ptr + gate_base + 0 * hidden_size, mask=mask, other=0.0).to(tl.float32)
    gate_f = tl.load(gate_pre_ptr + gate_base + 1 * hidden_size, mask=mask, other=0.0).to(tl.float32)
    gate_g = tl.load(gate_pre_ptr + gate_base + 2 * hidden_size, mask=mask, other=0.0).to(tl.float32)
    gate_o = tl.load(gate_pre_ptr + gate_base + 3 * hidden_size, mask=mask, other=0.0).to(tl.float32)

    # fuse bias
    gate_i += tl.load(bias_ptr + 0 * hidden_size + hid, mask=mask, other=0.0)
    gate_f += tl.load(bias_ptr + 1 * hidden_size + hid, mask=mask, other=0.0)
    gate_g += tl.load(bias_ptr + 2 * hidden_size + hid, mask=mask, other=0.0)
    gate_o += tl.load(bias_ptr + 3 * hidden_size + hid, mask=mask, other=0.0)

    c_prev = tl.load(c_prev_ptr + batch * c_prev_stride + hid, mask=mask, other=0.0).to(tl.float32)

    i = tl.sigmoid(gate_i)
    f = tl.sigmoid(gate_f)
    g = 2.0 * tl.sigmoid(gate_g * 2.0) - 1.0
    o = tl.sigmoid(gate_o)

    c_new = f * c_prev + i * g
    tanh_c = 2.0 * tl.sigmoid(c_new * 2.0) - 1.0
    h_new = o * tanh_c

    # stash gates for backward
    tl.store(gate_out_ptr + batch * gate_out_stride + hid + 0 * hidden_size, i, mask=mask)
    tl.store(gate_out_ptr + batch * gate_out_stride + hid + 1 * hidden_size, f, mask=mask)
    tl.store(gate_out_ptr + batch * gate_out_stride + hid + 2 * hidden_size, g, mask=mask)
    tl.store(gate_out_ptr + batch * gate_out_stride + hid + 3 * hidden_size, o, mask=mask)

    tl.store(c_out_ptr + batch * c_out_stride + hid, c_new, mask=mask)
    tl.store(h_out_ptr + batch * h_out_stride + hid, h_new.to(tl.bfloat16), mask=mask)


@triton.jit
def lstm_cell_bwd_pointwise_out_bf16(
    dY_ptr,  # fp16 [B, H]
    dh_next_ptr,  # fp32 [B, H]
    dc_next_ptr,  # fp32 [B, H]
    gate_out_ptr,  # fp32 [B, 4H] (i,f,g,o)
    c_prev_ptr,  # fp32 [B, H]
    c_out_ptr,  # fp32 [B, H]
    dGates_ptr,  # fp32 [B, 4H]
    dGates_half_ptr,  # bf16 [B, 4H]
    dc_prev_ptr,  # fp32 [B, H]
    batch_size: tl.constexpr,
    hidden_size: tl.constexpr,
    gate_stride,
    gate_out_stride,
    state_stride,
    c_prev_stride,
    c_out_stride,
    dGates_stride,
    dc_prev_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = batch_size * hidden_size
    mask = offs < total

    batch = offs // hidden_size
    hid = offs % hidden_size
    gate_dim = 4 * hidden_size

    dh_total = tl.load(dY_ptr + batch * state_stride + hid, mask=mask, other=0.0).to(tl.float32)
    dh_total += tl.load(dh_next_ptr + batch * state_stride + hid, mask=mask, other=0.0).to(tl.float32)
    dc_next = tl.load(dc_next_ptr + batch * dc_prev_stride + hid, mask=mask, other=0.0).to(tl.float32)

    gate_base = batch * gate_out_stride + hid
    i_gate = tl.load(gate_out_ptr + gate_base + 0 * hidden_size, mask=mask, other=0.0)
    f_gate = tl.load(gate_out_ptr + gate_base + 1 * hidden_size, mask=mask, other=0.0)
    g_gate = tl.load(gate_out_ptr + gate_base + 2 * hidden_size, mask=mask, other=0.0)
    o_gate = tl.load(gate_out_ptr + gate_base + 3 * hidden_size, mask=mask, other=0.0)

    c_prev = tl.load(c_prev_ptr + batch * c_prev_stride + hid, mask=mask, other=0.0).to(tl.float32)
    c_t = tl.load(c_out_ptr + batch * c_out_stride + hid, mask=mask, other=0.0).to(tl.float32)
    tanh_c = 2.0 * tl.sigmoid(c_t * 2.0) - 1.0

    do_gate = dh_total * tanh_c
    one_minus_tanh_sq = 1.0 - tanh_c * tanh_c
    dc_total = dc_next + dh_total * o_gate * one_minus_tanh_sq

    di = dc_total * g_gate
    df = dc_total * c_prev
    dg = dc_total * i_gate
    dc_prev = dc_total * f_gate

    dai = di * i_gate * (1.0 - i_gate)
    daf = df * f_gate * (1.0 - f_gate)
    dag = dg * (1.0 - g_gate * g_gate)
    dao = do_gate * o_gate * (1.0 - o_gate)

    # write dGates (column-major by gate then hidden)
    base = batch * dGates_stride + hid
    tl.store(dGates_ptr + base + 0 * hidden_size, dai, mask=mask)
    tl.store(dGates_ptr + base + 1 * hidden_size, daf, mask=mask)
    tl.store(dGates_ptr + base + 2 * hidden_size, dag, mask=mask)
    tl.store(dGates_ptr + base + 3 * hidden_size, dao, mask=mask)

    if dGates_half_ptr is not None:
        tl.store(dGates_half_ptr + base + 0 * hidden_size, dai.to(tl.bfloat16), mask=mask)
        tl.store(dGates_half_ptr + base + 1 * hidden_size, daf.to(tl.bfloat16), mask=mask)
        tl.store(dGates_half_ptr + base + 2 * hidden_size, dag.to(tl.bfloat16), mask=mask)
        tl.store(dGates_half_ptr + base + 3 * hidden_size, dao.to(tl.bfloat16), mask=mask)

    tl.store(dc_prev_ptr + batch * dc_prev_stride + hid, dc_prev, mask=mask)


__all__ = [
    "lstm_cell_recompute_out_fp16",
    "lstm_cell_recompute_out_bf16",
    "lstm_cell_bwd_pointwise_out_fp16",
    "lstm_cell_bwd_pointwise_out_bf16",
]
