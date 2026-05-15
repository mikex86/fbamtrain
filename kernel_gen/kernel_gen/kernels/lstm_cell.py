import triton
import triton.language as tl


@triton.jit
def lstm_cell_fwd_out_fp16(
    gates_ptr,  # fp16 [B, 4H]
    c_prev_ptr,  # fp16 [B, H]
    h_ptr,  # fp16 [B, H] (state layout)
    c_out_ptr,  # fp16 [B, H]
    y_ptr,  # fp16 [B, H] (output layout)
    batch_size: tl.constexpr,
    hidden_size: tl.constexpr,
    gate_stride,
    c_prev_stride,
    h_stride,
    c_out_stride,
    y_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = batch_size * hidden_size
    mask = offs < total

    batch = offs // hidden_size
    hid = offs % hidden_size

    gate_base = batch * gate_stride + hid

    gate_i = tl.load(gates_ptr + gate_base + 0 * hidden_size, mask=mask, other=0.0).to(tl.float32)
    gate_f = tl.load(gates_ptr + gate_base + 1 * hidden_size, mask=mask, other=0.0).to(tl.float32)
    gate_g = tl.load(gates_ptr + gate_base + 2 * hidden_size, mask=mask, other=0.0).to(tl.float32)
    gate_o = tl.load(gates_ptr + gate_base + 3 * hidden_size, mask=mask, other=0.0).to(tl.float32)

    c_prev = tl.load(c_prev_ptr + batch * c_prev_stride + hid, mask=mask, other=0.0).to(tl.float32)

    i = tl.sigmoid(gate_i).to(tl.float16)
    f = tl.sigmoid(gate_f).to(tl.float16)
    g = (2.0 * tl.sigmoid(gate_g * 2.0) - 1.0).to(tl.float16)
    o = tl.sigmoid(gate_o).to(tl.float16)

    c_prev_f16 = c_prev.to(tl.float16)
    c_new_f16 = f * c_prev_f16 + i * g
    tanh_c_f16 = (2.0 * tl.sigmoid(c_new_f16.to(tl.float32) * 2.0) - 1.0).to(tl.float16)
    h_new = o * tanh_c_f16

    tl.store(c_out_ptr + batch * c_out_stride + hid, c_new_f16, mask=mask)
    tl.store(h_ptr + batch * h_stride + hid, h_new, mask=mask)
    tl.store(y_ptr + batch * y_stride + hid, h_new, mask=mask)


@triton.jit
def lstm_cell_fwd_fp32_state_out_fp16(
    gates_ptr,  # fp32 [B, 4H]
    c_prev_ptr,  # fp32 [B, H]
    h_ptr,  # fp16 [B, H] (state layout)
    c_out_ptr,  # fp32 [B, H]
    y_ptr,  # fp16 [B, H] (output layout)
    batch_size: tl.constexpr,
    hidden_size: tl.constexpr,
    gate_stride,
    c_prev_stride,
    h_stride,
    c_out_stride,
    y_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = batch_size * hidden_size
    mask = offs < total

    batch = offs // hidden_size
    hid = offs % hidden_size

    gate_base = batch * gate_stride + hid

    gate_i = tl.load(gates_ptr + gate_base + 0 * hidden_size, mask=mask, other=0.0)
    gate_f = tl.load(gates_ptr + gate_base + 1 * hidden_size, mask=mask, other=0.0)
    gate_g = tl.load(gates_ptr + gate_base + 2 * hidden_size, mask=mask, other=0.0)
    gate_o = tl.load(gates_ptr + gate_base + 3 * hidden_size, mask=mask, other=0.0)

    c_prev = tl.load(c_prev_ptr + batch * c_prev_stride + hid, mask=mask, other=0.0)

    i = tl.sigmoid(gate_i)
    f = tl.sigmoid(gate_f)
    g = 2.0 * tl.sigmoid(gate_g * 2.0) - 1.0
    o = tl.sigmoid(gate_o)

    c_new = f * c_prev + i * g
    tanh_c = 2.0 * tl.sigmoid(c_new * 2.0) - 1.0
    h_new = o * tanh_c

    tl.store(c_out_ptr + batch * c_out_stride + hid, c_new, mask=mask)
    tl.store(h_ptr + batch * h_stride + hid, h_new.to(tl.float16), mask=mask)
    tl.store(y_ptr + batch * y_stride + hid, h_new.to(tl.float16), mask=mask)


@triton.jit
def lstm_cell_fwd_fp32_state_out_bf16(
    gates_ptr,  # fp32 [B, 4H]
    c_prev_ptr,  # fp32 [B, H]
    h_ptr,  # bf16 [B, H] (state layout)
    c_out_ptr,  # fp32 [B, H]
    y_ptr,  # bf16 [B, H] (output layout)
    batch_size: tl.constexpr,
    hidden_size: tl.constexpr,
    gate_stride,
    c_prev_stride,
    h_stride,
    c_out_stride,
    y_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = batch_size * hidden_size
    mask = offs < total

    batch = offs // hidden_size
    hid = offs % hidden_size

    gate_base = batch * gate_stride + hid

    gate_i = tl.load(gates_ptr + gate_base + 0 * hidden_size, mask=mask, other=0.0)
    gate_f = tl.load(gates_ptr + gate_base + 1 * hidden_size, mask=mask, other=0.0)
    gate_g = tl.load(gates_ptr + gate_base + 2 * hidden_size, mask=mask, other=0.0)
    gate_o = tl.load(gates_ptr + gate_base + 3 * hidden_size, mask=mask, other=0.0)

    c_prev = tl.load(c_prev_ptr + batch * c_prev_stride + hid, mask=mask, other=0.0)

    i = tl.sigmoid(gate_i)
    f = tl.sigmoid(gate_f)
    g = 2.0 * tl.sigmoid(gate_g * 2.0) - 1.0
    o = tl.sigmoid(gate_o)

    c_new = f * c_prev + i * g
    tanh_c = 2.0 * tl.sigmoid(c_new * 2.0) - 1.0
    h_new = o * tanh_c

    tl.store(c_out_ptr + batch * c_out_stride + hid, c_new, mask=mask)
    tl.store(h_ptr + batch * h_stride + hid, h_new.to(tl.bfloat16), mask=mask)
    tl.store(y_ptr + batch * y_stride + hid, h_new.to(tl.bfloat16), mask=mask)


__all__ = [
    "lstm_cell_fwd_out_fp16",
    "lstm_cell_fwd_fp32_state_out_fp16",
    "lstm_cell_fwd_fp32_state_out_bf16",
]
