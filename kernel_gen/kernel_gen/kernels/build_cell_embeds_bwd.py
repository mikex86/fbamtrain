import triton
import triton.language as tl


@triton.jit
def build_cell_embeds_bwd_cp(
        grad_out_ptr,  # *T, shape (N, D)
        cell_states_ptr,  # *u32, shape (N, 3)
        grad_cp_ptr,  # *T, shape (V, D)
        n_cells: tl.uint32,
        embed_dim: tl.uint32,
        vocab_size: tl.uint32,
        stride_out_n: tl.uint32,
        stride_out_d: tl.uint32,
        stride_cp_v: tl.uint32,
        stride_cp_d: tl.uint32,
        BLOCK_SIZE: tl.constexpr,
):
    out_dtype = grad_cp_ptr.dtype.element_ty

    pid = tl.program_id(axis=0)
    linear = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    linear = linear.to(tl.uint32)

    total = n_cells * embed_dim
    mask = linear < total

    cell_idx = linear // embed_dim
    d_off = linear - cell_idx * embed_dim
    tl.multiple_of(d_off, 16)
    tl.max_contiguous(d_off, 128)

    base = cell_states_ptr + cell_idx * 3
    cp_raw = tl.load(base + 0, mask=mask, other=0).to(tl.int32)
    cp_valid = cp_raw < vocab_size
    cp_idx = tl.minimum(cp_raw, vocab_size - 1)

    grad_ptrs = grad_out_ptr + cell_idx * stride_out_n + d_off * stride_out_d
    grad_vals = tl.load(grad_ptrs, mask=mask, other=0.0).to(tl.float32)

    out_ptrs = grad_cp_ptr + cp_idx * stride_cp_v + d_off * stride_cp_d
    tl.atomic_add(out_ptrs, grad_vals.to(out_dtype), mask=mask & cp_valid)


@triton.jit
def build_cell_embeds_bwd_cp_out_fp32(
        grad_out_ptr,  # *T, shape (N, D)
        cell_states_ptr,  # *u32, shape (N, 3)
        grad_cp_ptr,  # *fp32, shape (V, D)
        n_cells: tl.uint32,
        embed_dim: tl.uint32,
        vocab_size: tl.uint32,
        stride_out_n: tl.uint32,
        stride_out_d: tl.uint32,
        stride_cp_v: tl.uint32,
        stride_cp_d: tl.uint32,
        BLOCK_SIZE: tl.constexpr,
):
    out_dtype = grad_cp_ptr.dtype.element_ty

    pid = tl.program_id(axis=0)
    linear = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    linear = linear.to(tl.uint32)

    total = n_cells * embed_dim
    mask = linear < total

    cell_idx = linear // embed_dim
    d_off = linear - cell_idx * embed_dim
    tl.multiple_of(d_off, 16)
    tl.max_contiguous(d_off, 128)

    base = cell_states_ptr + cell_idx * 3
    cp_raw = tl.load(base + 0, mask=mask, other=0).to(tl.int32)
    cp_valid = cp_raw < vocab_size
    cp_idx = tl.minimum(cp_raw, vocab_size - 1)

    grad_ptrs = grad_out_ptr + cell_idx * stride_out_n + d_off * stride_out_d
    grad_vals = tl.load(grad_ptrs, mask=mask, other=0.0).to(tl.float32)

    out_ptrs = grad_cp_ptr + cp_idx * stride_cp_v + d_off * stride_cp_d
    tl.atomic_add(out_ptrs, grad_vals.to(out_dtype), mask=mask & cp_valid)


@triton.jit
def build_cell_embeds_bwd_pos(
        grad_out_ptr,  # *T, shape (N, D)
        grad_pos_ptr,  # *T, shape (P, D)
        batch: tl.uint32,
        max_positions: tl.uint32,
        embed_dim: tl.uint32,
        stride_out_n: tl.uint32,
        stride_out_d: tl.uint32,
        stride_pos_p: tl.uint32,
        stride_pos_d: tl.uint32,
        BLOCK_SIZE_X: tl.constexpr,
        BLOCK_SIZE_Y: tl.constexpr,
):
    out_dtype = grad_pos_ptr.dtype.element_ty

    # Flatten (P, D) program ids into a single launch dimension to avoid grid Y limits.
    pid = tl.program_id(axis=0).to(tl.uint32)
    grid_pos = max_positions
    grid_d = tl.cdiv(embed_dim, BLOCK_SIZE_X).to(tl.uint32)
    pid_pos = pid % grid_pos
    pid_d = pid // grid_pos

    if pid_pos >= max_positions or pid_d >= grid_d:
        return

    offs_d = pid_d * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    offs_d = offs_d.to(tl.uint32)
    mask_d = offs_d < embed_dim
    tl.multiple_of(offs_d, 16)
    tl.max_contiguous(offs_d, 128)

    acc = tl.zeros([BLOCK_SIZE_X], dtype=tl.float32)
    for b_start in tl.range(0, batch, BLOCK_SIZE_Y):
        offs_b = b_start + tl.arange(0, BLOCK_SIZE_Y)
        offs_b = offs_b.to(tl.uint32)
        mask_b = offs_b < batch
        cell_idx = offs_b * max_positions + pid_pos
        ptrs = grad_out_ptr + cell_idx[:, None] * stride_out_n + offs_d[None, :] * stride_out_d
        mask = mask_b[:, None] & mask_d[None, :]
        vals = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(vals, axis=0)

    out_ptrs = grad_pos_ptr + pid_pos * stride_pos_p + offs_d * stride_pos_d
    prev = tl.load(out_ptrs, mask=mask_d, other=0.0).to(tl.float32)
    tl.store(out_ptrs, (acc + prev).to(out_dtype), mask=mask_d)


@triton.jit
def build_cell_embeds_bwd_pos_out_fp32(
        grad_out_ptr,  # *T, shape (N, D)
        grad_pos_ptr,  # *fp32, shape (P, D)
        batch: tl.uint32,
        max_positions: tl.uint32,
        embed_dim: tl.uint32,
        stride_out_n: tl.uint32,
        stride_out_d: tl.uint32,
        stride_pos_p: tl.uint32,
        stride_pos_d: tl.uint32,
        BLOCK_SIZE_X: tl.constexpr,
        BLOCK_SIZE_Y: tl.constexpr,
):
    out_dtype = grad_pos_ptr.dtype.element_ty

    # Flatten (P, D) program ids into a single launch dimension to avoid grid Y limits.
    pid = tl.program_id(axis=0).to(tl.uint32)
    grid_pos = max_positions
    grid_d = tl.cdiv(embed_dim, BLOCK_SIZE_X).to(tl.uint32)
    pid_pos = pid % grid_pos
    pid_d = pid // grid_pos

    if pid_pos >= max_positions or pid_d >= grid_d:
        return

    offs_d = pid_d * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    offs_d = offs_d.to(tl.uint32)
    mask_d = offs_d < embed_dim
    tl.multiple_of(offs_d, 16)
    tl.max_contiguous(offs_d, 128)

    acc = tl.zeros([BLOCK_SIZE_X], dtype=tl.float32)
    for b_start in tl.range(0, batch, BLOCK_SIZE_Y):
        offs_b = b_start + tl.arange(0, BLOCK_SIZE_Y)
        offs_b = offs_b.to(tl.uint32)
        mask_b = offs_b < batch
        cell_idx = offs_b * max_positions + pid_pos
        ptrs = grad_out_ptr + cell_idx[:, None] * stride_out_n + offs_d[None, :] * stride_out_d
        mask = mask_b[:, None] & mask_d[None, :]
        vals = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(vals, axis=0)

    out_ptrs = grad_pos_ptr + pid_pos * stride_pos_p + offs_d * stride_pos_d
    prev = tl.load(out_ptrs, mask=mask_d, other=0.0).to(tl.float32)
    tl.store(out_ptrs, (acc + prev).to(out_dtype), mask=mask_d)


@triton.jit
def build_cell_embeds_bwd_color(
        grad_out_ptr,  # *T, shape (N, D)
        cell_states_ptr,  # *u32, shape (N, 3)
        grad_fg_r_ptr, grad_fg_g_ptr, grad_fg_b_ptr,
        grad_bg_r_ptr, grad_bg_g_ptr, grad_bg_b_ptr,
        n_cells: tl.uint32,
        embed_dim: tl.uint32,
        stride_out_n: tl.uint32,
        stride_out_d: tl.uint32,
        BLOCK_SIZE_X: tl.constexpr,
        BLOCK_SIZE_Y: tl.constexpr,
):
    out_dtype = grad_fg_r_ptr.dtype.element_ty

    # Flatten (D, N) program ids into a single launch dimension to avoid grid Y limits.
    pid = tl.program_id(axis=0).to(tl.uint32)
    grid_d = tl.cdiv(embed_dim, BLOCK_SIZE_X).to(tl.uint32)
    grid_n = tl.cdiv(n_cells, BLOCK_SIZE_Y).to(tl.uint32)
    pid_n = pid // grid_d
    pid_d = pid - pid_n * grid_d

    if pid_n >= grid_n:
        return

    offs_d = pid_d * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    offs_d = offs_d.to(tl.uint32)
    mask_d = offs_d < embed_dim
    tl.multiple_of(offs_d, 16)
    tl.max_contiguous(offs_d, 128)

    offs_n = pid_n * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    offs_n = offs_n.to(tl.uint32)
    mask_n = offs_n < n_cells

    grad_ptrs = grad_out_ptr + offs_n[:, None] * stride_out_n + offs_d[None, :] * stride_out_d
    grad_vals = tl.load(grad_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    base = cell_states_ptr + offs_n * 3
    fg_rgb = tl.load(base + 1, mask=mask_n, other=0).to(tl.int32)
    bg_rgb = tl.load(base + 2, mask=mask_n, other=0).to(tl.int32)

    inv255 = 1.0 / 255.0
    fg_r = ((fg_rgb >> 16) & 0xFF).to(tl.float32) * inv255
    fg_g = ((fg_rgb >> 8) & 0xFF).to(tl.float32) * inv255
    fg_b = (fg_rgb & 0xFF).to(tl.float32) * inv255
    bg_r = ((bg_rgb >> 16) & 0xFF).to(tl.float32) * inv255
    bg_g = ((bg_rgb >> 8) & 0xFF).to(tl.float32) * inv255
    bg_b = (bg_rgb & 0xFF).to(tl.float32) * inv255

    fg_r_sum = tl.sum(grad_vals * fg_r[:, None], axis=0)
    fg_g_sum = tl.sum(grad_vals * fg_g[:, None], axis=0)
    fg_b_sum = tl.sum(grad_vals * fg_b[:, None], axis=0)
    bg_r_sum = tl.sum(grad_vals * bg_r[:, None], axis=0)
    bg_g_sum = tl.sum(grad_vals * bg_g[:, None], axis=0)
    bg_b_sum = tl.sum(grad_vals * bg_b[:, None], axis=0)

    grad_fg_r_ptrs = grad_fg_r_ptr + offs_d
    grad_fg_g_ptrs = grad_fg_g_ptr + offs_d
    grad_fg_b_ptrs = grad_fg_b_ptr + offs_d
    grad_bg_r_ptrs = grad_bg_r_ptr + offs_d
    grad_bg_g_ptrs = grad_bg_g_ptr + offs_d
    grad_bg_b_ptrs = grad_bg_b_ptr + offs_d

    tl.atomic_add(grad_fg_r_ptrs, fg_r_sum.to(out_dtype), mask=mask_d)
    tl.atomic_add(grad_fg_g_ptrs, fg_g_sum.to(out_dtype), mask=mask_d)
    tl.atomic_add(grad_fg_b_ptrs, fg_b_sum.to(out_dtype), mask=mask_d)
    tl.atomic_add(grad_bg_r_ptrs, bg_r_sum.to(out_dtype), mask=mask_d)
    tl.atomic_add(grad_bg_g_ptrs, bg_g_sum.to(out_dtype), mask=mask_d)
    tl.atomic_add(grad_bg_b_ptrs, bg_b_sum.to(out_dtype), mask=mask_d)


@triton.jit
def build_cell_embeds_bwd_color_out_fp32(
        grad_out_ptr,  # *T, shape (N, D)
        cell_states_ptr,  # *u32, shape (N, 3)
        grad_fg_r_ptr, grad_fg_g_ptr, grad_fg_b_ptr,
        grad_bg_r_ptr, grad_bg_g_ptr, grad_bg_b_ptr,
        n_cells: tl.uint32,
        embed_dim: tl.uint32,
        stride_out_n: tl.uint32,
        stride_out_d: tl.uint32,
        BLOCK_SIZE_X: tl.constexpr,
        BLOCK_SIZE_Y: tl.constexpr,
):
    out_dtype = grad_fg_r_ptr.dtype.element_ty

    # Flatten (D, N) program ids into a single launch dimension to avoid grid Y limits.
    pid = tl.program_id(axis=0).to(tl.uint32)
    grid_d = tl.cdiv(embed_dim, BLOCK_SIZE_X).to(tl.uint32)
    grid_n = tl.cdiv(n_cells, BLOCK_SIZE_Y).to(tl.uint32)
    pid_n = pid // grid_d
    pid_d = pid - pid_n * grid_d

    if pid_n >= grid_n:
        return

    offs_d = pid_d * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    offs_d = offs_d.to(tl.uint32)
    mask_d = offs_d < embed_dim
    tl.multiple_of(offs_d, 16)
    tl.max_contiguous(offs_d, 128)

    offs_n = pid_n * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    offs_n = offs_n.to(tl.uint32)
    mask_n = offs_n < n_cells

    grad_ptrs = grad_out_ptr + offs_n[:, None] * stride_out_n + offs_d[None, :] * stride_out_d
    grad_vals = tl.load(grad_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    base = cell_states_ptr + offs_n * 3
    fg_rgb = tl.load(base + 1, mask=mask_n, other=0).to(tl.int32)
    bg_rgb = tl.load(base + 2, mask=mask_n, other=0).to(tl.int32)

    inv255 = 1.0 / 255.0
    fg_r = ((fg_rgb >> 16) & 0xFF).to(tl.float32) * inv255
    fg_g = ((fg_rgb >> 8) & 0xFF).to(tl.float32) * inv255
    fg_b = (fg_rgb & 0xFF).to(tl.float32) * inv255
    bg_r = ((bg_rgb >> 16) & 0xFF).to(tl.float32) * inv255
    bg_g = ((bg_rgb >> 8) & 0xFF).to(tl.float32) * inv255
    bg_b = (bg_rgb & 0xFF).to(tl.float32) * inv255

    fg_r_sum = tl.sum(grad_vals * fg_r[:, None], axis=0)
    fg_g_sum = tl.sum(grad_vals * fg_g[:, None], axis=0)
    fg_b_sum = tl.sum(grad_vals * fg_b[:, None], axis=0)
    bg_r_sum = tl.sum(grad_vals * bg_r[:, None], axis=0)
    bg_g_sum = tl.sum(grad_vals * bg_g[:, None], axis=0)
    bg_b_sum = tl.sum(grad_vals * bg_b[:, None], axis=0)

    grad_fg_r_ptrs = grad_fg_r_ptr + offs_d
    grad_fg_g_ptrs = grad_fg_g_ptr + offs_d
    grad_fg_b_ptrs = grad_fg_b_ptr + offs_d
    grad_bg_r_ptrs = grad_bg_r_ptr + offs_d
    grad_bg_g_ptrs = grad_bg_g_ptr + offs_d
    grad_bg_b_ptrs = grad_bg_b_ptr + offs_d

    tl.atomic_add(grad_fg_r_ptrs, fg_r_sum.to(out_dtype), mask=mask_d)
    tl.atomic_add(grad_fg_g_ptrs, fg_g_sum.to(out_dtype), mask=mask_d)
    tl.atomic_add(grad_fg_b_ptrs, fg_b_sum.to(out_dtype), mask=mask_d)
    tl.atomic_add(grad_bg_r_ptrs, bg_r_sum.to(out_dtype), mask=mask_d)
    tl.atomic_add(grad_bg_g_ptrs, bg_g_sum.to(out_dtype), mask=mask_d)
    tl.atomic_add(grad_bg_b_ptrs, bg_b_sum.to(out_dtype), mask=mask_d)
