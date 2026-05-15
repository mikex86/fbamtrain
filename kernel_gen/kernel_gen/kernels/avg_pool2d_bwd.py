import triton
import triton.language as tl


@triton.jit
def avg_pool2d_bwd_kernel(
    upstream_ptr,
    grad_input_ptr,
    outer_stride_upstream0,
    outer_stride_upstream1,
    pool_stride_upstream_h,
    pool_stride_upstream_w,
    outer_stride_grad0,
    outer_stride_grad1,
    pool_stride_grad_h,
    pool_stride_grad_w,
    outer_size0,
    outer_size1,
    pool_h_in,
    pool_w_in,
    pool_h_out,
    pool_w_out,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    inv_kernel_size,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    MAX_KERNEL_H: tl.constexpr,
    MAX_KERNEL_W: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_w = tl.cdiv(pool_w_out, BLOCK_SIZE_W)
    grid_h = tl.cdiv(pool_h_out, BLOCK_SIZE_H)
    tiles_per_outer = grid_w * grid_h
    pid_outer = pid // tiles_per_outer
    tile_id = pid - pid_outer * tiles_per_outer
    pid_h = tile_id // grid_w
    pid_w = tile_id - pid_h * grid_w

    outer_total = outer_size0 * outer_size1
    if pid_outer >= outer_total:
        return

    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_w = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)

    mask_h = offs_h < pool_h_out
    mask_w = offs_w < pool_w_out
    mask_tile = mask_h[:, None] & mask_w[None, :]

    idx_outer0 = pid_outer // outer_size1
    idx_outer1 = pid_outer % outer_size1

    idx_outer0_i64 = idx_outer0.to(tl.int64)
    idx_outer1_i64 = idx_outer1.to(tl.int64)
    outer_stride_upstream0_i64 = outer_stride_upstream0.to(tl.int64)
    outer_stride_upstream1_i64 = outer_stride_upstream1.to(tl.int64)
    outer_stride_grad0_i64 = outer_stride_grad0.to(tl.int64)
    outer_stride_grad1_i64 = outer_stride_grad1.to(tl.int64)

    base_upstream = (
        upstream_ptr + idx_outer0_i64 * outer_stride_upstream0_i64 + idx_outer1_i64 * outer_stride_upstream1_i64
    )
    base_grad = grad_input_ptr + idx_outer0_i64 * outer_stride_grad0_i64 + idx_outer1_i64 * outer_stride_grad1_i64

    pool_stride_upstream_h_i64 = pool_stride_upstream_h.to(tl.int64)
    pool_stride_upstream_w_i64 = pool_stride_upstream_w.to(tl.int64)
    pool_stride_grad_h_i64 = pool_stride_grad_h.to(tl.int64)
    pool_stride_grad_w_i64 = pool_stride_grad_w.to(tl.int64)
    offs_h_i64 = tl.broadcast_to(offs_h[:, None], mask_tile.shape).to(tl.int64)
    offs_w_i64 = tl.broadcast_to(offs_w[None, :], mask_tile.shape).to(tl.int64)

    upstream_ptrs = base_upstream + offs_h_i64 * pool_stride_upstream_h_i64 + offs_w_i64 * pool_stride_upstream_w_i64
    upstream = tl.load(upstream_ptrs, mask=mask_tile, other=0.0).to(tl.float32)
    scaled = upstream * inv_kernel_size

    start_h = offs_h * stride_h - padding_h
    start_w = offs_w * stride_w - padding_w

    for kh in tl.static_range(MAX_KERNEL_H):
        mask_kh = kh < kernel_h
        h_idx = start_h + kh
        for kw in tl.static_range(MAX_KERNEL_W):
            mask_kw = kw < kernel_w
            w_idx = start_w + kw

            h_idx_2d = tl.broadcast_to(h_idx[:, None], mask_tile.shape)
            w_idx_2d = tl.broadcast_to(w_idx[None, :], mask_tile.shape)

            valid_h = (h_idx_2d >= 0) & (h_idx_2d < pool_h_in)
            valid_w = (w_idx_2d >= 0) & (w_idx_2d < pool_w_in)

            mask = mask_tile & valid_h & valid_w & mask_kh & mask_kw
            h_safe = tl.where(mask, h_idx_2d, 0)
            w_safe = tl.where(mask, w_idx_2d, 0)

            grad_ptrs = (
                base_grad
                + h_safe.to(tl.int64) * pool_stride_grad_h_i64
                + w_safe.to(tl.int64) * pool_stride_grad_w_i64
            )
            tl.atomic_add(grad_ptrs, scaled.to(grad_input_ptr.dtype.element_ty), mask=mask)


@triton.jit
def avg_pool2d_bwd_noaccum_kernel(
    upstream_ptr,
    grad_input_ptr,
    outer_stride_upstream0,
    outer_stride_upstream1,
    pool_stride_upstream_h,
    pool_stride_upstream_w,
    outer_stride_grad0,
    outer_stride_grad1,
    pool_stride_grad_h,
    pool_stride_grad_w,
    outer_size0,
    outer_size1,
    pool_h_in,
    pool_w_in,
    pool_h_out,
    pool_w_out,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    inv_kernel_size,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    MAX_KERNEL_H: tl.constexpr,
    MAX_KERNEL_W: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_w = tl.cdiv(pool_w_out, BLOCK_SIZE_W)
    grid_h = tl.cdiv(pool_h_out, BLOCK_SIZE_H)
    tiles_per_outer = grid_w * grid_h
    pid_outer = pid // tiles_per_outer
    tile_id = pid - pid_outer * tiles_per_outer
    pid_h = tile_id // grid_w
    pid_w = tile_id - pid_h * grid_w

    outer_total = outer_size0 * outer_size1
    if pid_outer >= outer_total:
        return

    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_w = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)

    mask_h = offs_h < pool_h_out
    mask_w = offs_w < pool_w_out
    mask_tile = mask_h[:, None] & mask_w[None, :]

    idx_outer0 = pid_outer // outer_size1
    idx_outer1 = pid_outer % outer_size1

    idx_outer0_i64 = idx_outer0.to(tl.int64)
    idx_outer1_i64 = idx_outer1.to(tl.int64)
    outer_stride_upstream0_i64 = outer_stride_upstream0.to(tl.int64)
    outer_stride_upstream1_i64 = outer_stride_upstream1.to(tl.int64)
    outer_stride_grad0_i64 = outer_stride_grad0.to(tl.int64)
    outer_stride_grad1_i64 = outer_stride_grad1.to(tl.int64)

    base_upstream = (
        upstream_ptr + idx_outer0_i64 * outer_stride_upstream0_i64 + idx_outer1_i64 * outer_stride_upstream1_i64
    )
    base_grad = grad_input_ptr + idx_outer0_i64 * outer_stride_grad0_i64 + idx_outer1_i64 * outer_stride_grad1_i64

    pool_stride_upstream_h_i64 = pool_stride_upstream_h.to(tl.int64)
    pool_stride_upstream_w_i64 = pool_stride_upstream_w.to(tl.int64)
    pool_stride_grad_h_i64 = pool_stride_grad_h.to(tl.int64)
    pool_stride_grad_w_i64 = pool_stride_grad_w.to(tl.int64)
    offs_h_i64 = tl.broadcast_to(offs_h[:, None], mask_tile.shape).to(tl.int64)
    offs_w_i64 = tl.broadcast_to(offs_w[None, :], mask_tile.shape).to(tl.int64)

    upstream_ptrs = base_upstream + offs_h_i64 * pool_stride_upstream_h_i64 + offs_w_i64 * pool_stride_upstream_w_i64
    upstream = tl.load(upstream_ptrs, mask=mask_tile, other=0.0).to(tl.float32)
    scaled = upstream * inv_kernel_size

    start_h = offs_h * stride_h - padding_h
    start_w = offs_w * stride_w - padding_w

    for kh in tl.static_range(MAX_KERNEL_H):
        mask_kh = kh < kernel_h
        h_idx = start_h + kh
        for kw in tl.static_range(MAX_KERNEL_W):
            mask_kw = kw < kernel_w
            w_idx = start_w + kw

            h_idx_2d = tl.broadcast_to(h_idx[:, None], mask_tile.shape)
            w_idx_2d = tl.broadcast_to(w_idx[None, :], mask_tile.shape)

            valid_h = (h_idx_2d >= 0) & (h_idx_2d < pool_h_in)
            valid_w = (w_idx_2d >= 0) & (w_idx_2d < pool_w_in)

            mask = mask_tile & valid_h & valid_w & mask_kh & mask_kw
            h_safe = tl.where(mask, h_idx_2d, 0)
            w_safe = tl.where(mask, w_idx_2d, 0)

            grad_ptrs = (
                base_grad
                + h_safe.to(tl.int64) * pool_stride_grad_h_i64
                + w_safe.to(tl.int64) * pool_stride_grad_w_i64
            )
            tl.store(grad_ptrs, scaled.to(grad_input_ptr.dtype.element_ty), mask=mask)


@triton.jit
def avg_pool2d_bwd_noaccum_nhwc_2x2_s2_kernel(
    upstream_ptr,
    grad_input_ptr,
    outer_stride_upstream0,
    outer_stride_upstream1,
    pool_stride_upstream_h,
    pool_stride_upstream_w,
    outer_stride_grad0,
    outer_stride_grad1,
    pool_stride_grad_h,
    pool_stride_grad_w,
    outer_size0,
    outer_size1,
    pool_h_in,
    pool_w_in,
    pool_h_out,
    pool_w_out,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    inv_kernel_size,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    MAX_KERNEL_H: tl.constexpr,
    MAX_KERNEL_W: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    total_hw = pool_h_out * pool_w_out
    blocks_hw = tl.cdiv(total_hw, BLOCK_SIZE_H)
    blocks_c = tl.cdiv(outer_size1, BLOCK_SIZE_W)
    tiles_per_batch = blocks_hw * blocks_c

    n = pid // tiles_per_batch
    tile_id = pid - n * tiles_per_batch
    hw_block = tile_id // blocks_c
    c_block = tile_id - hw_block * blocks_c

    offs_hw = hw_block * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_c = c_block * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)

    oh = offs_hw // pool_w_out
    ow = offs_hw - oh * pool_w_out

    mask = (n < outer_size0) & (offs_hw[:, None] < total_hw) & (offs_c[None, :] < outer_size1)

    n_i64 = n.to(tl.int64)
    oh_i64 = tl.broadcast_to(oh[:, None], mask.shape).to(tl.int64)
    ow_i64 = tl.broadcast_to(ow[:, None], mask.shape).to(tl.int64)
    c_i64 = tl.broadcast_to(offs_c[None, :], mask.shape).to(tl.int64)

    outer_stride_upstream0_i64 = outer_stride_upstream0.to(tl.int64)
    outer_stride_upstream1_i64 = outer_stride_upstream1.to(tl.int64)
    pool_stride_upstream_h_i64 = pool_stride_upstream_h.to(tl.int64)
    pool_stride_upstream_w_i64 = pool_stride_upstream_w.to(tl.int64)
    outer_stride_grad0_i64 = outer_stride_grad0.to(tl.int64)
    outer_stride_grad1_i64 = outer_stride_grad1.to(tl.int64)
    pool_stride_grad_h_i64 = pool_stride_grad_h.to(tl.int64)
    pool_stride_grad_w_i64 = pool_stride_grad_w.to(tl.int64)

    upstream_ptrs = (
        upstream_ptr
        + n_i64 * outer_stride_upstream0_i64
        + oh_i64 * pool_stride_upstream_h_i64
        + ow_i64 * pool_stride_upstream_w_i64
        + c_i64 * outer_stride_upstream1_i64
    )
    scaled = tl.load(upstream_ptrs, mask=mask, other=0.0).to(tl.float32) * 0.25

    grad_base = (
        grad_input_ptr
        + n_i64 * outer_stride_grad0_i64
        + (oh_i64 * 2) * pool_stride_grad_h_i64
        + (ow_i64 * 2) * pool_stride_grad_w_i64
        + c_i64 * outer_stride_grad1_i64
    )
    grad = scaled.to(grad_input_ptr.dtype.element_ty)
    tl.store(grad_base, grad, mask=mask)
    tl.store(grad_base + pool_stride_grad_w_i64, grad, mask=mask)
    tl.store(grad_base + pool_stride_grad_h_i64, grad, mask=mask)
    tl.store(grad_base + pool_stride_grad_h_i64 + pool_stride_grad_w_i64, grad, mask=mask)
