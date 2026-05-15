import triton
import triton.language as tl


@triton.jit
def device_copy_strided_2d(
    dst_ptr: tl.pointer_type(tl.uint16),
    src_ptr: tl.pointer_type(tl.uint16),
    src_stride_elems: tl.int32,
    dst_stride_elems: tl.int32,
    width_elems: tl.int32,
    height: tl.int32,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)

    rows = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    cols = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)

    row_mask = rows < height
    col_mask = cols < width_elems
    mask = row_mask[:, None] & col_mask[None, :]

    src_offsets = rows[:, None] * src_stride_elems + cols[None, :]
    dst_offsets = rows[:, None] * dst_stride_elems + cols[None, :]

    src_block = src_ptr + src_offsets
    dst_block = dst_ptr + dst_offsets

    values = tl.load(src_block, mask=mask, other=0)
    tl.store(dst_block, values, mask=mask)
