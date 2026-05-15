import triton
import triton.language as tl


@triton.jit
def fill_zeros(
    out_ptr,
    n_bytes: tl.uint32,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_bytes

    zeros = tl.zeros([BLOCK_SIZE], dtype=tl.uint8)
    tl.store(out_ptr + offsets, zeros, mask=mask)
