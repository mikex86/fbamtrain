import triton
import triton.language as tl


@triton.jit
def fill_constant(
    out_ptr,
    n_elements: tl.uint32,
    value: tl.float32,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    vals = tl.full((BLOCK_SIZE,), value, out_ptr.dtype.element_ty)
    tl.store(out_ptr + offs, vals, mask=mask)
