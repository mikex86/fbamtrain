import math
import triton
import triton.language as tl


@triton.jit
def reduce_sum_partial(out_ptr, in_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    vals = tl.load(in_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    acc = tl.sum(vals, axis=0)
    tl.store(out_ptr + pid, acc)


def launch_reduce_sum_partial(x, block_size=256):
    if not x.is_contiguous():
        raise ValueError("reduce_sum_partial expects contiguous input")
    n = x.numel()
    out_len = (n + block_size - 1) // block_size
    out = x.new_empty((out_len,), dtype=triton.language.float32)
    grid = (out_len,)
    reduce_sum_partial[grid](out, x, n, BLOCK_SIZE=block_size)
    return out
