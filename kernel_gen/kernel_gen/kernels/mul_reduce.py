import math
from typing import Optional

import torch
import triton
import triton.language as tl

_MUL_CONTIG_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 64}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 256}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=2),
]

_MUL_COLUMN_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 32}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_SIZE": 64}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=2),
]

_MUL_COLUMN_SPLIT_CONFIGS = [
    triton.Config({"BLOCK_SIZE_X": 64, "BLOCK_SIZE_Y": 128}, num_warps=8, num_stages=2),
]


@triton.autotune(configs=_MUL_CONTIG_CONFIGS, key=["cols"])
@triton.jit
def mul_reduce_contiguous(
    out_ptr,
    lhs_ptr,
    rhs_ptr,
    outer,
    inner,
    cols,
    BLOCK_SIZE: tl.constexpr,
):
    pid_outer = tl.program_id(axis=0)
    pid_inner = tl.program_id(axis=1)

    if pid_outer >= outer or pid_inner >= inner:
        return

    base_offset = pid_outer * cols * inner + pid_inner
    offsets = tl.arange(0, BLOCK_SIZE)

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for col_start in tl.range(0, cols, BLOCK_SIZE):
        col_idx = col_start + offsets
        mask = col_idx < cols
        ptrs = base_offset + col_idx * inner
        lhs = tl.load(lhs_ptr + ptrs, mask=mask, other=0.0)
        rhs = tl.load(rhs_ptr + ptrs, mask=mask, other=0.0)
        prod = lhs.to(tl.float32) * rhs.to(tl.float32)
        acc += tl.where(mask, prod, 0.0)

    total = tl.sum(acc, axis=0)
    out_dtype = out_ptr.dtype.element_ty
    tl.store(out_ptr + pid_outer * inner + pid_inner, total.to(out_dtype))


@triton.autotune(configs=_MUL_COLUMN_CONFIGS, key=["inner"])
@triton.jit
def mul_reduce_column_tiled(
    out_ptr,
    lhs_ptr,
    rhs_ptr,
    outer,
    inner,
    cols,
    BLOCK_SIZE: tl.constexpr,
):
    pid_outer = tl.program_id(axis=0)
    pid_inner = tl.program_id(axis=1)

    if pid_outer >= outer:
        return

    inner_start = pid_inner * BLOCK_SIZE
    inner_offsets = inner_start + tl.arange(0, BLOCK_SIZE)
    inner_mask = inner_offsets < inner

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for col_start in tl.range(0, cols, 32):
        col_offsets = col_start + tl.arange(0, 32)
        col_mask = col_offsets < cols
        ptrs = pid_outer * cols * inner + col_offsets[:, None] * inner + inner_offsets[None, :]
        mask = col_mask[:, None] & inner_mask[None, :]
        lhs = tl.load(lhs_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
        rhs = tl.load(rhs_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(lhs * rhs, axis=0)

    out_dtype = out_ptr.dtype.element_ty
    tl.store(out_ptr + pid_outer * inner + inner_offsets, acc.to(out_dtype), mask=inner_mask)


@triton.autotune(configs=_MUL_COLUMN_SPLIT_CONFIGS, key=["inner", "cols"])
@triton.jit
def mul_reduce_column_split_partials(
    partials_ptr,
    lhs_ptr,
    rhs_ptr,
    outer,
    inner,
    cols,
    split_count,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
):
    pid_outer = tl.program_id(axis=0)
    pid_split = tl.program_id(axis=1)
    pid_inner = tl.program_id(axis=2)

    if pid_outer >= outer or pid_split >= split_count:
        return

    inner_offsets = pid_inner * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    col_offsets = pid_split * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    inner_mask = inner_offsets < inner
    col_mask = col_offsets < cols

    ptrs = pid_outer * cols * inner + col_offsets[:, None] * inner + inner_offsets[None, :]
    mask = col_mask[:, None] & inner_mask[None, :]
    lhs = tl.load(lhs_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
    rhs = tl.load(rhs_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
    acc = tl.sum(lhs * rhs, axis=0)

    out_offsets = pid_outer * split_count * inner + pid_split * inner + inner_offsets
    tl.store(partials_ptr + out_offsets, acc, mask=inner_mask)


def launch_mul_reduce(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    dim: int = -1,
    keepdim: bool = False,
    *,
    block_size: Optional[int] = None,
) -> torch.Tensor:
    if not lhs.is_contiguous() or not rhs.is_contiguous():
        raise ValueError("launch_mul_reduce expects contiguous tensors")
    if lhs.shape != rhs.shape:
        raise ValueError("launch_mul_reduce expects matching shapes")

    ndim = lhs.ndim
    if dim < 0:
        dim += ndim
    if dim < 0 or dim >= ndim:
        raise ValueError("dimension out of range")

    cols = lhs.shape[dim]
    if cols == 0:
        raise ValueError("launch_mul_reduce input must have non-zero size along reduced dimension")
    outer = math.prod(lhs.shape[:dim]) if dim > 0 else 1
    inner = math.prod(lhs.shape[dim + 1 :]) if dim + 1 < ndim else 1
    rows = outer * inner
    if rows == 0:
        raise ValueError("launch_mul_reduce input has zero-sized outer/inner dimensions")

    output_shape = list(lhs.shape)
    if keepdim:
        output_shape[dim] = 1
    else:
        output_shape.pop(dim)
    out = torch.empty(output_shape, dtype=torch.float32, device=lhs.device)

    out_flat = out.view(outer, inner)

    meta = {"BLOCK_SIZE": block_size} if block_size is not None else None
    use_column_tiled = inner != 1
    kernel = (mul_reduce_column_tiled if use_column_tiled else mul_reduce_contiguous)
    if meta:
        kernel = getattr(kernel, "fn", kernel)

    if use_column_tiled:
        grid = lambda META: (outer, triton.cdiv(inner, META["BLOCK_SIZE"]))
    else:
        grid = lambda META: (outer, inner)

    kernel[grid](
        out_flat,
        lhs,
        rhs,
        outer,
        inner,
        cols,
        **(meta or {}),
    )
    return out
