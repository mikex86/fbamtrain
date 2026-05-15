import torch
import triton
import triton.language as tl

_ELEMENTWISE_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=2),
]


def _resolve_kernel(kernel, overrides: dict | None):
    if overrides:
        kernel = getattr(kernel, "fn", kernel)
    return kernel


@triton.autotune(configs=_ELEMENTWISE_CONFIGS, key=["n_elements"])
@triton.jit
def div_elementwise(
    out_ptr,
    lhs_ptr,
    rhs_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    lhs = tl.load(lhs_ptr + offsets, mask=mask, other=0)
    rhs = tl.load(rhs_ptr + offsets, mask=mask, other=1)
    result = lhs / rhs
    tl.store(out_ptr + offsets, result, mask=mask)


@triton.autotune(configs=_ELEMENTWISE_CONFIGS, key=["n_elements"])
@triton.jit
def div_scalar(
    out_ptr,
    lhs_ptr,
    rhs_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    lhs = tl.load(lhs_ptr + offsets, mask=mask, other=0)
    rhs = tl.load(rhs_ptr)
    result = lhs / rhs
    tl.store(out_ptr + offsets, result, mask=mask)


@triton.autotune(configs=_ELEMENTWISE_CONFIGS, key=["n_elements"])
@triton.jit
def div_add(
    out_ptr,
    lhs_ptr,
    rhs_ptr,
    denom_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    lhs = tl.load(lhs_ptr + offsets, mask=mask, other=0)
    rhs = tl.load(rhs_ptr + offsets, mask=mask, other=0)
    denom = tl.load(denom_ptr + offsets, mask=mask, other=1)
    result = lhs + (rhs / denom)
    tl.store(out_ptr + offsets, result, mask=mask)


@triton.autotune(configs=_ELEMENTWISE_CONFIGS, key=["n_elements"])
@triton.jit
def div_scalar_add(
    out_ptr,
    lhs_ptr,
    rhs_ptr,
    denom_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    lhs = tl.load(lhs_ptr + offsets, mask=mask, other=0)
    rhs = tl.load(rhs_ptr + offsets, mask=mask, other=0)
    denom = tl.load(denom_ptr)
    result = lhs + (rhs / denom)
    tl.store(out_ptr + offsets, result, mask=mask)


@triton.autotune(configs=_ELEMENTWISE_CONFIGS, key=["n_elements"])
@triton.jit
def div_scalar_add_broadcast(
    out_ptr,
    lhs_ptr,
    rhs_ptr,
    denom_ptr,
    n_elements,
    inner_size,
    cols,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    idx = offsets
    inner = idx % inner_size
    tmp = idx // inner_size
    outer = tmp // cols
    rhs_index = outer * inner_size + inner

    lhs = tl.load(lhs_ptr + idx, mask=mask, other=0)
    rhs = tl.load(rhs_ptr + rhs_index, mask=mask, other=0)
    denom = tl.load(denom_ptr)
    result = lhs + (rhs / denom)
    tl.store(out_ptr + idx, result, mask=mask)


def launch_div_elementwise(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    block_size: int | None = None,
):
    """Utility to run the element-wise div kernel in eager mode for debugging."""
    if out is None:
        out = torch.empty_like(lhs)
    assert lhs.shape == rhs.shape == out.shape, "all tensors must share shape"
    assert lhs.is_contiguous() and rhs.is_contiguous() and out.is_contiguous(), "tensors must be contiguous"
    if lhs.dtype != rhs.dtype or lhs.dtype != out.dtype:
        raise ValueError("launch_div_elementwise expects matching dtypes")

    n_elements = lhs.numel()

    meta = {"BLOCK_SIZE": block_size} if block_size is not None else None
    kernel = _resolve_kernel(div_elementwise, meta)

    grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
    kernel[grid](
        out_ptr=out,
        lhs_ptr=lhs,
        rhs_ptr=rhs,
        n_elements=n_elements,
        **(meta or {}),
    )
    return out


def launch_div_scalar(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    block_size: int | None = None,
):
    """Utility to run the scalar div kernel in eager mode for debugging."""
    if out is None:
        out = torch.empty_like(lhs)
    assert rhs.numel() == 1, "rhs must be a scalar"
    assert lhs.is_contiguous() and rhs.is_contiguous() and out.is_contiguous(), "tensors must be contiguous"
    if lhs.dtype != rhs.dtype or lhs.dtype != out.dtype:
        raise ValueError("launch_div_scalar expects matching dtypes")

    n_elements = lhs.numel()

    meta = {"BLOCK_SIZE": block_size} if block_size is not None else None
    kernel = _resolve_kernel(div_scalar, meta)

    grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
    kernel[grid](
        out_ptr=out,
        lhs_ptr=lhs,
        rhs_ptr=rhs,
        n_elements=n_elements,
        **(meta or {}),
    )
    return out


def launch_div_add(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    denom: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    block_size: int | None = None,
):
    """Utility to run the div-add kernel in eager mode for debugging."""
    if out is None:
        out = torch.empty_like(lhs)
    assert lhs.shape == rhs.shape == out.shape, "lhs/rhs/out must share shape"
    assert lhs.is_contiguous() and rhs.is_contiguous() and denom.is_contiguous() and out.is_contiguous(), (
        "tensors must be contiguous"
    )
    if lhs.dtype != rhs.dtype or lhs.dtype != denom.dtype or lhs.dtype != out.dtype:
        raise ValueError("launch_div_add expects matching dtypes")

    n_elements = lhs.numel()

    meta = {"BLOCK_SIZE": block_size} if block_size is not None else None
    kernel = _resolve_kernel(div_add, meta)

    grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
    kernel[grid](
        out_ptr=out,
        lhs_ptr=lhs,
        rhs_ptr=rhs,
        denom_ptr=denom,
        n_elements=n_elements,
        **(meta or {}),
    )
    return out


def launch_div_scalar_add(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    denom: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    block_size: int | None = None,
):
    """Utility to run the scalar div-add kernel in eager mode for debugging."""
    if out is None:
        out = torch.empty_like(lhs)
    assert denom.numel() == 1, "denom must be a scalar"
    assert lhs.shape == rhs.shape == out.shape, "lhs/rhs/out must share shape"
    assert lhs.is_contiguous() and rhs.is_contiguous() and denom.is_contiguous() and out.is_contiguous(), (
        "tensors must be contiguous"
    )
    if lhs.dtype != rhs.dtype or lhs.dtype != denom.dtype or lhs.dtype != out.dtype:
        raise ValueError("launch_div_scalar_add expects matching dtypes")

    n_elements = lhs.numel()

    meta = {"BLOCK_SIZE": block_size} if block_size is not None else None
    kernel = _resolve_kernel(div_scalar_add, meta)

    grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
    kernel[grid](
        out_ptr=out,
        lhs_ptr=lhs,
        rhs_ptr=rhs,
        denom_ptr=denom,
        n_elements=n_elements,
        **(meta or {}),
    )
    return out


def launch_div_scalar_add_broadcast(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    denom: torch.Tensor,
    inner_size: int,
    cols: int,
    out: torch.Tensor | None = None,
    *,
    block_size: int | None = None,
):
    """Utility to run the scalar div-add broadcast kernel in eager mode for debugging."""
    if out is None:
        out = torch.empty_like(lhs)
    assert denom.numel() == 1, "denom must be a scalar"
    assert lhs.is_contiguous() and rhs.is_contiguous() and denom.is_contiguous() and out.is_contiguous(), (
        "tensors must be contiguous"
    )
    if lhs.dtype != rhs.dtype or lhs.dtype != denom.dtype or lhs.dtype != out.dtype:
        raise ValueError("launch_div_scalar_add_broadcast expects matching dtypes")

    n_elements = lhs.numel()

    meta = {"BLOCK_SIZE": block_size} if block_size is not None else None
    kernel = _resolve_kernel(div_scalar_add_broadcast, meta)

    grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
    kernel[grid](
        out_ptr=out,
        lhs_ptr=lhs,
        rhs_ptr=rhs,
        denom_ptr=denom,
        n_elements=n_elements,
        inner_size=inner_size,
        cols=cols,
        **(meta or {}),
    )
    return out
