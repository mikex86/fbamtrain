import triton
import triton.language as tl
from triton.language import erf  # pylint: disable=unused-import
import torch


def is_cuda():
    return torch.version.hip is None


DEVICE = torch.device('cuda' if is_cuda() else 'cpu')


def get_cuda_autotune_config():
    return [
        # Tuned on 8192x8192 matmul: best-performing configs first.
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 6},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
    ]


def get_hip_autotune_config():
    sizes = [
        {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 6},
        {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4},
        {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 6},
        {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 6},
        {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4},
        {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4},
        {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4},
        {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 6},
    ]
    return [triton.Config(s | {'matrix_instr_nonkdim': 16}, num_warps=8, num_stages=2) for s in sizes]


def get_autotune_config():
    if is_cuda():
        return get_cuda_autotune_config()
    else:
        return get_hip_autotune_config()


# Computes D = act(A @ B + C)
@triton.jit
def addmm_act(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr, d_ptr,
        # Matrix dimensions
        M, N, K,
        # Strides
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        stride_dm, stride_dn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        ACTIVATION: tl.constexpr,
        ACCUMULATE_IN_FP16: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        ACCUMULATE_OUTPUT: tl.constexpr,
):
    """Kernel for computing D = activation(A @ B + C).
    A: (M, K), B: (K, N), C/D: (M, N)
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    offs_am = offs_m % M
    offs_bn = offs_n % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    # Stride metadata already reflects transposed views; reuse the standard addressing.
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    tile_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    accumulator_dtype = tl.float16 if ACCUMULATE_IN_FP16 else tl.float32
    if HAS_BIAS and ACTIVATION != 3:
        c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
        c_tile = tl.load(c_ptrs, mask=tile_mask, other=0.0).to(accumulator_dtype)
    else:
        c_tile = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=accumulator_dtype)

    accumulator = c_tile
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b, out_dtype=accumulator_dtype)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if ACTIVATION == 1:  # ReLU
        accumulator = tl.maximum(accumulator, 0.0)
    elif ACTIVATION == 2:  # GELU
        accumulator = gelu(accumulator)
    elif ACTIVATION == 3:  # GELU backward (uses c_ptr as pre-activation)
        pre_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
        pre = tl.load(pre_ptrs, mask=tile_mask, other=0.0).to(tl.float32)
        grad = gelu_grad(pre)
        accumulator = accumulator.to(tl.float32) * grad

    if ACCUMULATE_OUTPUT:
        # Load existing output and accumulate
        d_ptrs = d_ptr + stride_dm * offs_m[:, None] + stride_dn * offs_n[None, :]
        existing = tl.load(d_ptrs, mask=tile_mask, other=0.0).to(accumulator_dtype)
        accumulator = accumulator + existing
        tl.store(d_ptrs, accumulator, mask=tile_mask)
    else:
        d_ptrs = d_ptr + stride_dm * offs_m[:, None] + stride_dn * offs_n[None, :]
        tl.store(d_ptrs, accumulator, mask=tile_mask)


# Computes D = act(A @ B + C) and writes pre-activation to E.
@triton.jit
def addmm_act_preact(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
        # Matrix dimensions
        M, N, K,
        # Strides
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        stride_dm, stride_dn,
        stride_em, stride_en,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        ACTIVATION: tl.constexpr,
        ACCUMULATE_IN_FP16: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        ACCUMULATE_OUTPUT: tl.constexpr,
):
    """Kernel for computing D = activation(A @ B + C) and E = A @ B + C."""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    offs_am = offs_m % M
    offs_bn = offs_n % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    tile_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    accumulator_dtype = tl.float16 if ACCUMULATE_IN_FP16 else tl.float32
    if HAS_BIAS and ACTIVATION != 3:
        c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
        c_tile = tl.load(c_ptrs, mask=tile_mask, other=0.0).to(accumulator_dtype)
    else:
        c_tile = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=accumulator_dtype)

    accumulator = c_tile
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b, out_dtype=accumulator_dtype)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    pre_ptrs = e_ptr + stride_em * offs_m[:, None] + stride_en * offs_n[None, :]
    tl.store(pre_ptrs, accumulator, mask=tile_mask)

    if ACTIVATION == 1:  # ReLU
        accumulator = tl.maximum(accumulator, 0.0)
    elif ACTIVATION == 2:  # GELU
        accumulator = gelu(accumulator)
    elif ACTIVATION == 3:  # GELU backward (uses c_ptr as pre-activation)
        pre_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
        pre = tl.load(pre_ptrs, mask=tile_mask, other=0.0).to(tl.float32)
        grad = gelu_grad(pre)
        accumulator = accumulator.to(tl.float32) * grad

    if ACCUMULATE_OUTPUT:
        d_ptrs = d_ptr + stride_dm * offs_m[:, None] + stride_dn * offs_n[None, :]
        existing = tl.load(d_ptrs, mask=tile_mask, other=0.0).to(accumulator_dtype)
        accumulator = accumulator + existing
        tl.store(d_ptrs, accumulator, mask=tile_mask)
    else:
        d_ptrs = d_ptr + stride_dm * offs_m[:, None] + stride_dn * offs_n[None, :]
        tl.store(d_ptrs, accumulator, mask=tile_mask)


@triton.jit
def gelu(x):
    """GELU activation function"""
    x_fp32 = x.to(tl.float32)
    result = 0.5 * x_fp32 * (1.0 + erf(x_fp32 / tl.sqrt(2.0)))
    return result.to(x.dtype)


@triton.jit
def gelu_grad(x):
    """GELU gradient for pre-activation values."""
    x_fp32 = x.to(tl.float32)
    erf_term = erf(x_fp32 / tl.sqrt(2.0))
    exp_term = tl.exp(-0.5 * x_fp32 * x_fp32)
    return 0.5 * (1.0 + erf_term) + 0.5 * x_fp32 * 0.7978845608028654 * exp_term


# -------------------------------
# Launcher: Triton addmm (no act)
# -------------------------------
def triton_addmm(a: torch.Tensor,
                 b: torch.Tensor,
                 c: torch.Tensor,
                 activation=None,
                 accumulate_in_fp16: bool = False,
                 accumulate_output: bool = False) -> torch.Tensor:
    """
    Compute D = A @ B + C with optional activation (None, 'relu', 'gelu').
    Passing activation=None disables activation (ACTIVATION=0).
    """
    assert a.is_cuda and b.is_cuda and c.is_cuda, "All tensors must be on CUDA"
    assert a.ndim == b.ndim == c.ndim == 2, "A, B, C must be 2D"
    M, K = a.shape
    Kb, N = b.shape
    Mc, Nc = c.shape
    assert K == Kb and M == Mc and N == Nc, f"Shape mismatch: A{a.shape}, B{b.shape}, C{c.shape}"

    # Map activation to the compile-time constant the kernel expects
    if isinstance(activation, str):
        activation = activation.lower()
    act_map = {None: 0, 'none': 0, 0: 0, 'relu': 1, 'gelu': 2}
    if activation not in act_map:
        raise ValueError(f"Unsupported activation: {activation}")
    ACTIVATION = act_map[activation]

    d = torch.empty_like(c)

    # Use a fixed configuration to match the compiled kernels used in the C++ path.
    config = triton.Config(
        {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8},
        num_warps=4,
        num_stages=4,
    )
    def grid(META):
        return (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)

    addmm_act[grid](
        a, b, c, d,
        M, N, K,
        *a.stride(),
        *b.stride(),
        *c.stride(),
        *d.stride(),
        num_warps=config.num_warps,
        num_stages=config.num_stages,
        ACTIVATION=ACTIVATION,
        ACCUMULATE_IN_FP16=accumulate_in_fp16,
        HAS_BIAS=True,
        ACCUMULATE_OUTPUT=accumulate_output,
        **config.kwargs,
    )
    return d


# -------------------------------
# Demo run: 1024 x 1024, no act
# -------------------------------
def run_addmm_1024():
    torch.manual_seed(0)
    dtype = torch.float16 if DEVICE.type == 'cuda' else torch.float32
    M = N = K = 1024

    A = torch.randn((M, K), device=DEVICE, dtype=dtype)
    B = torch.randn((K, N), device=DEVICE, dtype=dtype)
    C = torch.randn((M, N), device=DEVICE, dtype=dtype)

    # Run Triton kernel with no activation
    D = triton_addmm(A, B, C, activation=None)

    # Correctness check vs PyTorch (accumulate in fp32, then cast)
    ref = (A.to(torch.float32) @ B.to(torch.float32) + C.to(torch.float32)).to(dtype)
    max_abs_err = (D - ref).abs().max().item()
    print(f"Shapes: A{A.shape} @ B{B.shape} + C{C.shape} -> D{D.shape}")
    print(f"dtype: {dtype}, device: {DEVICE}")
    print(f"Max abs error vs PyTorch: {max_abs_err:.4e}")


if __name__ == "__main__":
    run_addmm_1024()
