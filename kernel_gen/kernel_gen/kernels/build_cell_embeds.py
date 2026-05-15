import torch
import triton
import triton.language as tl

NUM_CHANNELS_PER_CELL = 3
CELL_CODEPOINT_CHANNEL_IDX = 0
CELL_FG_COLOR_CHANNEL_IDX = 1
CELL_BG_COLOR_CHANNEL_IDX = 2


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=4),
    ],
    key=['embed_dim', 'n_cells']
)
@triton.jit
def build_cell_embeds_1d(
        out_ptr,  # *T, shape (N, D)
        input_cell_states,  # *u32, shape (N, 3) laid out contiguously

        cp_embed,  # *T, shape (V, D)
        position_embed,  # *T, shape (P, D)

        # Channel embedding vectors (each length D)
        fg_r_embed, fg_g_embed, fg_b_embed,
        bg_r_embed, bg_g_embed, bg_b_embed,

        # Shapes / strides
        n_cells: tl.uint32,  # N
        embed_dim: tl.uint32,  # D
        vocab_size: tl.uint32,  # V
        max_positions: tl.uint32,  # P

        stride_out_n: tl.uint32,  # usually D
        stride_out_d: tl.uint32,  # usually 1

        stride_cp_v: tl.uint32,  # usually D
        stride_cp_d: tl.uint32,  # usually 1

        stride_pos_p: tl.uint32,  # usually D
        stride_pos_d: tl.uint32,  # usually 1

        BLOCK_SIZE: tl.constexpr,
):
    out_dtype = out_ptr.dtype.element_ty

    # 1D program id over flattened space of size N * D
    pid = tl.program_id(axis=0)
    linear = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    linear = linear.to(tl.uint32)

    total = n_cells * embed_dim
    mask = linear < total

    # Map to (cell_idx, d_off)
    cell_idx = linear // embed_dim
    d_off = linear - cell_idx * embed_dim  # guaranteed in [0, D)
    # Provide compiler hints for better vectorization/coalescing across d_off
    tl.multiple_of(d_off, 16)
    tl.max_contiguous(d_off, 128)

    # Load per-cell triplet (cp_raw, fg_rgb, bg_rgb)
    base_u32 = input_cell_states + cell_idx * 3
    cp_raw = tl.load(base_u32 + 0, mask=mask, other=0, cache_modifier=".ca").to(tl.int32)
    fg_rgb = tl.load(base_u32 + 1, mask=mask, other=0, cache_modifier=".ca").to(tl.int32)
    bg_rgb = tl.load(base_u32 + 2, mask=mask, other=0, cache_modifier=".ca").to(tl.int32)

    # Codepoint validity: match PyTorch reference (>=0 and < V)
    cp_valid = (cp_raw >= 0) & (cp_raw < vocab_size)
    # Clamp index to [0, V-1] for safe addressing (even when invalid lanes are masked)
    cp_idx = tl.minimum(cp_raw, vocab_size - 1)
    cp_idx = tl.maximum(cp_idx, 0)

    # Decode colors: 0xRRGGBB -> per-channel intensities in [0, 1]
    fg_r = ((fg_rgb >> 16) & 0xFF).to(tl.float32)
    fg_g = ((fg_rgb >> 8) & 0xFF).to(tl.float32)
    fg_b = (fg_rgb & 0xFF).to(tl.float32)

    bg_r = ((bg_rgb >> 16) & 0xFF).to(tl.float32)
    bg_g = ((bg_rgb >> 8) & 0xFF).to(tl.float32)
    bg_b = (bg_rgb & 0xFF).to(tl.float32)

    inv255 = 1.0 / 255.0
    fg_r *= inv255;
    fg_g *= inv255;
    fg_b *= inv255
    bg_r *= inv255;
    bg_g *= inv255;
    bg_b *= inv255

    # Codepoint row: cp_embed[cp_idx, d_off]
    cp_ptrs = cp_embed + cp_idx * stride_cp_v + d_off * stride_cp_d
    cp_vec = tl.load(
        cp_ptrs,
        mask=mask & cp_valid,
        other=0.0,
        cache_modifier=".ca",
    ).to(tl.float32)

    pos_idx = cell_idx % max_positions
    pos_ptrs = position_embed + pos_idx * stride_pos_p + d_off * stride_pos_d
    pos_vec = tl.load(
        pos_ptrs,
        mask=mask,
        other=0.0,
        cache_modifier=".ca",
    ).to(tl.float32)

    # Channel vectors scaled by intensities (same d_off across cells)
    fg_r_col = tl.load(fg_r_embed + d_off, mask=mask, other=0.0, cache_modifier=".ca").to(tl.float32)
    fg_g_col = tl.load(fg_g_embed + d_off, mask=mask, other=0.0, cache_modifier=".ca").to(tl.float32)
    fg_b_col = tl.load(fg_b_embed + d_off, mask=mask, other=0.0, cache_modifier=".ca").to(tl.float32)

    bg_r_col = tl.load(bg_r_embed + d_off, mask=mask, other=0.0, cache_modifier=".ca").to(tl.float32)
    bg_g_col = tl.load(bg_g_embed + d_off, mask=mask, other=0.0, cache_modifier=".ca").to(tl.float32)
    bg_b_col = tl.load(bg_b_embed + d_off, mask=mask, other=0.0, cache_modifier=".ca").to(tl.float32)

    acc = (
            cp_vec
            + fg_r_col * fg_r + fg_g_col * fg_g + fg_b_col * fg_b
            + bg_r_col * bg_r + bg_g_col * bg_g + bg_b_col * bg_b
            + pos_vec
    )

    out_ptrs = out_ptr + cell_idx * stride_out_n + d_off * stride_out_d

    # bypass cache for writes to not evict embedding vectors which will be in L1/L2
    tl.store(out_ptrs, acc.to(out_dtype), mask=mask, cache_modifier=".cs")


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_X': 64, 'BLOCK_SIZE_Y': 4}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_X': 128, 'BLOCK_SIZE_Y': 4}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_X': 128, 'BLOCK_SIZE_Y': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_X': 256, 'BLOCK_SIZE_Y': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_X': 128, 'BLOCK_SIZE_Y': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_X': 256, 'BLOCK_SIZE_Y': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE_X': 128, 'BLOCK_SIZE_Y': 32}, num_warps=8, num_stages=2),
    ],
    key=['embed_dim', 'n_cells']
)
@triton.jit
def build_cell_embeds_2d(
        out_ptr,  # *T, shape (N, D)
        input_cell_states,  # *u32, shape (N, 3)

        cp_embed,  # *T, shape (V, D)
        position_embed,  # *T, shape (P, D)

        # Channel embedding vectors (each length D)
        fg_r_embed, fg_g_embed, fg_b_embed,
        bg_r_embed, bg_g_embed, bg_b_embed,

        # Shapes / strides
        n_cells: tl.uint32,  # N
        embed_dim: tl.uint32,  # D
        vocab_size: tl.uint32,  # V
        max_positions: tl.uint32,  # P

        stride_out_n: tl.uint32,  # usually D
        stride_out_d: tl.uint32,  # usually 1

        stride_cp_v: tl.uint32,  # usually D
        stride_cp_d: tl.uint32,  # usually 1

        stride_pos_p: tl.uint32,  # usually D
        stride_pos_d: tl.uint32,  # usually 1

        BLOCK_SIZE_X: tl.constexpr,
        BLOCK_SIZE_Y: tl.constexpr,
):
    out_dtype = out_ptr.dtype.element_ty

    # Flatten (N, D) program ids into a single launch dimension to avoid grid Y limits.
    pid = tl.program_id(0).to(tl.uint32)
    grid_d = tl.cdiv(embed_dim, BLOCK_SIZE_X).to(tl.uint32)
    pid_n = pid // grid_d
    pid_d = pid - pid_n * grid_d

    off_d = pid_d * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    off_d = off_d.to(tl.uint32)
    mask_d = off_d < embed_dim
    tl.multiple_of(off_d, 16)
    tl.max_contiguous(off_d, 128)

    # Load six channel vectors once per D-tile (keep in registers)
    fg_r_col = tl.load(fg_r_embed + off_d, mask=mask_d, other=0.0, cache_modifier=".ca").to(tl.float32)
    fg_g_col = tl.load(fg_g_embed + off_d, mask=mask_d, other=0.0, cache_modifier=".ca").to(tl.float32)
    fg_b_col = tl.load(fg_b_embed + off_d, mask=mask_d, other=0.0, cache_modifier=".ca").to(tl.float32)

    bg_r_col = tl.load(bg_r_embed + off_d, mask=mask_d, other=0.0, cache_modifier=".ca").to(tl.float32)
    bg_g_col = tl.load(bg_g_embed + off_d, mask=mask_d, other=0.0, cache_modifier=".ca").to(tl.float32)
    bg_b_col = tl.load(bg_b_embed + off_d, mask=mask_d, other=0.0, cache_modifier=".ca").to(tl.float32)

    base_n = pid_n * BLOCK_SIZE_Y
    inv255 = 1.0 / 255.0

    # Process a small batch of cells while reusing channel vectors
    for i in range(0, BLOCK_SIZE_Y):
        n = base_n + i
        n_u32 = n.to(tl.uint32)
        mask_n = n_u32 < n_cells

        base = input_cell_states + n_u32 * 3
        cp_raw = tl.load(base + 0, mask=mask_n, other=0).to(tl.int32)
        fg_rgb = tl.load(base + 1, mask=mask_n, other=0).to(tl.int32)
        bg_rgb = tl.load(base + 2, mask=mask_n, other=0).to(tl.int32)

        # validity + clamp
        cp_valid = (cp_raw >= 0) & (cp_raw < vocab_size)
        cp_idx = tl.where(cp_raw < 0, 0, tl.minimum(cp_raw, vocab_size - 1))

        # decode colors -> [0,1]
        fg_r = ((fg_rgb >> 16) & 0xFF).to(tl.float32) * inv255
        fg_g = ((fg_rgb >> 8) & 0xFF).to(tl.float32) * inv255
        fg_b = (fg_rgb & 0xFF).to(tl.float32) * inv255
        bg_r = ((bg_rgb >> 16) & 0xFF).to(tl.float32) * inv255
        bg_g = ((bg_rgb >> 8) & 0xFF).to(tl.float32) * inv255
        bg_b = (bg_rgb & 0xFF).to(tl.float32) * inv255

        # gather cp row for this cell & D-tile (stream via L2)
        cp_ptrs = cp_embed + cp_idx * stride_cp_v + off_d * stride_cp_d
        cp_vec = tl.load(cp_ptrs, mask=mask_d & mask_n & cp_valid, other=0.0, cache_modifier=".ca").to(tl.float32)

        pos_idx = n_u32 % max_positions
        pos_ptrs = position_embed + pos_idx * stride_pos_p + off_d * stride_pos_d
        pos_vec = tl.load(pos_ptrs, mask=mask_d & mask_n, other=0.0, cache_modifier=".ca").to(tl.float32)

        acc = cp_vec
        acc += fg_r_col * fg_r
        acc += fg_g_col * fg_g
        acc += fg_b_col * fg_b
        acc += bg_r_col * bg_r
        acc += bg_g_col * bg_g
        acc += bg_b_col * bg_b
        acc += pos_vec

        out_ptrs = out_ptr + n_u32 * stride_out_n + off_d * stride_out_d

        # bypass cache for writes to not evict embedding vectors which will be in L1/L2
        tl.store(out_ptrs, acc.to(out_dtype), mask=mask_d & mask_n, cache_modifier=".cs")


@torch.no_grad()
def build_cell_embeds_(
        out: torch.Tensor,  # (N, D) float32/float16/bfloat16, CUDA
        input_cell_states: torch.Tensor,  # (N, 3) int32, CUDA
        cp_embed: torch.Tensor,  # (V, D) same dtype/device as out
        position_embed: torch.Tensor,  # (P, D) same dtype/device as out
        fg_r_embed: torch.Tensor, fg_g_embed: torch.Tensor, fg_b_embed: torch.Tensor,
        bg_r_embed: torch.Tensor, bg_g_embed: torch.Tensor, bg_b_embed: torch.Tensor,
        use_2d: bool = None,  # None => heuristic
):
    assert out.is_cuda and input_cell_states.is_cuda and cp_embed.is_cuda and position_embed.is_cuda
    assert out.is_contiguous() and cp_embed.is_contiguous() and position_embed.is_contiguous()
    for v in (fg_r_embed, fg_g_embed, fg_b_embed, bg_r_embed, bg_g_embed, bg_b_embed):
        assert v.is_cuda and v.is_contiguous()
        assert v.dtype == out.dtype
    assert position_embed.dtype == out.dtype

    N, D = out.shape
    V, D2 = cp_embed.shape
    assert D == D2
    assert input_cell_states.shape == (N, NUM_CHANNELS_PER_CELL)
    P, D3 = position_embed.shape
    assert D3 == D
    for v in (fg_r_embed, fg_g_embed, fg_b_embed, bg_r_embed, bg_g_embed, bg_b_embed):
        assert v.shape == (D,)

    if P == 0:
        raise ValueError("position_embed must have at least one position")

    # Heuristic: use 2D when we can reuse channel vectors across several cells
    if use_2d is None:
        use_2d = (N >= 32 and D >= 128)

    if not use_2d:
        grid = lambda META: (triton.cdiv(N * D, META['BLOCK_SIZE']),)
        build_cell_embeds_1d[grid](
            out,
            input_cell_states,
            cp_embed,
            position_embed,
            fg_r_embed, fg_g_embed, fg_b_embed,
            bg_r_embed, bg_g_embed, bg_b_embed,
            N, D, V, P,
            out.stride(0), out.stride(1),
            cp_embed.stride(0), cp_embed.stride(1),
            position_embed.stride(0), position_embed.stride(1),
        )
    else:
        grid = lambda META: (
            triton.cdiv(D, META['BLOCK_SIZE_X']) * triton.cdiv(N, META['BLOCK_SIZE_Y']),
        )
        build_cell_embeds_2d[grid](
            out,
            input_cell_states,
            cp_embed,
            position_embed,
            fg_r_embed, fg_g_embed, fg_b_embed,
            bg_r_embed, bg_g_embed, bg_b_embed,
            N, D, V, P,
            out.stride(0), out.stride(1),
            cp_embed.stride(0), cp_embed.stride(1),
            position_embed.stride(0), position_embed.stride(1),
        )


@torch.no_grad()
def build_cell_embeds_torch(
        input_cell_states: torch.Tensor,  # (N, 3) int32, device=CUDA/CPU
        cp_embed: torch.Tensor,  # (V, D) float*, device matches output
        position_embed: torch.Tensor,  # (P, D) float*, device matches output
        fg_r_embed: torch.Tensor, fg_g_embed: torch.Tensor, fg_b_embed: torch.Tensor,
        bg_r_embed: torch.Tensor, bg_g_embed: torch.Tensor, bg_b_embed: torch.Tensor,
        out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Vectorized PyTorch version with identical semantics to the Triton kernels.
    Accumulates in float32, then casts to out_dtype.
    """
    assert input_cell_states.dim() == 2 and input_cell_states.size(1) == 3
    N = input_cell_states.size(0)
    V, D = cp_embed.shape
    P, D_pos = position_embed.shape
    if P == 0:
        raise ValueError("position_embed must have at least one position")
    if D_pos != D:
        raise ValueError("position_embed embedding dim must match cp_embed dim")
    if position_embed.dtype != cp_embed.dtype:
        raise ValueError("position_embed dtype must match cp_embed dtype")
    if position_embed.device != cp_embed.device:
        raise ValueError("position_embed device must match cp_embed device")

    # Decode channels
    states = input_cell_states.to(torch.int32)
    cp_raw = states[:, CELL_CODEPOINT_CHANNEL_IDX]
    fg_rgb = states[:, CELL_FG_COLOR_CHANNEL_IDX]
    bg_rgb = states[:, CELL_BG_COLOR_CHANNEL_IDX]

    # Codepoint contribution (mask out-of-range)
    cp_clamped = torch.minimum(cp_raw, torch.tensor(V - 1, dtype=torch.int32, device=states.device))
    cp_valid = (cp_raw >= 0) & (cp_raw < V)
    cp_vec = cp_embed[cp_clamped.to(torch.long)]
    cp_vec = torch.where(cp_valid.view(-1, 1), cp_vec, torch.zeros_like(cp_vec))

    # Colors -> [0, 1]
    def rgb_to_unit(rgb):
        r = ((rgb >> 16) & 0xFF).to(torch.float32)
        g = ((rgb >> 8) & 0xFF).to(torch.float32)
        b = (rgb & 0xFF).to(torch.float32)
        inv255 = 1.0 / 255.0
        return r * inv255, g * inv255, b * inv255

    fg_r, fg_g, fg_b = rgb_to_unit(fg_rgb)
    bg_r, bg_g, bg_b = rgb_to_unit(bg_rgb)

    # Compose (accumulate in fp32)
    acc = cp_vec.to(torch.float32)
    pos_indices = torch.arange(N, device=states.device, dtype=torch.int64) % P
    pos_vec = position_embed.index_select(0, pos_indices).to(torch.float32)
    acc = acc + pos_vec
    acc = acc + fg_r.view(-1, 1) * fg_r_embed.view(1, -1)
    acc = acc + fg_g.view(-1, 1) * fg_g_embed.view(1, -1)
    acc = acc + fg_b.view(-1, 1) * fg_b_embed.view(1, -1)
    acc = acc + bg_r.view(-1, 1) * bg_r_embed.view(1, -1)
    acc = acc + bg_g.view(-1, 1) * bg_g_embed.view(1, -1)
    acc = acc + bg_b.view(-1, 1) * bg_b_embed.view(1, -1)

    return acc.to(dtype=out_dtype, copy=True, memory_format=torch.contiguous_format)


# ---------------------------------------------------------------------------
# Test utilities
# ---------------------------------------------------------------------------
def _dtype_supported_on_cuda(dtype: torch.dtype) -> bool:
    if not torch.cuda.is_available():
        return False
    if dtype == torch.float16:
        return True
    if dtype == torch.float32:
        return True
    if dtype == torch.bfloat16:
        # bfloat16 generally requires Ampere+; check via PyTorch helper if present
        try:
            return torch.cuda.is_bf16_supported()
        except AttributeError:
            major, _ = torch.cuda.get_device_capability()
            return major >= 8
    return False


def _rand_inputs(
        N: int, D: int, V: int, P: int | None, dtype: torch.dtype, device: torch.device, seed: int = 0
):
    g = torch.Generator(device=device.type).manual_seed(seed)

    if P is None:
        P = max(1, N)

    cp = torch.randint(low=0, high=1, size=(N,), dtype=torch.int32, device=device, generator=g)
    fg_rgb = torch.randint(low=0, high=0x1000000, size=(N,), dtype=torch.int32, device=device, generator=g)  # 24-bit
    bg_rgb = torch.randint(low=0, high=0x1000000, size=(N,), dtype=torch.int32, device=device, generator=g)  # 24-bit
    states = torch.stack([cp, fg_rgb, bg_rgb], dim=1).contiguous()

    cp_embed = torch.randn((V, D), dtype=dtype, device=device, generator=g).contiguous()
    pos_embed = torch.randn((P, D), dtype=dtype, device=device, generator=g).contiguous()

    def rand_vec():
        return torch.randn((D,), dtype=dtype, device=device, generator=g).contiguous()

    fg_r = rand_vec();
    fg_g = rand_vec();
    fg_b = rand_vec()
    bg_r = rand_vec();
    bg_g = rand_vec();
    bg_b = rand_vec()

    out = torch.empty((N, D), dtype=dtype, device=device).contiguous()

    return (states, cp_embed, pos_embed, fg_r, fg_g, fg_b, bg_r, bg_g, bg_b, out)


def _assert_allclose(a: torch.Tensor, b: torch.Tensor, dtype: torch.dtype):
    if dtype == torch.float32:
        atol, rtol = 1e-6, 1e-5
    elif dtype == torch.float16:
        atol, rtol = 5e-3, 1e-2
    elif dtype == torch.bfloat16:
        atol, rtol = 8e-3, 2e-2
    else:
        atol, rtol = 1e-6, 1e-5
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        max_abs = (a - b).abs().max().item()
        max_rel = ((a - b).abs() / (b.abs() + 1e-12)).max().item()
        raise AssertionError(f"Mismatch: atol={atol}, rtol={rtol}, max_abs={max_abs:.3e}, max_rel={max_rel:.3e}")


def _bench_triton_output_gbps(N: int, D: int, V: int, dtype: torch.dtype, device: torch.device,
                              kernel_kind: str,
                              warmup: int = 20, iters: int = 60):
    """
    Measures output fill rate of the Triton kernels in GB/s (10^9 bytes/s).
    Uses best-of-N timing over CUDA events. Only counts bytes written to `out`.

    kernel_kind: '1d' or '2d'
    """
    assert kernel_kind in {"1d", "2d"}

    (states, cp_embed, pos_embed, fg_r, fg_g, fg_b, bg_r, bg_g, bg_b, out) = _rand_inputs(
        N, D, V, None, dtype=dtype, device=device, seed=N * 123 + D * 7 + V
    )

    # Warm-up (includes Triton JIT + autotune)
    for _ in range(warmup):
        build_cell_embeds_(out, states, cp_embed, pos_embed, fg_r, fg_g, fg_b, bg_r, bg_g, bg_b,
                           use_2d=(kernel_kind == '2d'))

    # Timed runs
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    best_ms = float('inf')
    for _ in range(iters):
        start.record()
        build_cell_embeds_(out, states, cp_embed, pos_embed, fg_r, fg_g, fg_b, bg_r, bg_g, bg_b,
                           use_2d=(kernel_kind == '2d'))
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end)  # milliseconds
        if ms < best_ms:
            best_ms = ms

    bytes_out = out.numel() * out.element_size()
    gbps = (bytes_out * 1e-6) / best_ms  # (bytes * 1e-6) / ms = GB/s with 1 GB = 1e9 bytes
    return gbps, best_ms, bytes_out


def _bench_torch_output_gbps(N: int, D: int, V: int, dtype: torch.dtype, device: torch.device,
                             warmup: int = 20, iters: int = 60):
    (states, cp_embed, pos_embed, fg_r, fg_g, fg_b, bg_r, bg_g, bg_b, out) = _rand_inputs(
        N, D, V, None, dtype=dtype, device=device, seed=N * 123 + D * 7 + V
    )

    # Warm-up (could include PyTorch JIT)
    for _ in range(warmup):
        build_cell_embeds_torch(states, cp_embed, pos_embed, fg_r, fg_g, fg_b, bg_r, bg_g, bg_b, out_dtype=dtype)
    # Timed runs
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    best_ms = float('inf')
    for _ in range(iters):
        start.record()
        build_cell_embeds_torch(states, cp_embed, pos_embed, fg_r, fg_g, fg_b, bg_r, bg_g, bg_b, out_dtype=dtype)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end)
        if ms < best_ms:
            best_ms = ms
    bytes_out = out.numel() * out.element_size()
    gbps = (bytes_out * 1e-6) / best_ms
    return gbps, best_ms, bytes_out


def benchmark_output_fill_rate():
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available; benchmark skipped.")
        return

    device = torch.device('cuda')
    dtypes = [torch.float32, torch.float16]
    if _dtype_supported_on_cuda(torch.bfloat16):
        dtypes.append(torch.bfloat16)

    shapes = [
        (8192, 256, 1024),
        (8192, 1024, 1024),
        (65536, 128, 4096),
        (65536, 1024, 256),
        (245760, 1024, 256),
    ]

    print("\n[BENCH] Triton output fill rate (GB/s = 10^9 bytes/s, best-of runs)")
    print("        warmup=20, iters=60\n")
    print(f"{'kernel':>6} {'dtype':>8} {'N':>8} {'D':>8} {'V':>8} {'GB/s':>10} {'best_ms':>10} {'bytes_out(MB)':>16}")

    for dtype in dtypes:
        for (N, D, V) in shapes:
            for kind in ("1d", "2d"):
                gbps, ms, bytes_out = _bench_triton_output_gbps(N, D, V, dtype, device, kernel_kind=kind)
                print(f"{kind:>6} {str(dtype).split('.')[-1]:>8} {N:8d} {D:8d} {V:8d} "
                      f"{gbps:10.2f} {ms:10.3f} {bytes_out / 1e6:16.2f}")

    print("\n[BENCH] Torch output fill rate (GB/s = 10^9 bytes/s, best-of runs)")
    print("        warmup=20, iters=60\n")
    print(f"{'dtype':>8} {'N':>8} {'D':>8} {'V':>8} {'GB/s':>10} {'best_ms':>10} {'bytes_out(MB)':>16}")
    for dtype in dtypes:
        for (N, D, V) in shapes:
            gbps, ms, bytes_out = _bench_torch_output_gbps(N, D, V, dtype, device)
            print(f"{str(dtype).split('.')[-1]:>8} {N:8d} {D:8d} {V:8d} "
                  f"{gbps:10.2f} {ms:10.3f} {bytes_out / 1e6:16.2f}")


def test_triton_vs_torch_random_shapes():
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available; Triton test skipped.")
        return

    device = torch.device('cuda')
    dtypes = [torch.float32, torch.float16]
    if _dtype_supported_on_cuda(torch.bfloat16):
        dtypes.append(torch.bfloat16)

    shapes = [
        (1, 1, 8),
        (4, 7, 32),
        (17, 129, 257),
        (32, 256, 512),
        (11, 513, 1024),
    ]

    for dtype in dtypes:
        for (N, D, V) in shapes:
            states, cp_embed, pos_embed, fg_r, fg_g, fg_b, bg_r, bg_g, bg_b, out = _rand_inputs(
                N, D, V, None, dtype=dtype, device=device, seed=N * 1000 + D * 10 + V
            )

            # Heuristic path (may pick 1D or 2D)
            build_cell_embeds_(
                out, states, cp_embed, pos_embed, fg_r, fg_g, fg_b, bg_r, bg_g, bg_b
            )

            ref = build_cell_embeds_torch(
                states, cp_embed, pos_embed, fg_r, fg_g, fg_b, bg_r, bg_g, bg_b, out_dtype=dtype
            ).contiguous()

            _assert_allclose(out, ref, dtype)
    print("[PASS] test_triton_vs_torch_random_shapes")


def test_1d_vs_2d_equivalence():
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available; Triton test skipped.")
        return

    device = torch.device('cuda')
    dtypes = [torch.float32, torch.float16]
    if _dtype_supported_on_cuda(torch.bfloat16):
        dtypes.append(torch.bfloat16)

    shapes = [
        (64, 128, 256),
        (8192, 256, 1024),
        (4096, 1024, 2048),
    ]

    for dtype in dtypes:
        for (N, D, V) in shapes:
            states, cp_embed, pos_embed, fg_r, fg_g, fg_b, bg_r, bg_g, bg_b, out1 = _rand_inputs(
                N, D, V, None, dtype=dtype, device=device, seed=N * 42 + D * 17 + V
            )
            out2 = torch.empty_like(out1)

            build_cell_embeds_(out1, states, cp_embed, pos_embed, fg_r, fg_g, fg_b, bg_r, bg_g, bg_b, use_2d=False)
            build_cell_embeds_(out2, states, cp_embed, pos_embed, fg_r, fg_g, fg_b, bg_r, bg_g, bg_b, use_2d=True)

            _assert_allclose(out1, out2, dtype)
    print("[PASS] test_1d_vs_2d_equivalence")


def test_known_values_sanity():
    """
    Simple determinism/sanity check on tiny inputs with hand-picked values.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32

    N, D, V = 2, 4, 5

    # States: cp in-range for row 0, out-of-range for row 1
    cp = torch.tensor([2, 99], dtype=torch.int32, device=device)
    fg_rgb = torch.tensor([0x112233, 0xFFFFFF], dtype=torch.int32, device=device)
    bg_rgb = torch.tensor([0x000000, 0x123456], dtype=torch.int32, device=device)
    states = torch.stack([cp, fg_rgb, bg_rgb], dim=1).contiguous()

    # Embeds
    torch.manual_seed(0)
    cp_embed = torch.arange(V * D, dtype=dtype, device=device).reshape(V, D).contiguous()
    fg_r = torch.ones(D, dtype=dtype, device=device)  # all 1s
    fg_g = torch.full((D,), 2.0, dtype=dtype, device=device)
    fg_b = torch.full((D,), -1.0, dtype=dtype, device=device)

    bg_r = torch.full((D,), 0.5, dtype=dtype, device=device)
    bg_g = torch.full((D,), -0.5, dtype=dtype, device=device)
    bg_b = torch.zeros(D, dtype=dtype, device=device)
    position_embed = torch.stack(
        [
            torch.full((D,), 0.25, dtype=dtype, device=device),
            torch.full((D,), -0.5, dtype=dtype, device=device),
            torch.full((D,), 0.75, dtype=dtype, device=device),
        ],
        dim=0,
    ).contiguous()

    # Torch reference
    ref = build_cell_embeds_torch(states, cp_embed, position_embed, fg_r, fg_g, fg_b, bg_r, bg_g, bg_b,
                                  out_dtype=dtype)

    # Manual expected row 0 (cp=2): cp_embed[2] = [8,9,10,11]
    # fg=0x11,0x22,0x33 -> [17,34,51]/255 ; bg=0, so only fg contributes.
    fg_r_s = 17 / 255.0;
    fg_g_s = 34 / 255.0;
    fg_b_s = 51 / 255.0
    expected0 = cp_embed[2] + position_embed[0] + fg_r_s * fg_r + fg_g_s * fg_g + fg_b_s * fg_b
    # Row 1: cp invalid -> zero; fg=white -> (1,1,1); bg=0x12,0x34,0x56 -> (18,52,86)/255
    fg_r_s2 = 1.0;
    fg_g_s2 = 1.0;
    fg_b_s2 = 1.0
    bg_r_s2 = 18 / 255.0;
    bg_g_s2 = 52 / 255.0;
    bg_b_s2 = 86 / 255.0
    expected1 = (position_embed[1] +
                 fg_r_s2 * fg_r + fg_g_s2 * fg_g + fg_b_s2 * fg_b +
                 bg_r_s2 * bg_r + bg_g_s2 * bg_g + bg_b_s2 * bg_b)
    expected = torch.stack([expected0, expected1], dim=0).contiguous()

    assert torch.allclose(ref, expected, atol=1e-6), "Sanity check mismatch on CPU/PyTorch reference."

    if torch.cuda.is_available():
        out = torch.empty((N, D), dtype=dtype, device=device).contiguous()
        # Force both paths to hit coverage in this tiny case
        build_cell_embeds_(out, states.to(device), cp_embed.to(device), position_embed.to(device),
                           fg_r.to(device), fg_g.to(device), fg_b.to(device),
                           bg_r.to(device), bg_g.to(device), bg_b.to(device), use_2d=False)
        _assert_allclose(out, expected, dtype)
        build_cell_embeds_(out, states.to(device), cp_embed.to(device), position_embed.to(device),
                           fg_r.to(device), fg_g.to(device), fg_b.to(device),
                           bg_r.to(device), bg_g.to(device), bg_b.to(device), use_2d=True)
        _assert_allclose(out, expected, dtype)
    print("[PASS] test_known_values_sanity")


if __name__ == "__main__":
    torch.manual_seed(1234)

    test_known_values_sanity()
    test_triton_vs_torch_random_shapes()
    test_1d_vs_2d_equivalence()

    benchmark_output_fill_rate()

    print("\nAll tests passed ✅")
