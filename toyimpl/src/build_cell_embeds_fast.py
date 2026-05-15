import torch
import triton
import triton.language as tl

NUM_CHANNELS_PER_CELL = 3  # cp, fg_rgb, bg_rgb


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

    # Program ids in D and N dimensions
    pid_d = tl.program_id(0)
    pid_n = tl.program_id(1)

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
def build_cell_embeds(
        input_cell_states: torch.Tensor,  # (N, 3) int32, CUDA
        cp_embed: torch.Tensor,  # (V, D) same dtype/device as out
        position_embed: torch.Tensor,  # (P, D) same dtype/device as out
        fg_r_embed: torch.Tensor, fg_g_embed: torch.Tensor, fg_b_embed: torch.Tensor,
        bg_r_embed: torch.Tensor, bg_g_embed: torch.Tensor, bg_b_embed: torch.Tensor,
):
    for v in (fg_r_embed, fg_g_embed, fg_b_embed, bg_r_embed, bg_g_embed, bg_b_embed, position_embed):
        assert v.is_cuda and v.is_contiguous()

    N, _ = input_cell_states.shape
    V, D = cp_embed.shape
    P, D_pos = position_embed.shape
    assert D == D_pos
    if P == 0:
        raise ValueError("position_embed must have at least one position")
    assert position_embed.dtype == cp_embed.dtype
    assert position_embed.device == cp_embed.device
    assert input_cell_states.shape == (N, NUM_CHANNELS_PER_CELL)
    for v in (fg_r_embed, fg_g_embed, fg_b_embed, bg_r_embed, bg_g_embed, bg_b_embed):
        assert v.shape == (D,)

    out = torch.empty((N, D), dtype=cp_embed.dtype, device=cp_embed.device)
    grid = lambda META: (
        triton.cdiv(D, META['BLOCK_SIZE_X']),
        triton.cdiv(N, META['BLOCK_SIZE_Y']),
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
    return out
