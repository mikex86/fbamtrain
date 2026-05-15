import sys

import torch

NUM_CHANNELS_PER_CELL = 3
CELL_CODEPOINT_CHANNEL_IDX = 0
CELL_FG_COLOR_CHANNEL_IDX = 1
CELL_BG_COLOR_CHANNEL_IDX = 2

def _build_cell_embeds_impl(
        input_cell_states: torch.Tensor,  # (N, 3) int32
        cp_embed: torch.Tensor,  # (V, D) float*
        position_embed: torch.Tensor,  # (P, D) float*
        fg_r_embed: torch.Tensor, fg_g_embed: torch.Tensor, fg_b_embed: torch.Tensor,
        bg_r_embed: torch.Tensor, bg_g_embed: torch.Tensor, bg_b_embed: torch.Tensor,
        out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    assert input_cell_states.dim() == 2 and input_cell_states.size(1) == 3
    V, D = cp_embed.shape
    P, D_pos = position_embed.shape
    if P == 0:
        raise ValueError("position_embed must have at least one row")
    if D != D_pos:
        raise ValueError("position_embed embedding dim mismatch")

    states = input_cell_states.to(torch.int32)
    cp_raw = states[:, CELL_CODEPOINT_CHANNEL_IDX]
    fg_rgb = states[:, CELL_FG_COLOR_CHANNEL_IDX]
    bg_rgb = states[:, CELL_BG_COLOR_CHANNEL_IDX]

    cp_clamped = torch.minimum(cp_raw, torch.tensor(V - 1, dtype=torch.int32, device=states.device))
    cp_valid = (cp_raw >= 0) & (cp_raw < V)
    cp_vec = cp_embed[cp_clamped.to(torch.long)]
    cp_vec = torch.where(cp_valid.view(-1, 1), cp_vec, torch.zeros_like(cp_vec))

    def rgb_to_unit(rgb):
        r = ((rgb >> 16) & 0xFF).to(torch.float32)
        g = ((rgb >> 8) & 0xFF).to(torch.float32)
        b = (rgb & 0xFF).to(torch.float32)
        inv255 = 1.0 / 255.0
        return r * inv255, g * inv255, b * inv255

    fg_r, fg_g, fg_b = rgb_to_unit(fg_rgb)
    bg_r, bg_g, bg_b = rgb_to_unit(bg_rgb)

    acc = cp_vec.to(torch.float32)
    pos_indices = torch.arange(input_cell_states.size(0), device=states.device, dtype=torch.int64) % P
    pos_vec = position_embed.index_select(0, pos_indices).to(torch.float32)
    acc = acc + pos_vec
    acc = acc + fg_r.view(-1, 1) * fg_r_embed.view(1, -1)
    acc = acc + fg_g.view(-1, 1) * fg_g_embed.view(1, -1)
    acc = acc + fg_b.view(-1, 1) * fg_b_embed.view(1, -1)
    acc = acc + bg_r.view(-1, 1) * bg_r_embed.view(1, -1)
    acc = acc + bg_g.view(-1, 1) * bg_g_embed.view(1, -1)
    acc = acc + bg_b.view(-1, 1) * bg_b_embed.view(1, -1)

    return acc.to(dtype=out_dtype, copy=True, memory_format=torch.contiguous_format)


if sys.version_info >= (3, 12):
    build_cell_embeds = torch.no_grad()(_build_cell_embeds_impl)
else:
    build_cell_embeds = torch.no_grad()(
        torch.compile(mode="max-autotune", fullgraph=True, dynamic=False)(_build_cell_embeds_impl)
    )


def build_cell_embeds_autograd(
        input_cell_states: torch.Tensor,  # (N, 3) int32
        cp_embed: torch.Tensor,  # (V, D) float*
        position_embed: torch.Tensor,  # (P, D) float*
        fg_r_embed: torch.Tensor, fg_g_embed: torch.Tensor, fg_b_embed: torch.Tensor,
        bg_r_embed: torch.Tensor, bg_g_embed: torch.Tensor, bg_b_embed: torch.Tensor,
        out_dtype=None,
) -> torch.Tensor:
    if input_cell_states.dim() != 2 or input_cell_states.size(1) != 3:
        raise ValueError(f"Expected input_cell_states shape (N, 3), got {input_cell_states.shape}")
    V, D = cp_embed.shape
    P, D_pos = position_embed.shape
    if P == 0:
        raise ValueError("position_embed must have at least one row")
    if D != D_pos:
        raise ValueError("position_embed embedding dim mismatch")

    states = input_cell_states.to(torch.int32)
    cp_raw = states[:, CELL_CODEPOINT_CHANNEL_IDX]
    fg_rgb = states[:, CELL_FG_COLOR_CHANNEL_IDX]
    bg_rgb = states[:, CELL_BG_COLOR_CHANNEL_IDX]

    cp_clamped = torch.minimum(cp_raw, torch.tensor(V - 1, dtype=torch.int32, device=states.device))
    cp_valid = (cp_raw >= 0) & (cp_raw < V)
    cp_vec = cp_embed[cp_clamped.to(torch.long)]
    cp_vec = torch.where(cp_valid.view(-1, 1), cp_vec, torch.zeros_like(cp_vec))

    def rgb_to_unit(rgb):
        r = ((rgb >> 16) & 0xFF).to(torch.float32)
        g = ((rgb >> 8) & 0xFF).to(torch.float32)
        b = (rgb & 0xFF).to(torch.float32)
        inv255 = 1.0 / 255.0
        return r * inv255, g * inv255, b * inv255

    fg_r, fg_g, fg_b = rgb_to_unit(fg_rgb)
    bg_r, bg_g, bg_b = rgb_to_unit(bg_rgb)

    acc = cp_vec.to(torch.float32)
    pos_indices = torch.arange(input_cell_states.size(0), device=states.device, dtype=torch.int64) % P
    pos_vec = position_embed.index_select(0, pos_indices).to(torch.float32)
    acc = acc + pos_vec
    acc = acc + fg_r.view(-1, 1) * fg_r_embed.view(1, -1)
    acc = acc + fg_g.view(-1, 1) * fg_g_embed.view(1, -1)
    acc = acc + fg_b.view(-1, 1) * fg_b_embed.view(1, -1)
    acc = acc + bg_r.view(-1, 1) * bg_r_embed.view(1, -1)
    acc = acc + bg_g.view(-1, 1) * bg_g_embed.view(1, -1)
    acc = acc + bg_b.view(-1, 1) * bg_b_embed.view(1, -1)

    if out_dtype is None:
        out_dtype = cp_embed.dtype
    return acc.to(dtype=out_dtype, copy=True, memory_format=torch.contiguous_format)
