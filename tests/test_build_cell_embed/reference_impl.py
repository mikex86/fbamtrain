import torch
import triton
import triton.language as tl
from safetensors.torch import save_file

BATCH_SIZE = 2
FRAME_ROWS = 2
FRAME_COLS = 3
EMBED_DIM = 8
MODEL_INIT_SEED = 123
NUM_CHANNELS_PER_CELL = 3
MAX_CODEPOINT = 255
BLOCK_SIZE = 1024


@triton.jit
def fill_normal(
        out_ptr,  # *float32
        n_elements: tl.uint32,
        mean: tl.float32,
        std: tl.float32,
        seed: tl.uint32,
        BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    nonce = 0x9E3779B9

    u = tl.rand(seed, offs)
    v = tl.rand(seed ^ nonce.to(tl.uint32), offs)

    u = tl.where(mask, tl.maximum(u, 1e-7), 0.5)
    v = tl.where(mask, v, 0.0)

    r = tl.sqrt(-2.0 * tl.log(u))
    theta = 6.283185307179586 * v
    z = r * tl.cos(theta)

    out = mean + std * z
    tl.store(out_ptr + offs, out, mask=mask)


def fill_normal_(tensor: torch.Tensor, mean: float, std: float, seed: int):
    n_elements = tensor.numel()
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    fill_normal[grid](tensor, n_elements, mean, std, seed, BLOCK_SIZE=BLOCK_SIZE)


def compute_cell_states(device: torch.device) -> torch.Tensor:
    n_cells = BATCH_SIZE * FRAME_ROWS * FRAME_COLS
    idx = torch.arange(n_cells, device=device, dtype=torch.int64)

    cp = ((idx * 13 + 7) % MAX_CODEPOINT).to(torch.int32)

    def make_color(mul_r: int, bias_r: int, mul_g: int, bias_g: int, mul_b: int, bias_b: int) -> torch.Tensor:
        r = ((idx * mul_r + bias_r) % 256).to(torch.int32)
        g = ((idx * mul_g + bias_g) % 256).to(torch.int32)
        b = ((idx * mul_b + bias_b) % 256).to(torch.int32)
        return (r << 16) | (g << 8) | b

    fg = make_color(17, 5, 19, 11, 23, 13)
    bg = make_color(29, 17, 31, 19, 37, 23)

    states = torch.stack([cp, fg, bg], dim=-1)
    return states.view(BATCH_SIZE, FRAME_ROWS, FRAME_COLS, NUM_CHANNELS_PER_CELL).to(torch.int32)


def build_cell_embeds(states: torch.Tensor,
                      cp_embed: torch.Tensor,
                      position_embed: torch.Tensor,
                      fg_r_embed: torch.Tensor,
                      fg_g_embed: torch.Tensor,
                      fg_b_embed: torch.Tensor,
                      bg_r_embed: torch.Tensor,
                      bg_g_embed: torch.Tensor,
                      bg_b_embed: torch.Tensor) -> torch.Tensor:
    states_flat = states.view(-1, NUM_CHANNELS_PER_CELL).to(torch.int32)
    cp_raw = states_flat[:, 0]
    fg_rgb = states_flat[:, 1]
    bg_rgb = states_flat[:, 2]

    vocab_size = cp_embed.size(0)
    cp_clamped = torch.minimum(cp_raw, torch.tensor(vocab_size - 1, device=states.device, dtype=torch.int32))
    cp_valid = (cp_raw >= 0) & (cp_raw < vocab_size)

    cp_vec = cp_embed[cp_clamped.to(torch.long)]
    cp_vec = torch.where(cp_valid.view(-1, 1), cp_vec, torch.zeros_like(cp_vec))

    def rgb_to_unit(rgb: torch.Tensor):
        rgb = rgb.to(torch.int32)
        r = ((rgb >> 16) & 0xFF).to(torch.float32) / 255.0
        g = ((rgb >> 8) & 0xFF).to(torch.float32) / 255.0
        b = (rgb & 0xFF).to(torch.float32) / 255.0
        return r, g, b

    fg_r, fg_g, fg_b = rgb_to_unit(fg_rgb)
    bg_r, bg_g, bg_b = rgb_to_unit(bg_rgb)

    acc = cp_vec.to(torch.float32)
    pos_dim = position_embed.size(0)
    pos_indices = torch.arange(states_flat.size(0), device=states.device, dtype=torch.int64) % pos_dim
    pos_vec = position_embed.index_select(0, pos_indices).to(torch.float32)
    acc = acc + pos_vec

    fg_r_vec = fg_r_embed.to(torch.float32)
    fg_g_vec = fg_g_embed.to(torch.float32)
    fg_b_vec = fg_b_embed.to(torch.float32)
    bg_r_vec = bg_r_embed.to(torch.float32)
    bg_g_vec = bg_g_embed.to(torch.float32)
    bg_b_vec = bg_b_embed.to(torch.float32)

    acc = acc + fg_r.view(-1, 1) * fg_r_vec.view(1, -1)
    acc = acc + fg_g.view(-1, 1) * fg_g_vec.view(1, -1)
    acc = acc + fg_b.view(-1, 1) * fg_b_vec.view(1, -1)
    acc = acc + bg_r.view(-1, 1) * bg_r_vec.view(1, -1)
    acc = acc + bg_g.view(-1, 1) * bg_g_vec.view(1, -1)
    acc = acc + bg_b.view(-1, 1) * bg_b_vec.view(1, -1)

    return acc.to(dtype=torch.bfloat16).view(BATCH_SIZE, FRAME_ROWS, FRAME_COLS, EMBED_DIM)


def main():
    device = torch.device('cuda')
    dtype = torch.bfloat16

    cp_embed = torch.empty((MAX_CODEPOINT, EMBED_DIM), dtype=dtype, device=device)
    position_embed = torch.empty((FRAME_ROWS * FRAME_COLS, EMBED_DIM), dtype=dtype, device=device)
    bg_r_embed = torch.empty((EMBED_DIM,), dtype=dtype, device=device)
    bg_g_embed = torch.empty((EMBED_DIM,), dtype=dtype, device=device)
    bg_b_embed = torch.empty((EMBED_DIM,), dtype=dtype, device=device)
    fg_r_embed = torch.empty((EMBED_DIM,), dtype=dtype, device=device)
    fg_g_embed = torch.empty((EMBED_DIM,), dtype=dtype, device=device)
    fg_b_embed = torch.empty((EMBED_DIM,), dtype=dtype, device=device)

    seed = MODEL_INIT_SEED
    fill_normal_(cp_embed, 0.0, 1.0, seed)
    seed += 1
    fill_normal_(position_embed, 0.0, 1.0, seed)
    seed += 1
    fill_normal_(bg_r_embed, 0.0, 1.0, seed)
    seed += 1
    fill_normal_(bg_g_embed, 0.0, 1.0, seed)
    seed += 1
    fill_normal_(bg_b_embed, 0.0, 1.0, seed)
    seed += 1
    fill_normal_(fg_r_embed, 0.0, 1.0, seed)
    seed += 1
    fill_normal_(fg_g_embed, 0.0, 1.0, seed)
    seed += 1
    fill_normal_(fg_b_embed, 0.0, 1.0, seed)
    seed += 1

    cell_states = compute_cell_states(device)
    output = build_cell_embeds(cell_states, cp_embed, position_embed, fg_r_embed, fg_g_embed, fg_b_embed,
                               bg_r_embed, bg_g_embed, bg_b_embed)
    save_file({'output': output.cpu()}, 'reference.safetensors')


if __name__ == '__main__':
    main()
