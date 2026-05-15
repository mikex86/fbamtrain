from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import torch
from safetensors.torch import save_file

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TOYIMPL = os.path.join(ROOT, "toyimpl", "src")
sys.path.insert(0, TOYIMPL)

import model as model_module  # noqa: E402


@dataclass
class FbamModelConfiguration:
    frame_cols: int
    frame_rows: int
    n_embed: int
    n_layer: int
    n_head: int
    bias: bool
    downsample_blocks: int
    max_code_point: int
    rms_norm_eps: float
    model_init_seed: int = 1337
    downsample_conv_mode: str = "dilated"
    downsample_conv_dilation: int = 2
    use_fp16_accumulation: bool = False
    frame_head_reduction_strategy: str = "mean"


BATCH_SIZE = 2
ROWS = 8
COLS = 8
MAX_CODE_POINT = 128
NUM_HEADS = 8
NUM_LAYERS = 1
NUM_DOWNSAMPLE_BLOCKS = 2
N_EMBD = 1024
RMS_EPS = 1e-5
MODEL_INIT_SEED = 1337


class RmsNormFallback(torch.nn.Module):
    def __init__(self, normalized_shape: int, device: torch.device | None = None, dtype: torch.dtype | None = None,
                 eps: float = 1e-5):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(normalized_shape, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(rms + self.eps)
        return x_norm * self.weight


if not hasattr(torch.nn, "RMSNorm"):
    torch.nn.RMSNorm = RmsNormFallback


def build_cell_embeds_autograd(
    input_cell_states: torch.Tensor,
    cp_embed: torch.Tensor,
    position_embed: torch.Tensor,
    fg_r_embed: torch.Tensor,
    fg_g_embed: torch.Tensor,
    fg_b_embed: torch.Tensor,
    bg_r_embed: torch.Tensor,
    bg_g_embed: torch.Tensor,
    bg_b_embed: torch.Tensor,
) -> torch.Tensor:
    states = input_cell_states.to(torch.int32)
    cp_raw = states[:, 0]
    fg_rgb = states[:, 1]
    bg_rgb = states[:, 2]

    vocab_size, embed_dim = cp_embed.shape
    max_positions = position_embed.shape[0]
    if max_positions == 0:
        raise ValueError("position_embed must have at least one row")

    cp_clamped = torch.minimum(cp_raw, torch.tensor(vocab_size - 1, device=states.device, dtype=torch.int32))
    cp_valid = (cp_raw >= 0) & (cp_raw < vocab_size)
    cp_vec = cp_embed[cp_clamped.to(torch.long)]
    cp_vec = torch.where(cp_valid.view(-1, 1), cp_vec, torch.zeros_like(cp_vec))

    def rgb_to_unit(rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r = ((rgb >> 16) & 0xFF).to(torch.float32)
        g = ((rgb >> 8) & 0xFF).to(torch.float32)
        b = (rgb & 0xFF).to(torch.float32)
        inv255 = 1.0 / 255.0
        return r * inv255, g * inv255, b * inv255

    fg_r, fg_g, fg_b = rgb_to_unit(fg_rgb)
    bg_r, bg_g, bg_b = rgb_to_unit(bg_rgb)

    acc = cp_vec.to(torch.float32)
    pos_indices = torch.arange(states.size(0), device=states.device, dtype=torch.int64) % max_positions
    pos_vec = position_embed.index_select(0, pos_indices).to(torch.float32)
    acc = acc + pos_vec
    acc = acc + fg_r.view(-1, 1) * fg_r_embed.view(1, -1)
    acc = acc + fg_g.view(-1, 1) * fg_g_embed.view(1, -1)
    acc = acc + fg_b.view(-1, 1) * fg_b_embed.view(1, -1)
    acc = acc + bg_r.view(-1, 1) * bg_r_embed.view(1, -1)
    acc = acc + bg_g.view(-1, 1) * bg_g_embed.view(1, -1)
    acc = acc + bg_b.view(-1, 1) * bg_b_embed.view(1, -1)
    return acc.to(dtype=cp_embed.dtype, copy=True, memory_format=torch.contiguous_format)


model_module.build_cell_embeds = build_cell_embeds_autograd
NUM_CHANNELS_PER_CELL = model_module.NUM_CHANNELS_PER_CELL


def compute_codepoint(index: torch.Tensor) -> torch.Tensor:
    return (index * 13 + 7) % MAX_CODE_POINT


def compute_color(
    index: torch.Tensor,
    r_mul: int,
    r_bias: int,
    g_mul: int,
    g_bias: int,
    b_mul: int,
    b_bias: int,
) -> torch.Tensor:
    r = (index * r_mul + r_bias) % 256
    g = (index * g_mul + g_bias) % 256
    b = (index * b_mul + b_bias) % 256
    return (r << 16) | (g << 8) | b


def build_cell_states(device: torch.device) -> torch.Tensor:
    total = BATCH_SIZE * ROWS * COLS
    idx = torch.arange(total, device=device, dtype=torch.int64)
    cp = compute_codepoint(idx)
    fg = compute_color(idx, 17, 5, 19, 11, 23, 13)
    bg = compute_color(idx, 29, 17, 31, 19, 37, 23)
    states = torch.stack([cp, fg, bg], dim=1).to(torch.int32)
    return states.view(BATCH_SIZE, ROWS, COLS, NUM_CHANNELS_PER_CELL)


def build_upstream(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    total = BATCH_SIZE * N_EMBD
    idx = torch.arange(total, device=device, dtype=torch.int64)
    vals = ((idx * 13 + 7) % 257).to(torch.float32) / 257.0 - 0.5
    return vals.view(BATCH_SIZE, N_EMBD).to(dtype)


def extract_param(params: dict[str, torch.Tensor], name: str) -> torch.Tensor:
    if name not in params:
        raise KeyError(f"Missing parameter {name}")
    param = params[name]
    if param.grad is None:
        raise RuntimeError(f"Missing gradient for {name}")
    return param.grad.detach()


def generate(dtype: torch.dtype, output_path: str) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required to generate reference tensors")

    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

    device = torch.device("cuda")
    config = FbamModelConfiguration(
        frame_cols=COLS,
        frame_rows=ROWS,
        n_embed=N_EMBD,
        n_layer=NUM_LAYERS,
        n_head=NUM_HEADS,
        bias=True,
        downsample_blocks=NUM_DOWNSAMPLE_BLOCKS,
        max_code_point=MAX_CODE_POINT,
        rms_norm_eps=RMS_EPS,
        model_init_seed=MODEL_INIT_SEED,
        downsample_conv_mode="dilated",
        downsample_conv_dilation=2,
        use_fp16_accumulation=False,
    )

    frame_head = model_module.FrameHeadModule(config, dtype=dtype, device=device, use_autograd_cell_embeds=True)
    cell_states = build_cell_states(device)
    upstream = build_upstream(device, dtype)

    output = frame_head(cell_states)
    output.backward(upstream)

    params = dict(frame_head.named_parameters())

    tensors: dict[str, torch.Tensor] = {
        "upstream": upstream.detach().cpu(),
        "frame_head.codepoint_embedding": extract_param(params, "codepoint_embed").cpu(),
        "frame_head.position_embedding.weight": extract_param(params, "position_embed").cpu(),
        "frame_head.fg_r_embed": extract_param(params, "fg_r_embed").cpu(),
        "frame_head.fg_g_embed": extract_param(params, "fg_g_embed").cpu(),
        "frame_head.fg_b_embed": extract_param(params, "fg_b_embed").cpu(),
        "frame_head.bg_r_embed": extract_param(params, "bg_r_embed").cpu(),
        "frame_head.bg_g_embed": extract_param(params, "bg_g_embed").cpu(),
        "frame_head.bg_b_embed": extract_param(params, "bg_b_embed").cpu(),
    }

    for i in range(NUM_DOWNSAMPLE_BLOCKS):
        ln_key = f"downsample_blocks.{i}.ln_1.weight"
        conv_key = f"downsample_blocks.{i}.conv.weight"
        tensors[f"frame_head.downsample_block.{i}.ln.weight"] = extract_param(params, ln_key).cpu()
        conv_grad = extract_param(params, conv_key).permute(0, 2, 3, 1).contiguous()
        tensors[f"frame_head.downsample_block.{i}.conv.weight"] = conv_grad.cpu()

    for i in range(NUM_LAYERS):
        tensors[f"frame_head.block.{i}.ln1.weight"] = extract_param(params, f"blocks.{i}.ln_1.weight").cpu()
        tensors[f"frame_head.block.{i}.attn.w_qkv"] = extract_param(
            params, f"blocks.{i}.attn.c_attn.weight"
        ).transpose(0, 1).contiguous().cpu()
        tensors[f"frame_head.block.{i}.attn.b_qkv"] = extract_param(
            params, f"blocks.{i}.attn.c_attn.bias"
        ).cpu()
        tensors[f"frame_head.block.{i}.attn.w_proj"] = extract_param(
            params, f"blocks.{i}.attn.c_proj.weight"
        ).transpose(0, 1).contiguous().cpu()
        tensors[f"frame_head.block.{i}.attn.b_proj"] = extract_param(
            params, f"blocks.{i}.attn.c_proj.bias"
        ).cpu()
        tensors[f"frame_head.block.{i}.ln2.weight"] = extract_param(params, f"blocks.{i}.ln_2.weight").cpu()

        tensors[f"frame_head.block.{i}.mlp.fc1.weight"] = extract_param(
            params, f"blocks.{i}.mlp.c_fc.weight"
        ).transpose(0, 1).contiguous().cpu()
        tensors[f"frame_head.block.{i}.mlp.fc1.bias"] = extract_param(
            params, f"blocks.{i}.mlp.c_fc.bias"
        ).cpu()
        tensors[f"frame_head.block.{i}.mlp.fc2.weight"] = extract_param(
            params, f"blocks.{i}.mlp.c_proj.weight"
        ).transpose(0, 1).contiguous().cpu()
        tensors[f"frame_head.block.{i}.mlp.fc2.bias"] = extract_param(
            params, f"blocks.{i}.mlp.c_proj.bias"
        ).cpu()

    save_file(tensors, output_path)


def main() -> None:
    out_dir = os.path.dirname(__file__)
    generate(torch.bfloat16, os.path.join(out_dir, "reference_bf16.safetensors"))
    generate(torch.float16, os.path.join(out_dir, "reference_fp16.safetensors"))


if __name__ == "__main__":
    main()
