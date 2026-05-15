from dataclasses import dataclass

import torch
import torch.nn
import torch.nn.functional as F

import init_utils
from build_cell_embeds import build_cell_embeds_autograd
from build_cell_embeds_fast import NUM_CHANNELS_PER_CELL, build_cell_embeds as build_cell_embeds_fast

MAX_BYTE = 255


@dataclass
class FrameHeadConfig:
    n_layer: int
    n_head: int
    n_embd: int
    bias: bool
    downsample_blocks: int
    rms_norm_eps: float
    downsample_conv_mode: str
    downsample_conv_dilation: int


class FullSelfAttention(torch.nn.Module):

    def __init__(self, config: FrameHeadConfig, device: torch.device, dtype: torch.dtype, seed: int):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError(f"Embedding dimension {config.n_embd} not divisible by number of heads {config.n_head}.")
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.c_attn = torch.nn.Linear(
            config.n_embd,
            3 * config.n_embd,
            bias=config.bias,
            device=device,
            dtype=dtype,
        )
        self.c_proj = torch.nn.Linear(
            config.n_embd,
            config.n_embd,
            bias=config.bias,
            device=device,
            dtype=dtype,
        )
        self._next_seed = self.reset_parameters(seed)

    def reset_parameters(self, seed: int) -> int:
        fan_in = self.n_embd
        with torch.no_grad():
            weight_t = self.c_attn.weight.transpose(0, 1).contiguous()
            init_utils.init_kaiming_uniform(weight_t, fan_in, seed)
            self.c_attn.weight.copy_(weight_t.transpose(0, 1))
        seed += 1
        if self.c_attn.bias is not None:
            init_utils.init_kaiming_uniform(self.c_attn.bias, fan_in, seed)
            seed += 1

        with torch.no_grad():
            weight_t = self.c_proj.weight.transpose(0, 1).contiguous()
            init_utils.init_kaiming_uniform(weight_t, self.n_embd, seed)
            self.c_proj.weight.copy_(weight_t.transpose(0, 1))
        seed += 1
        if self.c_proj.bias is not None:
            init_utils.init_kaiming_uniform(self.c_proj.bias, self.n_embd, seed)
            seed += 1
        return seed

    @property
    def next_seed(self) -> int:
        return self._next_seed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.size()

        qkv = self.c_attn(x)
        q, k, v = torch.chunk(qkv, 3, dim=2)

        head_dim = c // self.n_head
        q = q.view(b, t, self.n_head, head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_head, head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=False,
        )
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        y = self.c_proj(y)
        return y


class MLP(torch.nn.Module):

    def __init__(self, config: FrameHeadConfig, device: torch.device, dtype: torch.dtype, seed: int):
        super().__init__()
        hidden_dim = 4 * config.n_embd
        self.c_fc = torch.nn.Linear(
            config.n_embd,
            hidden_dim,
            bias=config.bias,
            device=device,
            dtype=dtype,
        )
        self.gelu = torch.nn.GELU()
        self.c_proj = torch.nn.Linear(
            hidden_dim,
            config.n_embd,
            bias=config.bias,
            device=device,
            dtype=dtype,
        )
        self._next_seed = self.reset_parameters(seed, config.n_embd, hidden_dim)

    def reset_parameters(self, seed: int, in_features: int, hidden_dim: int) -> int:
        with torch.no_grad():
            weight_t = self.c_fc.weight.transpose(0, 1).contiguous()
            init_utils.init_kaiming_uniform(weight_t, in_features, seed)
            self.c_fc.weight.copy_(weight_t.transpose(0, 1))
        seed += 1
        if self.c_fc.bias is not None:
            init_utils.init_kaiming_uniform(self.c_fc.bias, in_features, seed)
            seed += 1

        with torch.no_grad():
            weight_t = self.c_proj.weight.transpose(0, 1).contiguous()
            init_utils.init_kaiming_uniform(weight_t, hidden_dim, seed)
            self.c_proj.weight.copy_(weight_t.transpose(0, 1))
        seed += 1
        if self.c_proj.bias is not None:
            init_utils.init_kaiming_uniform(self.c_proj.bias, hidden_dim, seed)
            seed += 1
        return seed

    @property
    def next_seed(self) -> int:
        return self._next_seed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(torch.nn.Module):

    def __init__(self, config: FrameHeadConfig, device: torch.device, dtype: torch.dtype, seed: int):
        super().__init__()
        self.ln_1 = torch.nn.RMSNorm(config.n_embd, device=device, dtype=dtype, eps=config.rms_norm_eps)
        with torch.no_grad():
            self.ln_1.weight.fill_(1.0)

        self.attn = FullSelfAttention(config, device=device, dtype=dtype, seed=seed)
        seed = self.attn.next_seed

        self.ln_2 = torch.nn.RMSNorm(config.n_embd, device=device, dtype=dtype, eps=config.rms_norm_eps)

        with torch.no_grad():
            self.ln_2.weight.fill_(1.0)

        self.mlp = MLP(config, device=device, dtype=dtype, seed=seed)
        seed = self.mlp.next_seed
        self._next_seed = seed

    @property
    def next_seed(self) -> int:
        return self._next_seed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(self.ln_1(x))
        x = x + attn_out
        mlp_out = self.mlp(self.ln_2(x))
        x = x + mlp_out
        return x


class ConvDownsampleBlock(torch.nn.Module):

    def __init__(self, config: FrameHeadConfig, device: torch.device, dtype: torch.dtype, seed: int):
        super().__init__()
        self.mode = config.downsample_conv_mode.lower()
        if self.mode not in {"dilated", "geglu"}:
            raise ValueError(f"Unsupported downsample conv mode: {config.downsample_conv_mode}")

        self.dilation = int(config.downsample_conv_dilation)
        if self.dilation <= 0:
            raise ValueError("downsample_conv_dilation must be positive")

        self.ln_1 = torch.nn.RMSNorm(config.n_embd, device=device, dtype=dtype, eps=config.rms_norm_eps)
        with torch.no_grad():
            self.ln_1.weight.fill_(1.0)

        if self.mode == "dilated":
            out_channels = config.n_embd
            padding = self.dilation
            dilation = self.dilation
        else:
            out_channels = 2 * config.n_embd
            padding = 1
            dilation = 1

        self.conv = torch.nn.Conv2d(
            config.n_embd,
            out_channels,
            kernel_size=3,
            padding=padding,
            dilation=dilation,
            bias=False,
            device=device,
            dtype=dtype,
        )

        fan_in = config.n_embd * 3 * 3
        with torch.no_grad():
            weight_ohwc = torch.empty(
                (out_channels, 3, 3, config.n_embd), dtype=dtype, device=device
            )
            init_utils.init_kaiming_uniform(weight_ohwc, fan_in, seed)
            self.conv.weight.copy_(weight_ohwc.permute(0, 3, 1, 2))
        seed += 1
        self._next_seed = seed

    @property
    def next_seed(self) -> int:
        return self._next_seed

    def forward(self, x: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"ConvDownsampleBlock expects input of shape (B, T, C); got {x.shape}")
        b, seq, c = x.shape
        if seq != rows * cols:
            raise ValueError(
                f"ConvDownsampleBlock expects sequence length {rows * cols} but received {seq}"
            )

        x = self.ln_1(x)
        x_img = x.reshape(b, rows, cols, c).permute(0, 3, 1, 2)
        conv_out = self.conv(x_img)

        if self.mode == "geglu":
            val, gate = conv_out.chunk(2, dim=1)
            activated = val * F.gelu(gate, approximate="none")
        else:
            activated = F.gelu(conv_out, approximate="none")

        activated = activated.to(x.dtype)
        return activated.permute(0, 2, 3, 1).reshape(b, seq, c)


class FrameHeadModule(torch.nn.Module):

    def __init__(
            self,
            config: "FbamModelConfiguration",
            dtype: torch.dtype,
            device: torch.device,
            use_autograd_cell_embeds: bool = False,
    ):
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.rows = int(config.frame_rows)
        self.cols = int(config.frame_cols)
        self.n_embed = int(config.n_embed)
        self.tokens_per_frame = self.rows * self.cols
        self.max_code_point = int(config.max_code_point)
        self.frame_head_reduction_strategy = str(
            getattr(config, "frame_head_reduction_strategy", "last_pos")
        ).lower()
        if self.frame_head_reduction_strategy not in ("mean", "last_pos"):
            raise ValueError(
                "frame_head_reduction_strategy must be 'mean' or 'last_pos'. "
                f"Got '{self.frame_head_reduction_strategy}'."
            )
        self.build_cell_embeds = build_cell_embeds_autograd if use_autograd_cell_embeds else build_cell_embeds_fast

        frame_head_config = FrameHeadConfig(
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_embd=config.n_embed,
            bias=config.bias,
            downsample_blocks=config.downsample_blocks,
            rms_norm_eps=float(config.rms_norm_eps),
            downsample_conv_mode=getattr(config, "downsample_conv_mode", "dilated"),
            downsample_conv_dilation=int(getattr(config, "downsample_conv_dilation", 2)),
        )

        self.codepoint_embed = torch.nn.Parameter(
            torch.empty(self.max_code_point, self.n_embed, dtype=dtype, device=device)
        )

        self.bg_r_embed = torch.nn.Parameter(torch.empty(self.n_embed, dtype=dtype, device=device))
        self.bg_g_embed = torch.nn.Parameter(torch.empty(self.n_embed, dtype=dtype, device=device))
        self.bg_b_embed = torch.nn.Parameter(torch.empty(self.n_embed, dtype=dtype, device=device))

        self.fg_r_embed = torch.nn.Parameter(torch.empty(self.n_embed, dtype=dtype, device=device))
        self.fg_g_embed = torch.nn.Parameter(torch.empty(self.n_embed, dtype=dtype, device=device))
        self.fg_b_embed = torch.nn.Parameter(torch.empty(self.n_embed, dtype=dtype, device=device))

        self.position_embed = torch.nn.Parameter(
            torch.empty(self.tokens_per_frame, self.n_embed, dtype=dtype, device=device)
        )

        seed = int(config.model_init_seed)
        init_utils.init_normal(self.position_embed, mean=0.0, std=1.0, seed=seed)

        init_utils.init_normal(self.codepoint_embed, mean=0.0, std=1.0, seed=seed)
        seed += 1

        init_utils.init_normal(self.fg_r_embed, mean=0.0, std=1.0, seed=seed)
        seed += 1
        init_utils.init_normal(self.fg_g_embed, mean=0.0, std=1.0, seed=seed)
        seed += 1
        init_utils.init_normal(self.fg_b_embed, mean=0.0, std=1.0, seed=seed)
        seed += 1

        init_utils.init_normal(self.bg_r_embed, mean=0.0, std=1.0, seed=seed)
        seed += 1
        init_utils.init_normal(self.bg_g_embed, mean=0.0, std=1.0, seed=seed)
        seed += 1
        init_utils.init_normal(self.bg_b_embed, mean=0.0, std=1.0, seed=seed)
        seed += 1

        self.pool = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.downsample_blocks = torch.nn.ModuleList()
        for _ in range(frame_head_config.downsample_blocks):
            block = ConvDownsampleBlock(frame_head_config, device=device, dtype=dtype, seed=seed)
            seed = block.next_seed
            self.downsample_blocks.append(block)

        self.blocks = torch.nn.ModuleList()
        for _ in range(frame_head_config.n_layer):
            block = Block(frame_head_config, device=device, dtype=dtype, seed=seed)
            seed = block.next_seed
            self.blocks.append(block)

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"FrameHead initialized with {total_params} trainable parameters.")

    def forward(self, cell_states: torch.Tensor) -> torch.Tensor:
        if cell_states.dim() != 4:
            raise ValueError(f"Expected cell_states to be rank-4, got shape {cell_states.shape}")

        batch, rows, cols, channels = cell_states.shape
        if rows != self.rows or cols != self.cols:
            raise ValueError(
                f"Cell states size mismatch: model expects {self.rows}x{self.cols}, "
                f"but received {rows}x{cols}"
            )
        if channels != NUM_CHANNELS_PER_CELL:
            raise ValueError(
                f"Expected {NUM_CHANNELS_PER_CELL} frame channels, but received {channels}"
            )

        tmp_rows, tmp_cols = self.rows, self.cols
        for _ in self.downsample_blocks:
            if tmp_rows % 2 != 0 or tmp_cols % 2 != 0:
                raise ValueError("FrameHeadModule requires frame dimensions divisible by 2^downsample_blocks")
            tmp_rows //= 2
            tmp_cols //= 2
            if tmp_rows == 0 or tmp_cols == 0:
                raise ValueError("FrameHeadModule downsampling reduces spatial dimensions to zero")

        flat_states = cell_states.view(-1, channels)
        cell_embeddings = self.build_cell_embeds(
            flat_states,
            self.codepoint_embed,
            self.position_embed,
            self.fg_r_embed,
            self.fg_g_embed,
            self.fg_b_embed,
            self.bg_r_embed,
            self.bg_g_embed,
            self.bg_b_embed,
        )

        x = cell_embeddings.view(batch, self.tokens_per_frame, self.n_embed)

        current_rows, current_cols = self.rows, self.cols
        for block in self.downsample_blocks:
            x = block(x, current_rows, current_cols)
            if current_rows % 2 != 0 or current_cols % 2 != 0:
                raise ValueError("FrameHeadModule encountered odd spatial dimension during pooling")
            b, _, c = x.shape
            x_image = x.view(b, current_rows, current_cols, c).permute(0, 3, 1, 2).contiguous()
            x_image = self.pool(x_image)
            current_rows //= 2
            current_cols //= 2
            x = x_image.permute(0, 2, 3, 1).contiguous().view(b, current_rows * current_cols, c)

        for block in self.blocks:
            x = block(x)

        if self.frame_head_reduction_strategy == "mean":
            frame_embeddings = x.mean(dim=1)
        else:
            frame_embeddings = x[:, -1, :]
        return frame_embeddings.to(dtype=self.dtype)
