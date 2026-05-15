from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from .compile import ensure_workdir
from .runtime import load_cutlass_config, run_cutlass_conv2d


def _load_enabled_configs(path: Path) -> list[tuple[str, dict[str, object]]]:
    payload = json.loads(path.read_text())
    conv2d = payload.get("conv2d", {})
    if not isinstance(conv2d, dict):
        raise RuntimeError(f"Autotune config '{path}' is missing a 'conv2d' mapping.")
    return [(name, spec) for name, spec in conv2d.items() if spec.get("enabled", True)]


def _generate_inputs(spec: dict[str, object], device: torch.device, dtype: torch.dtype):
    batch = int(spec["batch"])
    in_channels = int(spec["in_channels"])
    out_channels = int(spec["out_channels"])
    height = int(spec["height"])
    width = int(spec["width"])
    kernel_h = int(spec.get("kernel_h", 3))
    kernel_w = int(spec.get("kernel_w", 3))

    input_scale = 1.0 / max(1.0, math.sqrt(in_channels))
    weight_scale = 1.0 / max(1.0, math.sqrt(in_channels * kernel_h * kernel_w))

    x = torch.randn((batch, in_channels, height, width), device=device, dtype=torch.float32)
    x = (x * input_scale).to(dtype)

    weight = torch.randn((out_channels, in_channels, kernel_h, kernel_w), device=device, dtype=torch.float32)
    weight = (weight * weight_scale).to(dtype)

    bias = torch.randn((out_channels,), device=device, dtype=torch.float32)
    bias = (bias * weight_scale).to(dtype)

    return x, weight, bias


def validate_cutlass_conv2d(*, autotune_config: Path, seed: int = 0) -> None:
    ensure_workdir()
    os.environ["FBAMTRAIN_CUTLASS_AUTOTUNE_CONFIG"] = str(autotune_config)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required to validate the CUTLASS Conv2d kernels")

    enabled_configs = _load_enabled_configs(autotune_config)
    if not enabled_configs:
        raise RuntimeError("No CUTLASS conv2d configurations are enabled for validation.")

    device = torch.device("cuda", torch.cuda.current_device())
    dtype = torch.bfloat16

    for idx, (name, spec) in enumerate(enabled_configs):
        torch.manual_seed(seed + idx)
        print(f"\n=== Validating configuration: {name} ===")

        load_cutlass_config(name)

        x, weight, bias = _generate_inputs(spec, device, dtype)
        x_cutlass = x.contiguous(memory_format=torch.channels_last)
        weight_cutlass = weight.contiguous(memory_format=torch.channels_last)

        stride = int(spec["stride"])
        padding = int(spec["padding"])
        dilation = int(spec["dilation"])

        stride_tuple = (stride, stride)
        padding_tuple = (padding, padding)
        dilation_tuple = (dilation, dilation)

        cutlass_out = run_cutlass_conv2d(
            x_cutlass,
            weight_cutlass,
            bias=bias,
            stride=stride_tuple,
            padding=padding_tuple,
            dilation=dilation_tuple,
        )

        torch_out = F.conv2d(
            x,
            weight,
            bias=bias,
            stride=stride_tuple,
            padding=padding_tuple,
            dilation=dilation_tuple,
        ).to(dtype).contiguous(memory_format=torch.channels_last)

        diff = (cutlass_out - torch_out).abs().float()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        print(f" Max abs diff: {max_diff:.6f}")
        print(f" Mean abs diff: {mean_diff:.6f}")


def main(args: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Validate CUTLASS conv2d kernels against PyTorch references.")
    parser.add_argument(
        "--autotune-config",
        type=Path,
        required=True,
        help="Path to the tuned CUTLASS config JSON (e.g. autotune_constants_sm90.json).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parsed = parser.parse_args(args=args)

    validate_cutlass_conv2d(autotune_config=parsed.autotune_config, seed=parsed.seed)


if __name__ == "__main__":  # pragma: no cover
    main()
