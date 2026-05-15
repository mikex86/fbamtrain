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


def _bench(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


def benchmark_cutlass_vs_pytorch(*, autotune_config: Path, iters: int = 100, warmup: int = 25, seed: int = 0) -> None:
    ensure_workdir()
    os.environ["FBAMTRAIN_CUTLASS_AUTOTUNE_CONFIG"] = str(autotune_config)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required to run the CUTLASS benchmark")

    enabled_configs = _load_enabled_configs(autotune_config)
    if not enabled_configs:
        raise RuntimeError("No CUTLASS conv2d configurations are enabled for benchmarking.")

    device = torch.device("cuda", torch.cuda.current_device())
    dtype = torch.bfloat16

    for idx, (name, spec) in enumerate(enabled_configs):
        torch.manual_seed(seed + idx)
        print(f"\n=== Benchmarking configuration: {name} ===")

        load_cutlass_config(name)

        x, weight, bias = _generate_inputs(spec, device, dtype)
        x_cutlass = x.contiguous(memory_format=torch.channels_last)
        weight_cutlass = weight.contiguous(memory_format=torch.channels_last)

        stride = int(spec["stride"])
        padding = int(spec["padding"])
        dilation = int(spec["dilation"])
        kernel_h = int(spec.get("kernel_h", 3))
        kernel_w = int(spec.get("kernel_w", 3))
        batch = int(spec["batch"])
        in_channels = int(spec["in_channels"])
        out_channels = int(spec["out_channels"])
        height = int(spec["height"])
        width = int(spec["width"])

        stride_tuple = (stride, stride)
        padding_tuple = (padding, padding)
        dilation_tuple = (dilation, dilation)

        def run_torch_case() -> torch.Tensor:
            return F.conv2d(
                x,
                weight,
                bias=bias,
                stride=stride_tuple,
                padding=padding_tuple,
                dilation=dilation_tuple,
            ).to(dtype).contiguous(memory_format=torch.channels_last)

        def run_cutlass_case() -> torch.Tensor:
            return run_cutlass_conv2d(
                x_cutlass,
                weight_cutlass,
                bias=bias,
                stride=stride_tuple,
                padding=padding_tuple,
                dilation=dilation_tuple,
            )

        torch_ms = _bench(run_torch_case, iters, warmup)
        cutlass_ms = _bench(run_cutlass_case, iters, warmup)

        with torch.no_grad():
            cutlass_out = run_cutlass_case()
            torch_out = run_torch_case()
            diff = (cutlass_out - torch_out).abs().float()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            print(f" Validation | max abs diff: {max_diff:.6f}, mean abs diff: {mean_diff:.6f}")

        eff_kernel_h = dilation * (kernel_h - 1) + 1
        eff_kernel_w = dilation * (kernel_w - 1) + 1
        out_h = (height + 2 * padding - eff_kernel_h) // stride + 1
        out_w = (width + 2 * padding - eff_kernel_w) // stride + 1

        total_flops = (
            2.0
            * batch
            * out_channels
            * out_h
            * out_w
            * kernel_h
            * kernel_w
            * in_channels
        )

        cutlass_tflops = total_flops / 1.0e12 / (cutlass_ms / 1.0e3)
        torch_tflops = total_flops / 1.0e12 / (torch_ms / 1.0e3)

        print(" Performance (bf16)")
        print(f"  CUTLASS: {cutlass_ms:.3f} ms | {cutlass_tflops:.2f} TFLOP/s")
        print(f"  PyTorch: {torch_ms:.3f} ms | {torch_tflops:.2f} TFLOP/s")


def main(args: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark CUTLASS vs PyTorch across tuned conv2d workloads.")
    parser.add_argument(
        "--autotune-config",
        type=Path,
        required=True,
        help="Path to the tuned CUTLASS config JSON (e.g. autotune_constants_sm90.json).",
    )
    parser.add_argument("--iters", type=int, default=100, help="Number of timed repetitions per configuration.")
    parser.add_argument("--warmup", type=int, default=25, help="Number of warmup iterations per configuration.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parsed = parser.parse_args(args=args)

    benchmark_cutlass_vs_pytorch(
        autotune_config=parsed.autotune_config,
        iters=parsed.iters,
        warmup=parsed.warmup,
        seed=parsed.seed,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
