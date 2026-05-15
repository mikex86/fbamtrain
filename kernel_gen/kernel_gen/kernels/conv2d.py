"""
Compiles all conv2d variants.
Do not run this file directory, please use compile_conv2d.sh instead to ensure it is run with the correct virtual environment!
The reason for this is a pinned triton version. See pyproject.toml for more details.
"""
import argparse
import json
from typing import Any, Dict, Iterable, Tuple

import torch
import triton
import triton.language as tl
import triton.testing

from .cutlass_codegen.runtime import run_cutlass_conv2d


def _pair(value: int | Iterable[int]) -> Tuple[int, int]:
    if isinstance(value, Iterable):
        items = tuple(int(v) for v in value)
        if len(items) == 1:
            return items[0], items[0]
        if len(items) == 2:
            return items
        raise ValueError("Expected iterable with one or two integers")
    return int(value), int(value)


SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)


def _dtype_code(dtype: torch.dtype) -> int:
    if dtype == torch.bfloat16:
        return 0
    if dtype == torch.float16:
        return 1
    raise ValueError(f"Unsupported dtype {dtype} (expected bfloat16 or float16)")


def _ensure_supported_dtype(dtype: torch.dtype) -> None:
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError("conv2d kernel expects BF16 or FP16 tensors")


@triton.autotune(
    # Keep num_stages=1 to avoid a Triton TMEM hoist compiler bug that surfaced during AOT compilation.
    configs=[
        triton.Config(
            {"BLOCK_W": 64, "BLOCK_OC": 64, "BLOCK_K": 32},
            num_warps=4,
            num_stages=1,
        ),
        triton.Config(
            {"BLOCK_W": 128, "BLOCK_OC": 64, "BLOCK_K": 32},
            num_warps=4,
            num_stages=1,
        ),
    ],
    key=[
        "batch",
        "in_channels",
        "out_channels",
        "out_h",
        "out_w",
        "kernel_h",
        "kernel_w",
        "dilation_h",
        "dilation_w",
    ],
)
@triton.jit
def conv2d_kernel(
        input_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        batch,
        in_channels,
        in_h,
        in_w,
        out_channels,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        pad_h,
        pad_w,
        out_h,
        out_w,
        in_stride_n,
        in_stride_h,
        in_stride_w,
        in_stride_c,
        weight_stride_kh,
        weight_stride_kw,
        weight_stride_ic,
        weight_stride_oc,
        out_stride_n,
        out_stride_h,
        out_stride_w,
        out_stride_c,
        has_bias,
        BLOCK_W: tl.constexpr,
        BLOCK_OC: tl.constexpr,
        BLOCK_K: tl.constexpr,
        KERNEL_H: tl.constexpr,
        KERNEL_W: tl.constexpr,
        IN_CHANNELS: tl.constexpr,
        OUT_DTYPE_CODE: tl.constexpr,
):
    out_dtype = tl.bfloat16 if OUT_DTYPE_CODE == 0 else tl.float16
    pid_oc = tl.program_id(0)
    pid_nh = tl.program_id(1)
    pid_ow = tl.program_id(2)

    oc_offsets = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    mask_oc = oc_offsets < out_channels
    oc_mat = tl.reshape(oc_offsets, (1, BLOCK_OC))
    mask_oc_mat = tl.reshape(mask_oc, (1, BLOCK_OC))

    n = pid_nh // out_h
    oh = pid_nh % out_h
    ow_offsets = pid_ow * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_ow = ow_offsets < out_w

    base_out = n * out_stride_n + oh * out_stride_h

    acc = tl.zeros((BLOCK_W, BLOCK_OC), dtype=tl.float32)
    if has_bias != 0:
        bias_vals = tl.load(bias_ptr + oc_offsets, mask=mask_oc, other=0.0)
        acc += tl.reshape(bias_vals.to(tl.float32), (1, BLOCK_OC))

    ih_base = oh * stride_h - pad_h
    iw_base = ow_offsets * stride_w - pad_w
    weight_oc_base = oc_mat * weight_stride_oc

    n_base = n * in_stride_n

    num_ic_blocks = (IN_CHANNELS + BLOCK_K - 1) // BLOCK_K

    for ic_block_idx in range(num_ic_blocks):
        ic_offsets = ic_block_idx * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_ic = ic_offsets < IN_CHANNELS

        ic_row = tl.reshape(ic_offsets, (1, BLOCK_K))
        ic_col = tl.reshape(ic_offsets, (BLOCK_K, 1))
        input_base = (
                n_base
                + ic_row * in_stride_c
        )
        weight_ic_base = ic_col * weight_stride_ic

        for kh in tl.static_range(KERNEL_H):
            ih = ih_base + kh * dilation_h
            mask_h = (ih >= 0) & (ih < in_h)
            input_h_base = input_base + ih * in_stride_h
            weight_h_base = weight_oc_base + weight_ic_base + kh * weight_stride_kh

            for kw in tl.static_range(KERNEL_W):
                iw = iw_base + kw * dilation_w
                mask_w = (iw >= 0) & (iw < in_w) & mask_ow & mask_h

                input_ptrs = (
                        input_ptr
                        + input_h_base
                        + tl.reshape(iw, (BLOCK_W, 1)) * in_stride_w
                )
                input_mask = mask_w[:, None] & tl.reshape(mask_ic, (1, BLOCK_K))
                input_vals = tl.load(input_ptrs, mask=input_mask, other=0.0)

                weight_ptrs = weight_ptr + weight_h_base + kw * weight_stride_kw
                weight_mask = tl.reshape(mask_ic, (BLOCK_K, 1)) & mask_oc_mat
                weight_vals = tl.load(
                    weight_ptrs,
                    mask=weight_mask,
                    other=0.0,
                    cache_modifier=".ca",
                )

                dot_out = tl.dot(input_vals, weight_vals)
                acc += dot_out

    out_ptrs = (
            output_ptr
            + base_out
            + ow_offsets[:, None] * out_stride_w
            + oc_mat * out_stride_c
    )
    tl.store(out_ptrs, acc.to(out_dtype), mask=mask_ow[:, None] & mask_oc_mat)


def _conv2d_triton_channels_last(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        stride: int | Tuple[int, int] = 1,
        padding: int | Tuple[int, int] = 0,
        dilation: int | Tuple[int, int] = 1,
) -> torch.Tensor:
    dtype = x.dtype
    _ensure_supported_dtype(dtype)
    if weight.dtype != dtype:
        raise ValueError("conv2d Triton kernel expects weight dtype to match input")
    if bias is not None and bias.dtype != dtype:
        raise ValueError("conv2d Triton kernel expects bias dtype to match input when provided")
    if x.ndim != 4:
        raise ValueError("Input tensor must be NCHW (N, C, H, W)")
    if weight.ndim != 4:
        raise ValueError("Weight tensor must be (out_channels, in_channels, k_h, k_w)")
    if not x.is_contiguous(memory_format=torch.channels_last):
        raise ValueError("channels-last fast path expects channels_last-contiguous input")

    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    dilation_h, dilation_w = _pair(dilation)

    batch, in_channels, in_h, in_w = x.shape
    out_channels, kernel_in_channels, kernel_h, kernel_w = weight.shape
    if kernel_in_channels != in_channels:
        raise ValueError("Weight in_channels must match input in_channels")

    eff_kernel_h = dilation_h * (kernel_h - 1) + 1
    eff_kernel_w = dilation_w * (kernel_w - 1) + 1

    out_h = (in_h + 2 * pad_h - eff_kernel_h)
    out_w = (in_w + 2 * pad_w - eff_kernel_w)
    if out_h < 0 or out_w < 0:
        raise ValueError("Invalid stride/padding/dilation/kernel combination producing negative output")
    if out_h % stride_h != 0 or out_w % stride_w != 0:
        raise ValueError("Invalid stride value producing fractional output dimension")

    out_h = out_h // stride_h + 1
    out_w = out_w // stride_w + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError("Invalid stride/padding/dilation/kernel combination producing empty output")

    x_nhwc = x.permute(0, 2, 3, 1)
    weight_hwio = weight.permute(2, 3, 1, 0).contiguous()
    y_nhwc = torch.empty((batch, out_h, out_w, out_channels), dtype=dtype, device=x.device)

    in_strides = x_nhwc.stride()
    weight_strides = weight_hwio.stride()
    out_strides = y_nhwc.stride()

    grid = lambda meta: (
        triton.cdiv(out_channels, meta["BLOCK_OC"]),
        batch * out_h,
        triton.cdiv(out_w, meta["BLOCK_W"]),
    )

    has_bias = 1 if bias is not None else 0
    bias_tensor = bias if bias is not None else torch.empty(1, dtype=dtype, device=x.device)

    out_dtype_code = _dtype_code(dtype)

    conv2d_kernel[grid](
        x_nhwc,
        weight_hwio,
        bias_tensor,
        y_nhwc,
        batch,
        in_channels,
        in_h,
        in_w,
        out_channels,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        pad_h,
        pad_w,
        out_h,
        out_w,
        in_strides[0],
        in_strides[1],
        in_strides[2],
        in_strides[3],
        weight_strides[0],
        weight_strides[1],
        weight_strides[2],
        weight_strides[3],
        out_strides[0],
        out_strides[1],
        out_strides[2],
        out_strides[3],
        has_bias,
        KERNEL_H=kernel_h,
        KERNEL_W=kernel_w,
        IN_CHANNELS=in_channels,
        OUT_DTYPE_CODE=out_dtype_code,
    )

    return y_nhwc.permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)


def _conv2d_triton(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        stride: int | Tuple[int, int] = 1,
        padding: int | Tuple[int, int] = 0,
        dilation: int | Tuple[int, int] = 1,
        check: bool = True,
) -> torch.Tensor:
    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    dilation_h, dilation_w = _pair(dilation)

    x_cl = x.contiguous(memory_format=torch.channels_last)
    weight_contig = weight.contiguous()

    return _conv2d_triton_channels_last(
        x_cl,
        weight_contig,
        bias=bias,
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
        dilation=(dilation_h, dilation_w),
    )


def _conv2d_cutlass(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        stride: int | Tuple[int, int] = 1,
        padding: int | Tuple[int, int] = 0,
        dilation: int | Tuple[int, int] = 1,
) -> torch.Tensor:
    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    dilation_h, dilation_w = _pair(dilation)

    if stride_h != 1 or stride_w != 1:
        raise ValueError("Cutlass kernel only supports unit stride")
    if pad_h != 2 or pad_w != 2:
        raise ValueError("Cutlass kernel only supports padding=2")
    if dilation_h != 2 or dilation_w != 2:
        raise ValueError("Cutlass kernel only supports dilation=2")

    if x.shape != (32, 1024, 48, 160):
        raise ValueError("Cutlass kernel is specialized for batch=32, channels=1024, height=48, width=160")
    if weight.shape != (1024, 1024, 3, 3):
        raise ValueError("Cutlass kernel is specialized for 1024 output/input channels with 3x3 kernels")
    if bias is not None and bias.shape != (1024,):
        raise ValueError("Cutlass kernel expects bias with shape (1024,)")

    x_cutlass = x.contiguous(memory_format=torch.channels_last)
    weight_cutlass = weight.contiguous(memory_format=torch.channels_last)

    return run_cutlass_conv2d(
        x_cutlass,
        weight_cutlass,
        bias,
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
        dilation=(dilation_h, dilation_w),
    )

def bench_conv2d(
        batch: int = 16,
        in_channels: int = 32,
        out_channels: int = 64,
        height: int = 64,
        width: int = 64,
        kernel_size: int | Tuple[int, int] = 3,
        stride: int | Tuple[int, int] = 1,
        padding: int | Tuple[int, int] = 1,
        dilation: int | Tuple[int, int] = 1,
        iters: int = 100,
        check: bool = True,
        backend: str = "triton",
        dtype: torch.dtype = torch.bfloat16,
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required to run the Conv2d benchmark")

    backend = backend.lower()
    if backend not in {"triton", "cutlass"}:
        raise ValueError(f"Unsupported backend '{backend}' (expected 'triton' or 'cutlass')")

    device = torch.device("cuda", torch.cuda.current_device())
    _ensure_supported_dtype(dtype)

    kernel_h, kernel_w = _pair(kernel_size)
    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    dilation_h, dilation_w = _pair(dilation)

    eff_kernel_h = dilation_h * (kernel_h - 1) + 1
    eff_kernel_w = dilation_w * (kernel_w - 1) + 1

    out_h = (height + 2 * pad_h - eff_kernel_h)
    out_w = (width + 2 * pad_w - eff_kernel_w)
    if out_h < 0 or out_w < 0:
        raise ValueError("Invalid stride/padding/dilation/kernel combination producing negative output")
    if out_h % stride_h != 0 or out_w % stride_w != 0:
        raise ValueError("Invalid stride value producing fractional output dimension")

    out_h = out_h // stride_h + 1
    out_w = out_w // stride_w + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError("Invalid stride/padding/dilation/kernel combination producing empty output")

    torch.manual_seed(0)
    x = torch.randn((batch, in_channels, height, width), device=device, dtype=dtype)
    weight = torch.randn((out_channels, in_channels, kernel_h, kernel_w), device=device, dtype=dtype)
    bias = torch.randn((out_channels,), device=device, dtype=dtype)

    def run_backend() -> torch.Tensor:
        if backend == "triton":
            return _conv2d_triton(
                x,
                weight,
                bias=bias,
                stride=(stride_h, stride_w),
                padding=(pad_h, pad_w),
                dilation=(dilation_h, dilation_w),
                check=check,
            )
        return _conv2d_cutlass(
            x,
            weight,
            bias=bias,
            stride=(stride_h, stride_w),
            padding=(pad_h, pad_w),
            dilation=(dilation_h, dilation_w),
        )

    def run_torch() -> torch.Tensor:
        y = torch.nn.functional.conv2d(
            x,
            weight,
            bias=bias,
            stride=(stride_h, stride_w),
            padding=(pad_h, pad_w),
            dilation=(dilation_h, dilation_w),
        )
        return y.to(dtype)

    if check:
        triton.testing.assert_close(run_backend(), run_torch(), atol=3.0, rtol=0.15)

    backend_ms = triton.testing.do_bench(run_backend, warmup=25, rep=iters)
    torch_ms = triton.testing.do_bench(run_torch, warmup=25, rep=iters)

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
    backend_tflops = total_flops / 1.0e12 / (backend_ms / 1.0e3)
    torch_tflops = total_flops / 1.0e12 / (torch_ms / 1.0e3)

    dtype_label = "BF16" if dtype == torch.bfloat16 else "FP16"
    print(
        f"Conv2d {dtype_label} batch={batch}, in_channels={in_channels}, out_channels={out_channels}, "
        f"shape=({height}, {width}), kernel=({kernel_h}, {kernel_w}), stride=({stride_h}, {stride_w}), "
        f"padding=({pad_h}, {pad_w}), dilation=({dilation_h}, {dilation_w})"
    )
    print(f" {backend.capitalize()}: {backend_ms:.3f} ms | {backend_tflops:.2f} TFLOP/s")
    print(f" PyTorch: {torch_ms:.3f} ms | {torch_tflops:.2f} TFLOP/s")

    return {
        "backend": backend,
        "backend_ms": backend_ms,
        "torch_ms": torch_ms,
        "backend_tflops": backend_tflops,
        "torch_tflops": torch_tflops,
    }


def autotune_conv2d_config(
        *,
        batch: int,
        in_channels: int,
        out_channels: int,
        height: int,
        width: int,
        kernel_h: int,
        kernel_w: int,
        stride_h: int,
        stride_w: int,
        pad_h: int,
        pad_w: int,
        dilation_h: int,
        dilation_w: int,
        dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, int]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required to autotune the Conv2d kernel")

    device = torch.device("cuda", torch.cuda.current_device())
    _ensure_supported_dtype(dtype)

    eff_kernel_h = dilation_h * (kernel_h - 1) + 1
    eff_kernel_w = dilation_w * (kernel_w - 1) + 1

    out_h = (height + 2 * pad_h - eff_kernel_h)
    out_w = (width + 2 * pad_w - eff_kernel_w)
    if out_h < 0 or out_w < 0:
        raise ValueError("Invalid stride/padding/dilation/kernel combination producing negative output")
    if out_h % stride_h != 0 or out_w % stride_w != 0:
        raise ValueError("Invalid stride value producing fractional output dimension")

    out_h = out_h // stride_h + 1
    out_w = out_w // stride_w + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError("Invalid stride/padding/dilation/kernel combination producing empty output")

    torch.manual_seed(0)
    x = torch.randn((batch, in_channels, height, width), device=device, dtype=dtype)
    weight = torch.randn((out_channels, in_channels, kernel_h, kernel_w), device=device, dtype=dtype)
    bias = torch.randn((out_channels,), device=device, dtype=dtype)

    conv2d_kernel.cache.clear()

    _ = _conv2d_triton(
        x,
        weight,
        bias=bias,
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
        dilation=(dilation_h, dilation_w),
        check=False,
    )

    if not conv2d_kernel.cache:
        raise RuntimeError("Failed to autotune Conv2d kernel; cache is empty")

    # take the config that corresponds to the most recent invocation
    best_config = conv2d_kernel.best_config
    if best_config is None:
        # fallback to any cached entry
        best_config = next(iter(conv2d_kernel.cache.values()))

    block_key = "BLOCK_W" if "BLOCK_W" in best_config.kwargs else "BLOCK_PIXELS"
    block_w = int(best_config.kwargs[block_key])

    return {
        "BLOCK_W": block_w,
        "BLOCK_OC": int(best_config.kwargs["BLOCK_OC"]),
        "BLOCK_K": int(best_config.kwargs["BLOCK_K"]),
        "num_warps": int(best_config.num_warps),
        "num_stages": int(best_config.num_stages),
    }


def _parse_pair(value: str) -> Tuple[int, int]:
    parts = value.split(",")
    if len(parts) == 1:
        v = int(parts[0])
        return v, v
    if len(parts) == 2:
        return int(parts[0]), int(parts[1])
    raise argparse.ArgumentTypeError("Expected value like '3' or '3,5'")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Triton Conv2d against PyTorch")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--in-channels", type=int, default=1024)
    parser.add_argument("--out-channels", type=int, default=1024)
    parser.add_argument("--height", type=int, default=48)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--kernel", type=_parse_pair, default="3")
    parser.add_argument("--stride", type=_parse_pair, default="1,1")
    parser.add_argument("--padding", type=_parse_pair, default="2,2")
    parser.add_argument("--dilation", type=_parse_pair, default="2,2")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--skip-assert", action="store_true")
    parser.add_argument("--report-best-config", action="store_true")
    parser.add_argument("--backend", choices=["triton", "cutlass"], default="triton")

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required to run the Conv2d benchmark")

    kernel_h, kernel_w = args.kernel
    stride_h, stride_w = args.stride
    pad_h, pad_w = args.padding
    dilation_h, dilation_w = args.dilation

    if args.report_best_config:
        if args.backend != "triton":
            raise ValueError("--report-best-config is only available for the Triton backend")
        best = autotune_conv2d_config(
            batch=args.batch,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            height=args.height,
            width=args.width,
            kernel_h=kernel_h,
            kernel_w=kernel_w,
            stride_h=stride_h,
            stride_w=stride_w,
            pad_h=pad_h,
            pad_w=pad_w,
            dilation_h=dilation_h,
            dilation_w=dilation_w,
        )
        print(json.dumps(best))
        return

    bench_conv2d(
        batch=args.batch,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        height=args.height,
        width=args.width,
        kernel_size=(kernel_h, kernel_w),
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
        dilation=(dilation_h, dilation_w),
        iters=args.iters,
        check=not args.skip_assert,
        backend=args.backend,
    )


if __name__ == "__main__":
    main()
