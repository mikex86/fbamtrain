import math

import torch
import triton
import triton.language as tl
import triton.testing


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_H": 8, "BLOCK_SIZE_W": 8, "MAX_KERNEL_H": 7, "MAX_KERNEL_W": 7},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_SIZE_H": 8, "BLOCK_SIZE_W": 16, "MAX_KERNEL_H": 7, "MAX_KERNEL_W": 7},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_SIZE_H": 16, "BLOCK_SIZE_W": 8, "MAX_KERNEL_H": 7, "MAX_KERNEL_W": 7},
            num_warps=4,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_SIZE_H": 16, "BLOCK_SIZE_W": 16, "MAX_KERNEL_H": 7, "MAX_KERNEL_W": 7},
            num_warps=8,
            num_stages=2,
        ),
    ],
    key=["pool_h_out", "pool_w_out", "kernel_h", "kernel_w"],
)
@triton.jit
def avg_pool2d_kernel(
    input_ptr,
    output_ptr,
    outer_stride_in0,
    outer_stride_in1,
    pool_stride_in_h,
    pool_stride_in_w,
    outer_stride_out0,
    outer_stride_out1,
    pool_stride_out_h,
    pool_stride_out_w,
    outer_size0,
    outer_size1,
    pool_h_in,
    pool_w_in,
    pool_h_out,
    pool_w_out,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    inv_kernel_size,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    MAX_KERNEL_H: tl.constexpr,
    MAX_KERNEL_W: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    grid_w = tl.cdiv(pool_w_out, BLOCK_SIZE_W)
    grid_h = tl.cdiv(pool_h_out, BLOCK_SIZE_H)
    tiles_per_outer = grid_w * grid_h
    pid_outer = pid // tiles_per_outer
    tile_id = pid - pid_outer * tiles_per_outer
    pid_h = tile_id // grid_w
    pid_w = tile_id - pid_h * grid_w

    outer_total = outer_size0 * outer_size1
    if pid_outer >= outer_total:
        return

    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_w = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)

    mask_h = offs_h < pool_h_out
    mask_w = offs_w < pool_w_out
    mask_tile = mask_h[:, None] & mask_w[None, :]

    idx_outer0 = pid_outer // outer_size1
    idx_outer1 = pid_outer % outer_size1

    idx_outer0_i64 = idx_outer0.to(tl.int64)
    idx_outer1_i64 = idx_outer1.to(tl.int64)
    outer_stride_in0_i64 = outer_stride_in0.to(tl.int64)
    outer_stride_in1_i64 = outer_stride_in1.to(tl.int64)
    outer_stride_out0_i64 = outer_stride_out0.to(tl.int64)
    outer_stride_out1_i64 = outer_stride_out1.to(tl.int64)

    base_in = input_ptr + idx_outer0_i64 * outer_stride_in0_i64 + idx_outer1_i64 * outer_stride_in1_i64
    base_out = output_ptr + idx_outer0_i64 * outer_stride_out0_i64 + idx_outer1_i64 * outer_stride_out1_i64

    start_h = offs_h * stride_h - padding_h
    start_w = offs_w * stride_w - padding_w

    pool_stride_in_h_i64 = pool_stride_in_h.to(tl.int64)
    pool_stride_in_w_i64 = pool_stride_in_w.to(tl.int64)
    pool_stride_out_h_i64 = pool_stride_out_h.to(tl.int64)
    pool_stride_out_w_i64 = pool_stride_out_w.to(tl.int64)
    offs_h_i64 = tl.broadcast_to(offs_h[:, None], mask_tile.shape).to(tl.int64)
    offs_w_i64 = tl.broadcast_to(offs_w[None, :], mask_tile.shape).to(tl.int64)

    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_W), dtype=tl.float32)

    for kh in tl.static_range(MAX_KERNEL_H):
        mask_kh = kh < kernel_h
        h_idx = start_h + kh
        for kw in tl.static_range(MAX_KERNEL_W):
            mask_kw = kw < kernel_w
            h_idx_2d = tl.broadcast_to(h_idx[:, None], mask_tile.shape)
            w_idx = start_w + kw
            w_idx_2d = tl.broadcast_to(w_idx[None, :], mask_tile.shape)

            valid_h = (h_idx_2d >= 0) & (h_idx_2d < pool_h_in)
            valid_w = (w_idx_2d >= 0) & (w_idx_2d < pool_w_in)
            mask = mask_tile & valid_h & valid_w & mask_kh & mask_kw

            h_safe = tl.where(mask, h_idx_2d, 0)
            w_safe = tl.where(mask, w_idx_2d, 0)

            ptrs = (
                base_in
                + h_safe.to(tl.int64) * pool_stride_in_h_i64
                + w_safe.to(tl.int64) * pool_stride_in_w_i64
            )
            vals = tl.load(ptrs, mask=mask, other=0.0)
            acc += vals.to(tl.float32)

    avg = acc * inv_kernel_size
    # Store in the output tensor's dtype (was incorrectly forced to BF16).
    out_dtype = output_ptr.dtype.element_ty
    tl.store(
        base_out + offs_h_i64 * pool_stride_out_h_i64 + offs_w_i64 * pool_stride_out_w_i64,
        avg.to(out_dtype),
        mask=mask_tile,
    )


@triton.jit
def avg_pool2d_nhwc_2x2_s2_kernel(
    input_ptr,
    output_ptr,
    outer_stride_in0,
    outer_stride_in1,
    pool_stride_in_h,
    pool_stride_in_w,
    outer_stride_out0,
    outer_stride_out1,
    pool_stride_out_h,
    pool_stride_out_w,
    outer_size0,
    outer_size1,
    pool_h_in,
    pool_w_in,
    pool_h_out,
    pool_w_out,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    inv_kernel_size,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    MAX_KERNEL_H: tl.constexpr,
    MAX_KERNEL_W: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    total_hw = pool_h_out * pool_w_out
    blocks_hw = tl.cdiv(total_hw, BLOCK_SIZE_H)
    blocks_c = tl.cdiv(outer_size1, BLOCK_SIZE_W)
    tiles_per_batch = blocks_hw * blocks_c

    n = pid // tiles_per_batch
    tile_id = pid - n * tiles_per_batch
    hw_block = tile_id // blocks_c
    c_block = tile_id - hw_block * blocks_c

    offs_hw = hw_block * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_c = c_block * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)

    oh = offs_hw // pool_w_out
    ow = offs_hw - oh * pool_w_out

    mask = (n < outer_size0) & (offs_hw[:, None] < total_hw) & (offs_c[None, :] < outer_size1)

    n_i64 = n.to(tl.int64)
    oh_i64 = tl.broadcast_to(oh[:, None], mask.shape).to(tl.int64)
    ow_i64 = tl.broadcast_to(ow[:, None], mask.shape).to(tl.int64)
    c_i64 = tl.broadcast_to(offs_c[None, :], mask.shape).to(tl.int64)

    outer_stride_in0_i64 = outer_stride_in0.to(tl.int64)
    outer_stride_in1_i64 = outer_stride_in1.to(tl.int64)
    pool_stride_in_h_i64 = pool_stride_in_h.to(tl.int64)
    pool_stride_in_w_i64 = pool_stride_in_w.to(tl.int64)
    outer_stride_out0_i64 = outer_stride_out0.to(tl.int64)
    outer_stride_out1_i64 = outer_stride_out1.to(tl.int64)
    pool_stride_out_h_i64 = pool_stride_out_h.to(tl.int64)
    pool_stride_out_w_i64 = pool_stride_out_w.to(tl.int64)

    in_base = (
        input_ptr
        + n_i64 * outer_stride_in0_i64
        + (oh_i64 * 2) * pool_stride_in_h_i64
        + (ow_i64 * 2) * pool_stride_in_w_i64
        + c_i64 * outer_stride_in1_i64
    )

    v00 = tl.load(in_base, mask=mask, other=0.0).to(tl.float32)
    v01 = tl.load(in_base + pool_stride_in_w_i64, mask=mask, other=0.0).to(tl.float32)
    v10 = tl.load(in_base + pool_stride_in_h_i64, mask=mask, other=0.0).to(tl.float32)
    v11 = tl.load(in_base + pool_stride_in_h_i64 + pool_stride_in_w_i64, mask=mask, other=0.0).to(tl.float32)
    avg = (v00 + v01 + v10 + v11) * 0.25

    out_ptrs = (
        output_ptr
        + n_i64 * outer_stride_out0_i64
        + oh_i64 * pool_stride_out_h_i64
        + ow_i64 * pool_stride_out_w_i64
        + c_i64 * outer_stride_out1_i64
    )
    tl.store(out_ptrs, avg.to(output_ptr.dtype.element_ty), mask=mask)


def _avg_pool2d_triton(
    x: torch.Tensor,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
) -> torch.Tensor:
    if x.dtype != torch.bfloat16 or x.ndim != 4:
        raise ValueError("avg_pool2d_triton expects a 4D BF16 tensor")

    device = x.device
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    pad_h, pad_w = padding

    if kernel_h <= 0 or kernel_w <= 0 or stride_h <= 0 or stride_w <= 0:
        raise ValueError("Invalid kernel or stride configuration")

    n, c, h_in, w_in = x.shape

    h_out = math.floor((h_in + 2 * pad_h - kernel_h) / stride_h) + 1
    w_out = math.floor((w_in + 2 * pad_w - kernel_w) / stride_w) + 1
    if h_out <= 0 or w_out <= 0:
        raise ValueError("avg_pool2d_triton produced empty output")

    y = torch.empty((n, c, h_out, w_out), device=device, dtype=x.dtype)

    in_strides = x.stride()
    out_strides = y.stride()

    outer_size0, outer_size1 = n, c
    outer_stride_in0, outer_stride_in1 = in_strides[0], in_strides[1]
    outer_stride_out0, outer_stride_out1 = out_strides[0], out_strides[1]

    pool_stride_in_h, pool_stride_in_w = in_strides[2], in_strides[3]
    pool_stride_out_h, pool_stride_out_w = out_strides[2], out_strides[3]

    inv_kernel = 1.0 / float(kernel_h * kernel_w)

    def grid(meta):
        grid_w = triton.cdiv(w_out, meta["BLOCK_SIZE_W"])
        grid_h = triton.cdiv(h_out, meta["BLOCK_SIZE_H"])
        return (grid_w * grid_h * outer_size0 * outer_size1,)

    avg_pool2d_kernel[grid](
        x,
        y,
        outer_stride_in0,
        outer_stride_in1,
        pool_stride_in_h,
        pool_stride_in_w,
        outer_stride_out0,
        outer_stride_out1,
        pool_stride_out_h,
        pool_stride_out_w,
        outer_size0,
        outer_size1,
        h_in,
        w_in,
        h_out,
        w_out,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        inv_kernel,
    )
    return y


def bench_avg_pool2d(
    batch: int = 16,
    channels: int = 32,
    height: int = 64,
    width: int = 64,
    kernel_h: int = 3,
    kernel_w: int = 3,
    stride_h: int = 2,
    stride_w: int = 2,
    pad_h: int = 0,
    pad_w: int = 0,
    iters: int = 100,
):
    device = triton.runtime.driver.active.get_active_torch_device()
    dtype = torch.bfloat16

    torch.manual_seed(0)
    x = torch.randn((batch, channels, height, width), device=device, dtype=dtype)

    def run_triton():
        return _avg_pool2d_triton(
            x,
            (kernel_h, kernel_w),
            (stride_h, stride_w),
            (pad_h, pad_w),
        )

    def run_torch():
        return torch.nn.functional.avg_pool2d(
            x.to(torch.float32),
            kernel_size=(kernel_h, kernel_w),
            stride=(stride_h, stride_w),
            padding=(pad_h, pad_w),
        ).to(dtype)

    triton.testing.assert_close(run_triton(), run_torch(), atol=2e-3, rtol=2e-3)

    triton_ms = triton.testing.do_bench(run_triton, warmup=25, rep=iters)
    torch_ms = triton.testing.do_bench(run_torch, warmup=25, rep=iters)

    print(
        f"AvgPool2d BF16 shape={(batch, channels, height, width)}, "
        f"K={(kernel_h, kernel_w)}, S={(stride_h, stride_w)}, P={(pad_h, pad_w)}"
    )
    print(f" Triton: {triton_ms:.3f} ms")
    print(f" PyTorch: {torch_ms:.3f} ms")

    return {"triton_ms": triton_ms, "torch_ms": torch_ms}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark Triton AvgPool2d against PyTorch")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--kernel-h", type=int, default=3)
    parser.add_argument("--kernel-w", type=int, default=3)
    parser.add_argument("--stride-h", type=int, default=2)
    parser.add_argument("--stride-w", type=int, default=2)
    parser.add_argument("--pad-h", type=int, default=0)
    parser.add_argument("--pad-w", type=int, default=0)
    parser.add_argument("--iters", type=int, default=100)

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required to run the benchmark")

    bench_avg_pool2d(
        batch=args.batch,
        channels=args.channels,
        height=args.height,
        width=args.width,
        kernel_h=args.kernel_h,
        kernel_w=args.kernel_w,
        stride_h=args.stride_h,
        stride_w=args.stride_w,
        pad_h=args.pad_h,
        pad_w=args.pad_w,
        iters=args.iters,
    )


if __name__ == "__main__":
    main()
