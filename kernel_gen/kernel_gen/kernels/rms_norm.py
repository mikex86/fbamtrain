import math

import torch
import triton
import triton.language as tl
import triton.testing


def _get_device():
    return triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def rms_norm_fwd(
    input_ptr,
    weight_ptr,
    output_ptr,
    num_rows,
    num_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= num_rows:
        return

    row_start = row_idx.to(tl.int64) * num_cols.to(tl.int64)
    offsets = tl.arange(0, BLOCK_SIZE)

    sum_squares = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for col_start in tl.range(0, num_cols, BLOCK_SIZE):
        cols = col_start + offsets
        mask = cols < num_cols
        x = tl.load(input_ptr + row_start + cols.to(tl.int64), mask=mask, other=0.0).to(tl.float32)
        sum_squares += tl.where(mask, x * x, 0.0)

    total_squares = tl.sum(sum_squares, axis=0)
    mean_square = total_squares / num_cols
    inv_rms = tl.rsqrt(mean_square + eps)

    for col_start in tl.range(0, num_cols, BLOCK_SIZE):
        cols = col_start + offsets
        mask = cols < num_cols
        x = tl.load(input_ptr + row_start + cols.to(tl.int64), mask=mask, other=0.0).to(tl.float32)
        gamma = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        y = x * inv_rms * gamma
        out_dtype = output_ptr.dtype.element_ty
        tl.store(output_ptr + row_start + cols.to(tl.int64), y.to(out_dtype), mask=mask)


@triton.jit
def rms_norm_bwd(
    input_ptr,
    weight_ptr,
    upstream_ptr,
    grad_input_ptr,
    xhat_ptr,
    num_rows,
    num_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= num_rows:
        return

    row_start = row_idx.to(tl.int64) * num_cols.to(tl.int64)
    offsets = tl.arange(0, BLOCK_SIZE)

    sum_squares = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    sum_dyw_x = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for col_start in tl.range(0, num_cols, BLOCK_SIZE):
        cols = col_start + offsets
        mask = cols < num_cols
        x = tl.load(input_ptr + row_start + cols.to(tl.int64), mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(upstream_ptr + row_start + cols.to(tl.int64), mask=mask, other=0.0).to(tl.float32)
        gamma = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        dyw = dy * gamma
        sum_squares += tl.where(mask, x * x, 0.0)
        sum_dyw_x += tl.where(mask, dyw * x, 0.0)

    total_squares = tl.sum(sum_squares, axis=0)
    dot = tl.sum(sum_dyw_x, axis=0)
    num_cols_f = tl.full([1], num_cols, tl.float32)
    mean_square = total_squares / num_cols_f
    inv_rms = tl.rsqrt(mean_square + eps)
    inv_rms3 = inv_rms * inv_rms * inv_rms
    scale = inv_rms3 * dot / num_cols_f

    out_dtype = grad_input_ptr.dtype.element_ty
    xhat_dtype = xhat_ptr.dtype.element_ty
    for col_start in tl.range(0, num_cols, BLOCK_SIZE):
        cols = col_start + offsets
        mask = cols < num_cols
        x = tl.load(input_ptr + row_start + cols.to(tl.int64), mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(upstream_ptr + row_start + cols.to(tl.int64), mask=mask, other=0.0).to(tl.float32)
        gamma = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        dyw = dy * gamma
        dx = dyw * inv_rms - x * scale
        x_hat = x * inv_rms
        tl.store(grad_input_ptr + row_start + cols.to(tl.int64), dx.to(out_dtype), mask=mask)
        tl.store(xhat_ptr + row_start + cols.to(tl.int64), x_hat.to(xhat_dtype), mask=mask)


def _rms_norm_triton(x, weight, eps, block_size=256):
    assert x.dtype == weight.dtype
    assert x.is_contiguous() and weight.is_contiguous()

    y = torch.empty_like(x)
    num_rows = math.prod(x.shape[:-1])
    num_cols = x.shape[-1]
    x_flat = x.view(num_rows, num_cols)
    y_flat = y.view(num_rows, num_cols)

    rms_norm_fwd[(num_rows,)](
        x_flat,
        weight,
        y_flat,
        num_rows,
        num_cols,
        eps,
        BLOCK_SIZE=block_size,
    )
    return y


def bench_rms_norm(B=32, T=4096, H=128, eps=1e-5, iters=100, block_size=256):
    device = _get_device()

    torch.manual_seed(0)
    x = torch.randn((B, T, H), device=device, dtype=torch.float16)
    gamma = torch.ones((H,), device=device, dtype=torch.float16)

    def run_triton():
        return _rms_norm_triton(x, gamma, eps, block_size)

    def run_torch():
        x32 = x.to(torch.float32)
        rms = torch.sqrt((x32 * x32).mean(dim=-1, keepdim=True) + eps)
        return (x32 / rms * gamma).to(torch.float16)

    triton.testing.assert_close(run_triton(), run_torch(), atol=1e-2, rtol=1e-2)

    triton_ms = triton.testing.do_bench(run_triton, warmup=25, rep=iters)
    torch_ms = triton.testing.do_bench(run_torch, warmup=25, rep=iters)

    hidden_size = H
    total_elems = B * T * hidden_size
    bytes_moved = total_elems * 2 * torch.finfo(x.dtype).bits / 8
    gbs_triton = bytes_moved / (triton_ms * 1e-3) / 1e9
    gbs_torch = bytes_moved / (torch_ms * 1e-3) / 1e9

    print(f"RMSNorm FP16 B={B},T={T},H={H}, eps={eps}, block={block_size}")
    print(f" Triton: {triton_ms:.3f} ms, {gbs_triton:.2f} GB/s")
    print(f" PyTorch: {torch_ms:.3f} ms, {gbs_torch:.2f} GB/s")

    return {
        "triton_ms": triton_ms,
        "torch_ms": torch_ms,
        "triton_gbs": gbs_triton,
        "torch_gbs": gbs_torch,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark Triton RMSNorm against PyTorch")
    parser.add_argument("--batch", "-B", type=int, default=32, help="Batch size")
    parser.add_argument("--tokens", "-T", type=int, default=4096, help="Sequence length / tokens")
    parser.add_argument("--hidden", "-H", type=int, default=128, help="Hidden size")
    parser.add_argument("--eps", type=float, default=1e-5, help="RMSNorm epsilon")
    parser.add_argument("--iters", type=int, default=100, help="Number of benchmark repetitions")
    parser.add_argument("--block-size", type=int, default=256, help="Kernel BLOCK_SIZE meta parameter")

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required to run the benchmark")

    bench_rms_norm(
        B=args.batch,
        T=args.tokens,
        H=args.hidden,
        eps=args.eps,
        iters=args.iters,
        block_size=args.block_size,
    )


if __name__ == "__main__":
    main()
