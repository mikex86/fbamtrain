import triton
import triton.language as tl
import triton.testing
import torch
import math


def _get_device():
    return triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def layer_norm_fwd(
        input_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        num_rows,
        num_cols,
        eps,
        BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= num_rows:
        return

    row_start = row_idx * num_cols
    offsets = tl.arange(0, BLOCK_SIZE)

    sum_x = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for col_start in tl.range(0, num_cols, BLOCK_SIZE):
        cols = col_start + offsets
        mask = cols < num_cols
        x = tl.load(input_ptr + row_start + cols, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sum_x += tl.where(mask, x, 0.0)

    total_sum = tl.sum(sum_x, axis=0)

    mean = total_sum / num_cols
    var_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for col_start in tl.range(0, num_cols, BLOCK_SIZE):
        cols = col_start + offsets
        mask = cols < num_cols

        x = tl.load(input_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
        diff = tl.where(mask, x - mean, 0.0)
        var_acc += diff * diff

    total_var = tl.sum(var_acc, axis=0)
    var = total_var / num_cols
    inv_std = tl.rsqrt(var + eps)

    for col_start in tl.range(0, num_cols, BLOCK_SIZE):
        cols = col_start + offsets
        mask = cols < num_cols

        x = tl.load(input_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
        gamma = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        beta = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        norm = (x - mean) * inv_std
        y = norm * gamma + beta
        out_dtype = output_ptr.dtype.element_ty
        tl.store(output_ptr + row_start + cols, y.to(out_dtype), mask=mask)


def _layer_norm_triton(x, weight, bias, eps, block_size=256):
    assert x.dtype == weight.dtype == bias.dtype
    assert x.is_contiguous() and weight.is_contiguous() and bias.is_contiguous()

    y = torch.empty_like(x)
    num_rows = math.prod(x.shape[:-1])
    num_cols = x.shape[-1]
    x_flat = x.view(num_rows, num_cols)
    y_flat = y.view(num_rows, num_cols)

    layer_norm_fwd[(num_rows,)](
        x_flat,
        weight,
        bias,
        y_flat,
        num_rows,
        num_cols,
        eps,
        BLOCK_SIZE=block_size,
    )
    return y


def bench_layer_norm(B=32, T=4096, H=128, eps=1e-5, iters=100, block_size=256):
    device = _get_device()

    torch.manual_seed(0)
    x = torch.randn((B, T, H), device=device, dtype=torch.float16)
    gamma = torch.ones((H,), device=device, dtype=torch.float16)
    beta = torch.zeros((H,), device=device, dtype=torch.float16)

    def run_triton():
        return _layer_norm_triton(x, gamma, beta, eps, block_size)

    def run_torch():
        return torch.nn.functional.layer_norm(x, (H,), gamma, beta, eps)

    triton.testing.assert_close(run_triton(), run_torch(), atol=1e-2, rtol=1e-2)

    triton_ms = triton.testing.do_bench(run_triton, warmup=25, rep=iters)
    torch_ms = triton.testing.do_bench(run_torch, warmup=25, rep=iters)

    hidden_size = H
    total_elems = B * T * hidden_size
    bytes_moved = total_elems * 2 * torch.finfo(torch.float16).bits / 8  # read + write
    gbs_triton = bytes_moved / (triton_ms * 1e-3) / 1e9
    gbs_torch = bytes_moved / (torch_ms * 1e-3) / 1e9

    print(f"LayerNorm FP16 B={B},T={T},H={H}, eps={eps}, block={block_size}")
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

    parser = argparse.ArgumentParser(description="Benchmark Triton LayerNorm against PyTorch")
    parser.add_argument("--batch", "-B", type=int, default=32, help="Batch size")
    parser.add_argument("--tokens", "-T", type=int, default=4096, help="Sequence length / tokens")
    parser.add_argument("--hidden", "-H", type=int, default=128, help="Hidden size")
    parser.add_argument("--eps", type=float, default=1e-5, help="LayerNorm epsilon")
    parser.add_argument("--iters", type=int, default=100, help="Number of benchmark repetitions")
    parser.add_argument("--block-size", type=int, default=256, help="Kernel BLOCK_SIZE meta parameter")

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required to run the benchmark")

    bench_layer_norm(
        B=args.batch,
        T=args.tokens,
        H=args.hidden,
        eps=args.eps,
        iters=args.iters,
        block_size=args.block_size,
    )


if __name__ == "__main__":
    main()
