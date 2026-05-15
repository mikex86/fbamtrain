import os
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import triton
import triton.language as tl
import triton.testing


_CONFIGS: Sequence[triton.Config] = [
    triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 256}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=16, num_stages=2),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=16, num_stages=3),
]


@triton.jit
def cross_entropy_on_targets_bwd(
    grad_ptr,
    logits_ptr,
    targets_ptr,
    upstream_ptr,
    rows,
    cols,
    BLOCK_SIZE: tl.constexpr,
    NUM_WARPS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if pid >= rows:
        return

    row_start = pid * cols
    offsets = tl.arange(0, BLOCK_SIZE)

    # Two-pass running max/sum to keep exp stable for large vocab.
    max_val = tl.full([], -float("inf"), tl.float32)
    exp_sum = tl.zeros((), dtype=tl.float32)
    for start in range(0, cols, BLOCK_SIZE):
        idx = start + offsets
        mask = idx < cols
        vals = tl.load(logits_ptr + row_start + idx, mask=mask, other=-float("inf"))
        vals_f = vals.to(tl.float32)
        block_max = tl.max(vals_f, axis=0)
        new_max = tl.maximum(max_val, block_max)
        exp_sum = exp_sum * tl.exp(max_val - new_max) + tl.sum(tl.exp(vals_f - new_max), axis=0)
        max_val = new_max

    target_idx = tl.load(targets_ptr + pid).to(tl.int32)
    upstream = tl.load(upstream_ptr + pid).to(tl.float32)

    for start in range(0, cols, BLOCK_SIZE):
        idx = start + offsets
        mask = idx < cols
        vals = tl.load(logits_ptr + row_start + idx, mask=mask, other=-float("inf")).to(tl.float32)
        prob = tl.exp(vals - max_val) / exp_sum
        grad = prob * upstream
        grad -= upstream * tl.where(idx == target_idx, 1.0, 0.0)
        tl.store(grad_ptr + row_start + idx, grad.to(grad_ptr.dtype.element_ty), mask=mask)


@dataclass
class CEBackwardOutput:
    grad: torch.Tensor
    time_ms: float
    tb_per_s: float
    best_config: Optional[triton.Config]


def _bytes_processed(rows: int, cols: int, dtype: torch.dtype) -> int:
    elem_bytes = torch.finfo(dtype).bits // 8
    read_logits = rows * cols * elem_bytes
    read_targets = rows * 4
    read_upstream = rows * 4
    write = rows * cols * elem_bytes
    return read_logits + read_targets + read_upstream + write


def launch_cross_entropy_bwd(
    logits: torch.Tensor,
    targets: torch.Tensor,
    upstream: torch.Tensor,
    *,
    config: triton.Config,
) -> torch.Tensor:
    if logits.ndim < 2:
        raise ValueError("logits must have at least 2 dimensions")
    rows = logits.numel() // logits.shape[-1]
    cols = logits.shape[-1]
    logits_flat = logits.view(rows, cols)
    targets_flat = targets.contiguous().view(rows)
    upstream_flat = upstream.contiguous().view(rows)

    bs = config.kwargs["BLOCK_SIZE"]
    grid = (rows,)
    grad = torch.empty_like(logits_flat)
    cross_entropy_on_targets_bwd[grid](
        grad,
        logits_flat,
        targets_flat,
        upstream_flat,
        rows,
        cols,
        BLOCK_SIZE=bs,
        NUM_WARPS=config.num_warps,
        NUM_STAGES=config.num_stages,
    )
    return grad.view_as(logits)


def _run_and_bench(
    logits: torch.Tensor,
    targets: torch.Tensor,
    upstream: torch.Tensor,
    iters: int,
    config: triton.Config,
) -> CEBackwardOutput:
    def runner():
        return launch_cross_entropy_bwd(logits, targets, upstream, config=config)

    # Correctness against PyTorch autograd.
    with torch.no_grad():
        logits_ref = logits.detach().clone().requires_grad_(True)
        loss = torch.nn.functional.cross_entropy(
            logits_ref.view(-1, logits.shape[-1]), targets.view(-1), reduction="none"
        )
        loss.backward(upstream.contiguous().view(-1))
        ref_grad = logits_ref.grad.detach()
    triton.testing.assert_close(runner(), ref_grad.to(dtype=logits.dtype), atol=5e-3, rtol=5e-3)

    ms = triton.testing.do_bench(runner, warmup=25, rep=iters)
    bytes_proc = _bytes_processed(rows=logits.numel() // logits.shape[-1], cols=logits.shape[-1], dtype=logits.dtype)
    tb_per_s = bytes_proc / (ms * 1e-3) / 1e12
    return CEBackwardOutput(grad=runner(), time_ms=ms, tb_per_s=tb_per_s, best_config=config)


def _find_best_config(logits: torch.Tensor, targets: torch.Tensor, upstream: torch.Tensor, iters: int) -> triton.Config:
    best_cfg = None
    best_tb = -1.0
    for cfg in _CONFIGS:
        def run():
            launch_cross_entropy_bwd(logits, targets, upstream, config=cfg)
        ms = triton.testing.do_bench(run, warmup=10, rep=max(iters, 20))
        tb = _bytes_processed(rows=logits.numel() // logits.shape[-1], cols=logits.shape[-1], dtype=logits.dtype) / (
            ms * 1e-3
        ) / 1e12
        if tb > best_tb:
            best_tb = tb
            best_cfg = cfg
    if os.environ.get("TRITON_PRINT_AUTOTUNING") == "1":
        print(
            f"[best] BLOCK_SIZE={best_cfg.kwargs['BLOCK_SIZE']} warps={best_cfg.num_warps} "
            f"stages={best_cfg.num_stages} TB/s≈{best_tb:.3f}"
        )
    return best_cfg


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark Triton cross entropy backward on targets")
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--cols", type=int, default=16384)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--block-size", type=int, default=512)
    args = parser.parse_args()

    os.environ.setdefault("TRITON_PRINT_AUTOTUNING", "1")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required to run the benchmark")
    device = torch.device("cuda")
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    torch.manual_seed(0)
    logits = torch.randn((args.rows, args.cols), device=device, dtype=dtype)
    targets = torch.randint(0, args.cols, (args.rows,), device=device, dtype=torch.int32)
    upstream_rows = torch.ones((args.rows,), device=device, dtype=torch.float32)

    cfg = triton.Config(
        {"BLOCK_SIZE": args.block_size},
        num_warps=max(4, args.block_size // 64),
        num_stages=2,
    )

    print(f"Benchmarking CE backward rows={args.rows} cols={args.cols} dtype={args.dtype}")
    best_cfg = _find_best_config(logits, targets, upstream_rows, iters=args.iters // 2 or 1)

    print("Running determinism/correctness checks...")
    _run_and_bench(logits, targets, upstream_rows, iters=10, config=cfg)

    out = _run_and_bench(logits, targets, upstream_rows, iters=args.iters, config=best_cfg)
    print(f"Backward: {out.time_ms:.4f} ms, {out.tb_per_s:.3f} TB/s")


if __name__ == "__main__":
    main()
