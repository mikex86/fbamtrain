import math
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
def cross_entropy_on_targets(
    out_ptr,
    logits_ptr,
    targets_ptr,
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

    max_val = tl.full([], -float("inf"), tl.float32)
    exp_sum = tl.zeros((), dtype=tl.float32)
    for start in range(0, cols, BLOCK_SIZE):
        idx = start + offsets
        mask = idx < cols
        vals = tl.load(logits_ptr + row_start + idx, mask=mask, other=-float("inf"))
        vals_f = vals.to(tl.float32)
        block_max = tl.max(vals_f, axis=0)
        new_max = tl.maximum(max_val, block_max)
        # Re-normalize existing sum to the new max and accumulate this block
        exp_sum = exp_sum * tl.exp(max_val - new_max) + tl.sum(tl.exp(vals_f - new_max), axis=0)
        max_val = new_max

    target_idx = tl.load(targets_ptr + pid, mask=None).to(tl.int64)
    target_val = tl.load(logits_ptr + row_start + target_idx, mask=None).to(tl.float32)
    loss = tl.log(exp_sum) + max_val - target_val
    tl.store(out_ptr + pid, loss.to(out_ptr.dtype.element_ty))


@triton.jit
def _reduce_sum(out_ptr, in_ptr, rows, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    if pid != 0:
        return

    offsets = tl.arange(0, BLOCK_SIZE)
    acc = 0.0
    for start in range(0, rows, BLOCK_SIZE):
        idx = start + offsets
        mask = idx < rows
        vals = tl.load(in_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(vals, axis=0)
    tl.store(out_ptr, acc)


def _bytes_processed(rows: int, cols: int, dtype: torch.dtype, reduction: str) -> int:
    elem_bytes = torch.finfo(dtype).bits // 8
    read_logits = rows * cols * elem_bytes
    read_targets = rows * 4
    write = rows * (elem_bytes if reduction == "mean" else 4)
    return read_logits + read_targets + write


@dataclass
class CEOutput:
    loss: torch.Tensor
    time_ms: float
    tb_per_s: float
    best_config: Optional[triton.Config]


def launch_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    reduction: str = "mean",
    config: triton.Config,
) -> torch.Tensor:
    if logits.ndim < 2:
        raise ValueError("logits must have at least 2 dimensions")
    rows = logits.numel() // logits.shape[-1]
    cols = logits.shape[-1]
    logits_flat = logits.view(rows, cols)
    targets_flat = targets.contiguous().view(rows)

    if reduction not in ("mean", "add"):
        raise ValueError("reduction must be 'mean' or 'add'")

    bs = config.kwargs["BLOCK_SIZE"]
    grid = (rows,)
    out_dtype = logits.dtype if reduction == "mean" else torch.float32
    out = torch.empty((rows,), device=logits.device, dtype=out_dtype)
    cross_entropy_on_targets[grid](
        out,
        logits_flat,
        targets_flat,
        rows,
        cols,
        BLOCK_SIZE=bs,
        NUM_WARPS=config.num_warps,
        NUM_STAGES=config.num_stages,
    )

    return out


def _run_and_bench(
    logits: torch.Tensor,
    targets: torch.Tensor,
    reduction: str,
    iters: int,
    config: triton.Config,
) -> CEOutput:
    def runner():
        return launch_cross_entropy(logits, targets, reduction=reduction, config=config)

    # Correctness check vs PyTorch
    with torch.no_grad():
        ref_loss = torch.nn.functional.cross_entropy(
            logits.float().view(-1, logits.shape[-1]),
            targets.view(-1).long(),
            reduction="none" if reduction == "mean" else "sum",
        )
        if reduction == "mean":
            ref_loss = ref_loss.view(logits.shape[:-1]).contiguous()
        else:
            ref_loss = ref_loss.view(1)

    triton.testing.assert_close(runner(), ref_loss.to(dtype=runner().dtype), atol=5e-3, rtol=5e-3)

    ms = triton.testing.do_bench(runner, warmup=25, rep=iters)
    bytes_proc = _bytes_processed(
        logits.numel() // logits.shape[-1], logits.shape[-1], logits.dtype, reduction
    )
    tb_per_s = bytes_proc / (ms * 1e-3) / 1e12
    return CEOutput(loss=runner(), time_ms=ms, tb_per_s=tb_per_s, best_config=config)


def _assert_deterministic(logits: torch.Tensor, targets: torch.Tensor, reduction: str, config: triton.Config):
    out1 = launch_cross_entropy(logits, targets, reduction=reduction, config=config)
    out2 = launch_cross_entropy(logits, targets, reduction=reduction, config=config)
    if not torch.equal(out1, out2):
        raise AssertionError(f"Determinism check failed for reduction={reduction}")


def _find_best_config(
    logits: torch.Tensor, targets: torch.Tensor, reduction: str, iters: int
) -> triton.Config:
    best_cfg = None
    best_tb = -1.0
    for cfg in _CONFIGS:
        def run():
            launch_cross_entropy(logits, targets, reduction=reduction, config=cfg)
        ms = triton.testing.do_bench(run, warmup=10, rep=max(iters, 20))
        tb = _bytes_processed(
            logits.numel() // logits.shape[-1], logits.shape[-1], logits.dtype, reduction
        ) / (ms * 1e-3) / 1e12
        if tb > best_tb:
            best_tb = tb
            best_cfg = cfg
    if os.environ.get("TRITON_PRINT_AUTOTUNING") == "1":
        print(
            f"[best] reduction={reduction} BLOCK_SIZE={best_cfg.kwargs['BLOCK_SIZE']} "
            f"warps={best_cfg.num_warps} stages={best_cfg.num_stages} TB/s≈{best_tb:.3f}"
        )
    return best_cfg


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark Triton cross entropy on targets")
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
    cfg = triton.Config({"BLOCK_SIZE": args.block_size}, num_warps=max(4, args.block_size // 64), num_stages=2)

    print(f"Benchmarking cross entropy rows={args.rows} cols={args.cols} dtype={args.dtype}")
    # Find best configs for each reduction
    cfg_mean = _find_best_config(logits, targets, reduction="mean", iters=args.iters // 2 or 1)
    cfg_add = _find_best_config(logits, targets, reduction="add", iters=args.iters // 2 or 1)

    print("Running deterministic checks...")
    _assert_deterministic(logits, targets, reduction="mean", config=cfg_mean)
    _assert_deterministic(logits, targets, reduction="add", config=cfg_add)

    mean_out = _run_and_bench(logits, targets, reduction="mean", iters=args.iters, config=cfg_mean)
    add_out = _run_and_bench(logits, targets, reduction="add", iters=args.iters, config=cfg_add)

    print(f"Mean reduction: {mean_out.time_ms:.4f} ms, {mean_out.tb_per_s:.3f} TB/s")
    print(f"Add reduction: {add_out.time_ms:.4f} ms, {add_out.tb_per_s:.3f} TB/s")
    print("Targets parity and determinism verified.")


if __name__ == "__main__":
    main()
