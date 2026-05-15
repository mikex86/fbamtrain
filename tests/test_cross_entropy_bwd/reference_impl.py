from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main():
    device = torch.device("cuda")
    base_ref = load_file(Path(__file__).parent.parent / "test_cross_entropy" / "reference.safetensors")

    logits_fp16 = base_ref["logits_fp16"].to(device)
    logits_bf16 = base_ref["logits_bf16"].to(device)

    seq, batch, vocab = logits_fp16.shape
    targets_vals = torch.tensor([1, 0, 5, 3, 2, 6], device=device, dtype=torch.long)
    targets = targets_vals.view(seq, batch).contiguous()

    upstream_rows = torch.ones(seq * batch, device=device, dtype=torch.float32)
    upstream_scalar = torch.tensor([1.0], device=device, dtype=torch.float32)

    def grad_rows(logits: torch.Tensor):
        logits = logits.detach().clone().requires_grad_(True)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, vocab), targets.view(-1), reduction="none"
        )
        loss.backward(upstream_rows)
        return logits.grad.detach().cpu()

    def grad_scalar(logits: torch.Tensor):
        logits = logits.detach().clone().requires_grad_(True)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, vocab), targets.view(-1), reduction="mean"
        )
        loss.backward(upstream_scalar.reshape(()))
        return logits.grad.detach().cpu()

    grads = {
        "logits_fp16": logits_fp16.detach().cpu(),
        "logits_bf16": logits_bf16.detach().cpu(),
        "upstream_rows": upstream_rows.detach().cpu(),
        "upstream_scalar": upstream_scalar.detach().cpu(),
        "grad_rows_fp16": grad_rows(logits_fp16),
        "grad_rows_bf16": grad_rows(logits_bf16),
        "grad_scalar_fp16": grad_scalar(logits_fp16),
        "grad_scalar_bf16": grad_scalar(logits_bf16),
    }

    save_file(grads, Path(__file__).parent / "reference.safetensors")


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
