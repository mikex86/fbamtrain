from pathlib import Path

import torch
from safetensors.torch import save_file

SEQ = 3
BATCH = 2
VOCAB = 8
HIDDEN = 16
TARGET_VALUES = [1, 0, 5, 3, 2, 6]


def run_case(dtype: torch.dtype, suffix: str, base_x: torch.Tensor | None = None):
    device = torch.device("cuda")

    if base_x is None:
        x = torch.randn(SEQ * BATCH, HIDDEN, device=device, dtype=dtype, requires_grad=True)
    else:
        x = base_x.detach().clone().to(device=device, dtype=dtype).requires_grad_(True)

    weight = torch.randn(HIDDEN, VOCAB, device=device, dtype=dtype, requires_grad=True)
    bias = torch.randn(VOCAB, device=device, dtype=dtype, requires_grad=True)

    targets = torch.tensor(TARGET_VALUES, device=device, dtype=torch.long)

    logits = x @ weight + bias
    loss = torch.nn.functional.cross_entropy(logits, targets, reduction="mean")
    loss.backward()

    return {
        f"x_{suffix}": x.detach().cpu(),
        f"weight_{suffix}": weight.detach().cpu(),
        f"bias_{suffix}": bias.detach().cpu(),
        f"grad_x_{suffix}": x.grad.detach().cpu(),
        f"grad_w_{suffix}": weight.grad.detach().cpu(),
        f"grad_b_{suffix}": bias.grad.detach().cpu(),
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required to generate reference")

    torch.manual_seed(0)
    fp16 = run_case(torch.float16, "fp16")
    bf16 = run_case(torch.bfloat16, "bf16", base_x=fp16["x_fp16"])

    combined = {**fp16, **bf16}

    save_file(combined, Path(__file__).parent / "reference.safetensors")


if __name__ == "__main__":
    main()
