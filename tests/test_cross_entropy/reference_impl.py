from pathlib import Path

import torch
from safetensors.torch import save_file

SEQ = 3
BATCH = 2
VOCAB = 8
TARGET_VALUES = [1, 0, 5, 3, 2, 6]


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required to generate reference")

    device = torch.device("cuda")
    torch.manual_seed(0)

    logits_fp16 = torch.randn(SEQ, BATCH, VOCAB, device=device, dtype=torch.float16)
    logits_bf16 = logits_fp16.float().to(torch.bfloat16)

    targets = torch.tensor(TARGET_VALUES, device=device, dtype=torch.int64).view(SEQ, BATCH)

    loss_fp16 = torch.nn.functional.cross_entropy(
        logits_fp16.float().view(-1, VOCAB), targets.view(-1), reduction="none"
    ).view(SEQ, BATCH).to(dtype=logits_fp16.dtype)
    loss_bf16 = torch.nn.functional.cross_entropy(
        logits_bf16.float().view(-1, VOCAB), targets.view(-1), reduction="none"
    ).view(SEQ, BATCH).to(dtype=logits_bf16.dtype)

    sum_fp32 = torch.nn.functional.cross_entropy(
        logits_fp16.float().view(-1, VOCAB), targets.view(-1), reduction="sum"
    ).unsqueeze(0)
    sum_bf32 = torch.nn.functional.cross_entropy(
        logits_bf16.float().view(-1, VOCAB), targets.view(-1), reduction="sum"
    ).unsqueeze(0)

    save_file(
        {
            "logits_fp16": logits_fp16.cpu(),
            "logits_bf16": logits_bf16.cpu(),
            "loss_fp16": loss_fp16.cpu(),
            "loss_bf16": loss_bf16.cpu(),
            "sum_fp32": sum_fp32.cpu(),
            "sum_bf32": sum_bf32.cpu(),
        },
        Path(__file__).parent / "reference.safetensors",
    )


if __name__ == "__main__":
    main()
