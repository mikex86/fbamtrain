from pathlib import Path

import torch
from safetensors.torch import save_file


def main():
    device = torch.device("cuda")
    torch.manual_seed(0)

    M, K, N = 128, 256, 64
    a = torch.randn(M, K, device=device, dtype=torch.float16)
    b = torch.randn(K, N, device=device, dtype=torch.float16)
    # Use fp16 inputs but run the matmul in fp32 for a tighter reference.
    a_fp32 = a.to(torch.float32)
    b_fp32 = b.to(torch.float32)
    out = torch.matmul(a_fp32, b_fp32).cpu()

    save_file(
        {
            "a": a.cpu(),
            "b": b.cpu(),
            "out": out,
        },
        Path(__file__).parent / "reference.safetensors",
    )


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for reference generation")
    main()
