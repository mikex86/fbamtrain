from pathlib import Path

import torch
from safetensors.torch import save_file


def build_case(dtype: torch.dtype) -> dict[str, torch.Tensor]:
    device = torch.device("cuda")
    torch.manual_seed(0)

    batch = 4
    in_features = 6
    out_features = 5

    x = torch.empty((batch, in_features), device=device, dtype=dtype).uniform_(-0.5, 0.5)
    w = torch.empty((in_features, out_features), device=device, dtype=dtype).uniform_(-0.5, 0.5)
    b = torch.empty((out_features,), device=device, dtype=dtype).uniform_(-0.5, 0.5)
    upstream = torch.empty((batch, out_features), device=device, dtype=dtype).uniform_(-0.5, 0.5)

    x = x.requires_grad_(True)
    w = w.requires_grad_(True)
    b = b.requires_grad_(True)

    y = x @ w + b
    y.backward(upstream)

    return {
        "x": x.detach().cpu(),
        "weight": w.detach().cpu(),
        "bias": b.detach().cpu(),
        "upstream": upstream.detach().cpu(),
        "grad_x": x.grad.detach().cpu(),
        "grad_w": w.grad.detach().cpu(),
        "grad_b": b.grad.detach().cpu(),
    }


def main() -> None:
    out_dir = Path(__file__).parent

    ref_bf16 = build_case(torch.bfloat16)
    save_file(ref_bf16, out_dir / "reference_bf16.safetensors")

    ref_fp16 = build_case(torch.float16)
    save_file(ref_fp16, out_dir / "reference_fp16.safetensors")


if __name__ == "__main__":
    main()
