from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file


def build_case(dtype: torch.dtype) -> dict[str, torch.Tensor]:
    device = torch.device("cuda")
    torch.manual_seed(0)

    batch = 4
    embed_dim = 16
    hidden_dim = 4 * embed_dim

    x = torch.empty((batch, embed_dim), device=device, dtype=dtype).uniform_(-0.5, 0.5)
    w1 = torch.empty((embed_dim, hidden_dim), device=device, dtype=dtype).uniform_(-0.5, 0.5)
    b1 = torch.empty((hidden_dim,), device=device, dtype=dtype).uniform_(-0.5, 0.5)
    w2 = torch.empty((hidden_dim, embed_dim), device=device, dtype=dtype).uniform_(-0.5, 0.5)
    b2 = torch.empty((embed_dim,), device=device, dtype=dtype).uniform_(-0.5, 0.5)
    upstream = torch.empty((batch, embed_dim), device=device, dtype=dtype).uniform_(-0.5, 0.5)

    x = x.requires_grad_(True)
    w1 = w1.requires_grad_(True)
    b1 = b1.requires_grad_(True)
    w2 = w2.requires_grad_(True)
    b2 = b2.requires_grad_(True)

    hidden = F.gelu(x @ w1 + b1, approximate="none")
    y = hidden @ w2 + b2
    y.backward(upstream)

    return {
        "x": x.detach().cpu(),
        "w1": w1.detach().cpu(),
        "b1": b1.detach().cpu(),
        "w2": w2.detach().cpu(),
        "b2": b2.detach().cpu(),
        "upstream": upstream.detach().cpu(),
        "grad_x": x.grad.detach().cpu(),
        "grad_w1": w1.grad.detach().cpu(),
        "grad_b1": b1.grad.detach().cpu(),
        "grad_w2": w2.grad.detach().cpu(),
        "grad_b2": b2.grad.detach().cpu(),
    }


def main() -> None:
    out_dir = Path(__file__).parent

    ref_bf16 = build_case(torch.bfloat16)
    save_file(ref_bf16, out_dir / "reference_bf16.safetensors")

    ref_fp16 = build_case(torch.float16)
    save_file(ref_fp16, out_dir / "reference_fp16.safetensors")


if __name__ == "__main__":
    main()
