from pathlib import Path

import torch
from safetensors.torch import save_file


def build_case(dtype: torch.dtype, eps: float) -> dict[str, torch.Tensor]:
    device = torch.device("cuda")
    torch.manual_seed(0)

    batch = 4
    seq = 3
    hidden = 5

    x = torch.empty((batch, seq, hidden), device=device, dtype=dtype).uniform_(-0.5, 0.5)
    w = torch.empty((hidden,), device=device, dtype=dtype).uniform_(-0.5, 0.5)
    upstream = torch.empty_like(x).uniform_(-0.5, 0.5)

    x = x.requires_grad_(True)
    w = w.requires_grad_(True)

    x_f = x.float()
    w_f = w.float()
    inv_rms = torch.rsqrt((x_f * x_f).mean(dim=-1, keepdim=True) + eps)
    y = (x_f * inv_rms * w_f).to(dtype)
    y.backward(upstream)

    return {
        "x": x.detach().cpu(),
        "weight": w.detach().cpu(),
        "upstream": upstream.detach().cpu(),
        "grad_x": x.grad.detach().cpu().to(dtype),
        "grad_w": w.grad.detach().cpu().to(dtype),
    }


def main() -> None:
    out_dir = Path(__file__).parent
    eps = 1e-5

    ref_bf16 = build_case(torch.bfloat16, eps)
    save_file(ref_bf16, out_dir / "reference_bf16.safetensors")

    ref_fp16 = build_case(torch.float16, eps)
    save_file(ref_fp16, out_dir / "reference_fp16.safetensors")


if __name__ == "__main__":
    main()
