from pathlib import Path

import torch
from safetensors.torch import save_file


def main():
    torch.manual_seed(1234)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for reference generation")

    device = torch.device("cuda")

    T, B, I, H = 16, 4, 32, 32

    # inputs and parameters in streaming layout
    x_fp16 = torch.randn(T, B, I, device=device, dtype=torch.float16)
    h0 = torch.randn(H, device=device, dtype=torch.float32)
    c0 = torch.randn(H, device=device, dtype=torch.float32)

    w_ih = torch.randn(I, 4 * H, device=device, dtype=torch.float32)
    w_hh = torch.randn(H, 4 * H, device=device, dtype=torch.float32)
    b_ih = torch.randn(4 * H, device=device, dtype=torch.float32)
    b_hh = torch.randn(4 * H, device=device, dtype=torch.float32)

    # torch.nn.LSTM expects weight layout (4H, I) / (4H, H)
    lstm = torch.nn.LSTM(
        input_size=I,
        hidden_size=H,
        num_layers=1,
        batch_first=False,
        bias=True,
        device=device,
        dtype=torch.float32,
    )

    with torch.no_grad():
        lstm.weight_ih_l0.copy_(w_ih.transpose(0, 1))
        lstm.weight_hh_l0.copy_(w_hh.transpose(0, 1))
        lstm.bias_ih_l0.copy_(b_ih)
        lstm.bias_hh_l0.copy_(b_hh)

    x_fp32 = x_fp16.float()
    h0_expanded = h0.view(1, 1, H).expand(1, B, H).contiguous()
    c0_expanded = c0.view(1, 1, H).expand(1, B, H).contiguous()

    with torch.no_grad():
        y, (h_n, c_n) = lstm(x_fp32, (h0_expanded, c0_expanded))

    save_file(
        {
            "x": x_fp16.cpu(),
            "h0": h0.cpu(),
            "c0": c0.cpu(),
            "w_ih": w_ih.cpu(),
            "w_hh": w_hh.cpu(),
            "b_ih": b_ih.cpu(),
            "b_hh": b_hh.cpu(),
            "y": y.to(dtype=torch.float16).cpu(),
            "h_n": h_n.squeeze(0).to(dtype=torch.float16).cpu(),
            "c_n": c_n.squeeze(0).to(dtype=torch.float16).cpu(),
        },
        Path(__file__).parent / "reference.safetensors",
    )


if __name__ == "__main__":
    main()
