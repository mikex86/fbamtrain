import math
import os
from pathlib import Path

import torch
from safetensors.torch import save_file


def main():
    torch.manual_seed(0)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False

    T = int(os.environ.get("LSTM_REF_T", 512))
    B = int(os.environ.get("LSTM_REF_B", 32))
    I = int(os.environ.get("LSTM_REF_I", 1024))
    H = int(os.environ.get("LSTM_REF_H", 1024))

    x = torch.randn(T, B, I, device=device, dtype=torch.float16)
    h0 = torch.randn(H, device=device, dtype=torch.float32)
    c0 = torch.randn(H, device=device, dtype=torch.float32)

    w_ih = torch.empty(I, 4 * H, device=device, dtype=torch.float32)
    w_hh = torch.empty(H, 4 * H, device=device, dtype=torch.float32)
    k = 1.0 / math.sqrt(H)
    w_ih.uniform_(-k, k)
    w_hh.uniform_(-k, k)
    b_ih = torch.zeros(4 * H, device=device, dtype=torch.float32)
    b_hh = torch.zeros(4 * H, device=device, dtype=torch.float32)

    # mirror the C++ path: quantize inputs/weights to fp16, but do matmul in fp32 with quantized values; cell state fp32
    h = h0.to(torch.float16).view(1, H).expand(B, H)
    c = c0.view(1, H).expand(B, H)  # keep fp32
    w_ih_q = w_ih.to(torch.float16)
    w_hh_q = w_hh.to(torch.float16)
    outputs = []
    for t in range(T):
        x_t = x[t].to(torch.float16)
        gates = torch.matmul(x_t, w_ih_q).to(torch.float32) + torch.matmul(h, w_hh_q).to(torch.float32)
        gates = gates + b_ih + b_hh

        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c = f * c + i * g  # fp32 cell state
        h = (o * torch.tanh(c)).to(torch.float16)  # hidden fp16
        outputs.append(h)

    y = torch.stack(outputs, dim=0)

    save_file(
        {
            "x": x.cpu(),
            "h0": h0.cpu(),
            "c0": c0.cpu(),
            "w_ih": w_ih.cpu(),
            "w_hh": w_hh.cpu(),
            "b_ih": b_ih.cpu(),
            "b_hh": b_hh.cpu(),
            "y": y.cpu(),
            "h_n": h.cpu(),
            "c_n": c.to(torch.float16).cpu(),
        },
        Path(__file__).parent / "reference.safetensors",
    )


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for reference generation")
    main()
