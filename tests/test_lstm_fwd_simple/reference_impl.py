import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file


def main(t_index: int):
    torch.manual_seed(0)
    device = torch.device("cuda")

    T, B, I, H = 8, 4, 6, 5
    if not (0 <= t_index < T):
        raise ValueError(f"t_index must be in [0, {T}), got {t_index}")

    x = torch.randn(T, B, I, device=device, dtype=torch.float16)
    h0 = torch.randn(H, device=device, dtype=torch.float32)
    c0 = torch.randn(H, device=device, dtype=torch.float32)

    # store weights in (input, 4 * hidden) and (hidden, 4 * hidden) layouts to match kernel expectations
    w_ih = torch.randn(I, 4 * H, device=device, dtype=torch.float32)
    w_hh = torch.randn(H, 4 * H, device=device, dtype=torch.float32)
    b_ih = torch.randn(4 * H, device=device, dtype=torch.float32)
    b_hh = torch.randn(4 * H, device=device, dtype=torch.float32)

    # compute gates in fp16 to match kernel behavior
    w_ih_f16 = w_ih.to(torch.float16)
    w_hh_f16 = w_hh.to(torch.float16)
    bias_sum_fp16 = (b_ih + b_hh).to(torch.float16)

    h = h0.to(torch.float16).view(1, H).expand(B, H)
    c = c0.to(torch.float16).view(1, H).expand(B, H)
    outputs = []
    for t in range(T):
        gates = torch.matmul(x[t], w_ih_f16) + torch.matmul(h, w_hh_f16)
        gates = gates + bias_sum_fp16

        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c = f * c + i * g
        h = o * torch.tanh(c)
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
            "y_t": y[t_index].cpu(),
            "h_n": h.cpu(),
            "c_n": c.cpu(),
        },
        Path(__file__).parent / "reference.safetensors",
    )


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for reference generation")
    parser = argparse.ArgumentParser()
    parser.add_argument("--t-index", type=int, default=3, help="Time step to store as y_t")
    args = parser.parse_args()
    main(args.t_index)
