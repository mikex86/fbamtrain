import math
from pathlib import Path

import torch
from safetensors.torch import save_file


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required to generate reference")

    torch.manual_seed(1234)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False

    seq_len, batch, input_size, hidden_size = 880, 1024, 1024, 1024

    x = torch.randn(seq_len, batch, input_size, device=device, dtype=torch.float16)
    h0 = torch.randn(hidden_size, device=device, dtype=torch.float32)
    c0 = torch.randn(hidden_size, device=device, dtype=torch.float32)

    # use a stable init to avoid saturating gates and match the C++ fp16-matmul path
    k = 1.0 / math.sqrt(hidden_size)
    w_ih = torch.empty(input_size, 4 * hidden_size, device=device, dtype=torch.float32).uniform_(-k, k)
    w_hh = torch.empty(hidden_size, 4 * hidden_size, device=device, dtype=torch.float32).uniform_(-k, k)
    b_ih = torch.zeros(4 * hidden_size, device=device, dtype=torch.float32)
    b_hh = torch.zeros(4 * hidden_size, device=device, dtype=torch.float32)
    # mirror the C++ path: quantize inputs/weights to fp16, matmul in fp32, cell state fp32
    h = h0.to(torch.float16).view(1, hidden_size).expand(batch, hidden_size)
    c = c0.view(1, hidden_size).expand(batch, hidden_size)
    w_ih_q = w_ih.to(torch.float16)
    w_hh_q = w_hh.to(torch.float16)
    outputs = []
    for t in range(seq_len):
        gates = torch.matmul(x[t], w_ih_q).to(torch.float32) + torch.matmul(h, w_hh_q).to(torch.float32)
        gates = gates + b_ih + b_hh
        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c = f * c + i * g
        h = (o * torch.tanh(c)).to(torch.float16)
        outputs.append(h)
    logits = torch.stack(outputs, dim=0)

    targets = torch.randint(low=0, high=hidden_size, size=(seq_len, batch), device=device, dtype=torch.int64)

    loss_sum = torch.nn.functional.cross_entropy(logits.float().view(-1, hidden_size), targets.view(-1),
                                                 reduction="sum")
    loss_mean = loss_sum / float(seq_len * batch)

    save_file(
        {
            "x": x.cpu(),
            "h0": h0.cpu(),
            "c0": c0.cpu(),
            "w_ih": w_ih.cpu(),
            "w_hh": w_hh.cpu(),
            "b_ih": b_ih.cpu(),
            "b_hh": b_hh.cpu(),
            "targets": targets.to(dtype=torch.float32).cpu(),
            "loss_sum": loss_sum.unsqueeze(0).cpu(),
            "loss_mean": loss_mean.unsqueeze(0).cpu(),
        },
        Path(__file__).parent / "reference.safetensors",
    )


if __name__ == "__main__":
    main()
