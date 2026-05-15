import math
from pathlib import Path

import torch
from safetensors.torch import save_file


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required to generate reference")

    torch.manual_seed(1234)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False

    seq_len, batch, input_size, hidden_size, vocab = 64, 8, 16, 16, 32

    x = torch.randn(seq_len, batch, input_size, device=device, dtype=torch.float16)
    h0 = torch.randn(hidden_size, device=device, dtype=torch.float32)
    c0 = torch.randn(hidden_size, device=device, dtype=torch.float32)

    k = 1.0 / math.sqrt(hidden_size)
    w_ih = torch.empty(input_size, 4 * hidden_size, device=device, dtype=torch.float32).uniform_(-k, k)
    w_hh = torch.empty(hidden_size, 4 * hidden_size, device=device, dtype=torch.float32).uniform_(-k, k)
    b_ih = torch.zeros(4 * hidden_size, device=device, dtype=torch.float32)
    b_hh = torch.zeros(4 * hidden_size, device=device, dtype=torch.float32)

    head_weight = torch.empty(hidden_size, vocab, device=device, dtype=torch.float16).uniform_(-k, k)

    targets = torch.randint(low=0, high=vocab, size=(seq_len, batch), device=device, dtype=torch.int64)

    lstm = torch.nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
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

    x_f = x.to(dtype=torch.float32).detach().clone().requires_grad_(True)
    h0_req = h0.detach().clone().requires_grad_(True)
    c0_req = c0.detach().clone().requires_grad_(True)
    h0_exp = h0_req.view(1, 1, hidden_size).expand(1, batch, hidden_size).contiguous()
    c0_exp = c0_req.view(1, 1, hidden_size).expand(1, batch, hidden_size).contiguous()

    y, _ = lstm(x_f, (h0_exp, c0_exp))

    logits = torch.matmul(y.to(torch.float16), head_weight).to(torch.float32)
    loss_sum = torch.nn.functional.cross_entropy(logits.view(-1, vocab), targets.view(-1), reduction="sum")
    loss_mean = loss_sum / float(seq_len * batch)
    loss_mean.backward()

    grads = {
        "x": x.detach().cpu(),
        "h0": h0.detach().cpu(),
        "c0": c0.detach().cpu(),
        "w_ih": w_ih.detach().cpu(),
        "w_hh": w_hh.detach().cpu(),
        "b_ih": b_ih.detach().cpu(),
        "b_hh": b_hh.detach().cpu(),
        "head_weight": head_weight.detach().cpu(),
        "targets": targets.to(dtype=torch.float32).cpu(),
        "loss_sum": loss_sum.detach().unsqueeze(0).cpu(),
        "loss_mean": loss_mean.detach().unsqueeze(0).cpu(),
        "grad_x": x_f.grad.detach().cpu(),
        "grad_h0": h0_req.grad.detach().cpu(),
        "grad_c0": c0_req.grad.detach().cpu(),
        "grad_w_ih": lstm.weight_ih_l0.grad.detach().transpose(0, 1).contiguous().cpu(),
        "grad_w_hh": lstm.weight_hh_l0.grad.detach().transpose(0, 1).contiguous().cpu(),
        "grad_b_ih": lstm.bias_ih_l0.grad.detach().cpu(),
        "grad_b_hh": lstm.bias_hh_l0.grad.detach().cpu(),
        "grad_head_weight": head_weight.grad.detach().cpu()
        if head_weight.grad is not None
        else torch.zeros_like(head_weight).cpu(),
    }

    save_file(grads, Path(__file__).parent / "reference.safetensors")


if __name__ == "__main__":
    main()
