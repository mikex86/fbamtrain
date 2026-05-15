from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main():
    torch.manual_seed(1234)
    device = torch.device("cuda")

    base_ref = load_file(Path(__file__).parent.parent / "test_lstm_streaming_torch" / "reference.safetensors")

    x = base_ref["x"].to(device)
    h0 = base_ref["h0"].to(device)
    c0 = base_ref["c0"].to(device)
    w_ih = base_ref["w_ih"].to(device)
    w_hh = base_ref["w_hh"].to(device)
    b_ih = base_ref["b_ih"].to(device)
    b_hh = base_ref["b_hh"].to(device)

    seq, batch, input_size = x.shape
    hidden_size = h0.shape[0]

    dy = torch.randn(seq, batch, hidden_size, device=device, dtype=torch.float32)

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
    h0_expanded = h0_req.view(1, 1, hidden_size).expand(1, batch, hidden_size).contiguous()
    c0_expanded = c0_req.view(1, 1, hidden_size).expand(1, batch, hidden_size).contiguous()

    y, (h_n, c_n) = lstm(x_f, (h0_expanded, c0_expanded))
    loss = (y * dy).sum()
    loss.backward()

    grads = {
        "x": x.detach().cpu(),
        "h0": h0.detach().cpu(),
        "c0": c0.detach().cpu(),
        "w_ih": w_ih.detach().cpu(),
        "w_hh": w_hh.detach().cpu(),
        "b_ih": b_ih.detach().cpu(),
        "b_hh": b_hh.detach().cpu(),
        "dy": dy.detach().cpu(),
        "grad_x": x_f.grad.detach().cpu(),
        "grad_h0": h0_req.grad.detach().cpu(),
        "grad_c0": c0_req.grad.detach().cpu(),
        "grad_w_ih": lstm.weight_ih_l0.grad.detach().transpose(0, 1).contiguous().cpu(),
        "grad_w_hh": lstm.weight_hh_l0.grad.detach().transpose(0, 1).contiguous().cpu(),
        "grad_b_ih": lstm.bias_ih_l0.grad.detach().cpu(),
        "grad_b_hh": lstm.bias_hh_l0.grad.detach().cpu(),
    }

    save_file(grads, Path(__file__).parent / "reference.safetensors")


if __name__ == "__main__":
    main()
