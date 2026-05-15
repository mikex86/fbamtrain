from pathlib import Path

import torch
from safetensors.torch import save_file


def main():
    torch.manual_seed(0)
    device = torch.device("cuda")

    batch, hidden = 8, 16
    gates_fp16 = torch.randn(batch, 4 * hidden, device=device, dtype=torch.float16)
    c_prev_fp16 = torch.randn(batch, hidden, device=device, dtype=torch.float16)

    gates_fp32 = torch.randn(batch, 4 * hidden, device=device, dtype=torch.float32)
    c_prev_fp32 = torch.randn(batch, hidden, device=device, dtype=torch.float32)

    def run_reference(gates_f: torch.Tensor, c_prev_f: torch.Tensor):
        i = torch.sigmoid(gates_f[:, 0 * hidden: 1 * hidden])
        f = torch.sigmoid(gates_f[:, 1 * hidden: 2 * hidden])
        g = torch.tanh(gates_f[:, 2 * hidden: 3 * hidden])
        o = torch.sigmoid(gates_f[:, 3 * hidden: 4 * hidden])

        c_out_ref = f * c_prev_f + i * g
        h_out_ref = o * torch.tanh(c_out_ref)
        return h_out_ref, c_out_ref

    h_fp16, c_fp16 = run_reference(gates_fp16.float(), c_prev_fp16.float())
    h_fp32, c_fp32 = run_reference(gates_fp32, c_prev_fp32)

    save_file(
        {
            "gates": gates_fp16.cpu(),
            "c_prev": c_prev_fp16.cpu(),
            "expected_h": h_fp16.half().cpu(),
            "expected_c": c_fp16.half().cpu(),
            "gates_fp32_state": gates_fp32.cpu(),
            "c_prev_fp32_state": c_prev_fp32.cpu(),
            "expected_h_fp32_state": h_fp32.half().cpu(),
            "expected_c_fp32_state": c_fp32.cpu(),
        },
        Path(__file__).parent / "reference.safetensors",
    )


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for reference generation")
    main()
