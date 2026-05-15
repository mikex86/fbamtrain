from pathlib import Path

import torch
from safetensors.torch import save_file


def build_case(dtype: torch.dtype) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    device = torch.device("cuda")

    k = 128
    m = 1024
    n = 4096

    a_ta_base = torch.empty((k, m), device=device, dtype=dtype).uniform_(-0.5, 0.5)
    b_ta_base = torch.empty((k, n), device=device, dtype=dtype).uniform_(-0.5, 0.5)
    out_ta = a_ta_base.t() @ b_ta_base

    a_tb_base = torch.empty((k, n), device=device, dtype=dtype).uniform_(-0.5, 0.5)
    b_tb_base = torch.empty((m, n), device=device, dtype=dtype).uniform_(-0.5, 0.5)
    out_tb = a_tb_base @ b_tb_base.t()

    return {
        "a_ta_base": a_ta_base.detach().cpu(),
        "b_ta_base": b_ta_base.detach().cpu(),
        "out_ta": out_ta.detach().cpu(),
        "a_tb_base": a_tb_base.detach().cpu(),
        "b_tb_base": b_tb_base.detach().cpu(),
        "out_tb": out_tb.detach().cpu(),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required")

    out_dir = Path(__file__).parent
    ref_bf16 = build_case(torch.bfloat16)
    save_file(ref_bf16, out_dir / "reference_bf16.safetensors")

    ref_fp16 = build_case(torch.float16)
    save_file(ref_fp16, out_dir / "reference_fp16.safetensors")


if __name__ == "__main__":
    main()
