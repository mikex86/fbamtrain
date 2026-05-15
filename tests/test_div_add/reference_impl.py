import torch
from safetensors.torch import save_file


ROWS = 64
COLS = 128
TIME = 5
SEED = 123


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required to generate reference tensors")

    torch.manual_seed(SEED)
    device = torch.device("cuda")

    lhs_scalar = torch.randn((ROWS, COLS), device=device, dtype=torch.bfloat16)
    rhs_scalar = torch.randn((ROWS, COLS), device=device, dtype=torch.bfloat16)
    denom_scalar = torch.randn((1,), device=device, dtype=torch.bfloat16)

    lhs_elem = torch.randn((ROWS, COLS), device=device, dtype=torch.bfloat16)
    rhs_elem = torch.randn((ROWS, COLS), device=device, dtype=torch.bfloat16)
    denom_elem = torch.randn((ROWS, COLS), device=device, dtype=torch.bfloat16)

    out_scalar = (lhs_scalar.float() + rhs_scalar.float() / denom_scalar.float()).to(torch.bfloat16)
    out_elem = (lhs_elem.float() + rhs_elem.float() / denom_elem.float()).to(torch.bfloat16)

    lhs_bcast = torch.randn((ROWS, TIME, COLS), device=device, dtype=torch.bfloat16)
    rhs_bcast = torch.randn((ROWS, 1, COLS), device=device, dtype=torch.bfloat16)
    denom_bcast = torch.randn((1,), device=device, dtype=torch.bfloat16)
    out_bcast = (lhs_bcast.float() + rhs_bcast.float() / denom_bcast.float()).to(torch.bfloat16)

    save_file(
        {
            "lhs_scalar": lhs_scalar.cpu(),
            "rhs_scalar": rhs_scalar.cpu(),
            "denom_scalar": denom_scalar.cpu(),
            "out_scalar": out_scalar.cpu(),
            "lhs_elem": lhs_elem.cpu(),
            "rhs_elem": rhs_elem.cpu(),
            "denom_elem": denom_elem.cpu(),
            "out_elem": out_elem.cpu(),
            "lhs_bcast": lhs_bcast.cpu(),
            "rhs_bcast": rhs_bcast.cpu(),
            "denom_bcast": denom_bcast.cpu(),
            "out_bcast": out_bcast.cpu(),
        },
        "reference.safetensors",
    )


if __name__ == "__main__":
    main()
