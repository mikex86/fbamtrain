import torch
from safetensors.torch import save_file


BATCH = 4
ROWS = 5
COLS = 6
SEED = 42


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required to generate reference tensors")

    torch.manual_seed(SEED)
    device = torch.device("cuda")

    x = torch.randn((BATCH, ROWS, COLS), device=device, dtype=torch.bfloat16)
    upstream_keep = torch.randn((BATCH, ROWS, 1), device=device, dtype=torch.bfloat16)
    upstream_time = torch.randn((BATCH, COLS), device=device, dtype=torch.bfloat16)

    grad_keep = (upstream_keep.float() / float(COLS)).to(torch.bfloat16).expand(BATCH, ROWS, COLS).contiguous()
    grad_time = (
        (upstream_time.float() / float(ROWS))
        .to(torch.bfloat16)
        .view(BATCH, 1, COLS)
        .expand(BATCH, ROWS, COLS)
        .contiguous()
    )

    save_file(
        {
            "x": x.cpu(),
            "upstream_keep": upstream_keep.cpu(),
            "upstream_time": upstream_time.cpu(),
            "grad_keep": grad_keep.cpu(),
            "grad_time": grad_time.cpu(),
        },
        "reference.safetensors",
    )


if __name__ == "__main__":
    main()
