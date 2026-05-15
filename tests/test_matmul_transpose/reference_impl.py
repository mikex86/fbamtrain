import torch
from safetensors.torch import save_file


def main():
    torch.manual_seed(123)
    device = "cuda"

    # Shapes chosen to exercise non-contiguous transposed views.
    M, K, N = 8, 6, 8

    a_nn = torch.randn((M, K), device=device, dtype=torch.float16)
    b_nn = torch.randn((K, N), device=device, dtype=torch.float16)

    # Contiguous bases for transposed views.
    a_t = torch.randn((K, M), device=device, dtype=torch.float16)
    b_t = torch.randn((N, K), device=device, dtype=torch.float16)

    out_nn = a_nn @ b_nn
    out_ta = a_t.t() @ b_nn
    out_tb = a_nn @ b_t.t()
    out_tab = a_t.t() @ b_t.t()

    save_file(
        {
            "a_nn": a_nn.cpu(),
            "b_nn": b_nn.cpu(),
            "a_t": a_t.cpu(),
            "b_t": b_t.cpu(),
            "out_nn": out_nn.cpu(),
            "out_ta": out_ta.cpu(),
            "out_tb": out_tb.cpu(),
            "out_tab": out_tab.cpu(),
        },
        "reference.safetensors",
    )


if __name__ == "__main__":
    main()
