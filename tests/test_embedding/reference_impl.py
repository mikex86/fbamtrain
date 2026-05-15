import torch
import triton
import triton.language as tl
from safetensors.torch import save_file

BATCH = 4
SEQ_LEN = 64
VOCAB_SIZE = 512
EMBED_DIM = 128
BLOCK_SIZE = 1024
MODEL_INIT_SEED = 1337


@triton.jit
def fill_normal(
        out_ptr,  # *float32
        n_elements: tl.uint32,
        mean: tl.float32,
        std: tl.float32,
        seed: tl.uint32,
        BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    nonce = 0x9E3779B9

    u = tl.rand(seed, offs)
    v = tl.rand(seed ^ nonce.to(tl.uint32), offs)

    u = tl.where(mask, tl.maximum(u, 1e-7), 0.5)
    v = tl.where(mask, v, 0.0)

    r = tl.sqrt(-2.0 * tl.log(u))
    theta = 6.283185307179586 * v
    z = r * tl.cos(theta)

    out = mean + std * z
    tl.store(out_ptr + offs, out, mask=mask)


def fill_normal_(tensor: torch.Tensor, mean: float, std: float, seed: int):
    n_elements = tensor.numel()
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    fill_normal[grid](tensor, n_elements, mean, std, seed, BLOCK_SIZE=BLOCK_SIZE)


def compute_indices(total: int) -> torch.Tensor:
    linear = torch.arange(total, dtype=torch.int32, device="cuda")
    return (linear * 17 + 11) % VOCAB_SIZE


def main():
    device = torch.device("cuda")
    weight = torch.empty((VOCAB_SIZE, EMBED_DIM), dtype=torch.bfloat16, device=device)
    fill_normal_(weight, 0.0, 1.0, MODEL_INIT_SEED)

    indices = compute_indices(BATCH * SEQ_LEN).view(BATCH, SEQ_LEN)
    output = torch.nn.functional.embedding(indices.to(torch.long), weight)

    save_file({
        "output": output.cpu(),
    }, "reference.safetensors")


if __name__ == "__main__":
    main()
