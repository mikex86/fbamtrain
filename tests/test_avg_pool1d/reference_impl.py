import torch
import triton
import triton.language as tl
from safetensors.torch import save_file


@triton.jit
def fill_uniform(out_ptr, n_elements: tl.uint32, low: tl.float32, high: tl.float32, seed: tl.uint32,
                 BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    random = tl.rand(seed, offsets.to(tl.uint32))
    values = low + (high - low) * random
    tl.store(out_ptr + offsets, values, mask=mask)


BLOCK_SIZE = 1024


def fill_uniform_(tensor, low=0.0, high=1.0, seed=0):
    n_elements = tensor.numel()
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    fill_uniform[grid](tensor, n_elements, low, high, seed, BLOCK_SIZE=BLOCK_SIZE)


BATCH = 8
CHANNELS = 16
LENGTH = 128
KERNEL_SIZE = 3
STRIDE = 2
SEED = 1337


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required to generate reference tensors")

    device = torch.device("cuda")
    x = torch.zeros((BATCH, CHANNELS, LENGTH), dtype=torch.bfloat16, device=device)
    fill_uniform_(x, -1.0, 1.0, seed=SEED)

    y = torch.nn.functional.avg_pool1d(x.to(torch.float32), kernel_size=KERNEL_SIZE, stride=STRIDE).to(torch.bfloat16)

    save_file({"output": y.cpu()}, "reference.safetensors")


if __name__ == "__main__":
    main()
