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


BATCH = 32
IN_CHANNELS = 32
OUT_CHANNELS = 64
HEIGHT = 48
WIDTH = 160
KERNEL_SIZE = 3
STRIDE = 1
PADDING = 1
DILATION = 1
INPUT_LOW = -0.5
INPUT_HIGH = 0.5
INPUT_SEED = 2024
INIT_SEED = 1337


def kaiming_uniform_(tensor: torch.Tensor, fan_in: int, seed: int):
    bound = (1.0 / float(fan_in)) ** 0.5
    fill_uniform_(tensor, -bound, bound, seed)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16

    x_nchw = torch.empty((BATCH, IN_CHANNELS, HEIGHT, WIDTH), dtype=dtype, device=device)
    fill_uniform_(x_nchw, INPUT_LOW, INPUT_HIGH, INPUT_SEED)
    x_nhwc = x_nchw.permute(0, 2, 3, 1).contiguous()

    weight_hwio = torch.empty((OUT_CHANNELS, KERNEL_SIZE, KERNEL_SIZE, IN_CHANNELS), dtype=dtype, device=device)

    fan_in = IN_CHANNELS * KERNEL_SIZE * KERNEL_SIZE
    kaiming_uniform_(weight_hwio, fan_in, INIT_SEED)
    weight_oihw = weight_hwio.permute(0, 3, 1, 2).contiguous()

    y = torch.nn.functional.conv2d(
        x_nhwc.permute(0, 3, 1, 2).contiguous().to(torch.float32),
        weight_oihw.to(torch.float32),
        stride=(STRIDE, STRIDE),
        padding=(PADDING, PADDING),
        dilation=(DILATION, DILATION),
    ).to(dtype)
    y_nhwc = y.permute(0, 2, 3, 1).contiguous()

    save_file({"output": y_nhwc.cpu()}, "reference.safetensors")


if __name__ == "__main__":
    main()
