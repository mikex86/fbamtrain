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
UPSTREAM_LOW = -0.25
UPSTREAM_HIGH = 0.25
INPUT_SEED = 2024
UPSTREAM_SEED = 1331
INIT_SEED = 1337


def kaiming_uniform_(tensor: torch.Tensor, fan_in: int, seed: int):
    bound = (1.0 / float(fan_in)) ** 0.5
    fill_uniform_(tensor, -bound, bound, seed)


def generate(dtype: torch.dtype, output_path: str):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required to generate reference tensors")

    device = torch.device("cuda")

    x_nhwc = torch.empty((BATCH, HEIGHT, WIDTH, IN_CHANNELS), dtype=dtype, device=device)
    fill_uniform_(x_nhwc, INPUT_LOW, INPUT_HIGH, INPUT_SEED)
    x_nchw = x_nhwc.permute(0, 3, 1, 2).contiguous()
    x_nchw.requires_grad_(True)

    weight_hwio = torch.empty((OUT_CHANNELS, KERNEL_SIZE, KERNEL_SIZE, IN_CHANNELS), dtype=dtype, device=device)
    fan_in = IN_CHANNELS * KERNEL_SIZE * KERNEL_SIZE
    kaiming_uniform_(weight_hwio, fan_in, INIT_SEED)
    weight_oihw = weight_hwio.permute(0, 3, 1, 2).contiguous()
    weight_oihw.requires_grad_(True)

    y = torch.nn.functional.conv2d(
        x_nchw,
        weight_oihw,
        bias=None,
        stride=(STRIDE, STRIDE),
        padding=(PADDING, PADDING),
        dilation=(DILATION, DILATION),
    )

    out_h = y.shape[2]
    out_w = y.shape[3]
    upstream_nhwc = torch.empty((BATCH, out_h, out_w, OUT_CHANNELS), dtype=dtype, device=device)
    fill_uniform_(upstream_nhwc, UPSTREAM_LOW, UPSTREAM_HIGH, UPSTREAM_SEED)
    upstream_nchw = upstream_nhwc.permute(0, 3, 1, 2).contiguous()

    y.backward(upstream_nchw)

    grad_input_nhwc = x_nchw.grad.permute(0, 2, 3, 1).contiguous()
    grad_weight_hwio = weight_oihw.grad.permute(0, 2, 3, 1).contiguous()

    save_file(
        {
            "input": x_nhwc.detach().cpu(),
            "weight": weight_hwio.detach().cpu(),
            "upstream": upstream_nhwc.detach().cpu(),
            "grad_input": grad_input_nhwc.detach().cpu(),
            "grad_weight": grad_weight_hwio.detach().cpu(),
        },
        output_path,
    )


def main():
    generate(torch.bfloat16, "reference_bf16.safetensors")
    generate(torch.float16, "reference_fp16.safetensors")


if __name__ == "__main__":
    main()
