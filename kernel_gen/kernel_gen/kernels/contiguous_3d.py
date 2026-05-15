import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_X": 256, "BLOCK_SIZE_Y": 4, "BLOCK_SIZE_Z": 128, "GROUP_X": 8}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE_X": 128, "BLOCK_SIZE_Y": 8, "BLOCK_SIZE_Z": 128, "GROUP_X": 8}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE_X": 128, "BLOCK_SIZE_Y": 4, "BLOCK_SIZE_Z": 64, "GROUP_X": 4}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_X": 64, "BLOCK_SIZE_Y": 16, "BLOCK_SIZE_Z": 64, "GROUP_X": 4}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE_X": 64, "BLOCK_SIZE_Y": 8, "BLOCK_SIZE_Z": 128, "GROUP_X": 4}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_X": 64, "BLOCK_SIZE_Y": 4, "BLOCK_SIZE_Z": 64, "GROUP_X": 4}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_X": 32, "BLOCK_SIZE_Y": 8, "BLOCK_SIZE_Z": 128, "GROUP_X": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_X": 32, "BLOCK_SIZE_Y": 4, "BLOCK_SIZE_Z": 64, "GROUP_X": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_X": 128, "BLOCK_SIZE_Y": 1, "BLOCK_SIZE_Z": 128, "GROUP_X": 4}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE_X": 64, "BLOCK_SIZE_Y": 1, "BLOCK_SIZE_Z": 128, "GROUP_X": 4}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE_X": 64, "BLOCK_SIZE_Y": 1, "BLOCK_SIZE_Z": 256, "GROUP_X": 4}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE_X": 32, "BLOCK_SIZE_Y": 1, "BLOCK_SIZE_Z": 256, "GROUP_X": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_X": 32, "BLOCK_SIZE_Y": 1, "BLOCK_SIZE_Z": 512, "GROUP_X": 8}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE_X": 32, "BLOCK_SIZE_Y": 1, "BLOCK_SIZE_Z": 256, "GROUP_X": 1}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_X": 16, "BLOCK_SIZE_Y": 1, "BLOCK_SIZE_Z": 256, "GROUP_X": 1}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_X": 16, "BLOCK_SIZE_Y": 2, "BLOCK_SIZE_Z": 128, "GROUP_X": 1}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_X": 64, "BLOCK_SIZE_Y": 1, "BLOCK_SIZE_Z": 256, "GROUP_X": 4}, num_warps=4, num_stages=1),
    ],
    key=["X", "Y", "Z"],
)
@triton.jit
def contiguous_3d(
    out_ptr,
    in_ptr,
    stride_x,
    stride_y,
    stride_z,
    X,
    Y,
    Z,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
    BLOCK_SIZE_Z: tl.constexpr,
    GROUP_X: tl.constexpr,
):
    pid_x = tl.program_id(0)
    pid_yz = tl.program_id(1)

    num_tiles_x = (X + BLOCK_SIZE_X - 1) // BLOCK_SIZE_X
    num_tiles_y = (Y + BLOCK_SIZE_Y - 1) // BLOCK_SIZE_Y
    num_tiles_z = (Z + BLOCK_SIZE_Z - 1) // BLOCK_SIZE_Z

    group_id = pid_x // GROUP_X
    first_x = group_id * GROUP_X
    x_tile = first_x + (pid_x % GROUP_X)
    if x_tile >= num_tiles_x:
        return

    y_tile = pid_yz % num_tiles_y
    z_tile = pid_yz // num_tiles_y
    if y_tile >= num_tiles_y or z_tile >= num_tiles_z:
        return

    x = x_tile * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    z = z_tile * BLOCK_SIZE_Z + tl.arange(0, BLOCK_SIZE_Z)

    tl.multiple_of(x, BLOCK_SIZE_X)
    tl.multiple_of(z, BLOCK_SIZE_Z)

    x_offsets = x[:, None] * stride_x
    z_offsets = z[None, :] * stride_z

    in_xz = in_ptr + x_offsets + z_offsets
    out_x = (x * Y)[:, None] * Z

    y_base = y_tile * BLOCK_SIZE_Y

    in_row_base = in_xz + y_base * stride_y
    out_row_base = out_ptr + out_x + y_base * Z

    full_tile = ((x_tile + 1) * BLOCK_SIZE_X <= X) & ((y_tile + 1) * BLOCK_SIZE_Y <= Y) & ((z_tile + 1) * BLOCK_SIZE_Z <= Z)

    if tl.constexpr(BLOCK_SIZE_Y == 1):
        if full_tile:
            slab = tl.load(in_row_base)
            tl.store(out_row_base + z[None, :], slab)
        else:
            base_mask = (x[:, None] < X) & (z[None, :] < Z) & (y_base < Y)
            slab = tl.load(in_row_base, mask=base_mask, other=0)
            tl.store(out_row_base + z[None, :], slab, mask=base_mask)
    else:
        if full_tile:
            for y_iter in tl.static_range(BLOCK_SIZE_Y):
                in_ptrs = in_row_base + y_iter * stride_y
                out_ptrs = out_row_base + y_iter * Z + z[None, :]
                slab = tl.load(in_ptrs)
                tl.store(out_ptrs, slab)
        else:
            base_mask = (x[:, None] < X) & (z[None, :] < Z)
            for y_iter in tl.static_range(BLOCK_SIZE_Y):
                y_idx = y_base + y_iter
                y_mask = base_mask & (y_idx < Y)
                in_ptrs = in_row_base + y_iter * stride_y
                out_ptrs = out_row_base + y_iter * Z + z[None, :]
                slab = tl.load(in_ptrs, mask=y_mask, other=0)
                tl.store(out_ptrs, slab, mask=y_mask)

    tl.max_contiguous(x, BLOCK_SIZE_X)
    tl.max_contiguous(z, BLOCK_SIZE_Z)


def contiguous_3d_into(x: torch.Tensor, out_flat: torch.Tensor):
    """
    Write a flat contiguous copy of 3D CUDA tensor x (shape [X, Y, Z]) into out_flat.
    Assumes x.stride(-1) == 1 (last dim Z is contiguous).
    """
    assert x.ndim == 3 and x.is_cuda, "x must be 3D CUDA"
    assert x.stride(-1) == 1, "expected last dim (Z) contiguous"
    X, Y, Z = x.shape
    sx, sy, sz = x.stride()

    grid = lambda META: (
        triton.cdiv(X, META["BLOCK_SIZE_X"]),
        triton.cdiv(Y, META["BLOCK_SIZE_Y"]) * triton.cdiv(Z, META["BLOCK_SIZE_Z"]),
    )

    contiguous_3d[grid](
        out_ptr=out_flat,
        in_ptr=x,
        stride_x=sx,
        stride_y=sy,
        stride_z=sz,
        X=X,
        Y=Y,
        Z=Z,
    )


def contiguous_3d_triton_flat(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty(x.numel(), device=x.device, dtype=x.dtype)
    contiguous_3d_into(x, out)
    return out


def time_cuda_ms(fn, iters=100, warmup=10):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda"

    B, T, C = 64, 4096, 256
    src = torch.randn(B, T, C, device=device, dtype=torch.bfloat16)

    out_triton = contiguous_3d_triton_flat(src)
    ref = src.contiguous().view(-1)
    print("Correct:", torch.equal(out_triton, ref), "max abs:", (out_triton - ref).abs().max().item())

    out_triton_buf = torch.empty(src.numel(), device=device, dtype=src.dtype)
    out_torch_buf = torch.empty_like(src.contiguous())

    triton_ms = time_cuda_ms(lambda: contiguous_3d_into(src, out_triton_buf))
    torch_copy_ms = time_cuda_ms(lambda: out_torch_buf.copy_(src))
    torch_alloc_ms = time_cuda_ms(lambda: src.contiguous())

    bytes_rw = src.numel() * src.element_size() * 2
    print(f"Triton avg: {triton_ms * 1e3:.1f} µs  (~{(bytes_rw / (triton_ms / 1e3)) / 1e9:.0f} GB/s)")
    print(f"Torch copy_ avg: {torch_copy_ms * 1e3:.1f} µs")
    print(f"Torch contiguous() avg: {torch_alloc_ms * 1e3:.1f} µs")
