import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_X": 256, "BLOCK_SIZE_Y": 4, "BLOCK_SIZE_Z": 64, "BLOCK_SIZE_W": 8, "GROUP_X": 8}, num_warps=8,
                      num_stages=2),
        triton.Config({"BLOCK_SIZE_X": 128, "BLOCK_SIZE_Y": 8, "BLOCK_SIZE_Z": 64, "BLOCK_SIZE_W": 8, "GROUP_X": 8}, num_warps=8,
                      num_stages=2),
        triton.Config({"BLOCK_SIZE_X": 128, "BLOCK_SIZE_Y": 4, "BLOCK_SIZE_Z": 128, "BLOCK_SIZE_W": 8, "GROUP_X": 8}, num_warps=8,
                      num_stages=2),
        triton.Config({"BLOCK_SIZE_X": 128, "BLOCK_SIZE_Y": 4, "BLOCK_SIZE_Z": 64, "BLOCK_SIZE_W": 4, "GROUP_X": 8}, num_warps=4,
                      num_stages=2),
        triton.Config({"BLOCK_SIZE_X": 64, "BLOCK_SIZE_Y": 8, "BLOCK_SIZE_Z": 64, "BLOCK_SIZE_W": 4, "GROUP_X": 8}, num_warps=4,
                      num_stages=2),
    ],
    key=["X", "Y", "Z", "W"],
)
@triton.jit
def contiguous_4d(
        out_ptr, in_ptr,
        stride_x, stride_y, stride_z, stride_w,
        X, Y, Z, W,
        BLOCK_SIZE_X: tl.constexpr,
        BLOCK_SIZE_Y: tl.constexpr,
        BLOCK_SIZE_Z: tl.constexpr,
        BLOCK_SIZE_W: tl.constexpr,
        GROUP_X: tl.constexpr,
):
    # program ids:
    #   axis-0 -> grouped X tiles
    #   axis-1 -> (Y tiles * W tiles)
    #   axis-2 -> Z index (one Z per program)
    pid_x = tl.program_id(0)
    pid_yw = tl.program_id(1)
    z = tl.program_id(2)

    # group X tiles for better locality across adjacent Y/W tiles
    num_tiles_x = (X + BLOCK_SIZE_X - 1) // BLOCK_SIZE_X
    group_id = pid_x // GROUP_X
    first_x = group_id * GROUP_X
    x_tile = first_x + (pid_x % GROUP_X)
    if x_tile >= num_tiles_x or z >= Z:
        return

    num_tiles_y = (Y + BLOCK_SIZE_Y - 1) // BLOCK_SIZE_Y
    num_tiles_w = (W + BLOCK_SIZE_Z * BLOCK_SIZE_W - 1) // (BLOCK_SIZE_Z * BLOCK_SIZE_W)

    # decode pid_yw -> (y_tile, w_tile)
    y_tile = pid_yw % num_tiles_y
    w_tile = pid_yw // num_tiles_y
    if w_tile >= num_tiles_w:
        return

    # index vectors
    x = x_tile * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)  # [BX]
    w0 = w_tile * (BLOCK_SIZE_Z * BLOCK_SIZE_W) + tl.arange(0, BLOCK_SIZE_Z * BLOCK_SIZE_W)  # [BW*V]

    # base offsets
    in_base = z * stride_z
    # flat row-major: (((x*Y + y)*Z + z)*W + w)
    tl.multiple_of(w0, BLOCK_SIZE_W)  # help vectorization on W

    # iterate the Y tile in a compact loop
    for yy in range(BLOCK_SIZE_Y):
        y_idx = y_tile * BLOCK_SIZE_Y + yy
        valid_y = y_idx < Y

        # load slab [BX, BW*V], last dim W is unit-stride -> coalesced, vectorizable
        in_ptrs = in_ptr \
                  + x[:, None] * stride_x \
                  + y_idx * stride_y \
                  + in_base \
                  + w0[None, :] * stride_w
        mask = (x[:, None] < X) & valid_y & (w0[None, :] < W)
        slab = tl.load(in_ptrs, mask=mask, other=0)

        # store slab to flat contiguous output: (((x*Y + y_idx)*Z + z)*W + w0)
        out_row = (((x * Y + y_idx) * Z + z) * W)[:, None]
        out_ptrs = out_ptr + out_row + w0[None, :]
        tl.store(out_ptrs, slab, mask=mask)

    # codegen hints
    tl.max_contiguous(x, BLOCK_SIZE_X)
    tl.max_contiguous(w0, BLOCK_SIZE_Z * BLOCK_SIZE_W)


def contiguous_4d_into(x: torch.Tensor, out_flat: torch.Tensor):
    """
    Write a flat contiguous copy of 4D CUDA tensor x (shape [X, Y, Z, W]) into out_flat.
    Assumes x.stride(-1) == 1 (last dim W is contiguous, e.g., from a transpose that keeps last dim).
    """
    assert x.ndim == 4 and x.is_cuda, "x must be 4D CUDA"
    assert x.stride(-1) == 1, "expected last dim (W) contiguous"
    X, Y, Z, W = x.shape
    sx, sy, sz, sw = x.stride()

    grid = lambda META: (
        triton.cdiv(X, META["BLOCK_SIZE_X"]),
        triton.cdiv(Y, META["BLOCK_SIZE_Y"]) * triton.cdiv(W, META["BLOCK_SIZE_Z"] * META["BLOCK_SIZE_W"]),
        Z,
    )

    contiguous_4d[grid](
        out_ptr=out_flat, in_ptr=x,
        stride_x=sx, stride_y=sy, stride_z=sz, stride_w=sw,
        X=X, Y=Y, Z=Z, W=W,
    )


def contiguous_4d_triton_flat(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty(x.numel(), device=x.device, dtype=x.dtype)
    contiguous_4d_into(x, out)
    return out


def time_cuda_ms(fn, iters=100, warmup=10):
    start = torch.cuda.Event(enable_timing=True);
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters): fn()
    end.record();
    end.synchronize()
    return start.elapsed_time(end) / iters


if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda"

    # Example: start from (B,H,T,HS) and form (X,Y,Z,W) = (B,T,H,C) with transpose(1,2)
    B, H, T, HS = 32, 8, 7680, 128
    src = torch.randn(B, H, T, HS, device=device, dtype=torch.bfloat16)
    y = src.transpose(1, 2)  # shape (B,T,H,C) => treat as (X,Y,Z,W)

    # Correctness
    out_triton = contiguous_4d_triton_flat(y)
    ref = y.contiguous().view(-1)
    print("Correct:", torch.equal(out_triton, ref), "max abs:", (out_triton - ref).abs().max().item())

    # Timings (no allocs in timed lambdas)
    out_triton_buf = torch.empty(y.numel(), device=device, dtype=y.dtype)
    out_torch_buf = torch.empty_like(y.contiguous())

    triton_ms = time_cuda_ms(lambda: contiguous_4d_into(y, out_triton_buf))
    torch_copy_ms = time_cuda_ms(lambda: out_torch_buf.copy_(y))
    torch_alloc_ms = time_cuda_ms(lambda: y.contiguous())

    bytes_rw = y.numel() * y.element_size() * 2
    print(f"Triton avg: {triton_ms * 1e3:.1f} µs  (~{(bytes_rw / (triton_ms / 1e3)) / 1e9:.0f} GB/s)")
    print(f"Torch copy_ avg: {torch_copy_ms * 1e3:.1f} µs")
    print(f"Torch contiguous() avg: {torch_alloc_ms * 1e3:.1f} µs")
