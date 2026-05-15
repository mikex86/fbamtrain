import triton
import triton.language as tl


@triton.jit
def fill_normal(
        out_ptr,  # *float32
        n_elements: tl.uint32,  # number of outputs
        mean: tl.float32,  # mean of the normal
        std: tl.float32,  # std  of the normal
        seed: tl.uint32,  # RNG seed
        BLOCK_SIZE: tl.constexpr,  # block size
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    nonce = 0x9E3779B9

    # Two independent uniform(0,1) streams; xor the seed for decorrelation
    u = tl.rand(seed, offs)
    v = tl.rand(seed ^ nonce.to(tl.uint32), offs)

    # Numerical safety: avoid log(0) and ignore out-of-bounds lanes
    u = tl.where(mask, tl.maximum(u, 1e-7), 0.5)
    v = tl.where(mask, v, 0.0)

    # Box–Muller transform -> N(0,1)
    r = tl.sqrt(-2.0 * tl.log(u))
    theta = 6.283185307179586 * v
    z = r * tl.cos(theta)

    # Affine transform to N(mean, std^2) and store
    out = mean + std * z
    tl.store(out_ptr + offs, out, mask=mask)
