import torch
import triton
import triton.language as tl
import triton.testing


def _default_device() -> torch.device:
    return triton.runtime.driver.active.get_active_torch_device()


configs = [
    triton.Config({'BLOCK_SIZE': 32}, num_warps=1, num_stages=2),
    triton.Config({'BLOCK_SIZE': 64}, num_warps=2, num_stages=2),
    triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_SIZE': 256}, num_warps=8, num_stages=2),
]


@triton.autotune(configs=configs, key=['embedding_dim'])
@triton.jit
def embedding_lookup(
        out_ptr,
        table_ptr,
        indices_ptr,
        num_indices,
        embedding_dim,
        stride_out_row,
        stride_out_col,
        stride_table_row,
        stride_table_col,
        BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    if row_id >= num_indices:
        return

    col_block = tl.program_id(1)
    cols = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < embedding_dim

    index = tl.load(indices_ptr + row_id).to(tl.int32)

    out_offsets = row_id * stride_out_row + cols * stride_out_col
    table_offsets = index * stride_table_row + cols * stride_table_col

    vals = tl.load(table_ptr + table_offsets, mask=mask, other=0.0)
    tl.store(out_ptr + out_offsets, vals, mask=mask)


def launch_embedding_lookup(
        table: torch.Tensor,
        indices: torch.Tensor,
        out: torch.Tensor | None = None,
        *,
        block_size: int | None = None,
):
    if table.dtype != torch.bfloat16:
        raise ValueError("embedding table must be bfloat16")
    if not table.is_contiguous():
        table = table.contiguous()
    if indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("indices must be int32 or int64")
    if not indices.is_cuda:
        raise ValueError("indices must be on CUDA device")

    indices_i32 = indices.to(torch.int32).contiguous()

    if out is None:
        out = torch.empty(*indices.shape, table.shape[1], dtype=torch.bfloat16, device=table.device)
    else:
        if out.dtype != torch.bfloat16:
            raise ValueError("output tensor must be bfloat16")
        if out.shape != (*indices.shape, table.shape[1]):
            raise ValueError("output tensor has incorrect shape")
        if not out.is_contiguous():
            out = out.contiguous()

    out_view = out.view(-1, table.shape[1])
    table_view = table.view(table.shape[0], table.shape[1])

    stride_out_row, stride_out_col = out_view.stride()
    stride_table_row, stride_table_col = table_view.stride()

    stride_out_row = int(stride_out_row)
    stride_out_col = int(stride_out_col)
    stride_table_row = int(stride_table_row)
    stride_table_col = int(stride_table_col)

    num_indices = indices_i32.numel()
    embedding_dim = table.shape[1]

    num_indices = int(num_indices)
    embedding_dim = int(embedding_dim)

    if block_size is None:
        grid = lambda META: (num_indices, triton.cdiv(embedding_dim, META['BLOCK_SIZE']))
        embedding_lookup[grid](
            out_ptr=out_view,
            table_ptr=table_view,
            indices_ptr=indices_i32,
            num_indices=num_indices,
            embedding_dim=embedding_dim,
            stride_out_row=stride_out_row,
            stride_out_col=stride_out_col,
            stride_table_row=stride_table_row,
            stride_table_col=stride_table_col,
        )
    else:
        block_size = int(block_size)
        grid = (num_indices, triton.cdiv(embedding_dim, block_size))
        embedding_lookup[grid](
            out_ptr=out_view,
            table_ptr=table_view,
            indices_ptr=indices_i32,
            num_indices=num_indices,
            embedding_dim=embedding_dim,
            stride_out_row=stride_out_row,
            stride_out_col=stride_out_col,
            stride_table_row=stride_table_row,
            stride_table_col=stride_table_col,
            BLOCK_SIZE=block_size,
        )

    return out


def benchmark_embedding(
        num_embeddings: int = 4096,
        embedding_dim: int = 256,
        batch: int = 16,
        tokens: int = 128,
        iters: int = 100,
):
    device = _default_device()
    torch.manual_seed(0)

    weight = torch.randn((num_embeddings, embedding_dim), device=device, dtype=torch.bfloat16)
    indices = torch.randint(0, num_embeddings, (batch, tokens), device=device, dtype=torch.int32)

    out_triton = torch.empty(batch, tokens, embedding_dim, device=device, dtype=torch.bfloat16)

    def run_triton():
        launch_embedding_lookup(weight, indices, out_triton)
        return out_triton

    def run_torch():
        return torch.nn.functional.embedding(indices.to(torch.long), weight)

    triton.testing.assert_close(run_triton(), run_torch(), atol=1e-2, rtol=1e-2)

    triton_ms = triton.testing.do_bench(run_triton, warmup=25, rep=iters)
    torch_ms = triton.testing.do_bench(run_torch, warmup=25, rep=iters)

    bytes_per_sample = weight.element_size() * embedding_dim
    throughput_triton = batch * tokens * bytes_per_sample / (triton_ms * 1e-3) / 1e9
    throughput_torch = batch * tokens * bytes_per_sample / (torch_ms * 1e-3) / 1e9

    print(f"EmbeddingLookup BF16 E={num_embeddings}, D={embedding_dim}, B={batch}, T={tokens}")
    print(f" Triton: {triton_ms:.3f} ms ({throughput_triton:.2f} GB/s)")
    print(f" Torch : {torch_ms:.3f} ms ({throughput_torch:.2f} GB/s)")

    return {
        "triton_ms": triton_ms,
        "torch_ms": torch_ms,
        "triton_gb_s": throughput_triton,
        "torch_gb_s": throughput_torch,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark Triton embedding kernel against PyTorch")
    parser.add_argument("--num-embeddings", type=int, default=4096, help="Vocabulary size")
    parser.add_argument("--embedding-dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--tokens", type=int, default=128, help="Sequence length")
    parser.add_argument("--iters", type=int, default=100, help="Number of benchmark iterations")

    args = parser.parse_args()

    benchmark_embedding(
        num_embeddings=args.num_embeddings,
        embedding_dim=args.embedding_dim,
        batch=args.batch,
        tokens=args.tokens,
        iters=args.iters,
    )


if __name__ == "__main__":
    main()
