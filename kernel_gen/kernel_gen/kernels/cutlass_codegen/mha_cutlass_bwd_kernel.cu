#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/default_gemm_configuration.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/numeric_types.h>

#include "third_party/cutlass/examples/41_fused_multi_head_attention/kernel_backward.h"

namespace cutlass {
namespace gemm {
namespace device {

// Example-41 backward depends on DefaultGemmConfiguration entries that are not
// provided for newer arch tags in this CUTLASS path. Forwarding these arch tags
// to Sm80 keeps the configuration policy in CUTLASS while allowing explicit
// sm89/sm90/sm100 instantiation.
template <typename ElementA, typename ElementB, typename ElementC, typename ElementAccumulator>
struct DefaultGemmConfiguration<arch::OpClassTensorOp, arch::Sm89, ElementA, ElementB, ElementC, ElementAccumulator>
    : DefaultGemmConfiguration<arch::OpClassTensorOp, arch::Sm80, ElementA, ElementB, ElementC, ElementAccumulator> {};

template <typename ElementA, typename ElementB, typename ElementC, typename ElementAccumulator>
struct DefaultGemmConfiguration<arch::OpClassTensorOp, arch::Sm90, ElementA, ElementB, ElementC, ElementAccumulator>
    : DefaultGemmConfiguration<arch::OpClassTensorOp, arch::Sm80, ElementA, ElementB, ElementC, ElementAccumulator> {};

template <typename ElementA, typename ElementB, typename ElementC, typename ElementAccumulator>
struct DefaultGemmConfiguration<arch::OpClassTensorOp, arch::Sm100, ElementA, ElementB, ElementC, ElementAccumulator>
    : DefaultGemmConfiguration<arch::OpClassTensorOp, arch::Sm80, ElementA, ElementB, ElementC, ElementAccumulator> {};

}  // namespace device
}  // namespace gemm
}  // namespace cutlass

#ifndef CUTLASS_MHA_BWD_KERNEL_NAME
#error "CUTLASS_MHA_BWD_KERNEL_NAME not specified!"
#endif

#ifndef CUTLASS_MHA_BWD_FP16
#error "CUTLASS_MHA_BWD_FP16 not specified!"
#endif

#ifndef CUTLASS_MHA_BWD_HEAD_DIM
#error "CUTLASS_MHA_BWD_HEAD_DIM not specified!"
#endif

#ifndef CUTLASS_MHA_BWD_BLOCK_I
#error "CUTLASS_MHA_BWD_BLOCK_I not specified!"
#endif

#ifndef CUTLASS_MHA_BWD_BLOCK_J
#error "CUTLASS_MHA_BWD_BLOCK_J not specified!"
#endif

#ifndef CUTLASS_MHA_BWD_SM_ARCH
#error "CUTLASS_MHA_BWD_SM_ARCH not specified!"
#endif

#ifndef CUTLASS_MHA_BWD_SM_VERSION
#error "CUTLASS_MHA_BWD_SM_VERSION not specified!"
#endif

namespace {

#if CUTLASS_MHA_BWD_FP16
using Element = cutlass::half_t;
#else
using Element = cutlass::bfloat16_t;
#endif

constexpr int kHeadDim = CUTLASS_MHA_BWD_HEAD_DIM;
constexpr int kBlockSizeI = CUTLASS_MHA_BWD_BLOCK_I;
constexpr int kBlockSizeJ = CUTLASS_MHA_BWD_BLOCK_J;
constexpr int kMaxK = kHeadDim;
// Example-41 backward does not provide a native Sm100 instantiation path in
// this single-kernel ABI. For sm100 codegen, use Sm90 template instantiation.
using RequestedArch =
    std::conditional_t<(CUTLASS_MHA_BWD_SM_VERSION >= 100), cutlass::arch::Sm90, CUTLASS_MHA_BWD_SM_ARCH>;
static_assert(
    (CUTLASS_MHA_BWD_SM_VERSION >= 100) ||
        (RequestedArch::kMinComputeCapability == CUTLASS_MHA_BWD_SM_VERSION),
    "CUTLASS_MHA_BWD_SM_VERSION must match CUTLASS_MHA_BWD_SM_ARCH for non-sm100 builds.");
constexpr bool kIsHalf = cutlass::sizeof_bits<Element>::value <= 16;
constexpr bool kOutputInRF = kIsHalf && kMaxK <= kBlockSizeI;
constexpr bool kPreload = kIsHalf && RequestedArch::kMinComputeCapability >= 80 && kOutputInRF;

using Attention = AttentionBackwardKernel<
    RequestedArch,
    Element,
    true,   // kIsAligned_
    false,  // kApplyDropout_
    kPreload,
    kBlockSizeI,
    kBlockSizeJ,
    kMaxK,
    false,  // kKeysQueriesAlignedToBlockSize
    true    // kEnableSplitKeys
    >;

using Params = typename Attention::Params;
using SharedStorage = typename Attention::SharedStorage;

} // namespace

extern "C" {

__device__ __constant__ int cutlass_mha_bwd_shared_mem_bytes = sizeof(SharedStorage);
__device__ __constant__ int cutlass_mha_bwd_num_warps = Attention::kNumWarpsPerBlock;
__device__ __constant__ int cutlass_mha_bwd_block_size_i = kBlockSizeI;
__device__ __constant__ int cutlass_mha_bwd_block_size_j = kBlockSizeJ;
__device__ __constant__ int cutlass_mha_bwd_head_dim = kHeadDim;
__device__ __constant__ int cutlass_mha_bwd_gradq_tile_elements = Attention::MatmulGradQ::AccumTileGmem::kElementsStored;
__device__ __constant__ int cutlass_mha_bwd_gradq_temp_bytes = sizeof(typename Attention::GradQTempStorage);

extern "C" __global__ void CUTLASS_MHA_BWD_KERNEL_NAME(
    const Element *q_ptr,
    uint32_t stride_q_z,
    uint32_t stride_q_h,
    uint32_t stride_q_t,
    const Element *k_ptr,
    uint32_t stride_k_z,
    uint32_t stride_k_h,
    uint32_t stride_k_t,
    const Element *v_ptr,
    uint32_t stride_v_z,
    uint32_t stride_v_h,
    uint32_t stride_v_t,
    const Element *o_ptr,
    uint32_t stride_o_z,
    uint32_t stride_o_h,
    uint32_t stride_o_t,
    float softmax_scale,
    const Element *do_ptr,
    uint32_t stride_do_z,
    uint32_t stride_do_h,
    uint32_t stride_do_t,
    Element *dq_ptr,
    Element *dk_ptr,
    Element *dv_ptr,
    float *lse_ptr,
    float *delta_ptr,
    uint32_t batch_size,
    uint32_t num_heads,
    uint32_t n_ctx,
    void *workspace_ptr) {
  if (batch_size == 0 || num_heads == 0 || n_ctx == 0) {
    return;
  }

  const int32_t stride_t = static_cast<int32_t>(stride_q_t);
  const int32_t stride_h = static_cast<int32_t>(stride_q_h);
  const int64_t stride_b = static_cast<int64_t>(stride_q_z);
  const int32_t stride_do_t_i = static_cast<int32_t>(stride_do_t);
  const int64_t stride_do_b = static_cast<int64_t>(stride_do_z);
  const int64_t stride_do_h_i64 = static_cast<int64_t>(stride_do_h);
  const int64_t stride_o_b = static_cast<int64_t>(stride_o_z);
  const int64_t stride_o_h_i64 = static_cast<int64_t>(stride_o_h);
  (void)stride_o_t;

  Params p{};
  p.query_ptr = const_cast<Element *>(q_ptr);
  p.key_ptr = const_cast<Element *>(k_ptr);
  p.value_ptr = const_cast<Element *>(v_ptr);
  p.bias_ptr = nullptr;
  p.logsumexp_ptr = lse_ptr;
  p.output_ptr = const_cast<Element *>(o_ptr);
  p.grad_output_ptr = const_cast<Element *>(do_ptr);
  p.delta_ptr = reinterpret_cast<decltype(p.delta_ptr)>(delta_ptr);
  p.cu_seqlens_q_ptr = nullptr;
  p.cu_seqlens_k_ptr = nullptr;

  p.grad_query_ptr = dq_ptr;
  p.grad_key_ptr = dk_ptr;
  p.grad_value_ptr = dv_ptr;
  p.grad_bias_ptr = nullptr;

  p.workspace = reinterpret_cast<decltype(p.workspace)>(workspace_ptr);

  p.scale = softmax_scale;
  p.head_dim = kHeadDim;
  p.head_dim_value = kHeadDim;
  p.num_queries = static_cast<int32_t>(n_ctx);
  p.num_keys = static_cast<int32_t>(n_ctx);
  p.num_heads = static_cast<int32_t>(num_heads);
  p.custom_mask_type = Attention::NoCustomMask;
  p.num_batches = static_cast<int32_t>(batch_size);
  p.num_splits_key = static_cast<int16_t>(gridDim.x);

  p.q_strideM = stride_t;
  p.k_strideM = static_cast<int32_t>(stride_k_t);
  p.v_strideM = static_cast<int32_t>(stride_v_t);
  p.bias_strideM = 0;
  p.gO_strideM = stride_do_t_i;
  p.gB_strideM = 0;
  const int32_t expected_gqkv_stride = static_cast<int32_t>(num_heads) * kHeadDim;
  const int32_t gqkv_multiplier = expected_gqkv_stride > 0 ? stride_t / expected_gqkv_stride : 1;
  p.gQKV_strideM_multiplier = static_cast<int8_t>(gqkv_multiplier);

  p.q_strideH = stride_h;
  p.k_strideH = static_cast<int32_t>(stride_k_h);
  p.v_strideH = static_cast<int32_t>(stride_v_h);
  p.bias_strideH = 0;
  p.o_strideB = stride_o_b;
  p.o_strideH = stride_o_h_i64;
  p.q_strideB = stride_b;
  p.k_strideB = static_cast<int64_t>(stride_k_z);
  p.v_strideB = static_cast<int64_t>(stride_v_z);
  p.bias_strideB = 0;
  p.lse_strideB = static_cast<int64_t>(num_heads) * n_ctx;
  p.lse_strideH = static_cast<int64_t>(n_ctx);
  p.delta_strideB = static_cast<int64_t>(num_heads) * n_ctx;
  p.delta_strideH = static_cast<int64_t>(n_ctx);

  p.gO_strideB = stride_do_b;
  p.gQ_strideB = stride_b;
  p.gK_strideB = static_cast<int64_t>(stride_k_z);
  p.gV_strideB = static_cast<int64_t>(stride_v_z);
  p.gB_strideB = 0;
  p.gO_strideH = stride_do_h_i64;
  p.gQ_strideH = static_cast<int64_t>(stride_h);
  p.gK_strideH = static_cast<int64_t>(stride_k_h);
  p.gV_strideH = static_cast<int64_t>(stride_v_h);
  p.gB_strideH = 0;

  if (!p.advance_to_block()) {
    return;
  }

  Attention::attention_kernel(p);
}

} // extern "C"
