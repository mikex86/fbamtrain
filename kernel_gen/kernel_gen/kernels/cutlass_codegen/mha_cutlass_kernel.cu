#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/default_gemm_configuration.h>
#include <cutlass/numeric_types.h>

#include "third_party/cutlass/examples/41_fused_multi_head_attention/kernel_forward.h"

#ifndef CUTLASS_MHA_KERNEL_NAME
#error "CUTLASS_MHA_KERNEL_NAME not specified!"
#endif

#ifndef CUTLASS_MHA_FP16
#error "CUTLASS_MHA_FP16 not specified!"
#endif

#ifndef CUTLASS_MHA_HEAD_DIM
#error "CUTLASS_MHA_HEAD_DIM not specified!"
#endif

#ifndef CUTLASS_MHA_QUERIES_PER_BLOCK
#error "CUTLASS_MHA_QUERIES_PER_BLOCK not specified!"
#endif

#ifndef CUTLASS_MHA_KEYS_PER_BLOCK
#error "CUTLASS_MHA_KEYS_PER_BLOCK not specified!"
#endif

#ifndef CUTLASS_MHA_IS_ALIGNED
#error "CUTLASS_MHA_IS_ALIGNED not specified!"
#endif

#ifndef CUTLASS_MHA_SUPPORTS_DROPOUT
#error "CUTLASS_MHA_SUPPORTS_DROPOUT not specified!"
#endif

#ifndef CUTLASS_MHA_SUPPORTS_BIAS
#error "CUTLASS_MHA_SUPPORTS_BIAS not specified!"
#endif

#ifndef CUTLASS_MHA_WRITE_LSE
#error "CUTLASS_MHA_WRITE_LSE not specified!"
#endif

#ifndef CUTLASS_MHA_SM_ARCH
#error "CUTLASS_MHA_SM_ARCH not specified!"
#endif

#ifndef CUTLASS_MHA_SM_VERSION
#error "CUTLASS_MHA_SM_VERSION not specified!"
#endif

namespace cutlass {
namespace gemm {
namespace device {

// Example-41 forward depends on DefaultGemmConfiguration entries that are
// provided for Sm80 in this CUTLASS path. Forward newer arch tags to Sm80
// so we keep using CUTLASS' own configuration policy without literal tiles.
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

namespace {

#if CUTLASS_MHA_FP16
using Element = cutlass::half_t;
#else
using Element = cutlass::bfloat16_t;
#endif

constexpr bool kIsAligned = CUTLASS_MHA_IS_ALIGNED != 0;
constexpr bool kSupportsDropout = CUTLASS_MHA_SUPPORTS_DROPOUT != 0;
constexpr bool kSupportsBias = CUTLASS_MHA_SUPPORTS_BIAS != 0;

constexpr int kQueriesPerBlock = CUTLASS_MHA_QUERIES_PER_BLOCK;
constexpr int kKeysPerBlock = CUTLASS_MHA_KEYS_PER_BLOCK;
constexpr int kHeadDim = CUTLASS_MHA_HEAD_DIM;
// Example-41 forward does not provide a native Sm100 instantiation path in
// this single-kernel ABI. For sm100 codegen, instantiate the FMHA template
// stack with Sm90 while still emitting sm100-targeted binaries.
using RequestedArch =
    std::conditional_t<(CUTLASS_MHA_SM_VERSION >= 100), cutlass::arch::Sm90, CUTLASS_MHA_SM_ARCH>;

static_assert(
    (CUTLASS_MHA_SM_VERSION >= 100) || (RequestedArch::kMinComputeCapability == CUTLASS_MHA_SM_VERSION),
    "CUTLASS_MHA_SM_VERSION must match CUTLASS_MHA_SM_ARCH for non-sm100 builds.");

using Attention = AttentionKernel<
    Element,
    RequestedArch,
    kIsAligned,
    kQueriesPerBlock,
    kKeysPerBlock,
    kHeadDim,
    kSupportsDropout,
    kSupportsBias>;

using Params = typename Attention::Params;
using SharedStorage = typename Attention::SharedStorage;

}  // namespace

extern "C" {

__device__ __constant__ int cutlass_mha_shared_mem_bytes = sizeof(SharedStorage);
__device__ __constant__ int cutlass_mha_num_warps = Attention::kNumWarpsPerBlock;

extern "C" __global__ void CUTLASS_MHA_KERNEL_NAME(
    float softmax_scale,
    float *scratch_m_ptr,
    uint32_t batch_size,
    uint32_t num_heads,
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
    Element *o_ptr,
    uint32_t stride_o_z,
    uint32_t stride_o_h,
    uint32_t stride_o_t,
    uint32_t n_ctx,
    uint32_t stride_m_z,
    uint32_t stride_m_h,
    uint32_t stride_m_t,
    void *workspace_ptr) {
  (void)stride_m_z;
  (void)stride_m_h;
  (void)stride_m_t;

  if (batch_size == 0 || num_heads == 0 || n_ctx == 0) {
    return;
  }

  Params p{};
  p.query_ptr = const_cast<Element *>(q_ptr);
  p.key_ptr = const_cast<Element *>(k_ptr);
  p.value_ptr = const_cast<Element *>(v_ptr);
  p.attn_bias_ptr = nullptr;
  p.seqstart_q_ptr = nullptr;
  p.seqstart_k_ptr = nullptr;
  p.seqlen_k_ptr = nullptr;
  p.output_ptr = o_ptr;
  p.output_accum_ptr = reinterpret_cast<decltype(p.output_accum_ptr)>(workspace_ptr);
#if CUTLASS_MHA_WRITE_LSE
  p.logsumexp_ptr = scratch_m_ptr;
#else
  (void)scratch_m_ptr;
  p.logsumexp_ptr = nullptr;
#endif
  p.scale = softmax_scale;
  p.num_batches = static_cast<int32_t>(batch_size);
  p.num_heads = static_cast<int32_t>(num_heads);
  p.num_queries = static_cast<int32_t>(n_ctx);
  p.num_keys = static_cast<int32_t>(n_ctx);
  p.head_dim = kHeadDim;
  p.head_dim_value = kHeadDim;
  p.custom_mask_type = Attention::NoCustomMask;
  p.causal_diagonal_offset = 0;
  p.use_dropout = false;
  p.dropout_prob = 0.0f;
  p.dropout_batch_head_rng_offset = 0;

  p.q_strideM = static_cast<int32_t>(stride_q_t);
  p.q_strideH = static_cast<int32_t>(stride_q_h);
  p.q_strideB = static_cast<int64_t>(stride_q_z);

  p.k_strideM = static_cast<int32_t>(stride_k_t);
  p.k_strideH = static_cast<int32_t>(stride_k_h);
  p.k_strideB = static_cast<int64_t>(stride_k_z);

  p.v_strideM = static_cast<int32_t>(stride_v_t);
  p.v_strideH = static_cast<int32_t>(stride_v_h);
  p.v_strideB = static_cast<int64_t>(stride_v_z);

  p.bias_strideM = 0;
  p.bias_strideH = 0;
  p.bias_strideB = 0;
  p.o_strideM = static_cast<int32_t>(stride_o_t);

  if (!p.advance_to_block()) {
    return;
  }

  Attention::attention_kernel(p);
}

}  // extern "C"
