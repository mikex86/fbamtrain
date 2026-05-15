#include <cuda_runtime.h>

#include <cstdint>
#include <cutlass/conv/conv2d_problem_size.h>
#include <cutlass/conv/device/implicit_gemm_convolution.h>
#include <cutlass/conv/kernel/default_conv2d_wgrad.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/layout/tensor.h>
#include <cutlass/numeric_types.h>

#ifndef CUTLASS_CONV_ENABLE_CONFIG_CHECK
#error "CUTLASS_CONV_ENABLE_CONFIG_CHECK not specified!"
#endif


#ifndef CUTLASS_CONV_KERNEL_NAME
#error "CUTLASS_CONV_KERNEL_NAME not specified!"
#endif

#ifndef CUTLASS_CONV_OPERATOR_CLASS
#error "CUTLASS_CONV_OPERATOR_CLASS not specified!"
#endif

#ifndef CUTLASS_CONV_ITERATOR_ALGO
#error "CUTLASS_CONV_ITERATOR_ALGO not specified!"
#endif

#ifndef CUTLASS_CONV_STRIDE_SUPPORT
#error "CUTLASS_CONV_STRIDE_SUPPORT not specified!"
#endif

#ifndef CUTLASS_CONV_ALIGNMENT_A
#error "CUTLASS_CONV_ALIGNMENT_A not specified!"
#endif

#ifndef CUTLASS_CONV_ALIGNMENT_B
#error "CUTLASS_CONV_ALIGNMENT_B not specified!"
#endif

#ifndef CUTLASS_CONV_THREADBLOCK_M
#error "CUTLASS_CONV_THREADBLOCK_M not specified!"
#endif

#ifndef CUTLASS_CONV_THREADBLOCK_N
#error "CUTLASS_CONV_THREADBLOCK_N not specified!"
#endif

#ifndef CUTLASS_CONV_THREADBLOCK_K
#error "CUTLASS_CONV_THREADBLOCK_K not specified!"
#endif

#ifndef CUTLASS_CONV_WARP_M
#error "CUTLASS_CONV_WARP_M not specified!"
#endif

#ifndef CUTLASS_CONV_WARP_N
#error "CUTLASS_CONV_WARP_N not specified!"
#endif

#ifndef CUTLASS_CONV_WARP_K
#error "CUTLASS_CONV_WARP_K not specified!"
#endif

#ifndef CUTLASS_CONV_INSTRUCTION_M
#error "CUTLASS_CONV_INSTRUCTION_M not specified!"
#endif

#ifndef CUTLASS_CONV_INSTRUCTION_N
#error "CUTLASS_CONV_INSTRUCTION_N not specified!"
#endif

#ifndef CUTLASS_CONV_INSTRUCTION_K
#error "CUTLASS_CONV_INSTRUCTION_K not specified!"
#endif

#ifndef CUTLASS_CONV_NUM_STAGES
#error "CUTLASS_CONV_NUM_STAGES not specified!"
#endif

#ifndef CUTLASS_CONV_EXPECT_STRIDE_H
#error "CUTLASS_CONV_EXPECT_STRIDE_H not specified!"
#endif

#ifndef CUTLASS_CONV_EXPECT_STRIDE_W
#error "CUTLASS_CONV_EXPECT_STRIDE_W not specified!"
#endif

#ifndef CUTLASS_CONV_EXPECT_PADDING_H
#error "CUTLASS_CONV_EXPECT_PADDING_H not specified!"
#endif

#ifndef CUTLASS_CONV_EXPECT_PADDING_W
#error "CUTLASS_CONV_EXPECT_PADDING_W not specified!"
#endif

#ifndef CUTLASS_CONV_EXPECT_DILATION_H
#error "CUTLASS_CONV_EXPECT_DILATION_H not specified!"
#endif

#ifndef CUTLASS_CONV_EXPECT_DILATION_W
#error "CUTLASS_CONV_EXPECT_DILATION_W not specified!"
#endif

#ifndef CUTLASS_CONV_EXPECT_GROUPS
#error "CUTLASS_CONV_EXPECT_GROUPS not specified!"
#endif

#ifndef CUTLASS_CONV_EXPECT_KERNEL_H
#error "CUTLASS_CONV_EXPECT_KERNEL_H not specified!"
#endif

#ifndef CUTLASS_CONV_EXPECT_KERNEL_W
#error "CUTLASS_CONV_EXPECT_KERNEL_W not specified!"
#endif

#ifndef CUTLASS_CONV_META_IN_CHANNELS
#error "CUTLASS_CONV_META_IN_CHANNELS not specified!"
#endif

#ifndef CUTLASS_CONV_FP16
#error "CUTLASS_CONV_FP16 not specified!"
#endif

#ifndef CUTLASS_CONV_FP16_ACCUM
#error "CUTLASS_CONV_FP16_ACCUM not specified!"
#endif

#ifndef CUTLASS_CONV_SM_ARCH
#error "CUTLASS_CONV_SM_ARCH not specified!"
#endif

namespace {

using ElementInputA =
#if CUTLASS_CONV_FP16
    cutlass::half_t;
#else
    cutlass::bfloat16_t;
#endif
using ElementInputB = ElementInputA;
using ElementOutput = ElementInputA;
#if CUTLASS_CONV_FP16_ACCUM
using ElementAccumulator = cutlass::half_t;
using ElementCompute = cutlass::half_t;
#else
using ElementAccumulator = float;
using ElementCompute = float;
#endif

using LayoutInputA = cutlass::layout::TensorNHWC;
using LayoutInputB = cutlass::layout::TensorNHWC;
using LayoutOutput = cutlass::layout::TensorNHWC;

using MMAOp = CUTLASS_CONV_OPERATOR_CLASS;
using SmArch = CUTLASS_CONV_SM_ARCH;

using ThreadblockShape = cutlass::gemm::GemmShape<
    CUTLASS_CONV_THREADBLOCK_M,
    CUTLASS_CONV_THREADBLOCK_N,
    CUTLASS_CONV_THREADBLOCK_K>;
using WarpShape = cutlass::gemm::GemmShape<
    CUTLASS_CONV_WARP_M,
    CUTLASS_CONV_WARP_N,
    CUTLASS_CONV_WARP_K>;
using InstructionShape = cutlass::gemm::GemmShape<
    CUTLASS_CONV_INSTRUCTION_M,
    CUTLASS_CONV_INSTRUCTION_N,
    CUTLASS_CONV_INSTRUCTION_K>;

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementCompute>;

using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
constexpr int NumStages = CUTLASS_CONV_NUM_STAGES;

using Conv2dWgradKernel = typename cutlass::conv::kernel::DefaultConv2dWgrad<
    ElementInputA,
    LayoutInputA,
    ElementInputB,
    LayoutInputB,
    ElementOutput,
    LayoutOutput,
    ElementAccumulator,
    MMAOp,
    SmArch,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOp,
    SwizzleThreadBlock,
    NumStages,
    cutlass::arch::OpMultiplyAdd,
    CUTLASS_CONV_ITERATOR_ALGO,
    CUTLASS_CONV_STRIDE_SUPPORT,
    CUTLASS_CONV_ALIGNMENT_A,
    CUTLASS_CONV_ALIGNMENT_B>::Kernel;

using ConvOperation = cutlass::conv::device::ImplicitGemmConvolution<Conv2dWgradKernel>;
using Params = typename Conv2dWgradKernel::Params;
using ProblemSize = cutlass::conv::Conv2dProblemSize;
using TensorCoord4D = cutlass::Tensor4DCoord;

struct LaunchConfig {
  int grid_x;
  int grid_y;
  int grid_z;
  int block_x;
  int block_y;
  int block_z;
  int shared_mem_bytes;
  std::uint64_t workspace_bytes;
};

struct Conv2dConfig {
  int N;
  int H;
  int W;
  int C;
  int K;
  int R;
  int S;
  int pad_h;
  int pad_w;
  int stride_h;
  int stride_w;
  int dilation_h;
  int dilation_w;
  int groups;
  int split_k_slices;
};

struct CutlassConv2dMeta {
  int num_warps;
  int shared_mem_bytes;
  int block_pixels;
  int block_oc;
  int block_k;
  int in_channels;
  int kernel_h;
  int kernel_w;
  int stride_h;
  int stride_w;
  int padding_h;
  int padding_w;
  int dilation_h;
  int dilation_w;
  int groups;
};

CUTLASS_DEVICE int compute_output_extent(int input, int kernel, int padding, int stride, int dilation) {
  int effective = dilation * (kernel - 1) + 1;
  return (input + 2 * padding - effective) / stride + 1;
}

#if CUTLASS_CONV_ENABLE_CONFIG_CHECK
CUTLASS_DEVICE bool config_matches_expected(const Conv2dConfig &cfg) {
  if (cfg.stride_h != CUTLASS_CONV_EXPECT_STRIDE_H || cfg.stride_w != CUTLASS_CONV_EXPECT_STRIDE_W) {
    return false;
  }
  if (cfg.pad_h != CUTLASS_CONV_EXPECT_PADDING_H || cfg.pad_w != CUTLASS_CONV_EXPECT_PADDING_W) {
    return false;
  }
  if (cfg.dilation_h != CUTLASS_CONV_EXPECT_DILATION_H || cfg.dilation_w != CUTLASS_CONV_EXPECT_DILATION_W) {
    return false;
  }
  if (cfg.groups != CUTLASS_CONV_EXPECT_GROUPS) {
    return false;
  }
  if (cfg.R != CUTLASS_CONV_EXPECT_KERNEL_H || cfg.S != CUTLASS_CONV_EXPECT_KERNEL_W) {
    return false;
  }
  if (CUTLASS_CONV_META_IN_CHANNELS > 0 && cfg.C != CUTLASS_CONV_META_IN_CHANNELS) {
    return false;
  }
  return true;
}
#endif

CUTLASS_DEVICE ProblemSize make_problem_size(const Conv2dConfig &cfg, int out_h, int out_w) {
  int groups = cfg.groups > 0 ? cfg.groups : 1;
  int split_k = cfg.split_k_slices > 0 ? cfg.split_k_slices : 1;

  return ProblemSize(
      cfg.N,
      cfg.H,
      cfg.W,
      cfg.C,
      cfg.K,
      cfg.R,
      cfg.S,
      out_h,
      out_w,
      cfg.pad_h,
      cfg.pad_w,
      cfg.stride_h,
      cfg.stride_w,
      cfg.dilation_h,
      cfg.dilation_w,
      cutlass::conv::Mode::kCrossCorrelation,
      split_k,
      groups);
}

CUTLASS_DEVICE ConvOperation::Arguments make_arguments(
    const Conv2dConfig &cfg,
    int out_h,
    int out_w,
    ElementInputA const *ptr_a,
    ElementInputB const *ptr_b,
    ElementOutput *ptr_d,
    ElementCompute alpha,
    ElementCompute beta) {
  TensorCoord4D input_extent(cfg.N, cfg.H, cfg.W, cfg.C);
  TensorCoord4D weight_extent(cfg.K, cfg.R, cfg.S, cfg.C / (cfg.groups > 0 ? cfg.groups : 1));
  TensorCoord4D output_extent(cfg.N, out_h, out_w, cfg.K);

  auto problem_size = make_problem_size(cfg, out_h, out_w);

  auto *mutable_a = const_cast<ElementInputA *>(ptr_a);
  auto *mutable_b = const_cast<ElementInputB *>(ptr_b);

  typename Conv2dWgradKernel::TensorRefA ref_A(mutable_a, LayoutInputA::packed(output_extent));
  typename Conv2dWgradKernel::TensorRefB ref_B(mutable_b, LayoutInputB::packed(input_extent));

  typename Conv2dWgradKernel::TensorRefC ref_C(ptr_d, LayoutOutput::packed(weight_extent));
  typename Conv2dWgradKernel::TensorRefC ref_D(ptr_d, LayoutOutput::packed(weight_extent));

  EpilogueOp::Params output_op(alpha, beta);

  return ConvOperation::Arguments(
      problem_size,
      ref_A,
      ref_B,
      ref_C,
      ref_D,
      output_op,
      cutlass::conv::SplitKMode::kSerial);
}

}  // namespace

extern "C" {

struct CutlassConv2dConfig {
  int N;
  int H;
  int W;
  int C;
  int K;
  int R;
  int S;
  int pad_h;
  int pad_w;
  int stride_h;
  int stride_w;
  int dilation_h;
  int dilation_w;
  int groups;
  int split_k_slices;
};

__device__ __constant__ int cutlass_conv2d_shared_mem_bytes = sizeof(typename Conv2dWgradKernel::SharedStorage);
__device__ __constant__ int cutlass_conv2d_num_warps = ConvOperation::kWarpCount;

extern "C" __global__ void CUTLASS_CONV_KERNEL_NAME(
    CutlassConv2dConfig cfg,
    ElementInputA const *ptr_dout,
    ElementInputB const *ptr_input,
    ElementOutput *ptr_dweight,
    ElementCompute alpha,
    ElementCompute beta,
    int *workspace) {
  extern __shared__ int SharedStorageBase[];
  auto *shared_storage = reinterpret_cast<typename Conv2dWgradKernel::SharedStorage *>(SharedStorageBase);

  Conv2dConfig conv_cfg{cfg.N,
                        cfg.H,
                        cfg.W,
                        cfg.C,
                        cfg.K,
                        cfg.R,
                        cfg.S,
                        cfg.pad_h,
                        cfg.pad_w,
                        cfg.stride_h,
                        cfg.stride_w,
                        cfg.dilation_h,
                        cfg.dilation_w,
                        cfg.groups,
                        cfg.split_k_slices};

#if CUTLASS_CONV_ENABLE_CONFIG_CHECK
  if (!config_matches_expected(conv_cfg)) {
    return;
  }
#endif

  int out_h = compute_output_extent(conv_cfg.H, conv_cfg.R, conv_cfg.pad_h, conv_cfg.stride_h, conv_cfg.dilation_h);
  int out_w = compute_output_extent(conv_cfg.W, conv_cfg.S, conv_cfg.pad_w, conv_cfg.stride_w, conv_cfg.dilation_w);

  auto args = make_arguments(
      conv_cfg,
      out_h,
      out_w,
      ptr_dout,
      ptr_input,
      ptr_dweight,
      ElementCompute(alpha),
      ElementCompute(beta));

  Params params(args, workspace);

  Conv2dWgradKernel op;
  op(params, *shared_storage);
}

}  // extern "C"
