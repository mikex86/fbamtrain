#include <cuda_runtime.h>

#include <cstdint>
#include <cmath>
#include <type_traits>
#include <new>

#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/thread/linear_combination_bias_elementwise.h>
#include <cutlass/epilogue/thread/linear_combination_generic.h>
#include <cutlass/epilogue/threadblock/default_epilogue_tensor_op.h>
#include <cutlass/gemm/device/default_gemm_configuration.h>
#include <cutlass/gemm/device/gemm_universal_with_broadcast.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/kernel/default_gemm_with_broadcast.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
#include <cutlass/transform/threadblock/regular_tile_access_iterator_tensor_op.h>
#include <cutlass/transform/threadblock/regular_tile_iterator_tensor_op.h>

namespace cutlass {
namespace gemm {
namespace device {

template <typename ElementA, typename ElementB, typename ElementC, typename ElementAccumulator>
struct DefaultGemmConfiguration<arch::OpClassTensorOp, arch::Sm89, ElementA, ElementB, ElementC, ElementAccumulator> {
  static int const kAlignmentA = 128 / sizeof_bits<ElementA>::value;
  static int const kAlignmentB = 128 / sizeof_bits<ElementA>::value;

  using ThreadblockShape = GemmShape<128, 256, 64>;
  using WarpShape = GemmShape<64, 64, 64>;
  using InstructionShape = GemmShape<16, 8, 16>;
  static int const kStages = 3;

  using EpilogueOutputOp = epilogue::thread::LinearCombination<
      ElementC,
      128 / sizeof_bits<ElementC>::value,
      ElementAccumulator,
      ElementAccumulator>;

  using Operator = arch::OpMultiplyAdd;
};

template <typename ElementA, typename ElementB, typename ElementC, typename ElementAccumulator>
struct DefaultGemmConfiguration<arch::OpClassTensorOp, arch::Sm90, ElementA, ElementB, ElementC, ElementAccumulator> {
  static int const kAlignmentA = 128 / sizeof_bits<ElementA>::value;
  static int const kAlignmentB = 128 / sizeof_bits<ElementA>::value;

  using ThreadblockShape = GemmShape<128, 256, 64>;
  using WarpShape = GemmShape<64, 64, 64>;
  using InstructionShape = GemmShape<16, 8, 16>;
  static int const kStages = 3;

  using EpilogueOutputOp = epilogue::thread::LinearCombination<
      ElementC,
      128 / sizeof_bits<ElementC>::value,
      ElementAccumulator,
      ElementAccumulator>;

  using Operator = arch::OpMultiplyAdd;
};

}  // namespace device
}  // namespace gemm
}  // namespace cutlass

#ifndef CUTLASS_GEMM_ENABLE_CONFIG_CHECK
#error "CUTLASS_GEMM_ENABLE_CONFIG_CHECK not specified!"
#endif

#ifndef CUTLASS_GEMM_KERNEL_NAME
#error "CUTLASS_GEMM_KERNEL_NAME not specified!"
#endif

#ifndef CUTLASS_GEMM_OPERATOR_CLASS
#error "CUTLASS_GEMM_OPERATOR_CLASS not specified!"
#endif

#ifndef CUTLASS_GEMM_ALIGNMENT_A
#error "CUTLASS_GEMM_ALIGNMENT_A not specified!"
#endif

#ifndef CUTLASS_GEMM_ALIGNMENT_B
#error "CUTLASS_GEMM_ALIGNMENT_B not specified!"
#endif

#ifndef CUTLASS_GEMM_THREADBLOCK_M
#error "CUTLASS_GEMM_THREADBLOCK_M not specified!"
#endif

#ifndef CUTLASS_GEMM_THREADBLOCK_N
#error "CUTLASS_GEMM_THREADBLOCK_N not specified!"
#endif

#ifndef CUTLASS_GEMM_THREADBLOCK_K
#error "CUTLASS_GEMM_THREADBLOCK_K not specified!"
#endif

#ifndef CUTLASS_GEMM_WARP_M
#error "CUTLASS_GEMM_WARP_M not specified!"
#endif

#ifndef CUTLASS_GEMM_WARP_N
#error "CUTLASS_GEMM_WARP_N not specified!"
#endif

#ifndef CUTLASS_GEMM_WARP_K
#error "CUTLASS_GEMM_WARP_K not specified!"
#endif

#ifndef CUTLASS_GEMM_INSTRUCTION_M
#error "CUTLASS_GEMM_INSTRUCTION_M not specified!"
#endif

#ifndef CUTLASS_GEMM_INSTRUCTION_N
#error "CUTLASS_GEMM_INSTRUCTION_N not specified!"
#endif

#ifndef CUTLASS_GEMM_INSTRUCTION_K
#error "CUTLASS_GEMM_INSTRUCTION_K not specified!"
#endif

#ifndef CUTLASS_GEMM_NUM_STAGES
#error "CUTLASS_GEMM_NUM_STAGES not specified!"
#endif

#ifndef CUTLASS_GEMM_ACTIVATION
#error "CUTLASS_GEMM_ACTIVATION not specified!"
#endif

#ifndef CUTLASS_GEMM_BIAS_OP
#error "CUTLASS_GEMM_BIAS_OP not specified!"
#endif

#ifndef CUTLASS_GEMM_SPLIT_K_SLICES
#error "CUTLASS_GEMM_SPLIT_K_SLICES not specified!"
#endif

#ifndef CUTLASS_GEMM_FP16
#error "CUTLASS_GEMM_FP16 not specified!"
#endif

#ifndef CUTLASS_GEMM_FP16_ACCUM
#error "CUTLASS_GEMM_FP16_ACCUM not specified!"
#endif

#ifndef CUTLASS_GEMM_FP32_OUTPUT
#error "CUTLASS_GEMM_FP32_OUTPUT not specified!"
#endif

#ifndef CUTLASS_GEMM_WITH_BIAS
#error "CUTLASS_GEMM_WITH_BIAS not specified!"
#endif

#ifndef CUTLASS_GEMM_TRANSPOSE_A
#error "CUTLASS_GEMM_TRANSPOSE_A not specified!"
#endif

#ifndef CUTLASS_GEMM_TRANSPOSE_B
#error "CUTLASS_GEMM_TRANSPOSE_B not specified!"
#endif

#ifndef CUTLASS_GEMM_WRITE_OUT_PREACT
#error "CUTLASS_GEMM_WRITE_OUT_PREACT not specified!"
#endif

#ifndef CUTLASS_GEMM_ELEMENTS_PER_THREAD
#error "CUTLASS_GEMM_ELEMENTS_PER_THREAD not specified!"
#endif

namespace {

using ElementInput =
#if CUTLASS_GEMM_FP16
    cutlass::half_t;
#else
    cutlass::bfloat16_t;
#endif

using ElementOutput =
#if CUTLASS_GEMM_FP32_OUTPUT
    float;
#else
    ElementInput;
#endif

#if CUTLASS_GEMM_FP16_ACCUM
using ElementAccumulator = cutlass::half_t;
using ElementCompute = cutlass::half_t;
#else
using ElementAccumulator = float;
using ElementCompute = float;
#endif

using LayoutInputA =
    std::conditional_t<CUTLASS_GEMM_TRANSPOSE_A, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>;
using LayoutInputB =
    std::conditional_t<CUTLASS_GEMM_TRANSPOSE_B, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>;
using LayoutOutput = cutlass::layout::RowMajor;

using OperatorClass = CUTLASS_GEMM_OPERATOR_CLASS;
#ifndef CUTLASS_GEMM_SM_ARCH
#error "CUTLASS_GEMM_SM_ARCH not specified!"
#endif

using RequestedSmArch = CUTLASS_GEMM_SM_ARCH;
// CUTLASS GemmUniversalWithBroadcast (this kernel path) does not provide a
// direct Sm100 instantiation yet. Compile sm_100 codegen while instantiating
// the template stack with Sm90, which matches CUTLASS coverage here.
using SmArch = std::conditional_t<std::is_same_v<RequestedSmArch, cutlass::arch::Sm100>,
                                  cutlass::arch::Sm90,
                                  RequestedSmArch>;

using ThreadblockShape = cutlass::gemm::GemmShape<
    CUTLASS_GEMM_THREADBLOCK_M,
    CUTLASS_GEMM_THREADBLOCK_N,
    CUTLASS_GEMM_THREADBLOCK_K>;
using WarpShape = cutlass::gemm::GemmShape<
    CUTLASS_GEMM_WARP_M,
    CUTLASS_GEMM_WARP_N,
    CUTLASS_GEMM_WARP_K>;
using InstructionShape = cutlass::gemm::GemmShape<
    CUTLASS_GEMM_INSTRUCTION_M,
    CUTLASS_GEMM_INSTRUCTION_N,
    CUTLASS_GEMM_INSTRUCTION_K>;

constexpr int kElementsPerAccess = CUTLASS_GEMM_ELEMENTS_PER_THREAD;

template <typename T>
using ActivationFunctorTemplate = std::conditional_t<CUTLASS_GEMM_ACTIVATION == 2,
                                                     cutlass::epilogue::thread::GELU<T>,
                                                     cutlass::epilogue::thread::Identity<T>>;

using ActivationFunctor = ActivationFunctorTemplate<ElementCompute>;

template <typename ElementC_, typename ElementAccumulator_, typename ElementCompute_, typename ElementZ_,
          typename ElementT_, int ElementsPerAccess, bool StoreT_ = false, typename ElementVector_ = ElementC_>
class LinearCombinationGeluBwd {
public:
  using ElementOutput = ElementC_;
  using ElementD = ElementOutput;
  using ElementC = ElementC_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;
  using ElementScalar = ElementCompute;
  using ElementZ = ElementZ_;
  using ElementT = ElementT_;
  using ElementVector = ElementVector_;
  static int const kElementsPerAccess = ElementsPerAccess;
  static int const kCount = kElementsPerAccess;
  static bool const IsEltActSupported = false;
  static bool const kIsSingleSource = false;

  using FragmentAccumulator = cutlass::Array<ElementAccumulator, kElementsPerAccess>;
  using FragmentCompute = cutlass::Array<ElementCompute, kElementsPerAccess>;
  using FragmentC = cutlass::Array<ElementC, kElementsPerAccess>;
  using FragmentZ = cutlass::Array<ElementZ, kElementsPerAccess>;
  using FragmentT = cutlass::Array<ElementT, kElementsPerAccess>;
  using FragmentBias = cutlass::Array<ElementVector, kElementsPerAccess>;

  static bool const kStoreZ = true;
  static bool const kStoreT = StoreT_;
  static bool const kIsHeavy = false;

  struct Params {
    ElementCompute alpha;
    ElementCompute beta;
    ElementCompute const *alpha_ptr;
    ElementCompute const *beta_ptr;

    CUTLASS_HOST_DEVICE
    Params(): alpha(ElementCompute(1)), beta(ElementCompute(0)), alpha_ptr(nullptr), beta_ptr(nullptr) {}

    CUTLASS_HOST_DEVICE
    Params(ElementCompute alpha_, ElementCompute beta_): alpha(alpha_), beta(beta_), alpha_ptr(nullptr), beta_ptr(nullptr) {}

    CUTLASS_HOST_DEVICE
    Params(ElementCompute const *alpha_ptr_, ElementCompute const *beta_ptr_)
        : alpha(0), beta(0), alpha_ptr(alpha_ptr_), beta_ptr(beta_ptr_) {}
  };

private:
  ElementCompute alpha_;
  ElementCompute beta_;
  bool skip_elementwise_;

  CUTLASS_HOST_DEVICE
  ElementCompute gelu_grad(ElementCompute value) const {
    const float x = static_cast<float>(value);
    const float erf_term = erff(x * 0.7071067811865475f);
    const float exp_term = expf(-0.5f * x * x);
    const float grad = 0.5f * (1.0f + erf_term) + 0.5f * x * 0.7978845608028654f * exp_term;
    return static_cast<ElementCompute>(grad);
  }

public:
  CUTLASS_HOST_DEVICE
  LinearCombinationGeluBwd(Params const &params) {
    alpha_ = (params.alpha_ptr ? *params.alpha_ptr : params.alpha);
    beta_ = (params.beta_ptr ? *params.beta_ptr : params.beta);
    skip_elementwise_ = false;
  }

  CUTLASS_HOST_DEVICE
  bool is_source_needed() const {
    // Always load sources so we can access pre-activation from C2.
    return true;
  }

  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {
    if (k_partition) {
      beta_ = ElementCompute(1);
    }
    if (k_partition != k_partition_count - 1) {
      skip_elementwise_ = true;
    }
  }

  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentZ &frag_Z,
    FragmentT &frag_T,
    FragmentAccumulator const &AB,
    FragmentC const &frag_C1,
    FragmentC const &frag_C2,
    FragmentCompute const & /*frag_Broadcast*/) const {
    FragmentCompute tmp_Accum = cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kElementsPerAccess>()(AB);
    FragmentCompute tmp_C1 = cutlass::NumericArrayConverter<ElementCompute, ElementC, kElementsPerAccess>()(frag_C1);
    FragmentCompute tmp_C2 = cutlass::NumericArrayConverter<ElementCompute, ElementC, kElementsPerAccess>()(frag_C2);
    FragmentCompute result_Z;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElementsPerAccess; ++i) {
      ElementCompute accum = alpha_ * tmp_Accum[i];
      ElementCompute grad = gelu_grad(tmp_C2[i]);
      ElementCompute z = accum * grad + beta_ * tmp_C1[i];
      result_Z[i] = skip_elementwise_ ? (accum + beta_ * tmp_C1[i]) : z;
    }

    cutlass::NumericArrayConverter<ElementZ, ElementCompute, kElementsPerAccess> convert_z;
    frag_Z = convert_z(result_Z);

    if constexpr (kStoreT) {
      cutlass::NumericArrayConverter<ElementT, ElementCompute, kElementsPerAccess> convert_t;
      frag_T = convert_t(result_Z);
    }
  }

  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentZ &frag_Z,
    FragmentT &frag_T,
    FragmentAccumulator const &AB,
    FragmentCompute const & /*frag_Broadcast*/) const {
    FragmentCompute tmp_Accum = cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kElementsPerAccess>()(AB);
    FragmentCompute result_Z;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElementsPerAccess; ++i) {
      ElementCompute accum = alpha_ * tmp_Accum[i];
      result_Z[i] = accum;
    }

    cutlass::NumericArrayConverter<ElementZ, ElementCompute, kElementsPerAccess> convert_z;
    frag_Z = convert_z(result_Z);

    if constexpr (kStoreT) {
      cutlass::NumericArrayConverter<ElementT, ElementCompute, kElementsPerAccess> convert_t;
      frag_T = convert_t(result_Z);
    }
  }
};

template <typename ElementC_, typename ElementAccumulator_, typename ElementCompute_, typename ElementZ_,
          typename ElementT_, int ElementsPerAccess, typename ElementwiseOp_ = cutlass::epilogue::thread::Identity<ElementCompute_>,
          typename BinaryOp_ = cutlass::plus<ElementCompute_>, bool StoreT_ = false, typename ElementVector_ = ElementC_>
class LinearCombinationBiasElementwisePostActAccum {
public:
  using ElementOutput = ElementC_;
  using ElementD = ElementOutput;
  using ElementC = ElementC_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;
  using ElementScalar = ElementCompute;
  using ElementZ = ElementZ_;
  using ElementT = ElementT_;
  using ElementVector = ElementVector_;
  static int const kElementsPerAccess = ElementsPerAccess;
  static int const kCount = kElementsPerAccess;

  static bool const IsEltActSupported = true;

  using ElementwiseOp = ElementwiseOp_;
  using BinaryOp = BinaryOp_;
  using ElementwiseOpDispatcher = cutlass::epilogue::thread::detail::ElementwiseOpDispatcher<ElementwiseOp>;
  using ElementwiseArguments = typename ElementwiseOpDispatcher::Arguments;

  static bool const kIsSingleSource = true;

  using FragmentAccumulator = cutlass::Array<ElementAccumulator, kElementsPerAccess>;
  using FragmentCompute = cutlass::Array<ElementCompute, kElementsPerAccess>;
  using FragmentC = cutlass::Array<ElementC, kElementsPerAccess>;
  using FragmentZ = cutlass::Array<ElementZ, kElementsPerAccess>;
  using FragmentT = cutlass::Array<ElementT, kElementsPerAccess>;
  using FragmentBias = cutlass::Array<ElementVector, kElementsPerAccess>;

  static bool const kIsHeavy = cutlass::epilogue::thread::kIsHeavy_member_or_false<ElementwiseOp>::value;
  static bool const kStoreZ = true;
  static bool const kStoreT = StoreT_;

  struct Params {
    ElementCompute alpha;
    ElementCompute beta;
    ElementCompute const *alpha_ptr;
    ElementCompute const *beta_ptr;
    ElementwiseArguments elementwise;

    CUTLASS_HOST_DEVICE
    Params(): alpha(ElementCompute(1)), beta(ElementCompute(0)), alpha_ptr(nullptr), beta_ptr(nullptr) {}

    CUTLASS_HOST_DEVICE
    Params(ElementCompute alpha_, ElementCompute beta_, ElementwiseArguments elementwise_ = ElementwiseArguments{})
        : alpha(alpha_), beta(beta_), alpha_ptr(nullptr), beta_ptr(nullptr), elementwise(elementwise_) {}

    CUTLASS_HOST_DEVICE
    Params(ElementCompute const *alpha_ptr_, ElementCompute const *beta_ptr_,
           ElementwiseArguments elementwise_ = ElementwiseArguments{})
        : alpha(0), beta(0), alpha_ptr(alpha_ptr_), beta_ptr(beta_ptr_), elementwise(elementwise_) {}
  };

private:
  ElementCompute alpha_;
  ElementCompute beta_;
  ElementwiseArguments const &elementwise_;
  bool skip_elementwise_;

public:
  CUTLASS_HOST_DEVICE
  LinearCombinationBiasElementwisePostActAccum(Params const &params): elementwise_(params.elementwise) {
    alpha_ = (params.alpha_ptr ? *params.alpha_ptr : params.alpha);
    beta_ = (params.beta_ptr ? *params.beta_ptr : params.beta);
    skip_elementwise_ = false;
  }

  CUTLASS_HOST_DEVICE
  bool is_source_needed() const {
    return beta_ != ElementCompute(0);
  }

  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {
    if (k_partition) {
      beta_ = ElementCompute(1);
    }
    if (k_partition != k_partition_count - 1) {
      skip_elementwise_ = true;
    }
  }

  template <typename ElementwiseArgs>
  CUTLASS_HOST_DEVICE
  void operator()(
      FragmentZ &frag_Z,
      FragmentT &frag_T,
      FragmentAccumulator const &AB,
      FragmentC const &frag_C,
      FragmentCompute const &V,
      ElementwiseArgs const &elementwise_args) const {
    ElementwiseOp elementwise_op;
    BinaryOp binary_op;

    FragmentCompute tmp_Accum = cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kElementsPerAccess>()(AB);
    FragmentCompute tmp_C = cutlass::NumericArrayConverter<ElementCompute, ElementC, kElementsPerAccess>()(frag_C);
    FragmentCompute result_Z;
    FragmentCompute result_T;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElementsPerAccess; ++i) {
      ElementCompute pre = binary_op(alpha_ * tmp_Accum[i], V[i]);
      ElementCompute act = skip_elementwise_ ? pre : elementwise_op(pre, elementwise_args);
      ElementCompute z = act + beta_ * tmp_C[i];
      result_T[i] = pre;
      result_Z[i] = z;
    }

    cutlass::NumericArrayConverter<ElementZ, ElementCompute, kElementsPerAccess> convert_z;
    frag_Z = convert_z(result_Z);

    if constexpr (kStoreT) {
      cutlass::NumericArrayConverter<ElementT, ElementCompute, kElementsPerAccess> convert_t;
      frag_T = convert_t(result_T);
    }
  }

  template <typename ElementwiseArgs>
  CUTLASS_HOST_DEVICE
  void operator()(
      FragmentZ &frag_Z,
      FragmentT &frag_T,
      FragmentAccumulator const &AB,
      FragmentCompute const &V,
      ElementwiseArgs const &elementwise_args) const {
    ElementwiseOp elementwise_op;
    BinaryOp binary_op;

    FragmentCompute tmp_Accum = cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kElementsPerAccess>()(AB);
    FragmentCompute result_Z;
    FragmentCompute result_T;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElementsPerAccess; ++i) {
      ElementCompute pre = binary_op(alpha_ * tmp_Accum[i], V[i]);
      ElementCompute act = skip_elementwise_ ? pre : elementwise_op(pre, elementwise_args);
      result_T[i] = pre;
      result_Z[i] = act;
    }

    cutlass::NumericArrayConverter<ElementZ, ElementCompute, kElementsPerAccess> convert_z;
    frag_Z = convert_z(result_Z);

    if constexpr (kStoreT) {
      cutlass::NumericArrayConverter<ElementT, ElementCompute, kElementsPerAccess> convert_t;
      frag_T = convert_t(result_T);
    }
  }

  CUTLASS_HOST_DEVICE
  void operator()(
      FragmentZ &frag_Z,
      FragmentT &frag_T,
      FragmentAccumulator const &AB,
      FragmentC const &frag_C,
      FragmentCompute const &V) const {
    ElementwiseOp elementwise_op;
    BinaryOp binary_op;

    FragmentCompute tmp_Accum = cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kElementsPerAccess>()(AB);
    FragmentCompute tmp_C = cutlass::NumericArrayConverter<ElementCompute, ElementC, kElementsPerAccess>()(frag_C);
    FragmentCompute result_Z;
    FragmentCompute result_T;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElementsPerAccess; ++i) {
      ElementCompute pre = binary_op(alpha_ * tmp_Accum[i], V[i]);
      ElementCompute act = skip_elementwise_ ? pre : elementwise_op(pre);
      ElementCompute z = act + beta_ * tmp_C[i];
      result_T[i] = pre;
      result_Z[i] = z;
    }

    cutlass::NumericArrayConverter<ElementZ, ElementCompute, kElementsPerAccess> convert_z;
    frag_Z = convert_z(result_Z);

    if constexpr (kStoreT) {
      cutlass::NumericArrayConverter<ElementT, ElementCompute, kElementsPerAccess> convert_t;
      frag_T = convert_t(result_T);
    }
  }

  CUTLASS_HOST_DEVICE
  void operator()(
      FragmentZ &frag_Z,
      FragmentT &frag_T,
      FragmentAccumulator const &AB,
      FragmentCompute const &V) const {
    ElementwiseOp elementwise_op;
    BinaryOp binary_op;

    FragmentCompute tmp_Accum = cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kElementsPerAccess>()(AB);
    FragmentCompute result_Z;
    FragmentCompute result_T;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElementsPerAccess; ++i) {
      ElementCompute pre = binary_op(alpha_ * tmp_Accum[i], V[i]);
      ElementCompute act = skip_elementwise_ ? pre : elementwise_op(pre);
      result_T[i] = pre;
      result_Z[i] = act;
    }

    cutlass::NumericArrayConverter<ElementZ, ElementCompute, kElementsPerAccess> convert_z;
    frag_Z = convert_z(result_Z);

    if constexpr (kStoreT) {
      cutlass::NumericArrayConverter<ElementT, ElementCompute, kElementsPerAccess> convert_t;
      frag_T = convert_t(result_T);
    }
  }
};

#if CUTLASS_GEMM_WITH_BIAS
#if CUTLASS_GEMM_BIAS_OP == 1
using EpilogueOp = LinearCombinationGeluBwd<
    ElementOutput,
    ElementAccumulator,
    ElementCompute,
    ElementOutput,
    ElementOutput,
    kElementsPerAccess,
    false,
    ElementOutput>;
#elif CUTLASS_GEMM_ACTIVATION == 2
using EpilogueOp = LinearCombinationBiasElementwisePostActAccum<
    ElementOutput,
    ElementAccumulator,
    ElementCompute,
    ElementOutput,
    ElementOutput,
    kElementsPerAccess,
    ActivationFunctor,
    cutlass::plus<ElementCompute>,
    CUTLASS_GEMM_WRITE_OUT_PREACT,
    ElementOutput>;
#else
using EpilogueOp = cutlass::epilogue::thread::LinearCombinationBiasElementwise<
    ElementOutput,
    ElementAccumulator,
    ElementCompute,
    ElementOutput,
    ElementOutput,
    kElementsPerAccess,
    ActivationFunctor,
    cutlass::plus<ElementCompute>,
    CUTLASS_GEMM_WRITE_OUT_PREACT,
    ElementOutput>;
#endif
#else
using EpilogueOp = typename std::conditional_t<
    (CUTLASS_GEMM_ACTIVATION == 0),
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput,
        kElementsPerAccess,
        ElementAccumulator,
        ElementCompute>,
        cutlass::epilogue::thread::LinearCombinationGeneric<
        ActivationFunctorTemplate,
        ElementOutput,
        kElementsPerAccess,
        ElementAccumulator,
        ElementCompute>>;
#endif

#if CUTLASS_GEMM_WITH_BIAS
using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmWithBroadcast<
    ElementInput,
    LayoutInputA,
    cutlass::ComplexTransform::kNone,
    CUTLASS_GEMM_ALIGNMENT_A,
    ElementInput,
    LayoutInputB,
    cutlass::ComplexTransform::kNone,
    CUTLASS_GEMM_ALIGNMENT_B,
    ElementOutput,
    LayoutOutput,
    ElementAccumulator,
    OperatorClass,
    SmArch,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<CUTLASS_GEMM_SWIZZLE_SIZE>,
    CUTLASS_GEMM_NUM_STAGES,
    typename cutlass::gemm::device::DefaultGemmConfiguration<
        OperatorClass,
        SmArch,
        ElementInput,
        ElementInput,
        ElementOutput,
        ElementAccumulator>::Operator>::GemmKernel;
#else
using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmUniversal<
    ElementInput,
    LayoutInputA,
    cutlass::ComplexTransform::kNone,
    CUTLASS_GEMM_ALIGNMENT_A,
    ElementInput,
    LayoutInputB,
    cutlass::ComplexTransform::kNone,
    CUTLASS_GEMM_ALIGNMENT_B,
    ElementOutput,
    LayoutOutput,
    ElementAccumulator,
    OperatorClass,
    SmArch,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<CUTLASS_GEMM_SWIZZLE_SIZE>,
    CUTLASS_GEMM_NUM_STAGES,
    typename cutlass::gemm::device::DefaultGemmConfiguration<
        OperatorClass,
        SmArch,
        ElementInput,
        ElementInput,
        ElementOutput,
        ElementAccumulator>::Operator>::GemmKernel;
#endif

using SharedStorage = typename GemmKernel::SharedStorage;
using Params = typename GemmKernel::Params;
using Arguments = typename GemmKernel::Arguments;

constexpr int kNumWarps = GemmKernel::kThreadCount / 32;

struct CutlassGemmConfig {
  int m;
  int n;
  int k;
  int lda;
  int ldb;
  int ldd;
  int bias_stride;
  int split_k_slices;
};

#if CUTLASS_GEMM_ENABLE_CONFIG_CHECK
CUTLASS_DEVICE bool config_matches_expected(const CutlassGemmConfig &cfg) {
  if (cfg.split_k_slices > 0 && CUTLASS_GEMM_SPLIT_K_SLICES > 0 &&
      cfg.split_k_slices != CUTLASS_GEMM_SPLIT_K_SLICES) {
    return false;
  }
  return true;
}
#endif

template <typename T>
__host__ __device__ T clamp_positive(T value) {
  return value > T(0) ? value : T(0);
}

}  // namespace

extern "C" {

__device__ __constant__ int cutlass_gemm_shared_mem_bytes = sizeof(SharedStorage);
__device__ __constant__ int cutlass_gemm_num_warps = kNumWarps;

extern "C" __global__ void CUTLASS_GEMM_KERNEL_NAME(
    CutlassGemmConfig cfg,
    ElementInput const *ptr_a,
    ElementInput const *ptr_b,
    ElementOutput const *ptr_bias,
    ElementOutput *ptr_d,
#if CUTLASS_GEMM_WRITE_OUT_PREACT
    ElementOutput *ptr_preact,
#endif
    ElementCompute alpha,
    ElementCompute beta,
    void *workspace) {
  (void)workspace;
  extern __shared__ int shared_storage_raw[];
  auto *shared_storage = reinterpret_cast<SharedStorage *>(shared_storage_raw);

#if CUTLASS_GEMM_ENABLE_CONFIG_CHECK
  if (!config_matches_expected(cfg)) {
    return;
  }
#endif

  const int m = clamp_positive(cfg.m);
  const int n = clamp_positive(cfg.n);
  const int k = clamp_positive(cfg.k);

  if (m == 0 || n == 0 || k == 0) {
    return;
  }

  cutlass::gemm::GemmCoord problem_size(m, n, k);
  typename EpilogueOp::Params epilogue_params;
  epilogue_params.alpha = ElementCompute(alpha);
  epilogue_params.beta = ElementCompute(beta);
  epilogue_params.alpha_ptr = nullptr;
  epilogue_params.beta_ptr = nullptr;

  const int lda = cfg.lda;
  const int ldb = cfg.ldb;
  const int ldd = cfg.ldd;
  const int ldr = cfg.bias_stride;

  Params params{};
  params.problem_size = problem_size;
  params.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  params.batch_count = 1;
  params.batch_stride_D = 0;
  params.semaphore = nullptr;

  params.params_A = typename GemmKernel::Mma::IteratorA::Params(lda);
  params.params_B = typename GemmKernel::Mma::IteratorB::Params(ldb);
#if CUTLASS_GEMM_BIAS_OP == 1
  params.params_C1 = typename GemmKernel::Epilogue::OutputTileIterator::Params(ldd);
  params.params_C2 = typename GemmKernel::Epilogue::OutputTileIterator::Params(ldr);
#else
  params.params_C = typename GemmKernel::Epilogue::OutputTileIterator::Params(ldd);
#endif
  params.params_D = typename GemmKernel::Epilogue::OutputTileIterator::Params(ldd);
#if CUTLASS_GEMM_WITH_BIAS
  params.params_Tensor = typename GemmKernel::Epilogue::TensorTileIterator::Params(ldd);
#endif

  new (&params.output_op) typename EpilogueOp::Params(epilogue_params);

  params.ptr_A = const_cast<ElementInput *>(ptr_a);
  params.ptr_B = const_cast<ElementInput *>(ptr_b);
#if CUTLASS_GEMM_BIAS_OP == 1
  params.ptr_C1 = const_cast<ElementOutput *>(ptr_d);
  params.ptr_C2 = const_cast<ElementOutput *>(ptr_d);
#else
  params.ptr_C = const_cast<ElementOutput *>(ptr_d);
#endif
  params.ptr_D = static_cast<void *>(ptr_d);
#if CUTLASS_GEMM_WITH_BIAS
  params.ptr_Vector = const_cast<ElementOutput *>(ptr_bias);
  params.ptr_Tensor = nullptr;
#if CUTLASS_GEMM_BIAS_OP == 1
  params.ptr_Vector = nullptr;
  params.ptr_Tensor = nullptr;
  params.ptr_C2 = const_cast<ElementOutput *>(ptr_bias);
#endif
#else
  (void)ptr_bias;
#endif

  params.batch_stride_A = 0;
  params.batch_stride_B = 0;
#if CUTLASS_GEMM_BIAS_OP == 1
  params.batch_stride_C1 = 0;
  params.batch_stride_C2 = 0;
#else
  params.batch_stride_C = 0;
#endif
  params.batch_stride_D = 0;
#if CUTLASS_GEMM_WITH_BIAS
  params.batch_stride_Vector = 0;
  params.batch_stride_Tensor = 0;
#endif

#if CUTLASS_GEMM_WITH_BIAS
  params.ldr = static_cast<typename LayoutOutput::Stride::Index>(ldr);
#endif

#if CUTLASS_GEMM_WRITE_OUT_PREACT
  params.ptr_Tensor = static_cast<ElementOutput *>(ptr_preact);
#endif

  params.grid_tiled_shape =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<CUTLASS_GEMM_SWIZZLE_SIZE>::get_tiled_shape(
          params.problem_size,
          {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
          params.batch_count);
  params.swizzle_log_tile =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<CUTLASS_GEMM_SWIZZLE_SIZE>::get_log_tile(
          params.grid_tiled_shape);

  params.gemm_k_size = problem_size.k();
  params.grid_tiled_shape.k() = 1;

  GemmKernel op;
  op(params, *shared_storage);
}

}  // extern "C"
