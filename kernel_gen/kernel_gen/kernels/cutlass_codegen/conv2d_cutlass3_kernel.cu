#include <cuda_runtime.h>

#include <cstdint>

#include <cutlass/device_kernel.h>

#include "conv2d_cutlass3_wgrad_traits.cuh"

#ifndef CUTLASS_CONV_KERNEL_NAME
#error "CUTLASS_CONV_KERNEL_NAME not specified!"
#endif

#ifndef CUTLASS_CONV_OP
#error "CUTLASS_CONV_OP not specified!"
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

#ifndef CUTLASS_CONV_THREADBLOCK_M
#error "CUTLASS_CONV_THREADBLOCK_M not specified!"
#endif

#ifndef CUTLASS_CONV_THREADBLOCK_N
#error "CUTLASS_CONV_THREADBLOCK_N not specified!"
#endif

#ifndef CUTLASS_CONV_THREADBLOCK_K
#error "CUTLASS_CONV_THREADBLOCK_K not specified!"
#endif

#ifndef CUTLASS_CONV_CLUSTER_M
#define CUTLASS_CONV_CLUSTER_M 1
#endif

#ifndef CUTLASS_CONV_CLUSTER_N
#define CUTLASS_CONV_CLUSTER_N 1
#endif

#ifndef CUTLASS_CONV_CLUSTER_K
#define CUTLASS_CONV_CLUSTER_K 1
#endif

#ifndef CUTLASS_CONV_ALIGNMENT_A
#define CUTLASS_CONV_ALIGNMENT_A 8
#endif

#ifndef CUTLASS_CONV_ALIGNMENT_B
#define CUTLASS_CONV_ALIGNMENT_B 8
#endif

namespace
{
using Element =
#if CUTLASS_CONV_FP16
    cutlass::half_t;
#else
    cutlass::bfloat16_t;
#endif

#if CUTLASS_CONV_FP16_ACCUM
using ElementAccumulator = cutlass::half_t;
#else
using ElementAccumulator = float;
#endif

using Traits = fbamtrain::kernel_gen::cutlass3_conv2d::Conv2dTraits<
    CUTLASS_CONV_OP, CUTLASS_CONV_SM_ARCH, Element, ElementAccumulator, CUTLASS_CONV_THREADBLOCK_M,
    CUTLASS_CONV_THREADBLOCK_N, CUTLASS_CONV_THREADBLOCK_K, CUTLASS_CONV_CLUSTER_M, CUTLASS_CONV_CLUSTER_N,
    CUTLASS_CONV_CLUSTER_K, CUTLASS_CONV_ALIGNMENT_A, CUTLASS_CONV_ALIGNMENT_B>;

using ConvKernel = typename Traits::ConvKernel;
using Params = typename Traits::Params;

struct alignas(alignof(Params)) Conv2dKernelParams
{
    unsigned char storage[sizeof(Params)];
};

static_assert(sizeof(Conv2dKernelParams) == sizeof(Params));
static_assert(alignof(Conv2dKernelParams) >= alignof(Params));
} // namespace

extern "C"
{
__device__ __constant__ int cutlass_conv2d_shared_mem_bytes = ConvKernel::SharedStorageSize;
__device__ __constant__ int cutlass_conv2d_num_warps = ConvKernel::MaxThreadsPerBlock / 32;

__global__
#ifdef __CUDACC__
    __launch_bounds__(ConvKernel::MaxThreadsPerBlock, ConvKernel::MinBlocksPerMultiprocessor)
#endif
        void CUTLASS_CONV_KERNEL_NAME(CUTLASS_GRID_CONSTANT Conv2dKernelParams const raw_params)
{
    extern __shared__ char smem[];
    auto const &params = *reinterpret_cast<Params const *>(&raw_params);

    ConvKernel op;
    op(params, smem);
    cutlass::arch::synclog_print();
}
} // extern "C"
