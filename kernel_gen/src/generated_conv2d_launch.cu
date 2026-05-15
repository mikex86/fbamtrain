#include <nv/generated_conv2d_launch.h>

#include <cuda.h>

#include <cstring>
#include <exception>
#include <sstream>

#include "conv2d_cutlass3_wgrad_traits.cuh"

namespace fbamtrain::kernel_gen::nv
{
namespace
{
int QuerySmCount(const int device_ordinal, std::string &error)
{
    CUdevice device{};
    if (const CUresult result = cuDeviceGet(&device, device_ordinal); result != CUDA_SUCCESS)
    {
        const char *msg = nullptr;
        cuGetErrorString(result, &msg);
        error = std::string("cuDeviceGet failed: ") + (msg ? msg : "unknown CUDA error");
        return 0;
    }

    int sm_count = 0;
    if (const CUresult result = cuDeviceGetAttribute(&sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
        result != CUDA_SUCCESS)
    {
        const char *msg = nullptr;
        cuGetErrorString(result, &msg);
        error = std::string("cuDeviceGetAttribute(MULTIPROCESSOR_COUNT) failed: ") +
                (msg ? msg : "unknown CUDA error");
        return 0;
    }
    return sm_count;
}

template <typename Element, cutlass::conv::Operator ConvOp>
bool PrepareConv2d(const kernel_meta_conv2d_cutlass_t &meta, const GeneratedConv2dRequest &request,
                   GeneratedConv2dLaunchPlan &plan, std::string &error)
{
    using Traits = fbamtrain::kernel_gen::cutlass3_conv2d::Conv2dTraits<
        ConvOp,
        cutlass::arch::Sm90, Element, float, 128, 128, 64, 2, 4, 1, 8, 8>;
    using ConvKernel = typename Traits::ConvKernel;
    using Params = typename Traits::Params;

    if (request.groups != 1)
    {
        error = "CUTLASS3 conv2d helper only supports groups=1";
        return false;
    }
    if (request.r != static_cast<int32_t>(meta.kernel_h) || request.s != static_cast<int32_t>(meta.kernel_w))
    {
        error = "kernel size does not match selected CUTLASS3 wgrad binary";
        return false;
    }

    const int sm_count = QuerySmCount(request.device_ordinal, error);
    if (sm_count <= 0)
    {
        return false;
    }

    try
    {
        cutlass::KernelHardwareInfo hw_info{};
        hw_info.device_id = request.device_ordinal;
        hw_info.sm_count = sm_count;

        auto problem_shape = Traits::make_problem_shape(
            request.n, request.h, request.w, request.c, request.k, request.r, request.s, request.pad_h,
            request.pad_w, request.stride_h, request.stride_w, request.dilation_h, request.dilation_w,
            request.groups);
        auto arguments = Traits::make_arguments(problem_shape, static_cast<Element const *>(request.ptr_a),
                                                static_cast<Element const *>(request.ptr_b),
                                                static_cast<Element *>(request.ptr_c),
                                                static_cast<Element *>(request.ptr_d),
                                                request.alpha, request.beta, hw_info);
        if (!ConvKernel::can_implement(arguments))
        {
            error = "CUTLASS3 conv2d kernel cannot implement requested shape";
            return false;
        }

        Params params = ConvKernel::to_underlying_arguments(arguments, nullptr);
        plan.params.resize(sizeof(Params));
        std::memcpy(plan.params.data(), &params, sizeof(Params));

        const dim3 grid = ConvKernel::get_grid_shape(params);
        plan.grid_dim_x = grid.x;
        plan.grid_dim_y = grid.y;
        plan.grid_dim_z = grid.z;
        plan.block_dim_x = ConvKernel::MaxThreadsPerBlock;
        plan.block_dim_y = 1;
        plan.block_dim_z = 1;
        plan.shared_mem_bytes = ConvKernel::SharedStorageSize;
        plan.cluster_dim_x = 2;
        plan.cluster_dim_y = 4;
        plan.cluster_dim_z = 1;
        return true;
    }
    catch (const std::exception &ex)
    {
        error = ex.what();
        return false;
    }
}
} // namespace

template <typename Element>
bool PrepareConv2dForOperation(const kernel_meta_conv2d_cutlass_t &meta,
                               const GeneratedConv2dRequest &request,
                               GeneratedConv2dLaunchPlan &plan,
                               std::string &error)
{
    switch (request.operation)
    {
        case GeneratedConv2dOperation::Fprop:
            return PrepareConv2d<Element, cutlass::conv::Operator::kFprop>(meta, request, plan, error);
        case GeneratedConv2dOperation::Dgrad:
            return PrepareConv2d<Element, cutlass::conv::Operator::kDgrad>(meta, request, plan, error);
        case GeneratedConv2dOperation::Wgrad:
            return PrepareConv2d<Element, cutlass::conv::Operator::kWgrad>(meta, request, plan, error);
    }
    error = "unsupported conv2d operation";
    return false;
}

bool PrepareGeneratedConv2dLaunch(const kernel_meta_conv2d_cutlass_t &meta,
                                  const GeneratedConv2dRequest &request,
                                  GeneratedConv2dLaunchPlan &plan,
                                  std::string &error)
{
    switch (request.dtype)
    {
        case GeneratedConv2dDType::BF16:
            return PrepareConv2dForOperation<cutlass::bfloat16_t>(meta, request, plan, error);
        case GeneratedConv2dDType::FP16:
            return PrepareConv2dForOperation<cutlass::half_t>(meta, request, plan, error);
    }
    error = "unsupported dtype";
    return false;
}
} // namespace fbamtrain::kernel_gen::nv
