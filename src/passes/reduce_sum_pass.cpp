#include "passes.h"
#include "shape_utils.h"

#include "passes/pass_utils.h"
#include <kernels/kernel_binaries.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <memory>

namespace
{
    struct ReduceGeometry
    {
        uint64_t outer{};
        uint64_t inner{};
        uint64_t cols{};
    };

    constexpr uint64_t kSplitReduceMinCols = 1024;
    constexpr uint64_t kSplitReduceMinInner = 64;

    uint64_t CeilDivU64(const uint64_t x, const uint64_t y) { return (x + y - 1U) / y; }

    ReduceGeometry ComputeReduceGeometry(const std::shared_ptr<pi::tensorlib::RealTensor> &input,
                                         const size_t reduction_dim_u)
    {
        uint64_t outer = 1;
        for (size_t dim = 0; dim < reduction_dim_u; ++dim)
        {
            outer *= input->shape()[dim];
        }

        uint64_t inner = 1;
        for (size_t dim = reduction_dim_u + 1; dim < input->shape().ndims(); ++dim)
        {
            inner *= input->shape()[dim];
        }

        return ReduceGeometry{.outer = outer, .inner = inner, .cols = input->shape()[reduction_dim_u]};
    }

    size_t NextExecutionEntryId(const pi::tensorlib::ExecutionPlan &execution_plan)
    {
        size_t next_entry_id = 0;
        for (const auto &entry : execution_plan.entries)
        {
            next_entry_id = std::max(next_entry_id, entry.id + 1);
        }
        return next_entry_id;
    }

    std::shared_ptr<pi::tensorlib::RealTensor> CreatePlanTemporary(pi::tensorlib::ExecutionPlan &execution_plan,
                                                                   const std::vector<uint64_t> &dims,
                                                                   const pi::tensorlib::DataType dtype,
                                                                   const pi::tensorlib::Device device)
    {
        pi::tensorlib::Shape shape{dims};
        auto tensor = std::make_shared<pi::tensorlib::RealTensor>(shape, pi::tensorlib::Strides(shape), dtype, device,
                                                                  false, 0, false);
        execution_plan.real_tensors.emplace(tensor->id(), tensor);
        return tensor;
    }

    bool ShouldUseSplitColumnReduce(const bool column_tiled, const ReduceGeometry &geometry)
    {
#if PI_TENSORLIB_ENABLE_CUDA
        if (IsEnvFlagEnabled("FBAMTRAIN_DISABLE_SPLIT_REDUCE"))
        {
            return false;
        }
        return column_tiled && geometry.cols >= kSplitReduceMinCols && geometry.inner >= kSplitReduceMinInner;
#else
        (void)column_tiled;
        (void)geometry;
        return false;
#endif
    }
} // namespace

static pi::tensorlib::ComputeKernelDescriptor CreateSumComputeKernelDescriptor(const int64_t reduction_dim,
                                                                               const bool keepdim,
                                                                               const pi::tensorlib::DataType dtype,
                                                                               const bool column_tiled)
{
    const bool is_fp32 = dtype == pi::tensorlib::DataType::FLOAT32;
    const auto kernel =
        is_fp32 ? (column_tiled ? ksum_reduce_column_tiled_fp32_out_fp32 : ksum_reduce_contiguous_fp32_out_fp32)
                : (column_tiled ? SelectKernelForHalf(ksum_reduce_column_tiled_bf16_out_fp32,
                                                      ksum_reduce_column_tiled_fp16_out_fp32, dtype)
                                : SelectKernelForHalf(ksum_reduce_contiguous_bf16_out_fp32,
                                                      ksum_reduce_contiguous_fp16_out_fp32, dtype));
    const std::string kernel_name =
        is_fp32 ? std::string(column_tiled ? "sum_reduce_column_tiled_fp32_out_fp32"
                                           : "sum_reduce_contiguous_fp32_out_fp32")
                : std::string(column_tiled ? "sum_reduce_column_tiled_" : "sum_reduce_contiguous_") +
                      std::string(KernelSuffixForHalf(dtype)) + "_out_fp32";
    const std::string dtype_name = pi::tensorlib::GetDataTypeName(dtype);

    return pi::tensorlib::ComputeKernelDescriptor{
        .kernel_name = kernel_name,
        .function_name = kernel.function_name,
        .expected_arg_count = kernel.arg_count,
        .argument_provider = [reduction_dim, keepdim, kernel, dtype, dtype_name,
                              column_tiled](const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                                            const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
            -> pi::tensorlib::KernelLaunchArguments
        {
            if (inputs.size() != 1 || outputs.size() != 1)
            {
                throw std::runtime_error("sum operation expects exactly one input and one output tensor");
            }

            const auto &input = inputs[0];
            const auto &output = outputs[0];

            if (input->dtype() != dtype)
            {
                throw std::runtime_error("sum operation currently supports " + std::string(dtype_name) +
                                         " input tensors only");
            }
            if (output->dtype() != pi::tensorlib::DataType::FLOAT32)
            {
                throw std::runtime_error("sum operation expects FLOAT32 output tensors");
            }

            if (input->device().device_type != pi::tensorlib::DeviceType::GPU ||
                output->device().device_type != pi::tensorlib::DeviceType::GPU)
            {
                throw std::runtime_error("sum operation requires GPU tensors on the same device");
            }
            const auto device_ordinal = input->device().ordinal;
            if (output->device().ordinal != device_ordinal)
            {
                throw std::runtime_error("sum operation requires GPU tensors on the same device");
            }

            const auto input_ndims = static_cast<int64_t>(input->shape().ndims());
            if (input_ndims == 0)
            {
                throw std::runtime_error("sum operation does not support scalar inputs");
            }
            int64_t reduction_dim_norm = reduction_dim;
            if (reduction_dim_norm < 0)
            {
                reduction_dim_norm += input_ndims;
            }
            if (reduction_dim_norm < 0 || reduction_dim_norm >= input_ndims)
            {
                throw std::runtime_error("sum operation reduction dimension is out of bounds");
            }
            const auto reduction_dim_u = static_cast<size_t>(reduction_dim_norm);

            const auto cols64 = input->shape()[reduction_dim_u];
            if (cols64 == 0)
            {
                throw std::runtime_error("sum operation requires non-zero reduction dimension size");
            }

            const auto total_elements = input->shape().numel();
            if (total_elements % cols64 != 0)
            {
                throw std::runtime_error("sum operation received incompatible shape for reduction");
            }

            const ReduceGeometry geometry = ComputeReduceGeometry(input, reduction_dim_u);
            const auto rows64 = geometry.outer * geometry.inner;
            if (rows64 == 0 || rows64 != total_elements / cols64)
            {
                throw std::runtime_error("sum operation computed invalid output shape");
            }

            if (geometry.outer > std::numeric_limits<uint32_t>::max() ||
                geometry.inner > std::numeric_limits<uint32_t>::max() ||
                geometry.cols > std::numeric_limits<uint32_t>::max())
            {
                throw std::runtime_error("sum operation exceeds supported tensor size");
            }

            if (!pi::tensorlib::shape_utils::IsRowMajorContiguous(input->shape(), input->strides()) ||
                !pi::tensorlib::shape_utils::IsRowMajorContiguous(output->shape(), output->strides()))
            {
                throw std::runtime_error("sum operation requires contiguous tensors");
            }

            if (keepdim)
            {
                if (output->shape().ndims() != input->shape().ndims())
                {
                    throw std::runtime_error("sum keepdim output must match input rank");
                }
                for (size_t dim = 0; dim < input->shape().ndims(); ++dim)
                {
                    const auto expected = (dim == reduction_dim_u) ? uint64_t{1} : input->shape()[dim];
                    if (output->shape()[dim] != expected)
                    {
                        throw std::runtime_error("sum keepdim output shape mismatch");
                    }
                }
            }
            else
            {
                if (output->shape().ndims() + 1 != input->shape().ndims())
                {
                    throw std::runtime_error("sum output rank should be input rank - 1 when keepdim is false");
                }
                size_t out_dim = 0;
                for (size_t dim = 0; dim < input->shape().ndims(); ++dim)
                {
                    if (dim == reduction_dim_u)
                    {
                        continue;
                    }
                    if (output->shape()[out_dim] != input->shape()[dim])
                    {
                        throw std::runtime_error("sum output shape mismatch when keepdim is false");
                    }
                    ++out_dim;
                }
            }

            void *output_ptr = output->dataptr();
            void *input_ptr = input->dataptr();

            const auto outer = static_cast<uint32_t>(geometry.outer);
            const auto inner = static_cast<uint32_t>(geometry.inner);
            const auto cols = static_cast<uint32_t>(geometry.cols);
            const uint32_t grid_y = column_tiled ? CEIL_DIV(inner, kernel.meta.block_size) : inner;

            pi::tensorlib::KernelLaunchArguments arguments{
                .args = {output_ptr, input_ptr, outer, inner, cols, static_cast<void *>(nullptr)},
                .grid_dim_x = outer,
                .grid_dim_y = grid_y,
                .grid_dim_z = 1,
                .block_dim_x = TRITON_WARP_SIZE * kernel.num_warps,
                .block_dim_y = 1,
                .block_dim_z = 1,
                .shared_mem_bytes = kernel.shared_mem_bytes,
                .device_ordinal = device_ordinal};
            return arguments;
        },
        .cuda_descriptor = pi::tensorlib::MakeCudaKernelDescriptor(kernel.data, kernel.size),
        .hip_descriptor = pi::tensorlib::MakeHipKernelDescriptor(kernel.data, kernel.size),
    };
}

#if PI_TENSORLIB_ENABLE_CUDA
static pi::tensorlib::ComputeKernelDescriptor
CreateSumSplitPartialComputeKernelDescriptor(const int64_t reduction_dim, const pi::tensorlib::DataType dtype)
{
    const bool is_fp32 = dtype == pi::tensorlib::DataType::FLOAT32;
    const auto kernel = is_fp32 ? ksum_reduce_column_split_partials_fp32_out_fp32
                                : SelectKernelForHalf(ksum_reduce_column_split_partials_bf16_out_fp32,
                                                      ksum_reduce_column_split_partials_fp16_out_fp32, dtype);
    const std::string kernel_name = is_fp32 ? "sum_reduce_column_split_partials_fp32_out_fp32"
                                            : std::string("sum_reduce_column_split_partials_") +
                                                  std::string(KernelSuffixForHalf(dtype)) + "_out_fp32";
    const std::string dtype_name = pi::tensorlib::GetDataTypeName(dtype);

    return pi::tensorlib::ComputeKernelDescriptor{
        .kernel_name = kernel_name,
        .function_name = kernel.function_name,
        .expected_arg_count = kernel.arg_count,
        .argument_provider = [reduction_dim, kernel, dtype, dtype_name,
                              kernel_name](const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                                           const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
            -> pi::tensorlib::KernelLaunchArguments
        {
            if (inputs.size() != 1 || outputs.size() != 1)
            {
                throw std::runtime_error(kernel_name + " expects exactly one input and one output tensor");
            }

            const auto &input = inputs[0];
            const auto &partials = outputs[0];
            if (input->dtype() != dtype)
            {
                throw std::runtime_error(kernel_name + " requires " + dtype_name + " input tensors");
            }
            if (partials->dtype() != pi::tensorlib::DataType::FLOAT32)
            {
                throw std::runtime_error(kernel_name + " requires FLOAT32 partial output tensors");
            }

            const auto device_ordinal =
                ValidateSameDeviceOrdinal(kernel_name, {input, partials}, pi::tensorlib::DeviceType::GPU);
            if (!pi::tensorlib::shape_utils::IsRowMajorContiguous(input->shape(), input->strides()) ||
                !pi::tensorlib::shape_utils::IsRowMajorContiguous(partials->shape(), partials->strides()))
            {
                throw std::runtime_error(kernel_name + " requires contiguous tensors");
            }

            const auto input_ndims = static_cast<int64_t>(input->shape().ndims());
            int64_t reduction_dim_norm = reduction_dim;
            if (reduction_dim_norm < 0)
            {
                reduction_dim_norm += input_ndims;
            }
            if (reduction_dim_norm < 0 || reduction_dim_norm >= input_ndims)
            {
                throw std::runtime_error(kernel_name + " reduction dimension is out of bounds");
            }
            const auto reduction_dim_u = static_cast<size_t>(reduction_dim_norm);
            if (partials->shape().ndims() != input->shape().ndims())
            {
                throw std::runtime_error(kernel_name + " partial rank mismatch");
            }

            const ReduceGeometry geometry = ComputeReduceGeometry(input, reduction_dim_u);
            const uint64_t split_count = CeilDivU64(geometry.cols, kernel.meta.block_size_y);
            for (size_t dim = 0; dim < input->shape().ndims(); ++dim)
            {
                const uint64_t expected = dim == reduction_dim_u ? split_count : input->shape()[dim];
                if (partials->shape()[dim] != expected)
                {
                    throw std::runtime_error(kernel_name + " partial shape mismatch");
                }
            }
            if (geometry.outer > std::numeric_limits<uint32_t>::max() ||
                geometry.inner > std::numeric_limits<uint32_t>::max() ||
                geometry.cols > std::numeric_limits<uint32_t>::max() ||
                split_count > std::numeric_limits<uint32_t>::max())
            {
                throw std::runtime_error(kernel_name + " exceeds supported tensor size");
            }

            return pi::tensorlib::KernelLaunchArguments{
                .args = {partials->dataptr(), input->dataptr(), static_cast<uint32_t>(geometry.outer),
                         static_cast<uint32_t>(geometry.inner), static_cast<uint32_t>(geometry.cols),
                         static_cast<uint32_t>(split_count), static_cast<void *>(nullptr)},
                .grid_dim_x = static_cast<uint32_t>(geometry.outer),
                .grid_dim_y = static_cast<uint32_t>(split_count),
                .grid_dim_z = static_cast<uint32_t>(CEIL_DIV(geometry.inner, kernel.meta.block_size_x)),
                .block_dim_x = TRITON_WARP_SIZE * kernel.num_warps,
                .block_dim_y = 1,
                .block_dim_z = 1,
                .shared_mem_bytes = kernel.shared_mem_bytes,
                .device_ordinal = device_ordinal};
        },
        .cuda_descriptor = pi::tensorlib::MakeCudaKernelDescriptor(kernel.data, kernel.size),
        .hip_descriptor = pi::tensorlib::MakeHipKernelDescriptor(kernel.data, kernel.size),
    };
}

static void LowerSplitColumnReduce(pi::tensorlib::ExecutionPlan &execution_plan, const size_t entry_idx,
                                   const int64_t reduction_dim, const bool keepdim, const pi::tensorlib::DataType dtype,
                                   const ReduceGeometry &geometry)
{
    auto &reduce_entry = execution_plan.entries[entry_idx];
    const auto input = reduce_entry.inputs[0];
    const auto output = reduce_entry.outputs[0];
    const auto reduction_dim_u = static_cast<size_t>(reduction_dim);

    const auto split_kernel = dtype == pi::tensorlib::DataType::FLOAT32
                                  ? ksum_reduce_column_split_partials_fp32_out_fp32
                                  : SelectKernelForHalf(ksum_reduce_column_split_partials_bf16_out_fp32,
                                                        ksum_reduce_column_split_partials_fp16_out_fp32, dtype);
    const uint64_t split_count = CeilDivU64(geometry.cols, split_kernel.meta.block_size_y);
    std::vector<uint64_t> partial_dims = input->shape().dims();
    partial_dims[reduction_dim_u] = split_count;
    auto partials =
        CreatePlanTemporary(execution_plan, partial_dims, pi::tensorlib::DataType::FLOAT32, input->device());

    const auto stream_desc = reduce_entry.gpu_stream_desc;
    const bool is_useful = reduce_entry.is_useful;
    size_t next_entry_id = NextExecutionEntryId(execution_plan);

    pi::tensorlib::ExecutionEntry create_partials{};
    create_partials.id = next_entry_id++;
    create_partials.op_type = pi::tensorlib::OpType::CREATE_TENSOR;
    create_partials.outputs = {partials};
    create_partials.is_useful = false;
    create_partials.gpu_stream_desc = stream_desc;

    reduce_entry.op_type = std::nullopt;
    reduce_entry.inputs = {input};
    reduce_entry.outputs = {partials};
    reduce_entry.kernel_descriptor = CreateSumSplitPartialComputeKernelDescriptor(reduction_dim, dtype);
    reduce_entry.flop_estimate = input->shape().numel();

    const bool final_column_tiled = partials->strides()[reduction_dim_u] != 1;
    pi::tensorlib::ExecutionEntry final_reduce{};
    final_reduce.id = next_entry_id++;
    final_reduce.op_type = std::nullopt;
    final_reduce.inputs = {partials};
    final_reduce.outputs = {output};
    final_reduce.is_useful = is_useful;
    final_reduce.kernel_descriptor =
        CreateSumComputeKernelDescriptor(reduction_dim, keepdim, pi::tensorlib::DataType::FLOAT32, final_column_tiled);
    final_reduce.flop_estimate = output->shape().numel() * split_count;
    final_reduce.gpu_stream_desc = stream_desc;

    pi::tensorlib::ExecutionEntry delete_partials{};
    delete_partials.id = next_entry_id++;
    delete_partials.op_type = pi::tensorlib::OpType::DELETE_TENSOR;
    delete_partials.inputs = {partials};
    delete_partials.is_useful = false;
    delete_partials.gpu_stream_desc = stream_desc;

    execution_plan.entries.insert(execution_plan.entries.begin() + static_cast<std::ptrdiff_t>(entry_idx),
                                  std::move(create_partials));
    execution_plan.entries.insert(execution_plan.entries.begin() + static_cast<std::ptrdiff_t>(entry_idx + 2),
                                  std::move(final_reduce));
    execution_plan.entries.insert(execution_plan.entries.begin() + static_cast<std::ptrdiff_t>(entry_idx + 3),
                                  std::move(delete_partials));
}
#endif

void ReduceSumImplPass::transform(pi::tensorlib::ExecutionPlan &execution_plan)
{
    for (size_t entry_idx = 0; entry_idx < execution_plan.entries.size(); ++entry_idx)
    {
        if (execution_plan.entries[entry_idx].op_type != pi::tensorlib::OpType::REDUCE_SUM)
        {
            continue;
        }

        if (execution_plan.entries[entry_idx].inputs.size() != 1 ||
            execution_plan.entries[entry_idx].outputs.size() != 1)
        {
            throw std::runtime_error("sum operation expects exactly one input and one output tensor");
        }

        const auto dim_it = execution_plan.entries[entry_idx].attributes.find("dim");
        const auto keepdim_it = execution_plan.entries[entry_idx].attributes.find("keepdim");
        if (dim_it == execution_plan.entries[entry_idx].attributes.end() ||
            keepdim_it == execution_plan.entries[entry_idx].attributes.end())
        {
            throw std::runtime_error("sum operation is missing required attributes");
        }

        auto reduction_dim = std::any_cast<int64_t>(dim_it->second);
        const auto input = execution_plan.entries[entry_idx].inputs[0];
        const auto output = execution_plan.entries[entry_idx].outputs[0];
        const auto input_ndims = static_cast<int64_t>(input->shape().ndims());
        if (reduction_dim < 0)
        {
            reduction_dim += input_ndims;
        }
        if (reduction_dim < 0 || reduction_dim >= input_ndims)
        {
            throw std::runtime_error("sum operation reduction dimension is out of bounds");
        }
        const auto keepdim = std::any_cast<bool>(keepdim_it->second);

        const auto dtype = input->dtype();

        if (!IsSupportedHalfType(dtype) && dtype != pi::tensorlib::DataType::FLOAT32)
        {
            continue;
        }

        const auto output_dtype = output->dtype();
        const bool output_is_fp32 = output_dtype == pi::tensorlib::DataType::FLOAT32;
        if (!output_is_fp32)
        {
            continue;
        }

        const auto reduction_dim_u = static_cast<size_t>(reduction_dim);
        const auto stride = input->strides()[reduction_dim_u];
        const bool column_tiled = stride != 1;
        const ReduceGeometry geometry = ComputeReduceGeometry(input, reduction_dim_u);

#if PI_TENSORLIB_ENABLE_CUDA
        if (ShouldUseSplitColumnReduce(column_tiled, geometry))
        {
            LowerSplitColumnReduce(execution_plan, entry_idx, reduction_dim, keepdim, dtype, geometry);
            continue;
        }
#endif

        auto &reduce_entry = execution_plan.entries[entry_idx];

        reduce_entry.op_type = std::nullopt;
        reduce_entry.kernel_descriptor = CreateSumComputeKernelDescriptor(reduction_dim, keepdim, dtype, column_tiled);
        reduce_entry.flop_estimate = output->shape().numel() * input->shape()[reduction_dim_u];
    }
}
