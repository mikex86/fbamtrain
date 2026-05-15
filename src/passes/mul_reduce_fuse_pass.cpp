#include "passes.h"
#include "shape_utils.h"

#include "passes/pass_utils.h"
#include <kernels/kernel_binaries.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

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

    void RemoveEntryById(pi::tensorlib::ExecutionPlan &execution_plan, const size_t entry_id)
    {
        const auto it = std::ranges::find_if(execution_plan.entries, [entry_id](const pi::tensorlib::ExecutionEntry &e)
                                             { return e.id == entry_id; });
        if (it != execution_plan.entries.end())
        {
            pi::tensorlib::passes::RemoveOperation(execution_plan, *it);
        }
    }

    void RemoveTensorCreateDeleteEntries(pi::tensorlib::ExecutionPlan &execution_plan, const uint64_t tensor_id)
    {
        std::vector<size_t> remove_ids;
        for (const auto &entry : execution_plan.entries)
        {
            if (entry.op_type == pi::tensorlib::OpType::CREATE_TENSOR)
            {
                for (const auto &out : entry.outputs)
                {
                    if (out && out->id() == tensor_id)
                    {
                        remove_ids.push_back(entry.id);
                    }
                }
            }
            if (entry.op_type == pi::tensorlib::OpType::DELETE_TENSOR)
            {
                for (const auto &in : entry.inputs)
                {
                    if (in && in->id() == tensor_id)
                    {
                        remove_ids.push_back(entry.id);
                    }
                }
            }
        }
        for (const auto id : remove_ids)
        {
            RemoveEntryById(execution_plan, id);
        }
    }
} // namespace

static pi::tensorlib::ComputeKernelDescriptor
CreateMulReduceComputeKernelDescriptor(const int64_t reduction_dim, const bool keepdim,
                                       const pi::tensorlib::DataType dtype, const bool column_tiled)
{
    const auto kernel =
        column_tiled
            ? SelectKernelForHalf(kmul_reduce_column_tiled_bf16_out_fp32, kmul_reduce_column_tiled_fp16_out_fp32, dtype)
            : SelectKernelForHalf(kmul_reduce_contiguous_bf16_out_fp32, kmul_reduce_contiguous_fp16_out_fp32, dtype);
    const std::string kernel_name = std::string(column_tiled ? "mul_reduce_column_tiled_" : "mul_reduce_contiguous_") +
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
            if (inputs.size() != 2 || outputs.size() != 1)
            {
                throw std::runtime_error("mul_reduce expects two inputs and one output tensor");
            }

            const auto &lhs = inputs[0];
            const auto &rhs = inputs[1];
            const auto &output = outputs[0];

            if (lhs->dtype() != dtype || rhs->dtype() != dtype)
            {
                throw std::runtime_error("mul_reduce requires " + std::string(dtype_name) + " input tensors");
            }
            if (output->dtype() != pi::tensorlib::DataType::FLOAT32)
            {
                throw std::runtime_error("mul_reduce expects FLOAT32 output tensors");
            }

            if (lhs->device().device_type != pi::tensorlib::DeviceType::GPU ||
                rhs->device().device_type != pi::tensorlib::DeviceType::GPU ||
                output->device().device_type != pi::tensorlib::DeviceType::GPU)
            {
                throw std::runtime_error("mul_reduce requires GPU tensors on the same device");
            }
            const auto device_ordinal = ValidateSameDeviceOrdinal("mul_reduce", {lhs, rhs, output});

            if (lhs->shape().ndims() == 0)
            {
                throw std::runtime_error("mul_reduce does not support scalar inputs");
            }
            if (lhs->shape().dims() != rhs->shape().dims())
            {
                throw std::runtime_error("mul_reduce expects input tensors with identical shapes");
            }

            const auto input_ndims = static_cast<int64_t>(lhs->shape().ndims());
            int64_t reduction_dim_norm = reduction_dim;
            if (reduction_dim_norm < 0)
            {
                reduction_dim_norm += input_ndims;
            }
            if (reduction_dim_norm < 0 || reduction_dim_norm >= input_ndims)
            {
                throw std::runtime_error("mul_reduce reduction dimension is out of bounds");
            }
            const auto reduction_dim_u = static_cast<size_t>(reduction_dim_norm);

            const auto cols64 = lhs->shape()[reduction_dim_u];
            if (cols64 == 0)
            {
                throw std::runtime_error("mul_reduce requires non-zero reduction dimension size");
            }

            const auto total_elements = lhs->shape().numel();
            if (total_elements % cols64 != 0)
            {
                throw std::runtime_error("mul_reduce received incompatible shape for reduction");
            }

            uint64_t outer64 = 1;
            for (size_t dim = 0; dim < reduction_dim_u; ++dim)
            {
                outer64 *= lhs->shape()[dim];
            }
            uint64_t inner64 = 1;
            for (size_t dim = reduction_dim_u + 1; dim < lhs->shape().ndims(); ++dim)
            {
                inner64 *= lhs->shape()[dim];
            }

            const auto rows64 = outer64 * inner64;
            if (rows64 == 0 || rows64 != total_elements / cols64)
            {
                throw std::runtime_error("mul_reduce computed invalid output shape");
            }

            if (outer64 > std::numeric_limits<uint32_t>::max() || inner64 > std::numeric_limits<uint32_t>::max() ||
                cols64 > std::numeric_limits<uint32_t>::max())
            {
                throw std::runtime_error("mul_reduce exceeds supported tensor size");
            }

            if (!pi::tensorlib::shape_utils::IsRowMajorContiguous(lhs->shape(), lhs->strides()) ||
                !pi::tensorlib::shape_utils::IsRowMajorContiguous(rhs->shape(), rhs->strides()) ||
                !pi::tensorlib::shape_utils::IsRowMajorContiguous(output->shape(), output->strides()))
            {
                throw std::runtime_error("mul_reduce requires contiguous tensors");
            }

            if (keepdim)
            {
                if (output->shape().ndims() != lhs->shape().ndims())
                {
                    throw std::runtime_error("mul_reduce keepdim output must match input rank");
                }
                for (size_t dim = 0; dim < lhs->shape().ndims(); ++dim)
                {
                    const auto expected = (dim == reduction_dim_u) ? uint64_t{1} : lhs->shape()[dim];
                    if (output->shape()[dim] != expected)
                    {
                        throw std::runtime_error("mul_reduce keepdim output shape mismatch");
                    }
                }
            }
            else
            {
                if (output->shape().ndims() + 1 != lhs->shape().ndims())
                {
                    throw std::runtime_error("mul_reduce output rank should be input rank - 1 when keepdim is false");
                }
                size_t out_dim = 0;
                for (size_t dim = 0; dim < lhs->shape().ndims(); ++dim)
                {
                    if (dim == reduction_dim_u)
                    {
                        continue;
                    }
                    if (output->shape()[out_dim] != lhs->shape()[dim])
                    {
                        throw std::runtime_error("mul_reduce output shape mismatch when keepdim is false");
                    }
                    ++out_dim;
                }
            }

            void *output_ptr = output->dataptr();
            void *lhs_ptr = lhs->dataptr();
            void *rhs_ptr = rhs->dataptr();

            const auto outer = static_cast<uint32_t>(outer64);
            const auto inner = static_cast<uint32_t>(inner64);
            const auto cols = static_cast<uint32_t>(cols64);
            const uint32_t grid_y = column_tiled ? CEIL_DIV(inner, kernel.meta.block_size) : inner;

            if (const char *env = std::getenv("FBAMTRAIN_DEBUG_MUL_REDUCE_LAUNCH"); env != nullptr && env[0] != '\0')
            {
                std::clog << "[MulReduceLaunch] out_id=" << output->id() << " out_ptr=" << output_ptr
                          << " lhs_id=" << lhs->id() << " rhs_id=" << rhs->id() << " outer=" << outer
                          << " inner=" << inner << " cols=" << cols << '\n';
            }

            pi::tensorlib::KernelLaunchArguments arguments{
                .args = {output_ptr, lhs_ptr, rhs_ptr, outer, inner, cols, static_cast<void *>(nullptr)},
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

static pi::tensorlib::ComputeKernelDescriptor
CreateFp32SumComputeKernelDescriptor(const int64_t reduction_dim, const bool keepdim, const bool column_tiled)
{
    const auto kernel = column_tiled ? ksum_reduce_column_tiled_fp32_out_fp32 : ksum_reduce_contiguous_fp32_out_fp32;
    const std::string kernel_name =
        column_tiled ? "sum_reduce_column_tiled_fp32_out_fp32" : "sum_reduce_contiguous_fp32_out_fp32";

    return pi::tensorlib::ComputeKernelDescriptor{
        .kernel_name = kernel_name,
        .function_name = kernel.function_name,
        .expected_arg_count = kernel.arg_count,
        .argument_provider = [reduction_dim, keepdim, kernel, kernel_name,
                              column_tiled](const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                                            const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
            -> pi::tensorlib::KernelLaunchArguments
        {
            if (inputs.size() != 1 || outputs.size() != 1)
            {
                throw std::runtime_error(kernel_name + " expects one input and one output tensor");
            }

            const auto &input = inputs[0];
            const auto &output = outputs[0];
            if (input->dtype() != pi::tensorlib::DataType::FLOAT32 ||
                output->dtype() != pi::tensorlib::DataType::FLOAT32)
            {
                throw std::runtime_error(kernel_name + " requires FLOAT32 tensors");
            }

            const auto device_ordinal =
                ValidateSameDeviceOrdinal(kernel_name, {input, output}, pi::tensorlib::DeviceType::GPU);

            if (!pi::tensorlib::shape_utils::IsRowMajorContiguous(input->shape(), input->strides()) ||
                !pi::tensorlib::shape_utils::IsRowMajorContiguous(output->shape(), output->strides()))
            {
                throw std::runtime_error(kernel_name + " requires contiguous tensors");
            }

            const auto input_ndims = static_cast<int64_t>(input->shape().ndims());
            if (input_ndims == 0)
            {
                throw std::runtime_error(kernel_name + " does not support scalar inputs");
            }

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

            const ReduceGeometry geometry = ComputeReduceGeometry(input, reduction_dim_u);
            if (geometry.cols == 0)
            {
                throw std::runtime_error(kernel_name + " requires non-zero reduction dimension size");
            }

            const auto rows64 = geometry.outer * geometry.inner;
            if (rows64 == 0 || rows64 != input->shape().numel() / geometry.cols)
            {
                throw std::runtime_error(kernel_name + " computed invalid output shape");
            }
            if (geometry.outer > std::numeric_limits<uint32_t>::max() ||
                geometry.inner > std::numeric_limits<uint32_t>::max() ||
                geometry.cols > std::numeric_limits<uint32_t>::max())
            {
                throw std::runtime_error(kernel_name + " exceeds supported tensor size");
            }

            if (keepdim)
            {
                if (output->shape().ndims() != input->shape().ndims())
                {
                    throw std::runtime_error(kernel_name + " keepdim output must match input rank");
                }
                for (size_t dim = 0; dim < input->shape().ndims(); ++dim)
                {
                    const auto expected = (dim == reduction_dim_u) ? uint64_t{1} : input->shape()[dim];
                    if (output->shape()[dim] != expected)
                    {
                        throw std::runtime_error(kernel_name + " keepdim output shape mismatch");
                    }
                }
            }
            else
            {
                if (output->shape().ndims() + 1 != input->shape().ndims())
                {
                    throw std::runtime_error(kernel_name +
                                             " output rank should be input rank - 1 when keepdim is false");
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
                        throw std::runtime_error(kernel_name + " output shape mismatch when keepdim is false");
                    }
                    ++out_dim;
                }
            }

            const auto outer = static_cast<uint32_t>(geometry.outer);
            const auto inner = static_cast<uint32_t>(geometry.inner);
            const auto cols = static_cast<uint32_t>(geometry.cols);
            const uint32_t grid_y = column_tiled ? CEIL_DIV(inner, kernel.meta.block_size) : inner;

            return pi::tensorlib::KernelLaunchArguments{
                .args = {output->dataptr(), input->dataptr(), outer, inner, cols, static_cast<void *>(nullptr)},
                .grid_dim_x = outer,
                .grid_dim_y = grid_y,
                .grid_dim_z = 1,
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

#if PI_TENSORLIB_ENABLE_CUDA
static pi::tensorlib::ComputeKernelDescriptor
CreateMulReduceSplitPartialComputeKernelDescriptor(const int64_t reduction_dim, const pi::tensorlib::DataType dtype)
{
    const auto kernel = SelectKernelForHalf(kmul_reduce_column_split_partials_bf16_out_fp32,
                                            kmul_reduce_column_split_partials_fp16_out_fp32, dtype);
    const std::string kernel_name =
        std::string("mul_reduce_column_split_partials_") + std::string(KernelSuffixForHalf(dtype)) + "_out_fp32";
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
            if (inputs.size() != 2 || outputs.size() != 1)
            {
                throw std::runtime_error(kernel_name + " expects two inputs and one output tensor");
            }

            const auto &lhs = inputs[0];
            const auto &rhs = inputs[1];
            const auto &partials = outputs[0];
            if (lhs->dtype() != dtype || rhs->dtype() != dtype)
            {
                throw std::runtime_error(kernel_name + " requires " + dtype_name + " input tensors");
            }
            if (partials->dtype() != pi::tensorlib::DataType::FLOAT32)
            {
                throw std::runtime_error(kernel_name + " requires FLOAT32 partial output tensors");
            }
            if (lhs->shape().dims() != rhs->shape().dims())
            {
                throw std::runtime_error(kernel_name + " expects input tensors with identical shapes");
            }

            const auto device_ordinal =
                ValidateSameDeviceOrdinal(kernel_name, {lhs, rhs, partials}, pi::tensorlib::DeviceType::GPU);
            if (!pi::tensorlib::shape_utils::IsRowMajorContiguous(lhs->shape(), lhs->strides()) ||
                !pi::tensorlib::shape_utils::IsRowMajorContiguous(rhs->shape(), rhs->strides()) ||
                !pi::tensorlib::shape_utils::IsRowMajorContiguous(partials->shape(), partials->strides()))
            {
                throw std::runtime_error(kernel_name + " requires contiguous tensors");
            }

            const auto input_ndims = static_cast<int64_t>(lhs->shape().ndims());
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

            if (partials->shape().ndims() != lhs->shape().ndims())
            {
                throw std::runtime_error(kernel_name + " partial rank mismatch");
            }

            const ReduceGeometry geometry = ComputeReduceGeometry(lhs, reduction_dim_u);
            const uint64_t split_count = CeilDivU64(geometry.cols, kernel.meta.block_size_y);
            for (size_t dim = 0; dim < lhs->shape().ndims(); ++dim)
            {
                const uint64_t expected = dim == reduction_dim_u ? split_count : lhs->shape()[dim];
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
                .args = {partials->dataptr(), lhs->dataptr(), rhs->dataptr(), static_cast<uint32_t>(geometry.outer),
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

static void LowerSplitMulReduce(pi::tensorlib::ExecutionPlan &execution_plan, const size_t reduce_idx,
                                const int64_t reduction_dim, const bool keepdim, const pi::tensorlib::DataType dtype,
                                const std::shared_ptr<pi::tensorlib::RealTensor> &reduce_input,
                                const std::shared_ptr<pi::tensorlib::RealTensor> &mul_lhs,
                                const std::shared_ptr<pi::tensorlib::RealTensor> &mul_rhs,
                                const std::shared_ptr<pi::tensorlib::RealTensor> &output,
                                const ReduceGeometry &geometry)
{
    auto &reduce_entry = execution_plan.entries[reduce_idx];
    const auto reduction_dim_u = static_cast<size_t>(reduction_dim);
    const auto split_kernel = SelectKernelForHalf(kmul_reduce_column_split_partials_bf16_out_fp32,
                                                  kmul_reduce_column_split_partials_fp16_out_fp32, dtype);
    const uint64_t split_count = CeilDivU64(geometry.cols, split_kernel.meta.block_size_y);
    std::vector<uint64_t> partial_dims = reduce_input->shape().dims();
    partial_dims[reduction_dim_u] = split_count;
    auto partials =
        CreatePlanTemporary(execution_plan, partial_dims, pi::tensorlib::DataType::FLOAT32, reduce_input->device());

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
    reduce_entry.inputs = {mul_lhs, mul_rhs};
    reduce_entry.outputs = {partials};
    reduce_entry.kernel_descriptor = CreateMulReduceSplitPartialComputeKernelDescriptor(reduction_dim, dtype);
    reduce_entry.flop_estimate = reduce_input->shape().numel();

    const bool final_column_tiled = partials->strides()[reduction_dim_u] != 1;
    pi::tensorlib::ExecutionEntry final_reduce{};
    final_reduce.id = next_entry_id++;
    final_reduce.op_type = std::nullopt;
    final_reduce.inputs = {partials};
    final_reduce.outputs = {output};
    final_reduce.is_useful = is_useful;
    final_reduce.kernel_descriptor = CreateFp32SumComputeKernelDescriptor(reduction_dim, keepdim, final_column_tiled);
    final_reduce.flop_estimate = output->shape().numel() * split_count;
    final_reduce.gpu_stream_desc = stream_desc;

    pi::tensorlib::ExecutionEntry delete_partials{};
    delete_partials.id = next_entry_id++;
    delete_partials.op_type = pi::tensorlib::OpType::DELETE_TENSOR;
    delete_partials.inputs = {partials};
    delete_partials.is_useful = false;
    delete_partials.gpu_stream_desc = stream_desc;

    execution_plan.entries.insert(execution_plan.entries.begin() + static_cast<std::ptrdiff_t>(reduce_idx),
                                  std::move(create_partials));
    execution_plan.entries.insert(execution_plan.entries.begin() + static_cast<std::ptrdiff_t>(reduce_idx + 2),
                                  std::move(final_reduce));
    execution_plan.entries.insert(execution_plan.entries.begin() + static_cast<std::ptrdiff_t>(reduce_idx + 3),
                                  std::move(delete_partials));
}
#endif

void FuseMulReducePass::transform(pi::tensorlib::ExecutionPlan &execution_plan)
{
    const bool debug_mul_reduce = std::getenv("FBAMTRAIN_DEBUG_MUL_REDUCE") != nullptr;

    bool changed = true;
    while (changed)
    {
        changed = false;
        for (size_t i = 0; i < execution_plan.entries.size(); ++i)
        {
            auto &mul_op = execution_plan.entries[i];
            if (mul_op.op_type != pi::tensorlib::OpType::MUL)
            {
                continue;
            }
            if (mul_op.inputs.size() != 2 || mul_op.outputs.size() != 1)
            {
                continue;
            }

            const size_t mul_entry_id = mul_op.id;
            const auto &mul_lhs = mul_op.inputs[0];
            const auto &mul_rhs = mul_op.inputs[1];
            const auto &mul_out = mul_op.outputs[0];
            if (!mul_lhs || !mul_rhs || !mul_out)
            {
                continue;
            }

            size_t reduce_idx = execution_plan.entries.size();
            for (size_t j = 0; j < execution_plan.entries.size(); ++j)
            {
                const auto &candidate = execution_plan.entries[j];
                if (candidate.op_type != pi::tensorlib::OpType::REDUCE_SUM)
                {
                    continue;
                }
                if (candidate.inputs.size() != 1 || candidate.outputs.size() != 1)
                {
                    continue;
                }
                const auto &reduce_input = candidate.inputs[0];
                if (reduce_input && reduce_input->id() == mul_out->id())
                {
                    reduce_idx = j;
                    break;
                }
            }
            if (reduce_idx == execution_plan.entries.size())
            {
                continue;
            }

            auto &reduce_op = execution_plan.entries[reduce_idx];
            const size_t reduce_entry_id = reduce_op.id;
            const uint64_t mul_out_id = mul_out->id();

            size_t mul_use_count = 0;
            for (const auto &entry : execution_plan.entries)
            {
                if (entry.op_type == std::nullopt || entry.op_type == pi::tensorlib::OpType::DELETE_TENSOR ||
                    entry.id == mul_entry_id)
                {
                    continue;
                }
                for (const auto &input : entry.inputs)
                {
                    if (input && input->id() == mul_out->id())
                    {
                        ++mul_use_count;
                    }
                }
            }
            if (mul_use_count != 1)
            {
                continue;
            }

            const auto dim_it = reduce_op.attributes.find("dim");
            const auto keepdim_it = reduce_op.attributes.find("keepdim");
            if (dim_it == reduce_op.attributes.end() || keepdim_it == reduce_op.attributes.end())
            {
                continue;
            }

            auto reduction_dim = std::any_cast<int64_t>(dim_it->second);
            const auto keepdim = std::any_cast<bool>(keepdim_it->second);

            const auto dtype = mul_lhs->dtype();
            if (!IsSupportedHalfType(dtype) || mul_rhs->dtype() != dtype)
            {
                continue;
            }

            const auto reduce_input = reduce_op.inputs[0];
            if (!reduce_input)
            {
                continue;
            }

            if (mul_lhs->shape().numel() != reduce_input->shape().numel() ||
                mul_rhs->shape().numel() != reduce_input->shape().numel())
            {
                continue;
            }

            if (!pi::tensorlib::shape_utils::IsRowMajorContiguous(mul_lhs) ||
                !pi::tensorlib::shape_utils::IsRowMajorContiguous(mul_rhs) ||
                !pi::tensorlib::shape_utils::IsRowMajorContiguous(reduce_input))
            {
                continue;
            }

            const auto output = reduce_op.outputs[0];
            if (!output)
            {
                continue;
            }

            const auto output_dtype = output->dtype();
            const bool output_is_fp32 = output_dtype == pi::tensorlib::DataType::FLOAT32;
            if (!output_is_fp32)
            {
                continue;
            }

            const auto input_ndims = static_cast<int64_t>(reduce_input->shape().ndims());
            if (input_ndims == 0)
            {
                continue;
            }
            if (reduction_dim < 0)
            {
                reduction_dim += input_ndims;
            }
            if (reduction_dim < 0 || reduction_dim >= input_ndims)
            {
                continue;
            }
            const auto reduction_dim_u = static_cast<size_t>(reduction_dim);
            const auto stride = reduce_input->strides()[reduction_dim_u];
            const bool column_tiled = stride != 1;
            const ReduceGeometry geometry = ComputeReduceGeometry(reduce_input, reduction_dim_u);

            const std::shared_ptr<pi::tensorlib::RealTensor> reduce_output = output;

#if PI_TENSORLIB_ENABLE_CUDA
            if (ShouldUseSplitColumnReduce(column_tiled, geometry))
            {
                LowerSplitMulReduce(execution_plan, reduce_idx, reduction_dim, keepdim, dtype, reduce_input, mul_lhs,
                                    mul_rhs, reduce_output, geometry);
            }
            else
#endif
            {
                auto &reduce_entry = execution_plan.entries[reduce_idx];
                reduce_entry.inputs = {mul_lhs, mul_rhs};
                reduce_entry.outputs[0] = reduce_output;
                reduce_entry.op_type = std::nullopt;
                reduce_entry.kernel_descriptor =
                    CreateMulReduceComputeKernelDescriptor(reduction_dim, keepdim, dtype, column_tiled);
                reduce_entry.flop_estimate = reduce_output->shape().numel() * reduce_input->shape()[reduction_dim_u];
            }

            if (debug_mul_reduce)
            {
                const auto reduce_entry_it = std::ranges::find_if(
                    execution_plan.entries,
                    [reduce_entry_id](const pi::tensorlib::ExecutionEntry &e) { return e.id == reduce_entry_id; });
                if (reduce_entry_it != execution_plan.entries.end() && reduce_entry_it->kernel_descriptor.has_value())
                {
                    std::cout << "[MulReduce] Fusing mul+reduce into "
                              << reduce_entry_it->kernel_descriptor->kernel_name << " for entry id "
                              << reduce_entry_it->id << '\n'
                              << std::flush;
                }
            }

            RemoveTensorCreateDeleteEntries(execution_plan, mul_out_id);
            RemoveEntryById(execution_plan, mul_entry_id);
            changed = true;
            break;
        }
    }
}
