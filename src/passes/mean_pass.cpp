#include "passes.h"
#include "shape_utils.h"

#include <kernels/kernel_binaries.h>
#include "passes/pass_utils.h"

#include <limits>

static pi::tensorlib::ComputeKernelDescriptor
CreateMeanComputeKernelDescriptor(const int64_t reduction_dim, const bool keepdim,
                                  const pi::tensorlib::DataType dtype, const bool column_tiled)
{
    const auto kernel = column_tiled ? SelectKernelForHalf(kmean_reduce_column_tiled_bf16,
                                                           kmean_reduce_column_tiled_fp16, dtype)
                                     : SelectKernelForHalf(kmean_reduce_contiguous_bf16,
                                                           kmean_reduce_contiguous_fp16, dtype);
    const std::string kernel_name = std::string(column_tiled ? "mean_reduce_column_tiled_" : "mean_reduce_contiguous_") +
                                    std::string(KernelSuffixForHalf(dtype));
    const std::string dtype_name = pi::tensorlib::GetDataTypeName(dtype);

    return pi::tensorlib::ComputeKernelDescriptor{
        .kernel_name = kernel_name,
        .function_name = kernel.function_name,
        .expected_arg_count = kernel.arg_count,
        .argument_provider = [reduction_dim, keepdim, kernel, dtype, dtype_name, column_tiled](
                                 const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                                 const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
            -> pi::tensorlib::KernelLaunchArguments
        {
            if (inputs.size() != 1 || outputs.size() != 1)
            {
                throw std::runtime_error("mean operation expects exactly one input and one output tensor");
            }

            const auto &input = inputs[0];
            const auto &output = outputs[0];

            if (input->dtype() != dtype || output->dtype() != dtype)
            {
                throw std::runtime_error("mean operation currently supports " + std::string(dtype_name) + " tensors only");
            }

            if (input->device().device_type != pi::tensorlib::DeviceType::GPU ||
                output->device().device_type != pi::tensorlib::DeviceType::GPU)
            {
                throw std::runtime_error("mean operation requires GPU tensors on the same device");
            }
            const auto device_ordinal = input->device().ordinal;
            if (output->device().ordinal != device_ordinal)
            {
                throw std::runtime_error("mean operation requires GPU tensors on the same device");
            }

            const auto input_ndims = static_cast<int64_t>(input->shape().ndims());
            if (input_ndims == 0)
            {
                throw std::runtime_error("mean operation does not support scalar inputs");
            }
            int64_t reduction_dim_norm = reduction_dim;
            if (reduction_dim_norm < 0)
            {
                reduction_dim_norm += input_ndims;
            }
            if (reduction_dim_norm < 0 || reduction_dim_norm >= input_ndims)
            {
                throw std::runtime_error("mean operation reduction dimension is out of bounds");
            }
            const auto reduction_dim_u = static_cast<size_t>(reduction_dim_norm);

            const auto cols64 = input->shape()[reduction_dim_u];
            if (cols64 == 0)
            {
                throw std::runtime_error("mean operation requires non-zero reduction dimension size");
            }

            const auto total_elements = input->shape().numel();
            if (total_elements % cols64 != 0)
            {
                throw std::runtime_error("mean operation received incompatible shape for reduction");
            }

            uint64_t outer64 = 1;
            for (size_t dim = 0; dim < reduction_dim_u; ++dim)
            {
                outer64 *= input->shape()[dim];
            }
            uint64_t inner64 = 1;
            for (size_t dim = reduction_dim_u + 1; dim < input->shape().ndims(); ++dim)
            {
                inner64 *= input->shape()[dim];
            }

            const auto rows64 = outer64 * inner64;
            if (rows64 == 0 || rows64 != total_elements / cols64)
            {
                throw std::runtime_error("mean operation computed invalid output shape");
            }

            if (outer64 > std::numeric_limits<uint32_t>::max() || inner64 > std::numeric_limits<uint32_t>::max() ||
                cols64 > std::numeric_limits<uint32_t>::max())
            {
                throw std::runtime_error("mean operation exceeds supported tensor size");
            }

            if (!pi::tensorlib::shape_utils::IsRowMajorContiguous(input->shape(), input->strides()) || !pi::tensorlib::shape_utils::IsRowMajorContiguous(output->shape(), output->strides()))
            {
                throw std::runtime_error("mean operation requires contiguous tensors");
            }

            if (keepdim)
            {
                if (output->shape().ndims() != input->shape().ndims())
                {
                    throw std::runtime_error("mean keepdim output must match input rank");
                }
                for (size_t dim = 0; dim < input->shape().ndims(); ++dim)
                {
                    const auto expected = (dim == reduction_dim_u) ? uint64_t{1} : input->shape()[dim];
                    if (output->shape()[dim] != expected)
                    {
                        throw std::runtime_error("mean keepdim output shape mismatch");
                    }
                }
            }
            else
            {
                if (output->shape().ndims() + 1 != input->shape().ndims())
                {
                    throw std::runtime_error("mean output rank should be input rank - 1 when keepdim is false");
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
                        throw std::runtime_error("mean output shape mismatch when keepdim is false");
                    }
                    ++out_dim;
                }
            }

            void *output_ptr = output->dataptr();
            void *input_ptr = input->dataptr();

            const auto outer = static_cast<uint32_t>(outer64);
            const auto inner = static_cast<uint32_t>(inner64);
            const auto cols = static_cast<uint32_t>(cols64);
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

void MeanImplPass::transform(pi::tensorlib::ExecutionPlan &execution_plan)
{
    for (auto &entry : execution_plan.entries)
    {
        if (entry.op_type != pi::tensorlib::OpType::MEAN)
        {
            continue;
        }

        if (entry.inputs.size() != 1 || entry.outputs.size() != 1)
        {
            throw std::runtime_error("mean operation expects exactly one input and one output tensor");
        }

        const auto dim_it = entry.attributes.find("dim");
        const auto keepdim_it = entry.attributes.find("keepdim");
        if (dim_it == entry.attributes.end() || keepdim_it == entry.attributes.end())
        {
            throw std::runtime_error("mean operation is missing required attributes");
        }

        auto reduction_dim = std::any_cast<int64_t>(dim_it->second);
        const auto input_ndims = static_cast<int64_t>(entry.inputs[0]->shape().ndims());
        if (reduction_dim < 0)
        {
            reduction_dim += input_ndims;
        }
        const auto keepdim = std::any_cast<bool>(keepdim_it->second);

        const auto &input = entry.inputs[0];
        const auto &output = entry.outputs[0];
        const auto dtype = input->dtype();

        if (!IsSupportedHalfType(dtype) || output->dtype() != dtype)
        {
            continue;
        }

        const auto reduction_dim_u = static_cast<size_t>(reduction_dim);
        const auto stride = input->strides()[reduction_dim_u];
        const bool column_tiled = stride != 1;

        entry.op_type = std::nullopt;
        entry.kernel_descriptor = CreateMeanComputeKernelDescriptor(reduction_dim, keepdim, dtype, column_tiled);
        // flops: sum (cols-1 adds) + divide per element -> approx (cols) per output element
        const auto reduced_dim = static_cast<size_t>(reduction_dim);
        entry.flop_estimate =
            static_cast<uint64_t>(output->shape().numel()) * static_cast<uint64_t>(input->shape()[reduced_dim]);
    }
}
