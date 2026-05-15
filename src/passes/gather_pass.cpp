#include "passes.h"

#include "passes/pass_utils.h"
#include <kernels/kernel_binaries.h>

#include <limits>
#include <string>

static pi::tensorlib::ComputeKernelDescriptor
CreateEmbeddingLookupComputeKernelDescriptor(const pi::tensorlib::DataType dtype)
{
    const auto kernel = SelectKernelForHalf(kembedding_lookup_bf16, kembedding_lookup_fp16, dtype);
    const std::string kernel_name = std::string("embedding_lookup_") + std::string(KernelSuffixForHalf(dtype));

    return pi::tensorlib::ComputeKernelDescriptor{
        .kernel_name = kernel_name,
        .function_name = kernel.function_name,
        .expected_arg_count = kernel.arg_count,
        .argument_provider = [kernel, dtype](const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                                             const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
            -> pi::tensorlib::KernelLaunchArguments
        {
            if (inputs.size() != 2)
            {
                throw std::runtime_error("embedding_lookup_bf16 expects two inputs");
            }
            if (outputs.size() != 1)
            {
                throw std::runtime_error("embedding_lookup_bf16 expects a single output");
            }

            const auto &table = inputs[0];
            const auto &indices = inputs[1];
            const auto &output = outputs[0];

            if (table->dtype() != dtype || output->dtype() != dtype)
            {
                throw std::runtime_error("embedding_lookup requires BFLOAT16 or FLOAT16 weight and output tensors");
            }
            if (indices->dtype() != pi::tensorlib::DataType::UINT32)
            {
                throw std::runtime_error("embedding_lookup expects UINT32 indices");
            }
            if (table->shape().ndims() != 2)
            {
                throw std::runtime_error("embedding_lookup weight must be rank-2");
            }

            if (table->device().device_type != pi::tensorlib::DeviceType::GPU ||
                indices->device().device_type != pi::tensorlib::DeviceType::GPU ||
                output->device().device_type != pi::tensorlib::DeviceType::GPU)
            {
                throw std::runtime_error("embedding_lookup requires GPU tensors");
            }

            const auto device_ordinal =
                ValidateSameDeviceOrdinal("embedding_lookup", {table, indices, output});

            const auto embedding_dim_64 = table->shape()[1];
            if (embedding_dim_64 == 0)
            {
                throw std::runtime_error("embedding_lookup embedding dimension must be > 0");
            }
            if (embedding_dim_64 > std::numeric_limits<uint32_t>::max())
            {
                throw std::runtime_error("embedding_lookup embedding dimension exceeds supported range");
            }
            const auto embedding_dim = static_cast<uint32_t>(embedding_dim_64);

            const auto num_indices_64 = indices->shape().numel();
            if (num_indices_64 == 0)
            {
                throw std::runtime_error("embedding_lookup requires at least one index");
            }
            if (num_indices_64 > std::numeric_limits<uint32_t>::max())
            {
                throw std::runtime_error("embedding_lookup index count exceeds supported range");
            }
            const auto num_indices = static_cast<uint32_t>(num_indices_64);

            const auto output_elements = output->shape().numel();
            if (output_elements != num_indices_64 * embedding_dim_64)
            {
                throw std::runtime_error("embedding_lookup output shape mismatch");
            }

            if (indices->shape().ndims() == 0)
            {
                throw std::runtime_error("embedding_lookup indices rank must be >= 1");
            }
            if (output->shape().ndims() == 0)
            {
                throw std::runtime_error("embedding_lookup output rank must be >= 1");
            }

            if (indices->strides()[indices->shape().ndims() - 1] != 1)
            {
                throw std::runtime_error("embedding_lookup expects contiguous indices tensor");
            }
            if (table->strides()[1] != 1)
            {
                throw std::runtime_error("embedding_lookup expects column-contiguous weight tensor");
            }
            if (output->strides()[output->shape().ndims() - 1] != 1)
            {
                throw std::runtime_error("embedding_lookup expects the last output dimension to be contiguous");
            }

            const auto stride_table_row_64 = table->strides()[0];
            if (stride_table_row_64 > std::numeric_limits<uint32_t>::max())
            {
                throw std::runtime_error("embedding_lookup table row stride exceeds supported range");
            }
            const auto stride_table_row = static_cast<uint32_t>(stride_table_row_64);
            const auto stride_table_col = static_cast<uint32_t>(table->strides()[1]);

            const auto stride_out_row = embedding_dim;
            const auto stride_out_col = static_cast<uint32_t>(output->strides()[output->shape().ndims() - 1]);

            void *output_ptr = output->dataptr();
            void *table_ptr = table->dataptr();
            void *indices_ptr = indices->dataptr();

            pi::tensorlib::KernelLaunchArguments arguments{
                .args = {output_ptr, table_ptr, indices_ptr, num_indices, embedding_dim, stride_out_row, stride_out_col,
                         stride_table_row, stride_table_col, static_cast<void *>(nullptr)},
                .grid_dim_x = num_indices,
                .grid_dim_y = CEIL_DIV(embedding_dim, kernel.meta.block_size),
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

void GatherImplPass::transform(pi::tensorlib::ExecutionPlan &execution_plan)
{
    for (auto &entry : execution_plan.entries)
    {
        if (entry.op_type != pi::tensorlib::OpType::GATHER)
        {
            continue;
        }
        if (entry.inputs.size() != 2)
        {
            continue;
        }
        if (entry.outputs.size() != 1)
        {
            continue;
        }

        const auto &table = entry.inputs[0];
        const auto &indices = entry.inputs[1];
        const auto &output = entry.outputs[0];

        const auto dtype = table->dtype();
        if (!IsSupportedHalfType(dtype) || output->dtype() != dtype)
        {
            continue;
        }
        if (indices->dtype() != pi::tensorlib::DataType::UINT32)
        {
            continue;
        }

        entry.op_type = std::nullopt;
        entry.kernel_descriptor = CreateEmbeddingLookupComputeKernelDescriptor(dtype);
        entry.flop_estimate = 0; // memory-bound gather
    }
}
