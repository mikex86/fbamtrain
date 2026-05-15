#include "passes.h"

#include "passes/pass_utils.h"
#include <kernels/kernel_binaries.h>

#include <cstdlib>
#include <iostream>
#include <limits>
#include <unordered_set>

namespace
{
    void DebugCastLaunch(const std::shared_ptr<pi::tensorlib::RealTensor> &input,
                         const std::shared_ptr<pi::tensorlib::RealTensor> &output,
                         const void *output_ptr, const uint64_t numel)
    {
        if (const char *env = std::getenv("FBAMTRAIN_DEBUG_CAST_LAUNCH"); env != nullptr && env[0] != '\0')
        {
            std::clog << "[CastLaunch] in_id=" << input->id() << " out_id=" << output->id()
                      << " out_ptr=" << output_ptr << " numel=" << numel << '\n';
        }
    }
} // namespace

static pi::tensorlib::ComputeKernelDescriptor CreateCastBf16ToFp32KernelDescriptor()
{
    return pi::tensorlib::ComputeKernelDescriptor{
        .kernel_name = "cast_bf16_to_fp32",
        .function_name = kcast_bf16_to_fp32.function_name,
        .expected_arg_count = kcast_bf16_to_fp32.arg_count,
        .argument_provider = [](const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                                const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
            -> pi::tensorlib::KernelLaunchArguments
        {
            if (inputs.size() != 1 || outputs.size() != 1)
            {
                throw std::runtime_error("cast_bf16_to_fp32 expects exactly one input and one output tensor");
            }

            const auto &input = inputs[0];
            const auto &output = outputs[0];

            if (input->dtype() != pi::tensorlib::DataType::BFLOAT16 ||
                output->dtype() != pi::tensorlib::DataType::FLOAT32)
            {
                throw std::runtime_error("cast_bf16_to_fp32 requires BF16 input and FP32 output tensors");
            }
            if (input->device().device_type != pi::tensorlib::DeviceType::GPU ||
                output->device().device_type != pi::tensorlib::DeviceType::GPU)
            {
                throw std::runtime_error("cast_bf16_to_fp32 currently supports GPU tensors only");
            }

            const auto numel_input = input->shape().numel();
            const auto numel_output = output->shape().numel();
            if (numel_input != numel_output)
            {
                throw std::runtime_error("cast_bf16_to_fp32 requires input and output tensors with matching sizes");
            }
            if (numel_input > std::numeric_limits<uint32_t>::max())
            {
                throw std::runtime_error("cast_bf16_to_fp32 supports up to 2^32-1 elements");
            }

            const auto device_ordinal =
                ValidateSameDeviceOrdinal("cast_bf16_to_fp32", {input, output});

            auto *input_ptr = input->dataptr();
            auto *output_ptr = output->dataptr();
            const auto n_elements = static_cast<uint32_t>(numel_input);

            DebugCastLaunch(input, output, output_ptr, numel_input);

            pi::tensorlib::KernelLaunchArguments arguments{
                .args = {input_ptr, output_ptr, n_elements, static_cast<void *>(nullptr)},
                .grid_dim_x = CEIL_DIV(n_elements, kcast_bf16_to_fp32.meta.block_size),
                .grid_dim_y = 1,
                .grid_dim_z = 1,
                .block_dim_x = TRITON_WARP_SIZE * kcast_bf16_to_fp32.num_warps,
                .block_dim_y = 1,
                .block_dim_z = 1,
                .shared_mem_bytes = kcast_bf16_to_fp32.shared_mem_bytes,
                .device_ordinal = device_ordinal,
            };
            return arguments;
        },
        .cuda_descriptor = pi::tensorlib::MakeCudaKernelDescriptor(kcast_bf16_to_fp32.data, kcast_bf16_to_fp32.size),
        .hip_descriptor = pi::tensorlib::MakeHipKernelDescriptor(kcast_bf16_to_fp32.data, kcast_bf16_to_fp32.size),
    };
}

static pi::tensorlib::ComputeKernelDescriptor CreateCastFp32ToBf16KernelDescriptor()
{
    return pi::tensorlib::ComputeKernelDescriptor{
        .kernel_name = "cast_fp32_to_bf16",
        .function_name = kcast_fp32_to_bf16.function_name,
        .expected_arg_count = kcast_fp32_to_bf16.arg_count,
        .argument_provider = [](const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                                const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
            -> pi::tensorlib::KernelLaunchArguments
        {
            if (inputs.size() != 1 || outputs.size() != 1)
            {
                throw std::runtime_error("cast_fp32_to_bf16 expects exactly one input and one output tensor");
            }

            const auto &input = inputs[0];
            const auto &output = outputs[0];

            if (input->dtype() != pi::tensorlib::DataType::FLOAT32 ||
                output->dtype() != pi::tensorlib::DataType::BFLOAT16)
            {
                throw std::runtime_error("cast_fp32_to_bf16 requires FP32 input and BF16 output tensors");
            }
            if (input->device().device_type != pi::tensorlib::DeviceType::GPU ||
                output->device().device_type != pi::tensorlib::DeviceType::GPU)
            {
                throw std::runtime_error("cast_fp32_to_bf16 currently supports GPU tensors only");
            }

            const auto numel_input = input->shape().numel();
            const auto numel_output = output->shape().numel();
            if (numel_input != numel_output)
            {
                throw std::runtime_error("cast_fp32_to_bf16 requires input and output tensors with matching sizes");
            }
            if (numel_input > std::numeric_limits<uint32_t>::max())
            {
                throw std::runtime_error("cast_fp32_to_bf16 supports up to 2^32-1 elements");
            }

            const auto device_ordinal =
                ValidateSameDeviceOrdinal("cast_fp32_to_bf16", {input, output});

            auto *input_ptr = input->dataptr();
            auto *output_ptr = output->dataptr();
            const auto n_elements = static_cast<uint32_t>(numel_input);

            DebugCastLaunch(input, output, output_ptr, numel_input);

            pi::tensorlib::KernelLaunchArguments arguments{
                .args = {input_ptr, output_ptr, n_elements, static_cast<void *>(nullptr)},
                .grid_dim_x = CEIL_DIV(n_elements, kcast_fp32_to_bf16.meta.block_size),
                .grid_dim_y = 1,
                .grid_dim_z = 1,
                .block_dim_x = TRITON_WARP_SIZE * kcast_fp32_to_bf16.num_warps,
                .block_dim_y = 1,
                .block_dim_z = 1,
                .shared_mem_bytes = kcast_fp32_to_bf16.shared_mem_bytes,
                .device_ordinal = device_ordinal,
            };
            return arguments;
        },
        .cuda_descriptor = pi::tensorlib::MakeCudaKernelDescriptor(kcast_fp32_to_bf16.data, kcast_fp32_to_bf16.size),
        .hip_descriptor = pi::tensorlib::MakeHipKernelDescriptor(kcast_fp32_to_bf16.data, kcast_fp32_to_bf16.size),
    };
}

static pi::tensorlib::ComputeKernelDescriptor CreateCastBf16ToFp16KernelDescriptor()
{
    return pi::tensorlib::ComputeKernelDescriptor{
        .kernel_name = "cast_bf16_to_fp16",
        .function_name = kcast_bf16_to_fp16.function_name,
        .expected_arg_count = kcast_bf16_to_fp16.arg_count,
        .argument_provider = [](const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                                const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
            -> pi::tensorlib::KernelLaunchArguments
        {
            if (inputs.size() != 1 || outputs.size() != 1)
            {
                throw std::runtime_error("cast_bf16_to_fp16 expects exactly one input and one output tensor");
            }

            const auto &input = inputs[0];
            const auto &output = outputs[0];

            if (input->dtype() != pi::tensorlib::DataType::BFLOAT16 ||
                output->dtype() != pi::tensorlib::DataType::FLOAT16)
            {
                throw std::runtime_error("cast_bf16_to_fp16 requires BF16 input and FP16 output tensors");
            }
            if (input->device().device_type != pi::tensorlib::DeviceType::GPU ||
                output->device().device_type != pi::tensorlib::DeviceType::GPU)
            {
                throw std::runtime_error("cast_bf16_to_fp16 currently supports GPU tensors only");
            }

            const auto numel_input = input->shape().numel();
            const auto numel_output = output->shape().numel();
            if (numel_input != numel_output)
            {
                throw std::runtime_error("cast_bf16_to_fp16 requires input and output tensors with matching sizes");
            }
            if (numel_input > std::numeric_limits<uint32_t>::max())
            {
                throw std::runtime_error("cast_bf16_to_fp16 supports up to 2^32-1 elements");
            }

            const auto device_ordinal =
                ValidateSameDeviceOrdinal("cast_bf16_to_fp16", {input, output});

            auto *input_ptr = input->dataptr();
            auto *output_ptr = output->dataptr();
            const auto n_elements = static_cast<uint32_t>(numel_input);

            DebugCastLaunch(input, output, output_ptr, numel_input);

            pi::tensorlib::KernelLaunchArguments arguments{
                .args = {input_ptr, output_ptr, n_elements, static_cast<void *>(nullptr)},
                .grid_dim_x = CEIL_DIV(n_elements, kcast_bf16_to_fp16.meta.block_size),
                .grid_dim_y = 1,
                .grid_dim_z = 1,
                .block_dim_x = TRITON_WARP_SIZE * kcast_bf16_to_fp16.num_warps,
                .block_dim_y = 1,
                .block_dim_z = 1,
                .shared_mem_bytes = kcast_bf16_to_fp16.shared_mem_bytes,
                .device_ordinal = device_ordinal,
            };
            return arguments;
        },
        .cuda_descriptor = pi::tensorlib::MakeCudaKernelDescriptor(kcast_bf16_to_fp16.data, kcast_bf16_to_fp16.size),
        .hip_descriptor = pi::tensorlib::MakeHipKernelDescriptor(kcast_bf16_to_fp16.data, kcast_bf16_to_fp16.size),
    };
}

static pi::tensorlib::ComputeKernelDescriptor CreateCastFp16ToBf16KernelDescriptor()
{
    return pi::tensorlib::ComputeKernelDescriptor{
        .kernel_name = "cast_fp16_to_bf16",
        .function_name = kcast_fp16_to_bf16.function_name,
        .expected_arg_count = kcast_fp16_to_bf16.arg_count,
        .argument_provider = [](const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                                const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
            -> pi::tensorlib::KernelLaunchArguments
        {
            if (inputs.size() != 1 || outputs.size() != 1)
            {
                throw std::runtime_error("cast_fp16_to_bf16 expects exactly one input and one output tensor");
            }

            const auto &input = inputs[0];
            const auto &output = outputs[0];

            if (input->dtype() != pi::tensorlib::DataType::FLOAT16 ||
                output->dtype() != pi::tensorlib::DataType::BFLOAT16)
            {
                throw std::runtime_error("cast_fp16_to_bf16 requires FP16 input and BF16 output tensors");
            }
            if (input->device().device_type != pi::tensorlib::DeviceType::GPU ||
                output->device().device_type != pi::tensorlib::DeviceType::GPU)
            {
                throw std::runtime_error("cast_fp16_to_bf16 currently supports GPU tensors only");
            }

            const auto numel_input = input->shape().numel();
            const auto numel_output = output->shape().numel();
            if (numel_input != numel_output)
            {
                throw std::runtime_error("cast_fp16_to_bf16 requires input and output tensors with matching sizes");
            }
            if (numel_input > std::numeric_limits<uint32_t>::max())
            {
                throw std::runtime_error("cast_fp16_to_bf16 supports up to 2^32-1 elements");
            }

            const auto device_ordinal =
                ValidateSameDeviceOrdinal("cast_fp16_to_bf16", {input, output});

            auto *input_ptr = input->dataptr();
            auto *output_ptr = output->dataptr();
            const auto n_elements = static_cast<uint32_t>(numel_input);

            DebugCastLaunch(input, output, output_ptr, numel_input);

            pi::tensorlib::KernelLaunchArguments arguments{
                .args = {input_ptr, output_ptr, n_elements, static_cast<void *>(nullptr)},
                .grid_dim_x = CEIL_DIV(n_elements, kcast_fp16_to_bf16.meta.block_size),
                .grid_dim_y = 1,
                .grid_dim_z = 1,
                .block_dim_x = TRITON_WARP_SIZE * kcast_fp16_to_bf16.num_warps,
                .block_dim_y = 1,
                .block_dim_z = 1,
                .shared_mem_bytes = kcast_fp16_to_bf16.shared_mem_bytes,
                .device_ordinal = device_ordinal,
            };
            return arguments;
        },
        .cuda_descriptor = pi::tensorlib::MakeCudaKernelDescriptor(kcast_fp16_to_bf16.data, kcast_fp16_to_bf16.size),
        .hip_descriptor = pi::tensorlib::MakeHipKernelDescriptor(kcast_fp16_to_bf16.data, kcast_fp16_to_bf16.size),
    };
}

static pi::tensorlib::ComputeKernelDescriptor CreateCastFp16ToFp32KernelDescriptor()
{
    return pi::tensorlib::ComputeKernelDescriptor{
        .kernel_name = "cast_fp16_to_fp32",
        .function_name = kcast_fp16_to_fp32.function_name,
        .expected_arg_count = kcast_fp16_to_fp32.arg_count,
        .argument_provider = [](const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                                const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
            -> pi::tensorlib::KernelLaunchArguments
        {
            if (inputs.size() != 1 || outputs.size() != 1)
            {
                throw std::runtime_error("cast_fp16_to_fp32 expects exactly one input and one output tensor");
            }

            const auto &input = inputs[0];
            const auto &output = outputs[0];

            if (input->dtype() != pi::tensorlib::DataType::FLOAT16 ||
                output->dtype() != pi::tensorlib::DataType::FLOAT32)
            {
                throw std::runtime_error("cast_fp16_to_fp32 requires FP16 input and FP32 output tensors");
            }
            if (input->device().device_type != pi::tensorlib::DeviceType::GPU ||
                output->device().device_type != pi::tensorlib::DeviceType::GPU)
            {
                throw std::runtime_error("cast_fp16_to_fp32 currently supports GPU tensors only");
            }

            const auto numel_input = input->shape().numel();
            const auto numel_output = output->shape().numel();
            if (numel_input != numel_output)
            {
                throw std::runtime_error("cast_fp16_to_fp32 requires input and output tensors with matching sizes");
            }
            if (numel_input > std::numeric_limits<uint32_t>::max())
            {
                throw std::runtime_error("cast_fp16_to_fp32 supports up to 2^32-1 elements");
            }

            const auto device_ordinal =
                ValidateSameDeviceOrdinal("cast_fp16_to_fp32", {input, output});

            auto *input_ptr = input->dataptr();
            auto *output_ptr = output->dataptr();
            const auto n_elements = static_cast<uint32_t>(numel_input);

            DebugCastLaunch(input, output, output_ptr, numel_input);

            pi::tensorlib::KernelLaunchArguments arguments{
                .args = {input_ptr, output_ptr, n_elements, static_cast<void *>(nullptr)},
                .grid_dim_x = CEIL_DIV(n_elements, kcast_fp16_to_fp32.meta.block_size),
                .grid_dim_y = 1,
                .grid_dim_z = 1,
                .block_dim_x = TRITON_WARP_SIZE * kcast_fp16_to_fp32.num_warps,
                .block_dim_y = 1,
                .block_dim_z = 1,
                .shared_mem_bytes = kcast_fp16_to_fp32.shared_mem_bytes,
                .device_ordinal = device_ordinal,
            };
            return arguments;
        },
        .cuda_descriptor = pi::tensorlib::MakeCudaKernelDescriptor(kcast_fp16_to_fp32.data, kcast_fp16_to_fp32.size),
        .hip_descriptor = pi::tensorlib::MakeHipKernelDescriptor(kcast_fp16_to_fp32.data, kcast_fp16_to_fp32.size),
    };
}

static pi::tensorlib::ComputeKernelDescriptor CreateCastFp32ToFp16KernelDescriptor()
{
    return pi::tensorlib::ComputeKernelDescriptor{
        .kernel_name = "cast_fp32_to_fp16",
        .function_name = kcast_fp32_to_fp16.function_name,
        .expected_arg_count = kcast_fp32_to_fp16.arg_count,
        .argument_provider = [](const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                                const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
            -> pi::tensorlib::KernelLaunchArguments
        {
            if (inputs.size() != 1 || outputs.size() != 1)
            {
                throw std::runtime_error("cast_fp32_to_fp16 expects exactly one input and one output tensor");
            }

            const auto &input = inputs[0];
            const auto &output = outputs[0];

            if (input->dtype() != pi::tensorlib::DataType::FLOAT32 ||
                output->dtype() != pi::tensorlib::DataType::FLOAT16)
            {
                throw std::runtime_error("cast_fp32_to_fp16 requires FP32 input and FP16 output tensors");
            }
            if (input->device().device_type != pi::tensorlib::DeviceType::GPU ||
                output->device().device_type != pi::tensorlib::DeviceType::GPU)
            {
                throw std::runtime_error("cast_fp32_to_fp16 currently supports GPU tensors only");
            }

            const auto numel_input = input->shape().numel();
            const auto numel_output = output->shape().numel();
            if (numel_input != numel_output)
            {
                throw std::runtime_error("cast_fp32_to_fp16 requires input and output tensors with matching sizes");
            }
            if (numel_input > std::numeric_limits<uint32_t>::max())
            {
                throw std::runtime_error("cast_fp32_to_fp16 supports up to 2^32-1 elements");
            }

            const auto device_ordinal =
                ValidateSameDeviceOrdinal("cast_fp32_to_fp16", {input, output});

            auto *input_ptr = input->dataptr();
            auto *output_ptr = output->dataptr();
            const auto n_elements = static_cast<uint32_t>(numel_input);

            DebugCastLaunch(input, output, output_ptr, numel_input);

            pi::tensorlib::KernelLaunchArguments arguments{
                .args = {input_ptr, output_ptr, n_elements, static_cast<void *>(nullptr)},
                .grid_dim_x = CEIL_DIV(n_elements, kcast_fp32_to_fp16.meta.block_size),
                .grid_dim_y = 1,
                .grid_dim_z = 1,
                .block_dim_x = TRITON_WARP_SIZE * kcast_fp32_to_fp16.num_warps,
                .block_dim_y = 1,
                .block_dim_z = 1,
                .shared_mem_bytes = kcast_fp32_to_fp16.shared_mem_bytes,
                .device_ordinal = device_ordinal,
            };
            return arguments;
        },
        .cuda_descriptor = pi::tensorlib::MakeCudaKernelDescriptor(kcast_fp32_to_fp16.data, kcast_fp32_to_fp16.size),
        .hip_descriptor = pi::tensorlib::MakeHipKernelDescriptor(kcast_fp32_to_fp16.data, kcast_fp32_to_fp16.size),
    };
}
void CastImplPass::transform(pi::tensorlib::ExecutionPlan &execution_plan)
{
    size_t next_entry_id = 0;
    for (const auto &entry : execution_plan.entries)
    {
        next_entry_id = std::max(next_entry_id, entry.id + 1);
    }

    std::unordered_set<uint64_t> created_ids{};
    for (const auto &entry : execution_plan.entries)
    {
        if (entry.op_type == pi::tensorlib::OpType::CREATE_TENSOR && entry.outputs.size() == 1 && entry.outputs[0])
        {
            created_ids.insert(entry.outputs[0]->id());
        }
    }

    // TODO: Review why the fuck this is necessary (I can't imagine it is...)
    auto ensure_create = [&](const size_t insert_idx, const std::shared_ptr<pi::tensorlib::RealTensor> &output,
                             const bool is_useful) -> bool
    {
        if (!output || output->isView())
        {
            return false;
        }
        if (created_ids.contains(output->id()))
        {
            return false;
        }
        pi::tensorlib::ExecutionEntry create_entry{};
        create_entry.id = next_entry_id++;
        create_entry.op_type = pi::tensorlib::OpType::CREATE_TENSOR;
        create_entry.outputs = {output};
        create_entry.is_useful = is_useful;
        execution_plan.entries.insert(execution_plan.entries.begin() + static_cast<std::ptrdiff_t>(insert_idx),
                                      std::move(create_entry));
        created_ids.insert(output->id());
        return true;
    };

    size_t i = 0;
    while (i < execution_plan.entries.size())
    {
        if (execution_plan.entries[i].op_type != pi::tensorlib::OpType::CAST)
        {
            ++i;
            continue;
        }
        if (execution_plan.entries[i].inputs.size() != 1 || execution_plan.entries[i].outputs.size() != 1)
        {
            ++i;
            continue;
        }

        const auto input = execution_plan.entries[i].inputs[0];
        const auto output = execution_plan.entries[i].outputs[0];
        size_t entry_idx = i;
        if (output && input->id() != output->id())
        {
            if (ensure_create(i, output, execution_plan.entries[i].is_useful))
            {
                ++i;
                entry_idx = i;
            }
        }

        auto &entry = execution_plan.entries[entry_idx];
        const auto src_dtype = input->dtype();
        const auto dst_dtype = output->dtype();

        if (src_dtype == dst_dtype)
        {
            // TODO: This is dirty and horrible and sh
            entry.op_type = pi::tensorlib::OpType::DEVICE_COPY;
            if (!entry.gpu_stream_desc.isValid())
            {
                entry.gpu_stream_desc = pi::tensorlib::GpuStreamDescriptors::Main;
            }
            entry.flop_estimate = 0;
            ++i;
            continue;
        }

        if (src_dtype == pi::tensorlib::DataType::BFLOAT16 && dst_dtype == pi::tensorlib::DataType::FLOAT32)
        {
            entry.op_type = std::nullopt;
            entry.kernel_descriptor = CreateCastBf16ToFp32KernelDescriptor();
            entry.flop_estimate = 0;
        }
        else if (src_dtype == pi::tensorlib::DataType::FLOAT32 && dst_dtype == pi::tensorlib::DataType::BFLOAT16)
        {
            entry.op_type = std::nullopt;
            entry.kernel_descriptor = CreateCastFp32ToBf16KernelDescriptor();
            entry.flop_estimate = 0;
        }
        else if (src_dtype == pi::tensorlib::DataType::BFLOAT16 && dst_dtype == pi::tensorlib::DataType::FLOAT16)
        {
            entry.op_type = std::nullopt;
            entry.kernel_descriptor = CreateCastBf16ToFp16KernelDescriptor();
            entry.flop_estimate = 0;
        }
        else if (src_dtype == pi::tensorlib::DataType::FLOAT16 && dst_dtype == pi::tensorlib::DataType::BFLOAT16)
        {
            entry.op_type = std::nullopt;
            entry.kernel_descriptor = CreateCastFp16ToBf16KernelDescriptor();
            entry.flop_estimate = 0;
        }
        else if (src_dtype == pi::tensorlib::DataType::FLOAT16 && dst_dtype == pi::tensorlib::DataType::FLOAT32)
        {
            entry.op_type = std::nullopt;
            entry.kernel_descriptor = CreateCastFp16ToFp32KernelDescriptor();
            entry.flop_estimate = 0;
        }
        else if (src_dtype == pi::tensorlib::DataType::FLOAT32 && dst_dtype == pi::tensorlib::DataType::FLOAT16)
        {
            entry.op_type = std::nullopt;
            entry.kernel_descriptor = CreateCastFp32ToFp16KernelDescriptor();
            entry.flop_estimate = 0;
        } else
        {
            throw std::runtime_error("Unsupported cast operation: from " +
                                     pi::tensorlib::GetDataTypeName(src_dtype) + " to " +
                                     pi::tensorlib::GetDataTypeName(dst_dtype));
        }
        ++i;
    }
}
