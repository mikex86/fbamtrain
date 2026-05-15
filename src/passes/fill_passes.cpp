#include "passes.h"

#include "passes/pass_utils.h"
#include <kernels/kernel_binaries.h>

#include <any>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

static pi::tensorlib::ComputeKernelDescriptor CreateFillZerosComputeKernelDescriptor()
{
    kernel_bin_t<kernel_meta_elementwise_t> kernel = kfill_zeros;

    return pi::tensorlib::ComputeKernelDescriptor{
        .kernel_name = "fill_zeros",
        .function_name = kfill_zeros.function_name,
        .expected_arg_count = kernel.arg_count,
        .argument_provider = [kernel](const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                                      const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
            -> pi::tensorlib::KernelLaunchArguments
        {
            if (inputs.size() != 1)
            {
                throw std::runtime_error("fill_zeros expects exactly one input tensor");
            }
            if (outputs.size() != 1)
            {
                throw std::runtime_error("fill_zeros expects exactly one output tensor");
            }
            if (inputs[0]->id() != outputs[0]->id())
            {
                throw std::runtime_error("fill_zeros is inplace and requires identical input/output tensors");
            }

            const auto &out = outputs[0];
            if (out->device().device_type != pi::tensorlib::DeviceType::GPU)
            {
                throw std::runtime_error("fill_zeros currently supports GPU tensors only");
            }

            const auto device_ordinal = ValidateSameDeviceOrdinal("fill_zeros", {out});
            const size_t bytes_per_element = pi::tensorlib::GetDataTypeSize(out->dtype());
            const uint64_t total_bytes_64 = out->shape().numel() * bytes_per_element;
            if (total_bytes_64 > std::numeric_limits<uint32_t>::max())
            {
                throw std::runtime_error("fill_zeros tensor is too large for the current kernel implementation");
            }
            const uint32_t total_bytes = static_cast<uint32_t>(total_bytes_64);

            void *out_ptr = out->dataptr();
            if (const char *env = std::getenv("FBAMTRAIN_DEBUG_FILL_ZEROS"); env != nullptr && env[0] != '\0')
            {
                std::clog << "[FillZeros] tensor_id=" << out->id() << " dtype=" << static_cast<int>(out->dtype())
                          << " bytes=" << total_bytes << " device=" << out->device().ordinal << '\n';
            }
            pi::tensorlib::KernelLaunchArguments arguments{
                .args = {out_ptr, total_bytes, static_cast<void *>(nullptr)},
                .grid_dim_x = (total_bytes + kernel.meta.block_size - 1) / kernel.meta.block_size,
                .grid_dim_y = 1,
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
CreateFillUniformComputeKernelDescriptor(float min_value, float max_value, uint32_t seed,
                                         const pi::tensorlib::DataType data_type)
{
    std::string kernel_name_postfix{};
    kernel_bin_t<kernel_meta_elementwise_t> kfill_uniform{};
    switch (data_type)
    {
        case pi::tensorlib::DataType::FLOAT32:
        {
            kernel_name_postfix = "fp32";
            kfill_uniform = kfill_uniform_fp32;
            break;
        }

        case pi::tensorlib::DataType::BFLOAT16:
        {
            kernel_name_postfix = "bf16";
            kfill_uniform = kfill_uniform_bf16;
            break;
        }
        case pi::tensorlib::DataType::FLOAT16:
        {
            kernel_name_postfix = "fp16";
            kfill_uniform = kfill_uniform_fp16;
            break;
        }
        default:
            throw std::runtime_error("Unsupported data type for fill_uniform kernel");
    }
    return pi::tensorlib::ComputeKernelDescriptor{
        .kernel_name = "fill_uniform_" + kernel_name_postfix,
        .function_name = kfill_uniform.function_name,
        .expected_arg_count = kfill_uniform.arg_count,
        .argument_provider = [kfill_uniform, min_value, max_value,
                              seed](const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                                    const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
            -> pi::tensorlib::KernelLaunchArguments
        {
            if (inputs.size() != 1)
            {
                throw std::runtime_error("Invalid number of inputs for fill_uniform kernel");
            }
            if (outputs.size() != 1)
            {
                throw std::runtime_error("Invalid number of outputs for fill_uniform kernel");
            }

            if (inputs[0]->id() != outputs[0]->id())
            {
                throw std::runtime_error(
                    "fill_uniform is an in-place operation and requires input and output tensors to be the same");
            }

            const auto &out = outputs[0];

            if (out->device().device_type != pi::tensorlib::DeviceType::GPU)
            {
                throw std::runtime_error("fill_uniform currently supports GPU tensors only");
            }

            const auto device_ordinal = ValidateSameDeviceOrdinal("fill_uniform", {out});
            void *out_ptr = out->dataptr();
            uint32_t n_elements = out->shape().numel();

            pi::tensorlib::KernelLaunchArguments arguments{
                .args = {out_ptr, n_elements, min_value, max_value, seed, static_cast<void *>(nullptr)},
                .grid_dim_x = (n_elements + kfill_uniform.meta.block_size - 1) / kfill_uniform.meta.block_size,
                .grid_dim_y = 1,
                .grid_dim_z = 1,
                .block_dim_x = TRITON_WARP_SIZE * kfill_uniform.num_warps,
                .block_dim_y = 1,
                .block_dim_z = 1,
                .shared_mem_bytes = kfill_uniform.shared_mem_bytes,
                .device_ordinal = device_ordinal};
            return arguments;
        },
        .cuda_descriptor = pi::tensorlib::MakeCudaKernelDescriptor(kfill_uniform.data, kfill_uniform.size),
        .hip_descriptor = pi::tensorlib::MakeHipKernelDescriptor(kfill_uniform.data, kfill_uniform.size),
    };
}

static pi::tensorlib::ComputeKernelDescriptor
CreateFillConstantComputeKernelDescriptor(float value, const pi::tensorlib::DataType data_type)
{
    std::string kernel_name_postfix{};
    kernel_bin_t<kernel_meta_elementwise_t> kfill_constant{};
    switch (data_type)
    {
        case pi::tensorlib::DataType::FLOAT32:
            kernel_name_postfix = "fp32";
            kfill_constant = kfill_constant_fp32;
            break;
        case pi::tensorlib::DataType::BFLOAT16:
            kernel_name_postfix = "bf16";
            kfill_constant = kfill_constant_bf16;
            break;
        case pi::tensorlib::DataType::FLOAT16:
            kernel_name_postfix = "fp16";
            kfill_constant = kfill_constant_fp16;
            break;
        default:
            throw std::runtime_error("Unsupported data type for fill_constant kernel");
    }

    return pi::tensorlib::ComputeKernelDescriptor{
        .kernel_name = "fill_constant_" + kernel_name_postfix,
        .function_name = kfill_constant.function_name,
        .expected_arg_count = kfill_constant.arg_count,
        .argument_provider = [kfill_constant, value](
                                 const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                                 const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
            -> pi::tensorlib::KernelLaunchArguments
        {
            if (inputs.size() != 1)
            {
                throw std::runtime_error("fill_constant expects exactly one input tensor");
            }
            if (outputs.size() != 1)
            {
                throw std::runtime_error("fill_constant expects exactly one output tensor");
            }
            if (inputs[0]->id() != outputs[0]->id())
            {
                throw std::runtime_error("fill_constant is inplace and requires identical input/output tensors");
            }

            const auto &out = outputs[0];
            if (out->device().device_type != pi::tensorlib::DeviceType::GPU)
            {
                throw std::runtime_error("fill_constant currently supports GPU tensors only");
            }

            const auto device_ordinal = ValidateSameDeviceOrdinal("fill_constant", {out});
            void *out_ptr = out->dataptr();
            uint32_t n_elements = out->shape().numel();

            pi::tensorlib::KernelLaunchArguments arguments{
                .args = {out_ptr, n_elements, value, static_cast<void *>(nullptr)},
                .grid_dim_x = (n_elements + kfill_constant.meta.block_size - 1) / kfill_constant.meta.block_size,
                .grid_dim_y = 1,
                .grid_dim_z = 1,
                .block_dim_x = TRITON_WARP_SIZE * kfill_constant.num_warps,
                .block_dim_y = 1,
                .block_dim_z = 1,
                .shared_mem_bytes = kfill_constant.shared_mem_bytes,
                .device_ordinal = device_ordinal};
            return arguments;
        },
        .cuda_descriptor = pi::tensorlib::MakeCudaKernelDescriptor(kfill_constant.data, kfill_constant.size),
        .hip_descriptor = pi::tensorlib::MakeHipKernelDescriptor(kfill_constant.data, kfill_constant.size),
    };
}

void FillZerosImplPass::transform(pi::tensorlib::ExecutionPlan &execution_plan)
{
    for (auto &fill_op : execution_plan.entries)
    {
        if (fill_op.op_type == pi::tensorlib::OpType::FILL_ZEROS)
        {
            if (fill_op.inputs.size() != 1)
            {
                std::cout << "FillZerosImplPass: skipping op with inputs size " << fill_op.inputs.size() << std::endl;
                continue;
            }
            if (fill_op.outputs.size() != 1)
            {
                std::cout << "FillZerosImplPass: skipping op with outputs size " << fill_op.outputs.size() << std::endl;
                continue;
            }
            if (fill_op.inputs[0]->id() != fill_op.outputs[0]->id())
            {
                std::cout << "FillZerosImplPass: skipping op with different input/output tensor ids "
                          << fill_op.inputs[0]->id() << " != " << fill_op.outputs[0]->id() << std::endl;
                continue;
            }

            fill_op.op_type = std::nullopt;
            fill_op.kernel_descriptor = CreateFillZerosComputeKernelDescriptor();
            fill_op.flop_estimate = 0;
        }
    }
}

void FillUniformImplPass::transform(pi::tensorlib::ExecutionPlan &execution_plan)
{
    for (auto &fill_op : execution_plan.entries)
    {
        if (fill_op.op_type == pi::tensorlib::OpType::FILL_UNIFORM)
        {
            if (fill_op.inputs.size() != 1)
            {
                continue;
            }
            if (fill_op.outputs.size() != 1)
            {
                continue;
            }
            const auto &input = fill_op.inputs[0];
            const auto input_dtype = input->dtype();
            if (input_dtype != pi::tensorlib::DataType::BFLOAT16 && input_dtype != pi::tensorlib::DataType::FLOAT32 &&
                input_dtype != pi::tensorlib::DataType::FLOAT16)
            {
                continue;
            }
            fill_op.op_type = std::nullopt;

            const auto &attributes = fill_op.attributes;

            const auto min_value = std::any_cast<float>(attributes.at("min_value"));
            const auto max_value = std::any_cast<float>(attributes.at("max_value"));
            const auto seed = std::any_cast<uint32_t>(attributes.at("seed"));

            fill_op.kernel_descriptor =
                CreateFillUniformComputeKernelDescriptor(min_value, max_value, seed, input_dtype);
            fill_op.flop_estimate = 0;
        }
    }
}

static pi::tensorlib::ComputeKernelDescriptor
CreateFillNormalComputeKernelDescriptor(float mean, float std, uint32_t seed, const pi::tensorlib::DataType dtype)
{
    std::string kernel_name;
    kernel_bin_t<kernel_meta_elementwise_t> kernel{};
    switch (dtype)
    {
        case pi::tensorlib::DataType::BFLOAT16:
            kernel_name = "fill_normal_bf16";
            kernel = kfill_normal_bf16;
            break;
        case pi::tensorlib::DataType::FLOAT16:
            kernel_name = "fill_normal_fp16";
            kernel = kfill_normal_fp16;
            break;
        default:
            throw std::runtime_error("fill_normal only supports BF16 or FP16 tensors");
    }

    return pi::tensorlib::ComputeKernelDescriptor{
        .kernel_name = kernel_name,
        .function_name = kernel.function_name,
        .expected_arg_count = kernel.arg_count,
        .argument_provider = [kernel, mean, std,
                              seed](const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                                    const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
            -> pi::tensorlib::KernelLaunchArguments
        {
            if (inputs.size() != 1)
            {
                throw std::runtime_error("fill_normal expects exactly one input tensor");
            }
            if (outputs.size() != 1)
            {
                throw std::runtime_error("fill_normal expects exactly one output tensor");
            }

            if (inputs[0]->id() != outputs[0]->id())
            {
                throw std::runtime_error("fill_normal is an in-place operation and requires identical input/output");
            }

            const auto &out = outputs[0];

            if (out->device().device_type != pi::tensorlib::DeviceType::GPU)
            {
                throw std::runtime_error("fill_normal currently supports GPU tensors only");
            }

            const auto device_ordinal = ValidateSameDeviceOrdinal("fill_normal", {out});
            void *out_ptr = out->dataptr();
            uint32_t n_elements = out->shape().numel();

            pi::tensorlib::KernelLaunchArguments arguments{
                .args = {out_ptr, n_elements, mean, std, seed, static_cast<void *>(nullptr)},
                .grid_dim_x = (n_elements + kernel.meta.block_size - 1) / kernel.meta.block_size,
                .grid_dim_y = 1,
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

void FillNormalImplPass::transform(pi::tensorlib::ExecutionPlan &execution_plan)
{
    for (auto &fill_op : execution_plan.entries)
    {
        if (fill_op.op_type == pi::tensorlib::OpType::FILL_NORMAL)
        {
            if (fill_op.inputs.size() != 1)
            {
                continue;
            }
            const auto &input = fill_op.inputs[0];
            const auto input_dtype = input->dtype();
            if (input_dtype != pi::tensorlib::DataType::BFLOAT16 &&
                input_dtype != pi::tensorlib::DataType::FLOAT16)
            {
                continue;
            }
            fill_op.op_type = std::nullopt;

            const auto &attributes = fill_op.attributes;

            const auto mean = std::any_cast<float>(attributes.at("mean"));
            const auto std = std::any_cast<float>(attributes.at("std"));
            const auto seed = std::any_cast<uint32_t>(attributes.at("seed"));

            fill_op.kernel_descriptor = CreateFillNormalComputeKernelDescriptor(mean, std, seed, input_dtype);
            fill_op.flop_estimate = 0;
        }
    }
}

void FillConstantImplPass::transform(pi::tensorlib::ExecutionPlan &execution_plan)
{
    for (auto &fill_op : execution_plan.entries)
    {
        if (fill_op.op_type == pi::tensorlib::OpType::FILL_CONSTANT)
        {
            if (fill_op.inputs.size() != 1)
            {
                continue;
            }
            if (fill_op.outputs.size() != 1)
            {
                continue;
            }
            const auto &input = fill_op.inputs[0];
            const auto input_dtype = input->dtype();
            if (input_dtype != pi::tensorlib::DataType::BFLOAT16 && input_dtype != pi::tensorlib::DataType::FLOAT32 &&
                input_dtype != pi::tensorlib::DataType::FLOAT16)
            {
                continue;
            }
            fill_op.op_type = std::nullopt;

            const auto &attributes = fill_op.attributes;
            const auto value = std::any_cast<float>(attributes.at("value"));

            fill_op.kernel_descriptor = CreateFillConstantComputeKernelDescriptor(value, input_dtype);
            fill_op.flop_estimate = 0;
        }
    }
}
