#include "passes.h"

#include "passes/pass_utils.h"
#include "shape_utils.h"

#include <kernels/kernel_binaries.h>
#include <launch_utils.h>

#include <sstream>

namespace
{
    pi::tensorlib::ComputeKernelDescriptor CreateReducePartialKernelDescriptor(const uint32_t block_size)
    {
        if (block_size != kreduce_sum_partial_fp32.meta.block_size)
        {
            throw std::runtime_error("Unsupported block size for reduce_sum_partial kernel: " + std::to_string(block_size) +
                                     ". Supported block size: " + std::to_string(kreduce_sum_partial_fp32.meta.block_size));
        }

        const auto kernel = kreduce_sum_partial_fp32; // sm set via alias
        const std::string kernel_name = "reduce_sum_partial_fp32";
        return pi::tensorlib::ComputeKernelDescriptor{
            .kernel_name = kernel_name,
            .function_name = std::string(kernel.function_name),
            .expected_arg_count = kernel.arg_count,
            .argument_provider = [kernel, kernel_name, block_size](
                                     const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                                                     const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
                -> pi::tensorlib::KernelLaunchArguments {
                if (inputs.size() != 1 || outputs.size() != 1)
                {
                    throw std::runtime_error(kernel_name + " expects 1 input and 1 output");
                }
                const auto &in = inputs[0];
                const auto &out = outputs[0];
                if (in->dtype() != pi::tensorlib::DataType::FLOAT32 || out->dtype() != pi::tensorlib::DataType::FLOAT32)
                {
                    throw std::runtime_error(kernel_name + " supports FLOAT32 tensors only");
                }
                if (!pi::tensorlib::shape_utils::IsRowMajorContiguous(in->shape(), in->strides()) ||
                    !pi::tensorlib::shape_utils::IsRowMajorContiguous(out->shape(), out->strides()))
                {
                    throw std::runtime_error(kernel_name + " requires contiguous tensors");
                }
                const auto n = in->shape().numel();
                const auto expected_out = (n + block_size - 1) / block_size;
                if (out->shape().numel() != expected_out)
                {
                    throw std::runtime_error(kernel_name + " output size mismatch");
                }
                const auto device_ordinal =
                    ValidateSameDeviceOrdinal(kernel_name, {in, out}, pi::tensorlib::DeviceType::GPU);
                pi::tensorlib::KernelLaunchArguments args{
                    .args = {out->dataptr(), in->dataptr(), static_cast<uint32_t>(n),
                             static_cast<void *>(nullptr)},
                    .grid_dim_x = static_cast<uint32_t>(expected_out),
                    .grid_dim_y = 1,
                    .grid_dim_z = 1,
                    .block_dim_x = block_size,
                    .block_dim_y = 1,
                    .block_dim_z = 1,
                    .shared_mem_bytes = kernel.shared_mem_bytes,
                    .device_ordinal = device_ordinal};
                return args;
            },
            .cuda_descriptor = pi::tensorlib::MakeCudaKernelDescriptor(kernel.data, kernel.size),
            .hip_descriptor = pi::tensorlib::MakeHipKernelDescriptor(kernel.data, kernel.size)};
    }
} // namespace

void ReduceSumPartialImplPass::transform(pi::tensorlib::ExecutionPlan &execution_plan)
{
    for (auto &entry : execution_plan.entries)
    {
        if (entry.op_type != pi::tensorlib::OpType::REDUCE_SUM_PARTIAL)
        {
            continue;
        }
        if (entry.inputs.size() != 1 || entry.outputs.size() != 1)
        {
            throw std::runtime_error("REDUCE_SUM_PARTIAL expects 1 input and 1 output");
        }

        uint32_t block_size{};
        auto it = entry.attributes.find("block_size");
        if (it == entry.attributes.end())
        {
            throw std::runtime_error("REDUCE_SUM_PARTIAL missing 'block_size' attribute");
        }
        block_size = static_cast<uint32_t>(std::any_cast<int64_t>(it->second));

        entry.op_type = std::nullopt;
        entry.kernel_descriptor = CreateReducePartialKernelDescriptor(block_size);
    }
}
