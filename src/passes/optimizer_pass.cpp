#include "passes.h"

#include "passes/pass_utils.h"

#include <shape_utils.h>

#include <any>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>

namespace
{
    uint32_t NarrowToUint32(const uint64_t value, const char *name)
    {
        if (value > std::numeric_limits<uint32_t>::max())
        {
            throw std::runtime_error(std::string(name) + " exceeds uint32_t range for optimizer kernel.");
        }
        return static_cast<uint32_t>(value);
    }

    kernel_bin_t<kernel_meta_elementwise_t> SelectAdamwKernel(const pi::tensorlib::DataType dtype)
    {
        return SelectKernelForAdd(kadamw_step_bf16, kadamw_step_fp16, kadamw_step_fp32, dtype);
    }

    kernel_bin_t<kernel_meta_elementwise_t> SelectSgdKernel(const pi::tensorlib::DataType dtype)
    {
        return SelectKernelForAdd(ksgd_step_bf16, ksgd_step_fp16, ksgd_step_fp32, dtype);
    }

    pi::tensorlib::ComputeKernelDescriptor CreateAdamwComputeKernelDescriptor(const pi::tensorlib::DataType dtype,
                                                                              const float learning_rate,
                                                                              const float beta1, const float beta2,
                                                                              const float eps, const float weight_decay)
    {
        const auto kernel = SelectAdamwKernel(dtype);
        const std::string kernel_name = "adamw_step_" + std::string(KernelSuffixForAdd(dtype));

        return pi::tensorlib::ComputeKernelDescriptor{
            .kernel_name = kernel_name,
            .function_name = kernel.function_name,
            .expected_arg_count = kernel.arg_count,
            .argument_provider = [kernel, dtype, learning_rate, beta1, beta2, eps, weight_decay](
                                     const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                                     const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
                -> pi::tensorlib::KernelLaunchArguments
            {
                if (inputs.size() != 6 || outputs.size() != 3)
                {
                    throw std::runtime_error("optimizer_adamw expects 6 inputs and 3 outputs");
                }

                const auto &param = inputs[0];
                const auto &grad = inputs[1];
                const auto &m = inputs[2];
                const auto &v = inputs[3];
                const auto &bias_correction1 = inputs[4];
                const auto &bias_correction2 = inputs[5];

                if (param->dtype() != dtype || grad->dtype() != dtype)
                {
                    throw std::runtime_error("optimizer_adamw requires param/grad to match expected dtype");
                }
                if (m->dtype() != pi::tensorlib::DataType::FLOAT32 || v->dtype() != pi::tensorlib::DataType::FLOAT32)
                {
                    throw std::runtime_error("optimizer_adamw requires m/v state to be FLOAT32");
                }
                if (param->shape() != grad->shape() || param->shape() != m->shape() || param->shape() != v->shape())
                {
                    throw std::runtime_error("optimizer_adamw requires param/grad/state shapes to match");
                }
                if (!pi::tensorlib::shape_utils::IsRowMajorContiguous(param) ||
                    !pi::tensorlib::shape_utils::IsRowMajorContiguous(grad) ||
                    !pi::tensorlib::shape_utils::IsRowMajorContiguous(m) ||
                    !pi::tensorlib::shape_utils::IsRowMajorContiguous(v))
                {
                    throw std::runtime_error("optimizer_adamw requires contiguous tensors");
                }

                if (outputs[0]->id() != param->id() || outputs[1]->id() != m->id() || outputs[2]->id() != v->id())
                {
                    throw std::runtime_error("optimizer_adamw expects in-place outputs for param/m/v");
                }

                if (!bias_correction1 || !bias_correction2)
                {
                    throw std::runtime_error("optimizer_adamw requires bias correction tensors");
                }
                if (bias_correction1->dtype() != pi::tensorlib::DataType::FLOAT32 ||
                    bias_correction2->dtype() != pi::tensorlib::DataType::FLOAT32)
                {
                    throw std::runtime_error("optimizer_adamw requires bias correction tensors to be FLOAT32");
                }
                if (bias_correction1->shape().numel() != 1 || bias_correction2->shape().numel() != 1)
                {
                    throw std::runtime_error("optimizer_adamw requires bias correction tensors to be scalars");
                }
                if (bias_correction1->device().device_type != pi::tensorlib::DeviceType::GPU ||
                    bias_correction2->device().device_type != pi::tensorlib::DeviceType::GPU)
                {
                    throw std::runtime_error("optimizer_adamw requires bias correction tensors on GPU");
                }

                const auto device_ordinal =
                    ValidateSameDeviceOrdinal("optimizer_adamw", {param, grad, m, v, bias_correction1, bias_correction2});
                const uint32_t n_elements = NarrowToUint32(param->shape().numel(), "optimizer_adamw n_elements");

                pi::tensorlib::KernelLaunchArguments arguments{
                    .args = {param->dataptr(), grad->dataptr(), m->dataptr(), v->dataptr(),
                             bias_correction1->dataptr(), bias_correction2->dataptr(), n_elements, learning_rate,
                             beta1, beta2, eps, weight_decay,
                             static_cast<void *>(nullptr)},
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

    pi::tensorlib::ComputeKernelDescriptor CreateSgdComputeKernelDescriptor(const pi::tensorlib::DataType dtype,
                                                                            const float learning_rate,
                                                                            const float momentum,
                                                                            const float weight_decay,
                                                                            const int32_t nesterov)
    {
        const auto kernel = SelectSgdKernel(dtype);
        const std::string kernel_name = "sgd_step_" + std::string(KernelSuffixForAdd(dtype));

        return pi::tensorlib::ComputeKernelDescriptor{
            .kernel_name = kernel_name,
            .function_name = kernel.function_name,
            .expected_arg_count = kernel.arg_count,
            .argument_provider = [kernel, dtype, learning_rate, momentum, weight_decay, nesterov](
                                     const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                                     const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
                -> pi::tensorlib::KernelLaunchArguments
            {
                if (inputs.size() != 3 || outputs.size() != 2)
                {
                    throw std::runtime_error("optimizer_sgd expects 3 inputs and 2 outputs");
                }

                const auto &param = inputs[0];
                const auto &grad = inputs[1];
                const auto &velocity = inputs[2];

                if (param->dtype() != dtype || grad->dtype() != dtype)
                {
                    throw std::runtime_error("optimizer_sgd requires param/grad to match expected dtype");
                }
                if (velocity->dtype() != pi::tensorlib::DataType::FLOAT32)
                {
                    throw std::runtime_error("optimizer_sgd requires velocity state to be FLOAT32");
                }
                if (param->shape() != grad->shape() || param->shape() != velocity->shape())
                {
                    throw std::runtime_error("optimizer_sgd requires param/grad/state shapes to match");
                }
                if (!pi::tensorlib::shape_utils::IsRowMajorContiguous(param) ||
                    !pi::tensorlib::shape_utils::IsRowMajorContiguous(grad) ||
                    !pi::tensorlib::shape_utils::IsRowMajorContiguous(velocity))
                {
                    throw std::runtime_error("optimizer_sgd requires contiguous tensors");
                }

                if (outputs[0]->id() != param->id() || outputs[1]->id() != velocity->id())
                {
                    throw std::runtime_error("optimizer_sgd expects in-place outputs for param/velocity");
                }

                const auto device_ordinal = ValidateSameDeviceOrdinal("optimizer_sgd", {param, grad, velocity});
                const uint32_t n_elements = NarrowToUint32(param->shape().numel(), "optimizer_sgd n_elements");

                pi::tensorlib::KernelLaunchArguments arguments{
                    .args = {param->dataptr(), grad->dataptr(), velocity->dataptr(), n_elements, learning_rate,
                             momentum, weight_decay, nesterov, static_cast<void *>(nullptr)},
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
} // namespace

void OptimizerImplPass::transform(pi::tensorlib::ExecutionPlan &execution_plan)
{
    for (auto &entry : execution_plan.entries)
    {
        if (entry.op_type == pi::tensorlib::OpType::OPTIMIZER_ADAMW)
        {
            if (entry.inputs.empty())
            {
                continue;
            }
            const auto dtype = entry.inputs[0]->dtype();

            const auto learning_rate = std::any_cast<float>(entry.attributes.at("learning_rate"));
            const auto beta1 = std::any_cast<float>(entry.attributes.at("beta1"));
            const auto beta2 = std::any_cast<float>(entry.attributes.at("beta2"));
            const auto eps = std::any_cast<float>(entry.attributes.at("eps"));
            const auto weight_decay = std::any_cast<float>(entry.attributes.at("weight_decay"));

            entry.op_type = std::nullopt;
            entry.kernel_descriptor = CreateAdamwComputeKernelDescriptor(dtype, learning_rate, beta1, beta2, eps,
                                                                          weight_decay);
            continue;
        }

        if (entry.op_type == pi::tensorlib::OpType::OPTIMIZER_SGD)
        {
            if (entry.inputs.empty())
            {
                continue;
            }
            const auto dtype = entry.inputs[0]->dtype();

            const auto learning_rate = std::any_cast<float>(entry.attributes.at("learning_rate"));
            const auto momentum = std::any_cast<float>(entry.attributes.at("momentum"));
            const auto weight_decay = std::any_cast<float>(entry.attributes.at("weight_decay"));
            const auto nesterov = std::any_cast<int32_t>(entry.attributes.at("nesterov"));

            entry.op_type = std::nullopt;
            entry.kernel_descriptor =
                CreateSgdComputeKernelDescriptor(dtype, learning_rate, momentum, weight_decay, nesterov);
            continue;
        }
    }
}
