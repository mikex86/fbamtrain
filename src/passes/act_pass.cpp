#include "passes.h"
#include "shape_utils.h"

#include <kernels/kernel_binaries.h>
#include "passes/pass_utils.h"

#include <activation.h>

#include <any>
#include <limits>
#include <string>
#include <utility>

static pi::tensorlib::ComputeKernelDescriptor
CreateActGeluElementwiseComputeKernelDescriptor(const pi::tensorlib::DataType dtype)
{
    const auto kernel = SelectKernelForHalf(kact_gelu_elementwise_bf16, kact_gelu_elementwise_fp16, dtype);
    const std::string kernel_name =
        std::string("act_gelu_elementwise_") + std::string(KernelSuffixForHalf(dtype));
    const std::string dtype_name = pi::tensorlib::GetDataTypeName(dtype);

    return pi::tensorlib::ComputeKernelDescriptor{
        .kernel_name = kernel_name,
        .function_name = kernel.function_name,
        .expected_arg_count = kernel.arg_count,
        .argument_provider = [kernel, dtype, dtype_name](
                                 const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                                 const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
            -> pi::tensorlib::KernelLaunchArguments
        {
            if (inputs.size() != 1)
            {
                throw std::runtime_error("act_gelu_elementwise expects exactly one input tensor");
            }
            if (outputs.size() != 1)
            {
                throw std::runtime_error("act_gelu_elementwise expects exactly one output tensor");
            }

            const auto &input = inputs[0];
            const auto &output = outputs[0];

            if (input->dtype() != dtype || output->dtype() != dtype)
            {
                throw std::runtime_error("act_gelu_elementwise only supports " + dtype_name + " tensors");
            }

            if (input->device().device_type != pi::tensorlib::DeviceType::GPU ||
                output->device().device_type != pi::tensorlib::DeviceType::GPU)
            {
                throw std::runtime_error("act_gelu_elementwise requires GPU tensors");
            }
            const auto device_ordinal =
                ValidateSameDeviceOrdinal("act_gelu_elementwise", {input, output});

            if (input->shape().ndims() != output->shape().ndims())
            {
                throw std::runtime_error("act_gelu_elementwise expects input/output ranks to match");
            }
            for (size_t dim = 0; dim < input->shape().ndims(); ++dim)
            {
                if (input->shape()[dim] != output->shape()[dim])
                {
                    throw std::runtime_error("act_gelu_elementwise expects input/output shapes to match");
                }
            }

            const auto numel64 = output->shape().numel();
            if (numel64 == 0)
            {
                throw std::runtime_error("act_gelu_elementwise tensor must contain at least one element");
            }
            if (numel64 > std::numeric_limits<uint32_t>::max())
            {
                throw std::runtime_error("act_gelu_elementwise tensor size exceeds supported range");
            }

            if (!pi::tensorlib::shape_utils::IsRowMajorContiguous(input) || !pi::tensorlib::shape_utils::IsRowMajorContiguous(output))
            {
                throw std::runtime_error("act_gelu_elementwise requires contiguous tensors");
            }

            void *out_ptr = output->dataptr();
            void *in_ptr = input->dataptr();
            const auto n_elements = static_cast<uint32_t>(numel64);

            pi::tensorlib::KernelLaunchArguments arguments{
                .args = {out_ptr, in_ptr, n_elements, static_cast<void *>(nullptr)},
                .grid_dim_x = CEIL_DIV(n_elements, kernel.meta.block_size),
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
CreateActGeluBackwardComputeKernelDescriptor(const pi::tensorlib::DataType dtype)
{
    const auto kernel = SelectKernelForHalf(kact_gelu_bwd_elementwise_bf16, kact_gelu_bwd_elementwise_fp16, dtype);
    const std::string kernel_name =
        std::string("act_gelu_bwd_elementwise_") + std::string(KernelSuffixForHalf(dtype));
    const std::string dtype_name = pi::tensorlib::GetDataTypeName(dtype);

    return pi::tensorlib::ComputeKernelDescriptor{
        .kernel_name = kernel_name,
        .function_name = kernel.function_name,
        .expected_arg_count = kernel.arg_count,
        .argument_provider = [kernel, dtype, dtype_name](
                                 const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                                 const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
            -> pi::tensorlib::KernelLaunchArguments
        {
            if (inputs.size() != 2)
            {
                throw std::runtime_error("act_gelu_bwd_elementwise expects exactly two input tensors");
            }
            if (outputs.size() != 1)
            {
                throw std::runtime_error("act_gelu_bwd_elementwise expects exactly one output tensor");
            }

            const auto &input = inputs[0];
            const auto &upstream = inputs[1];
            const auto &output = outputs[0];

            if (input->dtype() != dtype || upstream->dtype() != dtype || output->dtype() != dtype)
            {
                throw std::runtime_error("act_gelu_bwd_elementwise only supports " + dtype_name + " tensors");
            }

            if (input->device().device_type != pi::tensorlib::DeviceType::GPU ||
                upstream->device().device_type != pi::tensorlib::DeviceType::GPU ||
                output->device().device_type != pi::tensorlib::DeviceType::GPU)
            {
                throw std::runtime_error("act_gelu_bwd_elementwise requires GPU tensors");
            }
            const auto device_ordinal =
                ValidateSameDeviceOrdinal("act_gelu_bwd_elementwise", {input, upstream, output});

            if (input->shape().ndims() != output->shape().ndims() ||
                upstream->shape().ndims() != output->shape().ndims())
            {
                throw std::runtime_error("act_gelu_bwd_elementwise expects input/output ranks to match");
            }
            for (size_t dim = 0; dim < output->shape().ndims(); ++dim)
            {
                if (input->shape()[dim] != output->shape()[dim] ||
                    upstream->shape()[dim] != output->shape()[dim])
                {
                    throw std::runtime_error("act_gelu_bwd_elementwise expects input/output shapes to match");
                }
            }

            const auto numel64 = output->shape().numel();
            if (numel64 == 0)
            {
                throw std::runtime_error("act_gelu_bwd_elementwise tensor must contain at least one element");
            }
            if (numel64 > std::numeric_limits<uint32_t>::max())
            {
                throw std::runtime_error("act_gelu_bwd_elementwise tensor size exceeds supported range");
            }

            if (!pi::tensorlib::shape_utils::IsRowMajorContiguous(input) ||
                !pi::tensorlib::shape_utils::IsRowMajorContiguous(upstream) ||
                !pi::tensorlib::shape_utils::IsRowMajorContiguous(output))
            {
                throw std::runtime_error("act_gelu_bwd_elementwise requires contiguous tensors");
            }

            void *out_ptr = output->dataptr();
            void *in_ptr = input->dataptr();
            void *up_ptr = upstream->dataptr();
            const auto n_elements = static_cast<uint32_t>(numel64);

            pi::tensorlib::KernelLaunchArguments arguments{
                .args = {out_ptr, in_ptr, up_ptr, n_elements, static_cast<void *>(nullptr)},
                .grid_dim_x = CEIL_DIV(n_elements, kernel.meta.block_size),
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
CreateActReluElementwiseComputeKernelDescriptor(const pi::tensorlib::DataType dtype)
{
    const auto kernel = SelectKernelForHalf(kact_relu_elementwise_bf16, kact_relu_elementwise_fp16, dtype);
    const std::string kernel_name =
        std::string("act_relu_elementwise_") + std::string(KernelSuffixForHalf(dtype));
    const std::string dtype_name = pi::tensorlib::GetDataTypeName(dtype);

    return pi::tensorlib::ComputeKernelDescriptor{
        .kernel_name = kernel_name,
        .function_name = kernel.function_name,
        .expected_arg_count = kernel.arg_count,
        .argument_provider = [kernel, dtype, dtype_name](
                                 const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                                 const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
            -> pi::tensorlib::KernelLaunchArguments
        {
            if (inputs.size() != 1)
            {
                throw std::runtime_error("act_relu_elementwise expects exactly one input tensor");
            }
            if (outputs.size() != 1)
            {
                throw std::runtime_error("act_relu_elementwise expects exactly one output tensor");
            }

            const auto &input = inputs[0];
            const auto &output = outputs[0];

            if (input->dtype() != dtype || output->dtype() != dtype)
            {
                throw std::runtime_error("act_relu_elementwise only supports " + dtype_name + " tensors");
            }

            if (input->device().device_type != pi::tensorlib::DeviceType::GPU ||
                output->device().device_type != pi::tensorlib::DeviceType::GPU)
            {
                throw std::runtime_error("act_relu_elementwise requires GPU tensors");
            }
            const auto device_ordinal =
                ValidateSameDeviceOrdinal("act_relu_elementwise", {input, output});

            if (input->shape().ndims() != output->shape().ndims())
            {
                throw std::runtime_error("act_relu_elementwise expects input/output ranks to match");
            }
            for (size_t dim = 0; dim < input->shape().ndims(); ++dim)
            {
                if (input->shape()[dim] != output->shape()[dim])
                {
                    throw std::runtime_error("act_relu_elementwise expects input/output shapes to match");
                }
            }

            const auto numel64 = output->shape().numel();
            if (numel64 == 0)
            {
                throw std::runtime_error("act_relu_elementwise tensor must contain at least one element");
            }
            if (numel64 > std::numeric_limits<uint32_t>::max())
            {
                throw std::runtime_error("act_relu_elementwise tensor size exceeds supported range");
            }

            if (!pi::tensorlib::shape_utils::IsRowMajorContiguous(input) || !pi::tensorlib::shape_utils::IsRowMajorContiguous(output))
            {
                throw std::runtime_error("act_relu_elementwise requires contiguous tensors");
            }

            void *out_ptr = output->dataptr();
            void *in_ptr = input->dataptr();
            const auto n_elements = static_cast<uint32_t>(numel64);

            pi::tensorlib::KernelLaunchArguments arguments{
                .args = {out_ptr, in_ptr, n_elements, static_cast<void *>(nullptr)},
                .grid_dim_x = CEIL_DIV(n_elements, kernel.meta.block_size),
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

void ActImplPass::transform(pi::tensorlib::ExecutionPlan &execution_plan)
{
    for (auto &entry : execution_plan.entries)
    {
        if (entry.op_type != pi::tensorlib::OpType::ACT_FN && entry.op_type != pi::tensorlib::OpType::ACT_FN_BWD)
        {
            continue;
        }
        const bool is_backward = entry.op_type == pi::tensorlib::OpType::ACT_FN_BWD;
        if (entry.outputs.size() != 1 || (!is_backward && entry.inputs.size() != 1) ||
            (is_backward && entry.inputs.size() != 2))
        {
            continue;
        }

        const auto attr_it = entry.attributes.find("activation_function");
        if (attr_it == entry.attributes.end())
        {
            continue;
        }

        pi::tensorlib::ActivationFunction activation{};
        try
        {
            activation = std::any_cast<pi::tensorlib::ActivationFunction>(attr_it->second);
        }
        catch (const std::bad_any_cast &)
        {
            continue;
        }

        const auto &input = entry.inputs[0];
        const auto &output = entry.outputs[0];
        const auto dtype = input->dtype();
        if (is_backward && entry.inputs[1]->dtype() != dtype)
        {
            continue;
        }

        if (!IsSupportedHalfType(dtype) || output->dtype() != dtype)
        {
            continue;
        }

        if (input->device().device_type != pi::tensorlib::DeviceType::GPU ||
            output->device().device_type != pi::tensorlib::DeviceType::GPU ||
            (is_backward && entry.inputs[1]->device().device_type != pi::tensorlib::DeviceType::GPU))
        {
            continue;
        }
        if (input->device().ordinal != output->device().ordinal ||
            (is_backward && entry.inputs[1]->device().ordinal != output->device().ordinal))
        {
            continue;
        }

        if (input->shape().ndims() != output->shape().ndims())
        {
            continue;
        }
        bool same_shape = true;
        for (size_t dim = 0; dim < input->shape().ndims(); ++dim)
        {
            if (input->shape()[dim] != output->shape()[dim] ||
                (is_backward && entry.inputs[1]->shape()[dim] != output->shape()[dim]))
            {
                same_shape = false;
                break;
            }
        }
        if (!same_shape)
        {
            continue;
        }

        const auto numel64 = output->shape().numel();
        if (numel64 == 0 || numel64 > std::numeric_limits<uint32_t>::max())
        {
            continue;
        }

        if (!pi::tensorlib::shape_utils::IsRowMajorContiguous(input) ||
            !pi::tensorlib::shape_utils::IsRowMajorContiguous(output) ||
            (is_backward && !pi::tensorlib::shape_utils::IsRowMajorContiguous(entry.inputs[1])))
        {
            continue;
        }

        pi::tensorlib::ComputeKernelDescriptor descriptor{};
        bool supported_activation = true;
        switch (activation)
        {
            case pi::tensorlib::ActivationFunction::GELU:
                descriptor = is_backward ? CreateActGeluBackwardComputeKernelDescriptor(dtype)
                                         : CreateActGeluElementwiseComputeKernelDescriptor(dtype);
                break;
            case pi::tensorlib::ActivationFunction::RELU:
                if (is_backward)
                {
                    supported_activation = false;
                }
                else
                {
                    descriptor = CreateActReluElementwiseComputeKernelDescriptor(dtype);
                }
                break;
            default:
                supported_activation = false;
                break;
        }

        if (!supported_activation)
        {
            continue;
        }

        entry.flop_estimate = EstimateActivationFlops(activation, numel64);
        entry.op_type = std::nullopt;
        entry.kernel_descriptor = descriptor;
    }
}
