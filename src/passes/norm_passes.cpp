#include "passes.h"

#include <kernels/kernel_binaries.h>
#include "passes/pass_utils.h"

#include <any>
#include <limits>
#include <string>

static pi::tensorlib::ComputeKernelDescriptor
CreateLayerNormComputeKernelDescriptor(float eps, const kernel_bin_t<kernel_meta_elementwise_t> &kernel,
                                       const pi::tensorlib::DataType dtype)
{
    const std::string kernel_name =
        dtype == pi::tensorlib::DataType::BFLOAT16 ? "layer_norm_fwd_bf16" : "layer_norm_fwd_fp16";
    const std::string dtype_name = pi::tensorlib::GetDataTypeName(dtype);

    return pi::tensorlib::ComputeKernelDescriptor{
        .kernel_name = kernel_name,
        .function_name = kernel.function_name,
        .expected_arg_count = kernel.arg_count,
        .argument_provider = [eps, kernel, dtype,
                              dtype_name](const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                                          const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
            -> pi::tensorlib::KernelLaunchArguments
        {
            if (inputs.size() != 3)
            {
                throw std::runtime_error("layer_norm_fwd expects exactly three inputs");
            }
            if (outputs.size() != 1)
            {
                throw std::runtime_error("layer_norm_fwd expects exactly one output");
            }

            const auto &input = inputs[0];
            const auto &weight = inputs[1];
            const auto &bias = inputs[2];
            const auto &output = outputs[0];

            if (input->dtype() != dtype || weight->dtype() != dtype || bias->dtype() != dtype ||
                output->dtype() != dtype)
            {
                throw std::runtime_error("layer_norm_fwd requires " + dtype_name + " tensors");
            }

            if (input->device().device_type != pi::tensorlib::DeviceType::GPU ||
                weight->device().device_type != pi::tensorlib::DeviceType::GPU ||
                bias->device().device_type != pi::tensorlib::DeviceType::GPU ||
                output->device().device_type != pi::tensorlib::DeviceType::GPU)
            {
                throw std::runtime_error("layer_norm_fwd currently supports GPU tensors only");
            }

            const auto device_ordinal =
                ValidateSameDeviceOrdinal("layer_norm_fwd", {input, weight, bias, output});

            const auto hidden_size_64 = input->shape().dims().back();
            if (hidden_size_64 == 0)
            {
                throw std::runtime_error("layer_norm_fwd hidden size must be > 0");
            }
            if (hidden_size_64 > std::numeric_limits<uint32_t>::max())
            {
                throw std::runtime_error("layer_norm_fwd hidden size exceeds supported range");
            }

            const auto hidden_size = static_cast<uint32_t>(hidden_size_64);
            const auto numel = input->shape().numel();
            if (numel % hidden_size != 0)
            {
                throw std::runtime_error("layer_norm_fwd input numel must be divisible by hidden size");
            }

            const auto num_rows = static_cast<uint32_t>(numel / hidden_size);

            void *input_ptr = input->dataptr();
            void *weight_ptr = weight->dataptr();
            void *bias_ptr = bias->dataptr();
            void *output_ptr = output->dataptr();

            pi::tensorlib::KernelLaunchArguments arguments{
                .args = {input_ptr, weight_ptr, bias_ptr, output_ptr, num_rows, hidden_size, eps,
                         static_cast<void *>(nullptr)},
                .grid_dim_x = num_rows,
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

void LayerNormImplPass::transform(pi::tensorlib::ExecutionPlan &execution_plan)
{
    for (auto &op : execution_plan.entries)
    {
        if (op.op_type != pi::tensorlib::OpType::LAYER_NORM_FWD)
        {
            continue;
        }

        if (op.inputs.size() != 3)
        {
            throw std::runtime_error("layer_norm_fwd expects three inputs");
        }
        if (op.outputs.size() != 1)
        {
            throw std::runtime_error("layer_norm_fwd expects a single output");
        }

        const auto &input = op.inputs[0];
        const auto &weight = op.inputs[1];
        const auto &bias = op.inputs[2];
        const auto &output = op.outputs[0];

        const auto dtype = input->dtype();
        if (!IsSupportedHalfType(dtype) || weight->dtype() != dtype || bias->dtype() != dtype ||
            output->dtype() != dtype)
        {
            throw std::runtime_error("layer_norm_fwd requires BFLOAT16 or FLOAT16 tensors");
        }

        if (input->shape().ndims() < 1)
        {
            throw std::runtime_error("layer_norm_fwd expects input tensors with at least one dimension");
        }

        if (weight->shape().ndims() != 1 || bias->shape().ndims() != 1)
        {
            throw std::runtime_error("layer_norm_fwd weight and bias must be vectors");
        }

        const auto hidden_size = input->shape().dims().back();
        if (weight->shape()[0] != hidden_size || bias->shape()[0] != hidden_size)
        {
            throw std::runtime_error("layer_norm_fwd weight and bias must match input hidden dimension");
        }

        if (output->shape() != input->shape())
        {
            throw std::runtime_error("layer_norm_fwd output shape must match input shape");
        }

        if (input->strides()[input->shape().ndims() - 1] != 1 ||
            output->strides()[output->shape().ndims() - 1] != 1)
        {
            throw std::runtime_error("layer_norm_fwd requires contiguous last dimension");
        }

        if (weight->strides()[0] != 1 || bias->strides()[0] != 1)
        {
            throw std::runtime_error("layer_norm_fwd weight and bias must be contiguous");
        }

        const auto eps_it = op.attributes.find("eps");
        if (eps_it == op.attributes.end())
        {
            throw std::runtime_error("layer_norm_fwd missing eps attribute");
        }

        const auto eps = std::any_cast<float>(eps_it->second);

        op.op_type = std::nullopt;
        const auto kernel =
            (dtype == pi::tensorlib::DataType::BFLOAT16) ? klayer_norm_fwd_bf16 : klayer_norm_fwd_fp16;
        op.kernel_descriptor = CreateLayerNormComputeKernelDescriptor(eps, kernel, dtype);
        const uint64_t hidden = input->shape().dims().back();
        const uint64_t elements = input->shape().numel();
        const uint64_t rows = hidden == 0 ? 0 : elements / hidden;
        const uint64_t per_row_flops = 8 * hidden + 2; // mean + var reductions plus affine normalize
        op.flop_estimate = rows * per_row_flops;
    }
}

static pi::tensorlib::ComputeKernelDescriptor
CreateRmsNormComputeKernelDescriptor(float eps, const kernel_bin_t<kernel_meta_elementwise_t> &kernel,
                                     const pi::tensorlib::DataType dtype)
{
    const std::string kernel_name =
        dtype == pi::tensorlib::DataType::BFLOAT16 ? "rms_norm_fwd_bf16" : "rms_norm_fwd_fp16";
    const std::string dtype_name = pi::tensorlib::GetDataTypeName(dtype);

    return pi::tensorlib::ComputeKernelDescriptor{
        .kernel_name = kernel_name,
        .function_name = kernel.function_name,
        .expected_arg_count = kernel.arg_count,
        .argument_provider = [eps, kernel, dtype,
                              dtype_name](const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                                          const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
            -> pi::tensorlib::KernelLaunchArguments
        {
            if (inputs.size() != 2)
            {
                throw std::runtime_error("rms_norm_fwd expects two inputs");
            }
            if (outputs.size() != 1)
            {
                throw std::runtime_error("rms_norm_fwd expects a single output");
            }

            const auto &input = inputs[0];
            const auto &weight = inputs[1];
            const auto &output = outputs[0];

            if (input->dtype() != dtype || weight->dtype() != dtype || output->dtype() != dtype)
            {
                throw std::runtime_error("rms_norm_fwd requires " + dtype_name + " tensors");
            }

            if (input->device().device_type != pi::tensorlib::DeviceType::GPU ||
                weight->device().device_type != pi::tensorlib::DeviceType::GPU ||
                output->device().device_type != pi::tensorlib::DeviceType::GPU)
            {
                throw std::runtime_error("rms_norm_fwd currently supports GPU tensors only");
            }

            const auto device_ordinal =
                ValidateSameDeviceOrdinal("rms_norm_fwd", {input, weight, output});

            if (input->shape().ndims() < 1)
            {
                throw std::runtime_error("rms_norm_fwd expects input tensors with at least one dimension");
            }
            if (weight->shape().ndims() != 1)
            {
                throw std::runtime_error("rms_norm_fwd weight must be a vector");
            }

            const auto hidden_size = input->shape().dims().back();
            if (weight->shape()[0] != hidden_size)
            {
                throw std::runtime_error("rms_norm_fwd weight must match input hidden dimension");
            }
            if (output->shape() != input->shape())
            {
                throw std::runtime_error("rms_norm_fwd output shape must match input shape");
            }

            void *input_ptr = input->dataptr();
            void *weight_ptr = weight->dataptr();
            void *output_ptr = output->dataptr();

            const auto hidden_size_64 = hidden_size;
            if (hidden_size_64 > std::numeric_limits<uint32_t>::max())
            {
                throw std::runtime_error("rms_norm_fwd hidden size exceeds supported range");
            }

            const auto hidden_size_u32 = static_cast<uint32_t>(hidden_size_64);
            const auto numel = input->shape().numel();
            if (numel % hidden_size_u32 != 0)
            {
                throw std::runtime_error("rms_norm_fwd input numel must be divisible by hidden size");
            }
            const auto num_rows = static_cast<uint32_t>(numel / hidden_size_u32);

            pi::tensorlib::KernelLaunchArguments arguments{
                .args = {input_ptr, weight_ptr, output_ptr, num_rows, hidden_size_u32, eps,
                         static_cast<void *>(nullptr)},
                .grid_dim_x = num_rows,
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
CreateRmsNormBwdComputeKernelDescriptor(float eps, const kernel_bin_t<kernel_meta_elementwise_t> &kernel,
                                        const pi::tensorlib::DataType dtype)
{
    const std::string kernel_name =
        dtype == pi::tensorlib::DataType::BFLOAT16 ? "rms_norm_bwd_bf16" : "rms_norm_bwd_fp16";
    const std::string dtype_name = pi::tensorlib::GetDataTypeName(dtype);

    return pi::tensorlib::ComputeKernelDescriptor{
        .kernel_name = kernel_name,
        .function_name = kernel.function_name,
        .expected_arg_count = kernel.arg_count,
        .argument_provider = [eps, kernel, dtype,
                              dtype_name](const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                                          const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
            -> pi::tensorlib::KernelLaunchArguments
        {
            if (inputs.size() != 3)
            {
                throw std::runtime_error("rms_norm_bwd expects three inputs");
            }
            if (outputs.size() != 2)
            {
                throw std::runtime_error("rms_norm_bwd expects two outputs");
            }

            const auto &input = inputs[0];
            const auto &weight = inputs[1];
            const auto &upstream = inputs[2];
            const auto &grad_input = outputs[0];
            const auto &x_hat = outputs[1];

            if (input->dtype() != dtype || weight->dtype() != dtype || upstream->dtype() != dtype ||
                grad_input->dtype() != dtype || x_hat->dtype() != dtype)
            {
                throw std::runtime_error("rms_norm_bwd requires " + dtype_name + " tensors");
            }

            if (input->device().device_type != pi::tensorlib::DeviceType::GPU ||
                weight->device().device_type != pi::tensorlib::DeviceType::GPU ||
                upstream->device().device_type != pi::tensorlib::DeviceType::GPU ||
                grad_input->device().device_type != pi::tensorlib::DeviceType::GPU ||
                x_hat->device().device_type != pi::tensorlib::DeviceType::GPU)
            {
                throw std::runtime_error("rms_norm_bwd currently supports GPU tensors only");
            }

            const auto device_ordinal =
                ValidateSameDeviceOrdinal("rms_norm_bwd", {input, weight, upstream, grad_input, x_hat});

            if (input->shape().ndims() < 1)
            {
                throw std::runtime_error("rms_norm_bwd expects input tensors with at least one dimension");
            }
            if (weight->shape().ndims() != 1)
            {
                throw std::runtime_error("rms_norm_bwd weight must be a vector");
            }

            const auto hidden_size = input->shape().dims().back();
            if (weight->shape()[0] != hidden_size)
            {
                throw std::runtime_error("rms_norm_bwd weight must match input hidden dimension");
            }
            if (upstream->shape() != input->shape() || grad_input->shape() != input->shape() ||
                x_hat->shape() != input->shape())
            {
                throw std::runtime_error("rms_norm_bwd expects output shapes to match input shape");
            }

            void *input_ptr = input->dataptr();
            void *weight_ptr = weight->dataptr();
            void *upstream_ptr = upstream->dataptr();
            void *grad_input_ptr = grad_input->dataptr();
            void *xhat_ptr = x_hat->dataptr();

            const auto hidden_size_64 = hidden_size;
            if (hidden_size_64 > std::numeric_limits<uint32_t>::max())
            {
                throw std::runtime_error("rms_norm_bwd hidden size exceeds supported range");
            }

            const auto hidden_size_u32 = static_cast<uint32_t>(hidden_size_64);
            const auto numel = input->shape().numel();
            if (numel % hidden_size_u32 != 0)
            {
                throw std::runtime_error("rms_norm_bwd input numel must be divisible by hidden size");
            }
            const auto num_rows = static_cast<uint32_t>(numel / hidden_size_u32);

            pi::tensorlib::KernelLaunchArguments arguments{
                .args = {input_ptr, weight_ptr, upstream_ptr, grad_input_ptr, xhat_ptr, num_rows, hidden_size_u32,
                         eps, static_cast<void *>(nullptr)},
                .grid_dim_x = num_rows,
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

void RmsNormImplPass::transform(pi::tensorlib::ExecutionPlan &execution_plan)
{
    for (auto &op : execution_plan.entries)
    {
        if (op.op_type != pi::tensorlib::OpType::RMS_NORM_FWD &&
            op.op_type != pi::tensorlib::OpType::RMS_NORM_BWD)
        {
            continue;
        }
        const bool is_bwd = op.op_type == pi::tensorlib::OpType::RMS_NORM_BWD;
        if (!is_bwd && op.inputs.size() != 2)
        {
            throw std::runtime_error("rms_norm_fwd expects two inputs");
        }
        if (!is_bwd && op.outputs.size() != 1)
        {
            throw std::runtime_error("rms_norm_fwd expects a single output");
        }
        if (is_bwd && op.inputs.size() != 3)
        {
            throw std::runtime_error("rms_norm_bwd expects three inputs");
        }
        if (is_bwd && op.outputs.size() != 2)
        {
            throw std::runtime_error("rms_norm_bwd expects two outputs");
        }

        const auto &input = op.inputs[0];
        const auto &weight = op.inputs[1];
        const auto &output = op.outputs[0];

        const auto dtype = input->dtype();
        if (!IsSupportedHalfType(dtype) || weight->dtype() != dtype)
        {
            throw std::runtime_error("rms_norm requires BFLOAT16 or FLOAT16 tensors");
        }

        const auto eps_it = op.attributes.find("eps");
        if (eps_it == op.attributes.end())
        {
            throw std::runtime_error("rms_norm missing eps attribute");
        }
        const auto eps = std::any_cast<float>(eps_it->second);

        op.op_type = std::nullopt;
        if (!is_bwd)
        {
            const auto &out = output;
            if (out->dtype() != dtype)
            {
                throw std::runtime_error("rms_norm_fwd requires BFLOAT16 or FLOAT16 tensors");
            }
            const auto kernel =
                (dtype == pi::tensorlib::DataType::BFLOAT16) ? krms_norm_fwd_bf16 : krms_norm_fwd_fp16;
            op.kernel_descriptor = CreateRmsNormComputeKernelDescriptor(eps, kernel, dtype);
            const uint64_t hidden = input->shape().dims().back();
            const uint64_t elements = input->shape().numel();
            const uint64_t rows = hidden == 0 ? 0 : elements / hidden;
            // square each element, sum reduction, rsqrt, and two multiplies per element
            const uint64_t per_row_flops = 4 * hidden + 1;
            op.flop_estimate = rows * per_row_flops;
        }
        else
        {
            const auto &upstream = op.inputs[2];
            const auto &grad_input = op.outputs[0];
            const auto &x_hat = op.outputs[1];
            if (upstream->dtype() != dtype || grad_input->dtype() != dtype || x_hat->dtype() != dtype)
            {
                throw std::runtime_error("rms_norm_bwd requires BFLOAT16 or FLOAT16 tensors");
            }
            const auto kernel =
                (dtype == pi::tensorlib::DataType::BFLOAT16) ? krms_norm_bwd_bf16 : krms_norm_bwd_fp16;
            op.kernel_descriptor = CreateRmsNormBwdComputeKernelDescriptor(eps, kernel, dtype);
            const uint64_t hidden = input->shape().dims().back();
            const uint64_t elements = input->shape().numel();
            const uint64_t rows = hidden == 0 ? 0 : elements / hidden;
            // sum squares + dot + per-element ops for dx and x_hat
            const uint64_t per_row_flops = 8 * hidden + 2;
            op.flop_estimate = rows * per_row_flops;
        }
    }
}
