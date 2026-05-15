#include "passes.h"

#include "passes/pass_utils.h"
#include <kernels/kernel_binaries.h>

#include <array>
#include <limits>
#include <sstream>

static pi::tensorlib::ComputeKernelDescriptor CreateContiguous4dBf16ComputeKernelDescriptor()
{
    return pi::tensorlib::ComputeKernelDescriptor{
        .kernel_name = "contiguous_4d_bf16",
        .function_name = kcontiguous_4d_bf16.function_name,
        .expected_arg_count = kcontiguous_4d_bf16.arg_count,
        .argument_provider = [](const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                                const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
            -> pi::tensorlib::KernelLaunchArguments
        {
            if (inputs.size() != 1)
            {
                throw std::runtime_error("Invalid number of inputs for contiguous_4d_bf16 kernel");
            }
            if (outputs.size() != 1)
            {
                throw std::runtime_error("Invalid number of outputs for contiguous_4d_bf16 kernel");
            }
            if (inputs[0]->shape().ndims() != 4)
            {
                throw std::runtime_error("Input tensor must be 4D for contiguous_4d_bf16 kernel");
            }
            if (outputs[0]->shape().ndims() != 4)
            {
                throw std::runtime_error("Output tensor must be 4D for contiguous_4d_bf16 kernel");
            }
            const auto &input = inputs[0];
            const auto &output = outputs[0];

            const auto device_ordinal =
                ValidateSameDeviceOrdinal("contiguous_4d_bf16", {input, output});

            void *input_ptr = input->dataptr();   // input_ptr
            void *output_ptr = output->dataptr(); // output_ptr

            const auto stride_x = static_cast<uint32_t>(input->strides()[0]);
            const auto stride_y = static_cast<uint32_t>(input->strides()[1]);
            const auto stride_z = static_cast<uint32_t>(input->strides()[2]);
            const auto stride_w = static_cast<uint32_t>(input->strides()[3]);

            const auto x = static_cast<uint32_t>(input->shape()[0]);
            const auto y = static_cast<uint32_t>(input->shape()[1]);
            const auto z = static_cast<uint32_t>(input->shape()[2]);
            const auto w = static_cast<uint32_t>(input->shape()[3]);

            pi::tensorlib::KernelLaunchArguments arguments{
                .args =
                    {
                        output_ptr, input_ptr,

                        // strides
                        stride_x, stride_y, stride_z, stride_w,

                        // shape
                        x, y, z, w,

                        static_cast<void *>(nullptr), /* workspace */
                    },
                .grid_dim_x = CEIL_DIV(x, kcontiguous_4d_bf16.meta.block_size_x),
                .grid_dim_y =
                    CEIL_DIV(y, kcontiguous_4d_bf16.meta.block_size_y) *
                    CEIL_DIV(w, kcontiguous_4d_bf16.meta.block_size_w * kcontiguous_4d_bf16.meta.block_size_z),
                .grid_dim_z = z,
                .block_dim_x = TRITON_WARP_SIZE * kcontiguous_4d_bf16.num_warps,
                .block_dim_y = 1,
                .block_dim_z = 1,
                .shared_mem_bytes = kcontiguous_4d_bf16.shared_mem_bytes,
                .device_ordinal = device_ordinal};
            return arguments;
        },
        .cuda_descriptor = pi::tensorlib::MakeCudaKernelDescriptor(kcontiguous_4d_bf16.data, kcontiguous_4d_bf16.size),
        .hip_descriptor = pi::tensorlib::MakeHipKernelDescriptor(kcontiguous_4d_bf16.data, kcontiguous_4d_bf16.size),
    };
}

static pi::tensorlib::ComputeKernelDescriptor CreateContiguous4dFp16ComputeKernelDescriptor()
{
    return pi::tensorlib::ComputeKernelDescriptor{
        .kernel_name = "contiguous_4d_fp16",
        .function_name = kcontiguous_4d_fp16.function_name,
        .expected_arg_count = kcontiguous_4d_fp16.arg_count,
        .argument_provider =
            [](const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
               const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
            -> pi::tensorlib::KernelLaunchArguments
        {
            if (inputs.size() != 1)
            {
                throw std::runtime_error("Invalid number of inputs for contiguous_4d_fp16 kernel");
            }
            if (outputs.size() != 1)
            {
                throw std::runtime_error("Invalid number of outputs for contiguous_4d_fp16 kernel");
            }
            if (inputs[0]->shape().ndims() != 4)
            {
                throw std::runtime_error("Input tensor must be 4D for contiguous_4d_fp16 kernel");
            }
            if (outputs[0]->shape().ndims() != 4)
            {
                throw std::runtime_error("Output tensor must be 4D for contiguous_4d_fp16 kernel");
            }

            const auto &input = inputs[0];
            const auto &output = outputs[0];

            const auto device_ordinal =
                ValidateSameDeviceOrdinal("contiguous_4d_fp16", {input, output});

            void *input_ptr = input->dataptr();
            void *output_ptr = output->dataptr();

            const auto stride_x = static_cast<uint32_t>(input->strides()[0]);
            const auto stride_y = static_cast<uint32_t>(input->strides()[1]);
            const auto stride_z = static_cast<uint32_t>(input->strides()[2]);
            const auto stride_w = static_cast<uint32_t>(input->strides()[3]);

            const auto x = static_cast<uint32_t>(input->shape()[0]);
            const auto y = static_cast<uint32_t>(input->shape()[1]);
            const auto z = static_cast<uint32_t>(input->shape()[2]);
            const auto w = static_cast<uint32_t>(input->shape()[3]);

            pi::tensorlib::KernelLaunchArguments arguments{
                .args =
                    {
                        output_ptr, input_ptr,

                        // strides
                        stride_x, stride_y, stride_z, stride_w,

                        // shape
                        x, y, z, w,

                        static_cast<void *>(nullptr), /* workspace pointers (not used) */
                    },
                .grid_dim_x = CEIL_DIV(x, kcontiguous_4d_fp16.meta.block_size_x),
                .grid_dim_y =
                    CEIL_DIV(y, kcontiguous_4d_fp16.meta.block_size_y) *
                    CEIL_DIV(w, kcontiguous_4d_fp16.meta.block_size_w * kcontiguous_4d_fp16.meta.block_size_z),
                .grid_dim_z = z,
                .block_dim_x = TRITON_WARP_SIZE * kcontiguous_4d_fp16.num_warps,
                .block_dim_y = 1,
                .block_dim_z = 1,
                .shared_mem_bytes = kcontiguous_4d_fp16.shared_mem_bytes,
                .device_ordinal = device_ordinal};
            return arguments;
        },
        .cuda_descriptor = pi::tensorlib::MakeCudaKernelDescriptor(kcontiguous_4d_fp16.data, kcontiguous_4d_fp16.size),
        .hip_descriptor = pi::tensorlib::MakeHipKernelDescriptor(kcontiguous_4d_fp16.data, kcontiguous_4d_fp16.size),
    };
}

static pi::tensorlib::ComputeKernelDescriptor CreateContiguous3dBf16ComputeKernelDescriptor()
{
    return pi::tensorlib::ComputeKernelDescriptor{
        .kernel_name = "contiguous_3d_bf16",
        .function_name = kcontiguous_3d_bf16.function_name,
        .expected_arg_count = kcontiguous_3d_bf16.arg_count,
        .argument_provider = [](const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                                const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
            -> pi::tensorlib::KernelLaunchArguments
        {
            if (inputs.size() != 1)
            {
                throw std::runtime_error("Invalid number of inputs for contiguous_3d_bf16 kernel");
            }
            if (outputs.size() != 1)
            {
                throw std::runtime_error("Invalid number of outputs for contiguous_3d_bf16 kernel");
            }
            if (inputs[0]->shape().ndims() != 3)
            {
                throw std::runtime_error("Input tensor must be 3D for contiguous_3d_bf16 kernel");
            }
            if (outputs[0]->shape().ndims() != 3)
            {
                throw std::runtime_error("Output tensor must be 3D for contiguous_3d_bf16 kernel");
            }

            const auto &input = inputs[0];
            const auto &output = outputs[0];

            const auto device_ordinal =
                ValidateSameDeviceOrdinal("contiguous_3d_bf16", {input, output});

            void *input_ptr = input->dataptr();
            void *output_ptr = output->dataptr();

            const auto stride_x = static_cast<uint32_t>(input->strides()[0]);
            const auto stride_y = static_cast<uint32_t>(input->strides()[1]);
            const auto stride_z = static_cast<uint32_t>(input->strides()[2]);

            const auto x = static_cast<uint32_t>(input->shape()[0]);
            const auto y = static_cast<uint32_t>(input->shape()[1]);
            const auto z = static_cast<uint32_t>(input->shape()[2]);

            pi::tensorlib::KernelLaunchArguments arguments{
                .args =
                    {
                        output_ptr,
                        input_ptr,

                        stride_x,
                        stride_y,
                        stride_z,

                        x,
                        y,
                        z,

                        static_cast<void *>(nullptr),
                    },
                .grid_dim_x = CEIL_DIV(x, kcontiguous_3d_bf16.meta.block_size_x),
                .grid_dim_y = CEIL_DIV(y, kcontiguous_3d_bf16.meta.block_size_y) *
                              CEIL_DIV(z, kcontiguous_3d_bf16.meta.block_size_z),
                .grid_dim_z = 1,
                .block_dim_x = TRITON_WARP_SIZE * kcontiguous_3d_bf16.num_warps,
                .block_dim_y = 1,
                .block_dim_z = 1,
                .shared_mem_bytes = kcontiguous_3d_bf16.shared_mem_bytes,
                .device_ordinal = device_ordinal};
            return arguments;
        },
        .cuda_descriptor = pi::tensorlib::MakeCudaKernelDescriptor(kcontiguous_3d_bf16.data, kcontiguous_3d_bf16.size),
        .hip_descriptor = pi::tensorlib::MakeHipKernelDescriptor(kcontiguous_3d_bf16.data, kcontiguous_3d_bf16.size),
    };
}

static pi::tensorlib::ComputeKernelDescriptor CreateContiguous3dFp16ComputeKernelDescriptor()
{
    return pi::tensorlib::ComputeKernelDescriptor{
        .kernel_name = "contiguous_3d_fp16",
        .function_name = kcontiguous_3d_fp16.function_name,
        .expected_arg_count = kcontiguous_3d_fp16.arg_count,
        .argument_provider = [](const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                                const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
            -> pi::tensorlib::KernelLaunchArguments
        {
            if (inputs.size() != 1)
            {
                throw std::runtime_error("Invalid number of inputs for contiguous_3d_fp16 kernel");
            }
            if (outputs.size() != 1)
            {
                throw std::runtime_error("Invalid number of outputs for contiguous_3d_fp16 kernel");
            }
            if (inputs[0]->shape().ndims() != 3)
            {
                throw std::runtime_error("Input tensor must be 3D for contiguous_3d_fp16 kernel");
            }
            if (outputs[0]->shape().ndims() != 3)
            {
                throw std::runtime_error("Output tensor must be 3D for contiguous_3d_fp16 kernel");
            }

            const auto &input = inputs[0];
            const auto &output = outputs[0];

            const auto device_ordinal =
                ValidateSameDeviceOrdinal("contiguous_3d_fp16", {input, output});

            void *input_ptr = input->dataptr();
            void *output_ptr = output->dataptr();

            const auto stride_x = static_cast<uint32_t>(input->strides()[0]);
            const auto stride_y = static_cast<uint32_t>(input->strides()[1]);
            const auto stride_z = static_cast<uint32_t>(input->strides()[2]);

            const auto x = static_cast<uint32_t>(input->shape()[0]);
            const auto y = static_cast<uint32_t>(input->shape()[1]);
            const auto z = static_cast<uint32_t>(input->shape()[2]);

            pi::tensorlib::KernelLaunchArguments arguments{
                .args =
                    {
                        output_ptr,
                        input_ptr,

                        stride_x,
                        stride_y,
                        stride_z,

                        x,
                        y,
                        z,

                        static_cast<void *>(nullptr),
                    },
                .grid_dim_x = CEIL_DIV(x, kcontiguous_3d_fp16.meta.block_size_x),
                .grid_dim_y = CEIL_DIV(y, kcontiguous_3d_fp16.meta.block_size_y) *
                              CEIL_DIV(z, kcontiguous_3d_fp16.meta.block_size_z),
                .grid_dim_z = 1,
                .block_dim_x = TRITON_WARP_SIZE * kcontiguous_3d_fp16.num_warps,
                .block_dim_y = 1,
                .block_dim_z = 1,
                .shared_mem_bytes = kcontiguous_3d_fp16.shared_mem_bytes,
                .device_ordinal = device_ordinal};
            return arguments;
        },
        .cuda_descriptor = pi::tensorlib::MakeCudaKernelDescriptor(kcontiguous_3d_fp16.data, kcontiguous_3d_fp16.size),
        .hip_descriptor = pi::tensorlib::MakeHipKernelDescriptor(kcontiguous_3d_fp16.data, kcontiguous_3d_fp16.size),
    };
}

static pi::tensorlib::ComputeKernelDescriptor CreateContiguous2dBf16ComputeKernelDescriptor()
{
    return pi::tensorlib::ComputeKernelDescriptor{
        .kernel_name = "contiguous_2d_bf16",
        .function_name = kcontiguous_3d_bf16.function_name,
        .expected_arg_count = kcontiguous_3d_bf16.arg_count,
        .argument_provider = [](const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                                const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
            -> pi::tensorlib::KernelLaunchArguments
        {
            if (inputs.size() != 1 || outputs.size() != 1)
            {
                throw std::runtime_error("Invalid number of inputs/outputs for contiguous_2d_bf16 kernel");
            }
            if (inputs[0]->shape().ndims() != 2 || outputs[0]->shape().ndims() != 2)
            {
                throw std::runtime_error("contiguous_2d_bf16 expects 2D tensors");
            }

            const auto &input = inputs[0];
            const auto &output = outputs[0];
            const auto device_ordinal = ValidateSameDeviceOrdinal("contiguous_2d_bf16", {input, output});

            void *input_ptr = input->dataptr();
            void *output_ptr = output->dataptr();

            const auto stride_x = static_cast<uint32_t>(0);
            const auto stride_y = static_cast<uint32_t>(input->strides()[0]);
            const auto stride_z = static_cast<uint32_t>(input->strides()[1]);

            const auto x = static_cast<uint32_t>(1);
            const auto y = static_cast<uint32_t>(input->shape()[0]);
            const auto z = static_cast<uint32_t>(input->shape()[1]);

            return pi::tensorlib::KernelLaunchArguments{
                .args = {output_ptr, input_ptr, stride_x, stride_y, stride_z, x, y, z, static_cast<void *>(nullptr)},
                .grid_dim_x = CEIL_DIV(x, kcontiguous_3d_bf16.meta.block_size_x),
                .grid_dim_y = CEIL_DIV(y, kcontiguous_3d_bf16.meta.block_size_y) *
                              CEIL_DIV(z, kcontiguous_3d_bf16.meta.block_size_z),
                .grid_dim_z = 1,
                .block_dim_x = TRITON_WARP_SIZE * kcontiguous_3d_bf16.num_warps,
                .block_dim_y = 1,
                .block_dim_z = 1,
                .shared_mem_bytes = kcontiguous_3d_bf16.shared_mem_bytes,
                .device_ordinal = device_ordinal};
        },
        .cuda_descriptor = pi::tensorlib::MakeCudaKernelDescriptor(kcontiguous_3d_bf16.data, kcontiguous_3d_bf16.size),
        .hip_descriptor = pi::tensorlib::MakeHipKernelDescriptor(kcontiguous_3d_bf16.data, kcontiguous_3d_bf16.size),
    };
}

static pi::tensorlib::ComputeKernelDescriptor CreateContiguous2dFp16ComputeKernelDescriptor()
{
    return pi::tensorlib::ComputeKernelDescriptor{
        .kernel_name = "contiguous_2d_fp16",
        .function_name = kcontiguous_3d_fp16.function_name,
        .expected_arg_count = kcontiguous_3d_fp16.arg_count,
        .argument_provider = [](const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                                const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
            -> pi::tensorlib::KernelLaunchArguments
        {
            if (inputs.size() != 1 || outputs.size() != 1)
            {
                throw std::runtime_error("Invalid number of inputs/outputs for contiguous_2d_fp16 kernel");
            }
            if (inputs[0]->shape().ndims() != 2 || outputs[0]->shape().ndims() != 2)
            {
                throw std::runtime_error("contiguous_2d_fp16 expects 2D tensors");
            }

            const auto &input = inputs[0];
            const auto &output = outputs[0];
            const auto device_ordinal = ValidateSameDeviceOrdinal("contiguous_2d_fp16", {input, output});

            void *input_ptr = input->dataptr();
            void *output_ptr = output->dataptr();

            const auto stride_x = static_cast<uint32_t>(0);
            const auto stride_y = static_cast<uint32_t>(input->strides()[0]);
            const auto stride_z = static_cast<uint32_t>(input->strides()[1]);

            const auto x = static_cast<uint32_t>(1);
            const auto y = static_cast<uint32_t>(input->shape()[0]);
            const auto z = static_cast<uint32_t>(input->shape()[1]);

            return pi::tensorlib::KernelLaunchArguments{
                .args = {output_ptr, input_ptr, stride_x, stride_y, stride_z, x, y, z, static_cast<void *>(nullptr)},
                .grid_dim_x = CEIL_DIV(x, kcontiguous_3d_fp16.meta.block_size_x),
                .grid_dim_y = CEIL_DIV(y, kcontiguous_3d_fp16.meta.block_size_y) *
                              CEIL_DIV(z, kcontiguous_3d_fp16.meta.block_size_z),
                .grid_dim_z = 1,
                .block_dim_x = TRITON_WARP_SIZE * kcontiguous_3d_fp16.num_warps,
                .block_dim_y = 1,
                .block_dim_z = 1,
                .shared_mem_bytes = kcontiguous_3d_fp16.shared_mem_bytes,
                .device_ordinal = device_ordinal};
        },
        .cuda_descriptor = pi::tensorlib::MakeCudaKernelDescriptor(kcontiguous_3d_fp16.data, kcontiguous_3d_fp16.size),
        .hip_descriptor = pi::tensorlib::MakeHipKernelDescriptor(kcontiguous_3d_fp16.data, kcontiguous_3d_fp16.size),
    };
}
void ContiguousImplPass::transform(pi::tensorlib::ExecutionPlan &execution_plan)
{
    for (auto &fill_op : execution_plan.entries)
    {
        if (fill_op.op_type == pi::tensorlib::OpType::CONTIGUOUS)
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
            const auto dtype = input->dtype();
            if (dtype != pi::tensorlib::DataType::BFLOAT16 && dtype != pi::tensorlib::DataType::FLOAT16)
            {
                continue;
            }
            if (input->shape().ndims() == 2)
            {
                fill_op.op_type = std::nullopt; // mark as kernel
                fill_op.kernel_descriptor = (dtype == pi::tensorlib::DataType::BFLOAT16)
                                                ? CreateContiguous2dBf16ComputeKernelDescriptor()
                                                : CreateContiguous2dFp16ComputeKernelDescriptor();
                fill_op.flop_estimate = 0;
                continue;
            }
            if (input->shape().ndims() == 3)
            {
                fill_op.op_type = std::nullopt; // mark as kernel
                fill_op.kernel_descriptor = (dtype == pi::tensorlib::DataType::BFLOAT16)
                                                ? CreateContiguous3dBf16ComputeKernelDescriptor()
                                                : CreateContiguous3dFp16ComputeKernelDescriptor();
                fill_op.flop_estimate = 0;
                continue;
            }
            if (input->shape().ndims() == 4)
            {
                fill_op.op_type = std::nullopt; // mark as kernel
                fill_op.kernel_descriptor = (dtype == pi::tensorlib::DataType::BFLOAT16)
                                                ? CreateContiguous4dBf16ComputeKernelDescriptor()
                                                : CreateContiguous4dFp16ComputeKernelDescriptor();
                fill_op.flop_estimate = 0;
                continue;
            }
            // fallback: use generic device copy for other ranks.
            fill_op.op_type = pi::tensorlib::OpType::DEVICE_COPY;
            if (!fill_op.gpu_stream_desc.isValid())
            {
                fill_op.gpu_stream_desc = pi::tensorlib::GpuStreamDescriptors::Main;
            }
            fill_op.kernel_descriptor.reset();
        }
    }
}
