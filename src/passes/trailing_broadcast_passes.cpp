#include "passes.h"
#include "shape_utils.h"

#include <kernels/kernel_binaries.h>
#include "passes/pass_utils.h"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_set>

static pi::tensorlib::ComputeKernelDescriptor
CreateAddElementwiseComputeKernelDescriptor(const pi::tensorlib::DataType dtype)
{
    const auto kernel = SelectKernelForAdd(kadd_elementwise_bf16, kadd_elementwise_fp16, kadd_elementwise_fp32, dtype);
    const std::string kernel_name =
        std::string("add_elementwise_") + std::string(KernelSuffixForAdd(dtype));
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
                throw std::runtime_error("add_elementwise expects exactly two inputs");
            }
            if (outputs.size() != 1)
            {
                throw std::runtime_error("add_elementwise expects exactly one output");
            }

            const auto &lhs = inputs[0];
            const auto &rhs = inputs[1];
            const auto &output = outputs[0];

            if (lhs->dtype() != dtype || rhs->dtype() != dtype || output->dtype() != dtype)
            {
                throw std::runtime_error("add_elementwise only supports " + dtype_name + " tensors");
            }

            if (lhs->device().device_type != pi::tensorlib::DeviceType::GPU ||
                rhs->device().device_type != pi::tensorlib::DeviceType::GPU ||
                output->device().device_type != pi::tensorlib::DeviceType::GPU)
            {
                throw std::runtime_error("add_elementwise requires GPU tensors");
            }

            if (lhs->shape().ndims() != rhs->shape().ndims() || lhs->shape().ndims() != output->shape().ndims())
            {
                throw std::runtime_error("add_elementwise expects operands with matching rank");
            }

            for (size_t dim = 0; dim < lhs->shape().ndims(); ++dim)
            {
                if (lhs->shape()[dim] != rhs->shape()[dim] || lhs->shape()[dim] != output->shape()[dim])
                {
                    throw std::runtime_error("add_elementwise expects operands with matching shape");
                }
            }

            const auto numel64 = output->shape().numel();
            if (numel64 == 0)
            {
                throw std::runtime_error("add_elementwise tensor must contain at least one element");
            }
            if (numel64 > std::numeric_limits<uint32_t>::max())
            {
                throw std::runtime_error("add_elementwise tensor size exceeds supported range");
            }

            if (!pi::tensorlib::shape_utils::IsRowMajorContiguous(lhs) || !pi::tensorlib::shape_utils::IsRowMajorContiguous(rhs) || !pi::tensorlib::shape_utils::IsRowMajorContiguous(output))
            {
                throw std::runtime_error("add_elementwise requires contiguous tensors");
            }

            const auto device_ordinal =
                ValidateSameDeviceOrdinal("add_elementwise", {lhs, rhs, output});

            void *out_ptr = output->dataptr();
            void *lhs_ptr = lhs->dataptr();
            void *rhs_ptr = rhs->dataptr();
            const auto n_elements = static_cast<uint32_t>(numel64);

            pi::tensorlib::KernelLaunchArguments arguments{
                .args = {out_ptr, lhs_ptr, rhs_ptr, n_elements, static_cast<void *>(nullptr)},
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
CreateSqrtElementwiseComputeKernelDescriptor(const pi::tensorlib::DataType dtype)
{
    const auto kernel =
        SelectKernelForAdd(ksqrt_elementwise_bf16, ksqrt_elementwise_fp16, ksqrt_elementwise_fp32, dtype);
    const std::string kernel_name = std::string("sqrt_elementwise_") + std::string(KernelSuffixForAdd(dtype));
    const std::string dtype_name = pi::tensorlib::GetDataTypeName(dtype);

    return pi::tensorlib::ComputeKernelDescriptor{
        .kernel_name = kernel_name,
        .function_name = kernel.function_name,
        .expected_arg_count = kernel.arg_count,
        .argument_provider = [kernel, dtype, dtype_name, kernel_name](
                                 const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                                 const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
            -> pi::tensorlib::KernelLaunchArguments
        {
            if (inputs.size() != 1)
            {
                throw std::runtime_error(kernel_name + " expects exactly one input");
            }
            if (outputs.size() != 1)
            {
                throw std::runtime_error(kernel_name + " expects exactly one output");
            }

            const auto &input = inputs[0];
            const auto &output = outputs[0];

            if (input->dtype() != dtype || output->dtype() != dtype)
            {
                throw std::runtime_error(kernel_name + " only supports " + dtype_name + " tensors");
            }

            if (input->device().device_type != pi::tensorlib::DeviceType::GPU ||
                output->device().device_type != pi::tensorlib::DeviceType::GPU)
            {
                throw std::runtime_error(kernel_name + " requires GPU tensors");
            }

            if (input->shape().ndims() != output->shape().ndims())
            {
                throw std::runtime_error(kernel_name + " expects operands with matching rank");
            }

            for (size_t dim = 0; dim < input->shape().ndims(); ++dim)
            {
                if (input->shape()[dim] != output->shape()[dim])
                {
                    throw std::runtime_error(kernel_name + " expects operands with matching shape");
                }
            }

            const auto numel64 = output->shape().numel();
            if (numel64 == 0)
            {
                throw std::runtime_error(kernel_name + " tensor must contain at least one element");
            }
            if (numel64 > std::numeric_limits<uint32_t>::max())
            {
                throw std::runtime_error(kernel_name + " tensor size exceeds supported range");
            }

            if (!pi::tensorlib::shape_utils::IsRowMajorContiguous(input) ||
                !pi::tensorlib::shape_utils::IsRowMajorContiguous(output))
            {
                throw std::runtime_error(kernel_name + " requires contiguous tensors");
            }

            const auto device_ordinal = ValidateSameDeviceOrdinal(kernel_name, {input, output});

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

static pi::tensorlib::ComputeKernelDescriptor CreateAddElementwiseFp32OutFp16ComputeKernelDescriptor()
{
    const auto kernel = kadd_elementwise_fp32_out_fp16;
    const std::string kernel_name = "add_elementwise_fp32_out_fp16";

    return pi::tensorlib::ComputeKernelDescriptor{
        .kernel_name = kernel_name,
        .function_name = std::string(kernel.function_name),
        .expected_arg_count = kernel.arg_count,
        .argument_provider = [kernel, kernel_name](
                                 const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                                 const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
            -> pi::tensorlib::KernelLaunchArguments
        {
            if (inputs.size() != 2)
            {
                throw std::runtime_error(kernel_name + " expects exactly two inputs");
            }
            if (outputs.size() != 1)
            {
                throw std::runtime_error(kernel_name + " expects exactly one output");
            }

            const auto &lhs = inputs[0];
            const auto &rhs = inputs[1];
            const auto &output = outputs[0];

            if (lhs->dtype() != pi::tensorlib::DataType::FLOAT32 || rhs->dtype() != pi::tensorlib::DataType::FLOAT32 ||
                output->dtype() != pi::tensorlib::DataType::FLOAT16)
            {
                throw std::runtime_error(kernel_name + " expects fp32 inputs and fp16 output");
            }

            if (lhs->device().device_type != pi::tensorlib::DeviceType::GPU ||
                rhs->device().device_type != pi::tensorlib::DeviceType::GPU ||
                output->device().device_type != pi::tensorlib::DeviceType::GPU)
            {
                throw std::runtime_error(kernel_name + " requires GPU tensors");
            }

            if (lhs->shape().ndims() != rhs->shape().ndims() || lhs->shape().ndims() != output->shape().ndims())
            {
                throw std::runtime_error(kernel_name + " expects operands with matching rank");
            }

            for (size_t dim = 0; dim < lhs->shape().ndims(); ++dim)
            {
                if (lhs->shape()[dim] != rhs->shape()[dim] || lhs->shape()[dim] != output->shape()[dim])
                {
                    throw std::runtime_error(kernel_name + " expects operands with matching shape");
                }
            }

            const auto numel64 = output->shape().numel();
            if (numel64 == 0)
            {
                throw std::runtime_error(kernel_name + " tensor must contain at least one element");
            }
            if (numel64 > std::numeric_limits<uint32_t>::max())
            {
                throw std::runtime_error(kernel_name + " tensor size exceeds supported range");
            }

            if (!pi::tensorlib::shape_utils::IsRowMajorContiguous(lhs) ||
                !pi::tensorlib::shape_utils::IsRowMajorContiguous(rhs) ||
                !pi::tensorlib::shape_utils::IsRowMajorContiguous(output))
            {
                throw std::runtime_error(kernel_name + " requires contiguous tensors");
            }

            const auto device_ordinal = ValidateSameDeviceOrdinal(kernel_name, {lhs, rhs, output});

            void *out_ptr = output->dataptr();
            void *lhs_ptr = lhs->dataptr();
            void *rhs_ptr = rhs->dataptr();
            const auto n_elements = static_cast<uint32_t>(numel64);

            pi::tensorlib::KernelLaunchArguments arguments{
                .args = {out_ptr, lhs_ptr, rhs_ptr, n_elements, static_cast<void *>(nullptr)},
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

template <typename KernelT>
static pi::tensorlib::ComputeKernelDescriptor
CreateTrailingBroadcastDescriptor(const KernelT &kernel, const std::string &kernel_name, const std::string &op_name,
                                  const std::string &rhs_label, const pi::tensorlib::DataType dtype)
{
    const std::string dtype_name = pi::tensorlib::GetDataTypeName(dtype);

    return pi::tensorlib::ComputeKernelDescriptor{
        .kernel_name = kernel_name,
        .function_name = kernel.function_name,
        .expected_arg_count = kernel.arg_count,
        .argument_provider = [kernel, dtype, dtype_name, op_name, rhs_label](
                                 const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                                 const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
            -> pi::tensorlib::KernelLaunchArguments
        {
            if (inputs.size() != 2)
            {
                throw std::runtime_error(op_name + " expects exactly two inputs");
            }
            if (outputs.size() != 1)
            {
                throw std::runtime_error(op_name + " expects exactly one output");
            }

            const auto &lhs = inputs[0];
            const auto &rhs = inputs[1];
            const auto &output = outputs[0];

            if (lhs->dtype() != dtype || rhs->dtype() != dtype || output->dtype() != dtype)
            {
                throw std::runtime_error(op_name + " only supports " + dtype_name + " tensors");
            }

            if (lhs->device().device_type != pi::tensorlib::DeviceType::GPU ||
                rhs->device().device_type != pi::tensorlib::DeviceType::GPU ||
                output->device().device_type != pi::tensorlib::DeviceType::GPU)
            {
                throw std::runtime_error(op_name + " requires GPU tensors");
            }

            if (lhs->shape().ndims() != 2 || output->shape().ndims() != 2)
            {
                throw std::runtime_error(op_name + " expects 2D activation tensors");
            }
            const bool rhs_rank1 = rhs->shape().ndims() == 1;
            const bool rhs_rank2 = rhs->shape().ndims() == 2 && rhs->shape()[0] == 1;
            if (!rhs_rank1 && !rhs_rank2)
            {
                throw std::runtime_error(op_name + " expects 1D or (1, cols) " + rhs_label + " tensor");
            }

            if (lhs->shape()[0] != output->shape()[0] || lhs->shape()[1] != output->shape()[1])
            {
                throw std::runtime_error(op_name + " activation input/output shape mismatch");
            }
            const auto cols64 = output->shape()[1];
            if ((rhs_rank1 && rhs->shape()[0] != cols64) || (rhs_rank2 && rhs->shape()[1] != cols64))
            {
                throw std::runtime_error(op_name + " " + rhs_label + " must match trailing dimension");
            }

            const auto rows64 = output->shape()[0];
            if (rows64 == 0 || cols64 == 0)
            {
                throw std::runtime_error(op_name + " tensor dimensions must be non-zero");
            }
            if (rows64 > std::numeric_limits<uint32_t>::max() || cols64 > std::numeric_limits<uint32_t>::max())
            {
                throw std::runtime_error(op_name + " shape exceeds supported range");
            }

            const auto stride_in_row64 = lhs->strides()[0];
            const auto stride_in_col64 = lhs->strides()[1];
            const auto stride_out_row64 = output->strides()[0];
            const auto stride_out_col64 = output->strides()[1];
            const auto stride_rhs0_64 = rhs_rank1 ? rhs->strides()[0] : rhs->strides()[1];
            const auto stride_rhs_leading64 = rhs_rank2 ? rhs->strides()[0] : 0;

            if (stride_in_row64 != stride_out_row64 || stride_in_col64 != stride_out_col64)
            {
                throw std::runtime_error(op_name + " requires matching input/output layout");
            }
            if (stride_out_col64 != 1)
            {
                throw std::runtime_error(op_name + " expects contiguous inner dimension");
            }
            if (rhs_rank1)
            {
                if (stride_rhs0_64 != 1)
                {
                    throw std::runtime_error(op_name + " expects contiguous " + rhs_label + " tensor");
                }
            }
            else
            {
                if (stride_rhs0_64 != 1 || stride_rhs_leading64 != static_cast<int64_t>(cols64))
                {
                    throw std::runtime_error(op_name + " expects contiguous " + rhs_label + " tensor");
                }
            }

            if (stride_out_row64 > std::numeric_limits<uint32_t>::max())
            {
                throw std::runtime_error(op_name + " stride exceeds supported range");
            }
            if (stride_rhs0_64 > std::numeric_limits<uint32_t>::max())
            {
                throw std::runtime_error(op_name + " " + rhs_label + " stride exceeds supported range");
            }

            const auto device_ordinal = ValidateSameDeviceOrdinal(op_name, {lhs, rhs, output});

            void *out_ptr = output->dataptr();
            void *lhs_ptr = lhs->dataptr();
            void *rhs_ptr = rhs->dataptr();

            const auto rows = static_cast<uint32_t>(rows64);
            const auto cols = static_cast<uint32_t>(cols64);
            const auto stride_row = static_cast<uint32_t>(stride_out_row64);
            const auto stride_col = static_cast<uint32_t>(stride_out_col64);
            const auto stride_rhs = static_cast<uint32_t>(stride_rhs0_64);

            pi::tensorlib::KernelLaunchArguments arguments{
                .args = {out_ptr, lhs_ptr, rhs_ptr, rows, cols, stride_row, stride_col, stride_rhs,
                         static_cast<void *>(nullptr)},
                .grid_dim_x = rows,
                .grid_dim_y = CEIL_DIV(cols, kernel.meta.block_size),
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
CreateAddTrailingBroadcastComputeKernelDescriptor(const pi::tensorlib::DataType dtype)
{
    const auto kernel = SelectKernelForAdd(kadd_trailing_broadcast_bf16, kadd_trailing_broadcast_fp16,
                                           kadd_trailing_broadcast_fp32, dtype);
    const std::string kernel_name = kernel.kernel_name;
    return CreateTrailingBroadcastDescriptor(kernel, kernel_name, "add_trailing_broadcast", "bias", dtype);
}

static pi::tensorlib::ComputeKernelDescriptor
CreateMulElementwiseComputeKernelDescriptor(const pi::tensorlib::DataType dtype)
{
    const auto kernel = SelectKernelForAdd(kmul_elementwise_bf16, kmul_elementwise_fp16, kmul_elementwise_fp32, dtype);
    const std::string kernel_name =
        std::string("mul_elementwise_") + std::string(KernelSuffixForAdd(dtype));
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
                throw std::runtime_error("mul_elementwise expects exactly two inputs");
            }
            if (outputs.size() != 1)
            {
                throw std::runtime_error("mul_elementwise expects exactly one output");
            }

            const auto &lhs = inputs[0];
            const auto &rhs = inputs[1];
            const auto &output = outputs[0];

            if (lhs->dtype() != dtype || rhs->dtype() != dtype || output->dtype() != dtype)
            {
                throw std::runtime_error("mul_elementwise only supports " + dtype_name + " tensors");
            }

            if (lhs->device().device_type != pi::tensorlib::DeviceType::GPU ||
                rhs->device().device_type != pi::tensorlib::DeviceType::GPU ||
                output->device().device_type != pi::tensorlib::DeviceType::GPU)
            {
                throw std::runtime_error("mul_elementwise requires GPU tensors");
            }

            if (lhs->shape().ndims() != rhs->shape().ndims() || lhs->shape().ndims() != output->shape().ndims())
            {
                throw std::runtime_error("mul_elementwise expects operands with matching rank");
            }

            for (size_t dim = 0; dim < lhs->shape().ndims(); ++dim)
            {
                if (lhs->shape()[dim] != rhs->shape()[dim] || lhs->shape()[dim] != output->shape()[dim])
                {
                    throw std::runtime_error("mul_elementwise expects operands with matching shape");
                }
            }

            const auto numel64 = output->shape().numel();
            if (numel64 == 0)
            {
                throw std::runtime_error("mul_elementwise tensor must contain at least one element");
            }
            if (numel64 > std::numeric_limits<uint32_t>::max())
            {
                throw std::runtime_error("mul_elementwise tensor size exceeds supported range");
            }

            if (!pi::tensorlib::shape_utils::IsRowMajorContiguous(lhs) || !pi::tensorlib::shape_utils::IsRowMajorContiguous(rhs) || !pi::tensorlib::shape_utils::IsRowMajorContiguous(output))
            {
                throw std::runtime_error("mul_elementwise requires contiguous tensors");
            }

            const auto device_ordinal =
                ValidateSameDeviceOrdinal("mul_elementwise", {lhs, rhs, output});

            void *out_ptr = output->dataptr();
            void *lhs_ptr = lhs->dataptr();
            void *rhs_ptr = rhs->dataptr();
            const auto n_elements = static_cast<uint32_t>(numel64);

            if (const char *env = std::getenv("FBAMTRAIN_DEBUG_MUL_ELEMENTWISE"); env != nullptr && env[0] != '\0')
            {
                std::clog << "[MulElementwise] out_id=" << output->id() << " out_ptr=" << out_ptr
                          << " lhs_id=" << lhs->id() << " rhs_id=" << rhs->id()
                          << " numel=" << numel64 << '\n';
            }

            pi::tensorlib::KernelLaunchArguments arguments{
                .args = {out_ptr, lhs_ptr, rhs_ptr, n_elements, static_cast<void *>(nullptr)},
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
CreateMulScalarComputeKernelDescriptor(const pi::tensorlib::DataType dtype)
{
    const auto kernel = SelectKernelForAdd(kmul_scalar_bf16, kmul_scalar_fp16, kmul_scalar_fp32, dtype);
    const std::string kernel_name = std::string("mul_scalar_") + std::string(KernelSuffixForAdd(dtype));
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
                throw std::runtime_error("mul_scalar expects exactly two inputs");
            }
            if (outputs.size() != 1)
            {
                throw std::runtime_error("mul_scalar expects exactly one output");
            }

            const auto &lhs = inputs[0];
            const auto &rhs = inputs[1];
            const auto &output = outputs[0];

            if (lhs->dtype() != dtype || rhs->dtype() != dtype || output->dtype() != dtype)
            {
                throw std::runtime_error("mul_scalar only supports " + dtype_name + " tensors");
            }

            if (lhs->device().device_type != pi::tensorlib::DeviceType::GPU ||
                rhs->device().device_type != pi::tensorlib::DeviceType::GPU ||
                output->device().device_type != pi::tensorlib::DeviceType::GPU)
            {
                throw std::runtime_error("mul_scalar requires GPU tensors");
            }

            if (rhs->shape().numel() != 1)
            {
                throw std::runtime_error("mul_scalar expects a scalar multiplier");
            }

            if (lhs->shape().ndims() != output->shape().ndims())
            {
                throw std::runtime_error("mul_scalar expects operands with matching rank");
            }
            for (size_t dim = 0; dim < lhs->shape().ndims(); ++dim)
            {
                if (lhs->shape()[dim] != output->shape()[dim])
                {
                    throw std::runtime_error("mul_scalar expects lhs/output with matching shape");
                }
            }

            const auto numel64 = output->shape().numel();
            if (numel64 == 0)
            {
                throw std::runtime_error("mul_scalar tensor must contain at least one element");
            }
            if (numel64 > std::numeric_limits<uint32_t>::max())
            {
                throw std::runtime_error("mul_scalar tensor size exceeds supported range");
            }

            if (!pi::tensorlib::shape_utils::IsRowMajorContiguous(lhs) ||
                !pi::tensorlib::shape_utils::IsRowMajorContiguous(rhs) ||
                !pi::tensorlib::shape_utils::IsRowMajorContiguous(output))
            {
                throw std::runtime_error("mul_scalar requires contiguous tensors");
            }

            const auto device_ordinal =
                ValidateSameDeviceOrdinal("mul_scalar", {lhs, rhs, output});

            void *out_ptr = output->dataptr();
            void *lhs_ptr = lhs->dataptr();
            void *rhs_ptr = rhs->dataptr();
            const auto n_elements = static_cast<uint32_t>(numel64);

            pi::tensorlib::KernelLaunchArguments arguments{
                .args = {out_ptr, lhs_ptr, rhs_ptr, n_elements, static_cast<void *>(nullptr)},
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
CreateDivElementwiseComputeKernelDescriptor(const pi::tensorlib::DataType dtype)
{
    const auto kernel = SelectKernelForAdd(kdiv_elementwise_bf16, kdiv_elementwise_fp16, kdiv_elementwise_fp32, dtype);
    const std::string kernel_name =
        std::string("div_elementwise_") + std::string(KernelSuffixForAdd(dtype));
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
                throw std::runtime_error("div_elementwise expects exactly two inputs");
            }
            if (outputs.size() != 1)
            {
                throw std::runtime_error("div_elementwise expects exactly one output");
            }

            const auto &lhs = inputs[0];
            const auto &rhs = inputs[1];
            const auto &output = outputs[0];

            if (lhs->dtype() != dtype || rhs->dtype() != dtype || output->dtype() != dtype)
            {
                throw std::runtime_error("div_elementwise only supports " + dtype_name + " tensors");
            }

            if (lhs->device().device_type != pi::tensorlib::DeviceType::GPU ||
                rhs->device().device_type != pi::tensorlib::DeviceType::GPU ||
                output->device().device_type != pi::tensorlib::DeviceType::GPU)
            {
                throw std::runtime_error("div_elementwise requires GPU tensors");
            }

            if (lhs->shape().ndims() != rhs->shape().ndims() || lhs->shape().ndims() != output->shape().ndims())
            {
                throw std::runtime_error("div_elementwise expects operands with matching rank");
            }

            for (size_t dim = 0; dim < lhs->shape().ndims(); ++dim)
            {
                if (lhs->shape()[dim] != rhs->shape()[dim] || lhs->shape()[dim] != output->shape()[dim])
                {
                    throw std::runtime_error("div_elementwise expects operands with matching shape");
                }
            }

            const auto numel64 = output->shape().numel();
            if (numel64 == 0)
            {
                throw std::runtime_error("div_elementwise tensor must contain at least one element");
            }
            if (numel64 > std::numeric_limits<uint32_t>::max())
            {
                throw std::runtime_error("div_elementwise tensor size exceeds supported range");
            }

            if (!pi::tensorlib::shape_utils::IsRowMajorContiguous(lhs) || !pi::tensorlib::shape_utils::IsRowMajorContiguous(rhs) || !pi::tensorlib::shape_utils::IsRowMajorContiguous(output))
            {
                throw std::runtime_error("div_elementwise requires contiguous tensors");
            }

            const auto device_ordinal =
                ValidateSameDeviceOrdinal("div_elementwise", {lhs, rhs, output});

            void *out_ptr = output->dataptr();
            void *lhs_ptr = lhs->dataptr();
            void *rhs_ptr = rhs->dataptr();
            const auto n_elements = static_cast<uint32_t>(numel64);

            pi::tensorlib::KernelLaunchArguments arguments{
                .args = {out_ptr, lhs_ptr, rhs_ptr, n_elements, static_cast<void *>(nullptr)},
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
CreateDivScalarComputeKernelDescriptor(const pi::tensorlib::DataType dtype)
{
    const auto kernel = SelectKernelForAdd(kdiv_scalar_bf16, kdiv_scalar_fp16, kdiv_scalar_fp32, dtype);
    const std::string kernel_name = std::string("div_scalar_") + std::string(KernelSuffixForAdd(dtype));
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
                throw std::runtime_error("div_scalar expects exactly two inputs");
            }
            if (outputs.size() != 1)
            {
                throw std::runtime_error("div_scalar expects exactly one output");
            }

            const auto &lhs = inputs[0];
            const auto &rhs = inputs[1];
            const auto &output = outputs[0];

            if (lhs->dtype() != dtype || rhs->dtype() != dtype || output->dtype() != dtype)
            {
                throw std::runtime_error("div_scalar only supports " + dtype_name + " tensors");
            }

            if (lhs->device().device_type != pi::tensorlib::DeviceType::GPU ||
                rhs->device().device_type != pi::tensorlib::DeviceType::GPU ||
                output->device().device_type != pi::tensorlib::DeviceType::GPU)
            {
                throw std::runtime_error("div_scalar requires GPU tensors");
            }

            if (rhs->shape().numel() != 1)
            {
                throw std::runtime_error("div_scalar expects a scalar denominator");
            }

            if (lhs->shape().ndims() != output->shape().ndims())
            {
                throw std::runtime_error("div_scalar expects operands with matching rank");
            }
            for (size_t dim = 0; dim < lhs->shape().ndims(); ++dim)
            {
                if (lhs->shape()[dim] != output->shape()[dim])
                {
                    throw std::runtime_error("div_scalar expects lhs/output with matching shape");
                }
            }

            const auto numel64 = output->shape().numel();
            if (numel64 == 0)
            {
                throw std::runtime_error("div_scalar tensor must contain at least one element");
            }
            if (numel64 > std::numeric_limits<uint32_t>::max())
            {
                throw std::runtime_error("div_scalar tensor size exceeds supported range");
            }

            if (!pi::tensorlib::shape_utils::IsRowMajorContiguous(lhs) ||
                !pi::tensorlib::shape_utils::IsRowMajorContiguous(rhs) ||
                !pi::tensorlib::shape_utils::IsRowMajorContiguous(output))
            {
                throw std::runtime_error("div_scalar requires contiguous tensors");
            }

            const auto device_ordinal =
                ValidateSameDeviceOrdinal("div_scalar", {lhs, rhs, output});

            void *out_ptr = output->dataptr();
            void *lhs_ptr = lhs->dataptr();
            void *rhs_ptr = rhs->dataptr();
            const auto n_elements = static_cast<uint32_t>(numel64);

            pi::tensorlib::KernelLaunchArguments arguments{
                .args = {out_ptr, lhs_ptr, rhs_ptr, n_elements, static_cast<void *>(nullptr)},
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
CreateDivAddComputeKernelDescriptor(const pi::tensorlib::DataType dtype)
{
    const auto kernel = SelectKernelForAdd(kdiv_add_bf16, kdiv_add_fp16, kdiv_add_fp32, dtype);
    const std::string kernel_name = std::string("div_add_") + std::string(KernelSuffixForAdd(dtype));
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
            if (inputs.size() != 3)
            {
                throw std::runtime_error("div_add expects exactly three inputs");
            }
            if (outputs.size() != 1)
            {
                throw std::runtime_error("div_add expects exactly one output");
            }

            const auto &lhs = inputs[0];
            const auto &rhs = inputs[1];
            const auto &denom = inputs[2];
            const auto &output = outputs[0];

            if (lhs->dtype() != dtype || rhs->dtype() != dtype || denom->dtype() != dtype || output->dtype() != dtype)
            {
                throw std::runtime_error("div_add only supports " + dtype_name + " tensors");
            }

            if (lhs->device().device_type != pi::tensorlib::DeviceType::GPU ||
                rhs->device().device_type != pi::tensorlib::DeviceType::GPU ||
                denom->device().device_type != pi::tensorlib::DeviceType::GPU ||
                output->device().device_type != pi::tensorlib::DeviceType::GPU)
            {
                throw std::runtime_error("div_add requires GPU tensors");
            }

            if (lhs->shape().ndims() != rhs->shape().ndims() || lhs->shape().ndims() != output->shape().ndims())
            {
                throw std::runtime_error("div_add expects operands with matching rank");
            }
            for (size_t dim = 0; dim < lhs->shape().ndims(); ++dim)
            {
                if (lhs->shape()[dim] != rhs->shape()[dim] || lhs->shape()[dim] != output->shape()[dim] ||
                    lhs->shape()[dim] != denom->shape()[dim])
                {
                    throw std::runtime_error("div_add expects operands with matching shape");
                }
            }

            const auto numel64 = output->shape().numel();
            if (numel64 == 0)
            {
                throw std::runtime_error("div_add tensor must contain at least one element");
            }
            if (numel64 > std::numeric_limits<uint32_t>::max())
            {
                throw std::runtime_error("div_add tensor size exceeds supported range");
            }

            if (!pi::tensorlib::shape_utils::IsRowMajorContiguous(lhs) ||
                !pi::tensorlib::shape_utils::IsRowMajorContiguous(rhs) ||
                !pi::tensorlib::shape_utils::IsRowMajorContiguous(denom) ||
                !pi::tensorlib::shape_utils::IsRowMajorContiguous(output))
            {
                throw std::runtime_error("div_add requires contiguous tensors");
            }

            const auto device_ordinal =
                ValidateSameDeviceOrdinal("div_add", {lhs, rhs, denom, output});

            void *out_ptr = output->dataptr();
            void *lhs_ptr = lhs->dataptr();
            void *rhs_ptr = rhs->dataptr();
            void *denom_ptr = denom->dataptr();
            const auto n_elements = static_cast<uint32_t>(numel64);

            pi::tensorlib::KernelLaunchArguments arguments{
                .args = {out_ptr, lhs_ptr, rhs_ptr, denom_ptr, n_elements, static_cast<void *>(nullptr)},
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
CreateDivScalarAddComputeKernelDescriptor(const pi::tensorlib::DataType dtype)
{
    const auto kernel = SelectKernelForAdd(kdiv_scalar_add_bf16, kdiv_scalar_add_fp16, kdiv_scalar_add_fp32, dtype);
    const std::string kernel_name = std::string("div_scalar_add_") + std::string(KernelSuffixForAdd(dtype));
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
            if (inputs.size() != 3)
            {
                throw std::runtime_error("div_scalar_add expects exactly three inputs");
            }
            if (outputs.size() != 1)
            {
                throw std::runtime_error("div_scalar_add expects exactly one output");
            }

            const auto &lhs = inputs[0];
            const auto &rhs = inputs[1];
            const auto &denom = inputs[2];
            const auto &output = outputs[0];

            if (lhs->dtype() != dtype || rhs->dtype() != dtype || denom->dtype() != dtype || output->dtype() != dtype)
            {
                throw std::runtime_error("div_scalar_add only supports " + dtype_name + " tensors");
            }

            if (lhs->device().device_type != pi::tensorlib::DeviceType::GPU ||
                rhs->device().device_type != pi::tensorlib::DeviceType::GPU ||
                denom->device().device_type != pi::tensorlib::DeviceType::GPU ||
                output->device().device_type != pi::tensorlib::DeviceType::GPU)
            {
                throw std::runtime_error("div_scalar_add requires GPU tensors");
            }

            if (denom->shape().numel() != 1)
            {
                throw std::runtime_error("div_scalar_add expects a scalar denominator");
            }

            if (lhs->shape().ndims() != rhs->shape().ndims() || lhs->shape().ndims() != output->shape().ndims())
            {
                throw std::runtime_error("div_scalar_add expects operands with matching rank");
            }
            for (size_t dim = 0; dim < lhs->shape().ndims(); ++dim)
            {
                if (lhs->shape()[dim] != rhs->shape()[dim] || lhs->shape()[dim] != output->shape()[dim])
                {
                    throw std::runtime_error("div_scalar_add expects operands with matching shape");
                }
            }

            const auto numel64 = output->shape().numel();
            if (numel64 == 0)
            {
                throw std::runtime_error("div_scalar_add tensor must contain at least one element");
            }
            if (numel64 > std::numeric_limits<uint32_t>::max())
            {
                throw std::runtime_error("div_scalar_add tensor size exceeds supported range");
            }

            if (!pi::tensorlib::shape_utils::IsRowMajorContiguous(lhs) ||
                !pi::tensorlib::shape_utils::IsRowMajorContiguous(rhs) ||
                !pi::tensorlib::shape_utils::IsRowMajorContiguous(denom) ||
                !pi::tensorlib::shape_utils::IsRowMajorContiguous(output))
            {
                throw std::runtime_error("div_scalar_add requires contiguous tensors");
            }

            const auto device_ordinal =
                ValidateSameDeviceOrdinal("div_scalar_add", {lhs, rhs, denom, output});

            void *out_ptr = output->dataptr();
            void *lhs_ptr = lhs->dataptr();
            void *rhs_ptr = rhs->dataptr();
            void *denom_ptr = denom->dataptr();
            const auto n_elements = static_cast<uint32_t>(numel64);

            pi::tensorlib::KernelLaunchArguments arguments{
                .args = {out_ptr, lhs_ptr, rhs_ptr, denom_ptr, n_elements, static_cast<void *>(nullptr)},
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
CreateDivScalarAddBroadcastComputeKernelDescriptor(const pi::tensorlib::DataType dtype)
{
    const auto kernel =
        SelectKernelForAdd(kdiv_scalar_add_broadcast_bf16, kdiv_scalar_add_broadcast_fp16,
                           kdiv_scalar_add_broadcast_fp32, dtype);
    const std::string kernel_name = std::string("div_scalar_add_broadcast_") + std::string(KernelSuffixForAdd(dtype));
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
            if (inputs.size() != 3)
            {
                throw std::runtime_error("div_scalar_add_broadcast expects exactly three inputs");
            }
            if (outputs.size() != 1)
            {
                throw std::runtime_error("div_scalar_add_broadcast expects exactly one output");
            }

            const auto &lhs = inputs[0];
            const auto &rhs = inputs[1];
            const auto &denom = inputs[2];
            const auto &output = outputs[0];

            if (lhs->dtype() != dtype || rhs->dtype() != dtype || denom->dtype() != dtype || output->dtype() != dtype)
            {
                throw std::runtime_error("div_scalar_add_broadcast only supports " + dtype_name + " tensors");
            }

            if (lhs->device().device_type != pi::tensorlib::DeviceType::GPU ||
                rhs->device().device_type != pi::tensorlib::DeviceType::GPU ||
                denom->device().device_type != pi::tensorlib::DeviceType::GPU ||
                output->device().device_type != pi::tensorlib::DeviceType::GPU)
            {
                throw std::runtime_error("div_scalar_add_broadcast requires GPU tensors");
            }

            if (denom->shape().numel() != 1)
            {
                throw std::runtime_error("div_scalar_add_broadcast expects a scalar denominator");
            }

            if (lhs->shape().ndims() != rhs->shape().ndims() || lhs->shape().ndims() != output->shape().ndims())
            {
                throw std::runtime_error("div_scalar_add_broadcast expects operands with matching rank");
            }

            const auto numel64 = output->shape().numel();
            if (numel64 == 0)
            {
                throw std::runtime_error("div_scalar_add_broadcast tensor must contain at least one element");
            }
            if (numel64 > std::numeric_limits<uint32_t>::max())
            {
                throw std::runtime_error("div_scalar_add_broadcast tensor size exceeds supported range");
            }

            if (!pi::tensorlib::shape_utils::IsRowMajorContiguous(lhs) ||
                !pi::tensorlib::shape_utils::IsRowMajorContiguous(denom) ||
                !pi::tensorlib::shape_utils::IsRowMajorContiguous(output))
            {
                throw std::runtime_error("div_scalar_add_broadcast requires contiguous lhs/denom/output");
            }

            int64_t broadcast_dim = -1;
            for (size_t dim = 0; dim < rhs->shape().ndims(); ++dim)
            {
                if (rhs->strides()[dim] == 0)
                {
                    if (broadcast_dim >= 0)
                    {
                        throw std::runtime_error("div_scalar_add_broadcast expects a single broadcast dimension");
                    }
                    broadcast_dim = static_cast<int64_t>(dim);
                }
            }
            if (broadcast_dim < 0)
            {
                throw std::runtime_error("div_scalar_add_broadcast requires a broadcasted rhs");
            }

            uint64_t inner64 = 1;
            for (size_t dim = static_cast<size_t>(broadcast_dim) + 1; dim < rhs->shape().ndims(); ++dim)
            {
                inner64 *= rhs->shape()[dim];
            }
            const uint64_t cols64 = rhs->shape()[static_cast<size_t>(broadcast_dim)];
            const uint64_t outer64 = rhs->shape().numel() / (cols64 * inner64);
            if (outer64 == 0 || inner64 == 0 || cols64 == 0)
            {
                throw std::runtime_error("div_scalar_add_broadcast invalid broadcast shape");
            }
            if (outer64 > std::numeric_limits<uint32_t>::max() || inner64 > std::numeric_limits<uint32_t>::max() ||
                cols64 > std::numeric_limits<uint32_t>::max())
            {
                throw std::runtime_error("div_scalar_add_broadcast exceeds supported tensor size");
            }

            const auto device_ordinal =
                ValidateSameDeviceOrdinal("div_scalar_add_broadcast", {lhs, rhs, denom, output});

            void *out_ptr = output->dataptr();
            void *lhs_ptr = lhs->dataptr();
            void *rhs_ptr = rhs->dataptr();
            void *denom_ptr = denom->dataptr();
            const auto n_elements = static_cast<uint32_t>(numel64);
            const auto inner = static_cast<uint32_t>(inner64);
            const auto cols = static_cast<uint32_t>(cols64);

            pi::tensorlib::KernelLaunchArguments arguments{
                .args = {out_ptr, lhs_ptr, rhs_ptr, denom_ptr, n_elements, inner, cols,
                         static_cast<void *>(nullptr)},
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

void DivAddFusePass::transform(pi::tensorlib::ExecutionPlan &execution_plan)
{
    const bool debug_div_add = std::getenv("FBAMTRAIN_DEBUG_DIV_ADD") != nullptr;
    auto same_shape = [](const std::shared_ptr<pi::tensorlib::RealTensor> &a,
                         const std::shared_ptr<pi::tensorlib::RealTensor> &b) -> bool
    {
        if (a->shape().ndims() != b->shape().ndims())
        {
            return false;
        }
        for (size_t dim = 0; dim < a->shape().ndims(); ++dim)
        {
            if (a->shape()[dim] != b->shape()[dim])
            {
                return false;
            }
        }
        return true;
    };
    bool changed = true;
    while (changed)
    {
        changed = false;
        for (size_t i = 0; i < execution_plan.entries.size(); ++i)
        {
            auto &div_op = execution_plan.entries[i];
            if (div_op.op_type != pi::tensorlib::OpType::DIV)
            {
                continue;
            }
            if (div_op.inputs.size() != 2 || div_op.outputs.size() != 1)
            {
                continue;
            }

            const auto &div_lhs = div_op.inputs[0];
            const auto &div_rhs = div_op.inputs[1];
            const auto &div_out = div_op.outputs[0];
            if (div_lhs == nullptr || div_rhs == nullptr || div_out == nullptr)
            {
                continue;
            }
            const bool denom_is_scalar = div_rhs->shape().numel() == 1;
            if (div_out->id() != div_lhs->id())
            {
                continue;
            }

            size_t plus_idx = execution_plan.entries.size();
            int div_input_idx = -1;
            for (size_t j = 0; j < execution_plan.entries.size(); ++j)
            {
                const auto &plus_candidate = execution_plan.entries[j];
                if (plus_candidate.op_type != pi::tensorlib::OpType::PLUS || plus_candidate.inputs.size() != 2 ||
                    plus_candidate.outputs.size() != 1)
                {
                    continue;
                }
                if (plus_candidate.inputs[0] != nullptr && plus_candidate.inputs[0]->id() == div_out->id())
                {
                    plus_idx = j;
                    div_input_idx = 0;
                    break;
                }
                if (plus_candidate.inputs[1] != nullptr && plus_candidate.inputs[1]->id() == div_out->id())
                {
                    plus_idx = j;
                    div_input_idx = 1;
                    break;
                }
            }
            if (plus_idx == execution_plan.entries.size())
            {
                continue;
            }

            auto &plus_op = execution_plan.entries[plus_idx];
            const auto &plus_out = plus_op.outputs[0];
            if (plus_out == nullptr)
            {
                continue;
            }
            const int lhs_idx = (div_input_idx == 0) ? 1 : 0;
            const auto &plus_lhs = plus_op.inputs[lhs_idx];
            if (plus_lhs == nullptr)
            {
                continue;
            }
            if (plus_out->id() != plus_lhs->id())
            {
                continue;
            }

            size_t div_use_count = 0;
            for (const auto &entry : execution_plan.entries)
            {
                if (entry.op_type == std::nullopt || entry.id == div_op.id)
                {
                    continue;
                }
                for (const auto &input : entry.inputs)
                {
                    if (input != nullptr && input->id() == div_out->id())
                    {
                        ++div_use_count;
                    }
                }
            }
            if (div_use_count != 1)
            {
                continue;
            }

            const auto dtype = div_out->dtype();
            if (!IsSupportedAddType(dtype) || plus_lhs->dtype() != dtype || div_rhs->dtype() != dtype ||
                div_lhs->dtype() != dtype || plus_out->dtype() != dtype)
            {
                continue;
            }

            const auto numel64 = plus_out->shape().numel();
            const bool valid_size = numel64 > 0 && numel64 <= std::numeric_limits<uint32_t>::max();
            const bool gpu_devices = plus_lhs->device().device_type == pi::tensorlib::DeviceType::GPU &&
                                     div_lhs->device().device_type == pi::tensorlib::DeviceType::GPU &&
                                     div_rhs->device().device_type == pi::tensorlib::DeviceType::GPU &&
                                     plus_out->device().device_type == pi::tensorlib::DeviceType::GPU;
            const bool same_device = plus_lhs->device().ordinal == div_lhs->device().ordinal &&
                                     plus_lhs->device().ordinal == div_rhs->device().ordinal &&
                                     plus_lhs->device().ordinal == plus_out->device().ordinal;
            if (!valid_size || !gpu_devices || !same_device)
            {
                continue;
            }

            const bool div_lhs_contig = pi::tensorlib::shape_utils::IsRowMajorContiguous(div_lhs);
            if (!pi::tensorlib::shape_utils::IsRowMajorContiguous(plus_lhs) ||
                !pi::tensorlib::shape_utils::IsRowMajorContiguous(div_rhs) ||
                !pi::tensorlib::shape_utils::IsRowMajorContiguous(plus_out))
            {
                continue;
            }

            if (!same_shape(plus_lhs, div_lhs) || !same_shape(plus_lhs, plus_out))
            {
                continue;
            }
            if (!denom_is_scalar && !same_shape(div_rhs, div_lhs))
            {
                continue;
            }

            bool use_broadcast = false;
            if (denom_is_scalar && !div_lhs_contig)
            {
                int zero_stride_dims = 0;
                for (size_t dim = 0; dim < div_lhs->shape().ndims(); ++dim)
                {
                    if (div_lhs->strides()[dim] == 0)
                    {
                        ++zero_stride_dims;
                    }
                }
                use_broadcast = zero_stride_dims == 1;
            }

            if (denom_is_scalar && !div_lhs_contig && !use_broadcast)
            {
                continue;
            }

            plus_op.op_type = std::nullopt;
            if (denom_is_scalar)
            {
                plus_op.kernel_descriptor =
                    use_broadcast ? CreateDivScalarAddBroadcastComputeKernelDescriptor(dtype)
                                  : CreateDivScalarAddComputeKernelDescriptor(dtype);
            }
            else
            {
                plus_op.kernel_descriptor = CreateDivAddComputeKernelDescriptor(dtype);
            }
            plus_op.inputs = {plus_lhs, div_lhs, div_rhs};
            plus_op.flop_estimate = static_cast<uint64_t>(numel64) * 2u;

            if (debug_div_add && plus_op.kernel_descriptor.has_value())
            {
                std::cout << "[DivAdd] Fusing div+add into " << plus_op.kernel_descriptor->kernel_name
                          << " for entry id " << plus_op.id << '\n'
                          << std::flush;
            }

            pi::tensorlib::passes::RemoveOperation(execution_plan, div_op);
            changed = true;
            break;
        }
    }
}


void TrailingBroadcastAddImplPass::transform(pi::tensorlib::ExecutionPlan &execution_plan)
{
    for (auto &entry : execution_plan.entries)
    {
        if (entry.op_type != pi::tensorlib::OpType::PLUS)
        {
            continue;
        }
        if (entry.inputs.size() != 2 || entry.outputs.size() != 1)
        {
            continue;
        }

        const auto &lhs = entry.inputs[0];
        const auto &rhs = entry.inputs[1];
        const auto &output = entry.outputs[0];

        const auto lhs_dtype = lhs->dtype();
        const auto rhs_dtype = rhs->dtype();
        const auto output_dtype = output->dtype();

        if (lhs_dtype == pi::tensorlib::DataType::FLOAT32 && rhs_dtype == pi::tensorlib::DataType::FLOAT32 &&
            output_dtype == pi::tensorlib::DataType::FLOAT16)
        {
            const bool same_rank =
                lhs->shape().ndims() == rhs->shape().ndims() && lhs->shape().ndims() == output->shape().ndims();
            if (same_rank)
            {
                bool same_shape = true;
                for (size_t dim = 0; dim < lhs->shape().ndims(); ++dim)
                {
                    if (lhs->shape()[dim] != rhs->shape()[dim] || lhs->shape()[dim] != output->shape()[dim])
                    {
                        same_shape = false;
                        break;
                    }
                }

                const auto numel64 = output->shape().numel();
                const bool valid_size = numel64 > 0 && numel64 <= std::numeric_limits<uint32_t>::max();
                const bool gpu_devices = lhs->device().device_type == pi::tensorlib::DeviceType::GPU &&
                                         rhs->device().device_type == pi::tensorlib::DeviceType::GPU &&
                                         output->device().device_type == pi::tensorlib::DeviceType::GPU;
                const bool same_device = lhs->device().ordinal == rhs->device().ordinal &&
                                         lhs->device().ordinal == output->device().ordinal;
                if (same_shape && valid_size && gpu_devices && same_device && pi::tensorlib::shape_utils::IsRowMajorContiguous(lhs) &&
                    pi::tensorlib::shape_utils::IsRowMajorContiguous(rhs) && pi::tensorlib::shape_utils::IsRowMajorContiguous(output))
                {
                    entry.op_type = std::nullopt;
                    entry.kernel_descriptor = CreateAddElementwiseFp32OutFp16ComputeKernelDescriptor();
                    entry.flop_estimate = static_cast<uint64_t>(numel64);
                    continue;
                }
            }
        }

        const auto dtype = lhs_dtype;
        if (!IsSupportedAddType(dtype) || rhs_dtype != dtype || output_dtype != dtype)
        {
            continue;
        }

        const bool same_rank =
            lhs->shape().ndims() == rhs->shape().ndims() && lhs->shape().ndims() == output->shape().ndims();
        if (same_rank)
        {
            bool same_shape = true;
            for (size_t dim = 0; dim < lhs->shape().ndims(); ++dim)
            {
                if (lhs->shape()[dim] != rhs->shape()[dim] || lhs->shape()[dim] != output->shape()[dim])
                {
                    same_shape = false;
                    break;
                }
            }

            const auto numel64 = output->shape().numel();
            const bool valid_size = numel64 > 0 && numel64 <= std::numeric_limits<uint32_t>::max();
            const bool gpu_devices = lhs->device().device_type == pi::tensorlib::DeviceType::GPU &&
                                     rhs->device().device_type == pi::tensorlib::DeviceType::GPU &&
                                     output->device().device_type == pi::tensorlib::DeviceType::GPU;
            const bool same_device =
                lhs->device().ordinal == rhs->device().ordinal && lhs->device().ordinal == output->device().ordinal;
            if (same_shape && valid_size && gpu_devices && same_device && pi::tensorlib::shape_utils::IsRowMajorContiguous(lhs) &&
                pi::tensorlib::shape_utils::IsRowMajorContiguous(rhs) && pi::tensorlib::shape_utils::IsRowMajorContiguous(output))
            {
                entry.op_type = std::nullopt;
                entry.kernel_descriptor = CreateAddElementwiseComputeKernelDescriptor(dtype);
                entry.flop_estimate = static_cast<uint64_t>(numel64);
                continue;
            }
        }

        if (lhs->shape().ndims() != 2 || output->shape().ndims() != 2)
        {
            continue;
        }
        const bool bias_rank1 = rhs->shape().ndims() == 1;
        const bool bias_rank2 = rhs->shape().ndims() == 2 && rhs->shape()[0] == 1;
        if (!bias_rank1 && !bias_rank2)
        {
            continue;
        }

        if (lhs->shape()[0] != output->shape()[0] || lhs->shape()[1] != output->shape()[1])
        {
            continue;
        }

        const auto cols = output->shape()[1];
        if ((bias_rank1 && rhs->shape()[0] != cols) || (bias_rank2 && rhs->shape()[1] != cols))
        {
            continue;
        }

        const bool bias_contiguous = bias_rank1 ? rhs->strides()[0] == 1
                                                : (rhs->strides()[1] == 1 && rhs->strides()[0] == cols);
        if (lhs->strides()[0] != output->strides()[0] || lhs->strides()[1] != output->strides()[1] ||
            output->strides()[1] != 1 || !bias_contiguous)
        {
            continue;
        }

        if (output->shape()[0] > std::numeric_limits<uint32_t>::max() ||
            output->shape()[1] > std::numeric_limits<uint32_t>::max() ||
            output->strides()[0] > std::numeric_limits<uint32_t>::max() ||
            rhs->strides()[0] > std::numeric_limits<uint32_t>::max())
        {
            continue;
        }

        entry.op_type = std::nullopt;
        entry.kernel_descriptor = CreateAddTrailingBroadcastComputeKernelDescriptor(dtype);
        entry.flop_estimate = output->shape()[0] * output->shape()[1];
        continue;
    }
}

void TrailingBroadcastMulImplPass::transform(pi::tensorlib::ExecutionPlan &execution_plan)
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

    auto ensure_create = [&](size_t insert_idx, const std::shared_ptr<pi::tensorlib::RealTensor> &output,
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
        if (execution_plan.entries[i].op_type != pi::tensorlib::OpType::MUL ||
            execution_plan.entries[i].inputs.size() != 2 || execution_plan.entries[i].outputs.size() != 1)
        {
            ++i;
            continue;
        }

        const auto lhs = execution_plan.entries[i].inputs[0];
        const auto rhs = execution_plan.entries[i].inputs[1];
        const auto output = execution_plan.entries[i].outputs[0];

        const auto dtype = lhs->dtype();
        if (!IsSupportedAddType(dtype) || rhs->dtype() != dtype || output->dtype() != dtype)
        {
            ++i;
            continue;
        }

        const bool same_rank =
            lhs->shape().ndims() == rhs->shape().ndims() && lhs->shape().ndims() == output->shape().ndims();
        if (same_rank)
        {
            bool same_shape = true;
            for (size_t dim = 0; dim < lhs->shape().ndims(); ++dim)
            {
                if (lhs->shape()[dim] != rhs->shape()[dim] || lhs->shape()[dim] != output->shape()[dim])
                {
                    same_shape = false;
                    break;
                }
            }

            const auto numel64 = output->shape().numel();
            const bool valid_size = numel64 > 0 && numel64 <= std::numeric_limits<uint32_t>::max();
            const bool gpu_devices = lhs->device().device_type == pi::tensorlib::DeviceType::GPU &&
                                     rhs->device().device_type == pi::tensorlib::DeviceType::GPU &&
                                     output->device().device_type == pi::tensorlib::DeviceType::GPU;
            const bool same_device =
                lhs->device().ordinal == rhs->device().ordinal && lhs->device().ordinal == output->device().ordinal;
            if (same_shape && valid_size && gpu_devices && same_device && pi::tensorlib::shape_utils::IsRowMajorContiguous(lhs) &&
                pi::tensorlib::shape_utils::IsRowMajorContiguous(rhs) && pi::tensorlib::shape_utils::IsRowMajorContiguous(output))
            {
                size_t entry_idx = i;
                if (output && lhs->id() != output->id() && rhs->id() != output->id())
                {
                    if (ensure_create(i, output, execution_plan.entries[i].is_useful))
                    {
                        ++i;
                        entry_idx = i;
                    }
                }
                auto &entry = execution_plan.entries[entry_idx];
                entry.op_type = std::nullopt;
                entry.kernel_descriptor = CreateMulElementwiseComputeKernelDescriptor(dtype);
                entry.flop_estimate = static_cast<uint64_t>(numel64);
                ++i;
                continue;
            }
        }

        if (rhs->shape().numel() == 1)
        {
            const auto numel64 = output->shape().numel();
            const bool valid_size = numel64 > 0 && numel64 <= std::numeric_limits<uint32_t>::max();
            const bool gpu_devices = lhs->device().device_type == pi::tensorlib::DeviceType::GPU &&
                                     rhs->device().device_type == pi::tensorlib::DeviceType::GPU &&
                                     output->device().device_type == pi::tensorlib::DeviceType::GPU;
            const bool same_device =
                lhs->device().ordinal == rhs->device().ordinal && lhs->device().ordinal == output->device().ordinal;
            bool same_shape = lhs->shape().ndims() == output->shape().ndims();
            if (same_shape)
            {
                for (size_t dim = 0; dim < lhs->shape().ndims(); ++dim)
                {
                    if (lhs->shape()[dim] != output->shape()[dim])
                    {
                        same_shape = false;
                        break;
                    }
                }
            }
            if (same_shape && valid_size && gpu_devices && same_device &&
                pi::tensorlib::shape_utils::IsRowMajorContiguous(lhs) &&
                pi::tensorlib::shape_utils::IsRowMajorContiguous(rhs) &&
                pi::tensorlib::shape_utils::IsRowMajorContiguous(output))
            {
                size_t entry_idx = i;
                if (output && lhs->id() != output->id() && rhs->id() != output->id())
                {
                    if (ensure_create(i, output, execution_plan.entries[i].is_useful))
                    {
                        ++i;
                        entry_idx = i;
                    }
                }
                auto &entry = execution_plan.entries[entry_idx];
                entry.op_type = std::nullopt;
                entry.kernel_descriptor = CreateMulScalarComputeKernelDescriptor(dtype);
                entry.flop_estimate = static_cast<uint64_t>(numel64);
                ++i;
                continue;
            }
        }

        if (lhs->shape().ndims() != 2 || output->shape().ndims() != 2)
        {
            ++i;
            continue;
        }
        const bool rhs_rank1 = rhs->shape().ndims() == 1;
        const bool rhs_rank2 = rhs->shape().ndims() == 2 && rhs->shape()[0] == 1;
        if (!rhs_rank1 && !rhs_rank2)
        {
            ++i;
            continue;
        }
        if (lhs->shape()[0] != output->shape()[0] || lhs->shape()[1] != output->shape()[1])
        {
            ++i;
            continue;
        }
        const auto cols = output->shape()[1];
        if ((rhs_rank1 && rhs->shape()[0] != cols) || (rhs_rank2 && rhs->shape()[1] != cols))
        {
            ++i;
            continue;
        }
        const bool rhs_contiguous = rhs_rank1 ? rhs->strides()[0] == 1
                                              : (rhs->strides()[1] == 1 && rhs->strides()[0] == cols);
        if (lhs->strides()[0] != output->strides()[0] || lhs->strides()[1] != output->strides()[1] ||
            output->strides()[1] != 1 || !rhs_contiguous)
        {
            ++i;
            continue;
        }
        if (output->shape()[0] > std::numeric_limits<uint32_t>::max() ||
            output->shape()[1] > std::numeric_limits<uint32_t>::max() ||
            output->strides()[0] > std::numeric_limits<uint32_t>::max() ||
            rhs->strides()[0] > std::numeric_limits<uint32_t>::max())
        {
            ++i;
            continue;
        }

        size_t entry_idx = i;
        if (output && lhs->id() != output->id() && rhs->id() != output->id())
        {
            if (ensure_create(i, output, execution_plan.entries[i].is_useful))
            {
                ++i;
                entry_idx = i;
            }
        }
        auto &entry = execution_plan.entries[entry_idx];
        entry.op_type = std::nullopt;
        const auto kernel =
            SelectKernelForAdd(kmul_trailing_broadcast_bf16, kmul_trailing_broadcast_fp16, kmul_trailing_broadcast_fp32,
                               dtype);
        const std::string kernel_name = kernel.kernel_name;
        entry.kernel_descriptor =
            CreateTrailingBroadcastDescriptor(kernel, kernel_name, "mul_trailing_broadcast", "scale", dtype);
        entry.flop_estimate = output->shape()[0] * output->shape()[1];
        ++i;
        continue;
    }
        ++i;
}

void SqrtImplPass::transform(pi::tensorlib::ExecutionPlan &execution_plan)
{
    for (auto &entry : execution_plan.entries)
    {
        if (entry.op_type != pi::tensorlib::OpType::SQRT)
        {
            continue;
        }
        if (entry.inputs.size() != 1 || entry.outputs.size() != 1)
        {
            continue;
        }

        const auto &input = entry.inputs[0];
        const auto &output = entry.outputs[0];

        const auto dtype = input->dtype();
        if (!IsSupportedAddType(dtype) || output->dtype() != dtype)
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
            if (input->shape()[dim] != output->shape()[dim])
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
        const bool valid_size = numel64 > 0 && numel64 <= std::numeric_limits<uint32_t>::max();
        const bool gpu_devices = input->device().device_type == pi::tensorlib::DeviceType::GPU &&
                                 output->device().device_type == pi::tensorlib::DeviceType::GPU;
        const bool same_device = input->device().ordinal == output->device().ordinal;
        if (!valid_size || !gpu_devices || !same_device)
        {
            continue;
        }
        if (!pi::tensorlib::shape_utils::IsRowMajorContiguous(input) ||
            !pi::tensorlib::shape_utils::IsRowMajorContiguous(output))
        {
            continue;
        }

        entry.op_type = std::nullopt;
        entry.kernel_descriptor = CreateSqrtElementwiseComputeKernelDescriptor(dtype);
        entry.flop_estimate = static_cast<uint64_t>(numel64);
    }
}

void DivImplPass::transform(pi::tensorlib::ExecutionPlan &execution_plan)
{
    for (auto &entry : execution_plan.entries)
    {
        if (entry.op_type != pi::tensorlib::OpType::DIV)
        {
            continue;
        }
        if (entry.inputs.size() != 2 || entry.outputs.size() != 1)
        {
            continue;
        }

        const auto &lhs = entry.inputs[0];
        const auto &rhs = entry.inputs[1];
        const auto &output = entry.outputs[0];

        const auto dtype = lhs->dtype();
        if (!IsSupportedAddType(dtype) || rhs->dtype() != dtype || output->dtype() != dtype)
        {
            continue;
        }

        if (rhs->shape().numel() == 1)
        {
            const auto numel64 = output->shape().numel();
            const bool valid_size = numel64 > 0 && numel64 <= std::numeric_limits<uint32_t>::max();
            const bool gpu_devices = lhs->device().device_type == pi::tensorlib::DeviceType::GPU &&
                                     rhs->device().device_type == pi::tensorlib::DeviceType::GPU &&
                                     output->device().device_type == pi::tensorlib::DeviceType::GPU;
            const bool same_device =
                lhs->device().ordinal == rhs->device().ordinal && lhs->device().ordinal == output->device().ordinal;
            bool same_shape = lhs->shape().ndims() == output->shape().ndims();
            if (same_shape)
            {
                for (size_t dim = 0; dim < lhs->shape().ndims(); ++dim)
                {
                    if (lhs->shape()[dim] != output->shape()[dim])
                    {
                        same_shape = false;
                        break;
                    }
                }
            }
            if (same_shape && valid_size && gpu_devices && same_device &&
                pi::tensorlib::shape_utils::IsRowMajorContiguous(lhs) &&
                pi::tensorlib::shape_utils::IsRowMajorContiguous(rhs) &&
                pi::tensorlib::shape_utils::IsRowMajorContiguous(output))
            {
                entry.op_type = std::nullopt;
                entry.kernel_descriptor = CreateDivScalarComputeKernelDescriptor(dtype);
                entry.flop_estimate = static_cast<uint64_t>(numel64);
                continue;
            }
        }

        const bool same_rank =
            lhs->shape().ndims() == rhs->shape().ndims() && lhs->shape().ndims() == output->shape().ndims();
        if (!same_rank)
        {
            continue;
        }
        bool same_shape = true;
        for (size_t dim = 0; dim < lhs->shape().ndims(); ++dim)
        {
            if (lhs->shape()[dim] != rhs->shape()[dim] || lhs->shape()[dim] != output->shape()[dim])
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
        const bool valid_size = numel64 > 0 && numel64 <= std::numeric_limits<uint32_t>::max();
        const bool gpu_devices = lhs->device().device_type == pi::tensorlib::DeviceType::GPU &&
                                 rhs->device().device_type == pi::tensorlib::DeviceType::GPU &&
                                 output->device().device_type == pi::tensorlib::DeviceType::GPU;
        const bool same_device =
            lhs->device().ordinal == rhs->device().ordinal && lhs->device().ordinal == output->device().ordinal;
        if (!valid_size || !gpu_devices || !same_device)
        {
            continue;
        }
        if (!pi::tensorlib::shape_utils::IsRowMajorContiguous(lhs) ||
            !pi::tensorlib::shape_utils::IsRowMajorContiguous(rhs) ||
            !pi::tensorlib::shape_utils::IsRowMajorContiguous(output))
        {
            continue;
        }

        entry.op_type = std::nullopt;
        entry.kernel_descriptor = CreateDivElementwiseComputeKernelDescriptor(dtype);
        entry.flop_estimate = static_cast<uint64_t>(numel64);
    }
}
