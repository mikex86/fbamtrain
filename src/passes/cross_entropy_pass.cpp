#include "passes.h"

#include "passes/pass_utils.h"
#include "shape_utils.h"

#include <kernels/kernel_binaries.h>
#include <launch_utils.h>

#include <../../include/logger.h>

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <sstream>

namespace
{
    pi::tensorlib::ComputeKernelDescriptor
    CreateCrossEntropyKernelDescriptor(const pi::tensorlib::DataType dtype, const bool output_fp32)
    {
        const auto kernel = output_fp32
                                ? SelectKernelForHalf(kcross_entropy_on_targets_add_fp32_bf16,
                                                      kcross_entropy_on_targets_add_fp32_fp16, dtype)
                                : SelectKernelForHalf(kcross_entropy_on_targets_bf16, kcross_entropy_on_targets_fp16,
                                                      dtype);
        const std::string kernel_name =
            std::string("cross_entropy_on_targets_") +
            (output_fp32 ? "add_fp32_" : "") + std::string(KernelSuffixForHalf(dtype));

        const std::string dtype_name = pi::tensorlib::GetDataTypeName(dtype);

        return pi::tensorlib::ComputeKernelDescriptor{
            .kernel_name = kernel_name,
            .function_name = std::string(kernel.function_name),
            .expected_arg_count = kernel.arg_count,
            .argument_provider = [kernel, kernel_name, dtype, dtype_name](
                                     const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                                     const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
                -> pi::tensorlib::KernelLaunchArguments
            {
                if (inputs.size() != 2 || outputs.size() != 1)
                {
                    throw std::runtime_error(kernel_name +
                                             " expects exactly two inputs (logits, targets) and one output (loss)");
                }

                const auto &logits = inputs[0];
                const auto &targets = inputs[1];
                const auto &loss = outputs[0];

                if (!pi::tensorlib::shape_utils::IsRowMajorContiguous(logits->shape(), logits->strides()) ||
                    !pi::tensorlib::shape_utils::IsRowMajorContiguous(targets->shape(), targets->strides()) ||
                    !pi::tensorlib::shape_utils::IsRowMajorContiguous(loss->shape(), loss->strides()))
                {
                    throw std::runtime_error(kernel_name + " requires contiguous logits, targets, and output tensors");
                }

                if (logits->dtype() != dtype)
                {
                    throw std::runtime_error(kernel_name + " requires logits of type " + dtype_name);
                }
                if (loss->dtype() != pi::tensorlib::DataType::FLOAT32 && loss->dtype() != dtype)
                {
                    throw std::runtime_error(kernel_name + " requires output of type " + dtype_name + " or FLOAT32");
                }
                if (targets->dtype() != pi::tensorlib::DataType::UINT32)
                {
                    throw std::runtime_error(kernel_name + " requires UINT32 targets");
                }

                const auto &logits_shape = logits->shape();
                if (logits_shape.ndims() < 1)
                {
                    throw std::runtime_error(kernel_name + " expects logits with at least one dimension");
                }

                const auto vocab = logits_shape[logits_shape.ndims() - 1];
                if (vocab == 0)
                {
                    throw std::runtime_error(kernel_name + " requires non-empty class dimension");
                }

                const auto logits_elements = logits_shape.numel();
                if (logits_elements % vocab != 0)
                {
                    throw std::runtime_error(kernel_name + " expects logits.numel to be divisible by vocab size");
                }
                const auto rows64 = logits_elements / vocab;
                if (targets->shape().numel() != rows64)
                {
                    throw std::runtime_error(kernel_name +
                                             " expects targets to have the same logical elements as logits without "
                                             "the class dimension");
                }
                if (rows64 == 0)
                {
                    throw std::runtime_error(kernel_name + " requires at least one sample");
                }
                if (rows64 > std::numeric_limits<uint32_t>::max() ||
                    vocab > std::numeric_limits<uint32_t>::max())
                {
                    throw std::runtime_error(kernel_name + " supports up to 2^32-1 rows and classes");
                }

                if (loss->shape().numel() != rows64)
                {
                    throw std::runtime_error(kernel_name + " expects loss output sized to number of rows");
                }

                const auto device_ordinal = ValidateSameDeviceOrdinal(
                    kernel_name, {logits, targets, loss}, pi::tensorlib::DeviceType::GPU);

                void *loss_ptr = loss->dataptr();
                void *logits_ptr = logits->dataptr();
                void *targets_ptr = targets->dataptr();
                const auto rows = static_cast<uint32_t>(rows64);
                const auto cols = static_cast<uint32_t>(vocab);

                const uint32_t grid_x = rows;
                pi::tensorlib::KernelLaunchArguments arguments{
                    .args = {loss_ptr, logits_ptr, targets_ptr, rows, cols, static_cast<void *>(nullptr)},
                    .grid_dim_x = grid_x,
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

    pi::tensorlib::ComputeKernelDescriptor
    CreateCrossEntropyBwdKernelDescriptor(const pi::tensorlib::DataType dtype, const uint64_t rows,
                                          const uint64_t vocab)
    {
        const auto kernel = SelectKernelForHalf(kcross_entropy_on_targets_bwd_bf16, kcross_entropy_on_targets_bwd_fp16,
                                                dtype);
        const std::string kernel_name =
            std::string("cross_entropy_on_targets_bwd_") + std::string(KernelSuffixForHalf(dtype));
        const std::string dtype_name = pi::tensorlib::GetDataTypeName(dtype);

        return pi::tensorlib::ComputeKernelDescriptor{
            .kernel_name = kernel_name,
            .function_name = std::string(kernel.function_name),
            .expected_arg_count = kernel.arg_count,
            .argument_provider =
                [kernel, kernel_name, dtype, dtype_name, rows, vocab](
                    const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                    const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
                    -> pi::tensorlib::KernelLaunchArguments
            {
                if (inputs.size() != 3 || outputs.size() != 1)
                {
                    throw std::runtime_error(kernel_name + " expects logits, targets, upstream and one output");
                }
                const auto &logits_rt = inputs[0];
                const auto &targets_rt = inputs[1];
                const auto &upstream_rt = inputs[2];
                const auto &grad_rt = outputs[0];

                if (!pi::tensorlib::shape_utils::IsRowMajorContiguous(logits_rt->shape(), logits_rt->strides()) ||
                    !pi::tensorlib::shape_utils::IsRowMajorContiguous(targets_rt->shape(), targets_rt->strides()) ||
                    !pi::tensorlib::shape_utils::IsRowMajorContiguous(upstream_rt->shape(), upstream_rt->strides()) ||
                    !pi::tensorlib::shape_utils::IsRowMajorContiguous(grad_rt->shape(), grad_rt->strides()))
                {
                    throw std::runtime_error(kernel_name + " requires contiguous inputs/outputs");
                }

                if (logits_rt->dtype() != dtype || grad_rt->dtype() != dtype)
                {
                    throw std::runtime_error(kernel_name + " requires logits/grad of type " + dtype_name);
                }
                if (targets_rt->dtype() != pi::tensorlib::DataType::UINT32)
                {
                    throw std::runtime_error(kernel_name + " requires UINT32 targets");
                }
                if (upstream_rt->dtype() != pi::tensorlib::DataType::FLOAT32)
                {
                    throw std::runtime_error(kernel_name + " requires FLOAT32 upstream");
                }

                const auto device_ordinal =
                    ValidateSameDeviceOrdinal(kernel_name, {logits_rt, targets_rt, upstream_rt, grad_rt},
                                              pi::tensorlib::DeviceType::GPU);

                void *grad_ptr = grad_rt->dataptr();
                void *logits_ptr = logits_rt->dataptr();
                void *targets_ptr = targets_rt->dataptr();
                void *upstream_ptr = upstream_rt->dataptr();
                const uint32_t rows_u32 = static_cast<uint32_t>(rows);
                const uint32_t cols_u32 = static_cast<uint32_t>(vocab);

                pi::tensorlib::KernelLaunchArguments arguments{
                    .args = {grad_ptr, logits_ptr, targets_ptr, upstream_ptr, rows_u32, cols_u32,
                             static_cast<void *>(nullptr)},
                    .grid_dim_x = rows_u32,
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

void CrossEntropyOnTargetsImplPass::transform(pi::tensorlib::ExecutionPlan &execution_plan)
{
    const bool debug_ce = std::getenv("DEBUG_CE_PASS") != nullptr;
    auto describe_shape = [](const pi::tensorlib::Shape &shape) {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < shape.ndims(); ++i)
        {
            oss << shape[i];
            if (i + 1 < shape.ndims())
            {
                oss << ", ";
            }
        }
        oss << "]";
        return oss.str();
    };

    for (auto &entry : execution_plan.entries)
    {
        if (entry.op_type != pi::tensorlib::OpType::CROSS_ENTROPY_ON_TARGETS &&
            entry.op_type != pi::tensorlib::OpType::CROSS_ENTROPY_ON_TARGETS_BWD)
        {
            continue;
        }

        const auto stream_desc = entry.gpu_stream_desc;

        const bool is_bwd = entry.op_type == pi::tensorlib::OpType::CROSS_ENTROPY_ON_TARGETS_BWD;
        if ((!is_bwd && (entry.inputs.size() != 2 || entry.outputs.size() != 1)) ||
            (is_bwd && (entry.inputs.size() != 3 || entry.outputs.size() != 1)))
        {
            throw std::runtime_error(is_bwd ? "CROSS_ENTROPY_ON_TARGETS_BWD expects three inputs and one output"
                                            : "CROSS_ENTROPY_ON_TARGETS expects two inputs and one output");
        }

        const auto reduction_it = entry.attributes.find("reduction");
        if (reduction_it == entry.attributes.end())
        {
            throw std::runtime_error("CROSS_ENTROPY_ON_TARGETS missing reduction attribute");
        }
        const auto reduction = std::any_cast<std::string>(reduction_it->second);

        const auto &logits = entry.inputs[0];
        const auto dtype = logits->dtype();
        if (!IsSupportedHalfType(dtype))
        {
            throw std::runtime_error("CROSS_ENTROPY_ON_TARGETS supports BF16/FP16 logits only");
        }
        const auto vocab = logits->shape()[logits->shape().ndims() - 1];
        const auto rows = logits->shape().numel() / vocab;
        const auto &output = entry.outputs[0];
        if (debug_ce)
        {
            LOG(INFO) << "CE pass entry id=" << entry.id << " reduction=" << reduction << " rows=" << rows
                      << " cols=" << vocab << " logits shape=" << describe_shape(logits->shape())
                      << " targets shape=" << describe_shape(entry.inputs[1]->shape());
        }

        if (!pi::tensorlib::shape_utils::IsRowMajorContiguous(logits->shape(), logits->strides()) ||
            !pi::tensorlib::shape_utils::IsRowMajorContiguous(entry.inputs[1]->shape(), entry.inputs[1]->strides()) ||
            !pi::tensorlib::shape_utils::IsRowMajorContiguous(output->shape(), output->strides()))
        {
            throw std::runtime_error("CROSS_ENTROPY_ON_TARGETS requires contiguous tensors");
        }

        if (is_bwd)
        {
            const auto reduce_over_rows_it = entry.attributes.find("reduce_over_rows");
            if (reduce_over_rows_it == entry.attributes.end())
            {
                throw std::runtime_error("CROSS_ENTROPY_ON_TARGETS_BWD missing reduce_over_rows attribute");
            }
            const bool reduce_over_rows = std::any_cast<bool>(reduce_over_rows_it->second);
            const auto &upstream = entry.inputs[2];
            if (upstream->dtype() != pi::tensorlib::DataType::FLOAT32)
            {
                throw std::runtime_error("CROSS_ENTROPY_ON_TARGETS_BWD requires FLOAT32 upstream gradient");
            }
            if (reduce_over_rows)
            {
                const auto upstream_elements = upstream->shape().numel();
                if (upstream_elements != 1 && upstream_elements != rows)
                {
                    throw std::runtime_error("CROSS_ENTROPY_ON_TARGETS_BWD reduce_over_rows expects scalar upstream");
                }
            }
            else if (upstream->shape().numel() != rows)
            {
                throw std::runtime_error("CROSS_ENTROPY_ON_TARGETS_BWD expects upstream with one element per row");
            }

            entry.kernel_descriptor = CreateCrossEntropyBwdKernelDescriptor(dtype, rows, vocab);
            entry.op_type.reset();
            entry.attributes.clear();
            entry.gpu_stream_desc = stream_desc;
            entry.flop_estimate = rows * vocab * 2;
        }
        else
        {
            const bool output_fp32 = (output->dtype() == pi::tensorlib::DataType::FLOAT32);
            if (!output_fp32 && output->dtype() != dtype)
            {
                throw std::runtime_error("CROSS_ENTROPY_ON_TARGETS output dtype must be logits dtype or FLOAT32");
            }
            if (output->shape().numel() != rows)
            {
                throw std::runtime_error("CROSS_ENTROPY_ON_TARGETS output must have one element per row");
            }
            const auto kernel = CreateCrossEntropyKernelDescriptor(dtype, output_fp32);
            entry.kernel_descriptor = kernel;
            entry.op_type.reset();
            entry.attributes.clear();
            entry.gpu_stream_desc = stream_desc;
            entry.inputs = {entry.inputs[0], entry.inputs[1]};
            entry.outputs = {entry.outputs[0]};
            entry.flop_estimate = rows * (2 * vocab + 6);
        }
    }
}
