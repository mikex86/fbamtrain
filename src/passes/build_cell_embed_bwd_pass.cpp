#include "passes.h"

#include "passes/pass_utils.h"
#include <kernels/kernel_binaries.h>

#include <any>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

namespace
{
    pi::tensorlib::ComputeKernelDescriptor CreateBuildCellEmbedsBwdCpComputeKernelDescriptor(
        const pi::tensorlib::DataType dtype, const bool output_fp32)
    {
        const auto kernel = output_fp32
                                ? SelectKernelForHalf(kbuild_cell_embeds_bwd_cp_bf16_out_fp32,
                                                      kbuild_cell_embeds_bwd_cp_fp16_out_fp32, dtype)
                                : SelectKernelForHalf(kbuild_cell_embeds_bwd_cp_bf16, kbuild_cell_embeds_bwd_cp_fp16,
                                                      dtype);
        const std::string kernel_name =
            std::string("build_cell_embeds_bwd_cp_") + std::string(KernelSuffixForHalf(dtype)) +
            (output_fp32 ? "_out_fp32" : "");

        return pi::tensorlib::ComputeKernelDescriptor{
            .kernel_name = kernel_name,
            .function_name = kernel.function_name,
            .expected_arg_count = kernel.arg_count,
            .argument_provider = [kernel, dtype, output_fp32](
                                     const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                                                 const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
                -> pi::tensorlib::KernelLaunchArguments
            {
                if (inputs.size() != 2)
                {
                    throw std::runtime_error("build_cell_embeds_bwd_cp expects 2 inputs");
                }
                if (outputs.size() != 1)
                {
                    throw std::runtime_error("build_cell_embeds_bwd_cp expects 1 output");
                }

                const auto &grad_out = inputs[0];
                const auto &cell_states = inputs[1];
                const auto &grad_cp = outputs[0];

                if (grad_out->dtype() != dtype ||
                    grad_cp->dtype() != (output_fp32 ? pi::tensorlib::DataType::FLOAT32 : dtype))
                {
                    throw std::runtime_error("build_cell_embeds_bwd_cp expects matching BFLOAT16/FLOAT16 tensors");
                }
                if (cell_states->dtype() != pi::tensorlib::DataType::UINT32)
                {
                    throw std::runtime_error("build_cell_embeds_bwd_cp expects UINT32 cell state tensor");
                }

                const auto device_ordinal =
                    ValidateSameDeviceOrdinal("build_cell_embeds_bwd_cp", {grad_out, cell_states, grad_cp});

                if (grad_out->shape().ndims() != 2)
                {
                    throw std::runtime_error("build_cell_embeds_bwd_cp expects grad_out shaped (N, D)");
                }

                const auto N = static_cast<uint32_t>(grad_out->shape()[0]);
                const auto D = static_cast<uint32_t>(grad_out->shape()[1]);

                if (cell_states->shape().dims() != std::vector<uint64_t>{N, 3})
                {
                    throw std::runtime_error("build_cell_embeds_bwd_cp expects cell_states shaped (N, 3)");
                }
                if (grad_cp->shape().ndims() != 2 || grad_cp->shape()[1] != D)
                {
                    throw std::runtime_error("build_cell_embeds_bwd_cp expects grad_cp shaped (V, D)");
                }

                const auto stride_out_n = static_cast<uint32_t>(grad_out->strides()[0]);
                const auto stride_out_d = static_cast<uint32_t>(grad_out->strides()[1]);
                const auto stride_cp_v = static_cast<uint32_t>(grad_cp->strides()[0]);
                const auto stride_cp_d = static_cast<uint32_t>(grad_cp->strides()[1]);

                if (stride_out_n != D || stride_out_d != 1 || stride_cp_v != D || stride_cp_d != 1)
                {
                    throw std::runtime_error("build_cell_embeds_bwd_cp expects row-major grad tensors");
                }

                const auto V = static_cast<uint32_t>(grad_cp->shape()[0]);

                const uint64_t total = static_cast<uint64_t>(N) * static_cast<uint64_t>(D);
                const uint32_t grid_x = static_cast<uint32_t>((total + kernel.meta.block_size - 1) /
                                                              kernel.meta.block_size);

                pi::tensorlib::KernelLaunchArguments arguments{
                    .args = {grad_out->dataptr(),
                             cell_states->dataptr(),
                             grad_cp->dataptr(),
                             N,
                             D,
                             V,
                             stride_out_n,
                             stride_out_d,
                             stride_cp_v,
                             stride_cp_d,
                             static_cast<void *>(nullptr)},
                    .grid_dim_x = grid_x,
                    .grid_dim_y = 1,
                    .grid_dim_z = 1,
                    .block_dim_x = TRITON_WARP_SIZE * kernel.num_warps,
                    .block_dim_y = 1,
                    .block_dim_z = 1,
                    .shared_mem_bytes = kernel.shared_mem_bytes,
                    .device_ordinal = device_ordinal,
                };
                return arguments;
            },
            .cuda_descriptor = pi::tensorlib::MakeCudaKernelDescriptor(kernel.data, kernel.size),
            .hip_descriptor = pi::tensorlib::MakeHipKernelDescriptor(kernel.data, kernel.size),
        };
    }

    pi::tensorlib::ComputeKernelDescriptor CreateBuildCellEmbedsBwdColorComputeKernelDescriptor(
        const pi::tensorlib::DataType dtype, const bool output_fp32)
    {
        const auto kernel = output_fp32
                                ? SelectKernelForHalf(kbuild_cell_embeds_bwd_color_bf16_out_fp32,
                                                      kbuild_cell_embeds_bwd_color_fp16_out_fp32, dtype)
                                : SelectKernelForHalf(kbuild_cell_embeds_bwd_color_bf16,
                                                      kbuild_cell_embeds_bwd_color_fp16, dtype);
        const std::string kernel_name =
            std::string("build_cell_embeds_bwd_color_") + std::string(KernelSuffixForHalf(dtype)) +
            (output_fp32 ? "_out_fp32" : "");

        return pi::tensorlib::ComputeKernelDescriptor{
            .kernel_name = kernel_name,
            .function_name = kernel.function_name,
            .expected_arg_count = kernel.arg_count,
            .argument_provider = [kernel, dtype, output_fp32](
                                     const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                                                 const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
                -> pi::tensorlib::KernelLaunchArguments
            {
                if (inputs.size() != 2)
                {
                    throw std::runtime_error("build_cell_embeds_bwd_color expects 2 inputs");
                }
                if (outputs.size() != 6)
                {
                    throw std::runtime_error("build_cell_embeds_bwd_color expects 6 outputs");
                }

                const auto &grad_out = inputs[0];
                const auto &cell_states = inputs[1];

                const auto &grad_fg_r = outputs[0];
                const auto &grad_fg_g = outputs[1];
                const auto &grad_fg_b = outputs[2];
                const auto &grad_bg_r = outputs[3];
                const auto &grad_bg_g = outputs[4];
                const auto &grad_bg_b = outputs[5];

                const auto expected_out_dtype = output_fp32 ? pi::tensorlib::DataType::FLOAT32 : dtype;
                if (grad_out->dtype() != dtype || grad_fg_r->dtype() != expected_out_dtype ||
                    grad_fg_g->dtype() != expected_out_dtype || grad_fg_b->dtype() != expected_out_dtype ||
                    grad_bg_r->dtype() != expected_out_dtype || grad_bg_g->dtype() != expected_out_dtype ||
                    grad_bg_b->dtype() != expected_out_dtype)
                {
                    throw std::runtime_error("build_cell_embeds_bwd_color expects matching BFLOAT16/FLOAT16 tensors");
                }
                if (cell_states->dtype() != pi::tensorlib::DataType::UINT32)
                {
                    throw std::runtime_error("build_cell_embeds_bwd_color expects UINT32 cell state tensor");
                }

                const auto device_ordinal = ValidateSameDeviceOrdinal(
                    "build_cell_embeds_bwd_color",
                    {grad_out, cell_states, grad_fg_r, grad_fg_g, grad_fg_b, grad_bg_r, grad_bg_g, grad_bg_b});

                if (grad_out->shape().ndims() != 2)
                {
                    throw std::runtime_error("build_cell_embeds_bwd_color expects grad_out shaped (N, D)");
                }

                const auto N = static_cast<uint32_t>(grad_out->shape()[0]);
                const auto D = static_cast<uint32_t>(grad_out->shape()[1]);

                if (cell_states->shape().dims() != std::vector<uint64_t>{N, 3})
                {
                    throw std::runtime_error("build_cell_embeds_bwd_color expects cell_states shaped (N, 3)");
                }

                const auto check_embed = [&](const std::shared_ptr<pi::tensorlib::RealTensor> &tensor,
                                             const char *name)
                {
                    if (tensor->shape().ndims() != 1 || tensor->shape()[0] != D)
                    {
                        throw std::runtime_error(std::string(name) + " tensor must be of shape (D,)");
                    }
                    if (tensor->strides()[0] != 1)
                    {
                        throw std::runtime_error(std::string(name) + " tensor must be contiguous");
                    }
                };

                check_embed(grad_fg_r, "grad_fg_r");
                check_embed(grad_fg_g, "grad_fg_g");
                check_embed(grad_fg_b, "grad_fg_b");
                check_embed(grad_bg_r, "grad_bg_r");
                check_embed(grad_bg_g, "grad_bg_g");
                check_embed(grad_bg_b, "grad_bg_b");

                const auto stride_out_n = static_cast<uint32_t>(grad_out->strides()[0]);
                const auto stride_out_d = static_cast<uint32_t>(grad_out->strides()[1]);
                if (stride_out_n != D || stride_out_d != 1)
                {
                    throw std::runtime_error("build_cell_embeds_bwd_color expects row-major grad_out");
                }

                const uint32_t blocks_d = (D + kernel.meta.block_size_x - 1) / kernel.meta.block_size_x;
                const uint32_t blocks_n = (N + kernel.meta.block_size_y - 1) / kernel.meta.block_size_y;
                const uint64_t grid_x =
                    static_cast<uint64_t>(blocks_d) * static_cast<uint64_t>(blocks_n);
                if (grid_x > std::numeric_limits<uint32_t>::max())
                {
                    throw std::runtime_error("build_cell_embeds_bwd_color grid dimension overflow");
                }

                pi::tensorlib::KernelLaunchArguments arguments{
                    .args = {grad_out->dataptr(),
                             cell_states->dataptr(),
                             grad_fg_r->dataptr(),
                             grad_fg_g->dataptr(),
                             grad_fg_b->dataptr(),
                             grad_bg_r->dataptr(),
                             grad_bg_g->dataptr(),
                             grad_bg_b->dataptr(),
                             N,
                             D,
                             stride_out_n,
                             stride_out_d,
                             static_cast<void *>(nullptr)},
                    .grid_dim_x = static_cast<uint32_t>(grid_x),
                    .grid_dim_y = 1,
                    .grid_dim_z = 1,
                    .block_dim_x = TRITON_WARP_SIZE * kernel.num_warps,
                    .block_dim_y = 1,
                    .block_dim_z = 1,
                    .shared_mem_bytes = kernel.shared_mem_bytes,
                    .device_ordinal = device_ordinal,
                };
                return arguments;
            },
            .cuda_descriptor = pi::tensorlib::MakeCudaKernelDescriptor(kernel.data, kernel.size),
            .hip_descriptor = pi::tensorlib::MakeHipKernelDescriptor(kernel.data, kernel.size),
        };
    }

    pi::tensorlib::ComputeKernelDescriptor CreateBuildCellEmbedsBwdPosComputeKernelDescriptor(
        const pi::tensorlib::DataType dtype, const bool output_fp32)
    {
        const auto kernel = output_fp32
                                ? SelectKernelForHalf(kbuild_cell_embeds_bwd_pos_bf16_out_fp32,
                                                      kbuild_cell_embeds_bwd_pos_fp16_out_fp32, dtype)
                                : SelectKernelForHalf(kbuild_cell_embeds_bwd_pos_bf16, kbuild_cell_embeds_bwd_pos_fp16,
                                                      dtype);
        const std::string kernel_name =
            std::string("build_cell_embeds_bwd_pos_") + std::string(KernelSuffixForHalf(dtype)) +
            (output_fp32 ? "_out_fp32" : "");

        return pi::tensorlib::ComputeKernelDescriptor{
            .kernel_name = kernel_name,
            .function_name = kernel.function_name,
            .expected_arg_count = kernel.arg_count,
            .argument_provider = [kernel, dtype, output_fp32](
                                     const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                                     const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
                -> pi::tensorlib::KernelLaunchArguments
            {
                if (inputs.size() != 1)
                {
                    throw std::runtime_error("build_cell_embeds_bwd_pos expects 1 input");
                }
                if (outputs.size() != 1)
                {
                    throw std::runtime_error("build_cell_embeds_bwd_pos expects 1 output");
                }

                const auto &grad_out = inputs[0];
                const auto &grad_pos = outputs[0];

                const auto expected_out_dtype = output_fp32 ? pi::tensorlib::DataType::FLOAT32 : dtype;
                if (grad_out->dtype() != dtype || grad_pos->dtype() != expected_out_dtype)
                {
                    throw std::runtime_error("build_cell_embeds_bwd_pos expects matching BFLOAT16/FLOAT16 tensors");
                }

                const auto device_ordinal = ValidateSameDeviceOrdinal("build_cell_embeds_bwd_pos", {grad_out, grad_pos});

                if (grad_out->shape().ndims() != 2)
                {
                    throw std::runtime_error("build_cell_embeds_bwd_pos expects grad_out shaped (N, D)");
                }
                if (grad_pos->shape().ndims() != 2)
                {
                    throw std::runtime_error("build_cell_embeds_bwd_pos expects grad_pos shaped (P, D)");
                }

                const auto N = static_cast<uint32_t>(grad_out->shape()[0]);
                const auto D = static_cast<uint32_t>(grad_out->shape()[1]);
                const auto P = static_cast<uint32_t>(grad_pos->shape()[0]);

                if (grad_pos->shape()[1] != D)
                {
                    throw std::runtime_error("build_cell_embeds_bwd_pos expects matching embedding dimension");
                }
                if (P == 0 || N == 0 || (N % P) != 0)
                {
                    throw std::runtime_error("build_cell_embeds_bwd_pos expects N to be a multiple of P");
                }

                const auto stride_out_n = static_cast<uint32_t>(grad_out->strides()[0]);
                const auto stride_out_d = static_cast<uint32_t>(grad_out->strides()[1]);
                const auto stride_pos_p = static_cast<uint32_t>(grad_pos->strides()[0]);
                const auto stride_pos_d = static_cast<uint32_t>(grad_pos->strides()[1]);

                if (stride_out_n != D || stride_out_d != 1 || stride_pos_p != D || stride_pos_d != 1)
                {
                    throw std::runtime_error("build_cell_embeds_bwd_pos expects row-major grad tensors");
                }

                const auto batch = static_cast<uint32_t>(N / P);
                const uint32_t blocks_d = (D + kernel.meta.block_size_x - 1) / kernel.meta.block_size_x;
                const uint64_t grid_x = static_cast<uint64_t>(P) * static_cast<uint64_t>(blocks_d);
                if (grid_x > std::numeric_limits<uint32_t>::max())
                {
                    throw std::runtime_error("build_cell_embeds_bwd_pos grid dimension overflow");
                }

                pi::tensorlib::KernelLaunchArguments arguments{
                    .args = {grad_out->dataptr(),
                             grad_pos->dataptr(),
                             batch,
                             P,
                             D,
                             stride_out_n,
                             stride_out_d,
                             stride_pos_p,
                             stride_pos_d,
                             static_cast<void *>(nullptr)},
                    .grid_dim_x = static_cast<uint32_t>(grid_x),
                    .grid_dim_y = 1,
                    .grid_dim_z = 1,
                    .block_dim_x = TRITON_WARP_SIZE * kernel.num_warps,
                    .block_dim_y = 1,
                    .block_dim_z = 1,
                    .shared_mem_bytes = kernel.shared_mem_bytes,
                    .device_ordinal = device_ordinal,
                };
                return arguments;
            },
            .cuda_descriptor = pi::tensorlib::MakeCudaKernelDescriptor(kernel.data, kernel.size),
            .hip_descriptor = pi::tensorlib::MakeHipKernelDescriptor(kernel.data, kernel.size),
        };
    }
} // namespace

void BuildCellEmbedBwdPass::transform(pi::tensorlib::ExecutionPlan &execution_plan)
{
    size_t next_entry_id = 0;
    for (const auto &entry : execution_plan.entries)
    {
        next_entry_id = std::max(next_entry_id, entry.id + 1);
    }

    size_t i = 0;
    while (i < execution_plan.entries.size())
    {
        if (execution_plan.entries[i].op_type != pi::tensorlib::OpType::CUSTOM_OP)
        {
            ++i;
            continue;
        }

        auto &custom_op = execution_plan.entries[i];
        const auto &attributes = custom_op.attributes;
        const auto custom_op_name_it = attributes.find("custom_op_name");
        if (custom_op_name_it == attributes.end())
        {
            ++i;
            continue;
        }

        const auto custom_op_name = std::any_cast<const char *>(custom_op_name_it->second);
        if (std::string(custom_op_name) == "build_cell_embed_bwd_cp")
        {
            if (custom_op.inputs.size() != 2 || custom_op.outputs.size() != 1)
            {
                ++i;
                continue;
            }

            const auto &grad_out = custom_op.inputs[0];
            const auto dtype = grad_out->dtype();
            if (!IsSupportedHalfType(dtype))
            {
                throw std::runtime_error("build_cell_embed_bwd_cp expects BFLOAT16 or FLOAT16 gradients");
            }

            const auto &grad_cp = custom_op.outputs[0];
            const bool output_fp32 = grad_cp->dtype() == pi::tensorlib::DataType::FLOAT32;
            const size_t entry_idx = i;

            auto &entry = execution_plan.entries[entry_idx];
            entry.op_type = std::nullopt;
            entry.kernel_descriptor = CreateBuildCellEmbedsBwdCpComputeKernelDescriptor(dtype, output_fp32);
            const uint64_t elements = entry.inputs[0]->shape().numel();
            entry.flop_estimate = elements; // scaled adds
            if (const char *env = std::getenv("FBAMTRAIN_DEBUG_BUILD_CELL_EMBED_BWD"); env != nullptr && env[0] != '\0')
            {
                std::clog << "[BuildCellEmbedBwd] op=cp kernel="
                          << entry.kernel_descriptor->kernel_name << " output_fp32=" << output_fp32 << " out_id="
                          << entry.outputs[0]->id() << " out_dtype=" << static_cast<int>(entry.outputs[0]->dtype())
                          << '\n';
            }

            i = entry_idx + 1;
            continue;
        }

        if (std::string(custom_op_name) == "build_cell_embed_bwd_color")
        {
            if (custom_op.inputs.size() != 2 || custom_op.outputs.size() != 6)
            {
                ++i;
                continue;
            }

            const auto &grad_out = custom_op.inputs[0];
            const auto dtype = grad_out->dtype();
            if (!IsSupportedHalfType(dtype))
            {
                throw std::runtime_error("build_cell_embed_bwd_color expects BFLOAT16 or FLOAT16 gradients");
            }

            const auto &grad_fg_r = custom_op.outputs[0];
            const auto original_outputs = custom_op.outputs;
            const bool output_fp32 = grad_fg_r->dtype() == pi::tensorlib::DataType::FLOAT32;
            const size_t entry_idx = i;

            auto &entry = execution_plan.entries[entry_idx];
            entry.op_type = std::nullopt;
            entry.kernel_descriptor = CreateBuildCellEmbedsBwdColorComputeKernelDescriptor(dtype, output_fp32);
            const uint64_t elements = entry.inputs[0]->shape().numel();
            entry.flop_estimate = elements * 6; // six color channels
            if (const char *env = std::getenv("FBAMTRAIN_DEBUG_BUILD_CELL_EMBED_BWD"); env != nullptr && env[0] != '\0')
            {
                std::clog << "[BuildCellEmbedBwd] op=color kernel="
                          << entry.kernel_descriptor->kernel_name << " output_fp32=" << output_fp32
                          << " out0_id=" << entry.outputs[0]->id()
                          << " out0_dtype=" << static_cast<int>(entry.outputs[0]->dtype()) << '\n';
            }

            i = entry_idx + 1;
            continue;
        }

        if (std::string(custom_op_name) == "build_cell_embed_bwd_pos")
        {
            if (custom_op.inputs.size() != 1 || custom_op.outputs.size() != 1)
            {
                ++i;
                continue;
            }

            const auto &grad_out = custom_op.inputs[0];
            const auto dtype = grad_out->dtype();
            if (!IsSupportedHalfType(dtype))
            {
                throw std::runtime_error("build_cell_embed_bwd_pos expects BFLOAT16 or FLOAT16 gradients");
            }

            const auto &grad_pos = custom_op.outputs[0];
            const bool output_fp32 = grad_pos->dtype() == pi::tensorlib::DataType::FLOAT32;
            const size_t entry_idx = i;

            auto &entry = execution_plan.entries[entry_idx];
            entry.op_type = std::nullopt;
            entry.kernel_descriptor = CreateBuildCellEmbedsBwdPosComputeKernelDescriptor(dtype, output_fp32);
            const uint64_t elements = entry.inputs[0]->shape().numel();
            entry.flop_estimate = elements;
            if (const char *env = std::getenv("FBAMTRAIN_DEBUG_BUILD_CELL_EMBED_BWD"); env != nullptr && env[0] != '\0')
            {
                std::clog << "[BuildCellEmbedBwd] op=pos kernel=" << entry.kernel_descriptor->kernel_name
                          << " output_fp32=" << output_fp32 << " out_id=" << entry.outputs[0]->id()
                          << " out_dtype=" << static_cast<int>(entry.outputs[0]->dtype()) << '\n';
            }

            i = entry_idx + 1;
            continue;
        }

        ++i;
    }

}
