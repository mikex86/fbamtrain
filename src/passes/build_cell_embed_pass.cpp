#include "passes.h"

#include "passes/pass_utils.h"
#include <kernels/kernel_binaries.h>

#include <any>
#include <array>
#include <limits>
#include <string>

static pi::tensorlib::ComputeKernelDescriptor CreateBuildCellEmbedsComputeKernelDescriptor(
    const pi::tensorlib::DataType dtype)
{
    const auto kernel = SelectKernelForHalf(kbuild_cell_embeds_bf16, kbuild_cell_embeds_fp16, dtype);
    const std::string kernel_name = std::string("build_cell_embeds_") + std::string(KernelSuffixForHalf(dtype));

    return pi::tensorlib::ComputeKernelDescriptor{
        .kernel_name = kernel_name,
        .function_name = kernel.function_name,
        .expected_arg_count = kernel.arg_count,
        .argument_provider = [kernel, dtype](const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                                             const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
            -> pi::tensorlib::KernelLaunchArguments
        {
            if (inputs.size() != 10)
            {
                throw std::runtime_error("Invalid number of inputs for build_cell_embeds kernel");
            }
            if (outputs.size() != 1)
            {
                throw std::runtime_error("Invalid number of outputs for build_cell_embeds kernel");
            }

            if (inputs[0]->id() != outputs[0]->id())
            {
                throw std::runtime_error("build_cell_embeds is an in-place operation and requires identical input/output tensors");
            }

            const auto &out = outputs[0];
            const auto &input_cell_states = inputs[1];
            const auto &cp_embed = inputs[2];
            const auto &position_embed = inputs[3];
            const auto &fg_r_embed = inputs[4];
            const auto &fg_g_embed = inputs[5];
            const auto &fg_b_embed = inputs[6];
            const auto &bg_r_embed = inputs[7];
            const auto &bg_g_embed = inputs[8];
            const auto &bg_b_embed = inputs[9];

            if (out->dtype() != dtype || cp_embed->dtype() != dtype || position_embed->dtype() != dtype ||
                fg_r_embed->dtype() != dtype || fg_g_embed->dtype() != dtype || fg_b_embed->dtype() != dtype ||
                bg_r_embed->dtype() != dtype || bg_g_embed->dtype() != dtype || bg_b_embed->dtype() != dtype)
            {
                throw std::runtime_error("build_cell_embeds expects all embedding tensors to share the requested dtype");
            }
            if (input_cell_states->dtype() != pi::tensorlib::DataType::UINT32)
            {
                throw std::runtime_error("build_cell_embeds expects UINT32 cell state tensor");
            }

            const auto device_ordinal = ValidateSameDeviceOrdinal(
                "build_cell_embeds",
                {out, cp_embed, position_embed, fg_r_embed, fg_g_embed, fg_b_embed, bg_r_embed, bg_g_embed, bg_b_embed});

            if (out->shape().ndims() != 2)
            {
                throw std::runtime_error("output tensor must be of shape (N, D)");
            }
            const auto N = static_cast<uint32_t>(out->shape()[0]);
            const auto D = static_cast<uint32_t>(out->shape()[1]);

            if (input_cell_states->shape().dims() != std::vector<uint64_t>{N, 3})
            {
                throw std::runtime_error("input_cell_states tensor must be of shape (N, 3)");
            }
            if (cp_embed->shape().ndims() != 2 || cp_embed->shape()[1] != D)
            {
                throw std::runtime_error("cp_embed tensor must be of shape (V, D)");
            }
            if (position_embed->shape().ndims() != 2 || position_embed->shape()[1] != D)
            {
                throw std::runtime_error("position_embed tensor must be of shape (P, D)");
            }

            const auto V = static_cast<uint32_t>(cp_embed->shape()[0]);
            const auto P = static_cast<uint32_t>(position_embed->shape()[0]);
            if (P == 0)
            {
                throw std::runtime_error("position_embed must have at least one position");
            }

            const auto check_color_embed = [&](const std::shared_ptr<pi::tensorlib::RealTensor> &tensor,
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

            check_color_embed(fg_r_embed, "fg_r_embed");
            check_color_embed(fg_g_embed, "fg_g_embed");
            check_color_embed(fg_b_embed, "fg_b_embed");
            check_color_embed(bg_r_embed, "bg_r_embed");
            check_color_embed(bg_g_embed, "bg_g_embed");
            check_color_embed(bg_b_embed, "bg_b_embed");

            const auto stride_out_n = static_cast<uint32_t>(out->strides()[0]);
            const auto stride_out_d = static_cast<uint32_t>(out->strides()[1]);
            const auto stride_cp_v = static_cast<uint32_t>(cp_embed->strides()[0]);
            const auto stride_cp_d = static_cast<uint32_t>(cp_embed->strides()[1]);
            const auto stride_pos_p = static_cast<uint32_t>(position_embed->strides()[0]);
            const auto stride_pos_d = static_cast<uint32_t>(position_embed->strides()[1]);

            if (stride_out_n != D || stride_out_d != 1 || stride_cp_v != D || stride_cp_d != 1 ||
                stride_pos_p != D || stride_pos_d != 1)
            {
                throw std::runtime_error("build_cell_embeds expects row-major embeddings");
            }

            const uint32_t blocks_d =
                (D + kernel.meta.block_size_x - 1) / kernel.meta.block_size_x;
            const uint32_t blocks_n =
                (N + kernel.meta.block_size_y - 1) / kernel.meta.block_size_y;
            const uint64_t grid_x = static_cast<uint64_t>(blocks_d) * static_cast<uint64_t>(blocks_n);
            if (grid_x > std::numeric_limits<uint32_t>::max())
            {
                throw std::runtime_error("build_cell_embeds grid dimension overflow");
            }

            pi::tensorlib::KernelLaunchArguments arguments{
                .args = {
                    out->dataptr(),
                    input_cell_states->dataptr(),
                    cp_embed->dataptr(),
                    position_embed->dataptr(),
                    fg_r_embed->dataptr(),
                    fg_g_embed->dataptr(),
                    fg_b_embed->dataptr(),
                    bg_r_embed->dataptr(),
                    bg_g_embed->dataptr(),
                    bg_b_embed->dataptr(),
                    N,
                    D,
                    V,
                    P,
                    stride_out_n,
                    stride_out_d,
                    stride_cp_v,
                    stride_cp_d,
                    stride_pos_p,
                    stride_pos_d,
                    static_cast<void *>(nullptr)
                },
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

void BuildCellEmbedPass::transform(pi::tensorlib::ExecutionPlan &execution_plan)
{
    for (auto &custom_op : execution_plan.entries)
    {
        if (custom_op.op_type == pi::tensorlib::OpType::CUSTOM_OP)
        {
            const auto &attributes = custom_op.attributes;
            const auto custom_op_name_it = attributes.find("custom_op_name");
            if (custom_op_name_it == attributes.end())
            {
                continue;
            }
            if (const auto custom_op_name = std::any_cast<const char *>(custom_op_name_it->second);
                std::string(custom_op_name) != "build_cell_embed")
            {
                continue;
            }
            if (custom_op.inputs.size() != 10)
            {
                continue;
            }
            if (custom_op.outputs.size() != 1)
            {
                continue;
            }
            const auto &cell_embeddings = custom_op.inputs[0];
            const auto &cell_states = custom_op.inputs[1];

            const auto dtype = cell_embeddings->dtype();
            if (!IsSupportedHalfType(dtype))
            {
                throw std::runtime_error("cell_embeddings must be of type BFLOAT16 or FLOAT16");
            }
            if (cell_states->dtype() != pi::tensorlib::DataType::UINT32)
            {
                throw std::runtime_error("cell_states must be of type UINT32");
            }
            const auto &position_embed = custom_op.inputs[3];
            if (position_embed->dtype() != dtype)
            {
                throw std::runtime_error("position_embed must match cell_embeddings dtype");
            }

            custom_op.op_type = std::nullopt; // mark as kernel
            custom_op.kernel_descriptor = CreateBuildCellEmbedsComputeKernelDescriptor(dtype);
            const uint64_t elements = custom_op.outputs[0]->shape().numel();
            custom_op.flop_estimate = elements * 8; // combine cp/pos + six color embeds
        }
    }
}
