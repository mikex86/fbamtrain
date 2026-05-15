#include "passes.h"

#include <kernels/kernel_binaries.h>
#include "passes/pass_utils.h"

#include <array>
#include <iostream>
#include <stdexcept>

namespace
{
struct LstmKernelConfig
{
    const kernel_bin_t<kernel_meta_elementwise_t> *kernel;
    std::string name;
    std::array<pi::tensorlib::DataType, 5> dtypes;
};

bool MatchesDtypes(const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                   const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs,
                   const LstmKernelConfig &config)
{
    if (inputs.size() != 2 || (outputs.size() != 2 && outputs.size() != 3))
    {
        return false;
    }

    const auto &gates = inputs[0];
    const auto &c_prev = inputs[1];
    const auto &h_out = outputs[0];
    const auto &c_out = outputs[1];
    const auto &y_out = (outputs.size() >= 3) ? outputs[2] : outputs[0];

    return gates->dtype() == config.dtypes[0] && c_prev->dtype() == config.dtypes[1] &&
           h_out->dtype() == config.dtypes[2] && c_out->dtype() == config.dtypes[3] &&
           y_out->dtype() == config.dtypes[4];
}

pi::tensorlib::ComputeKernelDescriptor CreateLstmCellKernelDescriptor(const LstmKernelConfig &config)
{
    const auto *kernel = config.kernel;
    if (kernel == nullptr)
    {
        throw std::runtime_error("lstm_cell_fwd kernel config missing kernel binary");
    }

    return pi::tensorlib::ComputeKernelDescriptor{
        .kernel_name = config.name,
        .function_name = kernel->function_name,
        .expected_arg_count = kernel->arg_count,
        .argument_provider = [config, kernel](const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                                              const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
            -> pi::tensorlib::KernelLaunchArguments
        {
            if (inputs.size() != 2 || (outputs.size() != 2 && outputs.size() != 3))
            {
                throw std::runtime_error(
                    "LSTM_CELL_FWD expects two inputs (gates, c_prev) and two or three outputs (h, c, h_out_contig)");
            }
            const auto &gates = inputs[0];
            const auto &c_prev = inputs[1];
            const auto &h_out = outputs[0];
            const auto &c_out = outputs[1];
            const auto &y_out = (outputs.size() >= 3) ? outputs[2] : outputs[0];

            if (!MatchesDtypes(inputs, outputs, config))
            {
                throw std::runtime_error("LSTM_CELL_FWD dtype mismatch for kernel " + config.name);
            }
            if (gates->shape().ndims() != 2 || c_prev->shape().ndims() != 2 || h_out->shape().ndims() != 2 ||
                c_out->shape().ndims() != 2 || y_out->shape().ndims() != 2)
            {
                throw std::runtime_error("LSTM_CELL_FWD expects 2D tensors");
            }

            const auto batch = gates->shape()[0];
            const auto gate_dim = gates->shape()[1];
            if (gate_dim % 4 != 0)
            {
                throw std::runtime_error("LSTM_CELL_FWD gate dimension must be divisible by 4");
            }
            const auto hidden = gate_dim / 4;
            if (c_prev->shape()[0] != batch || c_prev->shape()[1] != hidden || h_out->shape()[0] != batch ||
                h_out->shape()[1] != hidden || c_out->shape()[0] != batch || c_out->shape()[1] != hidden ||
                y_out->shape()[0] != batch || y_out->shape()[1] != hidden)
            {
                throw std::runtime_error("LSTM_CELL_FWD shape mismatch");
            }

            const auto device_ordinal =
                ValidateSameDeviceOrdinal("lstm_cell_fwd", {gates, c_prev, h_out, c_out, y_out});
            const auto &gate_strides = gates->strides().strides();
            const auto &c_prev_strides = c_prev->strides().strides();
            const auto &h_out_strides = h_out->strides().strides();
            const auto &c_out_strides = c_out->strides().strides();
            const auto &y_out_strides = y_out->strides().strides();
            const auto gate_stride = !gate_strides.empty() ? gate_strides[0] : hidden * 4;
            const auto c_prev_stride = !c_prev_strides.empty() ? c_prev_strides[0] : hidden;
            const auto h_out_stride = !h_out_strides.empty() ? h_out_strides[0] : hidden;
            const auto c_out_stride = !c_out_strides.empty() ? c_out_strides[0] : hidden;
            const auto y_out_stride = !y_out_strides.empty() ? y_out_strides[0] : hidden;

            const uint32_t total = static_cast<uint32_t>(batch * hidden);
            const uint32_t block_elems = kernel->meta.block_size;
            if (block_elems == 0)
            {
                throw std::runtime_error("lstm_cell_fwd kernel meta block size is zero");
            }
            if (const char *env = std::getenv("FBAMTRAIN_DEBUG_LSTM"); env != nullptr)
            {
                std::clog << "[LSTM] block_elems=" << block_elems << " num_warps=" << kernel->num_warps << '\n';
            }

            pi::tensorlib::KernelLaunchArguments args{
                .args = {gates->dataptr(),
                         c_prev->dataptr(),
                         h_out->dataptr(),
                         c_out->dataptr(),
                         y_out->dataptr(),
                         static_cast<int32_t>(batch),
                         static_cast<int32_t>(hidden),
                         static_cast<int32_t>(gate_stride),
                         static_cast<int32_t>(c_prev_stride),
                         static_cast<int32_t>(h_out_stride),
                         static_cast<int32_t>(c_out_stride),
                         static_cast<int32_t>(y_out_stride),
                         static_cast<void *>(nullptr)},
                .grid_dim_x = (total + block_elems - 1) / block_elems,
                .grid_dim_y = 1,
                .grid_dim_z = 1,
                .block_dim_x = TRITON_WARP_SIZE * kernel->num_warps,
                .block_dim_y = 1,
                .block_dim_z = 1,
                .shared_mem_bytes = kernel->shared_mem_bytes,
                .device_ordinal = static_cast<int>(device_ordinal),
            };
            return args;
        },
        .cuda_descriptor = pi::tensorlib::MakeCudaKernelDescriptor(kernel->data, kernel->size),
        .hip_descriptor = pi::tensorlib::MakeHipKernelDescriptor(kernel->data, kernel->size),
    };
}
} // namespace

void LstmCellImplPass::transform(pi::tensorlib::ExecutionPlan &execution_plan)
{
    static const LstmKernelConfig fp16_config{&klstm_cell_fwd_out_fp16,
                                              "lstm_cell_fwd_out_fp16",
                                              {pi::tensorlib::DataType::FLOAT16,
                                               pi::tensorlib::DataType::FLOAT16,
                                               pi::tensorlib::DataType::FLOAT16,
                                               pi::tensorlib::DataType::FLOAT16,
                                               pi::tensorlib::DataType::FLOAT16}};
    static const LstmKernelConfig fp32_state_fp16_config{&klstm_cell_fwd_fp32_state_out_fp16,
                                                         "lstm_cell_fwd_fp32_state_out_fp16",
                                                         {pi::tensorlib::DataType::FLOAT32,
                                                          pi::tensorlib::DataType::FLOAT32,
                                                          pi::tensorlib::DataType::FLOAT16,
                                                          pi::tensorlib::DataType::FLOAT32,
                                                          pi::tensorlib::DataType::FLOAT16}};
    static const LstmKernelConfig fp32_state_bf16_config{&klstm_cell_fwd_fp32_state_out_bf16,
                                                         "lstm_cell_fwd_fp32_state_out_bf16",
                                                         {pi::tensorlib::DataType::FLOAT32,
                                                          pi::tensorlib::DataType::FLOAT32,
                                                          pi::tensorlib::DataType::BFLOAT16,
                                                          pi::tensorlib::DataType::FLOAT32,
                                                          pi::tensorlib::DataType::BFLOAT16}};

    static const auto fp16_descriptor = CreateLstmCellKernelDescriptor(fp16_config);
    static const auto fp32_fp16_descriptor = CreateLstmCellKernelDescriptor(fp32_state_fp16_config);
    static const auto fp32_bf16_descriptor = CreateLstmCellKernelDescriptor(fp32_state_bf16_config);

    for (auto &entry : execution_plan.entries)
    {
        if (entry.op_type != pi::tensorlib::OpType::LSTM_CELL_FWD)
        {
            continue;
        }
        if (entry.inputs.size() != 2 || (entry.outputs.size() != 2 && entry.outputs.size() != 3))
        {
            throw std::runtime_error("LSTM_CELL_FWD expects two inputs and two or three outputs");
        }

        const auto &gates = entry.inputs[0];
        const bool fp16_case = MatchesDtypes(entry.inputs, entry.outputs, fp16_config);
        const bool fp32_state_fp16_case = MatchesDtypes(entry.inputs, entry.outputs, fp32_state_fp16_config);
        const bool fp32_state_bf16_case = MatchesDtypes(entry.inputs, entry.outputs, fp32_state_bf16_config);
        if (!fp16_case && !fp32_state_fp16_case && !fp32_state_bf16_case)
        {
            throw std::runtime_error("Unsupported dtype combination for LSTM_CELL_FWD");
        }

        entry.op_type = std::nullopt;
        entry.kernel_descriptor = fp32_state_fp16_case   ? fp32_fp16_descriptor
                                 : fp32_state_bf16_case ? fp32_bf16_descriptor
                                                        : fp16_descriptor;
        const uint64_t gate_dim = gates->shape().ndims() >= 2 ? gates->shape()[1] : 0;
        const uint64_t hidden = gate_dim / 4;
        const uint64_t batch = gates->shape().ndims() >= 1 ? gates->shape()[0] : 0;
        const uint64_t elements = batch * hidden;
        // Per element: 4 bias adds + 3 sigmoids (~4 flops each) + g gate (mul/exp/div ~5)
        // + cell update (3) + tanh (5) + output mul (1) ≈ 30 flops.
        const uint64_t per_element = 30;
        entry.flop_estimate = elements * per_element;
    }
}
