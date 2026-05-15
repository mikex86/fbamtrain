#include "passes.h"

#include <kernels/kernel_binaries.h>
#include "passes/pass_utils.h"

#include <iostream>
#include <stdexcept>

namespace
{
struct LstmBwdKernelConfig
{
    const kernel_bin_t<kernel_meta_elementwise_t> *kernel;
    std::string name;
};

pi::tensorlib::ComputeKernelDescriptor CreateRecomputeKernelDescriptor(const LstmBwdKernelConfig &config)
{
    const auto *kernel = config.kernel;
    if (kernel == nullptr)
    {
        throw std::runtime_error("lstm_cell_recompute kernel config missing kernel binary");
    }

    return pi::tensorlib::ComputeKernelDescriptor{
        .kernel_name = config.name,
        .function_name = kernel->function_name,
        .expected_arg_count = kernel->arg_count,
        .argument_provider = [kernel](const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                                      const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
            -> pi::tensorlib::KernelLaunchArguments
        {
            if (inputs.size() != 3 || outputs.size() != 3)
            {
                throw std::runtime_error("LSTM_CELL_RECOMPUTE expects three inputs (gates, bias, c_prev) and three "
                                         "outputs (gate_out, h_out, c_out)");
            }
            const auto &gates = inputs[0];
            const auto &bias = inputs[1];
            const auto &c_prev = inputs[2];
            const auto &gate_out = outputs[0];
            const auto &h_out = outputs[1];
            const auto &c_out = outputs[2];

            const auto gate_dtype = gates->dtype();
            const auto h_out_dtype = h_out->dtype();
            if ((gate_dtype != pi::tensorlib::DataType::FLOAT16 && gate_dtype != pi::tensorlib::DataType::FLOAT32) ||
                bias->dtype() != pi::tensorlib::DataType::FLOAT32 || c_prev->dtype() != pi::tensorlib::DataType::FLOAT32 ||
                gate_out->dtype() != pi::tensorlib::DataType::FLOAT32 ||
                (h_out_dtype != pi::tensorlib::DataType::FLOAT16 && h_out_dtype != pi::tensorlib::DataType::BFLOAT16) ||
                c_out->dtype() != pi::tensorlib::DataType::FLOAT32)
            {
                throw std::runtime_error("LSTM_CELL_RECOMPUTE dtype mismatch");
            }

            if (gates->shape().ndims() != 2 || c_prev->shape().ndims() != 2 || gate_out->shape().ndims() != 2 ||
                h_out->shape().ndims() != 2 || c_out->shape().ndims() != 2 || bias->shape().ndims() != 1)
            {
                throw std::runtime_error("LSTM_CELL_RECOMPUTE expects gates/c_prev/gate_out/h_out/c_out to be 2D and "
                                         "bias to be 1D");
            }

            const auto batch = gates->shape()[0];
            const auto gate_dim = gates->shape()[1];
            if (gate_dim % 4 != 0)
            {
                throw std::runtime_error("LSTM_CELL_RECOMPUTE gate dimension must be divisible by 4");
            }
            const auto hidden = gate_dim / 4;
            if (c_prev->shape()[0] != batch || c_prev->shape()[1] != hidden || gate_out->shape()[0] != batch ||
                gate_out->shape()[1] != gate_dim || h_out->shape()[0] != batch || h_out->shape()[1] != hidden ||
                c_out->shape()[0] != batch || c_out->shape()[1] != hidden || bias->shape()[0] != gate_dim)
            {
                throw std::runtime_error("LSTM_CELL_RECOMPUTE shape mismatch");
            }

            if (const char *env = std::getenv("FBAMTRAIN_DEBUG_GEMM_VALUES"); env != nullptr)
            {
                static bool logged = false;
                if (!logged)
                {
                    logged = true;
                    auto dump_half = [](const std::shared_ptr<pi::tensorlib::RealTensor> &t, const char *label) {
                        auto storage = t->storage();
                        if (storage->device().device_type != pi::tensorlib::DeviceType::CPU)
                        {
                            storage = storage->toCPU();
                        }
                        const auto *ptr = static_cast<const uint16_t *>(storage->dataptr());
                        std::clog << "[lstm_recompute] " << label << "[0,0..4]=" << ptr[0] << ", " << ptr[1] << ", "
                                  << ptr[2] << ", " << ptr[3] << ", " << ptr[4] << '\n';
                    };
                    auto dump_f32 = [](const std::shared_ptr<pi::tensorlib::RealTensor> &t, const char *label) {
                        auto storage = t->storage();
                        if (storage->device().device_type != pi::tensorlib::DeviceType::CPU)
                        {
                            storage = storage->toCPU();
                        }
                        const auto *ptr = static_cast<const float *>(storage->dataptr());
                        std::clog << "[lstm_recompute] " << label << "[0,0..4]=" << ptr[0] << ", " << ptr[1] << ", "
                                  << ptr[2] << ", " << ptr[3] << ", " << ptr[4] << '\n';
                    };
                    dump_f32(gates, "gates");
                    dump_f32(bias, "bias");
                    dump_f32(c_prev, "c_prev");
                }
            }

            const auto device_ordinal =
                ValidateSameDeviceOrdinal("lstm_cell_recompute", {gates, bias, c_prev, gate_out, h_out, c_out});

            const auto &gate_strides = gates->strides().strides();
            const auto &c_prev_strides = c_prev->strides().strides();
            const auto &gate_out_strides = gate_out->strides().strides();
            const auto &h_out_strides = h_out->strides().strides();
            const auto &c_out_strides = c_out->strides().strides();
            const auto gate_stride = !gate_strides.empty() ? gate_strides[0] : gate_dim;
            const auto bias_stride = bias->strides().strides().empty() ? hidden * 4 : bias->strides()[0];
            const auto c_prev_stride = !c_prev_strides.empty() ? c_prev_strides[0] : hidden;
            const auto gate_out_stride = !gate_out_strides.empty() ? gate_out_strides[0] : gate_dim;
            const auto h_out_stride = !h_out_strides.empty() ? h_out_strides[0] : hidden;
            const auto c_out_stride = !c_out_strides.empty() ? c_out_strides[0] : hidden;

            const uint32_t total = static_cast<uint32_t>(batch * hidden);
            const uint32_t block_elems = kernel->meta.block_size;

            pi::tensorlib::KernelLaunchArguments args{
                .args = {gates->dataptr(),
                         bias->dataptr(),
                         c_prev->dataptr(),
                         gate_out->dataptr(),
                         h_out->dataptr(),
                         c_out->dataptr(),
                         static_cast<int32_t>(batch),
                         static_cast<int32_t>(hidden),
                         static_cast<int32_t>(gate_stride),
                         static_cast<int32_t>(bias_stride),
                         static_cast<int32_t>(c_prev_stride),
                         static_cast<int32_t>(gate_out_stride),
                         static_cast<int32_t>(h_out_stride),
                         static_cast<int32_t>(c_out_stride), static_cast<void *>(nullptr)},
                .grid_dim_x = (total + block_elems - 1) / block_elems,
                .grid_dim_y = 1,
                .grid_dim_z = 1,
                .block_dim_x = TRITON_WARP_SIZE * kernel->num_warps,
                .block_dim_y = 1,
                .block_dim_z = 1,
                .shared_mem_bytes = kernel->shared_mem_bytes,
                .device_ordinal = static_cast<int>(device_ordinal)};
            return args;
        },
        .cuda_descriptor = pi::tensorlib::MakeCudaKernelDescriptor(kernel->data, kernel->size),
        .hip_descriptor = pi::tensorlib::MakeHipKernelDescriptor(kernel->data, kernel->size)};
}

pi::tensorlib::ComputeKernelDescriptor CreateBwdPointwiseKernelDescriptor(const LstmBwdKernelConfig &config)
{
    const auto *kernel = config.kernel;
    if (kernel == nullptr)
    {
        throw std::runtime_error("lstm_cell_bwd_pointwise kernel config missing kernel binary");
    }
    return pi::tensorlib::ComputeKernelDescriptor{
        .kernel_name = config.name,
        .function_name = kernel->function_name,
        .expected_arg_count = kernel->arg_count,
        .argument_provider = [kernel](const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                                      const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
            -> pi::tensorlib::KernelLaunchArguments
        {
            if (inputs.size() != 6 || outputs.size() != 3)
            {
                throw std::runtime_error("LSTM_CELL_BWD expects six inputs (dY, dh_next, dc_next, gate_out, c_prev, "
                                         "c_out) and three outputs (dGates, dGates_half, dc_prev)");
            }
            const auto &dY = inputs[0];
            const auto &dh_next = inputs[1];
            const auto &dc_next = inputs[2];
            const auto &gate_out = inputs[3];
            const auto &c_prev = inputs[4];
            const auto &c_out = inputs[5];
            const auto &dGates = outputs[0];
            const auto &dGates_half = outputs[1];
            const auto &dc_prev = outputs[2];

            const auto dGates_half_dtype = dGates_half->dtype();
            if (dY->dtype() != pi::tensorlib::DataType::FLOAT32 || dh_next->dtype() != pi::tensorlib::DataType::FLOAT32 ||
                dc_next->dtype() != pi::tensorlib::DataType::FLOAT32 || gate_out->dtype() != pi::tensorlib::DataType::FLOAT32 ||
                c_prev->dtype() != pi::tensorlib::DataType::FLOAT32 || c_out->dtype() != pi::tensorlib::DataType::FLOAT32 ||
                dGates->dtype() != pi::tensorlib::DataType::FLOAT32 ||
                (dGates_half_dtype != pi::tensorlib::DataType::FLOAT16 &&
                 dGates_half_dtype != pi::tensorlib::DataType::BFLOAT16) ||
                dc_prev->dtype() != pi::tensorlib::DataType::FLOAT32)
            {
                throw std::runtime_error("LSTM_CELL_BWD dtype mismatch");
            }

            if (dY->shape().ndims() != 2 || dh_next->shape().ndims() != 2 || dc_next->shape().ndims() != 2 ||
                gate_out->shape().ndims() != 2 || c_prev->shape().ndims() != 2 || c_out->shape().ndims() != 2 ||
                dGates->shape().ndims() != 2 || dGates_half->shape().ndims() != 2 || dc_prev->shape().ndims() != 2)
            {
                throw std::runtime_error("LSTM_CELL_BWD expects 2D tensors");
            }

            const auto batch = dY->shape()[0];
            const auto hidden = dY->shape()[1];
            const auto gate_dim = gate_out->shape()[1];
            if (gate_dim != 4 * hidden)
            {
                throw std::runtime_error("LSTM_CELL_BWD gate_out shape mismatch");
            }
            if (dh_next->shape()[0] != batch || dh_next->shape()[1] != hidden || dc_next->shape()[0] != batch ||
                dc_next->shape()[1] != hidden || c_prev->shape()[0] != batch || c_prev->shape()[1] != hidden ||
                c_out->shape()[0] != batch || c_out->shape()[1] != hidden || dGates->shape()[0] != batch ||
                dGates->shape()[1] != gate_dim || dGates_half->shape()[0] != batch || dGates_half->shape()[1] != gate_dim ||
                dc_prev->shape()[0] != batch || dc_prev->shape()[1] != hidden)
            {
                throw std::runtime_error("LSTM_CELL_BWD shape mismatch");
            }

            if (const char *env = std::getenv("FBAMTRAIN_DEBUG_GEMM_VALUES"); env != nullptr)
            {
                static bool logged = false;
                if (!logged)
                {
                    logged = true;
                    auto dump_half = [](const std::shared_ptr<pi::tensorlib::RealTensor> &t, const char *label) {
                        auto storage = t->storage();
                        if (storage->device().device_type != pi::tensorlib::DeviceType::CPU)
                        {
                            storage = storage->toCPU();
                        }
                        const auto *ptr = static_cast<const uint16_t *>(storage->dataptr());
                        std::clog << "[lstm_bwd] " << label << "[0,0..4]=" << ptr[0] << ", " << ptr[1] << ", " << ptr[2]
                                  << ", " << ptr[3] << ", " << ptr[4] << '\n';
                    };
                    auto dump_f32 = [](const std::shared_ptr<pi::tensorlib::RealTensor> &t, const char *label) {
                        auto storage = t->storage();
                        if (storage->device().device_type != pi::tensorlib::DeviceType::CPU)
                        {
                            storage = storage->toCPU();
                        }
                        const auto *ptr = static_cast<const float *>(storage->dataptr());
                        std::clog << "[lstm_bwd] " << label << "[0,0..4]=" << ptr[0] << ", " << ptr[1] << ", " << ptr[2]
                                  << ", " << ptr[3] << ", " << ptr[4] << '\n';
                    };
                    dump_half(dY, "dY");
                    dump_f32(dh_next, "dh_next");
                    dump_f32(dc_next, "dc_next");
                    dump_f32(gate_out, "gate_out");
                    dump_f32(c_prev, "c_prev");
                    dump_f32(c_out, "c_out");
                }
            }

            const auto device_ordinal = ValidateSameDeviceOrdinal(
                "lstm_cell_bwd_pointwise",
                {dY, dh_next, dc_next, gate_out, c_prev, c_out, dGates, dGates_half, dc_prev});

            const auto gate_stride = gate_out->strides().strides().empty() ? gate_dim : gate_out->strides()[0];
            const auto gate_out_stride = gate_stride; // stored gate_out uses same leading stride
            const auto state_stride = dY->strides().strides().empty() ? hidden : dY->strides()[0];
            const auto c_prev_stride = c_prev->strides().strides().empty() ? hidden : c_prev->strides()[0];
            const auto c_out_stride = c_out->strides().strides().empty() ? hidden : c_out->strides()[0];
            const auto dG_stride = dGates->strides().strides().empty() ? gate_dim : dGates->strides()[0];
            const auto dc_stride = dc_prev->strides().strides().empty() ? hidden : dc_prev->strides()[0];

            const uint32_t total = static_cast<uint32_t>(batch * hidden);
            const uint32_t block_elems = kernel->meta.block_size;

            pi::tensorlib::KernelLaunchArguments args{
                .args = {dY->dataptr(),
                         dh_next->dataptr(),
                         dc_next->dataptr(),
                         gate_out->dataptr(),
                         c_prev->dataptr(),
                         c_out->dataptr(),
                         dGates->dataptr(),
                         dGates_half->dataptr(),
                         dc_prev->dataptr(),
                         static_cast<int32_t>(batch),
                         static_cast<int32_t>(hidden),
                         static_cast<int32_t>(gate_stride),
                         static_cast<int32_t>(gate_out_stride),
                         static_cast<int32_t>(state_stride),
                         static_cast<int32_t>(c_prev_stride),
                         static_cast<int32_t>(c_out_stride),
                         static_cast<int32_t>(dG_stride),
                         static_cast<int32_t>(dc_stride), static_cast<void *>(nullptr)},
                .grid_dim_x = (total + block_elems - 1) / block_elems,
                .grid_dim_y = 1,
                .grid_dim_z = 1,
                .block_dim_x = TRITON_WARP_SIZE * kernel->num_warps,
                .block_dim_y = 1,
                .block_dim_z = 1,
                .shared_mem_bytes = kernel->shared_mem_bytes,
                .device_ordinal = static_cast<int>(device_ordinal)};
            return args;
        },
        .cuda_descriptor = pi::tensorlib::MakeCudaKernelDescriptor(kernel->data, kernel->size),
        .hip_descriptor = pi::tensorlib::MakeHipKernelDescriptor(kernel->data, kernel->size)};
}
} // namespace

void LstmCellBwdImplPass::transform(pi::tensorlib::ExecutionPlan &execution_plan)
{
    static const LstmBwdKernelConfig recompute_fp16_config{&klstm_cell_recompute_out_fp16,
                                                           "lstm_cell_recompute_out_fp16"};
    static const LstmBwdKernelConfig recompute_bf16_config{&klstm_cell_recompute_out_bf16,
                                                           "lstm_cell_recompute_out_bf16"};
    static const LstmBwdKernelConfig bwd_fp16_config{&klstm_cell_bwd_pointwise_out_fp16,
                                                     "lstm_cell_bwd_pointwise_out_fp16"};
    static const LstmBwdKernelConfig bwd_bf16_config{&klstm_cell_bwd_pointwise_out_bf16,
                                                     "lstm_cell_bwd_pointwise_out_bf16"};

    static const auto recompute_fp16_descriptor = CreateRecomputeKernelDescriptor(recompute_fp16_config);
    static const auto recompute_bf16_descriptor = CreateRecomputeKernelDescriptor(recompute_bf16_config);
    static const auto bwd_fp16_descriptor = CreateBwdPointwiseKernelDescriptor(bwd_fp16_config);
    static const auto bwd_bf16_descriptor = CreateBwdPointwiseKernelDescriptor(bwd_bf16_config);

    for (auto &entry : execution_plan.entries)
    {
        if (!entry.op_type.has_value())
        {
            continue;
        }
        if (entry.op_type == pi::tensorlib::OpType::LSTM_CELL_RECOMPUTE)
        {
            if (entry.outputs.size() != 3)
            {
                throw std::runtime_error("LSTM_CELL_RECOMPUTE expects three outputs");
            }
            const auto &h_out = entry.outputs[1];
            const bool out_fp16 = h_out->dtype() == pi::tensorlib::DataType::FLOAT16;
            const bool out_bf16 = h_out->dtype() == pi::tensorlib::DataType::BFLOAT16;
            if (!out_fp16 && !out_bf16)
            {
                throw std::runtime_error("Unsupported dtype for LSTM_CELL_RECOMPUTE h_out");
            }
            entry.op_type = std::nullopt;
            entry.kernel_descriptor = out_bf16 ? recompute_bf16_descriptor : recompute_fp16_descriptor;
            const uint64_t gate_dim = entry.inputs.empty() ? 0 : entry.inputs[0]->shape()[1];
            const uint64_t hidden = gate_dim / 4;
            const uint64_t batch = entry.inputs.empty() ? 0 : entry.inputs[0]->shape()[0];
            // Matches forward cell: bias adds, three sigmoids, g gate (exp/div), cell update, tanh, output mul.
            entry.flop_estimate = batch * hidden * 30;
        }
        else if (entry.op_type == pi::tensorlib::OpType::LSTM_CELL_BWD)
        {
            if (entry.outputs.size() != 3)
            {
                throw std::runtime_error("LSTM_CELL_BWD expects three outputs");
            }
            const auto &dGates_half = entry.outputs[1];
            const bool out_fp16 = dGates_half->dtype() == pi::tensorlib::DataType::FLOAT16;
            const bool out_bf16 = dGates_half->dtype() == pi::tensorlib::DataType::BFLOAT16;
            if (!out_fp16 && !out_bf16)
            {
                throw std::runtime_error("Unsupported dtype for LSTM_CELL_BWD dGates_half");
            }
            entry.op_type = std::nullopt;
            entry.kernel_descriptor = out_bf16 ? bwd_bf16_descriptor : bwd_fp16_descriptor;
            const uint64_t hidden = entry.inputs.empty() ? 0 : entry.inputs[0]->shape()[1];
            const uint64_t batch = entry.inputs.empty() ? 0 : entry.inputs[0]->shape()[0];
            // Pointwise backward: tanh/sigmoid grads and gate products ≈ 28 flops per element.
            entry.flop_estimate = batch * hidden * 28;
        }
    }
}
