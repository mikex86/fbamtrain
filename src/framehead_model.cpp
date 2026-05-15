#include "framehead_model.h"

#include <attention.h>
#include <embedding.h>
#include <mlp.h>
#include <op_graph.h>
#include <rms_norm.h>

#include "config.h"
#include "executor.h"
#include "frame_utils.h"
#include "functional.h"

#include <array>
#include <cstdlib>
#include <memory>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace pi::tensorlib;

namespace fbamtrain
{
    enum class FrameHeadConvMode
    {
        Dilated,
        GeGLU,
    };

    namespace
    {
        FrameHeadConvMode ParseConvMode(const std::string &mode)
        {
            if (mode == "dilated")
            {
                return FrameHeadConvMode::Dilated;
            }
            if (mode == "geglu")
            {
                return FrameHeadConvMode::GeGLU;
            }
            throw std::invalid_argument("Unsupported downsample_conv_mode: " + mode +
                                        " (expected 'dilated' or 'geglu')");
        }

        uint64_t MeanReductionSize(const FbamModelConfiguration &config)
        {
            uint32_t rows = config.frame_rows;
            uint32_t cols = config.frame_cols;
            for (uint32_t i = 0; i < config.downsample_blocks; ++i)
            {
                rows /= 2U;
                cols /= 2U;
            }
            return static_cast<uint64_t>(rows) * static_cast<uint64_t>(cols);
        }

    } // namespace

    class FrameHeadConvBlock final : public Module<>
    {
        std::shared_ptr<RmsNorm> rms_norm_;
        TraceTensor weight_;
        FrameHeadConvMode mode_;
        Device device_;
        DataType dtype_;
        uint32_t embed_dim_;
        uint32_t rows_;
        uint32_t cols_;
        uint32_t dilation_;
        bool use_fp16_conv_acc_;
        GpuStreamDescriptor compute_stream_descriptor_;

      public:
        FrameHeadConvBlock(const std::string &name, const uint32_t embed_dim, const uint32_t rows, const uint32_t cols,
                           const FrameHeadConvMode mode, const uint32_t dilation, const float rms_eps,
                           const Device device, const DataType dtype, OpGraph &graph, uint32_t &init_seed,
                           const GpuStreamDescriptor &compute_stream_descriptor, const bool use_fp16_conv_acc = false)
            : Module(name),
              weight_(graph.createTensor({mode == FrameHeadConvMode::GeGLU ? 2 * embed_dim : embed_dim, 3, 3, embed_dim},
                                         dtype, device, compute_stream_descriptor, false)),
              mode_(mode), device_(device), dtype_(dtype), embed_dim_(embed_dim), rows_(rows), cols_(cols),
              dilation_(dilation), use_fp16_conv_acc_(use_fp16_conv_acc), compute_stream_descriptor_(compute_stream_descriptor)
        {
            if (rows_ == 0 || cols_ == 0)
            {
                throw std::invalid_argument("FrameHeadConvBlock requires positive spatial dimensions");
            }
            if (dilation_ == 0)
            {
                throw std::invalid_argument("FrameHeadConvBlock dilation must be greater than zero");
            }

            rms_norm_ = std::make_shared<RmsNorm>(name + ".ln", embed_dim_, device_, dtype_, rms_eps, graph,
                                                  false, compute_stream_descriptor_);

            const uint32_t fan_in = embed_dim_ * 3U * 3U;
            KaimingUniformInit(graph, weight_, fan_in, init_seed++, compute_stream_descriptor_);
        }

        [[nodiscard]] TraceTensor buildForward(OpGraph &graph, const std::initializer_list<TraceTensor> inputs,
                                               const bool save_input_for_backward) override
        {
            bw_context_.clear();
            const auto &input = inputs.begin()[0];
            const auto &input_shape = input.shape();
            if (input_shape.ndims() != 3)
            {
                throw std::invalid_argument("FrameHeadConvBlock expects input shaped (B, T, C)");
            }
            const uint64_t batch = input_shape[0];
            const uint64_t seq_len = input_shape[1];
            if (seq_len != rows_ * cols_)
            {
                throw std::invalid_argument("FrameHeadConvBlock: sequence length does not match rows*cols");
            }
            if (input_shape[2] != embed_dim_)
            {
                throw std::invalid_argument("FrameHeadConvBlock: embedding dimension mismatch");
            }

            TraceTensor norm = rms_norm_->buildForward(graph, {input}, save_input_for_backward);
            if (save_input_for_backward)
            {
                bw_context_.saveForBackward("norm", norm);
            }
            TraceTensor nhwc = norm.view(graph, {batch, rows_, cols_, embed_dim_});

            constexpr std::array<uint32_t, 2> stride{1, 1};
            const std::array<uint32_t, 2> dilation =
                mode_ == FrameHeadConvMode::Dilated ? std::array{dilation_, dilation_} : std::array<uint32_t, 2>{1, 1};
            const std::array<uint32_t, 2> padding =
                mode_ == FrameHeadConvMode::Dilated ? std::array{dilation_, dilation_} : std::array<uint32_t, 2>{1, 1};

            TraceTensor conv_out =
                Conv2d(graph, nhwc, weight_, nullptr, stride, padding, compute_stream_descriptor_, dilation,
                       use_fp16_conv_acc_);
            if (save_input_for_backward)
            {
                bw_context_.saveForBackward("conv_out_pre_act", conv_out);
            }
            if (mode_ == FrameHeadConvMode::Dilated)
            {
                conv_out = Gelu(graph, conv_out, compute_stream_descriptor_);
            }
            else
            {
                auto splits = conv_out.split(graph, 2, 3);
                TraceTensor value = splits.at(0);
                TraceTensor gate = splits.at(1);
                TraceTensor activated_gate = Gelu(graph, gate, compute_stream_descriptor_);
                if (save_input_for_backward)
                {
                    TraceTensor activated = graph.createTensor(value.shape().dims(), value.dtype(), value.device(),
                                                               compute_stream_descriptor_, false);
                    graph.recordOperation(OperationEntry{.type = OpType::MUL,
                                                         .inputs = {value, activated_gate},
                                                         .outputs = {activated},
                                                         .attributes = {},
                                                         .gpu_stream_desc = compute_stream_descriptor_});
                    conv_out = activated;
                }
                else
                {
                    InplaceMul(graph, value, activated_gate, compute_stream_descriptor_);
                    conv_out = value;
                }
            }

            TraceTensor activated_nhwc = conv_out;
            return activated_nhwc.view(graph, {batch, seq_len, embed_dim_});
        }

        [[nodiscard]] std::vector<ParameterEntry> parameters() const override
        {
            std::vector<ParameterEntry> params{};
            const auto norm_params = rms_norm_->parameters();
            params.reserve(norm_params.size());
            for (const auto &entry : norm_params)
            {
                params.emplace_back(entry);
            }
            params.emplace_back(ParameterEntry{.name = name_ + ".conv.weight", .tensor = weight_});
            return params;
        }

        void buildBackward(OpGraph &graph, const TraceTensor &upstream_grad,
                           const std::unordered_map<std::string, TraceTensor> &parameter_gradients,
                           const std::unordered_map<std::string, TraceTensor> &operand_gradients) override
        {
            if (!bw_context_.hasSaved("norm") || !bw_context_.hasSaved("conv_out_pre_act"))
            {
                throw std::runtime_error(
                    "FrameHeadConvBlock backward requested but required tensors were not retained");
            }

            const TraceTensor &norm = bw_context_.getSaved("norm");
            const TraceTensor &conv_out_pre_act = bw_context_.getSaved("conv_out_pre_act");

            if (upstream_grad.shape().ndims() != 3)
            {
                throw std::runtime_error("FrameHeadConvBlock backward expects upstream shaped (B, T, C)");
            }

            const uint64_t batch = upstream_grad.shape()[0];
            const uint64_t seq_len = upstream_grad.shape()[1];
            const uint64_t embed_dim = upstream_grad.shape()[2];
            if (seq_len != rows_ * cols_)
            {
                throw std::runtime_error("FrameHeadConvBlock backward sequence length mismatch");
            }
            if (embed_dim != embed_dim_)
            {
                throw std::runtime_error("FrameHeadConvBlock backward embedding dimension mismatch");
            }
            if (upstream_grad.dtype() != dtype_)
            {
                throw std::runtime_error("FrameHeadConvBlock backward dtype mismatch");
            }
            if (upstream_grad.device() != device_)
            {
                throw std::runtime_error("FrameHeadConvBlock backward device mismatch");
            }

            TraceTensor upstream_nhwc = upstream_grad.view(graph, {batch, rows_, cols_, embed_dim_});

            TraceTensor conv_grad = [&]()
            {
                if (mode_ == FrameHeadConvMode::Dilated)
                {
                    return GeluBackward(graph, conv_out_pre_act, upstream_nhwc, compute_stream_descriptor_);
                }

                auto splits = conv_out_pre_act.split(graph, 2, 3);
                const TraceTensor &value = splits.at(0);
                const TraceTensor &gate = splits.at(1);

                TraceTensor activated_gate = Gelu(graph, gate, compute_stream_descriptor_);
                TraceTensor grad_value = graph.createTensor(upstream_nhwc.shape().dims(), upstream_grad.dtype(),
                                                            upstream_grad.device(), compute_stream_descriptor_, false);
                graph.recordOperation(OperationEntry{.type = OpType::MUL,
                                                     .inputs = {upstream_nhwc, activated_gate},
                                                     .outputs = {grad_value},
                                                     .attributes = {},
                                                     .gpu_stream_desc = compute_stream_descriptor_});

                TraceTensor gate_upstream = graph.createTensor(upstream_nhwc.shape().dims(), upstream_grad.dtype(),
                                                               upstream_grad.device(), compute_stream_descriptor_, false);
                graph.recordOperation(OperationEntry{.type = OpType::MUL,
                                                     .inputs = {upstream_nhwc, value},
                                                     .outputs = {gate_upstream},
                                                     .attributes = {},
                                                     .gpu_stream_desc = compute_stream_descriptor_});
                TraceTensor grad_gate = GeluBackward(graph, gate, gate_upstream, compute_stream_descriptor_);

                TraceTensor conv_grad_local = graph.createTensor(conv_out_pre_act.shape().dims(), upstream_grad.dtype(),
                                                                 upstream_grad.device(), compute_stream_descriptor_, false);
                auto grad_splits = conv_grad_local.split(graph, 2, 3);
                DeviceCopy(graph, grad_value, grad_splits.at(0), compute_stream_descriptor_);
                DeviceCopy(graph, grad_gate, grad_splits.at(1), compute_stream_descriptor_);

                graph.deleteTensor(grad_value);
                graph.deleteTensor(gate_upstream);
                graph.deleteTensor(grad_gate);

                return conv_grad_local;
            }();

            TraceTensor norm_nhwc = norm.view(graph, {batch, rows_, cols_, embed_dim_});

            constexpr std::array<uint32_t, 2> stride{1, 1};
            const std::array<uint32_t, 2> dilation =
                mode_ == FrameHeadConvMode::Dilated ? std::array{dilation_, dilation_} : std::array<uint32_t, 2>{1, 1};
            const std::array<uint32_t, 2> padding =
                mode_ == FrameHeadConvMode::Dilated ? std::array{dilation_, dilation_} : std::array<uint32_t, 2>{1, 1};

            if (auto weight_it = parameter_gradients.find(name_ + ".conv.weight");
                weight_it != parameter_gradients.end())
            {
                Conv2dWgradInto(graph, norm_nhwc, conv_grad, weight_it->second, stride, padding, dilation,
                                compute_stream_descriptor_, /*accumulate_output=*/true);
            }

            TraceTensor grad_norm_nhwc =
                graph.createTensor(norm_nhwc.shape().dims(), upstream_grad.dtype(), upstream_grad.device(),
                                   compute_stream_descriptor_, false);
            Conv2dDgradInto(graph, conv_grad, weight_, grad_norm_nhwc, stride, padding, dilation,
                            compute_stream_descriptor_);

            TraceTensor grad_norm = grad_norm_nhwc.view(graph, {batch, seq_len, embed_dim_});

            std::unordered_map<std::string, TraceTensor> rms_operand_grads{};
            if (auto input_it = operand_gradients.find("input"); input_it != operand_gradients.end())
            {
                rms_operand_grads.emplace("input", input_it->second);
            }

            rms_norm_->buildBackward(graph, grad_norm, parameter_gradients, rms_operand_grads);

            graph.deleteTensor(grad_norm_nhwc);
            graph.deleteTensor(conv_grad);
            bw_context_.release(graph);
        }
    };

    class FrameHeadAttentionBlock final : public Module<>
    {
        std::shared_ptr<RmsNorm> rms_norm_1_;
        std::shared_ptr<FullMhaAttention> attention_;
        std::shared_ptr<RmsNorm> rms_norm_2_;
        std::shared_ptr<MlpBlock> mlp_;
        Device device_;
        DataType dtype_;
        GpuStreamDescriptor compute_stream_descriptor_;

      public:
        FrameHeadAttentionBlock(const std::string &name, const uint32_t embed_dim, const uint32_t num_heads,
                                const uint32_t hidden_dim, const float rms_eps, const Device device,
                                const DataType dtype, OpGraph &graph, uint32_t &init_seed, const bool has_bias,
                                const GpuStreamDescriptor &compute_stream_descriptor,
                                const bool use_fp16_flash_attn_acc = false, const bool use_fp16_matmul_acc = false)
            : Module(name), device_(device), dtype_(dtype), compute_stream_descriptor_(compute_stream_descriptor)
        {
            rms_norm_1_ = std::make_shared<RmsNorm>(name + ".ln1", embed_dim, device_, dtype_, rms_eps, graph, false, compute_stream_descriptor_);
            attention_ =
                std::make_shared<FullMhaAttention>(name + ".attn", embed_dim, num_heads, device_, dtype_, graph,
                                                   init_seed, compute_stream_descriptor_, has_bias, std::nullopt,
                                                   use_fp16_flash_attn_acc, use_fp16_matmul_acc,
                                                   use_fp16_matmul_acc);
            rms_norm_2_ = std::make_shared<RmsNorm>(name + ".ln2", embed_dim, device_, dtype_, rms_eps, graph, false, compute_stream_descriptor_);
            mlp_ = std::make_shared<MlpBlock>(name + ".mlp", embed_dim, hidden_dim, device_, dtype_, graph, init_seed,
                                              compute_stream_descriptor_, use_fp16_matmul_acc);
        }

        [[nodiscard]] TraceTensor buildForward(OpGraph &graph, const std::initializer_list<TraceTensor> inputs,
                                               const bool save_input_for_backward) override
        {
            bw_context_.clear();
            const auto &input = inputs.begin()[0];
            TraceTensor norm1 = rms_norm_1_->buildForward(graph, {input}, save_input_for_backward);

            TraceTensor attn_out = attention_->buildForward(graph, {norm1}, save_input_for_backward);

            const auto &shape = input.shape();
            const uint64_t batch = shape[0];
            const uint64_t seq_len = shape[1];
            const uint64_t embed_dim = shape[2];
            const uint64_t flattened_tokens = batch * seq_len;

            TraceTensor input_tokens = input.view(graph, {flattened_tokens, embed_dim});
            TraceTensor attn_flat = attn_out.view(graph, {flattened_tokens, embed_dim});

            TraceTensor residual_tokens = input_tokens;
            if (save_input_for_backward)
            {
                residual_tokens = graph.createTensor(input_tokens.shape().dims(), input_tokens.dtype(),
                                                     input_tokens.device(), compute_stream_descriptor_, false);
                graph.recordOperation(OperationEntry{.type = OpType::PLUS,
                                                     .inputs = {input_tokens, attn_flat},
                                                     .outputs = {residual_tokens},
                                                     .attributes = {},
                                                     .gpu_stream_desc = compute_stream_descriptor_});
            }
            else
            {
                InplaceAdd(graph, residual_tokens, attn_flat, compute_stream_descriptor_);
            }

            TraceTensor residual_after_attn = residual_tokens.view(graph, {batch, seq_len, embed_dim});

            TraceTensor norm2 = rms_norm_2_->buildForward(graph, {residual_after_attn}, save_input_for_backward);
            TraceTensor mlp_out = mlp_->buildForward(graph, {norm2}, save_input_for_backward);

            TraceTensor residual_flat = residual_after_attn.view(graph, {flattened_tokens, embed_dim});
            TraceTensor mlp_flat = mlp_out.view(graph, {flattened_tokens, embed_dim});

            TraceTensor output_tokens = residual_flat;
            if (save_input_for_backward)
            {
                output_tokens = graph.createTensor(residual_flat.shape().dims(), residual_flat.dtype(),
                                                   residual_flat.device(), compute_stream_descriptor_, false);
                graph.recordOperation(OperationEntry{.type = OpType::PLUS,
                                                     .inputs = {residual_flat, mlp_flat},
                                                     .outputs = {output_tokens},
                                                     .attributes = {},
                                                     .gpu_stream_desc = compute_stream_descriptor_});
            }
            else
            {
                InplaceAdd(graph, output_tokens, mlp_flat, compute_stream_descriptor_);
            }

            return output_tokens.view(graph, {batch, seq_len, embed_dim});
        }

        void buildBackward(OpGraph &graph, const TraceTensor &upstream_grad,
                           const std::unordered_map<std::string, TraceTensor> &parameter_gradients,
                           const std::unordered_map<std::string, TraceTensor> &operand_gradients) override
        {
            if (upstream_grad.shape().ndims() != 3)
            {
                throw std::runtime_error("FrameHeadAttentionBlock backward expects upstream shaped (B, T, C)");
            }
            const uint64_t batch = upstream_grad.shape()[0];
            const uint64_t seq_len = upstream_grad.shape()[1];
            const uint64_t embed_dim = upstream_grad.shape()[2];

            TraceTensor mlp_upstream_flat = upstream_grad.view(graph, {batch * seq_len, embed_dim});
            TraceTensor norm2_grad_flat = graph.createTensor(mlp_upstream_flat.shape().dims(), upstream_grad.dtype(),
                                                             upstream_grad.device(), compute_stream_descriptor_, false);

            std::unordered_map<std::string, TraceTensor> mlp_operand_grads{};
            mlp_operand_grads.emplace("input", norm2_grad_flat);
            mlp_->buildBackward(graph, mlp_upstream_flat, parameter_gradients, mlp_operand_grads);

            TraceTensor norm2_grad = norm2_grad_flat.view(graph, {batch, seq_len, embed_dim});

            TraceTensor residual_grad = graph.createTensor(upstream_grad.shape().dims(), upstream_grad.dtype(),
                                                           upstream_grad.device(), compute_stream_descriptor_, false);

            std::unordered_map<std::string, TraceTensor> norm2_operand_grads{};
            norm2_operand_grads.emplace("input", residual_grad);
            rms_norm_2_->buildBackward(graph, norm2_grad, parameter_gradients, norm2_operand_grads);

            InplaceAdd(graph, residual_grad, upstream_grad, compute_stream_descriptor_);

            TraceTensor norm1_grad = graph.createTensor(upstream_grad.shape().dims(), upstream_grad.dtype(),
                                                        upstream_grad.device(), compute_stream_descriptor_, false);
            std::unordered_map<std::string, TraceTensor> attn_operand_grads{};
            attn_operand_grads.emplace("input", norm1_grad);
            attention_->buildBackward(graph, residual_grad, parameter_gradients, attn_operand_grads);

            std::unordered_map<std::string, TraceTensor> norm1_operand_grads{};
            std::optional<TraceTensor> input_grad_target;
            if (auto input_it = operand_gradients.find("input"); input_it != operand_gradients.end())
            {
                input_grad_target = input_it->second;
                norm1_operand_grads.emplace("input", *input_grad_target);
            }
            rms_norm_1_->buildBackward(graph, norm1_grad, parameter_gradients, norm1_operand_grads);

            if (input_grad_target.has_value())
            {
                InplaceAdd(graph, *input_grad_target, residual_grad, compute_stream_descriptor_);
            }

            graph.deleteTensor(norm2_grad_flat);
            graph.deleteTensor(norm1_grad);
            graph.deleteTensor(residual_grad);
        }

        [[nodiscard]] std::vector<ParameterEntry> parameters() const override
        {
            std::vector<ParameterEntry> params{};
            const auto rms1_params = rms_norm_1_->parameters();
            const auto attn_params = attention_->parameters();
            const auto rms2_params = rms_norm_2_->parameters();
            const auto mlp_params = mlp_->parameters();
            params.reserve(rms1_params.size() + attn_params.size() + rms2_params.size() + mlp_params.size());
            for (const auto &entry : rms1_params)
            {
                params.emplace_back(entry);
            }
            for (const auto &entry : attn_params)
            {
                params.emplace_back(entry);
            }
            for (const auto &entry : rms2_params)
            {
                params.emplace_back(entry);
            }
            for (const auto &entry : mlp_params)
            {
                params.emplace_back(entry);
            }
            return params;
        }
    };

    FrameHeadModule::FrameHeadModule(const FbamModelConfiguration &config, OpGraph &graph, const Device device,
                                     const DataType dtype, const GpuStreamDescriptor &compute_stream_descriptor)
        : Module("frame_head"), rows_(config.frame_rows), cols_(config.frame_cols), vocab_size_(config.max_code_point),
          config_(config), device_(device), dtype_(dtype), compute_stream_descriptor_(compute_stream_descriptor),
          reduction_strategy_(config.frame_head_reduction_strategy),
          mean_(name_ + ".mean", /*dim=*/1, /*keepdim=*/false, MeanReductionSize(config), graph, dtype, device,
                compute_stream_descriptor)
    {
        if (config_.n_embed == 0)
        {
            throw std::invalid_argument("FrameHeadModule requires a non-zero embedding dimension");
        }
        if (config_.n_head == 0)
        {
            throw std::invalid_argument("FrameHeadModule requires at least one attention head");
        }
        if (config_.n_embed % config_.n_head != 0U)
        {
            throw std::invalid_argument("FrameHeadModule embedding dimension must be divisible by number of heads");
        }
        if (dtype_ != DataType::BFLOAT16 && dtype_ != DataType::FLOAT16)
        {
            throw std::invalid_argument("FrameHeadModule currently supports BFLOAT16 or FLOAT16 inputs");
        }
        if (config_.max_code_point == 0)
        {
            throw std::invalid_argument("FrameHeadModule requires a non-zero maximum code point");
        }
        if (config_.downsample_conv_dilation == 0)
        {
            throw std::invalid_argument("FrameHeadModule requires downsample_conv_dilation greater than zero");
        }

        switch (reduction_strategy_)
        {
            case FrameHeadReductionStrategy::Mean:
                break;
            case FrameHeadReductionStrategy::LastPos:
                break;
            default:
                throw std::invalid_argument("Unsupported frame_head_reduction_strategy enum value.");
        }

        const uint64_t positions = rows_ * cols_;
        auto init_seed = static_cast<uint32_t>(config_.model_init_seed);

        position_embedding_ = std::make_shared<Embedding>(name_ + ".position_embedding", positions, config_.n_embed,
                                                          device_, dtype_, graph, init_seed, compute_stream_descriptor_);
        const auto position_params = position_embedding_->parameters();
        if (position_params.size() != 1)
        {
            throw std::runtime_error("Position embedding expected a single parameter tensor");
        }
        position_embed_.emplace(position_params[0].tensor);

        codepoint_embed_.emplace(
            graph.createTensor({vocab_size_, config_.n_embed}, dtype_, device_, compute_stream_descriptor_, false));
        fg_r_embed_.emplace(graph.createTensor({config_.n_embed}, dtype_, device_, compute_stream_descriptor_, false));
        fg_g_embed_.emplace(graph.createTensor({config_.n_embed}, dtype_, device_, compute_stream_descriptor_, false));
        fg_b_embed_.emplace(graph.createTensor({config_.n_embed}, dtype_, device_, compute_stream_descriptor_, false));
        bg_r_embed_.emplace(graph.createTensor({config_.n_embed}, dtype_, device_, compute_stream_descriptor_, false));
        bg_g_embed_.emplace(graph.createTensor({config_.n_embed}, dtype_, device_, compute_stream_descriptor_, false));
        bg_b_embed_.emplace(graph.createTensor({config_.n_embed}, dtype_, device_, compute_stream_descriptor_, false));

        FillNormal(graph, *codepoint_embed_, 0.0f, 1.0f, init_seed++, compute_stream_descriptor_);
        FillNormal(graph, *fg_r_embed_, 0.0f, 1.0f, init_seed++, compute_stream_descriptor_);
        FillNormal(graph, *fg_g_embed_, 0.0f, 1.0f, init_seed++, compute_stream_descriptor_);
        FillNormal(graph, *fg_b_embed_, 0.0f, 1.0f, init_seed++, compute_stream_descriptor_);
        FillNormal(graph, *bg_r_embed_, 0.0f, 1.0f, init_seed++, compute_stream_descriptor_);
        FillNormal(graph, *bg_g_embed_, 0.0f, 1.0f, init_seed++, compute_stream_descriptor_);
        FillNormal(graph, *bg_b_embed_, 0.0f, 1.0f, init_seed++, compute_stream_descriptor_);

        const uint32_t hidden_dim = 4U * config_.n_embed;

        const auto conv_mode = ParseConvMode(config_.downsample_conv_mode);

        uint32_t validate_rows = rows_;
        uint32_t validate_cols = cols_;
        for (uint32_t index = 0; index < config_.downsample_blocks; ++index)
        {
            if (validate_rows % 2U != 0 || validate_cols % 2U != 0)
            {
                throw std::invalid_argument(
                    "FrameHeadModule requires frame dimensions divisible by 2^downsample_blocks");
            }
            validate_rows /= 2U;
            validate_cols /= 2U;
            if (validate_rows == 0 || validate_cols == 0)
            {
                throw std::invalid_argument("FrameHeadModule downsampling reduces spatial dimensions to zero");
            }
        }

        downsample_blocks_.reserve(config_.downsample_blocks);
        uint32_t current_rows = rows_;
        uint32_t current_cols = cols_;
        for (uint32_t index = 0; index < config_.downsample_blocks; ++index)
        {
            downsample_blocks_.push_back(std::make_shared<FrameHeadConvBlock>(
                name_ + ".downsample_block." + std::to_string(index), config_.n_embed, current_rows, current_cols,
                conv_mode, config_.downsample_conv_dilation, config_.rms_norm_eps, device_, dtype_, graph, init_seed,
                compute_stream_descriptor_, config.use_fp16_accumulation));
            current_rows /= 2U;
            current_cols /= 2U;
        }

        blocks_.reserve(config_.n_layer);
        for (uint32_t index = 0; index < config_.n_layer; ++index)
        {
            blocks_.push_back(std::make_shared<FrameHeadAttentionBlock>(
                name_ + ".block." + std::to_string(index), config_.n_embed, config_.n_head, hidden_dim,
                config_.rms_norm_eps, device_, dtype_, graph, init_seed, config_.bias, compute_stream_descriptor_,
                /*use_fp16_flash_attn_acc*/ config_.use_fp16_accumulation,
                /*use_fp16_matmul_acc*/ config_.use_fp16_accumulation));
        }
    }

    TraceTensor FrameHeadModule::buildForward(OpGraph &graph, std::initializer_list<TraceTensor> inputs,
                                              bool save_input_for_backward)
    {
        OpGraphGpuTxRange range = graph.createGpuTxRange("FrameHeadModule::buildForward");

        bw_context_.clear();
        const auto &input = inputs.begin()[0];

        if (input.device() != device_)
        {
            throw std::runtime_error("FrameHeadModule input device must match module device");
        }

        TraceTensor x = input;
        uint64_t batch = 0;
        uint64_t embed_dim = config_.n_embed;
        const uint64_t frame_tokens = static_cast<uint64_t>(rows_) * static_cast<uint64_t>(cols_);

        if (input.dtype() == DataType::UINT32)
        {
            if (input.shape().ndims() != 4)
            {
                throw std::invalid_argument("FrameHeadModule expects cell_states shaped (B, H, W, C)");
            }
            if (input.shape()[1] != rows_ || input.shape()[2] != cols_)
            {
                throw std::invalid_argument("FrameHeadModule cell_states spatial dimension mismatch");
            }
            if (input.shape()[3] != NUM_FRAME_CHANNELS)
            {
                throw std::invalid_argument("FrameHeadModule cell_states channel count mismatch");
            }

            batch = input.shape()[0];

            if (save_input_for_backward)
            {
                TraceTensor cell_states = input;
                bw_context_.saveForBackward("cell_states", cell_states);
            }

            TraceTensor cell_embeddings =
                graph.createTensor({batch, rows_, cols_, embed_dim}, dtype_, device_, compute_stream_descriptor_, false);
            TraceTensor cell_embeddings_flat =
                cell_embeddings.viewInferred(graph, {-1l, static_cast<int64_t>(embed_dim)});
            TraceTensor cell_states_flat = input.viewInferred(graph, {-1l, NUM_FRAME_CHANNELS});

            graph.recordOperation(OperationEntry{
                .type = OpType::CUSTOM_OP,
                .inputs = {cell_embeddings_flat, cell_states_flat, codepointEmbedding(), positionEmbedding(),
                           fgREmbedding(), fgGEmbedding(), fgBEmbedding(), bgREmbedding(), bgGEmbedding(),
                           bgBEmbedding()},
                .outputs = {cell_embeddings_flat},
                .attributes = {{"custom_op_name", "build_cell_embed"}},
                .gpu_stream_desc = compute_stream_descriptor_,
            });

            x = cell_embeddings.view(graph, {batch, frame_tokens, embed_dim});
        }
        else
        {
            if (input.dtype() != dtype_)
            {
                throw std::runtime_error("FrameHeadModule input dtype must match module dtype");
            }
            if (input.shape().ndims() != 3)
            {
                throw std::invalid_argument("FrameHeadModule expects input shaped (B, T, C)");
            }
            if (input.shape()[1] != frame_tokens)
            {
                throw std::invalid_argument("FrameHeadModule input sequence length mismatch");
            }
            if (input.shape()[2] != config_.n_embed)
            {
                throw std::invalid_argument("FrameHeadModule input embedding dimension mismatch");
            }

            x = input;
            batch = input.shape()[0];
            embed_dim = input.shape()[2];
        }

        uint32_t current_rows = rows_;
        uint32_t current_cols = cols_;
        for (const auto &block : downsample_blocks_)
        {
            TraceTensor prev_x = x;
            x = block->buildForward(graph, {x}, save_input_for_backward);
            if (current_rows % 2U != 0 || current_cols % 2U != 0)
            {
                throw std::runtime_error("FrameHeadModule encountered odd spatial dimension during pooling");
            }

            const uint32_t next_rows = current_rows / 2U;
            const uint32_t next_cols = current_cols / 2U;

            TraceTensor nhwc = x.view(graph, {batch, current_rows, current_cols, embed_dim});
            TraceTensor pooled =
                AvgPool2d(graph, nhwc, std::array<uint32_t, 2>{2, 2}, std::array<uint32_t, 2>{2, 2},
                          compute_stream_descriptor_, std::array<uint32_t, 2>{0, 0}, true);

            const uint32_t flattened = next_rows * next_cols;
            x = pooled.view(graph, {batch, flattened, embed_dim});

            current_rows = next_rows;
            current_cols = next_cols;

            if (!save_input_for_backward)
            {
                graph.deleteTensor(prev_x); // delete previous tensor to save memory
            }
        }

        for (const auto &block : blocks_)
        {
            x = block->buildForward(graph, {x}, save_input_for_backward);
        }

        if (reduction_strategy_ == FrameHeadReductionStrategy::Mean)
        {
            TraceTensor output = mean_.buildForward(graph, {x}, save_input_for_backward);
            return output;
        }
        else
        {
            const uint64_t last_index = MeanReductionSize(config_) - 1U;
            TraceTensor output = x.at(graph, /*dim=*/1, last_index);
            return output;
        }
    }

    std::vector<ParameterEntry> FrameHeadModule::parameters() const
    {
        std::vector<ParameterEntry> params{};
        params.reserve(7);
        params.emplace_back(ParameterEntry{.name = name_ + ".codepoint_embedding", .tensor = codepointEmbedding()});
        params.emplace_back(ParameterEntry{.name = name_ + ".fg_r_embed", .tensor = fgREmbedding()});
        params.emplace_back(ParameterEntry{.name = name_ + ".fg_g_embed", .tensor = fgGEmbedding()});
        params.emplace_back(ParameterEntry{.name = name_ + ".fg_b_embed", .tensor = fgBEmbedding()});
        params.emplace_back(ParameterEntry{.name = name_ + ".bg_r_embed", .tensor = bgREmbedding()});
        params.emplace_back(ParameterEntry{.name = name_ + ".bg_g_embed", .tensor = bgGEmbedding()});
        params.emplace_back(ParameterEntry{.name = name_ + ".bg_b_embed", .tensor = bgBEmbedding()});

        const auto position_params = position_embedding_->parameters();
        params.reserve(params.size() + position_params.size());
        for (const auto &entry : position_params)
        {
            params.emplace_back(entry);
        }

        for (const auto &block : downsample_blocks_)
        {
            const auto block_params = block->parameters();
            params.reserve(params.size() + block_params.size());
            for (const auto &entry : block_params)
            {
                params.emplace_back(entry);
            }
        }
        for (const auto &block : blocks_)
        {
            const auto block_params = block->parameters();
            params.reserve(params.size() + block_params.size());
            for (const auto &entry : block_params)
            {
                params.emplace_back(entry);
            }
        }
        const auto mean_params = mean_.parameters();
        params.reserve(params.size() + mean_params.size());
        for (const auto &entry : mean_params)
        {
            params.emplace_back(entry);
        }

        return params;
    }

    void FrameHeadModule::buildBackward(OpGraph &graph, const TraceTensor &backward_input,
                                        const std::unordered_map<std::string, TraceTensor> &parameter_gradients,
                                        const std::unordered_map<std::string, TraceTensor> &operand_gradients)
    {
        OpGraphGpuTxRange range = graph.createGpuTxRange("FrameHeadModule::buildBackward");

        const uint64_t tokens_per_frame = MeanReductionSize(config_);

        const auto &upstream_shape = backward_input.shape();
        if (upstream_shape.ndims() != 2)
        {
            throw std::runtime_error("FrameHead backward expects upstream grad shaped (B, C)");
        }
        const uint64_t batch = upstream_shape[0];
        const uint64_t embed = upstream_shape[1];
        if (embed != config_.n_embed)
        {
            throw std::runtime_error("FrameHead backward embedding dimension mismatch");
        }
        if (backward_input.dtype() != dtype_)
        {
            throw std::runtime_error("FrameHead backward dtype mismatch");
        }
        if (backward_input.device() != device_)
        {
            throw std::runtime_error("FrameHead backward device mismatch");
        }

        TraceTensor grad = graph.createTensor({batch, tokens_per_frame, embed}, backward_input.dtype(),
                                              backward_input.device(), compute_stream_descriptor_, false);
        FillZeros(graph, grad, compute_stream_descriptor_);

        if (reduction_strategy_ == FrameHeadReductionStrategy::Mean)
        {
            std::unordered_map<std::string, TraceTensor> mean_operand_grads{};
            mean_operand_grads.emplace("input", grad);
            std::unordered_map<std::string, TraceTensor> mean_param_grads{};
            mean_.buildBackward(graph, backward_input, mean_param_grads, mean_operand_grads);
        }
        else
        {
            const uint64_t last_index = tokens_per_frame - 1U;
            TraceTensor grad_last = grad.at(graph, /*dim=*/1, last_index);
            grad_last.populate(graph, backward_input, compute_stream_descriptor_);
        }

        for (auto &block : std::ranges::reverse_view(blocks_))
        {
            TraceTensor grad_prev =
                graph.createTensor(grad.shape().dims(), grad.dtype(), grad.device(), compute_stream_descriptor_, false);
            std::unordered_map<std::string, TraceTensor> operand_grads{};
            operand_grads.emplace("input", grad_prev);
            block->buildBackward(graph, grad, parameter_gradients, operand_grads);
            graph.deleteTensor(grad);
            grad = grad_prev;
        }

        std::vector<std::pair<uint32_t, uint32_t>> downsample_shapes{};
        downsample_shapes.reserve(downsample_blocks_.size());
        uint32_t current_rows = rows_;
        uint32_t current_cols = cols_;
        for (size_t idx = 0; idx < downsample_blocks_.size(); ++idx)
        {
            downsample_shapes.emplace_back(current_rows, current_cols);
            current_rows /= 2U;
            current_cols /= 2U;
        }

        for (size_t idx = downsample_blocks_.size(); idx-- > 0;)
        {
            const auto [rows_in, cols_in] = downsample_shapes[idx];
            const uint32_t rows_out = rows_in / 2U;
            const uint32_t cols_out = cols_in / 2U;

            TraceTensor grad_out_nhwc = grad.view(graph, {batch, rows_out, cols_out, embed});
            TraceTensor grad_input_nhwc =
                graph.createTensor({batch, rows_in, cols_in, embed}, grad.dtype(), grad.device(),
                                   compute_stream_descriptor_, false);

            std::unordered_map<std::string, std::any> pool_attrs{{"kernel_size", std::array<uint32_t, 2>{2, 2}},
                                                                 {"stride", std::array<uint32_t, 2>{2, 2}},
                                                                 {"padding", std::array<uint32_t, 2>{0, 0}},
                                                                 {"channels_last", true},
                                                                 {"accumulate", false}};

            graph.recordOperation(OperationEntry{.type = OpType::AVG_POOL2D_BWD,
                                                 .inputs = {grad_out_nhwc},
                                                 .outputs = {grad_input_nhwc},
                                                 .attributes = pool_attrs,
                                                 .gpu_stream_desc = compute_stream_descriptor_});

            const uint64_t pool_tokens = static_cast<uint64_t>(rows_in) * static_cast<uint64_t>(cols_in);
            TraceTensor grad_pool_flat = grad_input_nhwc.view(graph, {batch, pool_tokens, embed});

            TraceTensor grad_prev =
                graph.createTensor(grad_pool_flat.shape().dims(), grad.dtype(), grad.device(), compute_stream_descriptor_, false);
            std::unordered_map<std::string, TraceTensor> operand_grads{};
            operand_grads.emplace("input", grad_prev);
            downsample_blocks_[idx]->buildBackward(graph, grad_pool_flat, parameter_gradients, operand_grads);

            graph.deleteTensor(grad);
            grad = grad_prev;
        }

        const bool has_cell_states = bw_context_.hasSaved("cell_states");
        if (!has_cell_states)
        {
            if (auto input_it = operand_gradients.find("input"); input_it != operand_gradients.end())
            {
                if (input_it->second.shape() != grad.shape())
                {
                    throw std::runtime_error("FrameHead backward input gradient shape mismatch");
                }
                InplaceAdd(graph, input_it->second, grad, compute_stream_descriptor_);
            }
        }

        if (has_cell_states)
        {
            const TraceTensor &cell_states = bw_context_.getSaved("cell_states");
            if (cell_states.shape().ndims() != 4 || cell_states.shape()[1] != rows_ || cell_states.shape()[2] != cols_)
            {
                throw std::runtime_error("FrameHead backward cell_states shape mismatch");
            }

            TraceTensor grad_flat = grad.viewInferred(graph, {-1l, static_cast<int64_t>(embed)});
            TraceTensor cell_states_flat = cell_states.viewInferred(graph, {-1l, NUM_FRAME_CHANNELS});

            if (auto cp_it = parameter_gradients.find(name_ + ".codepoint_embedding");
                cp_it != parameter_gradients.end())
            {
                graph.recordOperation(OperationEntry{.type = OpType::CUSTOM_OP,
                                                     .inputs = {grad_flat, cell_states_flat},
                                                     .outputs = {cp_it->second},
                                                     .attributes = {{"custom_op_name", "build_cell_embed_bwd_cp"}},
                                                     .gpu_stream_desc = compute_stream_descriptor_});
            }

            const auto position_params = position_embedding_->parameters();
            if (!position_params.empty())
            {
                const std::string pos_key = position_params[0].name;
                if (auto pos_it = parameter_gradients.find(pos_key); pos_it != parameter_gradients.end())
                {
                    graph.recordOperation(OperationEntry{
                        .type = OpType::CUSTOM_OP,
                        .inputs = {grad_flat},
                        .outputs = {pos_it->second},
                        .attributes = {{"custom_op_name", "build_cell_embed_bwd_pos"}},
                        .gpu_stream_desc = compute_stream_descriptor_,
                    });
                }
            }

            auto fg_r_it = parameter_gradients.find(name_ + ".fg_r_embed");
            auto fg_g_it = parameter_gradients.find(name_ + ".fg_g_embed");
            auto fg_b_it = parameter_gradients.find(name_ + ".fg_b_embed");
            auto bg_r_it = parameter_gradients.find(name_ + ".bg_r_embed");
            auto bg_g_it = parameter_gradients.find(name_ + ".bg_g_embed");
            auto bg_b_it = parameter_gradients.find(name_ + ".bg_b_embed");

            const bool have_color_grads =
                fg_r_it != parameter_gradients.end() && fg_g_it != parameter_gradients.end() &&
                fg_b_it != parameter_gradients.end() && bg_r_it != parameter_gradients.end() &&
                bg_g_it != parameter_gradients.end() && bg_b_it != parameter_gradients.end();
            if (have_color_grads)
            {
                graph.recordOperation(OperationEntry{.type = OpType::CUSTOM_OP,
                                                     .inputs = {grad_flat, cell_states_flat},
                                                     .outputs = {fg_r_it->second, fg_g_it->second, fg_b_it->second,
                                                                 bg_r_it->second, bg_g_it->second, bg_b_it->second},
                                                     .attributes = {{"custom_op_name", "build_cell_embed_bwd_color"}},
                                                     .gpu_stream_desc = compute_stream_descriptor_});
            }
        }

        graph.deleteTensor(grad);
        bw_context_.release(graph);
    }

    const TraceTensor &FrameHeadModule::codepointEmbedding() const
    {
        if (!codepoint_embed_)
        {
            throw std::runtime_error("FrameHeadModule codepoint embedding not initialized");
        }
        return *codepoint_embed_;
    }

    const TraceTensor &FrameHeadModule::positionEmbedding() const
    {
        if (!position_embed_)
        {
            throw std::runtime_error("FrameHeadModule position embedding not initialized");
        }
        return *position_embed_;
    }

    const TraceTensor &FrameHeadModule::fgREmbedding() const
    {
        if (!fg_r_embed_)
        {
            throw std::runtime_error("FrameHeadModule FG R embedding not initialized");
        }
        return *fg_r_embed_;
    }

    const TraceTensor &FrameHeadModule::fgGEmbedding() const
    {
        if (!fg_g_embed_)
        {
            throw std::runtime_error("FrameHeadModule FG G embedding not initialized");
        }
        return *fg_g_embed_;
    }

    const TraceTensor &FrameHeadModule::fgBEmbedding() const
    {
        if (!fg_b_embed_)
        {
            throw std::runtime_error("FrameHeadModule FG B embedding not initialized");
        }
        return *fg_b_embed_;
    }

    const TraceTensor &FrameHeadModule::bgREmbedding() const
    {
        if (!bg_r_embed_)
        {
            throw std::runtime_error("FrameHeadModule BG R embedding not initialized");
        }
        return *bg_r_embed_;
    }

    const TraceTensor &FrameHeadModule::bgGEmbedding() const
    {
        if (!bg_g_embed_)
        {
            throw std::runtime_error("FrameHeadModule BG G embedding not initialized");
        }
        return *bg_g_embed_;
    }

    const TraceTensor &FrameHeadModule::bgBEmbedding() const
    {
        if (!bg_b_embed_)
        {
            throw std::runtime_error("FrameHeadModule BG B embedding not initialized");
        }
        return *bg_b_embed_;
    }

} // namespace fbamtrain
