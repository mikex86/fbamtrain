#pragma once

#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "functional.h"
#include "linear.h"
#include "module.h"
#include "op_graph.h"
#include "tensorlib.h"

namespace pi::tensorlib
{
    class FullMhaAttention final : public Module<>
    {
        Device device_;
        DataType dtype_;
        size_t embed_dim_;
        size_t num_heads_;
        size_t head_dim_;
        bool has_bias_;
        bool use_fp16_flash_attn_acc_;

        std::shared_ptr<Linear> qkv_proj_;
        std::shared_ptr<Linear> out_proj_;

        float softmax_scale_;
        GpuStreamDescriptor compute_stream_descriptor_;

      public:
        FullMhaAttention(const std::string &name, const size_t embed_dim, const size_t num_heads, const Device device,
                         const DataType dtype, OpGraph &graph, uint32_t &init_rng_seed,
                         const GpuStreamDescriptor &compute_stream_descriptor, const bool has_bias = true,
                         const std::optional<float> softmax_scale = std::nullopt,
                         const bool use_fp16_flash_attn_acc = false, const bool use_fp16_qkv_proj_acc = false,
                         const bool use_fp16_out_proj_acc = false)
            : Module(name), device_(device), dtype_(dtype), embed_dim_(embed_dim), num_heads_(num_heads),
              head_dim_(embed_dim / num_heads), has_bias_(has_bias), use_fp16_flash_attn_acc_(use_fp16_flash_attn_acc),
              softmax_scale_(softmax_scale.has_value() ? *softmax_scale
                                                       : 1.0f / std::sqrt(static_cast<float>(head_dim_))),
              compute_stream_descriptor_(compute_stream_descriptor)
        {
            if (embed_dim_ == 0)
            {
                throw std::invalid_argument("FullMhaAttention: embedding dimension must be greater than zero");
            }
            if (num_heads_ == 0)
            {
                throw std::invalid_argument("FullMhaAttention: number of heads must be greater than zero");
            }
            if (embed_dim_ % num_heads_ != 0)
            {
                throw std::invalid_argument(
                    "FullMhaAttention: embedding dimension must be divisible by number of heads");
            }

            qkv_proj_ =
                std::make_shared<Linear>(name + ".qkv", embed_dim_, 3 * embed_dim_, device_, dtype_, NONE, graph,
                                         init_rng_seed, compute_stream_descriptor, has_bias_, use_fp16_qkv_proj_acc);
            out_proj_ = std::make_shared<Linear>(name + ".proj", embed_dim_, embed_dim_, device_, dtype_, NONE, graph,
                                                 init_rng_seed, compute_stream_descriptor, has_bias_, use_fp16_out_proj_acc);
        }

        [[nodiscard]] TraceTensor buildForward(OpGraph &graph, const std::initializer_list<TraceTensor> inputs,
                                               bool save_input_for_backward = false) override
        {
            bw_context_.clear();
            const auto &input = inputs.begin()[0];
            const auto &input_shape = input.shape();
            if (input_shape.ndims() != 3)
            {
                throw std::invalid_argument("FullMhaAttention: input tensor must have shape (B, T, C)");
            }

            const uint64_t batch_size = input_shape[0];
            const uint64_t seq_len = input_shape[1];

            if (const uint64_t channels = input_shape[2]; channels != embed_dim_)
            {
                throw std::invalid_argument("FullMhaAttention: input embedding dimension does not match module");
            }

            if (input.dtype() != dtype_)
            {
                throw std::invalid_argument("FullMhaAttention: input dtype must match module dtype");
            }

            const uint64_t flat_tokens = batch_size * seq_len;
            const uint64_t embed_dim = embed_dim_;
            const uint64_t head_dim = embed_dim_ / num_heads_;

            TraceTensor qkv_flat = qkv_proj_->buildForward(graph, {input}, save_input_for_backward);
            if (save_input_for_backward)
            {
                bw_context_.saveForBackward("qkv", qkv_flat);
            }

            std::vector<TraceTensor> split_qkv = qkv_flat.split(graph, 3, -1);
            TraceTensor q = split_qkv[0].view(graph, {batch_size, seq_len, num_heads_, head_dim});
            TraceTensor k = split_qkv[1].view(graph, {batch_size, seq_len, num_heads_, head_dim});
            TraceTensor v = split_qkv[2].view(graph, {batch_size, seq_len, num_heads_, head_dim});

            std::optional<TraceTensor> scratch{};
            if (save_input_for_backward)
            {
                scratch = graph.createTensor({batch_size, static_cast<uint64_t>(num_heads_), seq_len},
                                             DataType::FLOAT32, device_, compute_stream_descriptor_, false);
            }
            TraceTensor attn =
                ScaledDotProductAttentionFwd(graph, q, k, v, softmax_scale_, false, use_fp16_flash_attn_acc_,
                                             scratch, compute_stream_descriptor_);
            if (save_input_for_backward)
            {
                bw_context_.saveForBackward("attn", attn);
                if (scratch.has_value())
                {
                    bw_context_.saveForBackward("scratch", *scratch);
                }
            }

            TraceTensor attn_flat = attn.view(graph, {flat_tokens, embed_dim});

            TraceTensor output = out_proj_->buildForward(graph, {attn_flat}, save_input_for_backward);

            if (!save_input_for_backward)
            {
                graph.deleteTensor(qkv_flat);
                graph.deleteTensor(attn);
            }

            return output.view(graph, {batch_size, seq_len, embed_dim});
        }

        [[nodiscard]] std::vector<ParameterEntry> parameters() const override
        {
            std::vector<ParameterEntry> params{};
            params.reserve(8);

            const auto qkv_params = qkv_proj_->parameters();
            for (const auto &entry : qkv_params)
            {
                const std::string param_name =
                    entry.name.find("weight") != std::string::npos ? name_ + ".w_qkv" : name_ + ".b_qkv";
                params.emplace_back(ParameterEntry{.name = param_name, .tensor = entry.tensor});
            }

            const auto proj_params = out_proj_->parameters();
            for (const auto &entry : proj_params)
            {
                const std::string param_name =
                    entry.name.find("weight") != std::string::npos ? name_ + ".w_proj" : name_ + ".b_proj";
                params.emplace_back(ParameterEntry{.name = param_name, .tensor = entry.tensor});
            }

            return params;
        }

        void buildBackward(OpGraph &graph, const TraceTensor &upstream_grad,
                           const std::unordered_map<std::string, TraceTensor> &parameter_gradients,
                           const std::unordered_map<std::string, TraceTensor> &operand_gradients) override
        {
            if (!bw_context_.hasSaved("qkv") || !bw_context_.hasSaved("attn") || !bw_context_.hasSaved("scratch"))
            {
                throw std::runtime_error("FullMhaAttention backward requested but required tensors were not retained");
            }

            const TraceTensor &qkv_flat = bw_context_.getSaved("qkv");
            const TraceTensor &attn = bw_context_.getSaved("attn");
            const TraceTensor &scratch = bw_context_.getSaved("scratch");

            if (upstream_grad.shape().ndims() != 3)
            {
                throw std::invalid_argument("FullMhaAttention backward expects upstream gradient shape (B, T, C)");
            }
            const uint64_t batch_size = upstream_grad.shape()[0];
            const uint64_t seq_len = upstream_grad.shape()[1];
            const uint64_t embed_dim = upstream_grad.shape()[2];
            const uint64_t num_heads = num_heads_;
            const uint64_t head_dim = embed_dim / num_heads_;

            if (num_heads != num_heads_ || head_dim != head_dim_)
            {
                throw std::runtime_error("FullMhaAttention backward retained tensor shape mismatch");
            }
            if (qkv_flat.shape().ndims() != 2 || qkv_flat.shape()[0] != batch_size * seq_len ||
                qkv_flat.shape()[1] != 3 * embed_dim)
            {
                throw std::runtime_error("FullMhaAttention backward saved qkv shape mismatch");
            }
            if (upstream_grad.dtype() != dtype_)
            {
                throw std::runtime_error("FullMhaAttention backward upstream dtype mismatch");
            }

            const uint64_t flat_tokens = batch_size * seq_len;

            auto qkv_saved_splits = qkv_flat.split(graph, 3, -1);
            if (qkv_saved_splits.size() != 3)
            {
                throw std::runtime_error("FullMhaAttention backward expected 3-way split for qkv tensor");
            }
            TraceTensor q = qkv_saved_splits[0].view(graph, {batch_size, seq_len, num_heads, head_dim});
            TraceTensor k = qkv_saved_splits[1].view(graph, {batch_size, seq_len, num_heads, head_dim});
            TraceTensor v = qkv_saved_splits[2].view(graph, {batch_size, seq_len, num_heads, head_dim});

            TraceTensor upstream_flat = upstream_grad.view(graph, {flat_tokens, embed_dim});
            TraceTensor attn_grad_flat = graph.createTensor({flat_tokens, embed_dim}, upstream_grad.dtype(),
                                                            upstream_grad.device(), compute_stream_descriptor_, false);

            std::unordered_map<std::string, TraceTensor> proj_param_grads{};
            for (const auto &entry : out_proj_->parameters())
            {
                const std::string key =
                    entry.name.find("weight") != std::string::npos ? name_ + ".w_proj" : name_ + ".b_proj";
                if (auto it = parameter_gradients.find(key); it != parameter_gradients.end())
                {
                    proj_param_grads.emplace(entry.name, it->second);
                }
            }

            std::unordered_map<std::string, TraceTensor> proj_operand_grads{};
            proj_operand_grads.emplace("input", attn_grad_flat);
            out_proj_->buildBackward(graph, upstream_flat, proj_param_grads, proj_operand_grads);

            TraceTensor attn_grad = attn_grad_flat.view(graph, {batch_size, seq_len, num_heads, head_dim});
            TraceTensor &attn_grad_contig = attn_grad;
            TraceTensor attn_contig = attn;

            TraceTensor qkv_grad_flat =
                graph.createTensor({flat_tokens, 3 * embed_dim}, dtype_, device_, compute_stream_descriptor_, false);
            auto qkv_splits = qkv_grad_flat.split(graph, 3, -1);
            if (qkv_splits.size() != 3)
            {
                throw std::runtime_error("FullMhaAttention backward expected 3-way split for qkv gradient");
            }

            TraceTensor grad_q = qkv_splits[0].view(graph, {batch_size, seq_len, num_heads, head_dim});
            TraceTensor grad_k = qkv_splits[1].view(graph, {batch_size, seq_len, num_heads, head_dim});
            TraceTensor grad_v = qkv_splits[2].view(graph, {batch_size, seq_len, num_heads, head_dim});

            ScaledDotProductAttentionBwdInto(graph, q, k, v, attn_contig, scratch, attn_grad_contig, grad_q, grad_k,
                                             grad_v, softmax_scale_, false, compute_stream_descriptor_);

            std::unordered_map<std::string, TraceTensor> qkv_param_grads{};
            for (const auto &entry : qkv_proj_->parameters())
            {
                const std::string key =
                    entry.name.find("weight") != std::string::npos ? name_ + ".w_qkv" : name_ + ".b_qkv";
                if (auto it = parameter_gradients.find(key); it != parameter_gradients.end())
                {
                    qkv_param_grads.emplace(entry.name, it->second);
                }
            }

            std::unordered_map<std::string, TraceTensor> qkv_operand_grads{};
            if (auto input_it = operand_gradients.find("input"); input_it != operand_gradients.end())
            {
                if (input_it->second.shape().ndims() != 3 || input_it->second.shape()[0] != batch_size ||
                    input_it->second.shape()[1] != seq_len || input_it->second.shape()[2] != embed_dim)
                {
                    throw std::runtime_error("FullMhaAttention backward input gradient shape mismatch");
                }
                TraceTensor input_flat = input_it->second.view(graph, {flat_tokens, embed_dim});
                qkv_operand_grads.emplace("input", input_flat);
            }

            qkv_proj_->buildBackward(graph, qkv_grad_flat, qkv_param_grads, qkv_operand_grads);

            graph.deleteTensor(attn_grad_flat);
            graph.deleteTensor(qkv_grad_flat);
            bw_context_.release(graph);
        }
    };
} // namespace pi::tensorlib
