#pragma once

#include <optional>

#include "tensorlib.h"
#include <stdexcept>

#include "module.h"
#include "op_graph.h"

#include "activation.h"
#include "functional.h"

namespace pi::tensorlib
{
    class Linear final : public Module<>
    {
        Device device_;
        TraceTensor weight_;
        std::optional<TraceTensor> bias_;
        ActivationFunction activation_function_;
        bool use_fp16_matmul_acc_;
        GpuStreamDescriptor compute_stream_descriptor_;

      public:
        Linear(const std::string &name, size_t input_features, size_t output_features, const Device device,
               const DataType data_type, const ActivationFunction activation_function, OpGraph &graph,
               uint32_t &init_rng_seed, const GpuStreamDescriptor &compute_stream_descriptor, const bool has_bias = true, const bool use_fp16_matmul_acc = false)
            : Module(name), device_(device),
              weight_(graph.createTensor({input_features, output_features}, data_type, device,
                                         compute_stream_descriptor, false)),
              bias_(has_bias ? std::make_optional(graph.createTensor({output_features}, data_type, device,
                                                                     compute_stream_descriptor, false))
                             : std::nullopt),
              activation_function_(activation_function),
              use_fp16_matmul_acc_(use_fp16_matmul_acc),
              compute_stream_descriptor_(compute_stream_descriptor)
        {
            KaimingUniformInit(graph, weight_, input_features, init_rng_seed++, compute_stream_descriptor_);

            if (bias_.has_value())
            {
                KaimingUniformInit(graph, *bias_, input_features, init_rng_seed++, compute_stream_descriptor_);
            }
        }

        [[nodiscard]] TraceTensor buildForward(OpGraph &graph, const std::initializer_list<TraceTensor> inputs,
                                               const bool save_input_for_backward) override
        {
            bw_context_.clear();
            const auto &input = inputs.begin()[0];

            const auto &input_shape = input.shape();
            if (input_shape.ndims() < 2)
            {
                throw std::invalid_argument("Input tensor must have at least 2 dimensions.");
            }

            const uint64_t input_features = weight_.shape()[0];
            const uint64_t output_features = weight_.shape()[1];
            TraceTensor x = input;
            if (save_input_for_backward)
            {
                bw_context_.saveForBackward("x", x);
            }
            if (input_shape.ndims() > 2)
            {
                // treat other dimensions as batch dimensions
                x = x.viewInferred(graph, {-1, static_cast<int64_t>(input_features)});
            }

            const uint64_t batch_size = x.shape()[0];

            // assert dtype matches
            if (x.dtype() != weight_.dtype())
            {
                throw std::invalid_argument("Input tensor dtype must match weight dtype.");
            }
            TraceTensor output =
                graph.createTensor({batch_size, output_features}, weight_.dtype(), device_, compute_stream_descriptor_, false);
            std::unordered_map<std::string, std::any> matmul_attributes{};
            if (weight_.dtype() == DataType::FLOAT16 && use_fp16_matmul_acc_)
            {
                matmul_attributes.emplace("use_fp16_matmul_acc", use_fp16_matmul_acc_);
            }
            if (save_input_for_backward && activation_function_ != NONE)
            {
                matmul_attributes.emplace("write_out_preact", true);
            }
            graph.recordOperation(
                OperationEntry{.type = OpType::MATMUL,
                               .inputs = {x, weight_},
                               .outputs = {output},
                               .attributes = std::move(matmul_attributes),
                               .gpu_stream_desc = compute_stream_descriptor_});

            if (bias_.has_value())
            {
                std::unordered_map<std::string, std::any> plus_attributes{};
                graph.recordOperation(OperationEntry{
                    .type = OpType::PLUS,
                    .inputs = {output, *bias_},
                    .outputs = {output},
                    .attributes = plus_attributes,
                    .gpu_stream_desc = compute_stream_descriptor_});
            }

            if (activation_function_ != NONE)
            {
                std::unordered_map<std::string, std::any> act_attributes{};
                act_attributes.emplace("activation_function", activation_function_);
                if (save_input_for_backward)
                {
                    bw_context_.saveForBackward("pre_act", output);
                    TraceTensor activated =
                        graph.createTensor(output.shape().dims(), output.dtype(), device_, compute_stream_descriptor_, false);
                    graph.recordOperation(OperationEntry{
                        .type = OpType::ACT_FN,
                        .inputs = {output},
                        .outputs = {activated},
                        .attributes = act_attributes,
                        .gpu_stream_desc = compute_stream_descriptor_});
                    output = activated;
                }
                else
                {
                    graph.recordOperation(OperationEntry{.type = OpType::ACT_FN,
                                                         .inputs = {output},
                                                         .outputs = {output},
                                                         .attributes = act_attributes,
                                                         .gpu_stream_desc = compute_stream_descriptor_});
                }
            }

            return output;
        }

        [[nodiscard]] std::vector<ParameterEntry> parameters() const override
        {
            std::vector params{ParameterEntry{.name = name_ + '.' + "weight", .tensor = weight_}};

            if (bias_.has_value())
            {
                params.emplace_back(ParameterEntry{.name = name_ + '.' + "bias", .tensor = bias_.value()});
            }
            return params;
        }

        void buildBackward(OpGraph &graph, const TraceTensor &upstream_grad,
                           const std::unordered_map<std::string, TraceTensor> &parameter_gradients,
                           const std::unordered_map<std::string, TraceTensor> &operand_gradients) override
        {
            if (!bw_context_.hasSaved("x"))
            {
                throw std::runtime_error("Linear backward requested but input was not retained");
            }
            const TraceTensor &x_saved = bw_context_.getSaved("x");
            TraceTensor x = x_saved;
            const auto output_features = upstream_grad.shape()[1];
            const auto input_features = weight_.shape()[0];
            if (x_saved.shape().ndims() > 2)
            {
                x = x_saved.viewInferred(graph, {-1, static_cast<int64_t>(input_features)});
            }
            if (x.shape().ndims() != 2)
            {
                throw std::runtime_error("Linear backward expects saved input to be rank-2 after flattening");
            }
            if (output_features != weight_.shape()[1] || x.shape()[1] != input_features)
            {
                throw std::runtime_error("Linear backward upstream/input shape mismatch");
            }

            std::unordered_map<std::string, std::any> base_matmul_attributes{};
            if (weight_.dtype() == DataType::FLOAT16 && use_fp16_matmul_acc_)
            {
                base_matmul_attributes.emplace("use_fp16_matmul_acc", use_fp16_matmul_acc_);
            }
            const auto compute_stream_descriptor = compute_stream_descriptor_;

            TraceTensor upstream_act_grad = upstream_grad;
            if (activation_function_ != NONE)
            {
                if (!bw_context_.hasSaved("pre_act"))
                {
                    throw std::runtime_error("Linear backward requested but pre-activation was not retained");
                }
                if (activation_function_ != GELU)
                {
                    throw std::runtime_error("Linear backward only implemented for ActivationFunction::GELU");
                }
                const TraceTensor &pre_act = bw_context_.getSaved("pre_act");
                upstream_act_grad = GeluBackward(graph, pre_act, upstream_grad, compute_stream_descriptor_);
            }

            const std::string weight_key = name_ + ".weight";
            if (auto weight_grad_it = parameter_gradients.find(weight_key); weight_grad_it != parameter_gradients.end())
            {
                auto weight_matmul_attributes = base_matmul_attributes;
                weight_matmul_attributes.emplace("accumulate_output", true);
                TraceTensor x_t = x.transpose(graph, {1, 0});
                graph.recordOperation(OperationEntry{.type = OpType::MATMUL,
                                                     .inputs = {x_t, upstream_act_grad},
                                                     .outputs = {weight_grad_it->second},
                                                     .attributes = weight_matmul_attributes,
                                                     .gpu_stream_desc = compute_stream_descriptor});
            }

            if (auto input_grad_it = operand_gradients.find("input"); input_grad_it != operand_gradients.end())
            {
                TraceTensor weight_t = weight_.transpose(graph, {1, 0});
                auto input_matmul_attributes = base_matmul_attributes;
                graph.recordOperation(
                    OperationEntry{.type = OpType::MATMUL,
                                   .inputs = {upstream_act_grad, weight_t},
                                   .outputs = {input_grad_it->second},
                                   .attributes = input_matmul_attributes,
                                   .gpu_stream_desc = compute_stream_descriptor});
                if (input_grad_it->second.shape() != x.shape())
                {
                    throw std::runtime_error("Linear backward input gradient shape mismatch");
                }
                if (input_grad_it->second.dtype() != x.dtype() || input_grad_it->second.device() != x.device())
                {
                    throw std::runtime_error("Linear backward input gradient dtype/device mismatch");
                }
            }

            if (bias_.has_value())
            {
                const std::string bias_key = name_ + ".bias";
                if (auto bias_grad_it = parameter_gradients.find(bias_key); bias_grad_it != parameter_gradients.end())
                {
                    TraceTensor db_vec_fp32 = ReduceSum(graph, upstream_act_grad, /*dim=*/0, /*keepdim=*/false,
                                                        DataType::FLOAT32, compute_stream_descriptor);
                    TraceTensor db_vec = db_vec_fp32.to(graph, bias_grad_it->second.dtype(), compute_stream_descriptor_);
                    InplaceAdd(graph, bias_grad_it->second, db_vec, compute_stream_descriptor_);
                    graph.deleteTensor(db_vec_fp32);
                    if (db_vec.id() != db_vec_fp32.id())
                    {
                        graph.deleteTensor(db_vec);
                    }
                }
            }

            if (activation_function_ != NONE)
            {
                graph.deleteTensor(upstream_act_grad);
            }

            bw_context_.release(graph);

        }

    };

} // namespace pi::tensorlib
