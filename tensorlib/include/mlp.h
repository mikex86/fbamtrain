#pragma once


#include "tensorlib.h"
#include "linear.h"

namespace pi::tensorlib
{
    class MlpBlock final : public Module<>
    {
        std::shared_ptr<Linear> fc1_;
        std::shared_ptr<Linear> fc2_;
        GpuStreamDescriptor compute_stream_descriptor_;

      public:
        MlpBlock(const std::string &name, uint32_t embed_dim, uint32_t hidden_dim,
                 const Device device, DataType dtype, OpGraph &graph, uint32_t &init_seed,
                 const GpuStreamDescriptor &compute_stream_descriptor, const bool use_fp16_matmul_acc = false)
            : Module(name), compute_stream_descriptor_(compute_stream_descriptor)
        {
            fc1_ = std::make_shared<Linear>(name + ".fc1", embed_dim, hidden_dim, device, dtype,
                                            GELU, graph, init_seed, compute_stream_descriptor_, true, use_fp16_matmul_acc);
            fc2_ = std::make_shared<Linear>(name + ".fc2", hidden_dim, embed_dim, device, dtype,
                                            NONE, graph, init_seed, compute_stream_descriptor_, true, use_fp16_matmul_acc);
        }

        [[nodiscard]] TraceTensor buildForward(OpGraph &graph, const std::initializer_list<TraceTensor> inputs,
                                               const bool save_input_for_backward) override
        {
            auto &input = inputs.begin()[0];
            TraceTensor hidden = fc1_->buildForward(graph, {input}, save_input_for_backward);
            return fc2_->buildForward(graph, {hidden}, save_input_for_backward);
        }

        void buildBackward(OpGraph &graph, const TraceTensor &upstream_grad,
                           const std::unordered_map<std::string, TraceTensor> &parameter_gradients,
                           const std::unordered_map<std::string, TraceTensor> &operand_gradients) override
        {
            const auto fc2_params = fc2_->parameters();
            if (fc2_params.empty())
            {
                throw std::runtime_error("MlpBlock backward expected fc2 parameters");
            }
            const auto &w2 = fc2_params[0].tensor;
            if (upstream_grad.shape().ndims() != 2)
            {
                throw std::runtime_error("MlpBlock backward expects 2D upstream gradient");
            }
            if (upstream_grad.shape()[1] != w2.shape()[1])
            {
                throw std::runtime_error("MlpBlock backward upstream shape mismatch");
            }
            const uint64_t batch = upstream_grad.shape()[0];
            const uint64_t hidden_dim = w2.shape()[0];

            TraceTensor hidden_grad = graph.createTensor({batch, hidden_dim}, upstream_grad.dtype(),
                                                         upstream_grad.device(), compute_stream_descriptor_, false);
            FillZeros(graph, hidden_grad, compute_stream_descriptor_);
            std::unordered_map<std::string, TraceTensor> fc2_operand_grads{};
            fc2_operand_grads.emplace("input", hidden_grad);
            fc2_->buildBackward(graph, upstream_grad, parameter_gradients, fc2_operand_grads);
            fc1_->buildBackward(graph, hidden_grad, parameter_gradients, operand_gradients);
            graph.deleteTensor(hidden_grad);
        }

        [[nodiscard]] std::vector<ParameterEntry> parameters() const override
        {
            std::vector<ParameterEntry> params{};
            const auto fc1_params = fc1_->parameters();
            const auto fc2_params = fc2_->parameters();
            params.reserve(fc1_params.size() + fc2_params.size());
            for (const auto &entry : fc1_params)
            {
                params.push_back(entry);
            }
            for (const auto &entry : fc2_params)
            {
                params.push_back(entry);
            }
            return params;
        }
    };
}; // namespace pi::tensorlib
