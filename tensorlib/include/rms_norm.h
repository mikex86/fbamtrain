#pragma once

#include <stdexcept>

#include "functional.h"
#include "module.h"
#include "op_graph.h"
#include "tensorlib.h"

namespace pi::tensorlib
{
    class RmsNorm final : public Module<>
    {
        Device device_;
        TraceTensor weight_;
        float eps_;
        bool inplace_;
        GpuStreamDescriptor compute_stream_descriptor_;

      public:
        RmsNorm(const std::string &name, size_t normalized_dim, const Device device, const DataType data_type,
                const float eps, OpGraph &graph, const bool inplace, const GpuStreamDescriptor &compute_stream_descriptor)
            : Module(name), device_(device),
              weight_(graph.createTensor({normalized_dim}, data_type, device, compute_stream_descriptor, false)),
              eps_(eps), inplace_(inplace), compute_stream_descriptor_(compute_stream_descriptor)
        {
            FillConstant(graph, weight_, 1.0f, compute_stream_descriptor_);
        }

        [[nodiscard]] TraceTensor buildForward(OpGraph &graph, const std::initializer_list<TraceTensor> inputs,
                                               const bool save_input_for_backward = false) override
        {
            bw_context_.clear();
            auto input = inputs.begin()[0];
            if (input.device() != device_)
            {
                throw std::runtime_error("RmsNorm input device must match module device");
            }
            if (input.shape().ndims() == 0)
            {
                throw std::invalid_argument("RmsNorm input must have at least one dimension");
            }
            if (const auto hidden_size = weight_.shape()[0]; input.shape().dims().back() != hidden_size)
            {
                throw std::invalid_argument("RmsNorm normalized dimension must match last dimension of input");
            }
            if (input.dtype() != weight_.dtype())
            {
                throw std::runtime_error("RmsNorm input dtype must match module dtype");
            }
            if (save_input_for_backward && inplace_)
            {
                throw std::runtime_error("RmsNorm cannot save input for backward when inplace is true");
            }
            if (save_input_for_backward)
            {
                bw_context_.saveForBackward("input", input);
            }
            if (inplace_)
            {
                RmsNormFwdInplace(graph, input, weight_, eps_, compute_stream_descriptor_);
                return std::move(input);
            }
            return RmsNormFwd(graph, input, weight_, eps_, compute_stream_descriptor_);
        }

        [[nodiscard]] std::vector<ParameterEntry> parameters() const override
        {
            return {ParameterEntry{.name = name_ + ".weight", .tensor = weight_}};
        }

        void buildBackward(OpGraph &graph, const TraceTensor &upstream_grad,
                           const std::unordered_map<std::string, TraceTensor> &parameter_gradients,
                           const std::unordered_map<std::string, TraceTensor> &operand_gradients) override
        {
            if (!bw_context_.hasSaved("input"))
            {
                throw std::runtime_error("RmsNorm backward requested but input was not retained");
            }

            const TraceTensor &input = bw_context_.getSaved("input");
            if (upstream_grad.shape() != input.shape())
            {
                throw std::runtime_error("RmsNorm backward upstream shape mismatch");
            }
            if (upstream_grad.dtype() != input.dtype() || upstream_grad.device() != input.device())
            {
                throw std::runtime_error("RmsNorm backward upstream dtype/device mismatch");
            }

            TraceTensor x_hat =
                graph.createTensor(input.shape().dims(), input.dtype(), input.device(), compute_stream_descriptor_, false);

            std::optional<TraceTensor> grad_input_target;
            bool delete_grad_input_temp = false;
            if (auto input_grad_it = operand_gradients.find("input"); input_grad_it != operand_gradients.end())
            {
                grad_input_target = input_grad_it->second;
                if (grad_input_target->shape() != input.shape())
                {
                    throw std::runtime_error("RmsNorm backward input grad shape mismatch");
                }
                if (grad_input_target->dtype() != input.dtype() || grad_input_target->device() != input.device())
                {
                    throw std::runtime_error("RmsNorm backward input grad dtype/device mismatch");
                }
            }
            else
            {
                grad_input_target = graph.createTensor(input.shape().dims(), input.dtype(), input.device(),
                                                       compute_stream_descriptor_, false);
                delete_grad_input_temp = true;
            }

            RmsNormBwd(graph, input, weight_, upstream_grad, *grad_input_target, x_hat, eps_, compute_stream_descriptor_);

            if (delete_grad_input_temp)
            {
                graph.deleteTensor(*grad_input_target);
            }

            const std::string weight_key = name_ + ".weight";
            if (auto weight_grad_it = parameter_gradients.find(weight_key);
                weight_grad_it != parameter_gradients.end())
            {
                const auto hidden = static_cast<int64_t>(input.shape().dims().back());
                TraceTensor upstream_view = upstream_grad.viewInferred(graph, {-1, hidden});
                TraceTensor x_hat_view = x_hat.viewInferred(graph, {-1, hidden});
                TraceTensor mul_tmp = graph.createTensor(upstream_view.shape().dims(), input.dtype(),
                                                         input.device(), compute_stream_descriptor_, false);
                graph.recordOperation(OperationEntry{.type = OpType::MUL,
                                                     .inputs = {upstream_view, x_hat_view},
                                                     .outputs = {mul_tmp},
                                                     .attributes = {},
                                                     .gpu_stream_desc = compute_stream_descriptor_});
                TraceTensor dw = ReduceSum(graph, mul_tmp, /*dim=*/0, /*keepdim=*/false,
                                           weight_grad_it->second.dtype(), compute_stream_descriptor_);
                InplaceAdd(graph, weight_grad_it->second, dw, compute_stream_descriptor_);
                graph.deleteTensor(mul_tmp);
                graph.deleteTensor(dw);
            }

            graph.deleteTensor(x_hat);
            bw_context_.release(graph);
        }
    };
} // namespace pi::tensorlib
