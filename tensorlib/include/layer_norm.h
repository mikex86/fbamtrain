#pragma once

#include <stdexcept>

#include "functional.h"
#include "module.h"
#include "op_graph.h"
#include "tensorlib.h"

namespace pi::tensorlib
{
    class LayerNorm final : public Module
    {
        Device device_;
        TraceTensor weight_;
        TraceTensor bias_;
        float eps_;
        bool elementwise_affine_;

      public:
        LayerNorm(const std::string &name, size_t normalized_dim, const Device device, const DataType data_type,
                  float eps, OpGraph &graph, uint32_t &init_rng_seed, bool elementwise_affine = true)
            : Module(name),
              device_(device),
              weight_(graph.createTensor({normalized_dim}, data_type, device)),
              bias_(graph.createTensor({normalized_dim}, data_type, device)),
              eps_(eps),
              elementwise_affine_(elementwise_affine)
        {
            FillConstant(graph, weight_, 1.0f);
            FillConstant(graph, bias_, 0.0f);
        }

        [[nodiscard]] TraceTensor buildForward(OpGraph &graph, std::initializer_list<TraceTensor> inputs,
                                               bool /*save_input_for_backward*/ = false) override
        {
            if (inputs.size() != 1)
            {
                throw std::invalid_argument("LayerNorm expects exactly one input tensor");
            }
            const TraceTensor &input = *inputs.begin();
            if (input.device() != device_)
            {
                throw std::runtime_error("LayerNorm input device must match module device");
            }
            if (input.dtype() != weight_.dtype())
            {
                throw std::runtime_error("LayerNorm input dtype must match module dtype");
            }
            if (input.shape().ndims() == 0)
            {
                throw std::invalid_argument("LayerNorm input must have at least one dimension");
            }
            const auto hidden_size = weight_.shape()[0];
            if (input.shape().dims().back() != hidden_size)
            {
                throw std::invalid_argument("LayerNorm normalized dimension must match last dimension of input");
            }
            return LayerNormFwd(graph, input, weight_, bias_, eps_);
        }

        void buildBackward(OpGraph & /*graph*/, const TraceTensor & /*backward_input*/,
                           std::unordered_map<std::string, TraceTensor> & /*parameter_gradients*/,
                           std::unordered_map<std::string, TraceTensor> & /*operand_gradients*/) override
        {
            throw std::runtime_error("LayerNorm backward is not implemented.");
        }

        [[nodiscard]] std::vector<ParameterEntry> parameters() const override
        {
            if (!elementwise_affine_)
            {
                return {};
            }
            return {ParameterEntry{.name = name_ + ".weight", .tensor = weight_},
                    ParameterEntry{.name = name_ + ".bias", .tensor = bias_}};
        }
    };
} // namespace pi::tensorlib
