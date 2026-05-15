#pragma once

#include <stdexcept>
#include <vector>

#include "functional.h"
#include "module.h"
#include "op_graph.h"
#include "tensorlib.h"

namespace pi::tensorlib
{
    class MeanModule final : public Module<>
    {
        int64_t dim_;
        bool keepdim_;
        uint64_t reduction_size_;
        TraceTensor denom_;
        float denom_value_;
        OpGraph *denom_graph_{nullptr};
        GpuStreamDescriptor compute_stream_descriptor_;

      public:
        MeanModule(const std::string &name, const int64_t dim, const bool keepdim, const uint64_t reduction_size,
                   OpGraph &graph, const DataType dtype, const Device device, const GpuStreamDescriptor &compute_stream_descriptor)
            : Module(name), dim_(dim), keepdim_(keepdim), reduction_size_(reduction_size),
              denom_(graph.createTensor({1}, dtype, device, compute_stream_descriptor, false)), denom_value_(static_cast<float>(reduction_size)),
              denom_graph_(&graph), compute_stream_descriptor_(compute_stream_descriptor)
        {
            if (reduction_size_ == 0)
            {
                throw std::runtime_error("Mean requires a non-zero reduction size");
            }
            FillConstant(graph, denom_, denom_value_, compute_stream_descriptor);
        }

        [[nodiscard]] TraceTensor buildForward(OpGraph &graph, const std::initializer_list<TraceTensor> inputs,
                                               const bool save_input_for_backward) override
        {
            const auto &input = inputs.begin()[0];
            const auto ndims = static_cast<int64_t>(input.shape().ndims());
            if (ndims == 0)
            {
                throw std::runtime_error("Mean forward does not support scalar inputs");
            }
            int64_t reduction_dim = dim_;
            if (reduction_dim < 0)
            {
                reduction_dim += ndims;
            }
            if (reduction_dim < 0 || reduction_dim >= ndims)
            {
                throw std::runtime_error("Mean forward reduction dimension out of range");
            }
            if (const uint64_t reduction_size = input.shape()[reduction_dim]; reduction_size != reduction_size_)
            {
                throw std::runtime_error("Mean forward reduction size mismatch for fixed module");
            }
            if (denom_.dtype() != input.dtype() || denom_.device() != input.device())
            {
                throw std::runtime_error("Mean forward denominator dtype/device mismatch");
            }
            return Mean(graph, input, dim_, keepdim_, compute_stream_descriptor_);
        }

        void buildBackward(OpGraph &graph, const TraceTensor &upstream_grad,
                           const std::unordered_map<std::string, TraceTensor> &,
                           const std::unordered_map<std::string, TraceTensor> &operand_gradients) override
        {
            const auto input_it = operand_gradients.find("input");
            if (input_it == operand_gradients.end())
            {
                return;
            }

            const TraceTensor &input_grad = input_it->second;
            const auto ndims = static_cast<int64_t>(input_grad.shape().ndims());
            if (ndims == 0)
            {
                throw std::runtime_error("Mean backward does not support scalar inputs");
            }

            int64_t reduction_dim = dim_;
            if (reduction_dim < 0)
            {
                reduction_dim += ndims;
            }
            if (reduction_dim < 0 || reduction_dim >= ndims)
            {
                throw std::runtime_error("Mean backward reduction dimension out of range");
            }
            const bool needs_scale = reduction_size_ > 1;

            TraceTensor upstream_view = upstream_grad;
            if (const size_t expected_upstream_ndims =
                    keepdim_ ? input_grad.shape().ndims() : input_grad.shape().ndims() - 1;
                upstream_grad.shape().ndims() != expected_upstream_ndims)
            {
                throw std::runtime_error("Mean backward upstream grad rank mismatch");
            }
            if (!keepdim_)
            {
                const auto &shape_dims = input_grad.shape().dims();
                std::vector<int64_t> view_dims(shape_dims.begin(), shape_dims.end());
                view_dims[static_cast<size_t>(reduction_dim)] = 1;
                upstream_view = upstream_grad.view(graph, view_dims);
            }

            const TraceTensor expanded =
                needs_scale ? upstream_view.broadcast(graph, reduction_dim, reduction_size_) : upstream_view;

            if (input_grad.shape() != expanded.shape())
            {
                throw std::runtime_error("Mean backward input gradient shape mismatch");
            }
            if (input_grad.dtype() != expanded.dtype() || input_grad.device() != expanded.device())
            {
                throw std::runtime_error("Mean backward input gradient dtype/device mismatch");
            }

            if (needs_scale)
            {
                if (!graph.hasTensor(denom_.id()))
                {
                    graph.createTensor(denom_);
                    FillConstant(graph, denom_, denom_value_, compute_stream_descriptor_);
                }
                if (denom_.dtype() != expanded.dtype() || denom_.device() != expanded.device())
                {
                    throw std::runtime_error("Mean backward denominator dtype/device mismatch");
                }
                InplaceDiv(graph, expanded, denom_, compute_stream_descriptor_);
            }
            InplaceAdd(graph, input_grad, expanded, compute_stream_descriptor_);
        }

        [[nodiscard]] std::vector<ParameterEntry> parameters() const override { return {}; }
    };
} // namespace pi::tensorlib
