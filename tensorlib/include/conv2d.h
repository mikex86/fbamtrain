#pragma once

#include <array>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "functional.h"
#include "module.h"

namespace pi::tensorlib
{
    class Conv2d final : public Module<>
    {
        Device device_;
        TraceTensor weight_;
        std::optional<TraceTensor> bias_;
        std::array<uint32_t, 2> stride_;
        std::array<uint32_t, 2> padding_;
        std::array<uint32_t, 2> dilation_;
        bool use_fp16_conv_acc_;
        GpuStreamDescriptor compute_stream_descriptor_;

        static std::array<uint32_t, 2> MakePair(const uint32_t value) { return {value, value}; }

      public:
        Conv2d(const std::string &name, uint32_t in_channels, uint32_t out_channels,
               const std::array<uint32_t, 2> &kernel_size, const std::array<uint32_t, 2> &stride,
               const std::array<uint32_t, 2> &padding, const std::array<uint32_t, 2> &dilation, const Device device,
               const DataType data_type, OpGraph &graph, uint32_t &init_rng_seed,
               const GpuStreamDescriptor &compute_stream_descriptor, const bool has_bias = true,
               const bool use_fp16_conv_acc = false)
            : Module(name), device_(device),
              weight_(graph.createTensor({out_channels, kernel_size[0], kernel_size[1], in_channels}, data_type, device,
                                         compute_stream_descriptor, false)),
              bias_(has_bias ? std::make_optional(
                                   graph.createTensor({out_channels}, data_type, device, compute_stream_descriptor, false))
                             : std::nullopt),
              stride_(stride), padding_(padding), dilation_(dilation), use_fp16_conv_acc_(use_fp16_conv_acc),
              compute_stream_descriptor_(compute_stream_descriptor)
        {
            const uint32_t fan_in = in_channels * kernel_size[0] * kernel_size[1];
            KaimingUniformInit(graph, weight_, fan_in, init_rng_seed++, compute_stream_descriptor_);

            if (bias_.has_value())
            {
                KaimingUniformInit(graph, *bias_, fan_in, init_rng_seed++, compute_stream_descriptor_);
            }
        }

        Conv2d(const std::string &name, const uint32_t in_channels, const uint32_t out_channels,
               const std::array<uint32_t, 2> &kernel_size, const std::array<uint32_t, 2> &stride,
               const std::array<uint32_t, 2> &padding, const Device device, const DataType data_type, OpGraph &graph,
               uint32_t &init_rng_seed, const GpuStreamDescriptor &compute_stream_descriptor, const bool has_bias = false,
               const bool use_fp16_conv_acc = false)
            : Conv2d(name, in_channels, out_channels, kernel_size, stride, padding, MakePair(1), device, data_type,
                     graph, init_rng_seed, compute_stream_descriptor, has_bias, use_fp16_conv_acc)
        {
        }

        Conv2d(const std::string &name, const uint32_t in_channels, const uint32_t out_channels,
               const uint32_t kernel_size, const uint32_t stride, const uint32_t padding, const Device device,
               const DataType data_type, OpGraph &graph, uint32_t &init_rng_seed,
               const GpuStreamDescriptor &compute_stream_descriptor, const bool has_bias = false,
               const bool use_fp16_conv_acc = false)
            : Conv2d(name, in_channels, out_channels, MakePair(kernel_size), MakePair(stride), MakePair(padding),
                     MakePair(1), device, data_type, graph, init_rng_seed, compute_stream_descriptor, has_bias,
                     use_fp16_conv_acc)
        {
        }

        Conv2d(const std::string &name, const uint32_t in_channels, const uint32_t out_channels,
               const uint32_t kernel_size, const uint32_t stride, const uint32_t padding, const uint32_t dilation,
               const Device device, const DataType data_type, OpGraph &graph, uint32_t &init_rng_seed,
               const GpuStreamDescriptor &compute_stream_descriptor, const bool has_bias = false,
               const bool use_fp16_conv_acc = false)
            : Conv2d(name, in_channels, out_channels, MakePair(kernel_size), MakePair(stride), MakePair(padding),
                     MakePair(dilation), device, data_type, graph, init_rng_seed, compute_stream_descriptor, has_bias,
                     use_fp16_conv_acc)
        {
        }

        [[nodiscard]] TraceTensor buildForward(OpGraph &graph, const std::initializer_list<TraceTensor> inputs,
                                               const bool save_input_for_backward = false) override
        {
            bw_context_.clear();
            auto input = inputs.begin()[0];
            if (save_input_for_backward)
            {
                bw_context_.saveForBackward("input", input);
            }
            const TraceTensor *bias_ptr = bias_.has_value() ? &bias_.value() : nullptr;
            return tensorlib::Conv2d(graph, input, weight_, bias_ptr, stride_, padding_, compute_stream_descriptor_,
                                     dilation_, use_fp16_conv_acc_);
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
            if (!bw_context_.hasSaved("input"))
            {
                throw std::runtime_error("Conv2d backward requested but input was not retained");
            }

            const TraceTensor &input = bw_context_.getSaved("input");
            if (input.shape().ndims() != 4 || upstream_grad.shape().ndims() != 4)
            {
                throw std::runtime_error("Conv2d backward expects 4D input and upstream tensors");
            }
            if (input.dtype() != upstream_grad.dtype())
            {
                throw std::runtime_error("Conv2d backward expects input and upstream to share dtype");
            }

            const std::string weight_key = name_ + ".weight";
            if (const auto weight_it = parameter_gradients.find(weight_key); weight_it != parameter_gradients.end())
            {
                Conv2dWgradInto(graph, input, upstream_grad, weight_it->second, stride_, padding_, dilation_,
                                compute_stream_descriptor_);
            }

            if (bias_.has_value())
            {
                const std::string bias_key = name_ + ".bias";
                if (const auto bias_it = parameter_gradients.find(bias_key); bias_it != parameter_gradients.end())
                {
                    const auto &grad_shape = upstream_grad.shape().dims();
                    const auto reduce_rows = grad_shape[0] * grad_shape[1] * grad_shape[2];
                    const auto out_channels = grad_shape[3];
                    const TraceTensor grad_flat = upstream_grad.view(graph, {reduce_rows, out_channels});
                    const TraceTensor bias_grad =
                        ReduceSum(graph, grad_flat, 0, false, upstream_grad.dtype(), compute_stream_descriptor_);
                    if (bias_it->second.shape() != bias_grad.shape())
                    {
                        throw std::runtime_error("Conv2d backward bias gradient shape mismatch");
                    }
                    DeviceCopy(graph, bias_grad, bias_it->second, compute_stream_descriptor_);
                }
            }

            if (const auto input_it = operand_gradients.find("input"); input_it != operand_gradients.end())
            {
                Conv2dDgradInto(graph, upstream_grad, weight_, input_it->second, stride_, padding_, dilation_,
                                compute_stream_descriptor_);
            }

            bw_context_.release(graph);
        }
    };
} // namespace pi::tensorlib
