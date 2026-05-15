#pragma once

#include <cstdint>
#include <stdexcept>

#include "functional.h"
#include "module.h"
#include "op_graph.h"

namespace pi::tensorlib
{
    class AvgPool1d final : public Module
    {
        Device device_;
        DataType dtype_;
        uint32_t kernel_size_;
        uint32_t stride_;
        int64_t pool_dim_;
        GpuStreamDescriptor compute_stream_descriptor_;

      public:
        AvgPool1d(const std::string &name, uint32_t kernel_size, uint32_t stride, Device device, DataType dtype,
                  const GpuStreamDescriptor &compute_stream_descriptor, int64_t pool_dim = 2)
            : Module(name), device_(device), dtype_(dtype), kernel_size_(kernel_size), stride_(stride),
              pool_dim_(pool_dim), compute_stream_descriptor_(compute_stream_descriptor)
        {
            if (kernel_size_ == 0)
            {
                throw std::invalid_argument("AvgPool1d kernel_size must be > 0");
            }
            if (stride_ == 0)
            {
                throw std::invalid_argument("AvgPool1d stride must be > 0");
            }
        }

        [[nodiscard]] TraceTensor buildForward(OpGraph &graph, std::initializer_list<TraceTensor> inputs,
                                               bool /*save_input_for_backward*/ = false) override
        {
            if (inputs.size() != 1)
            {
                throw std::invalid_argument("AvgPool1d expects exactly one input tensor");
            }
            const TraceTensor &input = *inputs.begin();
            if (input.device() != device_)
            {
                throw std::runtime_error("AvgPool1d input device must match module device");
            }
            if (input.dtype() != dtype_)
            {
                throw std::runtime_error("AvgPool1d input dtype must match module dtype");
            }
            if (input.shape().ndims() != 3)
            {
                throw std::invalid_argument("AvgPool1d expects input with shape (N, C, L)");
            }
            return tensorlib::AvgPool1d(graph, input, kernel_size_, stride_, compute_stream_descriptor_, pool_dim_);
        }

        void buildBackward(OpGraph & /*graph*/, const TraceTensor & /*backward_input*/,
                           std::unordered_map<std::string, TraceTensor> & /*parameter_gradients*/,
                           std::unordered_map<std::string, TraceTensor> & /*operand_gradients*/) override
        {
            throw std::runtime_error("AvgPool1d backward is not implemented.");
        }

        [[nodiscard]] std::vector<ParameterEntry> parameters() const override { return {}; }
    };
} // namespace pi::tensorlib
