#pragma once

#include <array>
#include <cstdint>
#include <stdexcept>

#include "functional.h"
#include "module.h"
#include "op_graph.h"

namespace pi::tensorlib
{
    class AvgPool2d final : public Module
    {
        Device device_;
        DataType dtype_;
        std::array<uint32_t, 2> kernel_size_;
        std::array<uint32_t, 2> stride_;
        std::array<uint32_t, 2> padding_;
        GpuStreamDescriptor compute_stream_descriptor_;

      public:
        AvgPool2d(const std::string &name, std::array<uint32_t, 2> kernel_size, std::array<uint32_t, 2> stride,
                  Device device, DataType dtype, const GpuStreamDescriptor &compute_stream_descriptor,
                  std::array<uint32_t, 2> padding = {0, 0})
            : Module(name), device_(device), dtype_(dtype), kernel_size_(kernel_size), stride_(stride),
              padding_(padding), compute_stream_descriptor_(compute_stream_descriptor)
        {
            if (kernel_size_[0] == 0 || kernel_size_[1] == 0)
            {
                throw std::invalid_argument("AvgPool2d kernel dimensions must be > 0");
            }
            if (stride_[0] == 0 || stride_[1] == 0)
            {
                throw std::invalid_argument("AvgPool2d strides must be > 0");
            }
        }

        [[nodiscard]] TraceTensor buildForward(OpGraph &graph, std::initializer_list<TraceTensor> inputs,
                                               bool /*save_input_for_backward*/ = false) override
        {
            if (inputs.size() != 1)
            {
                throw std::invalid_argument("AvgPool2d expects exactly one input tensor");
            }
            const TraceTensor &input = *inputs.begin();
            if (input.device() != device_)
            {
                throw std::runtime_error("AvgPool2d input device must match module device");
            }
            if (input.dtype() != dtype_)
            {
                throw std::runtime_error("AvgPool2d input dtype must match module dtype");
            }
            if (input.shape().ndims() != 4)
            {
                throw std::invalid_argument("AvgPool2d expects input with shape (N, C, H, W)");
            }
            return tensorlib::AvgPool2d(graph, input, kernel_size_, stride_, compute_stream_descriptor_, padding_);
        }

        void buildBackward(OpGraph & /*graph*/, const TraceTensor & /*backward_input*/,
                           std::unordered_map<std::string, TraceTensor> & /*parameter_gradients*/,
                           std::unordered_map<std::string, TraceTensor> & /*operand_gradients*/) override
        {
            throw std::runtime_error("AvgPool2d backward is not implemented.");
        }

        [[nodiscard]] std::vector<ParameterEntry> parameters() const override { return {}; }
    };
} // namespace pi::tensorlib
