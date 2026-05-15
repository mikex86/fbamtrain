#pragma once

#include <memory>

// Forward declaration of RealTensor class & GpuStream type
namespace pi::tensorlib
{
    class RealTensor;

    namespace gpustream
    {
        typedef void *GpuStream;
    } // namespace gpustream

} // namespace pi::tensorlib

namespace pi::tensorlib::internal::device_copy
{
    void PerformDeviceCopy(const std::shared_ptr<RealTensor> &input_tensor,
                           const std::shared_ptr<RealTensor> &output_tensor, gpustream::GpuStream stream);
}