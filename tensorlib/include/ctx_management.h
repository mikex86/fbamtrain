#pragma once

#include "gpu_stream.h"

#include <memory>
#include <optional>

namespace pi::tensorlib::internal::ctxmgmt
{
    typedef void *device_ctx_t;

    void GpuSetCurrentDevice(int ordinal);

    [[nodiscard]] device_ctx_t GpuGetDeviceCtx(int ordinal);

    [[nodiscard]] std::shared_ptr<gpustream::GpuStreamBundle> GetStreamBundle(int device_ordinal);

    [[nodiscard]] std::optional<int> GpuFindDeviceOrdinalForStream(gpustream::GpuStream stream);

} // namespace pi::tensorlib::internal::ctxmgmt
