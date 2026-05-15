#pragma once

#include "execution_plan.h"
#include "gpu_stream.h"

#include <stdexcept>

namespace pi::tensorlib::streamutils
{
    [[nodiscard]] inline gpustream::GpuStream
    GetStream(const std::shared_ptr<gpustream::GpuStreamBundle> &stream_bundle,
              const GpuStreamDescriptor &stream_desc)
    {
        if (!stream_desc.isValid())
        {
            throw std::runtime_error("Invalid stream descriptor for execution entry.");
        }
        switch (stream_desc.kind)
        {
            case StreamKind::Compute:
                return stream_bundle->getComputeStream(stream_desc.compute_stream_id.value_or(0));
            case StreamKind::H2D:
                return stream_bundle->h2d_stream;
            case StreamKind::D2H:
                return stream_bundle->d2h_stream;
            case StreamKind::Main:
                return stream_bundle->main_stream;
            case StreamKind::Cleanup:
                return stream_bundle->cleanup_stream;
            default:
                throw std::runtime_error("Unknown stream descriptor for execution entry.");
        }
    };

    [[nodiscard]] inline gpustream::GpuStream
    GetStream(const std::shared_ptr<gpustream::GpuStreamBundle> &stream_bundle, const ExecutionEntry &entry)
    {
        return GetStream(stream_bundle, entry.gpu_stream_desc);
    }
} // namespace pi::tensorlib::streamutils
