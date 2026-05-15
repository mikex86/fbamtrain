#pragma once

#include <optional>

namespace pi::tensorlib
{
    enum class StreamKind
    {
        Compute,
        H2D,
        D2H,
        Main,
        Cleanup
    };

    [[nodiscard]] inline bool IsValidStreamKind(const StreamKind kind)
    {
        switch (kind)
        {
            case StreamKind::Compute:
            case StreamKind::H2D:
            case StreamKind::D2H:
            case StreamKind::Main:
            case StreamKind::Cleanup:
                return true;
            default:
                return false;
        }
    }

    /**
     * Specifies a GPU stream used for an operation.
     *
     * A GPU stream is identified by its kind (compute, H2D, D2H, main) and, for compute streams,
     * an optional compute stream ID.
     *
     * The H2D stream is the stream "async" h2d copies are issued to.
     * The D2H stream is the stream "async" d2h copies are issued to.
     * The main stream is the default compute stream (stream ID 0).
     * The cleanup stream is a dedicated compute stream (stream ID -3).
     * The compute stream id identifies a specific compute stream that can be used for parallel execution.
     * StreamKind::Compute with compute_stream_id = 1 is equivalent to StreamKind::Main.
     */
    struct GpuStreamDescriptor
    {
        StreamKind kind = StreamKind::Main;
        std::optional<int> compute_stream_id = std::nullopt;

        [[nodiscard]] bool operator==(const GpuStreamDescriptor &rhs) const
        {
            return kind == rhs.kind && compute_stream_id == rhs.compute_stream_id;
        }

        [[nodiscard]] int getStreamId() const;
        [[nodiscard]] bool isValid() const;
    };

    struct GpuStreamDescriptors
    {
        static GpuStreamDescriptor Main;
        static GpuStreamDescriptor H2D;
        static GpuStreamDescriptor D2H;
        static GpuStreamDescriptor Cleanup;
    };

} // namespace pi::tensorlib
