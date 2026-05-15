#pragma once

#include "tensorlib.h"

#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace pi::tensorlib::gpustream
{
    /// Opaque handle to a GPU stream.
    typedef void *GpuStream;

    /** Bundle of GPU streams for different purposes
     * - compute_stream: for general compute operations & also memory transfers
     * - h2d_stream: for host to device transfers
     * - d2h_stream: for device to host transfers
     */
    struct GpuStreamBundle
    {
        int device_ordinal{0};

        /**
         * The main stream is used for compute operations and also memory transfers.
         * Whenever memory transfers cannot be safely overlapped with compute, they should be issued on this stream.
         * Together with the compute operations, memory transfers and kernel launches for a sequentially consistent
         * execution order. Only operations that *know* compute and subsequent kernel launches can be safely overlapped
         * with memory transfers should be issued on other streams.
         */
        GpuStream main_stream;

        /**
         * Stream for host to device transfers.
         * Only issue transfers on this stream that can be safely overlapped with compute operations on the main stream.
         * The user is responsible for ensuring transfers are correctly awaited before the data is used on device.
         */
        GpuStream h2d_stream;

        /**
         * Stream for device to host transfers.
         * Only issue transfers on this stream that can be safely overlapped with compute operations on the main stream.
         * The user is responsible for ensuring transfers are correctly awaited before the data is used on host.
         */
        GpuStream d2h_stream;

        /**
         * Stream for cleanup operations that should not interfere with compute or memory transfers.
         * Some kernels may need to launch cleanup work after compute is done, e.g. to reset temporary buffers.
         * Such cleanup work should be issued on this stream to avoid delaying subsequent compute or memory transfers.
         */
        GpuStream cleanup_stream;

        /**
         * Optional additional compute streams keyed by their logical id.
         * Stream id 0 maps to main_stream.
         */
        std::unordered_map<int, GpuStream> compute_streams;
        std::unordered_map<int, int> compute_stream_priorities;

        void init();

        /**
         * Retrieve (and lazily create) the compute stream for the given id.
         * Stream id 0 always returns main_stream.
         */
        [[nodiscard]] GpuStream getComputeStream(int stream_id);

        /**
         * Get the logical id for a given compute stream.
         * @param stream the compute stream
         * @return the logical id of the stream, or std::nullopt if the stream is not known.
         */
        [[nodiscard]] std::optional<int> getStreamId(GpuStream stream) const;

        /**
         * @return set of all known compute streams, including stream 0.
         */
        [[nodiscard]] std::unordered_set<GpuStream> getAllStreams() const;

        /**
         * Set the priority for a stream id. If the stream already exists it will be recreated.
         * Stream id 0 (main stream) cannot be reprioritized through this API.
         */
        void setComputeStreamPriority(int stream_id, int priority);

    private:
        mutable std::mutex streams_mutex_{};
        
        void EnsureStreamExists(GpuStream stream, int stream_id);
    };

    /**
     * @return a new gpu stream handle
     */
    [[nodiscard]] GpuStream CreateGpuStream(int priority = 0);

    /**
     * Destroys the supplied gpu stream handle
     * @param stream the stream handle to destroy
     */
    void DestroyGpuStream(GpuStream stream);

} // namespace pi::tensorlib::gpustream
