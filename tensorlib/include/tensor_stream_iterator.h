#pragma once

#include "tensorlib.h"
#include "transfer.h"
#include "stream_descriptor.h"

#include <optional>
#include <vector>

namespace pi::tensorlib
{

    class TensorStreamIterator
    {
        const TraceTensor &source_tensor_;
        size_t chunk_size_{};
        int dim_{};
        Device target_device_{};
        TransferType transfer_type_{};
        GpuStreamDescriptor copy_stream_desc_{GpuStreamDescriptors::Main};

        size_t current_index_{0};
        size_t total_size_{0};
        bool chunks_ready_{false};
        std::vector<TraceTensor> chunks_;

        void prepareChunks(OpGraph &graph);

    public:
        /**
         * Constructs a streaming iterator for the given tensor.
         * @param source_tensor the source tensor to iterate over
         * @param chunk_size the size of each chunk to be processed
         * @param dim the index of the dimension along which to stream
         * @param target_device the target device where data is streamed to
         * @param transfer_type the type of transfer to be used; Must match what is indicated by the source tensor and the target device
         */
        TensorStreamIterator(const TraceTensor &source_tensor, size_t chunk_size, int dim, Device target_device,
                             TransferType transfer_type,
                             std::optional<GpuStreamDescriptor> copy_stream_desc);

        [[nodiscard]] std::optional<TraceTensor> next(OpGraph &graph);
        [[nodiscard]] bool nextInplace(OpGraph &graph, const TraceTensor &dst);

    };
} // namespace pi::tensorlib
