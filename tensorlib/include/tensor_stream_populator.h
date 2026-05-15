#pragma once

#include "tensorlib.h"
#include "transfer.h"

#include <vector>

namespace pi::tensorlib
{
    class TensorStreamPopulator
    {
        const TraceTensor &dst_tensor_;
        int dim_{};
        TransferType transfer_type_{};
        uint64_t current_index_{0};
        uint64_t source_cursor_{0};

      public:
        TensorStreamPopulator(const TraceTensor &dst_tensor, int dim, TransferType transfer_type);

        /**
         * Writes out the src chunk to the destination tensor. interval specifies if every time step should be
         * populated. If interval=1, every time step is populated, if interval=2, every second time step is populated,
         * etc. When interval=1, the source chunk is copied directly to the destination tensor at the current index. If
         * interval>1, a strided copy is performed that may skip elements in the source tensor. The current index is
         * advanced by the amount of actual time steps populated during the call. The amount of actual time steps
         * populated is the size of the time dimension of the source chunk divided by interval.
         * When a source chunk is shorter than interval, the populator internally tracks the offset and will only
         * populate when the running cursor hits the interval boundary.
         * @param graph The operation graph to populate.
         * @param src_chunk The source chunk to copy from.
         * @param stream_desc Stream descriptor used for the copy operation.
         * @param interval The interval at which to populate the destination tensor.
         */
        void populateNext(OpGraph &graph, const TraceTensor &src_chunk, const GpuStreamDescriptor &stream_desc,
                          int interval = 1);
    };
} // namespace pi::tensorlib
