#include "tensor_stream_populator.h"

#include <stdexcept>

pi::tensorlib::TensorStreamPopulator::TensorStreamPopulator(const TraceTensor &dst_tensor,
                                                            const int dim, const TransferType transfer_type)
    : dst_tensor_(dst_tensor), dim_(dim), transfer_type_(transfer_type)
{
    if (dim_ < 0 || dim_ >= static_cast<int>(dst_tensor_.shape().ndims()))
    {
        throw std::invalid_argument("TensorStreamPopulator dimension is out of bounds for the destination tensor.");
    }
}

void pi::tensorlib::TensorStreamPopulator::populateNext(OpGraph &graph, const TraceTensor &src_chunk,
                                                        const GpuStreamDescriptor &stream_desc, const int interval)
{
    // validate transfer type
    {
        const auto [source_type, source_ord] = src_chunk.device();
        const auto [dest_type, dest_ord] = dst_tensor_.device();

        if (const TransferType actual_transfer_type = TransferTypeFrom(source_type, dest_type);
            transfer_type_ != actual_transfer_type)
        {
            throw std::runtime_error("Incompatible transfer type specified.");
        }
    }

    if (interval <= 0)
    {
        throw std::invalid_argument("populateNext interval must be positive");
    }
    const auto &src_shape = src_chunk.shape();
    if (dim_ >= static_cast<int>(src_shape.ndims()))
    {
        throw std::runtime_error("Source chunk rank smaller than populator dimension.");
    }

    const uint64_t src_timesteps = src_shape[dim_];
    const uint64_t total_steps_before = source_cursor_;
    const uint64_t dst_dim_size = dst_tensor_.shape()[dim_];

    uint64_t steps_to_populate = 0;
    uint64_t offset_within_chunk = 0;
    bool have_hits = false;
    // Align checkpoints to the first timestep whose global index is divisible by interval.
    offset_within_chunk = (interval - (total_steps_before % interval)) % interval;
    if (offset_within_chunk < src_timesteps)
    {
        have_hits = true;
        const uint64_t remaining_steps_in_chunk = src_timesteps - offset_within_chunk;
        steps_to_populate = 1 + (remaining_steps_in_chunk - 1) / static_cast<uint64_t>(interval);
    }

    if (current_index_ + steps_to_populate > dst_dim_size)
    {
        throw std::runtime_error("Destination tensor does not have enough remaining space.");
    }

    if (have_hits && steps_to_populate > 0)
    {
        const TraceTensor dst_slice =
            dst_tensor_.slice(graph, dim_, current_index_, steps_to_populate);
        if (dst_slice.dtype() != src_chunk.dtype())
        {
            throw std::runtime_error("Source and destination chunk dtype mismatch.");
        }

        const TraceTensor src_view =
            (interval == 1 && offset_within_chunk == 0)
                ? src_chunk
                : src_chunk.stridedSlice(graph, dim_, /*start=*/offset_within_chunk, steps_to_populate,
                                         static_cast<uint64_t>(interval));

        if (dst_slice.shape() != src_view.shape())
        {
            throw std::runtime_error("Source and destination chunk metadata mismatch.");
        }
        dst_slice.populate(graph, src_view, stream_desc);
        current_index_ += static_cast<int>(steps_to_populate);
    }

    source_cursor_ += src_timesteps;
}
