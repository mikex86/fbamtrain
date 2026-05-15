#include "tensor_stream_iterator.h"

#include "op_graph.h"
#include "shape_utils.h"

#include <stdexcept>

pi::tensorlib::TensorStreamIterator::TensorStreamIterator(const TraceTensor &source_tensor, const size_t chunk_size,
                                                          const int dim, const Device target_device,
                                                          const TransferType transfer_type,
                                                          const std::optional<GpuStreamDescriptor> copy_stream_desc)
    : source_tensor_(source_tensor), chunk_size_(chunk_size), dim_(dim), target_device_(target_device),
      transfer_type_(transfer_type), total_size_(source_tensor.shape().operator[](dim))
{
    // validate transfer type
    {
        const auto [source_type, source_ord] = source_tensor.device();
        const auto [dest_type, dest_ord] = target_device_;

        if (const TransferType actual_transfer_type = TransferTypeFrom(source_type, dest_type);
            transfer_type != actual_transfer_type)
        {
            throw std::runtime_error("Incompatible transfer type specified.");
        }
    }
    if (copy_stream_desc.has_value())
    {
        copy_stream_desc_ = *copy_stream_desc;
    }
    else
    {
        switch (transfer_type_)
        {
            case TransferType::H2D:
                copy_stream_desc_ = GpuStreamDescriptors::H2D;
                break;
            case TransferType::D2H:
                copy_stream_desc_ = GpuStreamDescriptors::D2H;
                break;
            default:
                copy_stream_desc_ = GpuStreamDescriptors::Main;
                break;
        }
    }
}

void pi::tensorlib::TensorStreamIterator::prepareChunks(OpGraph &graph)
{
    if (chunk_size_ <= 0)
    {
        throw std::runtime_error("chunk_size must be positive");
    }

    if (chunks_ready_)
    {
        return;
    }
    if (chunk_size_ > total_size_ || total_size_ % chunk_size_ != 0)
    {
        throw std::runtime_error("TensorStreamIterator chunk_size must evenly divide the streamed dimension");
    }
    const uint64_t num_chunks = total_size_ / chunk_size_;
    chunks_ = source_tensor_.split(graph, num_chunks, dim_);
    chunks_ready_ = true;
}

std::optional<pi::tensorlib::TraceTensor> pi::tensorlib::TensorStreamIterator::next(OpGraph &graph)
{
    prepareChunks(graph);
    if (current_index_ >= chunks_.size())
    {
        return std::nullopt;
    }

    const TraceTensor &next_chunk = chunks_[current_index_];
    current_index_++;
    return next_chunk.to(graph, target_device_, copy_stream_desc_);
}

bool pi::tensorlib::TensorStreamIterator::nextInplace(OpGraph &graph, const TraceTensor &dst)
{
    prepareChunks(graph);
    if (current_index_ >= chunks_.size())
    {
        return false;
    }

    const TraceTensor &next_chunk = chunks_[current_index_];
    if (next_chunk.shape() != dst.shape())
    {
        throw std::runtime_error("Destination tensor does not match the streamed chunk shape.");
    }
    if (next_chunk.dtype() != dst.dtype())
    {
        throw std::runtime_error("Destination tensor dtype does not match the streamed chunk dtype.");
    }

    const auto [dest_type, dest_ord] = dst.device();
    const auto [src_type, src_ord] = next_chunk.device();
    if (const TransferType actual_transfer_type = TransferTypeFrom(src_type, dest_type);
        actual_transfer_type != transfer_type_)
    {
        throw std::runtime_error("Incompatible transfer type specified for TensorStreamIterator::nextInto.");
    }

    graph.recordOperation(OperationEntry{.type = OpType::DEVICE_COPY,
                                         .inputs = {next_chunk},
                                         .outputs = {dst},
                                         .attributes = {},
                                         .gpu_stream_desc = copy_stream_desc_});

    current_index_++;
    return true;
}
