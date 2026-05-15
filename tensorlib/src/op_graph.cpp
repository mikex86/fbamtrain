#include "op_graph.h"

#include "ctx_management.h"
#include "transfer.h"

#include <cstdlib>
#include <optional>
#include <unordered_map>
#include <unordered_set>

namespace
{
    bool IsAliasOp(const pi::tensorlib::OpType type)
    {
        switch (type)
        {
            case pi::tensorlib::OpType::VIEW:
            case pi::tensorlib::OpType::TRANSPOSE:
            case pi::tensorlib::OpType::SPLIT:
            case pi::tensorlib::OpType::AT:
                return true;
            default:
                return false;
        }
    }

    bool IsStorageTouchingOp(const pi::tensorlib::OperationEntry &entry)
    {
        if (!entry.gpu_stream_desc.isValid())
        {
            return false;
        }
        if (entry.type == pi::tensorlib::OpType::DELETE_TENSOR || entry.type == pi::tensorlib::OpType::RECORD_EVENT ||
            entry.type == pi::tensorlib::OpType::AWAIT_EVENT ||
            entry.type == pi::tensorlib::OpType::BEGIN_GPUTX_RANGE ||
            entry.type == pi::tensorlib::OpType::END_GPUTX_RANGE)
        {
            return false;
        }
        return !IsAliasOp(entry.type);
    }

} // namespace

pi::tensorlib::OpGraph::OpGraph(const std::vector<GraphInputDescriptor> &input_descriptors,
                                const std::vector<GraphInputDescriptor> &parameter_descriptors)
    : input_descriptors_(input_descriptors), parameter_descriptors_(parameter_descriptors)
{
    for (const auto &entry : input_descriptors_)
    {
        if (!entry.tensor.retained())
        {
            throw std::runtime_error("Graph input '" + entry.name + "' must be marked retained.");
        }
    }
    for (const auto &entry : parameter_descriptors_)
    {
        if (!entry.tensor.retained())
        {
            throw std::runtime_error("Graph parameter '" + entry.name + "' must be marked retained.");
        }
    }
}

void pi::tensorlib::OpGraph::recordOperation(const OperationEntry &entry)
{
    // ensure that the inputs & outputs are not freed
    {
        for (const auto &input : entry.inputs)
        {
            input.validateNotFreed();
        }
        for (const auto &output : entry.outputs)
        {
            output.validateNotFreed();
        }
    }

    if (IsStorageTouchingOp(entry))
    {
        const int op_stream_id = entry.gpu_stream_desc.getStreamId();
        auto maybe_update_last_stream = [op_stream_id](const TraceTensor &tensor)
        {
            if (tensor.device().device_type != DeviceType::GPU)
            {
                return;
            }
            tensor.setLastStreamId(op_stream_id);
        };
        for (const auto &input : entry.inputs)
        {
            maybe_update_last_stream(input);
        }
        for (const auto &output : entry.outputs)
        {
            maybe_update_last_stream(output);
        }
    }

    // record the operation
    entries_.push_back(entry);
}

pi::tensorlib::TraceTensor pi::tensorlib::OpGraph::createTensor(const std::vector<uint64_t> &dims,
                                                                const DataType data_type, const Device device,
                                                                const GpuStreamDescriptor &stream_desc,
                                                                const bool pinned)
{
    auto tensor = TraceTensor::Create(dims, data_type, device, stream_desc, pinned);
    createTensor(tensor);
    return tensor;
}

void pi::tensorlib::OpGraph::createTensor(TraceTensor &tensor)
{
    if (created_tensor_ids_.contains(tensor.id()))
    {
        throw std::runtime_error("Tensor with id " + std::to_string(tensor.id()) + " is already defined in the graph.");
    }
    const int alloc_stream_id = tensor.allocatedStreamId();
    recordOperation(OperationEntry{.type = OpType::CREATE_TENSOR,
                                   .inputs = {},
                                   .outputs = {tensor},
                                   .attributes = {},
                                   .gpu_stream_desc = GpuStreamDescriptor{StreamKind::Compute, alloc_stream_id}});
    created_tensor_ids_.insert(tensor.id());
}

void pi::tensorlib::OpGraph::deleteTensor(TraceTensor &tensor)
{
    tensor.validateNotFreed();

    const int last_stream_id = tensor.lastStreamId();
    const auto delete_stream_desc = GpuStreamDescriptor{StreamKind::Compute, last_stream_id};
    recordOperation(OperationEntry{
        .type = OpType::DELETE_TENSOR,
        .inputs = {tensor},
        .outputs = {},
        .gpu_stream_desc = delete_stream_desc,
    });
    tensor.free();
    created_tensor_ids_.erase(tensor.id());
}

bool pi::tensorlib::OpGraph::hasTensor(const uint64_t id) const { return created_tensor_ids_.contains(id); }

// GpuStreamDescriptors

pi::tensorlib::GpuStreamDescriptor pi::tensorlib::GpuStreamDescriptors::Main = GpuStreamDescriptor{
    .kind = StreamKind::Main,
    .compute_stream_id = std::nullopt,
};

pi::tensorlib::GpuStreamDescriptor pi::tensorlib::GpuStreamDescriptors::H2D = GpuStreamDescriptor{
    .kind = StreamKind::H2D,
    .compute_stream_id = std::nullopt,
};

pi::tensorlib::GpuStreamDescriptor pi::tensorlib::GpuStreamDescriptors::D2H = GpuStreamDescriptor{
    .kind = StreamKind::D2H,
    .compute_stream_id = std::nullopt,
};

pi::tensorlib::GpuStreamDescriptor pi::tensorlib::GpuStreamDescriptors::Cleanup = GpuStreamDescriptor{
    .kind = StreamKind::Cleanup,
    .compute_stream_id = std::nullopt,
};

int pi::tensorlib::GpuStreamDescriptor::getStreamId() const
{
    switch (kind)
    {
        case StreamKind::Main:
            return 0;
        case StreamKind::Compute:
            if (!compute_stream_id.has_value())
            {
                throw std::runtime_error("Compute stream must have a valid compute_stream_id.");
            }
            return compute_stream_id.value();
        case StreamKind::H2D:
            return -1;
        case StreamKind::D2H:
            return -2;
        case StreamKind::Cleanup:
            return -3;
        default:
            throw std::runtime_error("Unknown stream kind.");
    }
}
bool pi::tensorlib::GpuStreamDescriptor::isValid() const
{
    switch (kind)
    {
        case StreamKind::Main:
        case StreamKind::H2D:
        case StreamKind::D2H:
        case StreamKind::Cleanup:
            if (compute_stream_id.has_value())
            {
                // compute_stream_id should not be specified when targeting common streams
                return false;
            }
            return true;
        case StreamKind::Compute:
            return compute_stream_id.has_value();
        default:
            return false;
    }
}

pi::tensorlib::OpGraphGpuTxRange::OpGraphGpuTxRange(OpGraph &parent_graph, const std::string &range_name)
    : parent_graph(parent_graph), range_name(range_name)
{
    parent_graph.recordOperation(OperationEntry{
        .type = OpType::BEGIN_GPUTX_RANGE,
        .inputs = {},
        .outputs = {},
        .attributes = {{"range_name", range_name}},
        .gpu_stream_desc = GpuStreamDescriptors::Main,
    });
}

pi::tensorlib::OpGraphGpuTxRange pi::tensorlib::OpGraphGpuTxRange::Start(OpGraph &parent_graph,
                                                                         const std::string &range_name)
{
    return {parent_graph, range_name};
}

pi::tensorlib::OpGraphGpuTxRange::~OpGraphGpuTxRange()
{
    parent_graph.recordOperation(OperationEntry{
        .type = OpType::END_GPUTX_RANGE,
        .inputs = {},
        .outputs = {},
        .attributes = {{"range_name", range_name}},
        .gpu_stream_desc = GpuStreamDescriptors::Main,
    });
}

pi::tensorlib::OpGraphGpuTxRange pi::tensorlib::OpGraph::createGpuTxRange(const std::string &range_name)
{
    return OpGraphGpuTxRange::Start(*this, range_name);
}

void pi::tensorlib::OpGraph::finalize()
{
    // checks if all inputs declared by entries are either defined
    // by input_descriptors_, parameter_descriptors_ or previous operations
    std::unordered_set<uint64_t> defined_inputs{};
    for (const auto &[name, tensor] : input_descriptors_)
    {
        defined_inputs.insert(tensor.id());
    }
    for (const auto &[name, tensor] : parameter_descriptors_)
    {
        defined_inputs.insert(tensor.id());
    }
    for (const auto &entry : entries_)
    {
        for (const auto &input : entry.inputs)
        {
            if (!defined_inputs.contains(input.id()))
            {
                throw std::runtime_error("Input tensor with id " + std::to_string(input.id()) +
                                         " is not defined in the graph.");
            }
        }
        for (const auto &output : entry.outputs)
        {
            defined_inputs.insert(output.id());
        }

        // int stream_id = GetOperationStreamId(entry);
        // TODO:
    }
}

pi::tensorlib::OpGraph::GpuEventHandle pi::tensorlib::OpGraph::createGpuEvent(const Device &device)
{
    if (device.device_type != DeviceType::GPU)
    {
        throw std::runtime_error("GpuEvent can only be created for GPU devices.");
    }

    const GpuEventHandle handle{event_alive_.size(), device.ordinal};
    event_alive_.push_back(true);
    return handle;
}

void pi::tensorlib::OpGraph::deleteGpuEvent(const GpuEventHandle &handle)
{
    if (handle.id >= event_alive_.size() || !event_alive_[handle.id])
    {
        throw std::runtime_error("Attempted to delete unknown or already deleted event handle");
    }
    event_alive_[handle.id] = false;
}

void pi::tensorlib::OpGraph::recordGpuEvent(const GpuEventHandle &handle, const GpuStreamDescriptor &stream_desc)
{
    if (handle.id >= event_alive_.size() || !event_alive_[handle.id])
    {
        throw std::runtime_error("Attempted to record an unknown or deleted event handle");
    }
    if (!stream_desc.isValid())
    {
        throw std::runtime_error("Invalid stream descriptor provided for recording GPU event");
    }
    recordOperation(
        OperationEntry{.type = OpType::RECORD_EVENT,
                       .inputs = {},
                       .outputs = {},
                       .attributes = {{"event_handle", handle.id}, {"device_ordinal", handle.device_ordinal}},
                       .gpu_stream_desc = stream_desc});
}

void pi::tensorlib::OpGraph::awaitGpuEvent(const GpuEventHandle &handle, const GpuStreamDescriptor &stream_desc)
{
    if (handle.id >= event_alive_.size() || !event_alive_[handle.id])
    {
        throw std::runtime_error("Attempted to await an unknown or deleted event handle");
    }
    if (!stream_desc.isValid())
    {
        throw std::runtime_error("Invalid stream descriptor provided for awaiting GPU event");
    }
    recordOperation(OperationEntry{.type = OpType::AWAIT_EVENT,
                                   .inputs = {},
                                   .outputs = {},
                                   .attributes = {{"event_handle", handle.id}},
                                   .gpu_stream_desc = stream_desc});
}

const std::vector<pi::tensorlib::GraphInputDescriptor> &pi::tensorlib::OpGraph::getInputDescriptors() const
{
    return input_descriptors_;
}

const std::vector<pi::tensorlib::GraphInputDescriptor> &pi::tensorlib::OpGraph::getParameterDescriptors() const
{
    return parameter_descriptors_;
}

const std::vector<pi::tensorlib::OperationEntry> &pi::tensorlib::OpGraph::getEntries() const { return entries_; }
