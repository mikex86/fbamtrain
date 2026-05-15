#include "tensorlib.h"

#include "allocator.h"
#include "cpudbg.h"
#include "ctx_management.h"
#include "op_graph.h"
#include "shape_utils.h"
#include "stream_utils.h"
#include "utils.h"

#include <cstdlib>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <optional>

uint64_t pi::tensorlib::Shape::operator[](int64_t index) const
{
    if (index < 0)
    {
        index += static_cast<int64_t>(dimensions_.size());
    }
    if (index < 0 || static_cast<size_t>(index) >= dimensions_.size())
    {
        throw std::out_of_range("Index out of range in Shape::operator[]");
    }
    return dimensions_[index];
}

uint64_t pi::tensorlib::Shape::numel() const
{
    uint64_t total_size = 1;
    for (const auto &dim : dimensions_)
    {
        total_size *= dim;
    }
    return total_size;
}

const std::vector<uint64_t> &pi::tensorlib::Shape::dims() const { return dimensions_; }

size_t pi::tensorlib::Shape::ndims() const { return dimensions_.size(); }

bool pi::tensorlib::Shape::operator==(const Shape &other) const { return dimensions_ == other.dimensions_; }

pi::tensorlib::Strides::Strides(const Shape &shape)
{
    const size_t n = shape.ndims();
    strides_.resize(n);
    if (n == 0)
        return;

    strides_[n - 1] = 1;

    for (size_t k = n - 1; k-- > 0;)
    {
        strides_[k] = strides_[k + 1] * shape[static_cast<int64_t>(k + 1)];
    }
}

pi::tensorlib::Strides::Strides(const std::vector<uint64_t> &strides) : strides_(strides) {}

uint64_t pi::tensorlib::Strides::operator[](const size_t index) const { return strides_[index]; }

const std::vector<uint64_t> &pi::tensorlib::Strides::strides() const { return strides_; }

static uint64_t GetNextTraceStorageId()
{
    static uint64_t next_id = 0;
    return next_id++;
}

pi::tensorlib::TraceStorage::TraceStorage(const int alloc_stream_id)
    : storage_id_(GetNextTraceStorageId()), alloc_stream_id_(alloc_stream_id), last_stream_id_(alloc_stream_id)
{
}

void pi::tensorlib::TraceStorage::validateNotFreed() const
{
    if (freed_)
    {
        throw std::runtime_error("Storage has been freed and cannot be accessed.");
    }
}

void pi::tensorlib::TraceStorage::markFreed() { freed_ = true; }

bool pi::tensorlib::TraceStorage::isFreed() const { return freed_; }

int pi::tensorlib::TraceStorage::allocatedStreamId() const { return alloc_stream_id_; }

uint64_t pi::tensorlib::TraceStorage::storageId() const { return storage_id_; }

void pi::tensorlib::TraceStorage::setLastStreamId(const int stream_id) { last_stream_id_ = stream_id; }

int pi::tensorlib::TraceStorage::lastStreamId() const { return last_stream_id_; }

std::shared_ptr<pi::tensorlib::TraceStorage> pi::tensorlib::TraceStorage::Create(const int alloc_stream_id)
{
    return std::make_shared<TraceStorage>(alloc_stream_id);
}

static uint64_t GetNextTraceTensorId()
{
    static uint64_t next_id = 0;
    return next_id++;
}

static int64_t GetNextRealTensorId()
{
    static int64_t next_id = -1;
    return next_id--;
}

namespace tensdbg
{
    static void MaybeBreak(const std::string &env_var, const uint64_t id)
    {
        static const std::optional<int64_t> break_id = [&env_var]() -> std::optional<int64_t>
        {
            const char *value = std::getenv(env_var.c_str());
            if (!value || value[0] == '\0')
            {
                return std::nullopt;
            }
            char *end = nullptr;
            const long long parsed = std::strtoll(value, &end, 10);
            if (end == value || (end != nullptr && *end != '\0'))
            {
                return std::nullopt;
            }
            return parsed;
        }();
        if (break_id.has_value() && break_id.value() == id)
        {
            cpudbg::DebugBreak();
        }
    }

    static void MaybeBreakOnTraceTensorId(const uint64_t id) { MaybeBreak("BREAK_ON_TRACE_TENSOR_ID", id); }

    static void MaybeBreakOnRealTensorId(const int64_t id) { MaybeBreak("BREAK_ON_REAL_TENSOR_ID", id); }
} // namespace tensdbg

// TraceTensor

pi::tensorlib::TraceTensor::TraceTensor(const Shape &shape, const DataType data_type, const Device device,
                                        const int alloc_stream_id, const bool pinned)
    : id_(GetNextTraceTensorId()), shape_(shape), strides_(shape), data_type_(data_type), device_(device),
      pinned_(pinned), storage_(TraceStorage::Create(alloc_stream_id))
{
    tensdbg::MaybeBreakOnTraceTensorId(id_);
}

pi::tensorlib::TraceTensor::TraceTensor(Shape shape, Strides strides, const DataType data_type, const Device device,
                                        const std::shared_ptr<TraceStorage> &storage, const bool is_view,
                                        const uint64_t storage_offset, const bool pinned)
    : is_view_(is_view), id_(GetNextTraceTensorId()), shape_(std::move(shape)), strides_(std::move(strides)),
      data_type_(data_type), device_(device), pinned_(pinned), storage_offset_(storage_offset), storage_(storage)
{
    validateNotFreed();
    tensdbg::MaybeBreakOnTraceTensorId(id_);
}

pi::tensorlib::TraceTensor pi::tensorlib::TraceTensor::Create(const std::vector<uint64_t> &dims,
                                                              const DataType data_type, const Device device,
                                                              const GpuStreamDescriptor &stream_desc,
                                                              const bool pinned)
{
    const int alloc_stream_id = stream_desc.getStreamId();
    auto t = TraceTensor{Shape(dims), data_type, device, alloc_stream_id, pinned};
    t.pinned_ = pinned;
    return t;
}

pi::tensorlib::TraceTensor pi::tensorlib::TraceTensor::view(OpGraph &graph,
                                                            const std::initializer_list<uint64_t> new_dims) const
{
    std::vector<int64_t> dims_vec{};
    dims_vec.reserve(new_dims.size());
    for (const auto dim : new_dims)
    {
        dims_vec.push_back(static_cast<int64_t>(dim));
    }
    return view(graph, dims_vec);
}

pi::tensorlib::TraceTensor pi::tensorlib::TraceTensor::viewInferred(OpGraph &graph,
                                                                    const std::initializer_list<int64_t> new_dims) const
{
    std::vector<int64_t> dims_vec{};
    dims_vec.reserve(new_dims.size());
    for (const auto dim : new_dims)
    {
        dims_vec.push_back(static_cast<int64_t>(dim));
    }
    return view(graph, dims_vec);
}

pi::tensorlib::TraceTensor pi::tensorlib::TraceTensor::view(OpGraph &graph, const std::vector<uint64_t> &new_dims) const
{
    std::vector<int64_t> dims_vec{};
    dims_vec.reserve(new_dims.size());
    for (const auto dim : new_dims)
    {
        dims_vec.push_back(static_cast<int64_t>(dim));
    }
    return view(graph, dims_vec);
}

pi::tensorlib::TraceTensor pi::tensorlib::TraceTensor::view(OpGraph &graph, const std::vector<int64_t> &new_dims) const
{
    validateNotFreed(); // validate not freed

    std::vector<uint64_t> new_dims_vec;
    new_dims_vec.reserve(new_dims.size());

    // handle -1 as the inferred dimension
    {
        const size_t num_elements = shape_.numel();
        size_t num_neg_ones = 0;
        size_t num_elements_specified = 1;
        for (const int64_t i : new_dims)
        {
            if (i == -1)
            {
                num_neg_ones++;
                if (num_neg_ones > 1)
                {
                    throw std::invalid_argument("view: only one dimension can be -1");
                }
            }
            else
            {
                num_elements_specified *= i;
            }
        }
        for (const int64_t dim : new_dims)
        {
            if (dim == -1)
            {
                new_dims_vec.push_back(num_elements / num_elements_specified);
            }
            else
            {
                new_dims_vec.push_back(static_cast<size_t>(dim));
            }
        }
    }

    Shape new_shape(new_dims_vec);
    if (shape_.numel() != new_shape.numel())
    {
        throw std::invalid_argument("view: total number of elements must not change; got " +
                                    std::to_string(shape_.numel()) + " -> " + std::to_string(new_shape.numel()));
    }

    const auto strides_opt = shape_utils::ComputeViewStrides(shape_, strides_, new_shape);
    if (!strides_opt)
    {
        throw std::invalid_argument("view: requested shape is not compatible with current strides");
    }
    const auto &strides = *strides_opt;
    auto tensor =
        TraceTensor{std::move(new_shape), strides, data_type_, device_, storage_, true, storage_offset_, pinned_};
    graph.recordOperation(OperationEntry{
        .type = OpType::VIEW,
        .inputs = {*this},
        .outputs = {tensor},
        .gpu_stream_desc = GpuStreamDescriptors::Main,
    });
    return tensor;
}
pi::tensorlib::TraceTensor pi::tensorlib::TraceTensor::transpose(OpGraph &graph,
                                                                 const std::initializer_list<int64_t> dim_indices) const
{
    validateNotFreed(); // validate not freed
    if (dim_indices.size() == 0)
    {
        throw std::invalid_argument("transpose: at least one dimension must be specified");
    }

    const auto tensor_rank = static_cast<int64_t>(shape_.ndims());
    std::vector<int64_t> dims;
    dims.reserve(dim_indices.size());
    for (int64_t dim : dim_indices)
    {
        if (dim < 0)
        {
            dim += tensor_rank;
        }
        if (dim < 0 || dim >= tensor_rank)
        {
            throw std::out_of_range("transpose: dimension index out of range");
        }
        dims.push_back(dim);
    }

    const auto &current_shape = shape_.dims();
    const auto &current_strides = strides_.strides();
    std::vector<uint64_t> new_shape_vec(current_shape.size());
    std::vector<uint64_t> new_strides_vec(current_strides.size());

    if (dims.size() == tensor_rank)
    {
        std::vector seen(static_cast<size_t>(tensor_rank), false);
        for (const auto dim : dims)
        {
            if (seen[static_cast<size_t>(dim)])
            {
                throw std::invalid_argument("transpose: duplicate dimension index specified");
            }
            seen[static_cast<size_t>(dim)] = true;
        }
        for (auto &&i : seen)
        {
            if (!i)
            {
                throw std::invalid_argument("transpose: permutation must include every dimension exactly once");
            }
        }
        for (size_t i = 0; i < dims.size(); ++i)
        {
            new_shape_vec[i] = current_shape[static_cast<size_t>(dims[i])];
            new_strides_vec[i] = current_strides[static_cast<size_t>(dims[i])];
        }
    }
    else if (dims.size() == 2)
    {
        if (dims[0] == dims[1])
        {
            throw std::invalid_argument("transpose: two identical dimensions were specified");
        }
        new_shape_vec = current_shape;
        new_strides_vec = current_strides;
        std::swap(new_shape_vec[static_cast<size_t>(dims[0])], new_shape_vec[static_cast<size_t>(dims[1])]);
        std::swap(new_strides_vec[static_cast<size_t>(dims[0])], new_strides_vec[static_cast<size_t>(dims[1])]);
    }
    else
    {
        throw std::invalid_argument("transpose: number of indices must be 2 or match tensor rank");
    }

    const Shape new_shape(new_shape_vec);
    const Strides new_strides(new_strides_vec);

    auto tensor = TraceTensor{new_shape, new_strides, data_type_, device_, storage_, true, storage_offset_, pinned_};
    graph.recordOperation(OperationEntry{
        .type = OpType::TRANSPOSE,
        .inputs = {*this},
        .outputs = {tensor},
        .gpu_stream_desc = GpuStreamDescriptors::Main,
    });
    return tensor;
}

pi::tensorlib::TraceTensor pi::tensorlib::TraceTensor::contiguous(
    OpGraph &graph, const GpuStreamDescriptor &compute_stream_descriptor) const
{
    validateNotFreed(); // validate not freed
    auto output = graph.createTensor(shape_.dims(), data_type_, device_, compute_stream_descriptor, false);
    graph.recordOperation(OperationEntry{
        .type = OpType::CONTIGUOUS,
        .inputs = {*this},
        .outputs = {output},
        .gpu_stream_desc = compute_stream_descriptor,
    });
    return output;
}

pi::tensorlib::TraceTensor pi::tensorlib::TraceTensor::slice(OpGraph &graph, int64_t dim, const uint64_t start,
                                                             const uint64_t length) const
{
    validateNotFreed();

    const size_t ndims = shape_.ndims();
    if (dim < 0)
    {
        dim += static_cast<int64_t>(ndims);
    }
    if (dim < 0 || static_cast<size_t>(dim) >= ndims)
    {
        throw std::out_of_range("slice: dimension index out of range.");
    }
    if (start + length > shape_[dim])
    {
        throw std::out_of_range("slice: requested range is out of bounds.");
    }

    std::vector<uint64_t> new_dims = shape_.dims();
    new_dims[static_cast<size_t>(dim)] = length;
    const auto &old_strides = strides_.strides();
    const uint64_t offset = storage_offset_ + start * old_strides[static_cast<size_t>(dim)];

    TraceTensor tensor{Shape(new_dims), strides_, data_type_, device_, storage_, true, offset, pinned_};
    graph.recordOperation(
        OperationEntry{.type = OpType::VIEW,
                       .inputs = {*this},
                       .outputs = {tensor},
                       .attributes = {},
                       .gpu_stream_desc = GpuStreamDescriptors::Main});
    return tensor;
}

pi::tensorlib::TraceTensor pi::tensorlib::TraceTensor::stridedSlice(OpGraph &graph, int64_t dim, const uint64_t start,
                                                                    const uint64_t length,
                                                                    const uint64_t stride_multiplier) const
{
    validateNotFreed();

    if (stride_multiplier == 0)
    {
        throw std::invalid_argument("stridedSlice: stride_multiplier must be positive.");
    }

    const size_t ndims = shape_.ndims();
    if (dim < 0)
    {
        dim += static_cast<int64_t>(ndims);
    }
    if (dim < 0 || static_cast<size_t>(dim) >= ndims)
    {
        throw std::out_of_range("stridedSlice: dimension index out of range.");
    }
    if (length == 0)
    {
        throw std::invalid_argument("stridedSlice: length must be positive.");
    }
    const uint64_t max_index = start + (length - 1) * stride_multiplier;
    if (max_index >= shape_[dim])
    {
        throw std::out_of_range("stridedSlice: requested range is out of bounds.");
    }

    std::vector<uint64_t> new_dims = shape_.dims();
    new_dims[static_cast<size_t>(dim)] = length;

    std::vector<uint64_t> new_strides_vec = strides_.strides();
    new_strides_vec[static_cast<size_t>(dim)] *= stride_multiplier;

    const auto &old_strides = strides_.strides();
    const uint64_t offset = storage_offset_ + start * old_strides[static_cast<size_t>(dim)];

    TraceTensor tensor{Shape(new_dims), Strides(new_strides_vec), data_type_, device_, storage_, true, offset, pinned_};
    graph.recordOperation(
        OperationEntry{.type = OpType::VIEW,
                       .inputs = {*this},
                       .outputs = {tensor},
                       .attributes = {},
                       .gpu_stream_desc = GpuStreamDescriptors::Main});
    return tensor;
}

pi::tensorlib::TraceTensor pi::tensorlib::TraceTensor::broadcast(OpGraph &graph, int64_t dim,
                                                                 const uint64_t new_size) const
{
    validateNotFreed();

    const size_t ndims = shape_.ndims();
    if (dim < 0)
    {
        dim += static_cast<int64_t>(ndims);
    }
    if (dim < 0 || static_cast<size_t>(dim) >= ndims)
    {
        throw std::out_of_range("broadcast: dimension index out of range.");
    }
    if (new_size == 0)
    {
        throw std::invalid_argument("broadcast: new_size must be positive.");
    }
    if (shape_[dim] != 1)
    {
        throw std::invalid_argument("broadcast: only singleton dimensions can be broadcast.");
    }

    std::vector<uint64_t> new_dims = shape_.dims();
    new_dims[static_cast<size_t>(dim)] = new_size;

    std::vector<uint64_t> new_strides_vec = strides_.strides();
    new_strides_vec[static_cast<size_t>(dim)] = 0;

    TraceTensor tensor{Shape(new_dims), Strides(new_strides_vec), data_type_, device_, storage_, true, storage_offset_,
                       pinned_};

    std::unordered_map<std::string, std::any> attributes{};
    attributes.emplace("dim", dim);
    attributes.emplace("broadcast_size", new_size);

    graph.recordOperation(
        OperationEntry{.type = OpType::VIEW,
                       .inputs = {*this},
                       .outputs = {tensor},
                       .attributes = attributes,
                       .gpu_stream_desc = GpuStreamDescriptors::Main});
    return tensor;
}

std::vector<pi::tensorlib::TraceTensor> pi::tensorlib::TraceTensor::split(OpGraph &graph, uint64_t num_splits,
                                                                          int64_t dimension) const
{
    validateNotFreed(); // validate not freed
    if (num_splits == 0)
    {
        throw std::invalid_argument("split: num_splits must be greater than 0");
    }
    if (dimension < 0)
    {
        dimension += static_cast<int64_t>(shape_.ndims());
    }
    if (dimension < 0 || dimension >= static_cast<int64_t>(shape_.ndims()))
    {
        throw std::out_of_range("split: dimension index out of range");
    }
    const uint64_t dim_size = shape_[dimension];
    if (dim_size % num_splits != 0)
    {
        throw std::invalid_argument("split: dimension size " + std::to_string(dim_size) +
                                    " is not divisible by num_splits " + std::to_string(num_splits));
    }
    const uint64_t split_size = dim_size / num_splits;

    std::vector<TraceTensor> outputs;
    outputs.reserve(num_splits);

    // create outputs
    for (uint64_t i = 0; i < num_splits; ++i)
    {
        std::vector<uint64_t> new_shape_vec = shape_.dims();
        new_shape_vec[dimension] = split_size;
        const Shape new_shape(new_shape_vec);
        const Strides new_strides = strides_; // same strides
        const auto storage_offset = storage_offset_ + i * split_size * strides_[dimension];
        auto tensor = TraceTensor{new_shape, new_strides, data_type_, device_, storage_, true, storage_offset, pinned_};
        outputs.push_back(tensor);
    }

    // attributes for the operation
    std::unordered_map<std::string, std::any> attributes{};
    attributes.reserve(2);
    attributes.emplace("num_splits", num_splits);
    attributes.emplace("dimension", dimension);

    // record operation
    graph.recordOperation(
        OperationEntry{.type = OpType::SPLIT,
                       .inputs = {*this},
                       .outputs = outputs,
                       .attributes = attributes,
                       .gpu_stream_desc = GpuStreamDescriptors::Main});

    return outputs;
}

pi::tensorlib::TraceTensor pi::tensorlib::TraceTensor::at(OpGraph &graph, int64_t dim, const uint64_t index) const
{
    validateNotFreed();

    const size_t ndims = shape_.ndims();
    if (ndims == 0)
    {
        throw std::out_of_range("Cannot index into a scalar tensor.");
    }

    if (dim < 0)
    {
        dim += static_cast<int64_t>(ndims);
    }
    if (dim < 0 || static_cast<size_t>(dim) >= ndims)
    {
        throw std::out_of_range("Dimension index out of range.");
    }

    if (index >= shape_[dim])
    {
        throw std::out_of_range("Index is out of bounds for the selected dimension.");
    }

    const size_t dim_index = static_cast<size_t>(dim);
    const auto &dims = shape_.dims();
    const auto &strides = strides_.strides();

    std::vector<uint64_t> subset_dims;
    subset_dims.reserve(ndims > 0 ? ndims - 1 : 0);
    std::vector<uint64_t> subset_strides;
    subset_strides.reserve(ndims > 0 ? ndims - 1 : 0);

    for (size_t i = 0; i < ndims; ++i)
    {
        if (i == dim_index)
        {
            continue;
        }
        subset_dims.push_back(dims[i]);
        subset_strides.push_back(strides[i]);
    }

    const uint64_t new_storage_offset = storage_offset_ + index * strides[dim_index];

    TraceTensor tensor{Shape(subset_dims), Strides(subset_strides), data_type_, device_, storage_, true,
                       new_storage_offset};

    std::unordered_map<std::string, std::any> attributes{};
    attributes.reserve(2);
    attributes.emplace("dim", dim);
    attributes.emplace("index", index);

    graph.recordOperation(OperationEntry{
        .type = OpType::AT,
        .inputs = {*this},
        .outputs = {tensor},
        .attributes = std::move(attributes),
        .gpu_stream_desc = GpuStreamDescriptors::Main});

    return tensor;
}

const pi::tensorlib::Shape &pi::tensorlib::TraceTensor::shape() const { return shape_; }

const pi::tensorlib::Strides &pi::tensorlib::TraceTensor::strides() const { return strides_; }

pi::tensorlib::DataType pi::tensorlib::TraceTensor::dtype() const { return data_type_; }

const pi::tensorlib::Device &pi::tensorlib::TraceTensor::device() const { return device_; }

std::string pi::tensorlib::TraceTensor::toString() const
{
    std::string result = "TraceTensor(shape=[";
    for (size_t i = 0; i < shape_.ndims(); ++i)
    {
        result += std::to_string(shape_[static_cast<int64_t>(i)]);
        if (i < shape_.ndims() - 1)
        {
            result += ", ";
        }
    }
    result += "], strides=[";
    for (size_t i = 0; i < strides_.strides().size(); ++i)
    {
        result += std::to_string(strides_[i]);
        if (i < strides_.strides().size() - 1)
        {
            result += ", ";
        }
    }
    result +=
        "], data_type=" + GetDataTypeName(data_type_) + ", deleted=" + (storage_->isFreed() ? "true" : "false") + ")";
    return result;
}

pi::tensorlib::TraceTensor pi::tensorlib::TraceTensor::to(OpGraph &graph, const Device device,
                                                          const GpuStreamDescriptor &stream_desc) const
{
    validateNotFreed(); // validate not freed

    if (device.device_type == device_.device_type && device.ordinal == device_.ordinal)
    {
        throw std::invalid_argument("to: target device is the same as current device");
    }
    const int stream_id = stream_desc.getStreamId();
    TraceTensor output{shape_, strides_, data_type_, device, TraceStorage::Create(stream_id), false, 0};
    graph.createTensor(output);
    graph.recordOperation(OperationEntry{.type = OpType::DEVICE_COPY,
                                         .inputs = {*this},
                                         .outputs = {output},
                                         .attributes = {},
                                         .gpu_stream_desc = stream_desc});
    return output;
}

pi::tensorlib::TraceTensor pi::tensorlib::TraceTensor::to(OpGraph &graph, const DataType data_type,
                                                          const GpuStreamDescriptor &stream_desc) const
{
    validateNotFreed();

    if (data_type == data_type_)
    {
        return *this;
    }

    const bool supported_cast =
        ((data_type_ == DataType::BFLOAT16 || data_type_ == DataType::FLOAT16) && data_type == DataType::FLOAT32) ||
        (data_type_ == DataType::FLOAT32 && (data_type == DataType::BFLOAT16 || data_type == DataType::FLOAT16)) ||
        (data_type_ == DataType::BFLOAT16 && data_type == DataType::FLOAT16) ||
        (data_type_ == DataType::FLOAT16 && data_type == DataType::BFLOAT16);
    if (!supported_cast)
    {
        throw std::invalid_argument("Unsupported cast combination requested");
    }

    TraceTensor output = graph.createTensor(shape_.dims(), data_type, device_, stream_desc,
                                            false); // TODO: support pinning here?
    graph.recordOperation(
        OperationEntry{.type = OpType::CAST,
                       .inputs = {*this},
                       .outputs = {output},
                       .attributes = {},
                       .gpu_stream_desc = stream_desc});
    return output;
}

void pi::tensorlib::TraceTensor::populate(OpGraph &graph, const TraceTensor &source,
                                          const GpuStreamDescriptor &stream_desc) const
{
    validateNotFreed(); // validate not freed

    graph.recordOperation(OperationEntry{.type = OpType::DEVICE_COPY,
                                         .inputs = {source},
                                         .outputs = {*this},
                                         .attributes = {},
                                         .gpu_stream_desc = stream_desc});
}

void pi::tensorlib::TraceTensor::free() const { storage_->markFreed(); }

void pi::tensorlib::TraceTensor::validateNotFreed() const { storage_->validateNotFreed(); }

uint64_t pi::tensorlib::TraceTensor::id() const { return id_; }
uint64_t pi::tensorlib::TraceTensor::storageOffset() const { return storage_offset_; }
uint64_t pi::tensorlib::TraceTensor::storageId() const { return storage_->storageId(); }
bool pi::tensorlib::TraceTensor::isView() const { return is_view_; }

void pi::tensorlib::TraceTensor::markRetained() const { storage_->markRetained(); }

bool pi::tensorlib::TraceTensor::retained() const { return storage_->retained(); }

int pi::tensorlib::TraceTensor::allocatedStreamId() const { return storage_->allocatedStreamId(); }

void pi::tensorlib::TraceTensor::setLastStreamId(const int stream_id) const { storage_->setLastStreamId(stream_id); }

int pi::tensorlib::TraceTensor::lastStreamId() const { return storage_->lastStreamId(); }

// Storage
static int64_t GetNextStorageId()
{
    static int64_t next_id = 0;
    return next_id++;
}

pi::tensorlib::Storage::Storage(const Device device, const size_t size_in_bytes, const bool pinned)
    : storage_id_(GetNextStorageId()), pinned_(pinned), device_(device), size_in_bytes_(size_in_bytes)
{
    if (device.device_type != DeviceType::CPU && pinned)
    {
        throw std::invalid_argument("Pinned memory is only supported for CPU device type.");
    }
}
std::shared_ptr<pi::tensorlib::Storage> pi::tensorlib::Storage::CreateFor(const DataType data_type, const Shape &shape,
                                                                          const Device device, bool pinned)
{
    size_t size_in_bytes = shape.numel() * GetDataTypeSize(data_type);
    return std::make_shared<Storage>(device, size_in_bytes, pinned);
}

#if PI_TENSORLIB_ENABLE_CUDA
#include <cuda.h>

#elif PI_TENSORLIB_ENABLE_HIP
#include <hip/hip_runtime.h>
#endif

void pi::tensorlib::Storage::initialize(void *data_ptr, allocator::Allocator *allocator)
{
    if (data_ptr_ != nullptr)
    {
        throw std::runtime_error("Storage has already been initialized with a data pointer.");
    }
    if (data_ptr == nullptr)
    {
        throw std::invalid_argument("Cannot initialize Storage with a null data pointer.");
    }
    data_ptr_ = data_ptr;
    allocator_ = allocator;
    freed_ = false; // mark as not freed
    last_stream_id_ = 0;
}

void pi::tensorlib::Storage::free()
{
    if (data_ptr_ == nullptr)
    {
        return;
    }

    if (freed_)
    {
        throw std::runtime_error("Storage has already been freed.");
    }
    freed_ = true;
    allocator_->deallocate(data_ptr_, device_.ordinal, last_stream_id_);
    data_ptr_ = nullptr;
}

void pi::tensorlib::Storage::validateNotFreed() const
{
    if (freed_)
    {
        throw std::runtime_error("Storage has been freed and cannot be accessed.");
    }
}

bool pi::tensorlib::Storage::isFreed() const { return freed_; }
void *pi::tensorlib::Storage::dataptr() const { return data_ptr_; }
const pi::tensorlib::Device &pi::tensorlib::Storage::device() const { return device_; }
size_t pi::tensorlib::Storage::sizeBytes() const { return size_in_bytes_; }
void pi::tensorlib::Storage::setLastStreamId(const int stream_id) { last_stream_id_ = stream_id; }
int pi::tensorlib::Storage::lastStreamId() const { return last_stream_id_; }
void pi::tensorlib::Storage::setAllocStreamId(const int stream_id) { alloc_stream_id_ = stream_id; }
int pi::tensorlib::Storage::allocStreamId() const { return alloc_stream_id_; }
uint64_t pi::tensorlib::Storage::storageId() const { return storage_id_; }

std::shared_ptr<pi::tensorlib::Storage> pi::tensorlib::Storage::toCPU()
{
    if (device_.device_type == DeviceType::CPU)
    {
        throw std::runtime_error("Storage is already on CPU, should not copy to CPU!");
    }
    validateNotFreed();
    if (data_ptr_ == nullptr)
    {
        throw std::runtime_error("Storage data pointer is null, cannot copy to CPU!");
    }
    const allocator::AllocatorRegistry &registry = allocator::DefaultAllocatorRegistry::instance();
    allocator::Allocator &cpu_allocator = registry.getAllocator(DeviceType::CPU);
    void *cpu_data_ptr = cpu_allocator.allocate(size_in_bytes_, 0, pinned_, 0, false);
    if (cpu_data_ptr == nullptr)
    {
        throw std::runtime_error("Failed to allocate CPU memory for Storage copy.");
    }
#if PI_TENSORLIB_ENABLE_CUDA
    if (const CUresult res = cuMemcpyDtoH(cpu_data_ptr, reinterpret_cast<CUdeviceptr>(data_ptr_), size_in_bytes_);
        res != CUDA_SUCCESS)
    {
        std::free(cpu_data_ptr);
        throw std::runtime_error("Failed to copy data from GPU to CPU.");
    }
    auto cpu_storage = std::make_shared<Storage>(Device{DeviceType::CPU, 0}, size_in_bytes_, pinned_);
    cpu_storage->initialize(cpu_data_ptr, &cpu_allocator);
    return cpu_storage;
#elif PI_TENSORLIB_ENABLE_HIP
    if (hipMemcpy(cpu_data_ptr, data_ptr_, size_in_bytes_, hipMemcpyDeviceToHost) != hipSuccess)
    {
        std::free(cpu_data_ptr);
        throw std::runtime_error("Failed to copy data from GPU to CPU.");
    }
    auto cpu_storage = std::make_shared<Storage>(Device{DeviceType::CPU, 0}, size_in_bytes_, pinned_);
    cpu_storage->initialize(cpu_data_ptr, &cpu_allocator);
    return cpu_storage;
#endif
}

void pi::tensorlib::Storage::copyFrom(const Storage &other, const GpuStreamDescriptor &stream_desc)
{
    validateNotFreed();
    other.validateNotFreed();

    if (size_in_bytes_ != other.size_in_bytes_)
    {
        throw std::invalid_argument("Storage sizes do not match for CopyFrom.");
    }
    if (data_ptr_ == nullptr || other.data_ptr_ == nullptr)
    {
        throw std::runtime_error("Cannot copy from/to uninitialized Storage data pointer.");
    }

    if (device_.device_type == DeviceType::CPU && other.device_.device_type == DeviceType::CPU)
    {
        std::memcpy(data_ptr_, other.data_ptr_, size_in_bytes_);
    }
#if PI_TENSORLIB_ENABLE_CUDA
    else if (device_.device_type == DeviceType::CPU && other.device_.device_type == DeviceType::GPU)
    {
        internal::ctxmgmt::GpuSetCurrentDevice(other.device_.ordinal);
        const auto stream_bundle = internal::ctxmgmt::GetStreamBundle(other.device_.ordinal);
        const auto stream = streamutils::GetStream(stream_bundle, stream_desc);
        const auto cu_stream = static_cast<CUstream>(stream);
        if (const CUresult res =
                cuMemcpyDtoHAsync(data_ptr_, reinterpret_cast<CUdeviceptr>(other.data_ptr_), size_in_bytes_, cu_stream);
            res != CUDA_SUCCESS)
        {
            throw std::runtime_error("Failed to copy data from GPU to CPU.");
        }
        cuStreamSynchronize(cu_stream);
    }
    else if (device_.device_type == DeviceType::GPU && other.device_.device_type == DeviceType::CPU)
    {
        internal::ctxmgmt::GpuSetCurrentDevice(device_.ordinal);
        const auto stream_bundle = internal::ctxmgmt::GetStreamBundle(device_.ordinal);
        const auto stream = streamutils::GetStream(stream_bundle, stream_desc);
        const auto cu_stream = static_cast<CUstream>(stream);
        if (const CUresult res =
                cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(data_ptr_), other.data_ptr_, size_in_bytes_, cu_stream);
            res != CUDA_SUCCESS)
        {
            throw std::runtime_error("Failed to copy data from CPU to GPU.");
        }
        cuStreamSynchronize(cu_stream);
    }
    else if (device_.device_type == DeviceType::GPU && other.device_.device_type == DeviceType::GPU)
    {
        if (device_.ordinal != other.device_.ordinal)
        {
            throw std::runtime_error("GPU-to-GPU CopyFrom across devices is not supported.");
        }
        internal::ctxmgmt::GpuSetCurrentDevice(device_.ordinal);
        const auto stream_bundle = internal::ctxmgmt::GetStreamBundle(device_.ordinal);
        const auto stream = streamutils::GetStream(stream_bundle, stream_desc);
        const auto cu_stream = static_cast<CUstream>(stream);
        if (const CUresult res =
                cuMemcpyDtoDAsync(reinterpret_cast<CUdeviceptr>(data_ptr_),
                                  reinterpret_cast<CUdeviceptr>(other.data_ptr_), size_in_bytes_, cu_stream);
            res != CUDA_SUCCESS)
        {
            throw std::runtime_error("Failed to copy data from GPU to GPU.");
        }
        cuStreamSynchronize(cu_stream);
    }
#elif PI_TENSORLIB_ENABLE_HIP
    else if (device_.device_type == DeviceType::CPU && other.device_.device_type == DeviceType::GPU)
    {
        internal::ctxmgmt::GpuSetCurrentDevice(other.device_.ordinal);
        const auto stream_bundle = internal::ctxmgmt::GetStreamBundle(other.device_.ordinal);
        const auto stream = streamutils::GetStream(stream_bundle, stream_desc);
        const auto hip_stream = static_cast<hipStream_t>(stream);
        if (hipMemcpyAsync(data_ptr_, other.data_ptr_, size_in_bytes_, hipMemcpyDeviceToHost, hip_stream) != hipSuccess)
        {
            throw std::runtime_error("Failed to copy data from GPU to CPU.");
        }
        hipStreamSynchronize(hip_stream);
    }
    else if (device_.device_type == DeviceType::GPU && other.device_.device_type == DeviceType::CPU)
    {
        internal::ctxmgmt::GpuSetCurrentDevice(device_.ordinal);
        const auto stream_bundle = internal::ctxmgmt::GetStreamBundle(device_.ordinal);
        const auto stream = streamutils::GetStream(stream_bundle, stream_desc);
        const auto hip_stream = static_cast<hipStream_t>(stream);
        if (hipMemcpyAsync(data_ptr_, other.data_ptr_, size_in_bytes_, hipMemcpyHostToDevice, hip_stream) != hipSuccess)
        {
            throw std::runtime_error("Failed to copy data from CPU to GPU.");
        }
        hipStreamSynchronize(hip_stream);
    }
    else if (device_.device_type == DeviceType::GPU && other.device_.device_type == DeviceType::GPU)
    {
        if (device_.ordinal != other.device_.ordinal)
        {
            throw std::runtime_error("GPU-to-GPU CopyFrom across devices is not supported.");
        }
        internal::ctxmgmt::GpuSetCurrentDevice(device_.ordinal);
        const auto stream_bundle = internal::ctxmgmt::GetStreamBundle(device_.ordinal);
        const auto stream = streamutils::GetStream(stream_bundle, stream_desc);
        const auto hip_stream = static_cast<hipStream_t>(stream);
        if (hipMemcpyAsync(data_ptr_, other.data_ptr_, size_in_bytes_, hipMemcpyDeviceToDevice, hip_stream) !=
            hipSuccess)
        {
            throw std::runtime_error("Failed to copy data from GPU to GPU.");
        }
        hipStreamSynchronize(hip_stream);
    }
#else
    throw std::runtime_error("CopyFrom between CPU and GPU is not supported in this build.");
#endif
}

pi::tensorlib::RealTensor::RealTensor(Shape shape, Strides strides, const DataType data_type, const Device device,
                                      const bool is_view, const uint64_t storage_offset, const bool pinned)
    : RealTensor(GetNextRealTensorId(), std::move(shape), std::move(strides), data_type, device, is_view,
                 storage_offset, pinned)
{
}

// RealTensor
pi::tensorlib::RealTensor::RealTensor(const int64_t id, Shape shape, Strides strides, const DataType data_type,
                                      const Device device, const bool is_view, const uint64_t storage_offset,
                                      const bool pinned)
    : id_(id), shape_(std::move(shape)), strides_(std::move(strides)), data_type_(data_type), device_(device),
      storage_offset_(storage_offset),
      storage_(is_view ? nullptr : Storage::CreateFor(data_type, shape_, device, pinned)), is_view_(is_view),
      pinned_(pinned)
{
    tensdbg::MaybeBreakOnRealTensorId(id_);
}

pi::tensorlib::RealTensor::~RealTensor()
{
    if (is_view_ || !storage_)
    {
        return;
    }
    if (storage_->dataptr() == nullptr || storage_->isFreed())
    {
        return;
    }
    free();
}

std::shared_ptr<pi::tensorlib::RealTensor> pi::tensorlib::RealTensor::CreateLike(const TraceTensor &trace_tensor)
{
    auto t = std::make_shared<RealTensor>(trace_tensor.id(), trace_tensor.shape(), trace_tensor.strides(),
                                          trace_tensor.dtype(), trace_tensor.device(), trace_tensor.isView(),
                                          trace_tensor.storageOffset(), trace_tensor.pinned());
    t->setRetained(trace_tensor.retained());
    return t;
}

static void *AllocateMemory(pi::tensorlib::allocator::Allocator &allocator, const pi::tensorlib::Shape &shape,
                            const pi::tensorlib::DataType data_type, const pi::tensorlib::Device &device,
                            const bool pinned, const int stream_id, const bool zero_initialize)
{
    const size_t size_in_bytes = shape.numel() * pi::tensorlib::GetDataTypeSize(data_type);
    return allocator.allocate(size_in_bytes, device.ordinal, pinned, stream_id, zero_initialize);
}

std::shared_ptr<pi::tensorlib::RealTensor>
pi::tensorlib::RealTensor::Allocate(const std::initializer_list<uint64_t> &dims, DataType data_type, Device device,
                                    const bool pinned)
{
    return AllocateOnStream(std::vector<uint64_t>(dims.begin(), dims.end()), data_type, device,
                            GpuStreamDescriptors::Main, pinned);
}

std::shared_ptr<pi::tensorlib::RealTensor>
pi::tensorlib::RealTensor::Allocate(const std::vector<uint64_t> &dims, DataType data_type, Device device,
                                    const bool pinned)
{
    return AllocateOnStream(dims, data_type, device, GpuStreamDescriptors::Main, pinned);
}

std::shared_ptr<pi::tensorlib::RealTensor>
pi::tensorlib::RealTensor::AllocateOnStream(const std::initializer_list<uint64_t> &dims, DataType data_type,
                                            Device device, const GpuStreamDescriptor &stream_desc,
                                            const bool pinned)
{
    return AllocateOnStream(std::vector<uint64_t>(dims.begin(), dims.end()), data_type, device, stream_desc, pinned);
}

std::shared_ptr<pi::tensorlib::RealTensor>
pi::tensorlib::RealTensor::AllocateOnStream(const std::vector<uint64_t> &dims, DataType data_type, Device device,
                                            const GpuStreamDescriptor &stream_desc, const bool pinned)
{
    const allocator::AllocatorRegistry &registry = allocator::DefaultAllocatorRegistry::instance();
    allocator::Allocator &allocator = registry.getAllocator(device.device_type);
    Shape shape(dims);
    auto tensor = std::make_shared<RealTensor>(shape, Strides(shape), data_type, device, false, 0, pinned);
    const int stream_id = stream_desc.getStreamId();
    void *memory =
        AllocateMemory(allocator, tensor->shape(), tensor->dtype(), tensor->device(), pinned, stream_id, false);
    tensor->storage()->initialize(memory, &allocator);
    tensor->storage()->setLastStreamId(stream_id);
    tensor->storage()->setAllocStreamId(stream_id);
    return tensor;
}

std::shared_ptr<pi::tensorlib::RealTensor> pi::tensorlib::RealTensor::to(Device device,
                                                                          const GpuStreamDescriptor &stream_desc) const
{
    if (device.device_type == device_.device_type && device.ordinal == device_.ordinal)
    {
        throw std::invalid_argument("to: target device is the same as current device");
    }

    const bool target_pinned = (device.device_type == DeviceType::CPU) ? pinned_ : false;
    auto output = std::make_shared<RealTensor>(shape_, strides_, data_type_, device, false, 0, target_pinned);
    const allocator::AllocatorRegistry &registry = allocator::DefaultAllocatorRegistry::instance();
    allocator::Allocator &allocator = registry.getAllocator(device.device_type);

    void *memory = AllocateMemory(allocator, output->shape(), output->dtype(), output->device(), target_pinned, stream_desc.getStreamId(), true);
    output->storage()->initialize(memory, &allocator);
    output->storage()->setLastStreamId(0);
    output->storage()->setAllocStreamId(0);

    // perform copy
    output->storage()->copyFrom(*storage_, stream_desc);
    const int stream_id = stream_desc.getStreamId();
    output->storage()->setLastStreamId(stream_id);
    return output;
}

void pi::tensorlib::RealTensor::setStorage(const std::shared_ptr<Storage> &storage)
{
    if (storage_ != nullptr && storage != nullptr) // we do allow re-setting to nullptr however
    {
        throw std::runtime_error("Cannot set storage on a RealTensor: storage has already been initialized.");
    }
    storage_ = storage;
}

const pi::tensorlib::Shape &pi::tensorlib::RealTensor::shape() const { return shape_; }
const pi::tensorlib::Strides &pi::tensorlib::RealTensor::strides() const { return strides_; }
pi::tensorlib::DataType pi::tensorlib::RealTensor::dtype() const { return data_type_; }
const std::shared_ptr<pi::tensorlib::Storage> &pi::tensorlib::RealTensor::storage() const { return storage_; }
const pi::tensorlib::Device &pi::tensorlib::RealTensor::device() const { return device_; }
uint64_t pi::tensorlib::RealTensor::id() const { return id_; }
bool pi::tensorlib::RealTensor::isView() const { return is_view_; }

void *pi::tensorlib::RealTensor::dataptr() const
{
    if (storage_ == nullptr)
    {
        throw std::runtime_error("RealTensor has no storage associated with it.");
    }
    return static_cast<uint8_t *>(storage_->dataptr()) + storage_offset_ * GetDataTypeSize(data_type_);
}
uint64_t pi::tensorlib::RealTensor::storageOffset() const { return storage_offset_; }

std::shared_ptr<pi::tensorlib::RealTensor> pi::tensorlib::RealTensor::at(int64_t dim, const uint64_t index) const
{
    const size_t ndims = shape_.ndims();
    if (ndims == 0)
    {
        throw std::out_of_range("Cannot index into a scalar tensor.");
    }

    if (dim < 0)
    {
        dim += static_cast<int64_t>(ndims);
    }
    if (dim < 0 || static_cast<size_t>(dim) >= ndims)
    {
        throw std::out_of_range("Dimension index out of range.");
    }

    if (index >= shape_[dim])
    {
        throw std::out_of_range("Index is out of bounds for the selected dimension.");
    }

    const size_t dim_index = static_cast<size_t>(dim);
    const auto &dims = shape_.dims();
    const auto &strides = strides_.strides();

    std::vector<uint64_t> subset_dims;
    subset_dims.reserve(ndims > 0 ? ndims - 1 : 0);
    std::vector<uint64_t> subset_strides;
    subset_strides.reserve(ndims > 0 ? ndims - 1 : 0);

    for (size_t i = 0; i < ndims; ++i)
    {
        if (i == dim_index)
        {
            continue;
        }
        subset_dims.push_back(dims[i]);
        subset_strides.push_back(strides[i]);
    }

    const uint64_t new_storage_offset = storage_offset_ + index * strides[dim_index];

    auto view = std::make_shared<RealTensor>(Shape(subset_dims), Strides(subset_strides), data_type_, device_, true,
                                             new_storage_offset);
    view->setStorage(storage_);
    return view;
}

void pi::tensorlib::RealTensor::free() const
{
    // only free if not a view
    if (is_view_)
    {
        return;
    }
    storage_->free();
}

void pi::tensorlib::RealTensor::print() const
{
    storage_->validateNotFreed();

    std::ostream &os = std::cout;

    os << "RealTensor(";

    if (storage_->dataptr() == nullptr)
    {
        os << "<uninitialized>";
        os << ", dtype=" << GetDataTypeName(data_type_) << ", device=" << DeviceTypeName(device_.device_type) << ":"
           << device_.ordinal << ")";
        return;
    }

    const auto &dims = shape_.dims();
    const auto &strides = strides_.strides();
    const size_t nd = shape_.ndims();
    const size_t elem_sz = GetDataTypeSize(data_type_);

    std::shared_ptr<Storage> storage = storage_;
    if (storage->device().device_type != DeviceType::CPU)
    {
        storage = storage->toCPU(); // will be auto-freed once out of scope
    }

    const auto *base = static_cast<uint8_t *>(storage->dataptr()) + storage_offset_ * elem_sz;

    auto print_scalar_at = [&](const uint64_t linear_index)
    {
        const unsigned char *p = base + linear_index * elem_sz;
        os.setf(static_cast<std::ios::fmtflags>(0), std::ios::floatfield);
        os << std::setprecision(6);

        switch (data_type_)
        {
            case DataType::FLOAT32:
            {
                float v;
                std::memcpy(&v, p, sizeof(float));
                os << v;
                break;
            }
            case DataType::BFLOAT16:
            {
                uint16_t bits16;
                std::memcpy(&bits16, p, sizeof(uint16_t));
                const uint32_t u32 = static_cast<uint32_t>(bits16) << 16;
                float v;
                std::memcpy(&v, &u32, sizeof(float));
                os << v;
                break;
            }
            case DataType::FLOAT16:
            {
                uint16_t bits16{};
                std::memcpy(&bits16, p, sizeof(uint16_t));
                const float v = utils::Fp32FromFp16(bits16);
                os << v;
                break;
            }
            case DataType::UINT32:
            {
                uint32_t v;
                std::memcpy(&v, p, sizeof(uint32_t));
                os << v;
                break;
            }
            case DataType::UINT64:
            {
                uint64_t v;
                std::memcpy(&v, p, sizeof(uint64_t));
                os << v;
                break;
            }
            default:
                os << "<unsupported dtype>";
        }
    };

    constexpr int INDENT_STEP = 2;
    auto indent = [&](const int n)
    {
        for (int i = 0; i < n; ++i)
            os.put(' ');
    };

    std::function<void(size_t /*dim*/, uint64_t /*base_idx*/, int /*indent*/)> rec;
    rec = [&](const size_t dim, const uint64_t base_idx, const int ind)
    {
        if (nd == 0)
        {
            // Scalar tensor: just the single value, no brackets.
            print_scalar_at(0);
            return;
        }

        os << '[';

        if (dims[dim] == 0)
        {
            // Empty dimension.
            os << ']';
            return;
        }

        for (uint64_t i = 0; i < dims[dim]; ++i)
        {
            const uint64_t child_base = base_idx + i * strides[dim];

            if (dim + 1 == nd) // last dim => print scalars inline
            {
                if (i > 0)
                    os << ", ";
                print_scalar_at(child_base);
            }
            else // inner dim => newline + recurse with indentation
            {
                if (i == 0)
                    os << '\n';
                else
                    os << ",\n";
                indent(ind + INDENT_STEP);
                rec(dim + 1, child_base, ind + INDENT_STEP);
            }
        }

        if (dim + 1 != nd)
        {
            os << '\n';
            indent(ind);
        }
        os << ']';
    };

    if (nd == 0)
    {
        // Scalar path: no brackets around data.
        print_scalar_at(0);
    }
    else
    {
        rec(0, 0, 0);
    }

    // Torch-like trailer with dtype/device (shape is handy for debugging).
    os << ", dtype=" << GetDataTypeName(data_type_) << ", device=" << DeviceTypeName(device_.device_type) << ":"
       << device_.ordinal << ")\n";
}
