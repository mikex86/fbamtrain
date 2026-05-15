#pragma once

#include "allocator.h"

#include <memory>
#include <span>
#include <vector>

namespace pi::tensorlib
{
    // Forward declaration
    class OpGraph;
    struct GpuStreamDescriptor;

    enum class DeviceType
    {
        CPU,
        GPU,
    };

    struct Device
    {
        DeviceType device_type;
        int ordinal; // e.g., GPU id

        bool operator==(const Device &other) const
        {
            return device_type == other.device_type && ordinal == other.ordinal;
        }
    };

    enum class DataType
    {
        // TODO: Add more data types as needed
        UINT32,
        UINT64,
        BFLOAT16,
        FLOAT16,
        FLOAT32,
    };

    class Shape
    {
        std::vector<uint64_t> dimensions_;

      public:
        explicit Shape(const std::vector<uint64_t> &dims) : dimensions_(dims) {}

        // override operator[] to access dimensions
        [[nodiscard]] uint64_t operator[](int64_t index) const;

        [[nodiscard]] uint64_t numel() const;

        [[nodiscard]] const std::vector<uint64_t> &dims() const;

        [[nodiscard]] size_t ndims() const;

        // equality
        bool operator==(const Shape &other) const;
    };

    class Strides
    {
        std::vector<uint64_t> strides_;

      public:
        Strides() = default;

        explicit Strides(const Shape &shape);

        explicit Strides(const std::vector<uint64_t> &strides);

        [[nodiscard]] uint64_t operator[](size_t index) const;

        [[nodiscard]] const std::vector<uint64_t> &strides() const;
    };

    class TraceStorage
    {
        uint64_t storage_id_{0};
        int alloc_stream_id_{0};
        int last_stream_id_{0};
        bool freed_{false};

        /// Marks storage whose tensors must survive beyond the lifetime of this graph (e.g., used by other graphs).
        bool retained_{false};

      public:
        // Constructor is considered private; use Create factory method.
        explicit TraceStorage(int alloc_stream_id);

        void validateNotFreed() const;

        void markFreed();

        [[nodiscard]] bool isFreed() const;

        void markRetained() { retained_ = true; }

        [[nodiscard]] bool retained() const { return retained_; }

        [[nodiscard]] int allocatedStreamId() const;
        [[nodiscard]] uint64_t storageId() const;
        void setLastStreamId(int stream_id);
        [[nodiscard]] int lastStreamId() const;

        static std::shared_ptr<TraceStorage> Create(int alloc_stream_id);
    };

    class TraceTensor
    {
        bool is_view_{false};
        uint64_t id_;
        Shape shape_;
        Strides strides_;
        DataType data_type_{};
        Device device_{};
        bool pinned_{false};

        /// storage offset in number of elements
        uint64_t storage_offset_{0};
        std::shared_ptr<TraceStorage> storage_;

        TraceTensor(const Shape &shape, DataType data_type, Device device, int alloc_stream_id, bool pinned = false);

        TraceTensor(Shape shape, Strides strides, DataType data_type, Device device,
                    const std::shared_ptr<TraceStorage> &storage, bool is_view, uint64_t storage_offset,
                    bool pinned = false);

      public:
        static TraceTensor Create(const std::vector<uint64_t> &dims, DataType data_type, Device device,
                                  const GpuStreamDescriptor &stream_desc, bool pinned = false);

        [[nodiscard]] TraceTensor view(OpGraph &graph, std::initializer_list<uint64_t> new_dims) const;

        [[nodiscard]] TraceTensor viewInferred(OpGraph &graph, std::initializer_list<int64_t> new_dims) const;

        [[nodiscard]] TraceTensor view(OpGraph &graph, const std::vector<uint64_t> &new_dims) const;

        [[nodiscard]] TraceTensor view(OpGraph &graph, const std::vector<int64_t> &new_dims) const;

        [[nodiscard]] TraceTensor transpose(OpGraph &graph, std::initializer_list<int64_t> dim_indices) const;

        [[nodiscard]] TraceTensor contiguous(OpGraph &graph,
                                             const GpuStreamDescriptor &compute_stream_descriptor) const;

        /// Returns a view slicing `length` elements along `dim` starting at `start`.
        [[nodiscard]] TraceTensor slice(OpGraph &graph, int64_t dim, uint64_t start, uint64_t length) const;

        /// Returns a view slicing `length` elements along `dim` starting at `start` with stride multiplier.
        [[nodiscard]] TraceTensor stridedSlice(OpGraph &graph, int64_t dim, uint64_t start, uint64_t length,
                                               uint64_t stride_multiplier) const;

        /// Broadcasts along `dim`, returning a view with the same underlying storage and zero stride on that dim.
        /// The size of the broadcast dimension must be 1 in the source tensor.
        [[nodiscard]] TraceTensor broadcast(OpGraph &graph, int64_t dim, uint64_t new_size) const;

        [[nodiscard]] std::vector<TraceTensor> split(OpGraph &graph, uint64_t num_splits, int64_t dimension) const;

        [[nodiscard]] TraceTensor at(OpGraph &graph, int64_t dim, uint64_t index) const;

        [[nodiscard]] const Shape &shape() const;

        [[nodiscard]] const Strides &strides() const;

        [[nodiscard]] const Device &device() const;

        [[nodiscard]] DataType dtype() const;

        [[nodiscard]] bool pinned() const { return pinned_; }

        [[nodiscard]] std::string toString() const;

        friend std::ostream &operator<<(std::ostream &os, const TraceTensor &tensor) { return os << tensor.toString(); }

        /// Copies this tensor to the specified device.
        [[nodiscard]] TraceTensor to(OpGraph &graph, Device device, const GpuStreamDescriptor &stream_desc) const;

        /// Casts this tensor to the specified data type.
        [[nodiscard]] TraceTensor to(OpGraph &graph, DataType data_type, const GpuStreamDescriptor &stream_desc) const;

        /// (Inplace) populates this tensor with contents from the provided source tensor.
        void populate(OpGraph &graph, const TraceTensor &source, const GpuStreamDescriptor &stream_desc) const;

        /**
         * Called to free the storage associated with this tensor.
         * This should not be called directly by users, but instead through
         * OpGraph::deleteTensor. It is public to allow the OpGraph to access it.
         */
        void free() const;

        void validateNotFreed() const;

        [[nodiscard]] uint64_t id() const;

        [[nodiscard]] uint64_t storageOffset() const;

        [[nodiscard]] bool isView() const;

        void markRetained() const;

        [[nodiscard]] bool retained() const;

        [[nodiscard]] int allocatedStreamId() const;
        [[nodiscard]] uint64_t storageId() const;
        void setLastStreamId(int stream_id) const;
        [[nodiscard]] int lastStreamId() const;
    };

    class Storage
    {
        uint64_t storage_id_{};
        bool freed_{false};
        bool pinned_{false};
        Device device_{};
        size_t size_in_bytes_;
        void *data_ptr_{nullptr};
        allocator::Allocator *allocator_ = nullptr;
        int last_stream_id_{0};
        int alloc_stream_id_{0};

      public:
        explicit Storage(Device device, size_t size_in_bytes, bool pinned);

        [[nodiscard]] std::shared_ptr<Storage> static CreateFor(DataType data_type, const Shape &shape, Device device,
                                                                bool pinned);

        void initialize(void *data_ptr, allocator::Allocator *allocator);
        void free();
        void validateNotFreed() const;

        void copyFrom(const Storage &other, const GpuStreamDescriptor &stream_desc);

        [[nodiscard]] bool isFreed() const;
        [[nodiscard]] void *dataptr() const;
        [[nodiscard]] const Device &device() const;
        [[nodiscard]] size_t sizeBytes() const;
        void setLastStreamId(int stream_id);
        [[nodiscard]] int lastStreamId() const;
        void setAllocStreamId(int stream_id);
        [[nodiscard]] int allocStreamId() const;
        [[nodiscard]] uint64_t storageId() const;

        [[nodiscard]] std::shared_ptr<Storage> toCPU();
    };

    class RealTensor
    {
        int64_t id_;
        Shape shape_;
        Strides strides_;
        DataType data_type_{};
        Device device_{};
        /// storage offset in number of elements
        uint64_t storage_offset_{0};
        std::shared_ptr<Storage> storage_;
        bool is_view_{false};
        bool pinned_{true};

        // Mirrors TraceTensor::retained(): tensors marked retained must not be freed by plan-level cleanup.
        bool retained_{false};
        bool has_alloc_event_{false};

      public:
        RealTensor(Shape shape, Strides strides, DataType data_type, Device device, bool is_view,
                   uint64_t storage_offset, bool pinned = false);

        RealTensor(int64_t id, Shape shape, Strides strides, DataType data_type, Device device, bool is_view,
                   uint64_t storage_offset, bool pinned = false);

        ~RealTensor();

        static std::shared_ptr<RealTensor> CreateLike(const TraceTensor &trace_tensor);

        static std::shared_ptr<RealTensor> Allocate(const std::initializer_list<uint64_t> &dims, DataType data_type,
                                                    Device device, bool pinned = false);

        static std::shared_ptr<RealTensor> Allocate(const std::vector<uint64_t> &dims, DataType data_type,
                                                    Device device, bool pinned = false);

        static std::shared_ptr<RealTensor> AllocateOnStream(const std::initializer_list<uint64_t> &dims,
                                                            DataType data_type, Device device,
                                                            const GpuStreamDescriptor &stream_desc,
                                                            bool pinned);

        static std::shared_ptr<RealTensor> AllocateOnStream(const std::vector<uint64_t> &dims, DataType data_type,
                                                            Device device, const GpuStreamDescriptor &stream_desc,
                                                            bool pinned);

        [[nodiscard]] std::shared_ptr<RealTensor> to(Device device, const GpuStreamDescriptor &stream_desc) const;

        void setStorage(const std::shared_ptr<Storage> &storage);

        [[nodiscard]] const Shape &shape() const;
        [[nodiscard]] const Strides &strides() const;
        [[nodiscard]] DataType dtype() const;
        [[nodiscard]] const std::shared_ptr<Storage> &storage() const;
        [[nodiscard]] const Device &device() const;
        [[nodiscard]] bool pinned() const { return pinned_; }
        [[nodiscard]] uint64_t id() const;
        [[nodiscard]] bool isView() const;
        [[nodiscard]] void *dataptr() const;
        [[nodiscard]] uint64_t storageOffset() const;
        [[nodiscard]] std::shared_ptr<RealTensor> at(int64_t dim, uint64_t index) const;

        void print() const;

        void free() const;

        void setRetained(const bool retained) { retained_ = retained; }

        [[nodiscard]] bool retained() const { return retained_; }
    };

    inline std::string GetDataTypeName(const DataType data_type)
    {
        switch (data_type)
        {
            case DataType::UINT32:
                return "uint32";
            case DataType::UINT64:
                return "uint64";
            case DataType::BFLOAT16:
                return "bfloat16";
            case DataType::FLOAT16:
                return "float16";
            case DataType::FLOAT32:
                return "float32";
            default:
                return "unknown";
        }
    }

    inline std::string DeviceTypeName(const DeviceType device_type)
    {
        switch (device_type)
        {
            case DeviceType::CPU:
                return "CPU";
            case DeviceType::GPU:
                return "GPU";
            default:
                return "unknown";
        }
    }

    inline size_t GetDataTypeSize(const DataType data_type)
    {
        switch (data_type)
        {
            case DataType::BFLOAT16:
            case DataType::FLOAT16:
                return 2;
            case DataType::FLOAT32:
            case DataType::UINT32:
                return 4;
            case DataType::UINT64:
                return 8;
            default:
                throw std::runtime_error("Unsupported data type");
        }
    }
} // namespace pi::tensorlib
