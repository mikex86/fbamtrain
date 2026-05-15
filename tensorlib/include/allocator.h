#pragma once

#include <cstdlib>
#include <memory>
#include <unordered_map>
#include <vector>

namespace pi::tensorlib
{
    enum class DeviceType;
};

namespace pi::tensorlib::allocator
{
    struct AllocatorMetrics
    {
        size_t in_use_bytes{};
        size_t peak_in_use_bytes{};
        size_t cached_bytes{};
        size_t peak_cached_bytes{};
        size_t reserved_bytes{};
        size_t peak_reserved_bytes{};
        size_t num_allocations{};
        size_t num_frees{};
        size_t num_cache_hits{};
    };

    class Allocator
    {
      public:
        virtual ~Allocator() = default;

        [[nodiscard]] virtual void *allocate(size_t size, int device_ordinal, bool pinned, int stream_id, bool zero_initialize) = 0;

        virtual void deallocate(void *ptr, int device_ordinal, int stream_id) = 0;

        [[nodiscard]] virtual AllocatorMetrics getMetrics() const = 0;
    };

    class CpuAllocator final : public Allocator
    {
        enum class AllocationKind
        {
            CUDA_PINNED,
            CUDA_REGISTERED,
            MALLOC
        };

        struct AllocationInfo
        {
            AllocationKind kind;
            std::vector<void *> registered_segments;
            size_t size;
        };

        std::unordered_map<void *, AllocationInfo> allocation_kinds_;
        size_t in_use_bytes_{};
        size_t peak_in_use_bytes_{};
        size_t num_allocations_{};
        size_t num_frees_{};

      public:
        void *allocate(size_t size, int device_ordinal, bool pinned, int stream_id, bool zero_initialize) override;

        void deallocate(void *ptr, int device_ordinal, int stream_id) override;

        [[nodiscard]] AllocatorMetrics getMetrics() const override;
    };

    class CachingAllocator final : public Allocator
    {
        struct ptr_info_t
        {
            size_t size;
            bool pinned;
            int stream_id;
        };
        std::unordered_map<size_t, std::unordered_map<int, std::vector<void *>>> allocated_ptrs_{};
        std::unordered_map<void *, ptr_info_t> ptr_info_{};
        
        size_t in_use_bytes_{};
        size_t peak_in_use_bytes_{};
        size_t cached_bytes_{};
        size_t peak_cached_bytes_{};
        size_t reserved_bytes_{};
        size_t peak_reserved_bytes_{};
        size_t num_allocations_{};
        size_t num_frees_{};
        size_t num_cache_hits_{};

        Allocator &allocator_;

      public:
        explicit CachingAllocator(Allocator &allocator);

        void *allocate(size_t size, int device_ordinal, bool pinned, int stream_id, bool zero_initialize) override;

        void deallocate(void *ptr, int device_ordinal, int stream_id) override;

        [[nodiscard]] AllocatorMetrics getMetrics() const override;

    };

    // Bump allocator that hands out slices of a fixed region. Frees are ignored, so callers must
    // ensure allocations outlive the allocator (e.g., for lifetime-bound scratch regions).
    class BumpAllocator final : public Allocator
    {
        unsigned char *base_ptr_{};
        size_t size_bytes_{};
        size_t offset_bytes_{};
        size_t alignment_bytes_{};
        int device_ordinal_{};
        size_t num_allocations_{};

      public:
        BumpAllocator(void *base_ptr, size_t size_bytes, int device_ordinal, size_t alignment_bytes = 256);

        void *allocate(size_t size, int device_ordinal, bool pinned, int stream_id, bool zero_initialize) override;

        void deallocate(void *ptr, int device_ordinal, int stream_id) override;

        [[nodiscard]] AllocatorMetrics getMetrics() const override;
    };


#if PI_TENSORLIB_ENABLE_CUDA
    class CudaAllocator final : public Allocator
    {
        std::unordered_map<void *, size_t> allocation_sizes_{};
        size_t in_use_bytes_{};
        size_t peak_in_use_bytes_{};
        size_t num_allocations_{};
        size_t num_frees_{};

      public:
        CudaAllocator();

        void *allocate(size_t size, int device_ordinal, bool pinned, int stream_id, bool zero_initialize) override;

        void deallocate(void *ptr, int device_ordinal, int stream_id) override;

        [[nodiscard]] AllocatorMetrics getMetrics() const override;
    };
#endif

#if PI_TENSORLIB_ENABLE_HIP
    class HipAllocator final : public Allocator
    {
        std::unordered_map<void *, size_t> allocation_sizes_{};
        size_t in_use_bytes_{};
        size_t peak_in_use_bytes_{};
        size_t num_allocations_{};
        size_t num_frees_{};

      public:
        HipAllocator() = default;

        void *allocate(size_t size, int device_ordinal, bool pinned, int stream_id, bool zero_initialize) override;

        void deallocate(void *ptr, int device_ordinal, int stream_id) override;

        [[nodiscard]] AllocatorMetrics getMetrics() const override;
    };
#endif

    class AllocatorRegistry
    {
        std::unordered_map<DeviceType, std::unique_ptr<Allocator>> allocators_{};

      protected:
        void registerAllocator(DeviceType deviceType, std::unique_ptr<Allocator> allocator);

      public:
        Allocator &getAllocator(DeviceType deviceType) const;
    };

    class DefaultAllocatorRegistry final : public AllocatorRegistry
    {
        DefaultAllocatorRegistry();

      public:
        static DefaultAllocatorRegistry &instance();
    };

    class CachingAllocatorRegistry final : public AllocatorRegistry
    {
        CachingAllocatorRegistry();

      public:
        static CachingAllocatorRegistry &instance();
    };

    class LocalAllocatorRegistry final : public AllocatorRegistry
    {
    public:
        LocalAllocatorRegistry() = default;

        void registerAllocator(DeviceType deviceType, std::unique_ptr<Allocator> allocator);
    };

} // namespace pi::tensorlib::allocator
