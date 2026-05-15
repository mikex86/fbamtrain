#include "allocator.h"

#include "ctx_management.h"
#include "tensorlib.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#if PI_TENSORLIB_ENABLE_CUDA
#include <cuda.h>
#endif // PI_TENSORLIB_ENABLE_CUDA
#if PI_TENSORLIB_ENABLE_HIP
#include <hip/hip_runtime.h>
#endif // PI_TENSORLIB_ENABLE_HIP

// CpuAllocator
void *pi::tensorlib::allocator::CpuAllocator::allocate(const size_t size, const int device_ordinal, const bool pinned,
                                                       const int stream_id, const bool zero_initialize)
{
#if PI_TENSORLIB_ENABLE_CUDA
    (void)stream_id;
    if (size == 0)
        return nullptr;

    static std::once_flag cu_init_once;
    static CUresult cu_init_status = CUDA_SUCCESS;
    std::call_once(cu_init_once, [] { cu_init_status = cuInit(0); });

    if (cu_init_status != CUDA_SUCCESS)
    {
        const char *err_msg = nullptr;
        cuGetErrorString(cu_init_status, &err_msg);
        throw std::runtime_error(std::string("cuInit failed in CpuAllocator::allocate: ") +
                                 (err_msg ? err_msg : "unknown error"));
    }
    (void)device_ordinal;
    if (pinned)
    {
        CUcontext current_ctx = nullptr;
        if (const CUresult ctx_status = cuCtxGetCurrent(&current_ctx); ctx_status != CUDA_SUCCESS)
        {
            const char *err_msg = nullptr;
            cuGetErrorString(ctx_status, &err_msg);
            throw std::runtime_error(std::string("cuCtxGetCurrent failed in CpuAllocator::allocate: ") +
                                     (err_msg ? err_msg : "unknown error"));
        }
        if (current_ctx == nullptr)
        {
            throw std::runtime_error(
                "Pinned CPU allocation requires an active CUDA context. Set the target GPU device before allocation.");
        }
    }

    constexpr size_t kChunk = 1ull * 1024ull * 1024ull * 1024ull; // 1 GiB
    const bool force_chunked = size >= kChunk;                    // large allocations prefer chunked registration

    void *h = nullptr;
    if (!force_chunked && pinned)
    {
        if (const CUresult sres = cuMemAllocHost_v2(&h, size); sres == CUDA_SUCCESS)
        {
            allocation_kinds_[h] =
                AllocationInfo{.kind = AllocationKind::CUDA_PINNED, .registered_segments = {}, .size = size};
            in_use_bytes_ += size;
            peak_in_use_bytes_ = std::max(peak_in_use_bytes_, in_use_bytes_);
            num_allocations_ += 1;
            if (zero_initialize)
            {
                std::memset(h, 0, size);
            }
            return h;
        }
    }

    // fallback: allocate pageable and register to still benefit from fast transfers
    void *raw = nullptr;
    constexpr size_t kPageAlignment = 4096;
    if (posix_memalign(&raw, kPageAlignment, size) != 0 || raw == nullptr)
    {
        throw std::runtime_error("CpuAllocator::allocate failed: posix_memalign returned nullptr");
    }
    h = raw;
    if (h == nullptr)
    {
        throw std::runtime_error("CpuAllocator::allocate failed: std::malloc returned nullptr");
    }
    std::vector<void *> segments{};
    size_t offset = 0;
    while (offset < size)
    {
        const size_t chunk = std::min(kChunk, size - offset);
        void *seg = static_cast<char *>(h) + offset;
        if (pinned)
        {
            const CUresult reg_res = cuMemHostRegister(seg, chunk, 0);
            if (reg_res != CUDA_SUCCESS)
            {
                // registration failed; fall back to pageable memory
                for (void *p : segments)
                {
                    cuMemHostUnregister(p);
                }
                allocation_kinds_[h] =
                    AllocationInfo{.kind = AllocationKind::MALLOC, .registered_segments = {}, .size = size};
                in_use_bytes_ += size;
                peak_in_use_bytes_ = std::max(peak_in_use_bytes_, in_use_bytes_);
                num_allocations_ += 1;

                throw std::runtime_error("CpuAllocator::allocate failed: cuMemHostRegister returned error");
            }
            segments.push_back(seg);
        }
        offset += chunk;
    }
    allocation_kinds_[h] = AllocationInfo{.kind = pinned ? AllocationKind::CUDA_REGISTERED : AllocationKind::MALLOC,
                                          .registered_segments = segments,
                                          .size = size};
    in_use_bytes_ += size;
    peak_in_use_bytes_ = std::max(peak_in_use_bytes_, in_use_bytes_);
    num_allocations_ += 1;
    if (zero_initialize)
    {
        std::memset(h, 0, size);
    }
    return h;
#else
    (void)device_ordinal;
    (void)pinned;
    (void)stream_id;
    void *ptr = std::malloc(size);
    if (ptr)
    {
        allocation_kinds_[ptr] =
            AllocationInfo{.kind = AllocationKind::MALLOC, .registered_segments = {}, .size = size};
        in_use_bytes_ += size;
        peak_in_use_bytes_ = std::max(peak_in_use_bytes_, in_use_bytes_);
        num_allocations_ += 1;
    }
    return ptr;
#endif
}

void pi::tensorlib::allocator::CpuAllocator::deallocate(void *ptr, int device_ordinal, int stream_id)
{
#if PI_TENSORLIB_ENABLE_CUDA
    (void)stream_id;
    if (ptr)
    {
        auto it = allocation_kinds_.find(ptr);
        if (it != allocation_kinds_.end())
        {
            in_use_bytes_ -= it->second.size;
            num_frees_ += 1;
            switch (it->second.kind)
            {
                case AllocationKind::CUDA_PINNED:
                    cuMemFreeHost(ptr);
                    break;
                case AllocationKind::CUDA_REGISTERED:
                    for (void *seg : it->second.registered_segments)
                    {
                        cuMemHostUnregister(seg);
                    }
                    std::free(ptr);
                    break;
                case AllocationKind::MALLOC:
                    std::free(ptr);
                    break;
            }
            allocation_kinds_.erase(it);
        }
        else
        {
            // best effort
            cuMemFreeHost(ptr);
        }
    }
#else
    (void)device_ordinal;
    (void)stream_id;
    if (ptr)
    {
        auto it = allocation_kinds_.find(ptr);
        if (it != allocation_kinds_.end())
        {
            in_use_bytes_ -= it->second.size;
            num_frees_ += 1;
            allocation_kinds_.erase(it);
        }
    }
    std::free(ptr);
#endif
}

pi::tensorlib::allocator::AllocatorMetrics pi::tensorlib::allocator::CpuAllocator::getMetrics() const
{
    return AllocatorMetrics{.in_use_bytes = in_use_bytes_,
                            .peak_in_use_bytes = peak_in_use_bytes_,
                            .cached_bytes = 0,
                            .peak_cached_bytes = 0,
                            .reserved_bytes = in_use_bytes_,
                            .peak_reserved_bytes = peak_in_use_bytes_,
                            .num_allocations = num_allocations_,
                            .num_frees = num_frees_,
                            .num_cache_hits = 0};
}

// CachingAllocator
pi::tensorlib::allocator::CachingAllocator::CachingAllocator(Allocator &allocator) : allocator_(allocator) {}

void *pi::tensorlib::allocator::CachingAllocator::allocate(const size_t size, const int device_ordinal,
                                                           const bool pinned, const int stream_id,
                                                           const bool zero_initialize)
{
    if (const auto it = allocated_ptrs_.find(size); it != allocated_ptrs_.end())
    {
        if (auto stream_it = it->second.find(stream_id); stream_it != it->second.end())
        {
            auto &candidates = stream_it->second;
            for (size_t idx = candidates.size(); idx-- > 0;)
            {
                const auto ptr = candidates[idx];
                const auto &info = ptr_info_.at(ptr);
                if (info.pinned != pinned || info.stream_id != stream_id)
                {
                    continue; // keep this entry for a future request with matching pinned flag
                }
                candidates.erase(candidates.begin() + static_cast<long>(idx));
                cached_bytes_ -= size;
                in_use_bytes_ += size;
                peak_in_use_bytes_ = std::max(peak_in_use_bytes_, in_use_bytes_);
                num_allocations_ += 1;
                num_cache_hits_ += 1;
                return ptr;
            }
        }
    }
    void *ptr = allocator_.allocate(size, device_ordinal, pinned, stream_id, zero_initialize);
    ptr_info_[ptr] = ptr_info_t{.size = size, .pinned = pinned, .stream_id = stream_id};
    in_use_bytes_ += size;
    reserved_bytes_ += size;
    peak_in_use_bytes_ = std::max(peak_in_use_bytes_, in_use_bytes_);
    peak_reserved_bytes_ = std::max(peak_reserved_bytes_, reserved_bytes_);
    num_allocations_ += 1;
    return ptr;
}

void pi::tensorlib::allocator::CachingAllocator::deallocate(void *ptr, int device_ordinal, const int stream_id)
{
    if (ptr == nullptr)
    {
        throw std::runtime_error("CachingAllocator: Tried to deallocate nullptr");
    }

    const auto it = ptr_info_.find(ptr);
    if (it == ptr_info_.end())
    {
        throw std::runtime_error("CachingAllocator: Tried to deallocate pointer not allocated by this allocator");
    }
    const auto size = it->second.size;
    it->second.stream_id = stream_id;
    allocated_ptrs_[size][stream_id].push_back(ptr);
    in_use_bytes_ -= size;
    cached_bytes_ += size;
    peak_cached_bytes_ = std::max(peak_cached_bytes_, cached_bytes_);
    num_frees_ += 1;
}

pi::tensorlib::allocator::AllocatorMetrics pi::tensorlib::allocator::CachingAllocator::getMetrics() const
{
    return AllocatorMetrics{.in_use_bytes = in_use_bytes_,
                            .peak_in_use_bytes = peak_in_use_bytes_,
                            .cached_bytes = cached_bytes_,
                            .peak_cached_bytes = peak_cached_bytes_,
                            .reserved_bytes = reserved_bytes_,
                            .peak_reserved_bytes = peak_reserved_bytes_,
                            .num_allocations = num_allocations_,
                            .num_frees = num_frees_,
                            .num_cache_hits = num_cache_hits_};
}

// BumpAllocator
namespace
{
    size_t AlignUp(const size_t value, const size_t alignment)
    {
        if (alignment == 0)
        {
            return value;
        }
        return ((value + alignment - 1) / alignment) * alignment;
    }
} // namespace

pi::tensorlib::allocator::BumpAllocator::BumpAllocator(void *base_ptr, const size_t size_bytes,
                                                       const int device_ordinal, const size_t alignment_bytes)
    : base_ptr_(static_cast<unsigned char *>(base_ptr)), size_bytes_(size_bytes), offset_bytes_(0),
      alignment_bytes_(alignment_bytes == 0 ? 1 : alignment_bytes), device_ordinal_(device_ordinal), num_allocations_(0)
{
    if (size_bytes_ > 0 && base_ptr_ == nullptr)
    {
        throw std::invalid_argument("BumpAllocator base pointer must be non-null for non-zero size.");
    }
}

void *pi::tensorlib::allocator::BumpAllocator::allocate(const size_t size, const int device_ordinal, const bool pinned,
                                                        const int stream_id, const bool zero_initialize)
{
    (void)pinned;
    (void)stream_id;
    (void)zero_initialize;
    if (size == 0)
    {
        return nullptr;
    }
    if (device_ordinal_ >= 0 && device_ordinal != device_ordinal_)
    {
        throw std::runtime_error("BumpAllocator device ordinal mismatch.");
    }
    const size_t aligned_offset = AlignUp(offset_bytes_, alignment_bytes_);
    if (aligned_offset + size > size_bytes_)
    {
        throw std::runtime_error("BumpAllocator exhausted the supplied memory region.");
    }
    void *ptr = base_ptr_ + aligned_offset;
    offset_bytes_ = aligned_offset + size;
    num_allocations_ += 1;
    return ptr;
}

void pi::tensorlib::allocator::BumpAllocator::deallocate(void *, const int, const int)
{
    // No-op: bump allocator storage is owned externally and freed as a whole.
}

pi::tensorlib::allocator::AllocatorMetrics pi::tensorlib::allocator::BumpAllocator::getMetrics() const
{
    return AllocatorMetrics{.in_use_bytes = offset_bytes_,
                            .peak_in_use_bytes = offset_bytes_,
                            .cached_bytes = 0,
                            .peak_cached_bytes = 0,
                            .reserved_bytes = size_bytes_,
                            .peak_reserved_bytes = size_bytes_,
                            .num_allocations = num_allocations_,
                            .num_frees = 0,
                            .num_cache_hits = 0};
}

// CudaAllocator
#if PI_TENSORLIB_ENABLE_CUDA

namespace
{
    void *CudaAllocBytes(const size_t size, CUstream stream)
    {
        CUdeviceptr ptr;
        if (const CUresult result = cuMemAllocAsync(&ptr, size, stream); result != CUDA_SUCCESS)
        {
            const char *err_str = nullptr;
            if (cuGetErrorString(result, &err_str) != CUDA_SUCCESS)
            {
                err_str = "unknown error";
            }
            const std::string err_msg = "CudaAllocBytes: cuMemAlloc_v2 failed with error " + std::string(err_str);
            throw std::runtime_error(err_msg);
        }
        return reinterpret_cast<void *>(ptr);
    }

    int GetDeviceOfPointer(const void *ptr)
    {
        const auto dptr = reinterpret_cast<CUdeviceptr>(const_cast<void *>(ptr));
        int device_ordinal;
        cuPointerGetAttribute(&device_ordinal, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, dptr);
        return device_ordinal;
    }

    void CudaDeallocateBytes(void *ptr, CUstream stream)
    {
        const auto dptr = reinterpret_cast<CUdeviceptr>(ptr);
        cuMemFreeAsync(dptr, stream);
    }

} // namespace

/**
 * This is the key to make async cuda mem alloc and free really fast and effectively eliminates our need for a caching
 * allocator.
 */
static void ConfigureMemoryPool(const int device_ordinal)
{
    // By setting the release threshold to UINT64_MAX, we prevent the cuda memory pool from releasing any memory
    // back to the OS, which would be slow and require synchronization. Instead, all freed memory remains in the pool
    // for future allocations, making subsequent allocations very fast. Additionally, we disable allow internal
    // dependencies to avoid any unexpected synchronization that could arise from those features.
    CUdevice device{};
    if (cuDeviceGet(&device, device_ordinal) != CUDA_SUCCESS)
    {
        throw std::runtime_error("ConfigureMemoryPool: cuDeviceGet failed");
    }

    CUmemoryPool mempool{};
    if (cuDeviceGetDefaultMemPool(&mempool, device) != CUDA_SUCCESS)
    {
        throw std::runtime_error("ConfigureMemoryPool: cudaDeviceGetDefaultMemPool failed");
    }

    uint64_t threshold = UINT64_MAX;
    cuMemPoolSetAttribute(mempool, CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, &threshold);

    int enable = 1;

    cuMemPoolSetAttribute(mempool, CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES, &enable);
    cuMemPoolSetAttribute(mempool, CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES, &enable);
    cuMemPoolSetAttribute(mempool, CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC, &enable);
}

static void EnsureMemoryPoolConfigured(const int device_ordinal)
{
    static std::unordered_map<int, bool> memory_pool_configured_{};
    if (!memory_pool_configured_.contains(device_ordinal))
    {
        ConfigureMemoryPool(device_ordinal);
        memory_pool_configured_[device_ordinal] = true;
    }
}

pi::tensorlib::allocator::CudaAllocator::CudaAllocator() = default;

void *pi::tensorlib::allocator::CudaAllocator::allocate(const size_t size, const int device_ordinal, const bool pinned,
                                                        const int stream_id, const bool zero_initialize)
{
    EnsureMemoryPoolConfigured(device_ordinal);

    (void)stream_id;
    if (pinned)
    {
        throw std::runtime_error("Pinned allocation requested on GPU allocator; pinning is CPU-only.");
    }
    if (size == 0)
    {
        throw std::runtime_error("CudaAllocator: Tried to allocate 0 bytes");
    }

    const auto &stream_bundle = internal::ctxmgmt::GetStreamBundle(device_ordinal);
    const auto stream = stream_bundle->getComputeStream(stream_id);
    const auto cu_stream = static_cast<CUstream>(stream);

    internal::ctxmgmt::GpuSetCurrentDevice(device_ordinal);

    void *ptr = ::CudaAllocBytes(size, cu_stream);
    allocation_sizes_[ptr] = size;
    in_use_bytes_ += size;
    peak_in_use_bytes_ = std::max(peak_in_use_bytes_, in_use_bytes_);
    num_allocations_ += 1;

    if (zero_initialize)
    {
        if (const CUresult sres = cuMemsetD8Async(reinterpret_cast<CUdeviceptr>(ptr), 0, size, cu_stream);
            sres != CUDA_SUCCESS)
        {
            const char *err_msg = nullptr;
            cuGetErrorString(sres, &err_msg);
            throw std::runtime_error(std::string("cuMemsetD8Async failed in CudaAllocator::allocate: ") +
                                     (err_msg ? err_msg : "unknown error"));
        }
    }
    return ptr;
}

void pi::tensorlib::allocator::CudaAllocator::deallocate(void *ptr, const int device_ordinal, const int stream_id)
{
    (void)stream_id;
    if (!ptr)
        return;

    if (const auto it = allocation_sizes_.find(ptr); it != allocation_sizes_.end())
    {
        in_use_bytes_ -= it->second;
        num_frees_ += 1;
        allocation_sizes_.erase(it);
    }

    if (const int detected_device_ordinal = GetDeviceOfPointer(ptr); device_ordinal != detected_device_ordinal)
    {
        // user did not pass the correct device ordinal; raise error for pedanticness
        throw std::runtime_error("CudaAllocator::deallocate: device ordinal mismatch");
    }

    const auto &stream_bundle = internal::ctxmgmt::GetStreamBundle(device_ordinal);
    const auto stream = stream_bundle->getComputeStream(stream_id);
    const auto cu_stream = static_cast<CUstream>(stream);

    internal::ctxmgmt::GpuSetCurrentDevice(device_ordinal);

    CudaDeallocateBytes(ptr, cu_stream);
}

pi::tensorlib::allocator::AllocatorMetrics pi::tensorlib::allocator::CudaAllocator::getMetrics() const
{
    return AllocatorMetrics{.in_use_bytes = in_use_bytes_,
                            .peak_in_use_bytes = peak_in_use_bytes_,
                            .cached_bytes = 0,
                            .peak_cached_bytes = 0,
                            .reserved_bytes = in_use_bytes_,
                            .peak_reserved_bytes = peak_in_use_bytes_,
                            .num_allocations = num_allocations_,
                            .num_frees = num_frees_,
                            .num_cache_hits = 0};
}
#endif // PI_TENSORLIB_ENABLE_CUDA

// HipAllocator
#if PI_TENSORLIB_ENABLE_HIP
namespace
{
    void *HipAllocBytes(const size_t size)
    {
        void *ptr = nullptr;
        if (const hipError_t result = hipMalloc(&ptr, size); result != hipSuccess)
        {
            const std::string err_msg = hipGetErrorString(result);
            throw std::runtime_error("HipAllocBytes: hipMalloc failed with error " + err_msg);
        }
        return ptr;
    }

    int HipGetDeviceOfPointer(const void *ptr)
    {
        int device_ordinal = -1;
        if (ptr == nullptr)
        {
            return device_ordinal;
        }
        if (const hipError_t result = hipPointerGetAttribute(&device_ordinal, HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                                                             reinterpret_cast<hipDeviceptr_t>(const_cast<void *>(ptr)));
            result != hipSuccess)
        {
            const std::string err_msg = hipGetErrorString(result);
            throw std::runtime_error("HipGetDeviceOfPointer: hipPointerGetAttribute failed with error " + err_msg);
        }
        return device_ordinal;
    }

    void HipDeallocateBytes(void *ptr)
    {
        if (const hipError_t result = hipFree(ptr); result != hipSuccess)
        {
            const std::string err_msg = hipGetErrorString(result);
            throw std::runtime_error("HipDeallocateBytes: hipFree failed with error " + err_msg);
        }
    }
} // namespace

void *pi::tensorlib::allocator::HipAllocator::allocate(const size_t size, const int device_ordinal, const bool pinned,
                                                       const int stream_id, const bool zero_initialize)
{
    (void)stream_id;
    if (pinned)
    {
        throw std::runtime_error("Pinned allocation requested on GPU allocator; pinning is CPU-only.");
    }
    if (size == 0)
    {
        throw std::runtime_error("HipAllocator: Tried to allocate 0 bytes");
    }

    internal::ctxmgmt::GpuSetCurrentDevice(device_ordinal);

    void *ptr = ::HipAllocBytes(size);
    allocation_sizes_[ptr] = size;
    in_use_bytes_ += size;
    peak_in_use_bytes_ = std::max(peak_in_use_bytes_, in_use_bytes_);
    num_allocations_ += 1;
    if (zero_initialize)
    {
        if (const hipError_t sres = hipMemset(ptr, 0, size); sres != hipSuccess)
        {
            const std::string err_msg = hipGetErrorString(sres);
            throw std::runtime_error("hipMemset failed in HipAllocator::allocate: " + err_msg);
        }
    }
    return ptr;
}

void pi::tensorlib::allocator::HipAllocator::deallocate(void *ptr, const int device_ordinal, const int stream_id)
{
    (void)stream_id;
    if (!ptr)
    {
        return;
    }

    if (const auto it = allocation_sizes_.find(ptr); it != allocation_sizes_.end())
    {
        in_use_bytes_ -= it->second;
        num_frees_ += 1;
        allocation_sizes_.erase(it);
    }

    if (const int detected_device_ordinal = HipGetDeviceOfPointer(ptr); device_ordinal != detected_device_ordinal)
    {
        throw std::runtime_error("HipAllocator::deallocate: device ordinal mismatch");
    }

    internal::ctxmgmt::GpuSetCurrentDevice(device_ordinal);

    if (const char *env = std::getenv("FBAMTRAIN_ALLOC_SYNC_ON_FREE"); env != nullptr && env[0] != '\0')
    {
        bool should_sync = true;
        if (const char *stream_env = std::getenv("FBAMTRAIN_ALLOC_SYNC_ON_FREE_STREAM");
            stream_env != nullptr && stream_env[0] != '\0')
        {
            char *end = nullptr;
            const long parsed = std::strtol(stream_env, &end, 10);
            if (end != stream_env)
            {
                should_sync = (static_cast<int>(parsed) == stream_id);
            }
        }
        if (should_sync)
        {
            if (const hipError_t sres = hipDeviceSynchronize(); sres != hipSuccess)
            {
                const std::string err_msg = hipGetErrorString(sres);
                throw std::runtime_error("hipDeviceSynchronize failed in HipAllocator::deallocate: " + err_msg);
            }
        }
    }

    HipDeallocateBytes(ptr);
}

pi::tensorlib::allocator::AllocatorMetrics pi::tensorlib::allocator::HipAllocator::getMetrics() const
{
    return AllocatorMetrics{.in_use_bytes = in_use_bytes_,
                            .peak_in_use_bytes = peak_in_use_bytes_,
                            .cached_bytes = 0,
                            .peak_cached_bytes = 0,
                            .reserved_bytes = in_use_bytes_,
                            .peak_reserved_bytes = peak_in_use_bytes_,
                            .num_allocations = num_allocations_,
                            .num_frees = num_frees_,
                            .num_cache_hits = 0};
}
#endif // PI_TENSORLIB_ENABLE_HIP

// AllocatorRegistry
void pi::tensorlib::allocator::AllocatorRegistry::registerAllocator(DeviceType deviceType,
                                                                    std::unique_ptr<Allocator> allocator)
{
    allocators_.emplace(deviceType, std::move(allocator));
}
pi::tensorlib::allocator::Allocator &
pi::tensorlib::allocator::AllocatorRegistry::getAllocator(const DeviceType deviceType) const
{
    return *allocators_.at(deviceType);
}

// DefaultAllocatorRegistry
pi::tensorlib::allocator::DefaultAllocatorRegistry::DefaultAllocatorRegistry()
{
    registerAllocator(DeviceType::CPU, std::make_unique<CpuAllocator>());
#if PI_TENSORLIB_ENABLE_CUDA
    registerAllocator(DeviceType::GPU, std::make_unique<CudaAllocator>());
#endif
#if PI_TENSORLIB_ENABLE_HIP
    registerAllocator(DeviceType::GPU, std::make_unique<HipAllocator>());
#endif
}

pi::tensorlib::allocator::DefaultAllocatorRegistry &pi::tensorlib::allocator::DefaultAllocatorRegistry::instance()
{
    static DefaultAllocatorRegistry instance{};
    return instance;
}

// CachingAllocatorRegistry
pi::tensorlib::allocator::CachingAllocatorRegistry::CachingAllocatorRegistry()
{
    const auto &default_registry = DefaultAllocatorRegistry::instance();
    registerAllocator(DeviceType::CPU,
                      std::make_unique<CachingAllocator>(default_registry.getAllocator(DeviceType::CPU)));
#if PI_TENSORLIB_ENABLE_CUDA
    registerAllocator(DeviceType::GPU, std::make_unique<CudaAllocator>());
#endif
#if PI_TENSORLIB_ENABLE_HIP
    registerAllocator(DeviceType::GPU,
                      std::make_unique<CachingAllocator>(default_registry.getAllocator(DeviceType::GPU)));
#endif
}
pi::tensorlib::allocator::CachingAllocatorRegistry &pi::tensorlib::allocator::CachingAllocatorRegistry::instance()
{
    static CachingAllocatorRegistry instance{};
    return instance;
}

void pi::tensorlib::allocator::LocalAllocatorRegistry::registerAllocator(const DeviceType deviceType,
                                                                         std::unique_ptr<Allocator> allocator)
{
    AllocatorRegistry::registerAllocator(deviceType, std::move(allocator)); // call super
}
