#include "gpu_stream.h"
#include "ctx_management.h"

#include <ranges>
#include <vector>

#include <stdexcept>
#include <string>
#include <unordered_set>

#if PI_TENSORLIB_ENABLE_CUDA
#include <cuda.h>
#endif

#if PI_TENSORLIB_ENABLE_HIP
#include <hip/hip_runtime.h>
#endif

namespace
{
    int ClampPriority(const int priority)
    {
#if PI_TENSORLIB_ENABLE_CUDA
        int least_priority{};
        int greatest_priority{};
        if (const CUresult range_res = cuCtxGetStreamPriorityRange(&least_priority, &greatest_priority);
            range_res != CUDA_SUCCESS)
        {
            throw std::runtime_error("Failed to query CUDA stream priority range: " + std::to_string(range_res));
        }
        const int clamped = std::max(greatest_priority, std::min(priority, least_priority));
        return clamped;
#elif PI_TENSORLIB_ENABLE_HIP
        int least_priority{};
        int greatest_priority{};
        if (const hipError_t range_res = hipDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);
            range_res != hipSuccess)
        {
            throw std::runtime_error("Failed to query HIP stream priority range: " + std::to_string(range_res));
        }
        const int clamped = std::max(greatest_priority, std::min(priority, least_priority));
        return clamped;
#else
        (void)priority;
        return 0;
#endif
    }
} // namespace

pi::tensorlib::gpustream::GpuStream pi::tensorlib::gpustream::CreateGpuStream(const int priority)
{
#if PI_TENSORLIB_ENABLE_CUDA
    CUstream stream{};
    const int clamped_priority = ClampPriority(priority);
    if (const CUresult result = cuStreamCreateWithPriority(&stream, CU_STREAM_NON_BLOCKING, clamped_priority);
        result != CUDA_SUCCESS)
    {
        throw std::runtime_error("Failed to create CUDA stream: " + std::to_string(result));
    }
    return stream;
#elif PI_TENSORLIB_ENABLE_HIP
    hipStream_t stream{};
    const int clamped_priority = ClampPriority(priority);
    if (const hipError_t result = hipStreamCreateWithPriority(&stream, hipStreamNonBlocking, clamped_priority);
        result != hipSuccess)
    {
        throw std::runtime_error("Failed to create HIP stream: " + std::to_string(result));
    }
    return stream;
#else
    throw std::runtime_error("No GPU backend enabled for stream creation.");
#endif
}

void pi::tensorlib::gpustream::DestroyGpuStream(GpuStream stream)
{
#if PI_TENSORLIB_ENABLE_CUDA
    const auto cuStream = static_cast<CUstream>(stream);
    if (const CUresult result = cuStreamDestroy(cuStream); result != CUDA_SUCCESS)
    {
        throw std::runtime_error("Failed to destroy CUDA stream: " + std::to_string(result));
    }
#elif PI_TENSORLIB_ENABLE_HIP
    const auto hipStream = static_cast<hipStream_t>(stream);
    if (const hipError_t result = hipStreamDestroy(hipStream); result != hipSuccess)
    {
        throw std::runtime_error("Failed to destroy HIP stream: " + std::to_string(result));
    }
#else
    throw std::runtime_error("No GPU backend enabled for stream destruction.");
#endif
}

void pi::tensorlib::gpustream::GpuStreamBundle::init()
{
    std::lock_guard<std::mutex> lock(streams_mutex_);
    EnsureStreamExists(main_stream, 0);
    EnsureStreamExists(h2d_stream, -1);
    EnsureStreamExists(d2h_stream, -2);
    EnsureStreamExists(cleanup_stream, -3);
}

void pi::tensorlib::gpustream::GpuStreamBundle::EnsureStreamExists(GpuStream stream, const int stream_id)
{
    if (stream == nullptr)
    {
        throw std::runtime_error("Stream with ID " + std::to_string(stream_id) + " does not exist.");
    }
    if (compute_streams.contains(stream_id))
    {
        return;
    }
    compute_streams.emplace(stream_id, stream);
}

pi::tensorlib::gpustream::GpuStream pi::tensorlib::gpustream::GpuStreamBundle::getComputeStream(const int stream_id)
{
    if (stream_id == 0)
    {
        return main_stream;
    }
    if (stream_id == -1)
    {
        return h2d_stream;
    }
    if (stream_id == -2)
    {
        return d2h_stream;
    }
    if (stream_id == -3)
    {
        return cleanup_stream;
    }

    {
        std::lock_guard<std::mutex> lock(streams_mutex_);
        if (const auto it = compute_streams.find(stream_id); it != compute_streams.end())
        {
            return it->second;
        }
    }

    internal::ctxmgmt::GpuSetCurrentDevice(device_ordinal);
    int priority = 0;
    {
        std::lock_guard<std::mutex> lock(streams_mutex_);
        if (compute_stream_priorities.contains(stream_id))
        {
            priority = compute_stream_priorities.at(stream_id);
        }
    }
    const auto stream = CreateGpuStream(priority);
    {
        std::lock_guard<std::mutex> lock(streams_mutex_);
        if (const auto it = compute_streams.find(stream_id); it != compute_streams.end())
        {
            // Another thread materialized the stream while we were creating this one.
            // Keep a single stream per id and destroy the duplicate.
            DestroyGpuStream(stream);
            return it->second;
        }
        compute_streams.emplace(stream_id, stream);
    }
    return stream;
}

std::optional<int> pi::tensorlib::gpustream::GpuStreamBundle::getStreamId(const GpuStream stream) const
{
    std::lock_guard<std::mutex> lock(streams_mutex_);
    if (main_stream == stream)
    {
        return 0;
    }
    if (h2d_stream == stream)
    {
        return -1;
    }
    if (d2h_stream == stream)
    {
        return -2;
    }
    if (cleanup_stream == stream)
    {
        return -3;
    }
    for (const auto &[id, s] : compute_streams)
    {
        if (s == stream)
        {
            return id;
        }
    }
    return std::nullopt;
}

std::unordered_set<pi::tensorlib::gpustream::GpuStream> pi::tensorlib::gpustream::GpuStreamBundle::getAllStreams() const
{
    std::lock_guard<std::mutex> lock(streams_mutex_);
    std::unordered_set<GpuStream> streams{};
    for (const auto &stream : compute_streams | std::views::values)
    {
        if (stream != nullptr)
        {
            streams.insert(stream);
        }
    }
    return streams;
}

void pi::tensorlib::gpustream::GpuStreamBundle::setComputeStreamPriority(const int stream_id, const int priority)
{
    if (stream_id == 0)
    {
        throw std::runtime_error("Main stream priority cannot be modified through setComputeStreamPriority");
    }

    GpuStream stream_to_destroy = nullptr;
    {
        std::lock_guard<std::mutex> lock(streams_mutex_);
        compute_stream_priorities[stream_id] = priority;

        if (const auto it = compute_streams.find(stream_id); it != compute_streams.end())
        {
            stream_to_destroy = it->second;
            compute_streams.erase(it);
        }
    }
    if (stream_to_destroy != nullptr)
    {
        DestroyGpuStream(stream_to_destroy);
    }
}
