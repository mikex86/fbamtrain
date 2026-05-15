#include "ctx_management.h"

#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace
{
    using StreamBundle = pi::tensorlib::gpustream::GpuStreamBundle;
    using StreamBundlePtr = std::shared_ptr<StreamBundle>;

    [[nodiscard]] StreamBundlePtr CreateStandardStreamBundle(const int device_ordinal)
    {
        pi::tensorlib::internal::ctxmgmt::GpuSetCurrentDevice(device_ordinal);

        // Highest priority for main compute; keep transfer streams at default priority.
        const auto main_stream = pi::tensorlib::gpustream::CreateGpuStream(/*priority=*/-1000); // max priority
        const auto d2h_stream = pi::tensorlib::gpustream::CreateGpuStream();
        const auto h2d_stream = pi::tensorlib::gpustream::CreateGpuStream();
        const auto cleanup_stream = pi::tensorlib::gpustream::CreateGpuStream(/*priority=*/1);

        auto bundle = std::make_shared<StreamBundle>();
        bundle->device_ordinal = device_ordinal;
        bundle->main_stream = main_stream;
        bundle->h2d_stream = h2d_stream;
        bundle->d2h_stream = d2h_stream;
        bundle->cleanup_stream = cleanup_stream;
        bundle->init();
        return bundle;
    }

    template <typename MapType, typename CreateFn>
    auto GetOrCreate(MapType &map, std::mutex &map_mutex, const int key, CreateFn &&create_fn) -> MapType::mapped_type
    {
        std::lock_guard<std::mutex> lock(map_mutex);
        const auto it = map.find(key);
        if (it != map.end())
        {
            return it->second;
        }
        auto value = create_fn(key);
        map.emplace(key, value);
        return value;
    }
} // namespace

#if PI_TENSORLIB_ENABLE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>

static std::once_flag cuda_init_once;
thread_local int current_device_ordinal = -1;
static std::unordered_map<int, CUcontext> device_contexts{};
static std::unordered_map<int, StreamBundlePtr> device_stream_bundles{};
static std::mutex device_contexts_mutex{};
static std::mutex device_stream_bundles_mutex{};

static CUcontext RetainPrimaryContext(const int ordinal)
{
    std::call_once(cuda_init_once, []()
    {
        if (const CUresult status = cuInit(0); status != CUDA_SUCCESS)
        {
            throw std::runtime_error("Failed to initialize CUDA, error code: " + std::to_string(status));
        }
    });
    CUdevice device{};
    if (const CUresult status = cuDeviceGet(&device, ordinal); status != CUDA_SUCCESS)
    {
        throw std::runtime_error("Failed to get CUDA device for ordinal " + std::to_string(ordinal) +
                                 ", error code: " + std::to_string(status));
    }
    CUcontext ctx{};
    if (const CUresult status = cuDevicePrimaryCtxRetain(&ctx, device); status != CUDA_SUCCESS)
    {
        throw std::runtime_error("Failed to retain CUDA primary context for device ordinal " +
                                 std::to_string(ordinal) + ", error code: " + std::to_string(status));
    }
    return ctx;
}

static CUcontext GetOrCreateCudaContext(const int ordinal)
{
    return GetOrCreate(device_contexts, device_contexts_mutex, ordinal, RetainPrimaryContext);
}

void pi::tensorlib::internal::ctxmgmt::GpuSetCurrentDevice(const int ordinal)
{
    if (current_device_ordinal == ordinal)
        return;

    CUcontext device_ctx = GetOrCreateCudaContext(ordinal);
    cuCtxSetCurrent(device_ctx);
    const cudaError_t rt_status = cudaSetDevice(ordinal);
    if (rt_status != cudaSuccess)
    {
        throw std::runtime_error("Failed to set CUDA runtime device ordinal " + std::to_string(ordinal) + ": " +
                                 std::string(cudaGetErrorString(rt_status)));
    }
    current_device_ordinal = ordinal;
}

pi::tensorlib::internal::ctxmgmt::device_ctx_t pi::tensorlib::internal::ctxmgmt::GpuGetDeviceCtx(const int ordinal)
{
    return GetOrCreateCudaContext(ordinal);
}

std::shared_ptr<pi::tensorlib::gpustream::GpuStreamBundle>
pi::tensorlib::internal::ctxmgmt::GetStreamBundle(const int device_ordinal)
{
    return GetOrCreate(device_stream_bundles, device_stream_bundles_mutex, device_ordinal,
                       [](const int ordinal)
                       { return CreateStandardStreamBundle(ordinal); });
}

std::optional<int> pi::tensorlib::internal::ctxmgmt::GpuFindDeviceOrdinalForStream(
    const pi::tensorlib::gpustream::GpuStream stream)
{
    if (stream == nullptr)
    {
        return std::nullopt;
    }
    std::lock_guard<std::mutex> lock(device_stream_bundles_mutex);
    for (const auto &[device_ordinal, bundle] : device_stream_bundles)
    {
        if (!bundle)
        {
            continue;
        }
        if (bundle->getStreamId(stream).has_value())
        {
            return device_ordinal;
        }
    }
    return std::nullopt;
}

#endif

#if PI_TENSORLIB_ENABLE_HIP
#include <hip/hip_runtime.h>

thread_local int hip_current_device_ordinal = -1;
static std::unordered_map<int, StreamBundlePtr> hip_device_stream_bundles{};
static std::mutex hip_device_stream_bundles_mutex{};

static void HipSetDevice(const int ordinal)
{
    if (hip_current_device_ordinal == ordinal)
    {
        return;
    }
    if (const hipError_t status = hipSetDevice(ordinal); status != hipSuccess)
    {
        throw std::runtime_error("Failed to set HIP device ordinal " + std::to_string(ordinal) + ": " +
                                 std::string(hipGetErrorString(status)));
    }
    hip_current_device_ordinal = ordinal;
}

void pi::tensorlib::internal::ctxmgmt::GpuSetCurrentDevice(const int ordinal) { HipSetDevice(ordinal); }

pi::tensorlib::internal::ctxmgmt::device_ctx_t pi::tensorlib::internal::ctxmgmt::GpuGetDeviceCtx(const int ordinal)
{
    HipSetDevice(ordinal);
    return nullptr;
}

std::shared_ptr<pi::tensorlib::gpustream::GpuStreamBundle>
pi::tensorlib::internal::ctxmgmt::GetStreamBundle(const int device_ordinal)
{
    return GetOrCreate(hip_device_stream_bundles, hip_device_stream_bundles_mutex, device_ordinal,
                       [](const int ordinal)
                       { return CreateStandardStreamBundle(ordinal); });
}

std::optional<int> pi::tensorlib::internal::ctxmgmt::GpuFindDeviceOrdinalForStream(
    const pi::tensorlib::gpustream::GpuStream stream)
{
    if (stream == nullptr)
    {
        return std::nullopt;
    }
    std::lock_guard<std::mutex> lock(hip_device_stream_bundles_mutex);
    for (const auto &[device_ordinal, bundle] : hip_device_stream_bundles)
    {
        if (!bundle)
        {
            continue;
        }
        if (bundle->getStreamId(stream).has_value())
        {
            return device_ordinal;
        }
    }
    return std::nullopt;
}
#endif
