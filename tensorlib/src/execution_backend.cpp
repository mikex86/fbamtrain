#include "execution_backend.h"

#include "ctx_management.h"
#include "device_copy.h"
#include "executor.h"
#include "gputx.h"
#include "kernel_cache.h"
#include "op_graph.h"
#include "stream_utils.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#ifdef PI_TENSORLIB_ENABLE_CUDA
#include <cuda.h>
#endif

#if PI_TENSORLIB_ENABLE_HIP
#include <hip/hip_runtime.h>
#endif

static void *AllocateMemory(pi::tensorlib::allocator::Allocator &allocator, const pi::tensorlib::Shape &shape,
                            const pi::tensorlib::DataType data_type, const pi::tensorlib::Device &device,
                            const bool pinned, const int stream_id, const bool zero_initialize)
{
    const size_t size_in_bytes = shape.numel() * pi::tensorlib::GetDataTypeSize(data_type);
    return allocator.allocate(size_in_bytes, device.ordinal, pinned, stream_id, zero_initialize);
}

static bool ShouldSyncEveryOp()
{
    const char *env = std::getenv("FBAMTRAIN_SYNC_EVERY_OP");
    return env != nullptr && env[0] != '\0';
}

static void SyncGpuStream(pi::tensorlib::gpustream::GpuStream stream)
{
    if (!stream)
    {
        return;
    }
#if PI_TENSORLIB_ENABLE_CUDA
    const auto cuda_stream = static_cast<CUstream>(stream);
    if (const CUresult result = cuStreamSynchronize(cuda_stream);
        result != CUDA_SUCCESS)
    {
        const char *err_msg{};
        cuGetErrorString(result, &err_msg);
        const std::string error_message = err_msg ? err_msg : "Unknown error";
        throw std::runtime_error("Failed to synchronize CUDA stream: CUresult : " + error_message);
    }
#elif PI_TENSORLIB_ENABLE_HIP
    const auto hip_stream = static_cast<hipStream_t>(stream);
    if (const hipError_t result = hipStreamSynchronize(hip_stream); result != hipSuccess)
    {
        throw std::runtime_error("Failed to synchronize HIP stream");
    }
#endif
}

struct ZeroInitRangeConfig
{
    size_t start{0};
    size_t end{0};
    bool invert{false};
};

static std::optional<ZeroInitRangeConfig> GetZeroInitRangeConfig()
{
    static std::optional<ZeroInitRangeConfig> cached = []() -> std::optional<ZeroInitRangeConfig> {
        const char *no_zero_env = std::getenv("NO_ZERO_INIT_RANGE");
        const char *zero_env = std::getenv("ZERO_INIT_RANGE");
        const char *env_value = nullptr;
        const char *env_name = nullptr;
        bool invert = false;

        if (no_zero_env != nullptr && no_zero_env[0] != '\0')
        {
            env_value = no_zero_env;
            env_name = "NO_ZERO_INIT_RANGE";
            invert = true;
        }
        else if (zero_env != nullptr && zero_env[0] != '\0')
        {
            env_value = zero_env;
            env_name = "ZERO_INIT_RANGE";
        }
        else
        {
            return std::nullopt;
        }

        const std::string range_str(env_value);
        const size_t sep = range_str.find(':');
        if (sep == std::string::npos)
        {
            throw std::runtime_error(std::string(env_name) + " must be formatted as start:end");
        }

        const std::string start_str = range_str.substr(0, sep);
        const std::string end_str = range_str.substr(sep + 1);
        if (start_str.empty() || end_str.empty())
        {
            throw std::runtime_error(std::string(env_name) + " must be formatted as start:end");
        }

        size_t start = 0;
        size_t end = 0;
        size_t start_idx = 0;
        size_t end_idx = 0;
        try
        {
            start = std::stoull(start_str, &start_idx, 10);
            end = std::stoull(end_str, &end_idx, 10);
        }
        catch (const std::exception &)
        {
            throw std::runtime_error(std::string(env_name) + " must be formatted as start:end");
        }

        if (start_idx != start_str.size() || end_idx != end_str.size())
        {
            throw std::runtime_error(std::string(env_name) + " must be formatted as start:end");
        }
        if (start > end)
        {
            throw std::runtime_error(std::string(env_name) + " start must be <= end");
        }

        return ZeroInitRangeConfig{.start = start, .end = end, .invert = invert};
    }();

    return cached;
}

static bool ShouldZeroInitialize(const size_t op_id)
{
    const auto config = GetZeroInitRangeConfig();
    if (!config.has_value())
    {
        return true;
    }

    const bool in_range = op_id >= config->start && op_id <= config->end;
    return config->invert ? !in_range : in_range;
}

struct KernelArgPack
{
    pi::tensorlib::KernelLaunchArguments kernel_args{};
    std::vector<void *> arguments{};
};

static KernelArgPack BuildKernelArgPack(const pi::tensorlib::KernelLaunchArguments &args)
{
    KernelArgPack pack{.kernel_args = args, .arguments = {}};
    auto push_ptr = [&pack](auto *p) { pack.arguments.push_back(static_cast<void *>(p)); };

    for (std::any &a : pack.kernel_args.args)
    {
        if (auto p = std::any_cast<void *>(&a))
        {
            push_ptr(p);
        }
        else if (const auto data = std::any_cast<pi::tensorlib::KernelDataArg>(&a))
        {
            push_ptr(data->bytes.data());
        }
        else if (auto p = std::any_cast<uint32_t>(&a))
        {
            push_ptr(p);
        }
        else if (auto p = std::any_cast<uint16_t>(&a))
        {
            push_ptr(p);
        }
        else if (auto p = std::any_cast<int32_t>(&a))
        {
            push_ptr(p);
        }
        else if (auto p = std::any_cast<float>(&a))
        {
            push_ptr(p);
        }
        else
        {
            throw std::runtime_error(std::string("unsupported kernel arg type: ") + a.type().name());
        }
    }

    return pack;
}

static void ValidateKernelArgumentCount(const pi::tensorlib::ComputeKernelDescriptor &kernel_descriptor,
                                        const std::vector<void *> &arguments)
{
    const uint32_t expected_arg_count = kernel_descriptor.expected_arg_count;
    if (expected_arg_count == UINT32_MAX)
    {
        throw std::runtime_error("Kernel argument count missing for " + kernel_descriptor.kernel_name);
    }

    if (arguments.size() != expected_arg_count)
    {
        throw std::runtime_error("Kernel argument count mismatch for " + kernel_descriptor.kernel_name +
                                 ": expected " + std::to_string(expected_arg_count) + ", got " +
                                 std::to_string(arguments.size()));
    }
}

std::size_t pi::tensorlib::DeviceHash::operator()(const Device &d) const noexcept
{
    using UT = std::underlying_type_t<DeviceType>;
    const std::size_t h1 = std::hash<UT>{}(static_cast<UT>(d.device_type));
    const std::size_t h2 = std::hash<int>{}(d.ordinal);

    // boost::hash_combine-style mix
    std::size_t seed = h1;
    seed ^= h2 + std::size_t{0x9e3779b97f4a7c15ull} + (seed << 6) + (seed >> 2);
    return seed;
}

struct pi::tensorlib::GpuEvent::Impl
{
    bool enable_timing_{};

    explicit Impl(const int device_ordinal, const bool enable_timing)
        : enable_timing_(enable_timing), device_ordinal_(device_ordinal)
    {
#if PI_TENSORLIB_ENABLE_CUDA
        if (device_ordinal_ < 0)
        {
            throw std::runtime_error("CUDA events require a non-negative device ordinal.");
        }
        internal::ctxmgmt::GpuSetCurrentDevice(device_ordinal_);

        int flags = CU_EVENT_DEFAULT;
        if (!enable_timing)
        {
            flags |= CU_EVENT_DISABLE_TIMING;
        }

        if (const CUresult result = cuEventCreate(&event_, flags); result != CUDA_SUCCESS)
        {
            const char *err_msg{};
            cuGetErrorString(result, &err_msg);
            const std::string error_message = err_msg ? err_msg : "Unknown error";
            throw std::runtime_error("Failed to create CUDA event: CUresult : " + error_message);
        }
#elif PI_TENSORLIB_ENABLE_HIP
        if (device_ordinal_ < 0)
        {
            throw std::runtime_error("HIP events require a non-negative device ordinal.");
        }
        internal::ctxmgmt::GpuSetCurrentDevice(device_ordinal_);

        int flags = hipEventDefault;
        if (!enable_timing)
        {
            flags |= hipEventDisableTiming;
        }

        if (const hipError_t result = hipEventCreateWithFlags(&event_, flags); result != hipSuccess)
        {
            const char *err_msg = hipGetErrorString(result);
            const std::string error_message = err_msg ? err_msg : "Unknown error";
            throw std::runtime_error("Failed to create HIP event: hipError_t : " + error_message);
        }
#else
        (void)device_ordinal_;
#endif
    }

    ~Impl()
    {
#if PI_TENSORLIB_ENABLE_CUDA || PI_TENSORLIB_ENABLE_HIP
        try
        {
            internal::ctxmgmt::GpuSetCurrentDevice(device_ordinal_);
        }
        catch (...)
        {
        }
#endif
#if PI_TENSORLIB_ENABLE_CUDA
        if (event_ != nullptr)
        {
            cuEventDestroy(event_);
            event_ = nullptr;
        }
#elif PI_TENSORLIB_ENABLE_HIP
        if (event_ != nullptr)
        {
            hipEventDestroy(event_);
            event_ = nullptr;
        }
#endif
    }

    void record(gpustream::GpuStream stream) const
    {
#if PI_TENSORLIB_ENABLE_CUDA
        const auto cuda_stream = static_cast<CUstream>(stream);
        internal::ctxmgmt::GpuSetCurrentDevice(device_ordinal_);
        if (const CUresult result = cuEventRecord(event_, cuda_stream); result != CUDA_SUCCESS)
        {
            const char *err_msg{};
            cuGetErrorString(result, &err_msg);
            const std::string error_message = err_msg ? err_msg : "Unknown error";
            throw std::runtime_error("Failed to record CUDA event: CUresult : " + error_message);
        }
#elif PI_TENSORLIB_ENABLE_HIP
        const auto hip_stream = static_cast<hipStream_t>(stream);
        internal::ctxmgmt::GpuSetCurrentDevice(device_ordinal_);
        if (const hipError_t result = hipEventRecord(event_, hip_stream); result != hipSuccess)
        {
            const char *err_msg = hipGetErrorString(result);
            const std::string error_message = err_msg ? err_msg : "Unknown error";
            throw std::runtime_error("Failed to record HIP event: hipError_t : " + error_message);
        }
#else
#error "Event recording is not supported for non-GPU backends."
#endif
    }

    void synchronize() const
    {
#if PI_TENSORLIB_ENABLE_CUDA
        internal::ctxmgmt::GpuSetCurrentDevice(device_ordinal_);
        if (const CUresult result = cuEventSynchronize(event_); result != CUDA_SUCCESS)
        {
            const char *err_msg{};
            cuGetErrorString(result, &err_msg);
            const std::string error_message = err_msg ? err_msg : "Unknown error";
            throw std::runtime_error("Failed to synchronize CUDA event: CUresult : " + error_message);
        }
#elif PI_TENSORLIB_ENABLE_HIP
        internal::ctxmgmt::GpuSetCurrentDevice(device_ordinal_);
        if (const hipError_t result = hipEventSynchronize(event_); result != hipSuccess)
        {
            const char *err_msg = hipGetErrorString(result);
            const std::string error_message = err_msg ? err_msg : "Unknown error";
            throw std::runtime_error("Failed to synchronize HIP event: hipError_t : " + error_message);
        }
#else
        // nothing to do
#endif
    }

    [[nodiscard]] bool isComplete() const
    {
#if PI_TENSORLIB_ENABLE_CUDA
        internal::ctxmgmt::GpuSetCurrentDevice(device_ordinal_);
        const CUresult result = cuEventQuery(event_);
        if (result == CUDA_SUCCESS)
        {
            return true;
        }
        if (result == CUDA_ERROR_NOT_READY)
        {
            return false;
        }
        const char *err_msg{};
        cuGetErrorString(result, &err_msg);
        const std::string error_message = err_msg ? err_msg : "Unknown error";
        throw std::runtime_error("Failed to query CUDA event: CUresult : " + error_message);
#elif PI_TENSORLIB_ENABLE_HIP
        internal::ctxmgmt::GpuSetCurrentDevice(device_ordinal_);
        const hipError_t result = hipEventQuery(event_);
        if (result == hipSuccess)
        {
            return true;
        }
        if (result == hipErrorNotReady)
        {
            return false;
        }
        const char *err_msg = hipGetErrorString(result);
        const std::string error_message = err_msg ? err_msg : "Unknown error";
        throw std::runtime_error("Failed to query HIP event: hipError_t : " + error_message);
#else
        return true;
#endif
    }

    [[nodiscard]] double elapsedMsSince(const Impl &start) const
    {
        if (!enable_timing_)
        {
            throw std::runtime_error("Timing was not enabled for this event.");
        }
#if PI_TENSORLIB_ENABLE_CUDA
        if (device_ordinal_ != start.device_ordinal_)
        {
            throw std::runtime_error("Cannot compute elapsed time between events on different CUDA devices.");
        }
        start.synchronize();
        synchronize();
        float milliseconds = 0.0f;
        if (const CUresult result = cuEventElapsedTime(&milliseconds, start.event_, event_); result != CUDA_SUCCESS)
        {
            const char *err_msg{};
            cuGetErrorString(result, &err_msg);
            const std::string error_message = err_msg ? err_msg : "Unknown error";
            throw std::runtime_error("Failed to compute elapsed CUDA event time: CUresult : " + error_message);
        }
        return milliseconds;
#elif PI_TENSORLIB_ENABLE_HIP
        if (device_ordinal_ != start.device_ordinal_)
        {
            throw std::runtime_error("Cannot compute elapsed time between events on different HIP devices.");
        }
        start.synchronize();
        synchronize();
        float milliseconds = 0.0f;
        if (const hipError_t result = hipEventElapsedTime(&milliseconds, start.event_, event_); result != hipSuccess)
        {
            const char *err_msg = hipGetErrorString(result);
            const std::string error_message = err_msg ? err_msg : "Unknown error";
            throw std::runtime_error("Failed to compute elapsed HIP event time: hipError_t : " + error_message);
        }
        return static_cast<double>(milliseconds);
#else
        const auto duration = timestamp_ - start.timestamp_;
        return std::chrono::duration<double, std::milli>(duration).count();
#endif
    }

    int device_ordinal_;
#if PI_TENSORLIB_ENABLE_CUDA
    CUevent event_{};
#elif PI_TENSORLIB_ENABLE_HIP
    hipEvent_t event_{};
#else
    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point timestamp_{Clock::now()};
#endif
};
pi::tensorlib::GpuEvent::GpuEvent(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

pi::tensorlib::GpuEvent::GpuEvent(GpuEvent &&) noexcept = default;

pi::tensorlib::GpuEvent &pi::tensorlib::GpuEvent::operator=(GpuEvent &&) noexcept = default;

pi::tensorlib::GpuEvent::~GpuEvent() = default;

void pi::tensorlib::GpuEvent::record(const gpustream::GpuStream gpu_stream) const
{
    if (!impl_)
    {
        throw std::runtime_error("Attempted to record a moved-from backend event.");
    }
    impl_->record(gpu_stream);
}

void pi::tensorlib::GpuEvent::synchronize() const
{
    if (!impl_)
    {
        throw std::runtime_error("Attempted to synchronize a moved-from backend event.");
    }
    impl_->synchronize();
}

bool pi::tensorlib::GpuEvent::isComplete() const
{
    if (!impl_)
    {
        throw std::runtime_error("Attempted to query a moved-from backend event.");
    }
    return impl_->isComplete();
}

double pi::tensorlib::GpuEvent::elapsedMsSince(const GpuEvent &start) const
{
    if (!impl_ || !start.impl_)
    {
        throw std::runtime_error("Attempted to compute elapsed time with a moved-from backend event.");
    }
    return impl_->elapsedMsSince(*start.impl_);
}

int pi::tensorlib::GpuEvent::deviceOrdinal() const { return impl_->device_ordinal_; }

pi::tensorlib::GpuEvent pi::tensorlib::GpuEvent::Create(int ordinal, bool enable_timing)
{
    return GpuEvent(std::make_unique<Impl>(ordinal, enable_timing));
}

pi::tensorlib::GpuEvent pi::tensorlib::ExecutionBackend::CreateEvent(const Device &device, const bool enable_timing)
{
    if (device.device_type != DeviceType::GPU)
    {
        throw std::runtime_error("Events are only available for GPU devices.");
    }
    return GpuEvent::Create(device.ordinal, enable_timing);
}

std::shared_ptr<pi::tensorlib::gpustream::GpuStreamBundle>
pi::tensorlib::ExecutionBackend::GetStreamBundle(const Device &device)
{
    if (device.device_type != DeviceType::GPU)
    {
        throw std::runtime_error("Stream bundles are only available for GPU devices.");
    }
    return internal::ctxmgmt::GetStreamBundle(device.ordinal);
}

void pi::tensorlib::ExecutionBackend::SetComputeStreamPriority(const Device &device,
                                                               const GpuStreamDescriptor &compute_stream_descriptor,
                                                               const int priority)
{
    if (device.device_type != DeviceType::GPU)
    {
        throw std::runtime_error("Stream priority can only be set for GPU devices.");
    }
    const auto bundle = internal::ctxmgmt::GetStreamBundle(device.ordinal);
    const int compute_stream_id = compute_stream_descriptor.getStreamId();
    bundle->setComputeStreamPriority(compute_stream_id, priority);
}

void pi::tensorlib::ExecutionBackend::executeOperation(ExecutionEntry &entry,
                                                       const allocator::AllocatorRegistry &allocator_registry,
                                                       const std::shared_ptr<gpustream::GpuStreamBundle> &stream_bundle)
{
    const bool sync_every_op = ShouldSyncEveryOp();
    if (stream_bundle)
    {
        internal::ctxmgmt::GpuSetCurrentDevice(stream_bundle->device_ordinal);
    }

    if (entry.op_type.has_value())
    {
        switch (const OpType type = *entry.op_type)
        {
            case OpType::CREATE_TENSOR:
            {
                if (!entry.inputs.empty())
                {
                    throw std::runtime_error("CREATE_TENSOR operation should not have inputs.");
                }
                // allocate storage for each output tensor
                if (!entry.gpu_stream_desc.isValid())
                {
                    throw std::runtime_error("CREATE_TENSOR operation has invalid stream descriptor.");
                }
                const int stream_id = entry.gpu_stream_desc.getStreamId();
                for (const auto &real_tensor : entry.outputs)
                {
                    const Device &device = real_tensor->device();
                    auto &allocator = allocator_registry.getAllocator(device.device_type);
                    void *data_ptr = AllocateMemory(allocator, real_tensor->shape(), real_tensor->dtype(), device,
                                                    real_tensor->pinned(), stream_id, false);
                    real_tensor->storage()->initialize(data_ptr, &allocator);
                    real_tensor->storage()->setLastStreamId(stream_id);
                    real_tensor->storage()->setAllocStreamId(stream_id);
                }
                break;
            }
            case OpType::DELETE_TENSOR:
            {
                if (!entry.outputs.empty())
                {
                    throw std::runtime_error("DELETE_TENSOR operation should not have outputs.");
                }

                for (auto &real_tensor : entry.inputs)
                {
                    real_tensor->free();
                }
                break;
            }
            case OpType::DEVICE_COPY:
            {
                if (entry.inputs.size() != 1 || entry.outputs.size() != 1)
                {
                    throw std::runtime_error("DEVICE_COPY operation should have exactly one input and one output.");
                }
                auto &input_tensor = entry.inputs[0];
                auto &output_tensor = entry.outputs[0];
                const auto input_device = input_tensor->device();
                const auto output_device = output_tensor->device();

                if (!stream_bundle)
                {
                    // no stream bundle provided, use default stream
                    // NOTE: This should only happen for CPU-only copies
                    internal::device_copy::PerformDeviceCopy(input_tensor, output_tensor, nullptr);
                    break;
                }

                if (!entry.gpu_stream_desc.isValid())
                {
                    throw std::runtime_error("DEVICE_COPY operation has invalid stream descriptor.");
                }
                const int copy_stream_id = entry.gpu_stream_desc.getStreamId();
                const gpustream::GpuStream stream = streamutils::GetStream(stream_bundle, entry);

                internal::device_copy::PerformDeviceCopy(input_tensor, output_tensor, stream);

                if (input_device.device_type == DeviceType::GPU)
                {
                    input_tensor->storage()->setLastStreamId(copy_stream_id);
                }
                if (output_device.device_type == DeviceType::GPU)
                {
                    output_tensor->storage()->setLastStreamId(copy_stream_id);
                }

                if (sync_every_op && stream_bundle)
                {
                    SyncGpuStream(stream);
                }
                break;
            }
            case OpType::VIEW:
            case OpType::TRANSPOSE:
            case OpType::AT:
            {
                if (entry.inputs.size() != 1 || entry.outputs.size() != 1)
                {
                    throw std::runtime_error(
                        "VIEW/TRANSPOSE/AT operation should have exactly one input and one output.");
                }
                const auto &input_tensor = entry.inputs[0];
                const auto &output_tensor = entry.outputs[0];
                output_tensor->setStorage(input_tensor->storage());
                break;
            }
            case OpType::SPLIT:
            {
                if (entry.inputs.size() != 1 || entry.outputs.empty())
                {
                    throw std::runtime_error("SPLIT operation should have exactly one input and at least one output.");
                }
                auto &input_tensor = entry.inputs[0];
                for (auto &output_tensor : entry.outputs)
                {
                    output_tensor->setStorage(input_tensor->storage());
                }
                break;
            }
            case OpType::RECORD_EVENT:
            {
                if (!stream_bundle)
                {
                    throw std::runtime_error("Event operations require a GPU stream bundle.");
                }
                auto event_it = entry.attributes.find("event_ptr");
                if (event_it == entry.attributes.end())
                {
                    throw std::runtime_error("Event operation missing event_ptr attribute");
                }
                auto event = std::any_cast<std::shared_ptr<GpuEvent>>(event_it->second);
                gpustream::GpuStream target_stream = streamutils::GetStream(stream_bundle, entry);
                event->record(target_stream);
                break;
            }
            case OpType::AWAIT_EVENT:
            {
                if (!stream_bundle)
                {
                    throw std::runtime_error("Event operations require a GPU stream bundle.");
                }
                auto event_it = entry.attributes.find("event_ptr");
                if (event_it == entry.attributes.end())
                {
                    throw std::runtime_error("Event operation missing event_ptr attribute");
                }
                auto event = std::any_cast<std::shared_ptr<GpuEvent>>(event_it->second);
                gpustream::GpuStream target_stream = streamutils::GetStream(stream_bundle, entry);
                GpuStreamWaitForEvent(*event, target_stream);
                break;
            }
            case OpType::BEGIN_GPUTX_RANGE:
            {
                const auto range_name_it = entry.attributes.find("range_name");
                if (range_name_it == entry.attributes.end())
                {
                    throw std::runtime_error("BEGIN_GPUTX_RANGE missing range_name attribute");
                }
                const auto range_name = std::any_cast<std::string>(range_name_it->second);
                auto *gpu_tx_range = MAKE_GPUTX_RANGE(range_name);
                open_ranges[range_name] = gpu_tx_range;
                break;
            }
            case OpType::END_GPUTX_RANGE:
            {
                const auto range_name_it = entry.attributes.find("range_name");
                if (range_name_it == entry.attributes.end())
                {
                    throw std::runtime_error("END_GPUTX_RANGE missing range_name attribute");
                }
                const auto range_name = std::any_cast<std::string>(range_name_it->second);
                const auto it = open_ranges.find(range_name);
                if (it == open_ranges.end())
                {
                    throw std::runtime_error("No open GPUTX range with name: " + range_name);
                }
                delete it->second; // invoke destructor of ScopedRange
                open_ranges.erase(it);
                break;
            }
            // CONTIGUOUS needs to be handled by a kernel, so we don't handle it here
            default:
            {
                throw std::runtime_error("No implementation for this operation type: " + GetOpTypeName(type));
            }
        }
    }
    else
    {
        // launch kernel
        if (entry.kernel_descriptor.has_value())
        {
            auto &kernel_descriptor = entry.kernel_descriptor.value();
            if (!stream_bundle)
            {
                throw std::runtime_error("No stream bundle provided for kernel execution.");
            }
            if (!entry.gpu_stream_desc.isValid())
            {
                throw std::runtime_error("Kernel launch entry has invalid stream descriptor.");
            }
            const auto stream = streamutils::GetStream(stream_bundle, entry);
            const int stream_id = entry.gpu_stream_desc.getStreamId();
            auto update_tensor_stream = [stream_id](const std::shared_ptr<RealTensor> &tensor)
            {
                if (tensor && tensor->device().device_type == DeviceType::GPU)
                {
                    tensor->storage()->setLastStreamId(stream_id);
                }
            };
            for (const auto &tensor : entry.inputs)
            {
                update_tensor_stream(tensor);
            }
            for (const auto &tensor : entry.outputs)
            {
                update_tensor_stream(tensor);
            }

            const KernelLaunchArguments args = kernel_descriptor.argument_provider(entry.inputs, entry.outputs);
            LaunchKernel(kernel_descriptor, args, stream);
            if (!kernel_descriptor.owned_allocations.empty())
            {
                for (const auto &tensor : kernel_descriptor.owned_allocations)
                {
                    if (tensor)
                    {
                        tensor->free();
                    }
                }
                kernel_descriptor.owned_allocations.clear();
            }
            if (sync_every_op)
            {
                SyncGpuStream(stream);
            }
        }
        else
        {
            throw std::runtime_error("No kernel descriptor provided for operation.");
        }
    }
    // add involved devices to the list for synchronization
    {
        for (const auto &tensor : entry.inputs)
        {
            devices_.insert(tensor->device());
        }
        for (const auto &tensor : entry.outputs)
        {
            devices_.insert(tensor->device());
        }
    }
}

void pi::tensorlib::ExecutionBackend::LaunchKernel(const ComputeKernelDescriptor &kernel_descriptor,
                                                   const KernelLaunchArguments &args, const gpustream::GpuStream stream)
{
    switch (kernel_descriptor.backend)
    {
        case KernelBackend::CUDA:
        {
#if PI_TENSORLIB_ENABLE_CUDA
            if (const char *env = std::getenv("FBAMTRAIN_DEBUG_KERNEL_LAUNCH"); env != nullptr && env[0] != '\0')
            {
                std::clog << "[Kernel] Launch " << kernel_descriptor.kernel_name << " ("
                          << kernel_descriptor.function_name << ") grid=(" << args.grid_dim_x << ',' << args.grid_dim_y
                          << ',' << args.grid_dim_z << ") block=(" << args.block_dim_x << ',' << args.block_dim_y << ','
                          << args.block_dim_z << ") cluster=(" << args.cluster_dim_x << ',' << args.cluster_dim_y
                          << ',' << args.cluster_dim_z << ") shared=" << args.shared_mem_bytes << '\n';
            }
            if (!stream)
            {
                throw std::runtime_error("CUDA kernel launch requires a valid stream.");
            }
            KernelCache &kernel_cache = KernelCache::getInstance();
            const int device_ordinal = args.device_ordinal;

            if (!kernel_descriptor.cuda_descriptor)
            {
                throw std::runtime_error("CUDA kernel descriptor missing for " + kernel_descriptor.kernel_name);
            }
            const auto &cuda_descriptor = *kernel_descriptor.cuda_descriptor;

            CUdevice cu_device{};
            if (const CUresult result = cuDeviceGet(&cu_device, device_ordinal); result != CUDA_SUCCESS)
            {
                const char *err_msg{};
                cuGetErrorString(result, &err_msg);
                const std::string error_message = err_msg ? err_msg : "Unknown error";
                throw std::runtime_error("Failed to get CUDA device for ordinal " + std::to_string(device_ordinal) +
                                         ": CUresult : " + error_message);
            }
            auto kernel = kernel_cache.getKernel(kernel_descriptor.kernel_name, device_ordinal);
            if (!kernel)
            {
                kernel_cache.loadKernel(kernel_descriptor.kernel_name, cuda_descriptor.module_data,
                                        cuda_descriptor.module_size, device_ordinal);
                kernel = kernel_cache.getKernel(kernel_descriptor.kernel_name, device_ordinal);

                const std::optional<kernel_func_t> function_opt = kernel_cache.getKernelFunction(
                    kernel_descriptor.kernel_name, kernel_descriptor.function_name, device_ordinal);
                if (!function_opt)
                {
                    throw std::runtime_error("Failed to get kernel function: " + kernel_descriptor.function_name);
                }
                const kernel_func_t function = *function_opt;

                assert(kernel.has_value() && "Kernel should be loaded at this point.");

                if (args.shared_mem_bytes >= 49152)
                {
                    int shared_optin{};
                    cuDeviceGetAttribute(&shared_optin, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                                         cu_device);
                    if (shared_optin >= 49152)
                    {
                        cuFuncSetCacheConfig(function, CU_FUNC_CACHE_PREFER_SHARED);

                        int shared_total{}, shared_static{};
                        cuDeviceGetAttribute(&shared_total, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
                                             cu_device);

                        cuFuncGetAttribute(&shared_static, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, function);

                        cuFuncSetAttribute(function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                           shared_optin - shared_static);
                    }
                }
            }
            assert(kernel && "Kernel should be loaded at this point.");

            const auto kernel_function = kernel_cache.getKernelFunction(
                kernel_descriptor.kernel_name, kernel_descriptor.function_name, device_ordinal);
            if (!kernel_function)
            {
                throw std::runtime_error("Failed to get kernel function: " + kernel_descriptor.function_name);
            }

            auto kernel_arg_pack = BuildKernelArgPack(args);
            ValidateKernelArgumentCount(kernel_descriptor, kernel_arg_pack.arguments);

            internal::ctxmgmt::GpuSetCurrentDevice(device_ordinal);

            const auto cuda_stream = static_cast<CUstream>(stream);
            const bool use_cluster_launch =
                args.cluster_dim_x > 1 || args.cluster_dim_y > 1 || args.cluster_dim_z > 1;
            CUresult result = CUDA_SUCCESS;
            if (use_cluster_launch)
            {
                CUlaunchAttribute launch_attribute{};
                launch_attribute.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
                launch_attribute.value.clusterDim.x = args.cluster_dim_x;
                launch_attribute.value.clusterDim.y = args.cluster_dim_y;
                launch_attribute.value.clusterDim.z = args.cluster_dim_z;

                CUlaunchConfig launch_config{};
                launch_config.gridDimX = args.grid_dim_x;
                launch_config.gridDimY = args.grid_dim_y;
                launch_config.gridDimZ = args.grid_dim_z;
                launch_config.blockDimX = args.block_dim_x;
                launch_config.blockDimY = args.block_dim_y;
                launch_config.blockDimZ = args.block_dim_z;
                launch_config.sharedMemBytes = args.shared_mem_bytes;
                launch_config.hStream = cuda_stream;
                launch_config.attrs = &launch_attribute;
                launch_config.numAttrs = 1;

                result = cuLaunchKernelEx(&launch_config, *kernel_function, kernel_arg_pack.arguments.data(), nullptr);
            }
            else
            {
                result = cuLaunchKernel(*kernel_function, args.grid_dim_x, args.grid_dim_y, args.grid_dim_z,
                                        args.block_dim_x, args.block_dim_y, args.block_dim_z, args.shared_mem_bytes,
                                        cuda_stream, kernel_arg_pack.arguments.data(), nullptr);
            }
            if (result != CUDA_SUCCESS)
            {
                const char *err_msg{};
                cuGetErrorString(result, &err_msg);
                const std::string error_message = err_msg ? err_msg : "Unknown error";
                throw std::runtime_error("Failed to launch CUDA kernel: " + kernel_descriptor.function_name +
                                         ": CUresult : " + error_message + " grid=(" + std::to_string(args.grid_dim_x) +
                                         "," + std::to_string(args.grid_dim_y) + "," + std::to_string(args.grid_dim_z) +
                                         ") block=(" + std::to_string(args.block_dim_x) + "," +
                                         std::to_string(args.block_dim_y) + "," + std::to_string(args.block_dim_z) +
                                         ") cluster=(" + std::to_string(args.cluster_dim_x) + "," +
                                         std::to_string(args.cluster_dim_y) + "," +
                                         std::to_string(args.cluster_dim_z) + ") shared=" +
                                         std::to_string(args.shared_mem_bytes));
            }
#else
            throw std::runtime_error("CUDA backend is not enabled in this build.");
#endif
            break;
        }
        case KernelBackend::HIP:
        {
#if PI_TENSORLIB_ENABLE_HIP
            if (const char *env = std::getenv("FBAMTRAIN_DEBUG_KERNEL_LAUNCH"); env != nullptr && env[0] != '\0')
            {
                std::clog << "[Kernel] Launch " << kernel_descriptor.kernel_name << " ("
                          << kernel_descriptor.function_name << ") grid=(" << args.grid_dim_x << ',' << args.grid_dim_y
                          << ',' << args.grid_dim_z << ") block=(" << args.block_dim_x << ',' << args.block_dim_y << ','
                          << args.block_dim_z << ") shared=" << args.shared_mem_bytes << '\n';
            }
            if (!stream)
            {
                throw std::runtime_error("HIP kernel launch requires a valid stream.");
            }
            KernelCache &kernel_cache = KernelCache::getInstance();
            const int device_ordinal = args.device_ordinal;

            internal::ctxmgmt::GpuSetCurrentDevice(device_ordinal);

            if (!kernel_descriptor.hip_descriptor)
            {
                throw std::runtime_error("HIP kernel descriptor missing for " + kernel_descriptor.kernel_name);
            }
            const auto &hip_descriptor = *kernel_descriptor.hip_descriptor;

            auto kernel = kernel_cache.getKernel(kernel_descriptor.kernel_name, device_ordinal);
            if (!kernel)
            {
                kernel_cache.loadKernel(kernel_descriptor.kernel_name, hip_descriptor.module_data,
                                        hip_descriptor.module_size, device_ordinal);
                kernel = kernel_cache.getKernel(kernel_descriptor.kernel_name, device_ordinal);

                const std::optional<kernel_func_t> function_opt = kernel_cache.getKernelFunction(
                    kernel_descriptor.kernel_name, kernel_descriptor.function_name, device_ordinal);
                if (!function_opt)
                {
                    throw std::runtime_error("Failed to get HIP kernel function: " + kernel_descriptor.function_name);
                }
            }
            assert(kernel && "Kernel should be loaded at this point.");

            const auto kernel_function = kernel_cache.getKernelFunction(
                kernel_descriptor.kernel_name, kernel_descriptor.function_name, device_ordinal);
            if (!kernel_function)
            {
                throw std::runtime_error("Failed to get HIP kernel function: " + kernel_descriptor.function_name);
            }

            auto kernel_arg_pack = BuildKernelArgPack(args);
            ValidateKernelArgumentCount(kernel_descriptor, kernel_arg_pack.arguments);

            const auto hip_stream = static_cast<hipStream_t>(stream);
            if (const hipError_t result =
                    hipModuleLaunchKernel(*kernel_function, args.grid_dim_x, args.grid_dim_y, args.grid_dim_z,
                                          args.block_dim_x, args.block_dim_y, args.block_dim_z, args.shared_mem_bytes,
                                          hip_stream, kernel_arg_pack.arguments.data(), nullptr);
                result != hipSuccess)
            {
                const char *err_msg = hipGetErrorString(result);
                const std::string error_message = err_msg ? err_msg : "Unknown error";
                throw std::runtime_error("Failed to launch HIP kernel: " + kernel_descriptor.function_name +
                                         ": hipError_t : " + error_message + " grid=(" +
                                         std::to_string(args.grid_dim_x) + "," + std::to_string(args.grid_dim_y) + "," +
                                         std::to_string(args.grid_dim_z) + ") block=(" +
                                         std::to_string(args.block_dim_x) + "," + std::to_string(args.block_dim_y) +
                                         "," + std::to_string(args.block_dim_z) + ") shared=" +
                                         std::to_string(args.shared_mem_bytes));
            }
#else
            throw std::runtime_error("HIP backend is not enabled in this build.");
#endif
            break;
        }
    }
}

void pi::tensorlib::ExecutionBackend::await(const gpustream::GpuStreamBundle &stream_bundle)
{
    for (const auto &[device_type, ordinal] : devices_)
    {
        if (device_type != DeviceType::GPU)
        {
            continue; // only GPU devices need synchronization
        }
        internal::ctxmgmt::GpuSetCurrentDevice(ordinal);

        std::unordered_set<gpustream::GpuStream> streams{};
        auto add_stream = [&streams](const gpustream::GpuStream stream)
        {
            if (stream != nullptr)
            {
                streams.insert(stream);
            }
        };
        add_stream(stream_bundle.main_stream);
        add_stream(stream_bundle.h2d_stream);
        add_stream(stream_bundle.d2h_stream);
        add_stream(stream_bundle.cleanup_stream);
        for (const auto &gpu_stream : stream_bundle.getAllStreams())
        {
            add_stream(gpu_stream);
        }

        for (const auto &gpu_stream : streams)
        {
#if PI_TENSORLIB_ENABLE_CUDA
            const auto cuda_stream = static_cast<CUstream>(gpu_stream);
            if (const CUresult result = cuStreamSynchronize(cuda_stream); result != CUDA_SUCCESS)
            {
                const char *err_msg{};
                cuGetErrorString(result, &err_msg);
                const std::string error_message = err_msg ? err_msg : "Unknown error";
                throw std::runtime_error("Failed to synchronize CUDA context: CUresult : " + error_message);
            }
#elif PI_TENSORLIB_ENABLE_HIP
            const auto hip_stream = static_cast<hipStream_t>(gpu_stream);
            if (const hipError_t result = hipStreamSynchronize(hip_stream); result != hipSuccess)
            {
                const char *err_msg = hipGetErrorString(result);
                const std::string error_message = err_msg ? err_msg : "Unknown error";
                throw std::runtime_error("Failed to synchronize HIP device: hipError_t : " + error_message);
            }
#endif
        }
    }
    devices_.clear(); // Clear the list of involved devices after synchronization
}

namespace
{
    void SetCurrentDeviceIfNeeded(const int device_ordinal)
    {
#if PI_TENSORLIB_ENABLE_CUDA || PI_TENSORLIB_ENABLE_HIP
        pi::tensorlib::internal::ctxmgmt::GpuSetCurrentDevice(device_ordinal);
#else
        (void)device_ordinal;
#endif
    }

    void SyncStreamBackend(const pi::tensorlib::gpustream::GpuStream stream)
    {
#if PI_TENSORLIB_ENABLE_CUDA
        const auto cuda_stream = static_cast<CUstream>(stream);
        if (const CUresult result = cuStreamSynchronize(cuda_stream); result != CUDA_SUCCESS)
        {
            const char *err_msg{};
            cuGetErrorString(result, &err_msg);
            const std::string error_message = err_msg ? err_msg : "Unknown error";
            throw std::runtime_error("Failed to synchronize CUDA stream: CUresult : " + error_message);
        }
#elif PI_TENSORLIB_ENABLE_HIP
        const auto hip_stream = static_cast<hipStream_t>(stream);
        if (const hipError_t result = hipStreamSynchronize(hip_stream); result != hipSuccess)
        {
            const char *err_msg = hipGetErrorString(result);
            const std::string error_message = err_msg ? err_msg : "Unknown error";
            throw std::runtime_error("Failed to synchronize HIP stream: hipError_t : " + error_message);
        }
#else
        (void)stream;
#endif
    }
} // namespace

void pi::tensorlib::ExecutionBackend::SynchronizeStreamBundle(const gpustream::GpuStreamBundle &stream_bundle)
{
    SetCurrentDeviceIfNeeded(stream_bundle.device_ordinal);

    std::unordered_set<gpustream::GpuStream> streams{};
    auto add_stream = [&streams](const gpustream::GpuStream stream)
    {
        if (stream != nullptr)
        {
            streams.insert(stream);
        }
    };
    add_stream(stream_bundle.main_stream);
    add_stream(stream_bundle.h2d_stream);
    add_stream(stream_bundle.d2h_stream);
    add_stream(stream_bundle.cleanup_stream);
    for (const auto &gpu_stream : stream_bundle.getAllStreams())
    {
        add_stream(gpu_stream);
    }

    for (const auto &gpu_stream : streams)
    {
        SyncStreamBackend(gpu_stream);
    }
}

void pi::tensorlib::ExecutionBackend::SynchronizeGpuStream(const gpustream::GpuStream stream, const int device_ordinal)
{
    if (stream == nullptr)
    {
        return;
    }
    SetCurrentDeviceIfNeeded(device_ordinal);
    SyncStreamBackend(stream);
}

void pi::tensorlib::ExecutionBackend::GpuWaitFor(const GpuEvent &event, const int device_ordinal)
{
    const auto stream_bundle = GetStreamBundle(Device{DeviceType::GPU, device_ordinal});
    if (!stream_bundle)
    {
        return;
    }
    for (const auto &stream : stream_bundle->getAllStreams())
    {
        GpuStreamWaitForEvent(event, stream);
    }
}
void pi::tensorlib::ExecutionBackend::GpuStreamWaitForEvent(const GpuEvent &event, gpustream::GpuStream waiting)
{
#if PI_TENSORLIB_ENABLE_CUDA
    internal::ctxmgmt::GpuSetCurrentDevice(event.deviceOrdinal());
    const auto cuda_waiting = static_cast<CUstream>(waiting);
    if (const CUresult wait_result = cuStreamWaitEvent(cuda_waiting, event.impl_->event_, 0);
        wait_result != CUDA_SUCCESS)
    {
        const char *err_msg{};
        cuGetErrorString(wait_result, &err_msg);
        const std::string error_message = err_msg ? err_msg : "Unknown error";
        throw std::runtime_error("Failed to make CUDA stream wait for event: CUresult : " + error_message);
    }
#elif PI_TENSORLIB_ENABLE_HIP
    const auto hip_waiting = static_cast<hipStream_t>(waiting);
    if (const hipError_t wait_result = hipStreamWaitEvent(hip_waiting, event.impl_->event_, 0);
        wait_result != hipSuccess)
    {
        const char *err_msg = hipGetErrorString(wait_result);
        const std::string error_message = err_msg ? err_msg : "Unknown error";
        throw std::runtime_error("Failed to make HIP stream wait for event: hipError_t : " + error_message);
    }
#else
    (void)event;
    (void)waiting;
#endif
}
void pi::tensorlib::ExecutionBackend::GpuStreamWaitFor(gpustream::GpuStream to_await, gpustream::GpuStream waiting)
{
#if PI_TENSORLIB_ENABLE_CUDA
    if (to_await == nullptr || waiting == nullptr)
    {
        throw std::runtime_error("Cannot wait on/null CUDA stream.");
    }
    if (to_await == waiting)
    {
        return;
    }

    auto device_ordinal = internal::ctxmgmt::GpuFindDeviceOrdinalForStream(waiting);
    if (!device_ordinal.has_value())
    {
        device_ordinal = internal::ctxmgmt::GpuFindDeviceOrdinalForStream(to_await);
    }
    if (device_ordinal.has_value())
    {
        internal::ctxmgmt::GpuSetCurrentDevice(*device_ordinal);
    }

    const auto cuda_to_await = static_cast<CUstream>(to_await);
    const auto cuda_waiting = static_cast<CUstream>(waiting);

    CUevent event{};
    auto destroy_event = [&event]()
    {
        if (event != nullptr)
        {
            cuEventDestroy(event);
            event = nullptr;
        }
    };
    if (const CUresult create_result = cuEventCreate(&event, CU_EVENT_DISABLE_TIMING); create_result != CUDA_SUCCESS)
    {
        const char *err_msg{};
        cuGetErrorString(create_result, &err_msg);
        const std::string error_message = err_msg ? err_msg : "Unknown error";
        throw std::runtime_error("Failed to create CUDA event for stream wait: CUresult : " + error_message);
    }

    if (const CUresult record_result = cuEventRecord(event, cuda_to_await); record_result != CUDA_SUCCESS)
    {
        destroy_event();
        const char *err_msg{};
        cuGetErrorString(record_result, &err_msg);
        const std::string error_message = err_msg ? err_msg : "Unknown error";
        throw std::runtime_error("Failed to record CUDA event for stream wait: CUresult : " + error_message);
    }

    if (const CUresult wait_result = cuStreamWaitEvent(cuda_waiting, event, 0); wait_result != CUDA_SUCCESS)
    {
        destroy_event();
        const char *err_msg{};
        cuGetErrorString(wait_result, &err_msg);
        const std::string error_message = err_msg ? err_msg : "Unknown error";
        throw std::runtime_error("Failed to make CUDA stream wait for event: CUresult : " + error_message);
    }

    destroy_event();
#elif PI_TENSORLIB_ENABLE_HIP
    if (to_await == nullptr || waiting == nullptr)
    {
        throw std::runtime_error("Cannot wait on/null HIP stream.");
    }
    if (to_await == waiting)
    {
        return;
    }

    const auto hip_to_await = static_cast<hipStream_t>(to_await);
    const auto hip_waiting = static_cast<hipStream_t>(waiting);

    hipEvent_t event{};
    auto destroy_event = [&event]()
    {
        if (event != nullptr)
        {
            hipEventDestroy(event);
            event = nullptr;
        }
    };
    if (const hipError_t create_result = hipEventCreateWithFlags(&event, hipEventDisableTiming);
        create_result != hipSuccess)
    {
        const char *err_msg = hipGetErrorString(create_result);
        const std::string error_message = err_msg ? err_msg : "Unknown error";
        throw std::runtime_error("Failed to create HIP event for stream wait: hipError_t : " + error_message);
    }

    if (const hipError_t record_result = hipEventRecord(event, hip_to_await); record_result != hipSuccess)
    {
        destroy_event();
        const char *err_msg = hipGetErrorString(record_result);
        const std::string error_message = err_msg ? err_msg : "Unknown error";
        throw std::runtime_error("Failed to record HIP event for stream wait: hipError_t : " + error_message);
    }

    if (const hipError_t wait_result = hipStreamWaitEvent(hip_waiting, event, 0); wait_result != hipSuccess)
    {
        destroy_event();
        const char *err_msg = hipGetErrorString(wait_result);
        const std::string error_message = err_msg ? err_msg : "Unknown error";
        throw std::runtime_error("Failed to make HIP stream wait for event: hipError_t : " + error_message);
    }

    destroy_event();
#endif
}

pi::tensorlib::ExecutionBackend &pi::tensorlib::ExecutionBackend::getInstance()
{
    static ExecutionBackend instance{};
    return instance;
}
