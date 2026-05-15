#include "executor.h"

#include "ctx_management.h"
#include "gpu_stream.h"
#include "op_graph.h"
#include <any>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#if PI_TENSORLIB_ENABLE_CUDA
#include <cuda.h>
#endif
#if PI_TENSORLIB_ENABLE_HIP
#include <hip/hip_runtime.h>
#endif

pi::tensorlib::Executor::Executor(ExecutionPlan execution_plan, ExecutionBackend &backend, const int device_ordinal)
    : backend_(backend), execution_plan_(std::move(execution_plan)), device_ordinal_(device_ordinal)
{
    if (device_ordinal_ < 0)
    {
        throw std::invalid_argument("Executor requires a non-negative GPU device ordinal.");
    }
    real_tensors = execution_plan_.real_tensors;
}

void pi::tensorlib::Executor::updateInputDescriptors(
    const std::vector<GraphExecutionInputDescriptor> &input_descriptors)
{
    execution_plan_.updateInputDescriptors(input_descriptors);
    real_tensors = execution_plan_.real_tensors;
    finished_ = false;
}

static void WaitDeviceIdle(pi::tensorlib::ExecutionBackend &backend, const int device_ordinal)
{
    using namespace pi::tensorlib;
    const auto stream_bundle = internal::ctxmgmt::GetStreamBundle(device_ordinal);
    backend.await(*stream_bundle);
}

static void SyncDevice(const int device_ordinal)
{
#if PI_TENSORLIB_ENABLE_CUDA
    pi::tensorlib::internal::ctxmgmt::GpuSetCurrentDevice(device_ordinal);
    if (const CUresult result = cuCtxSynchronize(); result != CUDA_SUCCESS)
    {
        const char *err_msg{};
        cuGetErrorString(result, &err_msg);
        const std::string error_message = err_msg ? err_msg : "Unknown error";
        throw std::runtime_error("Failed to synchronize CUDA context: CUresult : " + error_message);
    }
#elif PI_TENSORLIB_ENABLE_HIP
    pi::tensorlib::internal::ctxmgmt::GpuSetCurrentDevice(device_ordinal);
    if (const hipError_t result = hipDeviceSynchronize(); result != hipSuccess)
    {
        const char *err_msg = hipGetErrorString(result);
        const std::string error_message = err_msg ? err_msg : "Unknown error";
        throw std::runtime_error("Failed to synchronize HIP device: hipError_t : " + error_message);
    }
#else
    (void)device_ordinal;
#endif
}

static void SyncStream(const std::shared_ptr<pi::tensorlib::gpustream::GpuStreamBundle> &stream_bundle,
                       const int stream_id)
{
    if (!stream_bundle)
    {
        return;
    }
#if PI_TENSORLIB_ENABLE_CUDA
    const auto stream = stream_bundle->getComputeStream(stream_id);
    if (const CUresult result = cuStreamSynchronize(static_cast<CUstream>(stream)); result != CUDA_SUCCESS)
    {
        const char *err_msg{};
        cuGetErrorString(result, &err_msg);
        const std::string error_message = err_msg ? err_msg : "Unknown error";
        throw std::runtime_error("Failed to synchronize CUDA stream: CUresult : " + error_message);
    }
#elif PI_TENSORLIB_ENABLE_HIP
    const auto stream = stream_bundle->getComputeStream(stream_id);
    if (const hipError_t result = hipStreamSynchronize(static_cast<hipStream_t>(stream)); result != hipSuccess)
    {
        const char *err_msg = hipGetErrorString(result);
        const std::string error_message = err_msg ? err_msg : "Unknown error";
        throw std::runtime_error("Failed to synchronize HIP stream: hipError_t : " + error_message);
    }
#else
    (void)stream_bundle;
    (void)stream_id;
#endif
}

void pi::tensorlib::Executor::execute(const allocator::AllocatorRegistry &allocator_registry, const bool awaitExecution)
{
    executed_gpu_work_ = false;
    std::vector<std::shared_ptr<GpuEvent>> empty_events{};
    std::vector<std::shared_ptr<GpuEvent>> *events_ptr = &empty_events;
    if (execution_plan_.num_events > 0)
    {
        inflight_events_.emplace_back(execution_plan_.num_events);
        events_ptr = &inflight_events_.back();
    }
    auto &events = *events_ptr;

    // to allow for repeated execution of the same execution plan,
    // we need to free all tensors created during the previous execution by a CREATE_TENSOR op.
    // VIEW ops do not own the memory, so we do not free them - however, we do need to unset the storage
    // pointer to be re-initialized during plan execution.
    for (const auto &entry : execution_plan_.entries)
    {
        if (entry.op_type == OpType::CREATE_TENSOR)
        {
            for (const auto &output : entry.outputs)
            {
                output->free();
            }
        }
        else if (!entry.outputs.empty())
        {
            for (const auto &output : entry.outputs)
            {
                if (output->isView())
                {
                    output->setStorage(nullptr);
                }
            }
        }
    }

    // execute the operations using the backend
    for (auto &execution_entry : execution_plan_.entries)
    {
        int device_ordinal{};

        // find the "device_ordinal" attribute
        if (const auto device_ordinal_it = execution_entry.attributes.find("device_ordinal");
            device_ordinal_it == execution_entry.attributes.end())
        {
            device_ordinal = -1;
        }
        else
        {
            device_ordinal = std::any_cast<int>(device_ordinal_it->second);
        }

        if (execution_entry.op_type == OpType::RECORD_EVENT)
        {
            if (device_ordinal == -1)
            {
                throw std::runtime_error("Record event operation missing device_ordinal attribute");
            }
        } else if (execution_entry.op_type == OpType::AWAIT_EVENT)
        {
            // find the event which is being awaited
            const auto handle_it = execution_entry.attributes.find("event_handle");
            if (handle_it == execution_entry.attributes.end())
            {
                throw std::runtime_error("Await event operation missing event_handle attribute");
            }
            const auto event_id = std::any_cast<size_t>(handle_it->second);
            if (event_id >= events.size())
            {
                throw std::runtime_error("Event handle out of range");
            }
            if (!events[event_id])
            {
                throw std::runtime_error("Awaiting an uninitialized event");
            }
            device_ordinal = events[event_id]->deviceOrdinal();
        }

        // Determine device ordinal the operation will run on.
        // We assume the first input with a GPU device type determines the device ordinal.
        // If no such input exists, then we leave device_ordinal as -1, which indicates
        // that this is a CPU-only operation.
        if (device_ordinal == -1)
        {
            for (const auto &input : execution_entry.inputs)
            {
                const Device &device = input->device();
                if (device.device_type != DeviceType::GPU)
                {
                    continue;
                }
                device_ordinal = device.ordinal;
                goto device_ordinal_found;
            }
            for (const auto &output : execution_entry.outputs)
            {
                const Device &device = output->device();
                if (device.device_type != DeviceType::GPU)
                {
                    continue;
                }
                device_ordinal = device.ordinal;
                goto device_ordinal_found;
            }
            device_ordinal = -1;
        }
    device_ordinal_found:
        if (device_ordinal_ >= 0 && device_ordinal != -1 && device_ordinal != device_ordinal_)
        {
            throw std::runtime_error("Executor device ordinal mismatch with operation target device.");
        }

        if (execution_entry.op_type == OpType::RECORD_EVENT || execution_entry.op_type == OpType::AWAIT_EVENT)
        {
            const auto handle_it = execution_entry.attributes.find("event_handle");
            if (handle_it == execution_entry.attributes.end())
            {
                throw std::runtime_error("Event operation missing event_handle attribute");
            }
            const auto event_id = std::any_cast<size_t>(handle_it->second);
            if (event_id >= events.size())
            {
                throw std::runtime_error("Event handle out of range");
            }
            if (!events[event_id])
            {
                events[event_id] = std::make_shared<GpuEvent>(
                    ExecutionBackend::CreateEvent(Device{DeviceType::GPU, device_ordinal}, /*enable_timing=*/false));
            }
            execution_entry.attributes["event_ptr"] = events[event_id];
        }
        std::shared_ptr<gpustream::GpuStreamBundle> stream_bundle;
        if (device_ordinal != -1)
        {
            stream_bundle = internal::ctxmgmt::GetStreamBundle(device_ordinal);
            executed_gpu_work_ = true;
        }
        backend_.executeOperation(execution_entry, allocator_registry, stream_bundle);
    }

    if (awaitExecution)
    {
        if (executed_gpu_work_)
        {
            WaitDeviceIdle(backend_, device_ordinal_);
        }
        finished_ = true;
        inflight_events_.clear();
    }
}
std::optional<std::shared_ptr<pi::tensorlib::RealTensor>>
pi::tensorlib::Executor::getOutput(const TraceTensor &trace_tensor, const bool unsafe) const
{
    if (!unsafe)
    {
        if (!finished_)
        {
            throw std::runtime_error("Graph execution has not been awaited yet.");
        }
    }
    const auto it = real_tensors.find(trace_tensor.id());
    if (it == real_tensors.end())
    {
        return std::nullopt;
    }
    return it->second;
}

void pi::tensorlib::Executor::gpuWaitFor(const GpuEvent &event) const
{
    ExecutionBackend::GpuWaitFor(event, device_ordinal_);
}

void pi::tensorlib::Executor::await()
{
    if (finished_)
    {
        return;
    }
    if (executed_gpu_work_)
    {
        WaitDeviceIdle(backend_, device_ordinal_);
    }
    finished_ = true;
    inflight_events_.clear();
}

void pi::tensorlib::Executor::releaseTensors()
{
    execution_plan_.releaseTensors();
}
