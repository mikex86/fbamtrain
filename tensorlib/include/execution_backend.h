#pragma once

#include "allocator.h"
#include "execution_plan.h"
#include "gpu_stream.h"
#include "tensorlib.h"

#include <memory>
#include <unordered_set>

namespace pi::tensorlib
{
    // Forward declarations
    namespace gputx
    {
        class ScopedRange;
    };

    struct DeviceHash
    {
        std::size_t operator()(const Device &d) const noexcept;
    };

    class AllocationAwaitRegistry;

    class GpuEvent
    {
    public:
        GpuEvent(GpuEvent &&) noexcept;
        GpuEvent &operator=(GpuEvent &&) noexcept;
        ~GpuEvent();

        GpuEvent(const GpuEvent &) = delete;
        GpuEvent &operator=(const GpuEvent &) = delete;

        void record(gpustream::GpuStream gpu_stream) const;
        void synchronize() const;
        [[nodiscard]] bool isComplete() const;
        [[nodiscard]] double elapsedMsSince(const GpuEvent &start) const;
        [[nodiscard]] int deviceOrdinal() const;

        static GpuEvent Create(int ordinal, bool enable_timing);

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;

        explicit GpuEvent(std::unique_ptr<Impl> impl);
        friend class ExecutionBackend;
    };

    class ExecutionBackend
    {
        /// All devices used by operations since the last synchronize call
        std::unordered_set<Device, DeviceHash> devices_{};

        std::unordered_map<std::string, gputx::ScopedRange *> open_ranges;

        explicit ExecutionBackend() = default;

      public:
        void executeOperation(ExecutionEntry &entry, const allocator::AllocatorRegistry &allocator_registry,
                              const std::shared_ptr<gpustream::GpuStreamBundle> &stream_bundle);

        void await(const gpustream::GpuStreamBundle &stream_bundle);

        [[nodiscard]] static GpuEvent CreateEvent(const Device &device, bool enable_timing = false);

        [[nodiscard]] static std::shared_ptr<gpustream::GpuStreamBundle> GetStreamBundle(const Device &device);

        static void SetComputeStreamPriority(const Device &device, const GpuStreamDescriptor &compute_stream_descriptor,
                                             int priority);

        [[nodiscard]] static ExecutionBackend &getInstance();

        /// Wait for an event on a specific GPU stream
        static void GpuWaitFor(const GpuEvent &event, int device_ordinal);

        static void GpuStreamWaitForEvent(const GpuEvent &event, gpustream::GpuStream waiting);

        /// Synchronize all streams in a stream bundle on the host.
        static void SynchronizeStreamBundle(const gpustream::GpuStreamBundle &stream_bundle);

        /// Synchronize a single GPU stream on the host.
        static void SynchronizeGpuStream(gpustream::GpuStream stream, int device_ordinal);

        /**
         * Await one GPU stream on another GPU stream
         * @param to_await the stream that will be awaited
         * @param waiting the stream that will wait
         * @brief This function makes `waiting` wait until all operations in `to_await` are completed.
         */
        static void GpuStreamWaitFor(gpustream::GpuStream to_await, gpustream::GpuStream waiting);

        /**
         * Launch a compute kernel on a given GPU stream
         * @param kernel_descriptor the descriptor to launch
         * @param args the launch arguments
         * @param stream the GPU stream to launch on
         */
        static void LaunchKernel(const ComputeKernelDescriptor &kernel_descriptor, const KernelLaunchArguments &args,
                                 gpustream::GpuStream stream);

        ~ExecutionBackend() = default;
    };
} // namespace pi::tensorlib
