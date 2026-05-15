#pragma once

#include "execution_backend.h"
#include "execution_plan.h"

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace pi::tensorlib
{
    class RealTensor;
    class OpGraph;
    class ExecutionBackend;
    class TraceTensor;

    namespace allocator
    {
        class AllocatorRegistry;
    } // namespace allocator

    class Executor
    {
        ExecutionBackend &backend_;
        ExecutionPlan execution_plan_;
        int device_ordinal_{-1};
        bool finished_ = false;
        bool executed_gpu_work_{false};

        /// Keeps per-execution GPU events alive until work completes.
        std::vector<std::vector<std::shared_ptr<GpuEvent>>> inflight_events_{};

        /// Maps the ids of the trace tensors to their real tensor counterparts
        std::unordered_map<uint64_t, std::shared_ptr<RealTensor>> real_tensors{};

      public:
        explicit Executor(ExecutionPlan execution_plan, ExecutionBackend &backend, int device_ordinal);

        void updateInputDescriptors(const std::vector<GraphExecutionInputDescriptor> &input_descriptors);

        void execute(const allocator::AllocatorRegistry &allocator_registry, bool awaitExecution = true);

        void await();

        void releaseTensors();

        [[nodiscard]] std::optional<std::shared_ptr<RealTensor>> getOutput(const TraceTensor &trace_tensor,
                                                                           bool unsafe = false) const;

        /// Instructs the main-stream of all devices used in the last execution
        void gpuWaitFor(const GpuEvent &event) const;
    };
} // namespace pi::tensorlib
