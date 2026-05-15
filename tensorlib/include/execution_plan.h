#pragma once

#include "compute_kernel.h"
#include "stream_descriptor.h"

#include <any>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace pi::tensorlib
{
    /// Forward declarations
    enum class OpType;
    class RealTensor;
    class OpGraph;
    class TraceTensor;

    struct ExecutionEntry
    {
        size_t id{0};

        /// Either op_type or kernel_function must be set
        std::optional<OpType> op_type;
        std::optional<ComputeKernelDescriptor> kernel_descriptor;

        std::vector<std::shared_ptr<RealTensor>> inputs;
        std::vector<std::shared_ptr<RealTensor>> outputs;

        uint64_t flop_estimate{0};
        bool is_useful{true};

        std::unordered_map<std::string, std::any> attributes;

        /// Specifies the GPU stream on which the operation should execute.
        GpuStreamDescriptor gpu_stream_desc{};
    };

    struct GraphExecutionInputDescriptor
    {
        std::string name;
        std::shared_ptr<RealTensor> tensor;
    };

    struct ExecutionPlan
    {
        std::vector<ExecutionEntry> entries;
        std::unordered_map<uint64_t, std::shared_ptr<RealTensor>> real_tensors;
        std::unordered_map<std::string, uint64_t> input_names_to_real_tensor_ids;
        size_t num_events{0};

        static ExecutionPlan FromGraph(OpGraph &graph,
                                       const std::vector<GraphExecutionInputDescriptor> &input_descriptors,
                                       const std::vector<GraphExecutionInputDescriptor> &parameter_descriptors);

        /// Called to update the values of input tensors before execution
        void updateInputDescriptors(const std::vector<GraphExecutionInputDescriptor> &input_descriptors);

        /// Sum of flop_estimate across all entries.
        [[nodiscard]] uint64_t totalFlops() const;

        /// Sum of flop_estimate for entries marked useful.
        [[nodiscard]] uint64_t totalUsefulFlops() const;

        [[nodiscard]] std::optional<std::shared_ptr<RealTensor>>
        getRealTensor(const TraceTensor &trace_tensor) const;

        void releaseTensors() const;
    };

    namespace passes
    {
        class CompilerPass
        {
          public:
            virtual ~CompilerPass() = default;

            virtual void transform(ExecutionPlan &execution_plan) = 0;
        };

        void Transform(ExecutionPlan &execution_plan, const std::vector<std::unique_ptr<CompilerPass>> &passes);

        void RemoveOperation(ExecutionPlan &execution_plan, const ExecutionEntry &entry);

    } // namespace passes

} // namespace pi::tensorlib
