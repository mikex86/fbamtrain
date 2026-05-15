#include "execution_plan.h"

#include "op_graph.h"
#include "tensorlib.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <unordered_map>
#include <unordered_set>

static void ValidateCompareTraceAndRealTensor(const std::string &name, const pi::tensorlib::TraceTensor &trace_tensor,
                                              const std::shared_ptr<pi::tensorlib::RealTensor> &real_tensor)
{
    if (trace_tensor.shape() != real_tensor->shape())
    {
        throw std::runtime_error(
            "RealTensor provided as input to graph is incompatible with TraceTensor of input descriptor: '" + name +
            "'.");
    }
    if (trace_tensor.dtype() != real_tensor->dtype())
    {
        throw std::runtime_error(
            "RealTensor provided as input to graph is incompatible with TraceTensor of input descriptor: '" + name +
            "'.");
    }
    if (trace_tensor.device() != real_tensor->device())
    {
        throw std::runtime_error(
            "RealTensor provided as input to graph is incompatible with TraceTensor of input descriptor: '" + name +
            "'.");
    }
}

pi::tensorlib::ExecutionPlan
pi::tensorlib::ExecutionPlan::FromGraph(OpGraph &graph,
                                        const std::vector<GraphExecutionInputDescriptor> &input_descriptors,
                                        const std::vector<GraphExecutionInputDescriptor> &parameter_descriptors)
{
    // Maps the ids of the trace tensors to their real tensor counterparts
    std::unordered_map<uint64_t, std::shared_ptr<RealTensor>> real_tensors{};

    // check if tensors for all inputs have been provided
    std::unordered_set<std::string> provided_input_names{};
    std::unordered_map<std::string, TraceTensor> input_trace_tensors{};
    for (const auto &[name, tensor] : input_descriptors)
    {
        if (provided_input_names.contains(name))
        {
            throw std::runtime_error("Input name '" + name + "' is provided multiple times.");
        }
        provided_input_names.insert(name);
    }
    for (const auto &[name, tensor] : graph.getInputDescriptors())
    {
        if (!provided_input_names.contains(name))
        {
            throw std::runtime_error("Input name '" + name + "' is not provided.");
        }
        input_trace_tensors.emplace(name, tensor);
    }

    // check if tensors for all parameters have been provided
    std::unordered_set<std::string> expected_parameter_names{};
    std::unordered_map<std::string, TraceTensor> parameter_trace_tensors{};
    for (const auto &[name, tensor] : parameter_descriptors)
    {
        if (expected_parameter_names.contains(name))
        {
            throw std::runtime_error("Parameter name '" + name + "' is provided multiple times.");
        }
        expected_parameter_names.insert(name);
    }
    for (const auto &[name, tensor] : graph.getParameterDescriptors())
    {
        if (!expected_parameter_names.contains(name))
        {
            throw std::runtime_error("Parameter name '" + name + "' is not provided.");
        }
        expected_parameter_names.insert(name);
        parameter_trace_tensors.emplace(name, tensor);
    }

    // map the trace tensors to their real tensor counterparts
    std::unordered_map<std::string, uint64_t> input_names_to_tensor_ids{};
    for (const auto &[name, real_tensor] : input_descriptors)
    {
        const auto &trace_tensor = input_trace_tensors.at(name);
        ValidateCompareTraceAndRealTensor(name, trace_tensor, real_tensor);

        real_tensor->setRetained(trace_tensor.retained());
        real_tensors.emplace(trace_tensor.id(), real_tensor);
        input_names_to_tensor_ids.emplace(name, trace_tensor.id());
    }

    // map the parameters to their real tensor counterparts
    for (const auto &[name, real_tensor] : parameter_descriptors)
    {
        const auto &trace_tensor = parameter_trace_tensors.at(name);
        ValidateCompareTraceAndRealTensor(name, trace_tensor, real_tensor);
        real_tensor->setRetained(trace_tensor.retained());
        real_tensors.emplace(trace_tensor.id(), real_tensor);
    }

    // iterate over op graph entries and build execution entries with real tensors
    std::vector<ExecutionEntry> execution_entries{};
    execution_entries.reserve(graph.getEntries().size());

    size_t next_id = 0;
    for (const auto &entry : graph.getEntries())
    {
        std::vector<std::shared_ptr<RealTensor>> inputs;
        std::vector<std::shared_ptr<RealTensor>> outputs;
        for (const auto &trace_tensor : entry.inputs)
        {
            const auto real_tensor = real_tensors.at(trace_tensor.id());
            inputs.push_back(real_tensor);
        }
        for (const auto &trace_tensor : entry.outputs)
        {
            auto output = real_tensors.contains(trace_tensor.id()) ? real_tensors.at(trace_tensor.id())
                                                                   : RealTensor::CreateLike(trace_tensor);
            outputs.push_back(output);
            real_tensors.emplace(trace_tensor.id(), output);
        }
        ExecutionEntry execution_entry{.id = next_id++,
                                       .op_type = entry.type,
                                       .inputs = std::move(inputs),
                                       .outputs = std::move(outputs),
                                       .is_useful = entry.is_useful,
                                       .attributes = entry.attributes,
                                       .gpu_stream_desc = entry.gpu_stream_desc};
        execution_entries.push_back(execution_entry);
    }
    return ExecutionPlan{.entries = std::move(execution_entries),
                         .real_tensors = std::move(real_tensors),
                         .input_names_to_real_tensor_ids = input_names_to_tensor_ids,
                         .num_events = graph.numEvents()};
}

void pi::tensorlib::ExecutionPlan::updateInputDescriptors(
    const std::vector<GraphExecutionInputDescriptor> &input_descriptors)
{
    for (const auto &[name, real_tensor] : input_descriptors)
    {
        uint64_t trace_tensor_id;
        bool has_alloc_event = false;

        // find the trace tensor in the map
        {
            auto it = input_names_to_real_tensor_ids.find(name);
            if (it == input_names_to_real_tensor_ids.end())
            {
                throw std::runtime_error("Input name '" + name + "' not found in execution plan.");
            }
            trace_tensor_id = it->second;
        }
        
        auto old_it = real_tensors.find(trace_tensor_id);
        if (old_it == real_tensors.end())
        {
            throw std::runtime_error("Input name '" + name + "' not found in execution plan tensors.");
        }
        auto old_real_tensor = old_it->second;
        old_it->second = real_tensor;

        // update input_names_to_tensor_ids entry
        input_names_to_real_tensor_ids[name] = trace_tensor_id;

        // update the real tensor in the execution entries
        for (auto &entry : entries)
        {
            for (auto &input_tensor : entry.inputs)
            {
                if (input_tensor == old_real_tensor)
                {
                    input_tensor = real_tensor;
                }
            }
            for (auto &output_tensor : entry.outputs)
            {
                if (output_tensor == old_real_tensor)
                {
                    output_tensor = real_tensor;
                }
            }
        }
    }
}

void pi::tensorlib::passes::Transform(ExecutionPlan &execution_plan,
                                      const std::vector<std::unique_ptr<CompilerPass>> &passes)
{
    for (const auto &pass : passes)
    {
        pass->transform(execution_plan);
    }
}

uint64_t pi::tensorlib::ExecutionPlan::totalFlops() const
{
    uint64_t total = 0;
    for (const auto &entry : entries)
    {
        total += entry.flop_estimate;
    }
    return total;
}

std::optional<std::shared_ptr<pi::tensorlib::RealTensor>>
pi::tensorlib::ExecutionPlan::getRealTensor(const TraceTensor &trace_tensor) const
{
    const auto it = real_tensors.find(trace_tensor.id());
    if (it == real_tensors.end())
    {
        return std::nullopt;
    }
    return it->second;
}

void pi::tensorlib::ExecutionPlan::releaseTensors() const
{
    for (const auto &entry : entries)
    {
        if (entry.op_type != OpType::CREATE_TENSOR)
        {
            continue;
        }
        for (const auto &output : entry.outputs)
        {
            if (!output || output->isView() || output->retained())
            {
                continue;
            }
            const auto &storage = output->storage();
            if (!storage || storage->isFreed() || storage->dataptr() == nullptr)
            {
                continue;
            }
            output->free();
        }
    }
}

uint64_t pi::tensorlib::ExecutionPlan::totalUsefulFlops() const
{
    uint64_t total = 0;
    for (const auto &entry : entries)
    {
        if (entry.is_useful)
        {
            total += entry.flop_estimate;
        }
    }
    return total;
}

void pi::tensorlib::passes::RemoveOperation(ExecutionPlan &execution_plan, const ExecutionEntry &entry)
{
    // remove by .id
    const auto it =
        std::ranges::remove_if(execution_plan.entries, [&entry](const ExecutionEntry &e) { return e.id == entry.id; })
            .begin();
    if (it != execution_plan.entries.end())
    {
        execution_plan.entries.erase(it, execution_plan.entries.end());
    }
}
