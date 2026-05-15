#include "testing.h"

#include <allocator.h>
#include <execution_backend.h>
#include <executor.h>
#include <functional.h>
#include <passes.h>
#include <safe_tensors.h>
#include <tensorlib.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    constexpr pi::tensorlib::Device DEVICE_CPU{pi::tensorlib::DeviceType::CPU, 0};
    constexpr pi::tensorlib::Device DEVICE_GPU{pi::tensorlib::DeviceType::GPU, 0};

    constexpr int CROSS_ENTROPY_STREAM_ID = 1;
    constexpr int CROSS_ENTROPY_STREAM_PRIORITY = 1000;
    constexpr uint64_t STREAMING_CHUNK = 16;
    constexpr int RECOMPUTE_INTERVAL = 8;

    void ApplyDefaultPasses(pi::tensorlib::ExecutionPlan &execution_plan)
    {
        std::vector<std::unique_ptr<pi::tensorlib::passes::CompilerPass>> passes{};
        passes.emplace_back(std::make_unique<MatmulFusePass>());
        passes.emplace_back(std::make_unique<MatmulImplPass>());
        passes.emplace_back(std::make_unique<ActImplPass>());
        passes.emplace_back(std::make_unique<CastImplPass>());
        passes.emplace_back(std::make_unique<TrailingBroadcastAddImplPass>());
        passes.emplace_back(std::make_unique<TrailingBroadcastMulImplPass>());
        passes.emplace_back(std::make_unique<LstmCellImplPass>());
        passes.emplace_back(std::make_unique<CrossEntropyOnTargetsImplPass>());
        passes.emplace_back(std::make_unique<ReduceSumPartialImplPass>());
        passes.emplace_back(std::make_unique<DivImplPass>());
        passes.emplace_back(std::make_unique<FillZerosImplPass>());
        passes.emplace_back(std::make_unique<FillConstantImplPass>());
        passes.emplace_back(std::make_unique<FillUniformImplPass>());
        pi::tensorlib::passes::Transform(execution_plan, passes);
    }

    std::shared_ptr<pi::tensorlib::RealTensor>
    FetchTensor(const std::map<std::string, std::shared_ptr<pi::tensorlib::RealTensor>> &tensors, const char *name)
    {
        const auto it = tensors.find(name);
        if (it == tensors.end())
        {
            throw std::runtime_error(std::string("missing tensor: ") + name);
        }
        return it->second;
    }

    struct TargetPrefetchState
    {
        std::array<pi::tensorlib::TraceTensor, 2> buffers;
        int current{0};
        int prefetched{-1};
        bool first_call{true};
    };
} // namespace

int main()
{
    using namespace pi::tensorlib;

    ExecutionBackend &execution_backend = ExecutionBackend::getInstance();
    const auto main_stream_desc = GpuStreamDescriptors::Main;
    const auto ce_stream_desc = GpuStreamDescriptor{StreamKind::Compute, CROSS_ENTROPY_STREAM_ID};
    ExecutionBackend::SetComputeStreamPriority(DEVICE_GPU, ce_stream_desc, CROSS_ENTROPY_STREAM_PRIORITY);

    const auto &allocator_registry = allocator::CachingAllocatorRegistry::instance();

    const auto reference_tensors = safetensors::Load("reference.safetensors", /*pinned=*/true);
    const auto x_host = FetchTensor(reference_tensors, "x");
    const auto h0_host = FetchTensor(reference_tensors, "h0");
    const auto c0_host = FetchTensor(reference_tensors, "c0");
    const auto w_ih_host = FetchTensor(reference_tensors, "w_ih");
    const auto w_hh_host = FetchTensor(reference_tensors, "w_hh");
    const auto b_ih_host = FetchTensor(reference_tensors, "b_ih");
    const auto b_hh_host = FetchTensor(reference_tensors, "b_hh");
    const auto targets_ref = FetchTensor(reference_tensors, "targets");
    const auto expected_loss_sum = FetchTensor(reference_tensors, "loss_sum");
    const auto expected_loss_mean = FetchTensor(reference_tensors, "loss_mean");

    const auto h0_gpu = h0_host->to(DEVICE_GPU, GpuStreamDescriptors::Main);
    const auto c0_gpu = c0_host->to(DEVICE_GPU, GpuStreamDescriptors::Main);
    const auto w_ih_gpu = w_ih_host->to(DEVICE_GPU, GpuStreamDescriptors::Main);
    const auto w_hh_gpu = w_hh_host->to(DEVICE_GPU, GpuStreamDescriptors::Main);
    const auto b_ih_gpu = b_ih_host->to(DEVICE_GPU, GpuStreamDescriptors::Main);
    const auto b_hh_gpu = b_hh_host->to(DEVICE_GPU, GpuStreamDescriptors::Main);

    const auto &target_dims = targets_ref->shape().dims();
    const uint64_t target_elements = targets_ref->shape().numel();
    if (target_dims.size() != 2)
    {
        throw std::runtime_error("targets tensor expected to be 2D");
    }
    auto targets_host_uint32 =
        RealTensor::Allocate({target_dims[0], target_dims[1]}, DataType::UINT32, DEVICE_CPU, /*pinned=*/true);
    if (targets_ref->dtype() != DataType::FLOAT32)
    {
        throw std::runtime_error("reference targets expected FLOAT32 dtype");
    }
    const auto *src_targets = static_cast<const float *>(targets_ref->dataptr());
    auto *dst_targets = static_cast<uint32_t *>(targets_host_uint32->dataptr());
    for (uint64_t i = 0; i < target_elements; ++i)
    {
        dst_targets[i] = static_cast<uint32_t>(src_targets[i]);
    }

    auto loss_denominator_host = RealTensor::Allocate({1}, DataType::FLOAT32, DEVICE_CPU, /*pinned=*/true);
    static_cast<float *>(loss_denominator_host->dataptr())[0] = static_cast<float>(targets_ref->shape().numel());

    std::vector<GraphInputDescriptor> inputs{};
    inputs.emplace_back(GraphInputDescriptor{.name = "x",
                                             .tensor = TraceTensor::Create(x_host->shape().dims(), x_host->dtype(),
                                                                           DEVICE_CPU, main_stream_desc,
                                                                           /*pinned=*/true)});
    inputs.emplace_back(GraphInputDescriptor{
        .name = "h0",
        .tensor = TraceTensor::Create(h0_host->shape().dims(), DataType::FLOAT32, DEVICE_GPU, main_stream_desc)});
    inputs.emplace_back(GraphInputDescriptor{
        .name = "c0",
        .tensor = TraceTensor::Create(c0_host->shape().dims(), DataType::FLOAT32, DEVICE_GPU, main_stream_desc)});
    inputs.emplace_back(
        GraphInputDescriptor{.name = "targets",
                             .tensor = TraceTensor::Create(targets_host_uint32->shape().dims(), DataType::UINT32,
                                                           DEVICE_CPU, main_stream_desc, true)});
    inputs.emplace_back(GraphInputDescriptor{
        .name = "loss_denominator",
        .tensor = TraceTensor::Create({1}, DataType::FLOAT32, DEVICE_CPU, main_stream_desc, true)});

    std::vector<GraphInputDescriptor> params{};
    params.emplace_back(GraphInputDescriptor{
        .name = "w_ih",
        .tensor = TraceTensor::Create(w_ih_host->shape().dims(), DataType::FLOAT32, DEVICE_GPU, main_stream_desc)});
    params.emplace_back(GraphInputDescriptor{
        .name = "w_hh",
        .tensor = TraceTensor::Create(w_hh_host->shape().dims(), DataType::FLOAT32, DEVICE_GPU, main_stream_desc)});
    params.emplace_back(GraphInputDescriptor{
        .name = "b_ih",
        .tensor = TraceTensor::Create(b_ih_host->shape().dims(), DataType::FLOAT32, DEVICE_GPU, main_stream_desc)});
    params.emplace_back(GraphInputDescriptor{
        .name = "b_hh",
        .tensor = TraceTensor::Create(b_hh_host->shape().dims(), DataType::FLOAT32, DEVICE_GPU, main_stream_desc)});

    for (auto &entry : inputs)
    {
        entry.tensor.markRetained();
    }
    for (auto &entry : params)
    {
        entry.tensor.markRetained();
    }

    OpGraph graph(inputs, params);

    const uint64_t batch_size = x_host->shape().dims()[1];

    std::array<TraceTensor, 2> targets_chunk_gpu{
        graph.createTensor({STREAMING_CHUNK, batch_size}, DataType::UINT32, DEVICE_GPU, main_stream_desc, false),
        graph.createTensor({STREAMING_CHUNK, batch_size}, DataType::UINT32, DEVICE_GPU, main_stream_desc, false)};

    TraceTensor loss_sum = graph.createTensor({1}, DataType::FLOAT32, DEVICE_GPU, main_stream_desc, false);
    FillZeros(graph, loss_sum, ce_stream_desc);

    TraceTensor loss_denominator_gpu = graph.createTensor({1}, DataType::FLOAT32, DEVICE_GPU, main_stream_desc, false);
    DeviceCopy(graph, inputs[4].tensor, loss_denominator_gpu, main_stream_desc);

    auto target_state = std::make_shared<TargetPrefetchState>(
        TargetPrefetchState{.buffers = targets_chunk_gpu, .current = 0, .prefetched = -1, .first_call = true});
    auto hook_calls = std::make_shared<uint64_t>(0);
    auto hook_steps = std::make_shared<uint64_t>(0);

    const TraceTensor &targets_host_trace = inputs[3].tensor;

    StreamingChunkHook ce_hook = [targets_host_trace, loss_sum, target_state, hook_calls, hook_steps, ce_stream_desc](
                                     OpGraph &hook_graph, const TraceTensor &output_chunk, const uint64_t chunk_start,
                                     const uint64_t chunk_steps) -> std::optional<GpuStreamDescriptor>
    {
        const uint64_t total_steps = targets_host_trace.shape()[0];
        if (chunk_start + chunk_steps > total_steps)
        {
            throw std::out_of_range("chunk hook slice out of range");
        }
        if (chunk_steps > STREAMING_CHUNK)
        {
            throw std::out_of_range("chunk_steps exceeds streaming chunk size");
        }

        if (!target_state->first_call)
        {
            target_state->current = target_state->prefetched;
            target_state->prefetched = -1;
        }

        // Ensure the CE stream waits for main-stream writes for this chunk before reading logits.
        AwaitMainStream(hook_graph, DEVICE_GPU, ce_stream_desc);

        const auto &current_buffer = target_state->buffers[static_cast<size_t>(target_state->current)];

        if (target_state->first_call)
        {
            TraceTensor targets_host_slice = targets_host_trace.slice(hook_graph, 0, chunk_start, chunk_steps);
            TraceTensor targets_chunk_slice = current_buffer.slice(hook_graph, 0, 0, chunk_steps);
            // copy on CE compute stream to keep dependency local
            DeviceCopy(hook_graph, targets_host_slice, targets_chunk_slice, ce_stream_desc);
            target_state->first_call = false;
        }

        TraceTensor logits_slice = output_chunk.slice(hook_graph, 0, 0, chunk_steps);
        TraceTensor targets_slice = current_buffer.slice(hook_graph, 0, 0, chunk_steps);

        // Cross entropy per-row then hierarchical partial sum to scalar and accumulate.
        TraceTensor ce_sum = CrossEntropyOnTargets(hook_graph, logits_slice, targets_slice, Reduction::ADD,
                                                   ce_stream_desc, /*reduce_over_rows=*/true);
        InplaceAdd(hook_graph, loss_sum, ce_sum, ce_stream_desc);
        hook_graph.deleteTensor(ce_sum);

        ++*hook_calls;
        *hook_steps += chunk_steps;

        const uint64_t next_start = chunk_start + chunk_steps;
        if (next_start < total_steps)
        {
            const uint64_t next_steps = std::min<uint64_t>(STREAMING_CHUNK, total_steps - next_start);
            const int next_buf = target_state->current ^ 1;
            TraceTensor next_targets_host_slice = targets_host_trace.slice(hook_graph, 0, next_start, next_steps);
            TraceTensor next_targets_chunk_slice =
                target_state->buffers[static_cast<size_t>(next_buf)].slice(hook_graph, 0, 0, next_steps);
            // Copy targets for the next chunk on the CE compute stream to avoid queueing behind large H2D copies.
            DeviceCopy(hook_graph, next_targets_host_slice, next_targets_chunk_slice, ce_stream_desc);
            target_state->prefetched = next_buf;
        }
        return ce_stream_desc;
    };

    // TODO: This test is fucked
    auto lstm_result = StreamingLstmFwd(graph, inputs[0].tensor, inputs[1].tensor, inputs[2].tensor, params[0].tensor,
                                        params[1].tensor, params[2].tensor, params[3].tensor, DataType::FLOAT32,
                                        RECOMPUTE_INTERVAL, STREAMING_CHUNK, main_stream_desc, ce_hook);

    if (*hook_steps != x_host->shape().dims()[0] || *hook_calls != (x_host->shape().dims()[0] / STREAMING_CHUNK))
    {
        throw std::runtime_error("hook mismatch steps=" + std::to_string(*hook_steps) +
                                 " calls=" + std::to_string(*hook_calls));
    }

    AwaitAsyncTransfers(graph, TransferType::H2D, DEVICE_GPU, ce_stream_desc);
    AwaitComputeForTransfer(graph, TransferType::H2D, DEVICE_GPU, ce_stream_desc);
    AwaitAsyncTransfers(graph, TransferType::H2D, DEVICE_GPU, main_stream_desc);

    TraceTensor loss_mean = graph.createTensor({1}, DataType::FLOAT32, DEVICE_GPU, main_stream_desc, false);
    Div(graph, loss_sum, loss_denominator_gpu, loss_mean, ce_stream_desc);

    graph.finalize();

    std::vector<GraphExecutionInputDescriptor> exec_inputs{};
    exec_inputs.push_back(GraphExecutionInputDescriptor{.name = "x", .tensor = x_host});
    exec_inputs.push_back(GraphExecutionInputDescriptor{.name = "h0", .tensor = h0_gpu});
    exec_inputs.push_back(GraphExecutionInputDescriptor{.name = "c0", .tensor = c0_gpu});
    exec_inputs.push_back(GraphExecutionInputDescriptor{.name = "targets", .tensor = targets_host_uint32});
    exec_inputs.push_back(GraphExecutionInputDescriptor{.name = "loss_denominator", .tensor = loss_denominator_host});

    std::vector<GraphExecutionInputDescriptor> exec_params{};
    exec_params.push_back(GraphExecutionInputDescriptor{.name = "w_ih", .tensor = w_ih_gpu});
    exec_params.push_back(GraphExecutionInputDescriptor{.name = "w_hh", .tensor = w_hh_gpu});
    exec_params.push_back(GraphExecutionInputDescriptor{.name = "b_ih", .tensor = b_ih_gpu});
    exec_params.push_back(GraphExecutionInputDescriptor{.name = "b_hh", .tensor = b_hh_gpu});

    ExecutionPlan plan = ExecutionPlan::FromGraph(graph, exec_inputs, exec_params);
    ApplyDefaultPasses(plan);

    Executor executor{plan, execution_backend, 0};
    executor.execute(allocator_registry);

    const auto actual_loss_sum = executor.getOutput(loss_sum);
    const auto actual_loss_mean = executor.getOutput(loss_mean);
    if (!actual_loss_sum || !actual_loss_mean)
    {
        throw std::runtime_error("missing loss outputs");
    }

    testing::AssertSimilar(expected_loss_sum, *actual_loss_sum, 1.0);
    testing::AssertSimilar(expected_loss_mean, *actual_loss_mean, 5e-3);

    return 0;
}
