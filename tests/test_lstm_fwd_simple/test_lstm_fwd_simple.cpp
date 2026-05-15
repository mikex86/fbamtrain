#include "testing.h"

#include <allocator.h>
#include <execution_backend.h>
#include <executor.h>
#include <functional.h>
#include <op_graph.h>
#include <passes.h>
#include <safe_tensors.h>
#include <tensorlib.h>

#include <stdexcept>

using namespace pi::tensorlib;

namespace
{
    constexpr Device DEVICE_CPU{DeviceType::CPU, 0};
    constexpr Device DEVICE_GPU{DeviceType::GPU, 0};
} // namespace

int main()
{
    ExecutionBackend &execution_backend = ExecutionBackend::getInstance();
    const auto &allocator_registry = allocator::CachingAllocatorRegistry::instance();
    const auto main_stream_desc = GpuStreamDescriptors::Main;

    const auto tensors = safetensors::Load("reference.safetensors");
    const auto fetch = [&tensors](const char *name)
    {
        auto it = tensors.find(name);
        if (it == tensors.end())
        {
            throw std::runtime_error(std::string("missing tensor: ") + name);
        }
        return it->second;
    };

    const auto x_host = fetch("x");
    const auto h0_host = fetch("h0");
    const auto c0_host = fetch("c0");
    const auto w_ih_host = fetch("w_ih");
    const auto w_hh_host = fetch("w_hh");
    const auto b_ih_host = fetch("b_ih");
    const auto b_hh_host = fetch("b_hh");
    const auto expected_y = fetch("y");
    const auto expected_hn = fetch("h_n");
    const auto expected_cn = fetch("c_n");

    const auto h0_device = h0_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
    const auto c0_device = c0_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);

    const auto w_ih_device = w_ih_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
    const auto w_hh_device = w_hh_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
    const auto b_ih_device = b_ih_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
    const auto b_hh_device = b_hh_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);

    std::vector<GraphInputDescriptor> inputs{};
    inputs.emplace_back(GraphInputDescriptor{
        .name = "x", .tensor = TraceTensor::Create(x_host->shape().dims(), x_host->dtype(), DEVICE_CPU,
                                                   main_stream_desc)});
    inputs.emplace_back(GraphInputDescriptor{
        .name = "h0", .tensor = TraceTensor::Create(h0_host->shape().dims(), DataType::FLOAT32, DEVICE_GPU,
                                                    main_stream_desc)});
    inputs.emplace_back(GraphInputDescriptor{
        .name = "c0", .tensor = TraceTensor::Create(c0_host->shape().dims(), DataType::FLOAT32, DEVICE_GPU,
                                                    main_stream_desc)});

    std::vector<GraphInputDescriptor> params{};
    params.emplace_back(GraphInputDescriptor{
        .name = "w_ih", .tensor = TraceTensor::Create(w_ih_host->shape().dims(), w_ih_host->dtype(), DEVICE_GPU,
                                                      main_stream_desc)});
    params.emplace_back(GraphInputDescriptor{
        .name = "w_hh", .tensor = TraceTensor::Create(w_hh_host->shape().dims(), w_hh_host->dtype(), DEVICE_GPU,
                                                      main_stream_desc)});
    params.emplace_back(GraphInputDescriptor{
        .name = "b_ih", .tensor = TraceTensor::Create(b_ih_host->shape().dims(), b_ih_host->dtype(), DEVICE_GPU,
                                                      main_stream_desc)});
    params.emplace_back(GraphInputDescriptor{
        .name = "b_hh", .tensor = TraceTensor::Create(b_hh_host->shape().dims(), b_hh_host->dtype(), DEVICE_GPU,
                                                      main_stream_desc)});

    for (auto &entry : inputs)
    {
        entry.tensor.markRetained();
    }
    for (auto &entry : params)
    {
        entry.tensor.markRetained();
    }

    const std::vector<std::pair<int, int>> scenarios = {
        {1, 1}, {1, 2}, {1, 4}, {1, 8}, // baseline streaming sizes
        {2, 1}, {2, 2}, {2, 4},         // interval equals chunk or greater
        {3, 1}, {3, 2}, {3, 4},         // interval > chunk cases
        {4, 1}, {4, 2}, {4, 4},         // interval aligned with chunk divisor
        {5, 1}, {5, 2}, {5, 4},         // interval larger than chunk
        {8, 1}, {8, 2}, {8, 4}, {8, 8}  // interval spans the full sequence
    };

    for (const auto [recompute_interval, streaming_chunk] : scenarios)
    {
        OpGraph graph(inputs, params);

        auto result = StreamingLstmFwd(graph, inputs[0].tensor, inputs[1].tensor, inputs[2].tensor, params[0].tensor,
                                       params[1].tensor, params[2].tensor, params[3].tensor, DataType::FLOAT16,
                                       recompute_interval, streaming_chunk, main_stream_desc);

        graph.finalize();

        std::vector<GraphExecutionInputDescriptor> exec_inputs;
        exec_inputs.emplace_back(GraphExecutionInputDescriptor{.name = "x", .tensor = x_host});
        exec_inputs.emplace_back(GraphExecutionInputDescriptor{.name = "h0", .tensor = h0_device});
        exec_inputs.emplace_back(GraphExecutionInputDescriptor{.name = "c0", .tensor = c0_device});
        std::vector<GraphExecutionInputDescriptor> exec_params;
        exec_params.emplace_back(GraphExecutionInputDescriptor{.name = "w_ih", .tensor = w_ih_device});
        exec_params.emplace_back(GraphExecutionInputDescriptor{.name = "w_hh", .tensor = w_hh_device});
        exec_params.emplace_back(GraphExecutionInputDescriptor{.name = "b_ih", .tensor = b_ih_device});
        exec_params.emplace_back(GraphExecutionInputDescriptor{.name = "b_hh", .tensor = b_hh_device});

        ExecutionPlan plan = ExecutionPlan::FromGraph(graph, exec_inputs, exec_params);

        std::vector<std::unique_ptr<passes::CompilerPass>> passes{};
        passes.emplace_back(std::make_unique<MatmulFusePass>()); // fuse matmul + bias into addmm
        passes.emplace_back(std::make_unique<MatmulImplPass>()); // fall back for any unfused matmul
        passes.emplace_back(std::make_unique<ActImplPass>());
        passes.emplace_back(std::make_unique<CastImplPass>());
        passes.emplace_back(std::make_unique<TrailingBroadcastAddImplPass>());
        passes.emplace_back(std::make_unique<LstmCellImplPass>());
        pi::tensorlib::passes::Transform(plan, passes);

        Executor executor{plan, execution_backend, 0};
        executor.execute(allocator_registry);

        const auto y_opt = executor.getOutput(result.output);
        const auto hn_opt = executor.getOutput(result.h_n);
        const auto cn_opt = executor.getOutput(result.c_n);
        if (!y_opt || !hn_opt || !cn_opt)
        {
            throw std::runtime_error("failed to retrieve LSTM outputs");
        }

        try
        {
            testing::AssertSimilar(expected_y, *y_opt, 1e-1);
            testing::AssertSimilar(expected_hn, *hn_opt, 1e-1);
            testing::AssertSimilar(expected_cn, *cn_opt, 1e-1);
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error("scenario recompute_interval=" + std::to_string(recompute_interval) +
                                     ", streaming_chunk_size=" + std::to_string(streaming_chunk) +
                                     " failed: " + e.what());
        }
    }
    return 0;
}
