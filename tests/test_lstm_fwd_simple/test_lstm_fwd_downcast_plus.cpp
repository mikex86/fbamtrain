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
    constexpr int kReferenceTIndex = 3;
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
    const auto expected_y_t = fetch("y_t");
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

    OpGraph graph(inputs, params);

    const int recompute_interval = 2;
    const int streaming_chunk = 2;
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
    passes.emplace_back(std::make_unique<MatmulFusePass>());
    passes.emplace_back(std::make_unique<MatmulImplPass>());
    passes.emplace_back(std::make_unique<ActImplPass>());
    passes.emplace_back(std::make_unique<CastImplPass>());
    passes.emplace_back(std::make_unique<TrailingBroadcastAddImplPass>());
    passes.emplace_back(std::make_unique<LstmCellImplPass>());
    pi::tensorlib::passes::Transform(plan, passes);

    for (const auto &entry : plan.entries)
    {
        if (entry.op_type.has_value() && entry.op_type == pi::tensorlib::OpType::PLUS)
        {
            throw std::runtime_error("PLUS op was not lowered to a kernel");
        }
    }

    Executor executor{plan, execution_backend, 0};
    executor.execute(allocator_registry);

    const auto y_opt = executor.getOutput(result.output);
    const auto hn_opt = executor.getOutput(result.h_n);
    const auto cn_opt = executor.getOutput(result.c_n);
    if (!y_opt || !hn_opt || !cn_opt)
    {
        throw std::runtime_error("failed to retrieve LSTM outputs");
    }

    const auto y_t_opt = (*y_opt)->at(0, kReferenceTIndex);
    if (!y_t_opt)
    {
        throw std::runtime_error("failed to slice output at requested timestep");
    }

    testing::AssertSimilar(expected_y, *y_opt, 1e-2);
    testing::AssertSimilar(expected_y_t, y_t_opt, 1e-2);
    testing::AssertSimilar(expected_hn, *hn_opt, 1e-2);
    testing::AssertSimilar(expected_cn, *cn_opt, 1e-2);
    return 0;
}
