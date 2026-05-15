#include "testing.h"

#include <execution_backend.h>
#include <executor.h>
#include <functional.h>
#include <op_graph.h>
#include <passes.h>
#include <safe_tensors.h>
#include <tensorlib.h>

#include <stdexcept>
#include <string>

using namespace pi::tensorlib;

namespace
{
    constexpr Device DEVICE_CPU{DeviceType::CPU, 0};
    constexpr Device DEVICE_GPU{DeviceType::GPU, 0};
} // namespace

int main()
{
    ExecutionBackend &execution_backend = ExecutionBackend::getInstance();
    const auto &allocator_registry = allocator::DefaultAllocatorRegistry::instance();
    const auto main_stream_desc = GpuStreamDescriptors::Main;

    const auto tensors = safetensors::Load("reference.safetensors");
    const auto fetch = [&tensors](const char *name) -> const std::shared_ptr<RealTensor> &
    {
        auto it = tensors.find(name);
        if (it == tensors.end())
        {
            throw std::runtime_error(std::string("reference.safetensors missing tensor: ") + name);
        }
        return it->second;
    };

    constexpr uint64_t BATCH = 8;
    constexpr uint64_t HIDDEN = 16;

    const auto run_fp16_case = [&]()
    {
        const auto &gates_host = fetch("gates");
        const auto &cprev_host = fetch("c_prev");
        const auto &expected_h = fetch("expected_h");
        const auto &expected_c = fetch("expected_c");

        TraceTensor gates_cpu =
            TraceTensor::Create({BATCH, 4 * HIDDEN}, DataType::FLOAT16, DEVICE_CPU, main_stream_desc);
        TraceTensor cprev_cpu =
            TraceTensor::Create({BATCH, HIDDEN}, DataType::FLOAT16, DEVICE_CPU, main_stream_desc);
        gates_cpu.markRetained();
        cprev_cpu.markRetained();

        OpGraph graph({{.name = "gates_fp16", .tensor = gates_cpu}, {.name = "cprev_fp16", .tensor = cprev_cpu}}, {});

        TraceTensor gates_gpu =
            graph.createTensor({BATCH, 4 * HIDDEN}, DataType::FLOAT16, DEVICE_GPU, main_stream_desc, false);
        TraceTensor cprev_gpu =
            graph.createTensor({BATCH, HIDDEN}, DataType::FLOAT16, DEVICE_GPU, main_stream_desc, false);

        DeviceCopy(graph, gates_cpu, gates_gpu, GpuStreamDescriptors::Main);
        DeviceCopy(graph, cprev_cpu, cprev_gpu, GpuStreamDescriptors::Main);

        auto [h_gpu, c_gpu] = LstmCellFwd(graph, gates_gpu, cprev_gpu, main_stream_desc);

        TraceTensor h_cpu = graph.createTensor(h_gpu.shape().dims(), h_gpu.dtype(), DEVICE_CPU, main_stream_desc, false);
        TraceTensor c_cpu = graph.createTensor(c_gpu.shape().dims(), c_gpu.dtype(), DEVICE_CPU, main_stream_desc, false);

        DeviceCopy(graph, h_gpu, h_cpu, GpuStreamDescriptors::Main);
        DeviceCopy(graph, c_gpu, c_cpu, GpuStreamDescriptors::Main);

        graph.finalize();

        ExecutionPlan plan = ExecutionPlan::FromGraph(
            graph, {{.name = "gates_fp16", .tensor = gates_host}, {.name = "cprev_fp16", .tensor = cprev_host}}, {});

        std::vector<std::unique_ptr<passes::CompilerPass>> passes{};
        passes.emplace_back(std::make_unique<CastImplPass>());
        passes.emplace_back(std::make_unique<LstmCellImplPass>());
        pi::tensorlib::passes::Transform(plan, passes);

        Executor executor{plan, execution_backend, 0};
        executor.execute(allocator_registry);

        const auto actual_h_opt = executor.getOutput(h_cpu);
        const auto actual_c_opt = executor.getOutput(c_cpu);
        if (!actual_h_opt || !actual_c_opt)
        {
            throw std::runtime_error("Failed to retrieve LSTM cell outputs (fp16 case)");
        }

        testing::AssertSimilar(expected_h, *actual_h_opt, 1e-3);
        testing::AssertSimilar(expected_c, *actual_c_opt, 1e-3);
    };

    const auto run_fp32_state_case = [&]()
    {
        const auto &gates_host = fetch("gates_fp32_state");
        const auto &cprev_host = fetch("c_prev_fp32_state");
        const auto &expected_h = fetch("expected_h_fp32_state");
        const auto &expected_c = fetch("expected_c_fp32_state");

        TraceTensor gates_cpu =
            TraceTensor::Create({BATCH, 4 * HIDDEN}, DataType::FLOAT32, DEVICE_CPU, main_stream_desc);
        TraceTensor cprev_cpu =
            TraceTensor::Create({BATCH, HIDDEN}, DataType::FLOAT32, DEVICE_CPU, main_stream_desc);
        gates_cpu.markRetained();
        cprev_cpu.markRetained();

        OpGraph graph({{.name = "gates_fp32", .tensor = gates_cpu}, {.name = "cprev_fp32", .tensor = cprev_cpu}}, {});

        TraceTensor gates_gpu =
            graph.createTensor({BATCH, 4 * HIDDEN}, DataType::FLOAT32, DEVICE_GPU, main_stream_desc, false);
        TraceTensor cprev_gpu =
            graph.createTensor({BATCH, HIDDEN}, DataType::FLOAT32, DEVICE_GPU, main_stream_desc, false);

        DeviceCopy(graph, gates_cpu, gates_gpu, GpuStreamDescriptors::Main);
        DeviceCopy(graph, cprev_cpu, cprev_gpu, GpuStreamDescriptors::Main);

        TraceTensor h_gpu =
            graph.createTensor({BATCH, HIDDEN}, DataType::FLOAT16, DEVICE_GPU, main_stream_desc, false);
        TraceTensor c_gpu =
            graph.createTensor({BATCH, HIDDEN}, DataType::FLOAT32, DEVICE_GPU, main_stream_desc, false);
        TraceTensor y_gpu =
            graph.createTensor({BATCH, HIDDEN}, DataType::FLOAT16, DEVICE_GPU, main_stream_desc, false);

        LstmCellFwdInplace(graph, gates_gpu, cprev_gpu, h_gpu, c_gpu, y_gpu, main_stream_desc);

        TraceTensor h_cpu = graph.createTensor(h_gpu.shape().dims(), h_gpu.dtype(), DEVICE_CPU, main_stream_desc, false);
        TraceTensor y_cpu = graph.createTensor(y_gpu.shape().dims(), y_gpu.dtype(), DEVICE_CPU, main_stream_desc, false);
        TraceTensor c_cpu = graph.createTensor(c_gpu.shape().dims(), c_gpu.dtype(), DEVICE_CPU, main_stream_desc, false);

        DeviceCopy(graph, h_gpu, h_cpu, GpuStreamDescriptors::Main);
        DeviceCopy(graph, y_gpu, y_cpu, GpuStreamDescriptors::Main);
        DeviceCopy(graph, c_gpu, c_cpu, GpuStreamDescriptors::Main);

        graph.finalize();

        ExecutionPlan plan = ExecutionPlan::FromGraph(
            graph, {{.name = "gates_fp32", .tensor = gates_host}, {.name = "cprev_fp32", .tensor = cprev_host}}, {});

        std::vector<std::unique_ptr<passes::CompilerPass>> passes{};
        passes.emplace_back(std::make_unique<CastImplPass>());
        passes.emplace_back(std::make_unique<LstmCellImplPass>());
        pi::tensorlib::passes::Transform(plan, passes);

        Executor executor{plan, execution_backend, 0};
        executor.execute(allocator_registry);

        const auto actual_h_opt = executor.getOutput(h_cpu);
        const auto actual_y_opt = executor.getOutput(y_cpu);
        const auto actual_c_opt = executor.getOutput(c_cpu);
        if (!actual_h_opt || !actual_c_opt || !actual_y_opt)
        {
            throw std::runtime_error("Failed to retrieve LSTM cell outputs (fp32 state case)");
        }

        testing::AssertSimilar(expected_h, *actual_h_opt, 1e-3);
        testing::AssertSimilar(expected_h, *actual_y_opt, 1e-3);
        testing::AssertSimilar(expected_c, *actual_c_opt, 1e-4);
    };

    run_fp16_case();
    run_fp32_state_case();
    return 0;
}
