#include "benchmark_utils.h"

#include <allocator.h>
#include <execution_backend.h>
#include <executor.h>
#include <functional.h>
#include <passes.h>
#include <tensorlib.h>

#include <any>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace
{
    constexpr uint32_t M_DIM = 4096;
    constexpr uint32_t N_DIM = 4096;
    constexpr uint32_t K_DIM = 4096;

    void ApplyDefaultPasses(pi::tensorlib::ExecutionPlan &execution_plan)
    {
        std::vector<std::unique_ptr<pi::tensorlib::passes::CompilerPass>> passes{};
        passes.emplace_back(std::make_unique<MatmulFusePass>());
        passes.emplace_back(std::make_unique<MatmulImplPass>());
        passes.emplace_back(std::make_unique<FillZerosImplPass>());
        passes.emplace_back(std::make_unique<FillUniformImplPass>());
        pi::tensorlib::passes::Transform(execution_plan, passes);
    }

    double ComputeMatmulFlops()
    {
        const double m = static_cast<double>(M_DIM);
        const double n = static_cast<double>(N_DIM);
        const double k = static_cast<double>(K_DIM);
        return 2.0 * m * n * k;
    }
} // namespace

int main()
{
    const bench_utils::BenchmarkConfig config = bench_utils::LoadBenchmarkConfig();

    pi::tensorlib::ExecutionBackend &backend = pi::tensorlib::ExecutionBackend::getInstance();
    auto &allocator_registry = pi::tensorlib::allocator::CachingAllocatorRegistry::instance();

    const pi::tensorlib::Device device{
        .device_type = pi::tensorlib::DeviceType::GPU,
        .ordinal = 0,
    };
    const auto main_stream_desc = pi::tensorlib::GpuStreamDescriptors::Main;

    // Initialize matrices A and B.
    pi::tensorlib::OpGraph init_graph{{}, {}};
    pi::tensorlib::TraceTensor a = init_graph.createTensor({M_DIM, K_DIM}, config.dtype, device, main_stream_desc, false);
    pi::tensorlib::TraceTensor b = init_graph.createTensor({K_DIM, N_DIM}, config.dtype, device, main_stream_desc, false);
    a.markRetained();
    b.markRetained();

    uint32_t seed = 42;
    pi::tensorlib::FillUniform(init_graph, a, -0.5f, 0.5f, seed++, main_stream_desc);
    pi::tensorlib::FillUniform(init_graph, b, -0.5f, 0.5f, seed++, main_stream_desc);
    init_graph.finalize();

    pi::tensorlib::ExecutionPlan init_plan = pi::tensorlib::ExecutionPlan::FromGraph(init_graph, {}, {});
    ApplyDefaultPasses(init_plan);
    pi::tensorlib::Executor init_executor{init_plan, backend, 0};
    init_executor.execute(allocator_registry);

    const auto a_real_opt = init_executor.getOutput(a);
    const auto b_real_opt = init_executor.getOutput(b);
    if (!a_real_opt || !b_real_opt)
    {
        throw std::runtime_error("Failed to materialize matmul inputs after initialization");
    }

    // Build matmul graph.
    pi::tensorlib::OpGraph graph{{
                                     {.name = "a", .tensor = a},
                                     {.name = "b", .tensor = b},
                                 },
                                 {}};

    pi::tensorlib::TraceTensor matmul_out =
        graph.createTensor({M_DIM, N_DIM}, config.dtype, device, main_stream_desc, false);
    std::unordered_map<std::string, std::any> matmul_attrs{};
    if (config.dtype == pi::tensorlib::DataType::FLOAT16)
    {
        if (const char *env = std::getenv("BENCH_FP16_ACC"))
        {
            const std::string_view value{env};
            if (!(value.empty() || value == "0" || value == "false"))
            {
                matmul_attrs.emplace("use_fp16_matmul_acc", true);
            }
        }
    }
    graph.recordOperation(pi::tensorlib::OperationEntry{.type = pi::tensorlib::OpType::MATMUL,
                                                        .inputs = {a, b},
                                                        .outputs = {matmul_out},
                                                        .attributes = std::move(matmul_attrs),
                                                        .gpu_stream_desc = main_stream_desc});
    graph.finalize();

    pi::tensorlib::ExecutionPlan plan =
        pi::tensorlib::ExecutionPlan::FromGraph(graph,
                                                {
                                                    {.name = "a", .tensor = *a_real_opt},
                                                    {.name = "b", .tensor = *b_real_opt},
                                                },
                                                {});
    ApplyDefaultPasses(plan);

    pi::tensorlib::Executor executor{plan, backend, 0};

    // Warmup runs.
    for (std::size_t i = 0; i < config.warmup_runs; ++i)
    {
        executor.execute(allocator_registry);
    }

    // Timed runs.
    pi::tensorlib::GpuEvent start_event = pi::tensorlib::ExecutionBackend::CreateEvent(device, true);
    pi::tensorlib::GpuEvent end_event = pi::tensorlib::ExecutionBackend::CreateEvent(device, true);

    const auto stream_bundle = pi::tensorlib::ExecutionBackend::GetStreamBundle(device);
    auto main_stream = stream_bundle->main_stream;

    auto start_time = std::chrono::high_resolution_clock::now();
    start_event.record(main_stream);
    for (std::size_t i = 0; i < config.measure_runs; ++i)
    {
        executor.execute(allocator_registry, /*awaitExecution=*/false);
    }
    end_event.record(main_stream);
    end_event.synchronize();
    executor.await();
    auto end_time = std::chrono::high_resolution_clock::now();

    const double event_ms = end_event.elapsedMsSince(start_event);
    auto total_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1e3;

    const double avg_ms_host = total_ms / static_cast<double>(config.measure_runs);
    const double avg_ms_device = event_ms / static_cast<double>(config.measure_runs);
    const double avg_sec = avg_ms_device / 1e3;
    const double flops_per_iter = ComputeMatmulFlops();
    const double tflops = flops_per_iter / avg_sec / 1e12;

    // Materialize output once to ensure execution completes eagerly.
    const auto output_real_opt = executor.getOutput(matmul_out);
    if (!output_real_opt)
    {
        throw std::runtime_error("Failed to fetch matmul output");
    }

    bench_utils::PrintResult("matmul", config, avg_ms_device, tflops, flops_per_iter);
    std::cout << "avg_ms_host=" << avg_ms_host << " avg_ms_device=" << avg_ms_device << std::endl;
    return 0;
}
