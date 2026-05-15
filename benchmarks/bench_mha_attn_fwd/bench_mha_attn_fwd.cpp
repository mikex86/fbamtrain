#include "benchmark_utils.h"

#include <allocator.h>
#include <execution_backend.h>
#include <executor.h>
#include <functional.h>
#include <passes.h>
#include <tensorlib.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

namespace
{
    constexpr uint32_t BATCH = 32;
    constexpr uint32_t HEADS = 8;
    constexpr uint32_t SEQ_LEN = 4096;
    constexpr uint32_t HEAD_DIM = 128;

    void ApplyDefaultPasses(pi::tensorlib::ExecutionPlan &execution_plan)
    {
        std::vector<std::unique_ptr<pi::tensorlib::passes::CompilerPass>> passes{};
        passes.emplace_back(std::make_unique<MhaAttentionImplPass>());
        passes.emplace_back(std::make_unique<FillZerosImplPass>());
        passes.emplace_back(std::make_unique<FillUniformImplPass>());
        pi::tensorlib::passes::Transform(execution_plan, passes);
    }

    double ComputeAttentionFlops()
    {
        const double batch = static_cast<double>(BATCH);
        const double heads = static_cast<double>(HEADS);
        const double seq = static_cast<double>(SEQ_LEN);
        const double head_dim = static_cast<double>(HEAD_DIM);
        // Two batched GEMMs: Q*K^T and Attn*V. Each costs 2 * head_dim flops per output element.
        return 4.0 * batch * heads * seq * seq * head_dim;
    }
} // namespace

int main()
{
    const bench_utils::BenchmarkConfig config = bench_utils::LoadBenchmarkConfig();
    const auto main_stream_desc = pi::tensorlib::GpuStreamDescriptors::Main;

    pi::tensorlib::ExecutionBackend &backend = pi::tensorlib::ExecutionBackend::getInstance();
    auto &allocator_registry = pi::tensorlib::allocator::CachingAllocatorRegistry::instance();

    const pi::tensorlib::Device device{
        .device_type = pi::tensorlib::DeviceType::GPU,
        .ordinal = 0,
    };

    pi::tensorlib::OpGraph init_graph{{}, {}};
    pi::tensorlib::TraceTensor q =
        init_graph.createTensor({BATCH, SEQ_LEN, HEADS, HEAD_DIM}, config.dtype, device, main_stream_desc, false);
    pi::tensorlib::TraceTensor k =
        init_graph.createTensor({BATCH, SEQ_LEN, HEADS, HEAD_DIM}, config.dtype, device, main_stream_desc, false);
    pi::tensorlib::TraceTensor v =
        init_graph.createTensor({BATCH, SEQ_LEN, HEADS, HEAD_DIM}, config.dtype, device, main_stream_desc, false);
    q.markRetained();
    k.markRetained();
    v.markRetained();

    uint32_t seed = 42;
    pi::tensorlib::FillUniform(init_graph, q, -0.5f, 0.5f, seed++, main_stream_desc);
    pi::tensorlib::FillUniform(init_graph, k, -0.5f, 0.5f, seed++, main_stream_desc);
    pi::tensorlib::FillUniform(init_graph, v, -0.5f, 0.5f, seed++, main_stream_desc);
    init_graph.finalize();

    pi::tensorlib::ExecutionPlan init_plan = pi::tensorlib::ExecutionPlan::FromGraph(init_graph, {}, {});
    ApplyDefaultPasses(init_plan);
    pi::tensorlib::Executor init_executor{init_plan, backend, 0};
    init_executor.execute(allocator_registry);

    const auto q_real_opt = init_executor.getOutput(q);
    const auto k_real_opt = init_executor.getOutput(k);
    const auto v_real_opt = init_executor.getOutput(v);
    if (!q_real_opt || !k_real_opt || !v_real_opt)
    {
        throw std::runtime_error("Failed to fetch attention inputs after init execution");
    }

    pi::tensorlib::OpGraph graph{
        {
            {.name = "q", .tensor = q},
            {.name = "k", .tensor = k},
            {.name = "v", .tensor = v},
        },
        {}};

    const float softmax_scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(HEAD_DIM)));
    pi::tensorlib::TraceTensor output = pi::tensorlib::ScaledDotProductAttentionFwd(
        graph, q, k, v, softmax_scale, false, /*use_fp16_flash_attn_acc=*/false, std::nullopt, main_stream_desc);
    graph.finalize();

    pi::tensorlib::ExecutionPlan plan = pi::tensorlib::ExecutionPlan::FromGraph(
        graph,
        {
            {.name = "q", .tensor = *q_real_opt},
            {.name = "k", .tensor = *k_real_opt},
            {.name = "v", .tensor = *v_real_opt},
        },
        {});
    ApplyDefaultPasses(plan);

    pi::tensorlib::Executor executor{plan, backend, 0};

    for (std::size_t i = 0; i < config.warmup_runs; ++i)
    {
        executor.execute(allocator_registry);
    }

    auto start_event = pi::tensorlib::ExecutionBackend::CreateEvent(device, true);
    auto end_event = pi::tensorlib::ExecutionBackend::CreateEvent(device, true);

    const auto stream_bundle = pi::tensorlib::ExecutionBackend::GetStreamBundle(device);
    const auto main_stream = stream_bundle->main_stream;

    start_event.record(main_stream);
    for (std::size_t i = 0; i < config.measure_runs; ++i)
    {
        executor.execute(allocator_registry, false);
    }
    end_event.record(main_stream);

    const double total_ms = (config.measure_runs > 0) ? end_event.elapsedMsSince(start_event) : 0.0;

    executor.await();

    const double avg_ms = total_ms / static_cast<double>(config.measure_runs);
    const double avg_sec = avg_ms / 1e3;
    const double flops_per_iter = ComputeAttentionFlops();
    const double tflops = flops_per_iter / avg_sec / 1e12;

    const auto output_real_opt = executor.getOutput(output);
    if (!output_real_opt)
    {
        throw std::runtime_error("Failed to fetch attention output");
    }

    bench_utils::PrintResult("mha_attention_fwd", config, avg_ms, tflops, flops_per_iter);
    return 0;
}
