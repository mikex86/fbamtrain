#include <allocator.h>
#include <execution_backend.h>
#include <executor.h>
#include <functional.h>
#include <mfu.h>
#include <passes.h>
#include <tensorlib.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace
{
    constexpr uint32_t BATCH = 8;
    constexpr uint32_t HEADS = 8;
    constexpr uint32_t SEQ_LEN = 1024;
    constexpr uint32_t HEAD_DIM = 128;
    constexpr std::size_t WARMUP_ITERS = 8;
    constexpr std::size_t MEASURE_ITERS = 32;
    const auto main_stream_desc = pi::tensorlib::GpuStreamDescriptors::Main;

    struct RunResult
    {
        std::string label;
        double avg_ms{};
        double tflops{};
        double mfu{};
    };

    void ApplyDefaultPasses(pi::tensorlib::ExecutionPlan &plan)
    {
        std::vector<std::unique_ptr<pi::tensorlib::passes::CompilerPass>> passes{};
        passes.emplace_back(std::make_unique<MhaAttentionImplPass>());
        passes.emplace_back(std::make_unique<TrailingBroadcastMulImplPass>());
        passes.emplace_back(std::make_unique<TrailingBroadcastAddImplPass>());
        passes.emplace_back(std::make_unique<ReduceSumImplPass>());
        passes.emplace_back(std::make_unique<ContiguousImplPass>());
        passes.emplace_back(std::make_unique<CastImplPass>());
        passes.emplace_back(std::make_unique<FillConstantImplPass>());
        passes.emplace_back(std::make_unique<FillZerosImplPass>());
        passes.emplace_back(std::make_unique<FillUniformImplPass>());
        pi::tensorlib::passes::Transform(plan, passes);
    }

    double ComputeAttentionFlops()
    {
        const double batch = static_cast<double>(BATCH);
        const double heads = static_cast<double>(HEADS);
        const double seq = static_cast<double>(SEQ_LEN);
        const double head_dim = static_cast<double>(HEAD_DIM);
        return 4.0 * batch * heads * seq * seq * head_dim;
    }

    double ComputeAttentionBwdFlops()
    {
        const double batch = static_cast<double>(BATCH);
        const double heads = static_cast<double>(HEADS);
        const double seq = static_cast<double>(SEQ_LEN);
        const double head_dim = static_cast<double>(HEAD_DIM);
        return 6.0 * batch * heads * seq * seq * head_dim;
    }

    enum class MhaBackend
    {
        FlashAttention,
        Cutlass,
        Triton,
    };

    const char *BackendLabel(const MhaBackend backend)
    {
        switch (backend)
        {
            case MhaBackend::FlashAttention:
                return "flash";
            case MhaBackend::Cutlass:
                return "cutlass";
            case MhaBackend::Triton:
                return "triton";
        }
        return "unknown";
    }

    void ConfigureMhaBackend(const MhaBackend backend)
    {
        const char *env_name = "FBAMTRAIN_PREFER_MHA_BACKEND";
        switch (backend)
        {
            case MhaBackend::FlashAttention:
                setenv(env_name, "flash", 1);
                break;
            case MhaBackend::Cutlass:
                setenv(env_name, "cutlass", 1);
                break;
            case MhaBackend::Triton:
                setenv(env_name, "triton", 1);
                break;
        }
    }

    struct Inputs
    {
        std::shared_ptr<pi::tensorlib::RealTensor> q;
        std::shared_ptr<pi::tensorlib::RealTensor> k;
        std::shared_ptr<pi::tensorlib::RealTensor> v;
        std::shared_ptr<pi::tensorlib::RealTensor> upstream;
    };

    Inputs InitializeInputs(const pi::tensorlib::Device &device, const pi::tensorlib::DataType dtype)
    {
        pi::tensorlib::OpGraph init_graph{{}, {}};
        auto q = init_graph.createTensor({BATCH, SEQ_LEN, HEADS, HEAD_DIM}, dtype, device, main_stream_desc, false);
        auto k = init_graph.createTensor({BATCH, SEQ_LEN, HEADS, HEAD_DIM}, dtype, device, main_stream_desc, false);
        auto v = init_graph.createTensor({BATCH, SEQ_LEN, HEADS, HEAD_DIM}, dtype, device, main_stream_desc, false);
        auto upstream =
            init_graph.createTensor({BATCH, SEQ_LEN, HEADS, HEAD_DIM}, dtype, device, main_stream_desc, false);

        uint32_t seed = 42;
        pi::tensorlib::FillUniform(init_graph, q, -0.5f, 0.5f, seed++, main_stream_desc);
        pi::tensorlib::FillUniform(init_graph, k, -0.5f, 0.5f, seed++, main_stream_desc);
        pi::tensorlib::FillUniform(init_graph, v, -0.5f, 0.5f, seed++, main_stream_desc);
        pi::tensorlib::FillUniform(init_graph, upstream, -0.5f, 0.5f, seed++, main_stream_desc);
        init_graph.finalize();

        auto plan = pi::tensorlib::ExecutionPlan::FromGraph(init_graph, {}, {});
        ApplyDefaultPasses(plan);

        auto &backend = pi::tensorlib::ExecutionBackend::getInstance();
        auto &allocator_registry = pi::tensorlib::allocator::CachingAllocatorRegistry::instance();
        pi::tensorlib::Executor executor{plan, backend, 0};
        executor.execute(allocator_registry);

        const auto q_real = executor.getOutput(q);
        const auto k_real = executor.getOutput(k);
        const auto v_real = executor.getOutput(v);
        const auto upstream_real = executor.getOutput(upstream);
        if (!q_real || !k_real || !v_real || !upstream_real)
        {
            throw std::runtime_error("Failed to initialize MHA inputs");
        }
        return Inputs{*q_real, *k_real, *v_real, *upstream_real};
    }

    struct FwdOutputs
    {
        std::shared_ptr<pi::tensorlib::RealTensor> output;
        std::shared_ptr<pi::tensorlib::RealTensor> scratch;
    };

    FwdOutputs MaterializeFwdOutputs(const Inputs &inputs, const float softmax_scale, const bool use_fp16_accum,
                                     const MhaBackend backend)
    {
        using namespace pi::tensorlib;
        ConfigureMhaBackend(backend);
        const Device device{DeviceType::GPU, 0};
        const auto dtype = inputs.q->dtype();

        TraceTensor q = TraceTensor::Create(inputs.q->shape().dims(), dtype, device, main_stream_desc);
        TraceTensor k = TraceTensor::Create(inputs.k->shape().dims(), dtype, device, main_stream_desc);
        TraceTensor v = TraceTensor::Create(inputs.v->shape().dims(), dtype, device, main_stream_desc);
        TraceTensor scratch =
            TraceTensor::Create({BATCH, HEADS, SEQ_LEN}, DataType::FLOAT32, device, main_stream_desc);
        q.markRetained();
        k.markRetained();
        v.markRetained();

        OpGraph graph({{.name = "q", .tensor = q}, {.name = "k", .tensor = k}, {.name = "v", .tensor = v}}, {});
        TraceTensor output = ScaledDotProductAttentionFwd(graph, q, k, v, softmax_scale, false, use_fp16_accum, scratch,
                                                          main_stream_desc);
        graph.finalize();

        ExecutionPlan plan = ExecutionPlan::FromGraph(
            graph,
            {{.name = "q", .tensor = inputs.q}, {.name = "k", .tensor = inputs.k}, {.name = "v", .tensor = inputs.v}},
            {});
        ApplyDefaultPasses(plan);

        auto &exec_backend = ExecutionBackend::getInstance();
        auto &allocator_registry = allocator::CachingAllocatorRegistry::instance();
        Executor executor{plan, exec_backend, 0};
        executor.execute(allocator_registry);

        const auto output_real = executor.getOutput(output);
        const auto scratch_real = executor.getOutput(scratch);
        if (!output_real || !scratch_real)
        {
            throw std::runtime_error("Failed to materialize MHA forward outputs");
        }
        return FwdOutputs{*output_real, *scratch_real};
    }

    RunResult RunFwdBench(const Inputs &inputs, const float softmax_scale, const bool use_fp16_accum,
                          const MhaBackend backend)
    {
        using namespace pi::tensorlib;
        ConfigureMhaBackend(backend);

        const Device device{DeviceType::GPU, 0};
        const auto dtype = inputs.q->dtype();

        TraceTensor q = TraceTensor::Create(inputs.q->shape().dims(), dtype, device, main_stream_desc);
        TraceTensor k = TraceTensor::Create(inputs.k->shape().dims(), dtype, device, main_stream_desc);
        TraceTensor v = TraceTensor::Create(inputs.v->shape().dims(), dtype, device, main_stream_desc);
        TraceTensor output =
            TraceTensor::Create({BATCH, SEQ_LEN, HEADS, HEAD_DIM}, dtype, device, main_stream_desc);
        TraceTensor scratch =
            TraceTensor::Create({BATCH, HEADS, SEQ_LEN}, DataType::FLOAT32, device, main_stream_desc);
        q.markRetained();
        k.markRetained();
        v.markRetained();
        output.markRetained();
        scratch.markRetained();

        OpGraph graph({{.name = "q", .tensor = q},
                       {.name = "k", .tensor = k},
                       {.name = "v", .tensor = v},
                       {.name = "output", .tensor = output},
                       {.name = "scratch", .tensor = scratch}},
                      {});

        std::unordered_map<std::string, std::any> attributes{};
        attributes.emplace("softmax_scale", softmax_scale);
        attributes.emplace("causal", false);
        if (use_fp16_accum)
        {
            attributes.emplace("use_fp16_flash_attn_acc", true);
        }
        graph.recordOperation(OperationEntry{.type = OpType::MHA_ATTN_FWD,
                                             .inputs = {q, k, v},
                                             .outputs = {output, scratch},
                                             .attributes = std::move(attributes),
                                             .gpu_stream_desc = main_stream_desc});
        graph.finalize();

        auto output_real = RealTensor::Allocate({BATCH, SEQ_LEN, HEADS, HEAD_DIM}, dtype, device);
        auto scratch_real = RealTensor::Allocate({BATCH, HEADS, SEQ_LEN}, DataType::FLOAT32, device);

        ExecutionPlan plan = ExecutionPlan::FromGraph(
            graph,
            {{.name = "q", .tensor = inputs.q},
             {.name = "k", .tensor = inputs.k},
             {.name = "v", .tensor = inputs.v},
             {.name = "output", .tensor = output_real},
             {.name = "scratch", .tensor = scratch_real}},
            {});
        ApplyDefaultPasses(plan);

        auto &exec_backend = ExecutionBackend::getInstance();
        auto &allocator_registry = allocator::CachingAllocatorRegistry::instance();
        Executor executor{plan, exec_backend, 0};

        for (std::size_t i = 0; i < WARMUP_ITERS; ++i)
        {
            executor.execute(allocator_registry);
        }

        auto start_event = ExecutionBackend::CreateEvent(device, true);
        auto end_event = ExecutionBackend::CreateEvent(device, true);
        const auto stream_bundle = ExecutionBackend::GetStreamBundle(device);

        start_event.record(stream_bundle->main_stream);
        for (std::size_t i = 0; i < MEASURE_ITERS; ++i)
        {
            executor.execute(allocator_registry, false);
        }
        end_event.record(stream_bundle->main_stream);
        const double total_ms = end_event.elapsedMsSince(start_event);
        executor.await();

        const double avg_ms = total_ms / static_cast<double>(MEASURE_ITERS);
        const double avg_sec = avg_ms / 1e3;
        const double flops = ComputeAttentionFlops();
        const double tflops = flops / avg_sec / 1e12;

        const auto precision = use_fp16_accum ? PrecisionMode::FP16_ACC16 : PrecisionMode::FP16;
        const auto promised = GetPromisedTFlops(device.ordinal, precision);
        const double mfu = promised ? (tflops / static_cast<double>(*promised) * 100.0) : -1.0;

        std::string label = std::string("mha_fwd_") + BackendLabel(backend) + '_' +
                            (use_fp16_accum ? "fp16acc" : "fp32acc");
        std::cout << "[mha_mfu] label=" << label << " avg_ms=" << avg_ms << " tflops=" << tflops << " mfu=" << mfu;
        if (promised)
        {
            std::cout << " promised_tflops=" << *promised;
        }
        std::cout << '\n';

        return RunResult{label, avg_ms, tflops, mfu};
    }

    RunResult RunBwdBench(const Inputs &inputs, const FwdOutputs &fwd, const float softmax_scale,
                          const MhaBackend backend)
    {
        using namespace pi::tensorlib;
        ConfigureMhaBackend(backend);

        const Device device{DeviceType::GPU, 0};
        const auto dtype = inputs.q->dtype();

        TraceTensor q = TraceTensor::Create(inputs.q->shape().dims(), dtype, device, main_stream_desc);
        TraceTensor k = TraceTensor::Create(inputs.k->shape().dims(), dtype, device, main_stream_desc);
        TraceTensor v = TraceTensor::Create(inputs.v->shape().dims(), dtype, device, main_stream_desc);
        TraceTensor output = TraceTensor::Create(fwd.output->shape().dims(), dtype, device, main_stream_desc);
        TraceTensor scratch =
            TraceTensor::Create(fwd.scratch->shape().dims(), DataType::FLOAT32, device, main_stream_desc);
        TraceTensor upstream = TraceTensor::Create(inputs.upstream->shape().dims(), dtype, device, main_stream_desc);
        q.markRetained();
        k.markRetained();
        v.markRetained();
        output.markRetained();
        scratch.markRetained();
        upstream.markRetained();

        OpGraph graph({{.name = "q", .tensor = q},
                       {.name = "k", .tensor = k},
                       {.name = "v", .tensor = v},
                       {.name = "output", .tensor = output},
                       {.name = "scratch", .tensor = scratch},
                       {.name = "upstream", .tensor = upstream}},
                      {});
        (void)ScaledDotProductAttentionBwd(graph, q, k, v, output, scratch, upstream, softmax_scale,
                                           /*causal=*/false, main_stream_desc);
        graph.finalize();

        ExecutionPlan plan = ExecutionPlan::FromGraph(
            graph,
            {
                {.name = "q", .tensor = inputs.q},
                {.name = "k", .tensor = inputs.k},
                {.name = "v", .tensor = inputs.v},
                {.name = "output", .tensor = fwd.output},
                {.name = "scratch", .tensor = fwd.scratch},
                {.name = "upstream", .tensor = inputs.upstream},
            },
            {});
        ApplyDefaultPasses(plan);

        auto &exec_backend = ExecutionBackend::getInstance();
        auto &allocator_registry = allocator::CachingAllocatorRegistry::instance();
        Executor executor{plan, exec_backend, 0};

        for (std::size_t i = 0; i < WARMUP_ITERS; ++i)
        {
            executor.execute(allocator_registry);
        }

        auto start_event = ExecutionBackend::CreateEvent(device, true);
        auto end_event = ExecutionBackend::CreateEvent(device, true);
        const auto stream_bundle = ExecutionBackend::GetStreamBundle(device);

        start_event.record(stream_bundle->main_stream);
        for (std::size_t i = 0; i < MEASURE_ITERS; ++i)
        {
            executor.execute(allocator_registry, false);
        }
        end_event.record(stream_bundle->main_stream);
        const double total_ms = end_event.elapsedMsSince(start_event);
        executor.await();

        const double avg_ms = total_ms / static_cast<double>(MEASURE_ITERS);
        const double avg_sec = avg_ms / 1e3;
        const double flops = ComputeAttentionBwdFlops();
        const double tflops = flops / avg_sec / 1e12;

        const auto promised = GetPromisedTFlops(device.ordinal, PrecisionMode::FP16);
        const double mfu = promised ? (tflops / static_cast<double>(*promised) * 100.0) : -1.0;

        std::string label = std::string("mha_bwd_") + BackendLabel(backend);
        std::cout << "[mha_mfu] label=" << label << " avg_ms=" << avg_ms << " tflops=" << tflops << " mfu=" << mfu;
        if (promised)
        {
            std::cout << " promised_tflops=" << *promised;
        }
        std::cout << '\n';

        return RunResult{label, avg_ms, tflops, mfu};
    }
} // namespace

int main()
{
    using namespace pi::tensorlib;
    const Device device{DeviceType::GPU, 0};
    const DataType dtype = DataType::FLOAT16;
    const float softmax_scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(HEAD_DIM)));

    const auto inputs = InitializeInputs(device, dtype);

    std::vector<RunResult> results{};
    results.reserve(9);

    const bool use_fp16_accum = true;
    const bool use_fp32_accum = false;

    results.push_back(RunFwdBench(inputs, softmax_scale, use_fp32_accum, MhaBackend::FlashAttention));
    results.push_back(RunFwdBench(inputs, softmax_scale, use_fp16_accum, MhaBackend::FlashAttention));
    results.push_back(RunFwdBench(inputs, softmax_scale, use_fp32_accum, MhaBackend::Cutlass));
    results.push_back(RunFwdBench(inputs, softmax_scale, use_fp16_accum, MhaBackend::Cutlass));
    results.push_back(RunFwdBench(inputs, softmax_scale, use_fp32_accum, MhaBackend::Triton));
    results.push_back(RunFwdBench(inputs, softmax_scale, use_fp16_accum, MhaBackend::Triton));

    const auto fwd_flash = MaterializeFwdOutputs(inputs, softmax_scale, use_fp32_accum, MhaBackend::FlashAttention);
    const auto fwd_cutlass = MaterializeFwdOutputs(inputs, softmax_scale, use_fp32_accum, MhaBackend::Cutlass);
    const auto fwd_triton = MaterializeFwdOutputs(inputs, softmax_scale, use_fp32_accum, MhaBackend::Triton);
    results.push_back(RunBwdBench(inputs, fwd_flash, softmax_scale, MhaBackend::FlashAttention));
    results.push_back(RunBwdBench(inputs, fwd_cutlass, softmax_scale, MhaBackend::Cutlass));
    results.push_back(RunBwdBench(inputs, fwd_triton, softmax_scale, MhaBackend::Triton));

    for (const auto &entry : results)
    {
        if (entry.tflops <= 0.0)
        {
            std::cerr << "Invalid MFU result for " << entry.label << '\n';
            return 1;
        }
    }

    return 0;
}
