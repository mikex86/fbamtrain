#include <chrono>
#include <iostream>
#include <vector>

#include <allocator.h>
#include <execution_backend.h>
#include <executor.h>
#include <functional.h>
#include <mfu.h>
#include <op_graph.h>
#include <passes.h>
#include <tensorlib.h>
#include <testing.h>

#if PI_TENSORLIB_ENABLE_CUDA
#include <cuda_runtime.h>
#elif PI_TENSORLIB_ENABLE_HIP
#include <hip/hip_runtime.h>
#endif

using namespace pi::tensorlib;

namespace
{
    constexpr Device DEVICE_CPU{DeviceType::CPU, 0};
    constexpr Device DEVICE_GPU{DeviceType::GPU, 0};
    const auto main_stream_desc = GpuStreamDescriptors::Main;

    struct BenchmarkConfig
    {
        int T{512};
        int B{1024};
        int I{1024};
        int H{1024};
        int streaming_chunk{8};
        int recompute_interval{1};
    };

    void Warmup(Executor &executor, const allocator::AllocatorRegistry &alloc_registry, int iters)
    {
        for (int i = 0; i < iters; ++i)
        {
            executor.execute(alloc_registry, /*awaitExecution=*/true);
        }
    }

    double RunTimed(Executor &executor, const allocator::AllocatorRegistry &alloc_registry, int iters)
    {
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i)
        {
            executor.execute(alloc_registry, /*awaitExecution=*/true);
        }
        auto end = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(end - start).count();
    }

    ExecutionPlan BuildPlan(const BenchmarkConfig &cfg, ExecutionBackend &backend)
    {
        const uint64_t T = static_cast<uint64_t>(cfg.T);
        const uint64_t B = static_cast<uint64_t>(cfg.B);
        const uint64_t I = static_cast<uint64_t>(cfg.I);
        const uint64_t H = static_cast<uint64_t>(cfg.H);
        const uint64_t gate_dim = 4 * H;

        // Host inputs / caches
        auto x_host = RealTensor::Allocate({T, B, I}, DataType::FLOAT16, DEVICE_CPU, true);
        auto h_cache_host = RealTensor::Allocate({T, B, H}, DataType::FLOAT16, DEVICE_CPU, true);
        const uint64_t c_cache_steps =
            (T + static_cast<uint64_t>(cfg.recompute_interval) - 1) / static_cast<uint64_t>(cfg.recompute_interval);
        auto c_cache_host = RealTensor::Allocate({c_cache_steps, B, H}, DataType::FLOAT32, DEVICE_CPU, true);
        auto y_cache_host = RealTensor::Allocate({T, B, H}, DataType::FLOAT16, DEVICE_CPU, true);
        auto dy_host = RealTensor::Allocate({T, B, H}, DataType::FLOAT32, DEVICE_CPU, true);

        // Device states / params
        auto h0_device = RealTensor::Allocate({H}, DataType::FLOAT32, DEVICE_GPU);
        auto c0_device = RealTensor::Allocate({H}, DataType::FLOAT32, DEVICE_GPU);
        auto w_ih = RealTensor::Allocate({I, gate_dim}, DataType::FLOAT32, DEVICE_GPU);
        auto w_hh = RealTensor::Allocate({H, gate_dim}, DataType::FLOAT32, DEVICE_GPU);
        auto b_ih = RealTensor::Allocate({gate_dim}, DataType::FLOAT32, DEVICE_GPU);
        auto b_hh = RealTensor::Allocate({gate_dim}, DataType::FLOAT32, DEVICE_GPU);

        std::vector<GraphInputDescriptor> inputs{};
        inputs.push_back({"x", TraceTensor::Create(x_host->shape().dims(), x_host->dtype(), DEVICE_CPU,
                                                   main_stream_desc, /*pinned=*/true)});
        inputs.push_back(
            {"h0", TraceTensor::Create(h0_device->shape().dims(), h0_device->dtype(), DEVICE_GPU, main_stream_desc)});
        inputs.push_back(
            {"c0", TraceTensor::Create(c0_device->shape().dims(), c0_device->dtype(), DEVICE_GPU, main_stream_desc)});
        inputs.push_back({"y_cache", TraceTensor::Create(y_cache_host->shape().dims(), y_cache_host->dtype(),
                                                         DEVICE_CPU, main_stream_desc, /*pinned=*/true)});
        inputs.push_back({"h_cache", TraceTensor::Create(h_cache_host->shape().dims(), h_cache_host->dtype(),
                                                         DEVICE_CPU, main_stream_desc, /*pinned=*/true)});
        inputs.push_back({"c_cache", TraceTensor::Create(c_cache_host->shape().dims(), c_cache_host->dtype(),
                                                         DEVICE_CPU, main_stream_desc, /*pinned=*/true)});
        inputs.push_back({"dy", TraceTensor::Create(dy_host->shape().dims(), dy_host->dtype(), DEVICE_CPU,
                                                    main_stream_desc, /*pinned=*/true)});

        std::vector<GraphInputDescriptor> params{};
        params.push_back(
            {"w_ih", TraceTensor::Create(w_ih->shape().dims(), w_ih->dtype(), DEVICE_GPU, main_stream_desc)});
        params.push_back(
            {"w_hh", TraceTensor::Create(w_hh->shape().dims(), w_hh->dtype(), DEVICE_GPU, main_stream_desc)});
        params.push_back(
            {"b_ih", TraceTensor::Create(b_ih->shape().dims(), b_ih->dtype(), DEVICE_GPU, main_stream_desc)});
        params.push_back(
            {"b_hh", TraceTensor::Create(b_hh->shape().dims(), b_hh->dtype(), DEVICE_GPU, main_stream_desc)});

        for (auto &entry : inputs)
        {
            entry.tensor.markRetained();
        }
        for (auto &entry : params)
        {
            entry.tensor.markRetained();
        }

        OpGraph graph(inputs, params);

        auto lstm_result = StreamingLstmBwd(graph, inputs[0].tensor, inputs[1].tensor, inputs[2].tensor,
                                            params[0].tensor, params[1].tensor, params[2].tensor, params[3].tensor,
                                            inputs[3].tensor, // y_cache (unused but required)
                                            inputs[4].tensor, inputs[5].tensor, inputs[6].tensor, std::nullopt,
                                            std::nullopt, cfg.recompute_interval, cfg.streaming_chunk, std::nullopt,
                                            main_stream_desc);
        (void)lstm_result;

        graph.finalize();

        std::vector<GraphExecutionInputDescriptor> exec_inputs{};
        exec_inputs.push_back({"x", x_host});
        exec_inputs.push_back({"h0", h0_device});
        exec_inputs.push_back({"c0", c0_device});
        exec_inputs.push_back({"y_cache", y_cache_host});
        exec_inputs.push_back({"h_cache", h_cache_host});
        exec_inputs.push_back({"c_cache", c_cache_host});
        exec_inputs.push_back({"dy", dy_host});

        std::vector<GraphExecutionInputDescriptor> exec_params{};
        exec_params.push_back({"w_ih", w_ih});
        exec_params.push_back({"w_hh", w_hh});
        exec_params.push_back({"b_ih", b_ih});
        exec_params.push_back({"b_hh", b_hh});

        ExecutionPlan plan = ExecutionPlan::FromGraph(graph, exec_inputs, exec_params);

        std::vector<std::unique_ptr<passes::CompilerPass>> passes{};
        passes.emplace_back(std::make_unique<MatmulFusePass>());
        passes.emplace_back(std::make_unique<MatmulImplPass>());
        passes.emplace_back(std::make_unique<FillZerosImplPass>());
        passes.emplace_back(std::make_unique<FillConstantImplPass>());
        passes.emplace_back(std::make_unique<ActImplPass>());
        passes.emplace_back(std::make_unique<CastImplPass>());
        passes.emplace_back(std::make_unique<ContiguousImplPass>());
        passes.emplace_back(std::make_unique<TrailingBroadcastAddImplPass>());
        passes.emplace_back(std::make_unique<ReduceSumImplPass>());
        passes.emplace_back(std::make_unique<LstmCellImplPass>());
        passes.emplace_back(std::make_unique<LstmCellBwdImplPass>());
        pi::tensorlib::passes::Transform(plan, passes);

        return plan;
    }
} // namespace

int main()
{
    BenchmarkConfig cfg{};
    ExecutionBackend &backend = ExecutionBackend::getInstance();
    // Establish the target GPU context before any pinned host allocation.
    (void)ExecutionBackend::GetStreamBundle(DEVICE_GPU);
    const auto &alloc_registry = allocator::CachingAllocatorRegistry::instance();

    auto plan = BuildPlan(cfg, backend);
    const double total_flops = static_cast<double>(plan.totalFlops());
    const double useful_flops = static_cast<double>(plan.totalUsefulFlops());

    Executor executor{plan, backend, 0};

    Warmup(executor, alloc_registry, /*iters=*/3);
    const int bench_iters = 5;
    const double seconds = RunTimed(executor, alloc_registry, bench_iters);
    const double time_per_iter = seconds / static_cast<double>(bench_iters);
    const double useful_tflops = useful_flops / time_per_iter / 1e12;
    const double total_tflops = total_flops / time_per_iter / 1e12;

    const auto promised_tflops = GetPromisedTFlops(0, PrecisionMode::FP16);
    const double mfu = promised_tflops ? (useful_tflops / static_cast<double>(*promised_tflops) * 100.0) : -1.0;
    const double hfu = promised_tflops ? (total_tflops / static_cast<double>(*promised_tflops) * 100.0) : -1.0;

    std::cout << "Streaming LSTM backward benchmark\n";
    std::cout << "T=" << cfg.T << " B=" << cfg.B << " I=" << cfg.I << " H=" << cfg.H << "\n";
    std::cout << "Time/iter (s): " << time_per_iter << "\n";
    std::cout << "Useful FLOPs: " << useful_flops << "\n";
    std::cout << "Useful TFLOPs/s: " << useful_tflops << "\n";
    std::cout << "Promised TFLOPs/s: " << promised_tflops.value_or(-1.0f) << "\n";
    std::cout << "MFU (%): " << mfu << "\n";
    std::cout << "HFU (%): " << hfu << "\n";
    std::cout << "Total FLOPs: " << total_flops << "\n";

    return 0;
}
