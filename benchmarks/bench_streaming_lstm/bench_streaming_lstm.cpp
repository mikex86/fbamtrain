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
        // Host inputs
        auto x_host = RealTensor::Allocate(
            {static_cast<uint64_t>(cfg.T), static_cast<uint64_t>(cfg.B), static_cast<uint64_t>(cfg.I)},
            DataType::FLOAT16, DEVICE_CPU, true);
        auto h0_device =
            RealTensor::Allocate({static_cast<uint64_t>(cfg.H)}, DataType::FLOAT32, DEVICE_GPU);
        auto c0_device =
            RealTensor::Allocate({static_cast<uint64_t>(cfg.H)}, DataType::FLOAT32, DEVICE_GPU);

        // Params (fp32)
        auto w_ih = RealTensor::Allocate({static_cast<uint64_t>(cfg.I), static_cast<uint64_t>(4 * cfg.H)},
                                         DataType::FLOAT32, DEVICE_GPU);
        auto w_hh = RealTensor::Allocate({static_cast<uint64_t>(cfg.H), static_cast<uint64_t>(4 * cfg.H)},
                                         DataType::FLOAT32, DEVICE_GPU);
        auto b_ih = RealTensor::Allocate({static_cast<uint64_t>(4 * cfg.H)}, DataType::FLOAT32, DEVICE_GPU);
        auto b_hh = RealTensor::Allocate({static_cast<uint64_t>(4 * cfg.H)}, DataType::FLOAT32, DEVICE_GPU);

        std::vector<GraphInputDescriptor> inputs{};
        inputs.push_back({"x", TraceTensor::Create(x_host->shape().dims(), x_host->dtype(), DEVICE_CPU,
                                                   main_stream_desc, /*pinned=*/true)});
        inputs.push_back(
            {"h0", TraceTensor::Create(h0_device->shape().dims(), h0_device->dtype(), DEVICE_GPU, main_stream_desc)});
        inputs.push_back(
            {"c0", TraceTensor::Create(c0_device->shape().dims(), c0_device->dtype(), DEVICE_GPU, main_stream_desc)});

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

        auto lstm_result =
            StreamingLstmFwd(graph, inputs[0].tensor, inputs[1].tensor, inputs[2].tensor, params[0].tensor,
                             params[1].tensor, params[2].tensor, params[3].tensor, DataType::FLOAT16,
                             /*recompute_interval=*/16, /*streaming_chunk_size=*/cfg.streaming_chunk,
                             main_stream_desc);

        graph.finalize();

        std::vector<GraphExecutionInputDescriptor> exec_inputs{};
        exec_inputs.push_back({"x", x_host});
        exec_inputs.push_back({"h0", h0_device});
        exec_inputs.push_back({"c0", c0_device});
        std::vector<GraphExecutionInputDescriptor> exec_params{};
        exec_params.push_back({"w_ih", w_ih});
        exec_params.push_back({"w_hh", w_hh});
        exec_params.push_back({"b_ih", b_ih});
        exec_params.push_back({"b_hh", b_hh});

        ExecutionPlan plan = ExecutionPlan::FromGraph(graph, exec_inputs, exec_params);

        std::vector<std::unique_ptr<passes::CompilerPass>> passes{};
        passes.emplace_back(std::make_unique<MatmulFusePass>()); // fuse matmul + bias into addmm/addmm_gelu
        passes.emplace_back(std::make_unique<MatmulImplPass>());
        passes.emplace_back(std::make_unique<ActImplPass>());
        passes.emplace_back(std::make_unique<CastImplPass>());
        passes.emplace_back(std::make_unique<TrailingBroadcastAddImplPass>());
        passes.emplace_back(std::make_unique<LstmCellImplPass>());
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
    const double tflops = useful_flops / time_per_iter / 1e12;

    const auto promised_tflops = GetPromisedTFlops(0, PrecisionMode::FP16); // fp16 inputs / fp32 accumulate
    const double mfu = promised_tflops ? (tflops / static_cast<double>(*promised_tflops) * 100.0) : -1.0;

    std::cout << "Streaming LSTM benchmark\n";
    std::cout << "T=" << cfg.T << " B=" << cfg.B << " I=" << cfg.I << " H=" << cfg.H << "\n";
    std::cout << "Time/iter (s): " << time_per_iter << "\n";
    std::cout << "Useful FLOPs: " << useful_flops << "\n";
    std::cout << "Useful TFLOPs/s: " << tflops << "\n";
    std::cout << "Promised TFLOPs/s: " << promised_tflops.value_or(-1.0f) << "\n";
    if (mfu >= 0.0)
    {
        std::cout << "MFU (%): " << mfu << "\n";
    }
    else
    {
        std::cout << "MFU (%): unknown (promised TFLOPs unavailable for device)\n";
    }
    std::cout << "Total FLOPs: " << total_flops << "\n";

    return 0;
}
