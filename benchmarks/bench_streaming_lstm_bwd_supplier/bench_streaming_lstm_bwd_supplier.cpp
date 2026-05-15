#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <allocator.h>
#include <execution_backend.h>
#include <executor.h>
#include <functional.h>
#include <mfu.h>
#include <op_graph.h>
#include <passes.h>
#include <tensorlib.h>

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
        int V{4096};
        int streaming_chunk{8};
        int recompute_interval{1};
        int supplier_stream_id{1};
        int supplier_stream_priority{1000};
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

    ExecutionPlan BuildPlan(const BenchmarkConfig &cfg)
    {
        const uint64_t T = static_cast<uint64_t>(cfg.T);
        const uint64_t B = static_cast<uint64_t>(cfg.B);
        const uint64_t I = static_cast<uint64_t>(cfg.I);
        const uint64_t H = static_cast<uint64_t>(cfg.H);
        const uint64_t V = static_cast<uint64_t>(cfg.V);
        const uint64_t gate_dim = 4 * H;

        auto x_host = RealTensor::Allocate({T, B, I}, DataType::FLOAT16, DEVICE_CPU, true);
        auto h_cache_host = RealTensor::Allocate({T, B, H}, DataType::FLOAT16, DEVICE_CPU, true);
        const uint64_t c_cache_steps =
            (T + static_cast<uint64_t>(cfg.recompute_interval) - 1) / static_cast<uint64_t>(cfg.recompute_interval);
        auto c_cache_host = RealTensor::Allocate({c_cache_steps, B, H}, DataType::FLOAT32, DEVICE_CPU, true);
        auto y_cache_host = RealTensor::Allocate({T, B, H}, DataType::FLOAT16, DEVICE_CPU, true);
        auto targets_host = RealTensor::Allocate({T, B}, DataType::UINT32, DEVICE_CPU, true);

        auto h0_device = RealTensor::Allocate({H}, DataType::FLOAT32, DEVICE_GPU);
        auto c0_device = RealTensor::Allocate({H}, DataType::FLOAT32, DEVICE_GPU);
        auto w_ih = RealTensor::Allocate({I, gate_dim}, DataType::FLOAT32, DEVICE_GPU);
        auto w_hh = RealTensor::Allocate({H, gate_dim}, DataType::FLOAT32, DEVICE_GPU);
        auto b_ih = RealTensor::Allocate({gate_dim}, DataType::FLOAT32, DEVICE_GPU);
        auto b_hh = RealTensor::Allocate({gate_dim}, DataType::FLOAT32, DEVICE_GPU);
        const auto supplier_stream_desc = GpuStreamDescriptor{StreamKind::Compute, cfg.supplier_stream_id};
        auto head_weight =
            RealTensor::AllocateOnStream({H, V}, DataType::FLOAT16, DEVICE_GPU, supplier_stream_desc, false);

        std::memset(y_cache_host->dataptr(), 0, y_cache_host->storage()->sizeBytes());
        std::memset(targets_host->dataptr(), 0, targets_host->storage()->sizeBytes());

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
        inputs.push_back({"targets", TraceTensor::Create(targets_host->shape().dims(), targets_host->dtype(),
                                                         DEVICE_CPU, main_stream_desc, /*pinned=*/true)});

        std::vector<GraphInputDescriptor> params{};
        params.push_back(
            {"w_ih", TraceTensor::Create(w_ih->shape().dims(), w_ih->dtype(), DEVICE_GPU, main_stream_desc)});
        params.push_back(
            {"w_hh", TraceTensor::Create(w_hh->shape().dims(), w_hh->dtype(), DEVICE_GPU, main_stream_desc)});
        params.push_back(
            {"b_ih", TraceTensor::Create(b_ih->shape().dims(), b_ih->dtype(), DEVICE_GPU, main_stream_desc)});
        params.push_back(
            {"b_hh", TraceTensor::Create(b_hh->shape().dims(), b_hh->dtype(), DEVICE_GPU, main_stream_desc)});
        params.push_back({"head_weight", TraceTensor::Create(head_weight->shape().dims(), head_weight->dtype(),
                                                             DEVICE_GPU, supplier_stream_desc)});

        for (auto &entry : inputs)
        {
            entry.tensor.markRetained();
        }
        for (auto &entry : params)
        {
            entry.tensor.markRetained();
        }

        OpGraph graph(inputs, params);

        TraceTensor targets_gpu = graph.createTensor(inputs[6].tensor.shape().dims(), inputs[6].tensor.dtype(),
                                                     DEVICE_GPU, supplier_stream_desc, false);
        DeviceCopy(graph, inputs[6].tensor, targets_gpu, main_stream_desc);

        TraceTensor upstream_scalar =
            graph.createTensor({1}, DataType::FLOAT32, DEVICE_GPU, supplier_stream_desc, false);
        const float loss_scale = 1.0f / static_cast<float>(cfg.T * cfg.B);
        FillConstant(graph, upstream_scalar, loss_scale, supplier_stream_desc);

        TraceTensor grad_head_weight = graph.createTensor(head_weight->shape().dims(), DataType::FLOAT16, DEVICE_GPU,
                                                          supplier_stream_desc, false);
        FillZeros(graph, grad_head_weight, supplier_stream_desc);

        const TraceTensor head_weight_t = params[4].tensor;
        const TraceTensor head_weight_T = head_weight_t.transpose(graph, {1, 0});
        const TraceTensor &y_host_trace = inputs[3].tensor;
        const auto head_weight_dtype = head_weight->dtype();

        TraceTensor y_chunk_gpu = graph.createTensor({static_cast<uint64_t>(cfg.streaming_chunk), B, H},
                                                     DataType::FLOAT16, DEVICE_GPU, supplier_stream_desc, false);
        std::unordered_map<std::string, std::any> accumulate_attr{};
        accumulate_attr.emplace("accumulate_output", true);

        StreamingDySupplier dy_supplier =
            [&](OpGraph &hook_graph, const TraceTensor &dst, const uint64_t chunk_start,
                const uint64_t chunk_steps) -> std::optional<GpuStreamDescriptor>
        {
            const uint64_t total_steps = y_host_trace.shape()[0];
            if (chunk_start + chunk_steps > total_steps)
            {
                throw std::out_of_range("dy supplier slice out of range");
            }
            if (chunk_steps > static_cast<uint64_t>(cfg.streaming_chunk))
            {
                throw std::out_of_range("dy supplier chunk exceeds streaming_chunk size");
            }

            TraceTensor y_host_slice = y_host_trace.slice(hook_graph, 0, chunk_start, chunk_steps);
            TraceTensor y_chunk_slice = y_chunk_gpu.slice(hook_graph, 0, 0, chunk_steps);
            DeviceCopy(hook_graph, y_host_slice, y_chunk_slice, supplier_stream_desc);

            TraceTensor targets_chunk_slice = targets_gpu.slice(hook_graph, 0, chunk_start, chunk_steps);

            const uint64_t chunk_steps_u64 = chunk_steps;
            TraceTensor flat_y = y_chunk_slice.view(hook_graph, {chunk_steps_u64 * B, H});
            TraceTensor flat_logits = hook_graph.createTensor({flat_y.shape()[0], V}, head_weight_dtype, DEVICE_GPU,
                                                              supplier_stream_desc, false);
            hook_graph.recordOperation(OperationEntry{.type = OpType::MATMUL,
                                                      .inputs = {flat_y, head_weight_t},
                                                      .outputs = {flat_logits},
                                                      .is_useful = false,
                                                      .attributes = {},
                                                      .gpu_stream_desc = supplier_stream_desc});

            TraceTensor vocab_logits = flat_logits.view(hook_graph, {chunk_steps_u64, B, V});

            TraceTensor grad_logits =
                CrossEntropyOnTargetsBackward(hook_graph, vocab_logits, targets_chunk_slice, upstream_scalar,
                                              Reduction::ADD, supplier_stream_desc, /*reduce_over_rows=*/true);

            TraceTensor grad_logits_flat = grad_logits.view(hook_graph, {chunk_steps_u64 * B, V});

            TraceTensor dy_flat = dst.view(hook_graph, {chunk_steps_u64 * B, H});
            hook_graph.recordOperation(OperationEntry{.type = OpType::MATMUL,
                                                      .inputs = {grad_logits_flat, head_weight_T},
                                                      .outputs = {dy_flat},
                                                      .attributes = {},
                                                      .gpu_stream_desc = supplier_stream_desc});

            TraceTensor flat_y_T = flat_y.transpose(hook_graph, {1, 0});
            hook_graph.recordOperation(OperationEntry{.type = OpType::MATMUL,
                                                      .inputs = {flat_y_T, grad_logits_flat},
                                                      .outputs = {grad_head_weight},
                                                      .attributes = accumulate_attr,
                                                      .gpu_stream_desc = supplier_stream_desc});

            hook_graph.deleteTensor(flat_logits);
            hook_graph.deleteTensor(grad_logits);
            return supplier_stream_desc;
        };

        auto lstm_result = StreamingLstmBwdStreamedDy(
            graph, inputs[0].tensor, inputs[1].tensor, inputs[2].tensor, params[0].tensor, params[1].tensor,
            params[2].tensor, params[3].tensor, inputs[3].tensor, inputs[4].tensor, inputs[5].tensor, dy_supplier,
            supplier_stream_desc, DataType::FLOAT16, std::nullopt, std::nullopt, cfg.recompute_interval,
            cfg.streaming_chunk, std::nullopt, main_stream_desc);
        (void)lstm_result;

        graph.finalize();

        std::vector<GraphExecutionInputDescriptor> exec_inputs{};
        exec_inputs.push_back({"x", x_host});
        exec_inputs.push_back({"h0", h0_device});
        exec_inputs.push_back({"c0", c0_device});
        exec_inputs.push_back({"y_cache", y_cache_host});
        exec_inputs.push_back({"h_cache", h_cache_host});
        exec_inputs.push_back({"c_cache", c_cache_host});
        exec_inputs.push_back({"targets", targets_host});

        std::vector<GraphExecutionInputDescriptor> exec_params{};
        exec_params.push_back({"w_ih", w_ih});
        exec_params.push_back({"w_hh", w_hh});
        exec_params.push_back({"b_ih", b_ih});
        exec_params.push_back({"b_hh", b_hh});
        exec_params.push_back({"head_weight", head_weight});

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
        passes.emplace_back(std::make_unique<CrossEntropyOnTargetsImplPass>());
        pi::tensorlib::passes::Transform(plan, passes);

        return plan;
    }
} // namespace

int main()
{
    BenchmarkConfig cfg{};
    ExecutionBackend &backend = ExecutionBackend::getInstance();
    const auto supplier_stream_desc = GpuStreamDescriptor{StreamKind::Compute, cfg.supplier_stream_id};
    if (cfg.supplier_stream_id != 0)
    {
        ExecutionBackend::SetComputeStreamPriority(DEVICE_GPU, supplier_stream_desc, cfg.supplier_stream_priority);
    }
    const auto &alloc_registry = allocator::CachingAllocatorRegistry::instance();

    auto plan = BuildPlan(cfg);
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

    std::cout << "Streaming LSTM backward (streamed dY) benchmark\n";
    std::cout << "T=" << cfg.T << " B=" << cfg.B << " I=" << cfg.I << " H=" << cfg.H << " V=" << cfg.V << "\n";
    std::cout << "Time/iter (s): " << time_per_iter << "\n";
    std::cout << "Useful FLOPs: " << useful_flops << "\n";
    std::cout << "Useful TFLOPs/s: " << useful_tflops << "\n";
    std::cout << "Promised TFLOPs/s: " << promised_tflops.value_or(-1.0f) << "\n";
    std::cout << "MFU (%): " << mfu << "\n";
    std::cout << "HFU (%): " << hfu << "\n";
    std::cout << "Total FLOPs: " << total_flops << "\n";

    return 0;
}
