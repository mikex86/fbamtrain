#include <allocator.h>
#include <execution_backend.h>
#include <executor.h>
#include <functional.h>
#include <mfu.h>
#include <passes.h>
#include <tensorlib.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <unordered_map>
#include <utility>

namespace
{
    constexpr uint32_t M_DIM = 8192;
    constexpr uint32_t N_DIM = 8192;
    constexpr uint32_t K_DIM = 8192;
    constexpr uint32_t M_DIM_UNALIGNED = 8191;
    constexpr std::size_t WARMUP_ITERS = 8;
    constexpr std::size_t MEASURE_ITERS = 32;
    const auto main_stream_desc = pi::tensorlib::GpuStreamDescriptors::Main;

    struct MatmulRunResult
    {
        std::string label;
        double avg_ms{};
        double tflops{};
        double mfu{};
    };

    void ApplyDefaultPasses(pi::tensorlib::ExecutionPlan &execution_plan)
    {
        std::vector<std::unique_ptr<pi::tensorlib::passes::CompilerPass>> passes{};
        passes.emplace_back(std::make_unique<MatmulFusePass>());
        passes.emplace_back(std::make_unique<MatmulImplPass>());
        passes.emplace_back(std::make_unique<ActImplPass>());
        passes.emplace_back(std::make_unique<FillZerosImplPass>());
        passes.emplace_back(std::make_unique<FillUniformImplPass>());
        pi::tensorlib::passes::Transform(execution_plan, passes);
    }

    struct MatmulInputs
    {
        std::shared_ptr<pi::tensorlib::RealTensor> a_nn;       // (M, K)
        std::shared_ptr<pi::tensorlib::RealTensor> b_nn;       // (K, N)
        std::shared_ptr<pi::tensorlib::RealTensor> a_t;        // (K, M) for transposed-a cases
        std::shared_ptr<pi::tensorlib::RealTensor> b_t;        // (N, K) for transposed-b cases
        std::shared_ptr<pi::tensorlib::RealTensor> bias;       // (N)
        std::shared_ptr<pi::tensorlib::RealTensor> preact_out; // (M, N) pre-activation for GELU backward
    };

    struct GeluBwdInputs
    {
        std::shared_ptr<pi::tensorlib::RealTensor> a_nn;       // (M, K)
        std::shared_ptr<pi::tensorlib::RealTensor> b_t;        // (N, K) for transpose-b
        std::shared_ptr<pi::tensorlib::RealTensor> preact_out; // (M, N)
    };

    MatmulInputs InitializeInputs(const pi::tensorlib::Device &device, const pi::tensorlib::DataType dtype)
    {
        pi::tensorlib::OpGraph init_graph{{}, {}};
        pi::tensorlib::TraceTensor a_nn = init_graph.createTensor({M_DIM, K_DIM}, dtype, device, main_stream_desc, false);
        pi::tensorlib::TraceTensor b_nn = init_graph.createTensor({K_DIM, N_DIM}, dtype, device, main_stream_desc, false);
        // Bases sized for transposed views: a_t (K, M) -> transpose -> (M, K); b_t (N, K) -> transpose -> (K, N).
        pi::tensorlib::TraceTensor a_t = init_graph.createTensor({K_DIM, M_DIM}, dtype, device, main_stream_desc, false);
        pi::tensorlib::TraceTensor b_t = init_graph.createTensor({N_DIM, K_DIM}, dtype, device, main_stream_desc, false);
        pi::tensorlib::TraceTensor bias = init_graph.createTensor({N_DIM}, dtype, device, main_stream_desc, false);
        pi::tensorlib::TraceTensor preact_out =
            init_graph.createTensor({M_DIM, N_DIM}, dtype, device, main_stream_desc, false);

        uint32_t seed = 42;
        pi::tensorlib::FillUniform(init_graph, a_nn, -0.5f, 0.5f, seed++, main_stream_desc);
        pi::tensorlib::FillUniform(init_graph, b_nn, -0.5f, 0.5f, seed++, main_stream_desc);
        pi::tensorlib::FillUniform(init_graph, a_t, -0.5f, 0.5f, seed++, main_stream_desc);
        pi::tensorlib::FillUniform(init_graph, b_t, -0.5f, 0.5f, seed++, main_stream_desc);
        pi::tensorlib::FillUniform(init_graph, bias, -0.5f, 0.5f, seed++, main_stream_desc);
        pi::tensorlib::FillUniform(init_graph, preact_out, -0.5f, 0.5f, seed++, main_stream_desc);
        init_graph.finalize();

        pi::tensorlib::ExecutionPlan init_plan = pi::tensorlib::ExecutionPlan::FromGraph(init_graph, {}, {});
        ApplyDefaultPasses(init_plan);

        pi::tensorlib::ExecutionBackend &backend = pi::tensorlib::ExecutionBackend::getInstance();
        auto &allocator_registry = pi::tensorlib::allocator::CachingAllocatorRegistry::instance();
        pi::tensorlib::Executor init_executor{init_plan, backend, 0};
        init_executor.execute(allocator_registry);

        const auto a_nn_real = init_executor.getOutput(a_nn);
        const auto b_nn_real = init_executor.getOutput(b_nn);
        const auto a_t_real = init_executor.getOutput(a_t);
        const auto b_t_real = init_executor.getOutput(b_t);
        const auto bias_real = init_executor.getOutput(bias);
        const auto preact_out_real = init_executor.getOutput(preact_out);
        if (!a_nn_real || !b_nn_real || !a_t_real || !b_t_real || !bias_real || !preact_out_real)
        {
            throw std::runtime_error("Failed to materialize matmul inputs after initialization");
        }
        return MatmulInputs{*a_nn_real, *b_nn_real, *a_t_real, *b_t_real, *bias_real, *preact_out_real};
    }

    GeluBwdInputs InitializeGeluBwdInputs(const pi::tensorlib::Device &device, const pi::tensorlib::DataType dtype,
                                          const uint32_t m_dim, const uint32_t n_dim, const uint32_t k_dim)
    {
        pi::tensorlib::OpGraph init_graph{{}, {}};
        pi::tensorlib::TraceTensor a_nn = init_graph.createTensor({m_dim, k_dim}, dtype, device, main_stream_desc, false);
        pi::tensorlib::TraceTensor b_t = init_graph.createTensor({n_dim, k_dim}, dtype, device, main_stream_desc, false);
        pi::tensorlib::TraceTensor preact_out =
            init_graph.createTensor({m_dim, n_dim}, dtype, device, main_stream_desc, false);

        uint32_t seed = 4242;
        pi::tensorlib::FillUniform(init_graph, a_nn, -0.5f, 0.5f, seed++, main_stream_desc);
        pi::tensorlib::FillUniform(init_graph, b_t, -0.5f, 0.5f, seed++, main_stream_desc);
        pi::tensorlib::FillUniform(init_graph, preact_out, -0.5f, 0.5f, seed++, main_stream_desc);
        init_graph.finalize();

        pi::tensorlib::ExecutionPlan init_plan = pi::tensorlib::ExecutionPlan::FromGraph(init_graph, {}, {});
        ApplyDefaultPasses(init_plan);

        pi::tensorlib::ExecutionBackend &backend = pi::tensorlib::ExecutionBackend::getInstance();
        auto &allocator_registry = pi::tensorlib::allocator::CachingAllocatorRegistry::instance();
        pi::tensorlib::Executor init_executor{init_plan, backend, 0};
        init_executor.execute(allocator_registry);

        const auto a_nn_real = init_executor.getOutput(a_nn);
        const auto b_t_real = init_executor.getOutput(b_t);
        const auto preact_out_real = init_executor.getOutput(preact_out);
        if (!a_nn_real || !b_t_real || !preact_out_real)
        {
            throw std::runtime_error("Failed to materialize GELU bwd inputs after initialization");
        }
        return GeluBwdInputs{*a_nn_real, *b_t_real, *preact_out_real};
    }

    MatmulRunResult RunMatmul(const std::string &label, const pi::tensorlib::Device &device,
                              const pi::tensorlib::DataType dtype,
                              const std::shared_ptr<pi::tensorlib::RealTensor> &a_real,
                              const std::shared_ptr<pi::tensorlib::RealTensor> &b_real, const bool transpose_a,
                              const bool transpose_b, const bool use_fp16_accum,
                              const std::optional<const char *> prefer_env_override)
    {
        // Toggle preferred backend for this run.
        const char *env_name = "FBAMTRAIN_PREFER_GEMM_BACKEND";
        const char *old_env = std::getenv(env_name);
        if (prefer_env_override.has_value())
        {
            setenv(env_name, *prefer_env_override, 1);
        }
        else
        {
            unsetenv(env_name);
        }

        // Effective dimensions after optional transposes.
        const auto a_shape = a_real->shape();
        const auto b_shape = b_real->shape();
        const uint64_t m_dim = transpose_a ? a_shape[1] : a_shape[0];
        const uint64_t k_dim_a = transpose_a ? a_shape[0] : a_shape[1];
        const uint64_t k_dim_b = transpose_b ? b_shape[1] : b_shape[0];
        const uint64_t n_dim = transpose_b ? b_shape[0] : b_shape[1];
        if (k_dim_a != k_dim_b)
        {
            throw std::runtime_error("Incompatible shapes for matmul run");
        }

        pi::tensorlib::TraceTensor a_trace = pi::tensorlib::TraceTensor::Create(
            {static_cast<uint64_t>(a_shape[0]), static_cast<uint64_t>(a_shape[1])}, dtype, device, main_stream_desc);
        pi::tensorlib::TraceTensor b_trace = pi::tensorlib::TraceTensor::Create(
            {static_cast<uint64_t>(b_shape[0]), static_cast<uint64_t>(b_shape[1])}, dtype, device, main_stream_desc);
        a_trace.markRetained();
        b_trace.markRetained();
        pi::tensorlib::OpGraph graph{{{.name = "a", .tensor = a_trace}, {.name = "b", .tensor = b_trace}}, {}};

        pi::tensorlib::TraceTensor a_view = graph.getInputDescriptors()[0].tensor;
        pi::tensorlib::TraceTensor b_view = graph.getInputDescriptors()[1].tensor;
        if (transpose_a)
        {
            a_view = a_view.transpose(graph, {1, 0});
        }
        if (transpose_b)
        {
            b_view = b_view.transpose(graph, {1, 0});
        }

        pi::tensorlib::TraceTensor out = graph.createTensor({static_cast<uint64_t>(m_dim), static_cast<uint64_t>(n_dim)},
                                                            dtype, device, main_stream_desc, false);

        std::unordered_map<std::string, std::any> matmul_attrs{};
        if (use_fp16_accum && dtype == pi::tensorlib::DataType::FLOAT16)
        {
            matmul_attrs.emplace("use_fp16_matmul_acc", true);
        }

        graph.recordOperation(pi::tensorlib::OperationEntry{.type = pi::tensorlib::OpType::MATMUL,
                                                            .inputs = {a_view, b_view},
                                                            .outputs = {out},
                                                            .attributes = std::move(matmul_attrs),
                                                            .gpu_stream_desc = main_stream_desc});
        graph.finalize();

        pi::tensorlib::ExecutionPlan plan = pi::tensorlib::ExecutionPlan::FromGraph(graph,
                                                                                    {
                                                                                        {.name = "a", .tensor = a_real},
                                                                                        {.name = "b", .tensor = b_real},
                                                                                    },
                                                                                    {});
        ApplyDefaultPasses(plan);

        pi::tensorlib::ExecutionBackend &backend = pi::tensorlib::ExecutionBackend::getInstance();
        auto &allocator_registry = pi::tensorlib::allocator::CachingAllocatorRegistry::instance();
        pi::tensorlib::Executor executor{plan, backend, 0};

        // Warmup.
        for (std::size_t i = 0; i < WARMUP_ITERS; ++i)
        {
            executor.execute(allocator_registry);
        }

        // Timed runs (event + wall).
        pi::tensorlib::GpuEvent start_event = pi::tensorlib::ExecutionBackend::CreateEvent(device, true);
        pi::tensorlib::GpuEvent end_event = pi::tensorlib::ExecutionBackend::CreateEvent(device, true);

        const auto stream_bundle = pi::tensorlib::ExecutionBackend::GetStreamBundle(device);
        auto main_stream = stream_bundle->main_stream;

        auto start_time = std::chrono::high_resolution_clock::now();
        start_event.record(main_stream);
        for (std::size_t i = 0; i < MEASURE_ITERS; ++i)
        {
            executor.execute(allocator_registry, /*awaitExecution=*/false);
        }
        end_event.record(main_stream);
        end_event.synchronize();
        executor.await();
        auto end_time = std::chrono::high_resolution_clock::now();

        const double event_ms = end_event.elapsedMsSince(start_event);
        const double avg_ms_device = event_ms / static_cast<double>(MEASURE_ITERS);
        const double total_ms_host =
            std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1e3;
        (void)total_ms_host; // host timing reported via stdout below.

        // Materialize output once to guarantee completion.
        const auto out_real = executor.getOutput(out);
        if (!out_real)
        {
            throw std::runtime_error("Failed to fetch matmul output");
        }

        const double flops = 2.0 * static_cast<double>(M_DIM) * static_cast<double>(N_DIM) * static_cast<double>(K_DIM);
        const double tflops = flops / (avg_ms_device / 1e3) / 1e12;

        const pi::tensorlib::PrecisionMode precision =
            (dtype == pi::tensorlib::DataType::BFLOAT16)
                ? pi::tensorlib::PrecisionMode::BF16
                : (use_fp16_accum ? pi::tensorlib::PrecisionMode::FP16_ACC16 : pi::tensorlib::PrecisionMode::FP16);
        const auto promised = pi::tensorlib::GetPromisedTFlops(device.ordinal, precision);
        const double mfu = promised ? (tflops / static_cast<double>(*promised) * 100.0) : -1.0;

        // Restore env.
        if (old_env)
        {
            setenv(env_name, old_env, 1);
        }
        else
        {
            unsetenv(env_name);
        }

        std::cout << "[matmul_mfu] label=" << label << " avg_ms_device=" << avg_ms_device << " tflops=" << tflops
                  << " mfu=" << mfu;
        if (promised)
        {
            std::cout << " promised_tflops=" << *promised;
        }
        std::cout << std::endl;

        return MatmulRunResult{label, avg_ms_device, tflops, mfu};
    }

    MatmulRunResult RunMatmulGelu(const std::string &label, const pi::tensorlib::Device &device,
                                  const pi::tensorlib::DataType dtype, const MatmulInputs &inputs,
                                  const bool transpose_a, const bool transpose_b, const bool use_fp16_accum,
                                  const std::optional<const char *> prefer_env_override)
    {
        const char *env_name = "FBAMTRAIN_PREFER_GEMM_BACKEND";
        const char *old_env = std::getenv(env_name);
        if (prefer_env_override.has_value())
        {
            setenv(env_name, *prefer_env_override, 1);
        }
        else
        {
            unsetenv(env_name);
        }

        const auto &a_real = transpose_a ? inputs.a_t : inputs.a_nn;
        const auto &b_real = transpose_b ? inputs.b_t : inputs.b_nn;

        const auto a_shape = a_real->shape();
        const auto b_shape = b_real->shape();
        const uint64_t m_dim = transpose_a ? a_shape[1] : a_shape[0];
        const uint64_t k_dim_a = transpose_a ? a_shape[0] : a_shape[1];
        const uint64_t k_dim_b = transpose_b ? b_shape[1] : b_shape[0];
        const uint64_t n_dim = transpose_b ? b_shape[0] : b_shape[1];
        if (k_dim_a != k_dim_b)
        {
            throw std::runtime_error("Incompatible shapes for matmul gelu run");
        }

        pi::tensorlib::TraceTensor a_trace = pi::tensorlib::TraceTensor::Create(
            {static_cast<uint64_t>(a_shape[0]), static_cast<uint64_t>(a_shape[1])}, dtype, device, main_stream_desc);
        pi::tensorlib::TraceTensor b_trace = pi::tensorlib::TraceTensor::Create(
            {static_cast<uint64_t>(b_shape[0]), static_cast<uint64_t>(b_shape[1])}, dtype, device, main_stream_desc);
        a_trace.markRetained();
        b_trace.markRetained();
        pi::tensorlib::OpGraph graph{{{.name = "a", .tensor = a_trace}, {.name = "b", .tensor = b_trace}}, {}};

        pi::tensorlib::TraceTensor a_view = graph.getInputDescriptors()[0].tensor;
        pi::tensorlib::TraceTensor b_view = graph.getInputDescriptors()[1].tensor;
        if (transpose_a)
        {
            a_view = a_view.transpose(graph, {1, 0});
        }
        if (transpose_b)
        {
            b_view = b_view.transpose(graph, {1, 0});
        }

        pi::tensorlib::TraceTensor matmul_out =
            graph.createTensor({static_cast<uint64_t>(m_dim), static_cast<uint64_t>(n_dim)}, dtype, device,
                               main_stream_desc, false);

        std::unordered_map<std::string, std::any> matmul_attrs{};
        if (use_fp16_accum && dtype == pi::tensorlib::DataType::FLOAT16)
        {
            matmul_attrs.emplace("use_fp16_matmul_acc", true);
        }

        graph.recordOperation(pi::tensorlib::OperationEntry{.type = pi::tensorlib::OpType::MATMUL,
                                                            .inputs = {a_view, b_view},
                                                            .outputs = {matmul_out},
                                                            .attributes = std::move(matmul_attrs),
                                                            .gpu_stream_desc = main_stream_desc});

        const pi::tensorlib::TraceTensor out = pi::tensorlib::Gelu(graph, matmul_out, main_stream_desc);
        graph.finalize();

        pi::tensorlib::ExecutionPlan plan = pi::tensorlib::ExecutionPlan::FromGraph(graph,
                                                                                    {
                                                                                        {.name = "a", .tensor = a_real},
                                                                                        {.name = "b", .tensor = b_real},
                                                                                    },
                                                                                    {});
        ApplyDefaultPasses(plan);

        pi::tensorlib::ExecutionBackend &backend = pi::tensorlib::ExecutionBackend::getInstance();
        auto &allocator_registry = pi::tensorlib::allocator::CachingAllocatorRegistry::instance();
        pi::tensorlib::Executor executor{plan, backend, 0};

        for (std::size_t i = 0; i < WARMUP_ITERS; ++i)
        {
            executor.execute(allocator_registry);
        }

        pi::tensorlib::GpuEvent start_event = pi::tensorlib::ExecutionBackend::CreateEvent(device, true);
        pi::tensorlib::GpuEvent end_event = pi::tensorlib::ExecutionBackend::CreateEvent(device, true);

        const auto stream_bundle = pi::tensorlib::ExecutionBackend::GetStreamBundle(device);
        auto main_stream = stream_bundle->main_stream;

        start_event.record(main_stream);
        for (std::size_t i = 0; i < MEASURE_ITERS; ++i)
        {
            executor.execute(allocator_registry, /*awaitExecution=*/false);
        }
        end_event.record(main_stream);
        end_event.synchronize();
        executor.await();

        const double event_ms = end_event.elapsedMsSince(start_event);
        const double avg_ms_device = event_ms / static_cast<double>(MEASURE_ITERS);

        const auto out_real = executor.getOutput(out);
        if (!out_real)
        {
            throw std::runtime_error("Failed to fetch matmul gelu output");
        }

        const double flops = 2.0 * static_cast<double>(M_DIM) * static_cast<double>(N_DIM) * static_cast<double>(K_DIM);
        const double tflops = flops / (avg_ms_device / 1e3) / 1e12;

        const pi::tensorlib::PrecisionMode precision =
            (dtype == pi::tensorlib::DataType::BFLOAT16)
                ? pi::tensorlib::PrecisionMode::BF16
                : (use_fp16_accum ? pi::tensorlib::PrecisionMode::FP16_ACC16 : pi::tensorlib::PrecisionMode::FP16);
        const auto promised = pi::tensorlib::GetPromisedTFlops(device.ordinal, precision);
        const double mfu = promised ? (tflops / static_cast<double>(*promised) * 100.0) : -1.0;

        if (old_env)
        {
            setenv(env_name, old_env, 1);
        }
        else
        {
            unsetenv(env_name);
        }

        std::cout << "[matmul_mfu] label=" << label << " avg_ms_device=" << avg_ms_device << " tflops=" << tflops
                  << " mfu=" << mfu;
        if (promised)
        {
            std::cout << " promised_tflops=" << *promised;
        }
        std::cout << std::endl;

        return MatmulRunResult{label, avg_ms_device, tflops, mfu};
    }

    MatmulRunResult RunAddmmGeluPreact(const std::string &label, const pi::tensorlib::Device &device,
                                       const pi::tensorlib::DataType dtype, const MatmulInputs &inputs,
                                       const bool transpose_a, const bool transpose_b, const bool use_fp16_accum,
                                       const std::optional<const char *> prefer_env_override)
    {
        const char *env_name = "FBAMTRAIN_PREFER_GEMM_BACKEND";
        const char *old_env = std::getenv(env_name);
        if (prefer_env_override.has_value())
        {
            setenv(env_name, *prefer_env_override, 1);
        }
        else
        {
            unsetenv(env_name);
        }

        const auto &a_real = transpose_a ? inputs.a_t : inputs.a_nn;
        const auto &b_real = transpose_b ? inputs.b_t : inputs.b_nn;
        const auto &bias_real = inputs.bias;

        const auto a_shape = a_real->shape();
        const auto b_shape = b_real->shape();
        const uint64_t m_dim = transpose_a ? a_shape[1] : a_shape[0];
        const uint64_t k_dim_a = transpose_a ? a_shape[0] : a_shape[1];
        const uint64_t k_dim_b = transpose_b ? b_shape[1] : b_shape[0];
        const uint64_t n_dim = transpose_b ? b_shape[0] : b_shape[1];
        if (k_dim_a != k_dim_b)
        {
            throw std::runtime_error("Incompatible shapes for addmm gelu preact run");
        }

        pi::tensorlib::TraceTensor a_trace = pi::tensorlib::TraceTensor::Create(
            {static_cast<uint64_t>(a_shape[0]), static_cast<uint64_t>(a_shape[1])}, dtype, device, main_stream_desc);
        pi::tensorlib::TraceTensor b_trace = pi::tensorlib::TraceTensor::Create(
            {static_cast<uint64_t>(b_shape[0]), static_cast<uint64_t>(b_shape[1])}, dtype, device, main_stream_desc);
        pi::tensorlib::TraceTensor bias_trace =
            pi::tensorlib::TraceTensor::Create({static_cast<uint64_t>(bias_real->shape()[0])}, dtype, device,
                                               main_stream_desc);
        a_trace.markRetained();
        b_trace.markRetained();
        bias_trace.markRetained();
        pi::tensorlib::OpGraph graph{
            {
                {.name = "a", .tensor = a_trace},
                {.name = "b", .tensor = b_trace},
                {.name = "bias", .tensor = bias_trace},
            },
            {}};

        pi::tensorlib::TraceTensor a_view = graph.getInputDescriptors()[0].tensor;
        pi::tensorlib::TraceTensor b_view = graph.getInputDescriptors()[1].tensor;
        if (transpose_a)
        {
            a_view = a_view.transpose(graph, {1, 0});
        }
        if (transpose_b)
        {
            b_view = b_view.transpose(graph, {1, 0});
        }

        pi::tensorlib::TraceTensor matmul_out =
            graph.createTensor({static_cast<uint64_t>(m_dim), static_cast<uint64_t>(n_dim)}, dtype, device,
                               main_stream_desc, false);

        std::unordered_map<std::string, std::any> matmul_attrs{};
        if (use_fp16_accum && dtype == pi::tensorlib::DataType::FLOAT16)
        {
            matmul_attrs.emplace("use_fp16_matmul_acc", true);
        }
        matmul_attrs.emplace("write_out_preact", true);

        graph.recordOperation(pi::tensorlib::OperationEntry{.type = pi::tensorlib::OpType::MATMUL,
                                                            .inputs = {a_view, b_view},
                                                            .outputs = {matmul_out},
                                                            .attributes = std::move(matmul_attrs),
                                                            .gpu_stream_desc = main_stream_desc});

        const auto bias_view = graph.getInputDescriptors()[2].tensor;
        graph.recordOperation(pi::tensorlib::OperationEntry{.type = pi::tensorlib::OpType::PLUS,
                                                            .inputs = {matmul_out, bias_view},
                                                            .outputs = {matmul_out},
                                                            .gpu_stream_desc = main_stream_desc});

        const pi::tensorlib::TraceTensor out = pi::tensorlib::Gelu(graph, matmul_out, main_stream_desc);
        graph.finalize();

        pi::tensorlib::ExecutionPlan plan =
            pi::tensorlib::ExecutionPlan::FromGraph(graph,
                                                    {
                                                        {.name = "a", .tensor = a_real},
                                                        {.name = "b", .tensor = b_real},
                                                        {.name = "bias", .tensor = bias_real},
                                                    },
                                                    {});
        ApplyDefaultPasses(plan);

        pi::tensorlib::ExecutionBackend &backend = pi::tensorlib::ExecutionBackend::getInstance();
        auto &allocator_registry = pi::tensorlib::allocator::CachingAllocatorRegistry::instance();
        pi::tensorlib::Executor executor{plan, backend, 0};

        for (std::size_t i = 0; i < WARMUP_ITERS; ++i)
        {
            executor.execute(allocator_registry);
        }

        pi::tensorlib::GpuEvent start_event = pi::tensorlib::ExecutionBackend::CreateEvent(device, true);
        pi::tensorlib::GpuEvent end_event = pi::tensorlib::ExecutionBackend::CreateEvent(device, true);

        const auto stream_bundle = pi::tensorlib::ExecutionBackend::GetStreamBundle(device);
        auto main_stream = stream_bundle->main_stream;

        start_event.record(main_stream);
        for (std::size_t i = 0; i < MEASURE_ITERS; ++i)
        {
            executor.execute(allocator_registry, /*awaitExecution=*/false);
        }
        end_event.record(main_stream);
        end_event.synchronize();
        executor.await();

        const double event_ms = end_event.elapsedMsSince(start_event);
        const double avg_ms_device = event_ms / static_cast<double>(MEASURE_ITERS);

        const auto out_real = executor.getOutput(out);
        if (!out_real)
        {
            throw std::runtime_error("Failed to fetch addmm gelu preact output");
        }

        const double flops = 2.0 * static_cast<double>(M_DIM) * static_cast<double>(N_DIM) * static_cast<double>(K_DIM);
        const double tflops = flops / (avg_ms_device / 1e3) / 1e12;

        const pi::tensorlib::PrecisionMode precision =
            (dtype == pi::tensorlib::DataType::BFLOAT16)
                ? pi::tensorlib::PrecisionMode::BF16
                : (use_fp16_accum ? pi::tensorlib::PrecisionMode::FP16_ACC16 : pi::tensorlib::PrecisionMode::FP16);
        const auto promised = pi::tensorlib::GetPromisedTFlops(device.ordinal, precision);
        const double mfu = promised ? (tflops / static_cast<double>(*promised) * 100.0) : -1.0;

        if (old_env)
        {
            setenv(env_name, old_env, 1);
        }
        else
        {
            unsetenv(env_name);
        }

        std::cout << "[matmul_mfu] label=" << label << " avg_ms_device=" << avg_ms_device << " tflops=" << tflops
                  << " mfu=" << mfu;
        if (promised)
        {
            std::cout << " promised_tflops=" << *promised;
        }
        std::cout << std::endl;

        return MatmulRunResult{label, avg_ms_device, tflops, mfu};
    }

    MatmulRunResult RunMatmulGeluBwd(const std::string &label, const pi::tensorlib::Device &device,
                                     const pi::tensorlib::DataType dtype,
                                     const std::shared_ptr<pi::tensorlib::RealTensor> &a_real,
                                     const std::shared_ptr<pi::tensorlib::RealTensor> &b_real,
                                     const std::shared_ptr<pi::tensorlib::RealTensor> &pre_act_real,
                                     const bool transpose_a, const bool transpose_b, const bool use_fp16_accum,
                                     const std::optional<const char *> prefer_env_override)
    {
        const char *env_name = "FBAMTRAIN_PREFER_GEMM_BACKEND";
        const char *old_env = std::getenv(env_name);
        if (prefer_env_override.has_value())
        {
            setenv(env_name, *prefer_env_override, 1);
        }
        else
        {
            unsetenv(env_name);
        }

        const auto a_shape = a_real->shape();
        const auto b_shape = b_real->shape();
        const auto pre_shape = pre_act_real->shape();
        const uint64_t m_dim = transpose_a ? a_shape[1] : a_shape[0];
        const uint64_t k_dim_a = transpose_a ? a_shape[0] : a_shape[1];
        const uint64_t k_dim_b = transpose_b ? b_shape[1] : b_shape[0];
        const uint64_t n_dim = transpose_b ? b_shape[0] : b_shape[1];
        if (k_dim_a != k_dim_b)
        {
            throw std::runtime_error("Incompatible shapes for matmul gelu bwd run");
        }
        if (pre_shape[0] != m_dim || pre_shape[1] != n_dim)
        {
            throw std::runtime_error("Pre-activation shape does not match matmul output");
        }

        pi::tensorlib::TraceTensor a_trace = pi::tensorlib::TraceTensor::Create(
            {static_cast<uint64_t>(a_shape[0]), static_cast<uint64_t>(a_shape[1])}, dtype, device, main_stream_desc);
        pi::tensorlib::TraceTensor b_trace = pi::tensorlib::TraceTensor::Create(
            {static_cast<uint64_t>(b_shape[0]), static_cast<uint64_t>(b_shape[1])}, dtype, device, main_stream_desc);
        pi::tensorlib::TraceTensor pre_trace = pi::tensorlib::TraceTensor::Create(
            {static_cast<uint64_t>(pre_shape[0]), static_cast<uint64_t>(pre_shape[1])}, dtype, device,
            main_stream_desc);
        a_trace.markRetained();
        b_trace.markRetained();
        pre_trace.markRetained();
        pi::tensorlib::OpGraph graph{
            {
                {.name = "a", .tensor = a_trace},
                {.name = "b", .tensor = b_trace},
                {.name = "pre", .tensor = pre_trace},
            },
            {}};

        pi::tensorlib::TraceTensor a_view = graph.getInputDescriptors()[0].tensor;
        pi::tensorlib::TraceTensor b_view = graph.getInputDescriptors()[1].tensor;
        pi::tensorlib::TraceTensor pre_view = graph.getInputDescriptors()[2].tensor;

        if (transpose_a)
        {
            a_view = a_view.transpose(graph, {1, 0});
        }
        if (transpose_b)
        {
            b_view = b_view.transpose(graph, {1, 0});
        }

        pi::tensorlib::TraceTensor matmul_out =
            graph.createTensor({static_cast<uint64_t>(m_dim), static_cast<uint64_t>(n_dim)}, dtype, device,
                               main_stream_desc, false);

        std::unordered_map<std::string, std::any> matmul_attrs{};
        if (use_fp16_accum && dtype == pi::tensorlib::DataType::FLOAT16)
        {
            matmul_attrs.emplace("use_fp16_matmul_acc", true);
        }

        graph.recordOperation(pi::tensorlib::OperationEntry{.type = pi::tensorlib::OpType::MATMUL,
                                                            .inputs = {a_view, b_view},
                                                            .outputs = {matmul_out},
                                                            .attributes = std::move(matmul_attrs),
                                                            .gpu_stream_desc = main_stream_desc});

        const pi::tensorlib::TraceTensor out =
            pi::tensorlib::GeluBackward(graph, pre_view, matmul_out, main_stream_desc);
        graph.finalize();

        pi::tensorlib::ExecutionPlan plan =
            pi::tensorlib::ExecutionPlan::FromGraph(graph,
                                                    {
                                                        {.name = "a", .tensor = a_real},
                                                        {.name = "b", .tensor = b_real},
                                                        {.name = "pre", .tensor = pre_act_real},
                                                    },
                                                    {});
        ApplyDefaultPasses(plan);

        pi::tensorlib::ExecutionBackend &backend = pi::tensorlib::ExecutionBackend::getInstance();
        auto &allocator_registry = pi::tensorlib::allocator::CachingAllocatorRegistry::instance();
        pi::tensorlib::Executor executor{plan, backend, 0};

        for (std::size_t i = 0; i < WARMUP_ITERS; ++i)
        {
            executor.execute(allocator_registry);
        }

        pi::tensorlib::GpuEvent start_event = pi::tensorlib::ExecutionBackend::CreateEvent(device, true);
        pi::tensorlib::GpuEvent end_event = pi::tensorlib::ExecutionBackend::CreateEvent(device, true);

        const auto stream_bundle = pi::tensorlib::ExecutionBackend::GetStreamBundle(device);
        auto main_stream = stream_bundle->main_stream;

        start_event.record(main_stream);
        for (std::size_t i = 0; i < MEASURE_ITERS; ++i)
        {
            executor.execute(allocator_registry, /*awaitExecution=*/false);
        }
        end_event.record(main_stream);
        end_event.synchronize();
        executor.await();

        const double event_ms = end_event.elapsedMsSince(start_event);
        const double avg_ms_device = event_ms / static_cast<double>(MEASURE_ITERS);

        const auto out_real = executor.getOutput(out);
        if (!out_real)
        {
            throw std::runtime_error("Failed to fetch matmul gelu bwd output");
        }

        const double flops =
            2.0 * static_cast<double>(m_dim) * static_cast<double>(n_dim) * static_cast<double>(k_dim_a);
        const double tflops = flops / (avg_ms_device / 1e3) / 1e12;

        const pi::tensorlib::PrecisionMode precision =
            (dtype == pi::tensorlib::DataType::BFLOAT16)
                ? pi::tensorlib::PrecisionMode::BF16
                : (use_fp16_accum ? pi::tensorlib::PrecisionMode::FP16_ACC16 : pi::tensorlib::PrecisionMode::FP16);
        const auto promised = pi::tensorlib::GetPromisedTFlops(device.ordinal, precision);
        const double mfu = promised ? (tflops / static_cast<double>(*promised) * 100.0) : -1.0;

        if (old_env)
        {
            setenv(env_name, old_env, 1);
        }
        else
        {
            unsetenv(env_name);
        }

        std::cout << "[matmul_mfu] label=" << label << " avg_ms_device=" << avg_ms_device << " tflops=" << tflops
                  << " mfu=" << mfu;
        if (promised)
        {
            std::cout << " promised_tflops=" << *promised;
        }
        std::cout << std::endl;

        return MatmulRunResult{label, avg_ms_device, tflops, mfu};
    }
} // namespace

int main()
{
    constexpr pi::tensorlib::Device device{
        .device_type = pi::tensorlib::DeviceType::GPU,
        .ordinal = 0,
    };
    constexpr auto dtype = pi::tensorlib::DataType::FLOAT16;

    const auto inputs = InitializeInputs(device, dtype);

    // Cutlass path (default allow).
    const auto cutlass_res = RunMatmul("cutlass_fp16", device, dtype, inputs.a_nn, inputs.b_nn,
                                       /*transpose_a=*/false, /*transpose_b=*/false,
                                       /*use_fp16_accum=*/false, std::nullopt);
    const auto cutlass_ta_res = RunMatmul("cutlass_fp16_ta", device, dtype, inputs.a_t, inputs.b_nn,
                                          /*transpose_a=*/true, /*transpose_b=*/false,
                                          /*use_fp16_accum=*/false, std::nullopt);
    const auto cutlass_tb_res = RunMatmul("cutlass_fp16_tb", device, dtype, inputs.a_nn, inputs.b_t,
                                          /*transpose_a=*/false, /*transpose_b=*/true,
                                          /*use_fp16_accum=*/false, std::nullopt);
    const auto cutlass_tab_res = RunMatmul("cutlass_fp16_tab", device, dtype, inputs.a_t, inputs.b_t,
                                           /*transpose_a=*/true, /*transpose_b=*/true,
                                           /*use_fp16_accum=*/false, std::nullopt);

    // Triton path (prefer triton).
    const auto triton_res = RunMatmul("triton_fp16", device, dtype, inputs.a_nn, inputs.b_nn,
                                      /*transpose_a=*/false, /*transpose_b=*/false,
                                      /*use_fp16_accum=*/false, std::optional{"triton"});

    const auto triton_ta_res = RunMatmul("triton_fp16_ta", device, dtype, inputs.a_t, inputs.b_nn,
                                         /*transpose_a=*/true, /*transpose_b=*/false,
                                         /*use_fp16_accum=*/false, std::optional{"triton"});

    const auto triton_tb_res = RunMatmul("triton_fp16_tb", device, dtype, inputs.a_nn, inputs.b_t,
                                         /*transpose_a=*/false, /*transpose_b=*/true,
                                         /*use_fp16_accum=*/false, std::optional{"triton"});

    const auto triton_tab_res = RunMatmul("triton_fp16_tab", device, dtype, inputs.a_t, inputs.b_t,
                                          /*transpose_a=*/true, /*transpose_b=*/true,
                                          /*use_fp16_accum=*/false, std::optional{"triton"});

    // Fused GELU forward matmul (cutlass path).
    const auto cutlass_gelu_res = RunMatmulGelu("cutlass_gelu", device, dtype, inputs,
                                                /*transpose_a=*/false, /*transpose_b=*/false,
                                                /*use_fp16_accum=*/false, std::nullopt);

    // Fused GELU forward matmul (prefer triton).
    const auto triton_gelu_res = RunMatmulGelu("triton_gelu", device, dtype, inputs,
                                               /*transpose_a=*/false, /*transpose_b=*/false,
                                               /*use_fp16_accum=*/false, std::optional{"triton"});

    // Fused GELU backward matmul (cutlass default, triton fallback).
    const auto cutlass_gelu_bwd_res =
        RunMatmulGeluBwd("cutlass_gelu_bwd", device, dtype, inputs.a_nn, inputs.b_nn, inputs.preact_out,
                         /*transpose_a=*/false, /*transpose_b=*/false,
                         /*use_fp16_accum=*/false, std::nullopt);
    const auto cutlass_gelu_bwd_tb_aligned_res =
        RunMatmulGeluBwd("cutlass_gelu_bwd_tb_aligned", device, dtype, inputs.a_nn, inputs.b_t, inputs.preact_out,
                         /*transpose_a=*/false, /*transpose_b=*/true,
                         /*use_fp16_accum=*/false, std::nullopt);

    // Fused GELU backward matmul (prefer triton).
    const auto triton_gelu_bwd_res =
        RunMatmulGeluBwd("triton_gelu_bwd", device, dtype, inputs.a_nn, inputs.b_nn, inputs.preact_out,
                         /*transpose_a=*/false, /*transpose_b=*/false,
                         /*use_fp16_accum=*/false, std::optional{"triton"});

    const auto triton_gelu_bwd_tb_aligned_res =
        RunMatmulGeluBwd("triton_gelu_bwd_tb_aligned", device, dtype, inputs.a_nn, inputs.b_t, inputs.preact_out,
                         /*transpose_a=*/false, /*transpose_b=*/true,
                         /*use_fp16_accum=*/false, std::optional{"triton"});

    const auto gelu_bwd_unaligned_inputs = InitializeGeluBwdInputs(device, dtype, M_DIM_UNALIGNED, N_DIM, K_DIM);
    const auto cutlass_gelu_bwd_tb_unaligned_res =
        RunMatmulGeluBwd("cutlass_gelu_bwd_tb_unaligned", device, dtype, gelu_bwd_unaligned_inputs.a_nn,
                         gelu_bwd_unaligned_inputs.b_t, gelu_bwd_unaligned_inputs.preact_out,
                         /*transpose_a=*/false, /*transpose_b=*/true,
                         /*use_fp16_accum=*/false, std::nullopt);
    const auto triton_gelu_bwd_tb_unaligned_res =
        RunMatmulGeluBwd("triton_gelu_bwd_tb_unaligned", device, dtype, gelu_bwd_unaligned_inputs.a_nn,
                         gelu_bwd_unaligned_inputs.b_t, gelu_bwd_unaligned_inputs.preact_out,
                         /*transpose_a=*/false, /*transpose_b=*/true,
                         /*use_fp16_accum=*/false, std::optional{"triton"});

    // Fused addmm+gelu with pre-activation write-out (cutlass path).
    const auto cutlass_addmm_gelu_preact_res = RunAddmmGeluPreact("cutlass_addmm_gelu_preact", device, dtype, inputs,
                                                                  /*transpose_a=*/false, /*transpose_b=*/false,
                                                                  /*use_fp16_accum=*/false, std::nullopt);

    // Fused addmm+gelu with pre-activation write-out (triton path).
    const auto triton_addmm_gelu_preact_res = RunAddmmGeluPreact("triton_addmm_gelu_preact", device, dtype, inputs,
                                                                 /*transpose_a=*/false, /*transpose_b=*/false,
                                                                 /*use_fp16_accum=*/false, std::optional{"triton"});

    // Simple sanity checks to keep this as a unit test.
    if (cutlass_res.tflops <= 0.0 || cutlass_ta_res.tflops <= 0.0 || cutlass_tb_res.tflops <= 0.0 ||
        cutlass_tab_res.tflops <= 0.0 || triton_res.tflops <= 0.0 || triton_ta_res.tflops <= 0.0 ||
        triton_tb_res.tflops <= 0.0 || triton_tab_res.tflops <= 0.0 || cutlass_gelu_res.tflops <= 0.0 ||
        triton_gelu_res.tflops <= 0.0 || cutlass_gelu_bwd_res.tflops <= 0.0 ||
        cutlass_gelu_bwd_tb_aligned_res.tflops <= 0.0 || triton_gelu_bwd_res.tflops <= 0.0 ||
        triton_gelu_bwd_tb_aligned_res.tflops <= 0.0 || cutlass_gelu_bwd_tb_unaligned_res.tflops <= 0.0 ||
        triton_gelu_bwd_tb_unaligned_res.tflops <= 0.0 || cutlass_addmm_gelu_preact_res.tflops <= 0.0 ||
        triton_addmm_gelu_preact_res.tflops <= 0.0)
    {
        std::cerr << "Measured TFLOPs must be positive for all paths\n";
        return 1;
    }
    return 0;
}
