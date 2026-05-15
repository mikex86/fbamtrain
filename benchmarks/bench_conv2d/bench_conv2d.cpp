#include "benchmark_utils.h"

#include <allocator.h>
#include <conv2d.h>
#include <execution_backend.h>
#include <executor.h>
#include <functional.h>
#include <mfu.h>
#include <passes.h>
#include <tensorlib.h>

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace
{
    // Matches working_dir/run-configs/debug.json:
    // micro_batch_size=32, frame_rows=48, frame_cols=160, downsample_conv_dilation=2, n_embed=1024.
    constexpr uint32_t BATCH = 32;
    constexpr uint32_t IN_CHANNELS = 1024;
    constexpr uint32_t OUT_CHANNELS = 1024;
    constexpr uint32_t HEIGHT = 48;
    constexpr uint32_t WIDTH = 160;
    constexpr uint32_t KERNEL = 3;
    constexpr uint32_t STRIDE = 1;
    constexpr uint32_t PADDING = 2;
    constexpr uint32_t DILATION = 2;

    class ScopedEnvOverride
    {
      public:
        ScopedEnvOverride(const char *name, const char *value) : name_(name)
        {
            if (const char *old = std::getenv(name))
            {
                had_old_value_ = true;
                old_value_ = old;
            }
            setenv(name, value, 1);
        }

        ~ScopedEnvOverride()
        {
            if (had_old_value_)
            {
                setenv(name_.c_str(), old_value_.c_str(), 1);
            }
            else
            {
                unsetenv(name_.c_str());
            }
        }

        ScopedEnvOverride(const ScopedEnvOverride &) = delete;
        ScopedEnvOverride &operator=(const ScopedEnvOverride &) = delete;

      private:
        std::string name_;
        std::string old_value_;
        bool had_old_value_{false};
    };

    bench_utils::BenchmarkConfig LoadConv2dBenchmarkConfig()
    {
        bench_utils::BenchmarkConfig cfg{
            .warmup_runs = bench_utils::detail::ParseSizeEnv("BENCH_WARMUP", 8),
            .measure_runs = bench_utils::detail::ParseSizeEnv("BENCH_ITERS", 128),
            .dtype = bench_utils::detail::ParseBenchDtype(),
        };
        if (cfg.measure_runs == 0)
        {
            std::cerr << "[bench] Warning: BENCH_ITERS resolved to zero, forcing measure_runs=1\n";
            cfg.measure_runs = 1;
        }
        return cfg;
    }

    void ApplyDefaultPasses(pi::tensorlib::ExecutionPlan &execution_plan)
    {
        std::vector<std::unique_ptr<pi::tensorlib::passes::CompilerPass>> passes{};
        passes.emplace_back(std::make_unique<Conv2dImplPass>());
        passes.emplace_back(std::make_unique<ContiguousImplPass>());
        passes.emplace_back(std::make_unique<FillZerosImplPass>());
        passes.emplace_back(std::make_unique<FillUniformImplPass>());
        pi::tensorlib::passes::Transform(execution_plan, passes);
    }

    uint64_t ComputeOutputExtent(const uint32_t input, const uint32_t padding, const uint32_t dilation,
                                 const uint32_t kernel, const uint32_t stride)
    {
        const uint64_t effective_kernel = 1ULL + static_cast<uint64_t>(dilation) * (kernel - 1U);
        const int64_t numerator =
            static_cast<int64_t>(input) + 2LL * static_cast<int64_t>(padding) - static_cast<int64_t>(effective_kernel);
        return static_cast<uint64_t>(numerator / static_cast<int64_t>(stride) + 1);
    }

    double ComputeConv2dFwdFlops()
    {
        const double height_out = static_cast<double>(ComputeOutputExtent(HEIGHT, PADDING, DILATION, KERNEL, STRIDE));
        const double width_out = static_cast<double>(ComputeOutputExtent(WIDTH, PADDING, DILATION, KERNEL, STRIDE));
        const double points = static_cast<double>(BATCH) * height_out * width_out * static_cast<double>(OUT_CHANNELS);
        const double kernel_muladds = static_cast<double>(IN_CHANNELS) * static_cast<double>(KERNEL) *
                                      static_cast<double>(KERNEL) * 2.0;
        return points * kernel_muladds;
    }

    double ComputeConv2dDgradFlops()
    {
        const double points =
            static_cast<double>(BATCH) * static_cast<double>(HEIGHT) * static_cast<double>(WIDTH) *
            static_cast<double>(IN_CHANNELS);
        const double kernel_muladds = static_cast<double>(OUT_CHANNELS) * static_cast<double>(KERNEL) *
                                      static_cast<double>(KERNEL) * 2.0;
        return points * kernel_muladds;
    }

    double ComputeConv2dWgradFlops()
    {
        const double height_out = static_cast<double>(ComputeOutputExtent(HEIGHT, PADDING, DILATION, KERNEL, STRIDE));
        const double width_out = static_cast<double>(ComputeOutputExtent(WIDTH, PADDING, DILATION, KERNEL, STRIDE));
        const double points = static_cast<double>(BATCH) * height_out * width_out * static_cast<double>(OUT_CHANNELS);
        const double kernel_muladds = static_cast<double>(IN_CHANNELS) * static_cast<double>(KERNEL) *
                                      static_cast<double>(KERNEL) * 2.0;
        return points * kernel_muladds;
    }

    bool IsEnvTruthy(const char *env_name)
    {
        if (const char *env = std::getenv(env_name))
        {
            const std::string_view value{env};
            return !(value.empty() || value == "0" || value == "false");
        }
        return false;
    }

    double RunTimed(pi::tensorlib::Executor &executor,
                    pi::tensorlib::allocator::CachingAllocatorRegistry &allocator_registry,
                    const pi::tensorlib::Device &device, const bench_utils::BenchmarkConfig &config)
    {
        for (std::size_t i = 0; i < config.warmup_runs; ++i)
        {
            executor.execute(allocator_registry);
        }

        auto start_event = pi::tensorlib::ExecutionBackend::CreateEvent(device, true);
        auto end_event = pi::tensorlib::ExecutionBackend::CreateEvent(device, true);
        const auto stream_bundle = pi::tensorlib::ExecutionBackend::GetStreamBundle(device);
        const auto &main_stream = stream_bundle->main_stream;

        start_event.record(main_stream);
        for (std::size_t i = 0; i < config.measure_runs; ++i)
        {
            executor.execute(allocator_registry, false);
        }
        end_event.record(main_stream);
        end_event.synchronize();
        executor.await();

        const double total_ms = (config.measure_runs > 0) ? end_event.elapsedMsSince(start_event) : 0.0;
        return total_ms / static_cast<double>(config.measure_runs);
    }

    pi::tensorlib::PrecisionMode SelectPrecisionMode(const pi::tensorlib::DataType dtype,
                                                     const bool use_fp16_conv_acc)
    {
        using pi::tensorlib::DataType;
        using pi::tensorlib::PrecisionMode;
        if (dtype == DataType::BFLOAT16)
        {
            return PrecisionMode::BF16;
        }
        if (dtype == DataType::FLOAT16)
        {
            return use_fp16_conv_acc ? PrecisionMode::FP16_ACC16 : PrecisionMode::FP16;
        }
        return PrecisionMode::FP32;
    }

    void PrintConv2dResult(const std::string_view label, const bench_utils::BenchmarkConfig &config,
                           const double avg_ms, const double tflops, const double flops_per_iter,
                           const std::optional<float> promised)
    {
        const double mfu = promised ? (tflops / static_cast<double>(*promised) * 100.0) : -1.0;
        std::cout << "[conv2d] label=" << label << " dtype=" << pi::tensorlib::GetDataTypeName(config.dtype)
                  << " warmup=" << config.warmup_runs << " iters=" << config.measure_runs << " avg_ms=" << avg_ms
                  << " tflops=" << tflops << " mfu=" << mfu;
        if (promised)
        {
            std::cout << " promised_tflops=" << *promised;
        }
        std::cout << " flops_per_iter=" << flops_per_iter << '\n';
    }
} // namespace

int main()
{
    const bench_utils::BenchmarkConfig config = LoadConv2dBenchmarkConfig();
    const bool use_fp16_conv_acc = (config.dtype == pi::tensorlib::DataType::FLOAT16 && IsEnvTruthy("BENCH_FP16_ACC"));
    const auto main_stream_desc = pi::tensorlib::GpuStreamDescriptors::Main;

    pi::tensorlib::ExecutionBackend &backend = pi::tensorlib::ExecutionBackend::getInstance();
    auto &allocator_registry = pi::tensorlib::allocator::CachingAllocatorRegistry::instance();

    constexpr pi::tensorlib::Device device{
        .device_type = pi::tensorlib::DeviceType::GPU,
        .ordinal = 0,
    };

    pi::tensorlib::OpGraph init_graph{{}, {}};
    pi::tensorlib::TraceTensor input =
        init_graph.createTensor({BATCH, HEIGHT, WIDTH, IN_CHANNELS}, config.dtype, device, main_stream_desc, false);
    const uint64_t out_h = ComputeOutputExtent(HEIGHT, PADDING, DILATION, KERNEL, STRIDE);
    const uint64_t out_w = ComputeOutputExtent(WIDTH, PADDING, DILATION, KERNEL, STRIDE);
    pi::tensorlib::TraceTensor upstream =
        init_graph.createTensor({BATCH, out_h, out_w, OUT_CHANNELS}, config.dtype, device, main_stream_desc, false);
    uint32_t init_seed = 1337;
    pi::tensorlib::FillUniform(init_graph, input, -0.5f, 0.5f, init_seed++, main_stream_desc);
    pi::tensorlib::FillUniform(init_graph, upstream, -0.5f, 0.5f, init_seed++, main_stream_desc);

    struct pi::tensorlib::Conv2d conv("conv", IN_CHANNELS, OUT_CHANNELS, KERNEL, STRIDE, PADDING, DILATION, device,
                                      config.dtype, init_graph, init_seed, main_stream_desc, false, use_fp16_conv_acc);

    init_graph.finalize();

    pi::tensorlib::ExecutionPlan init_plan = pi::tensorlib::ExecutionPlan::FromGraph(init_graph, {}, {});
    ApplyDefaultPasses(init_plan);
    pi::tensorlib::Executor init_executor{init_plan, backend, 0};
    init_executor.execute(allocator_registry);

    const auto input_real_opt = init_executor.getOutput(input);
    if (!input_real_opt)
    {
        throw std::runtime_error("Failed to retrieve conv2d input tensor");
    }
    const auto upstream_real_opt = init_executor.getOutput(upstream);
    if (!upstream_real_opt)
    {
        throw std::runtime_error("Failed to retrieve conv2d upstream tensor");
    }

    const auto conv_params = conv.parameters();

    std::vector<pi::tensorlib::GraphInputDescriptor> parameter_descriptors{};
    parameter_descriptors.reserve(conv_params.size());
    for (const auto &[name, tensor] : conv_params)
    {
        auto retained_tensor = tensor;
        retained_tensor.markRetained();
        parameter_descriptors.push_back(pi::tensorlib::GraphInputDescriptor{.name = name, .tensor = retained_tensor});
    }

    input.markRetained();
    upstream.markRetained();
    pi::tensorlib::OpGraph graph(
        {
            {.name = "input", .tensor = input},
        },
        parameter_descriptors);

    pi::tensorlib::TraceTensor output = conv.buildForward(graph, {input});
    graph.finalize();

    std::vector<pi::tensorlib::GraphExecutionInputDescriptor> parameter_execution_inputs{};
    parameter_execution_inputs.reserve(conv_params.size());
    for (const auto &[name, tensor] : conv_params)
    {
        const auto tensor_real_opt = init_executor.getOutput(tensor);
        if (!tensor_real_opt)
        {
            throw std::runtime_error(std::string("Failed to resolve conv parameter tensor ") + name);
        }
        parameter_execution_inputs.push_back(
            pi::tensorlib::GraphExecutionInputDescriptor{.name = name, .tensor = *tensor_real_opt});
    }

    pi::tensorlib::ExecutionPlan plan =
        pi::tensorlib::ExecutionPlan::FromGraph(graph,
                                                {
                                                    {.name = "input", .tensor = *input_real_opt},
                                                },
                                                parameter_execution_inputs);
    ApplyDefaultPasses(plan);

    pi::tensorlib::Executor executor{plan, backend, 0};

    const double avg_ms = RunTimed(executor, allocator_registry, device, config);
    const double avg_sec = avg_ms / 1e3;
    const double flops_per_iter = ComputeConv2dFwdFlops();
    const double tflops = flops_per_iter / avg_sec / 1e12;
    const auto promised = pi::tensorlib::GetPromisedTFlops(device.ordinal,
                                                           SelectPrecisionMode(config.dtype, use_fp16_conv_acc));

    const auto output_real_opt = executor.getOutput(output);
    if (!output_real_opt)
    {
        throw std::runtime_error("Failed to fetch conv2d output tensor");
    }

    PrintConv2dResult("conv2d_fwd", config, avg_ms, tflops, flops_per_iter, promised);

    const auto run_wgrad_variant = [&](const char *implementation, const char *label)
    {
        ScopedEnvOverride backend_env{"FBAMTRAIN_PREFER_CONV2D_BACKEND", "cutlass"};
        ScopedEnvOverride implementation_env{"FBAMTRAIN_CONV2D_WGRAD_CUTLASS_IMPL", implementation};

        pi::tensorlib::OpGraph wgrad_graph(
            {
                {.name = "input", .tensor = input},
                {.name = "upstream", .tensor = upstream},
            },
            parameter_descriptors);

        auto grad_weight =
            wgrad_graph.createTensor(conv_params[0].tensor.shape().dims(), config.dtype, device, main_stream_desc, false);
        Conv2dWgradInto(wgrad_graph, input, upstream, grad_weight, {STRIDE, STRIDE}, {PADDING, PADDING},
                        {DILATION, DILATION}, main_stream_desc);
        wgrad_graph.finalize();

        pi::tensorlib::ExecutionPlan wgrad_plan =
            pi::tensorlib::ExecutionPlan::FromGraph(wgrad_graph,
                                                    {
                                                        {.name = "input", .tensor = *input_real_opt},
                                                        {.name = "upstream", .tensor = *upstream_real_opt},
                                                    },
                                                    parameter_execution_inputs);
        ApplyDefaultPasses(wgrad_plan);

        pi::tensorlib::Executor wgrad_executor{wgrad_plan, backend, 0};
        const double wgrad_avg_ms = RunTimed(wgrad_executor, allocator_registry, device, config);
        const double wgrad_avg_sec = wgrad_avg_ms / 1e3;
        const double wgrad_flops = ComputeConv2dWgradFlops();
        const double wgrad_tflops = wgrad_flops / wgrad_avg_sec / 1e12;

        const auto grad_weight_real_opt = wgrad_executor.getOutput(grad_weight);
        if (!grad_weight_real_opt)
        {
            throw std::runtime_error("Failed to fetch conv2d wgrad output tensor");
        }
        PrintConv2dResult(label, config, wgrad_avg_ms, wgrad_tflops, wgrad_flops, promised);
    };

    run_wgrad_variant("cutlass2", "conv2d_wgrad_cutlass2");
    run_wgrad_variant("cutlass3", "conv2d_wgrad_cutlass3");

    pi::tensorlib::OpGraph bwd_graph(
        {
            {.name = "input", .tensor = input},
            {.name = "upstream", .tensor = upstream},
        },
        parameter_descriptors);

    auto grad_input = bwd_graph.createTensor(input.shape().dims(), config.dtype, device, main_stream_desc, false);
    auto grad_weight =
        bwd_graph.createTensor(conv_params[0].tensor.shape().dims(), config.dtype, device, main_stream_desc, false);
    Conv2dDgradInto(bwd_graph, upstream, conv_params[0].tensor, grad_input, {STRIDE, STRIDE}, {PADDING, PADDING},
                    {DILATION, DILATION}, main_stream_desc);
    Conv2dWgradInto(bwd_graph, input, upstream, grad_weight, {STRIDE, STRIDE}, {PADDING, PADDING},
                    {DILATION, DILATION}, main_stream_desc);
    bwd_graph.finalize();

    pi::tensorlib::ExecutionPlan bwd_plan =
        pi::tensorlib::ExecutionPlan::FromGraph(bwd_graph,
                                                {
                                                    {.name = "input", .tensor = *input_real_opt},
                                                    {.name = "upstream", .tensor = *upstream_real_opt},
                                                },
                                                parameter_execution_inputs);
    ApplyDefaultPasses(bwd_plan);

    pi::tensorlib::Executor bwd_executor{bwd_plan, backend, 0};
    const double bwd_avg_ms = RunTimed(bwd_executor, allocator_registry, device, config);
    const double bwd_avg_sec = bwd_avg_ms / 1e3;
    const double bwd_flops = ComputeConv2dDgradFlops() + ComputeConv2dWgradFlops();
    const double bwd_tflops = bwd_flops / bwd_avg_sec / 1e12;
    PrintConv2dResult("conv2d_bwd", config, bwd_avg_ms, bwd_tflops, bwd_flops, promised);
    return 0;
}
