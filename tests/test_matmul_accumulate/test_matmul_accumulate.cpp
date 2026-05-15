#include "testing.h"

#include <allocator.h>
#include <execution_backend.h>
#include <executor.h>
#include <functional.h>
#include <linear.h>
#include <passes.h>
#include <tensorlib.h>
#include <utils.h>

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

using namespace pi::tensorlib;

namespace
{
    constexpr Device DEVICE_CPU{DeviceType::CPU, 0};
    constexpr Device DEVICE_GPU{DeviceType::GPU, 0};

    void ApplyPasses(ExecutionPlan &plan)
    {
        std::vector<std::unique_ptr<pi::tensorlib::passes::CompilerPass>> passes{};
        passes.emplace_back(std::make_unique<FillConstantImplPass>());
        passes.emplace_back(std::make_unique<FillUniformImplPass>());
        passes.emplace_back(std::make_unique<TrailingBroadcastAddImplPass>());
        passes.emplace_back(std::make_unique<ReduceSumImplPass>());
        passes.emplace_back(std::make_unique<ContiguousImplPass>());
        passes.emplace_back(std::make_unique<CastImplPass>());
        passes.emplace_back(std::make_unique<MatmulImplPass>());
        pi::tensorlib::passes::Transform(plan, passes);
    }

    std::shared_ptr<RealTensor> MakeHostTensor(std::initializer_list<uint64_t> shape, const std::vector<float> &values,
                                               const DataType dtype)
    {
        auto tensor = RealTensor::Allocate(shape, dtype, DEVICE_CPU);
        auto *dst = reinterpret_cast<uint16_t *>(tensor->dataptr());
        for (size_t i = 0; i < values.size(); ++i)
        {
            dst[i] = (dtype == DataType::BFLOAT16) ? utils::Bf16FromFp32(values[i]) : utils::Fp16FromFp32(values[i]);
        }
        return tensor;
    }

    void RunMatmulAccumulateCase(const int m, const int k, const int n, const bool disable_cutlass)
    {
        if (disable_cutlass)
        {
            setenv("FBAMTRAIN_PREFER_GEMM_BACKEND", "triton", 1);
        }
        else
        {
            setenv("FBAMTRAIN_PREFER_GEMM_BACKEND", "cutlass", 1);
        }
        setenv("FBAMTRAIN_DEBUG_GEMM", "1", 1);

        ExecutionBackend &backend = ExecutionBackend::getInstance();
        const auto &allocator_registry = allocator::DefaultAllocatorRegistry::instance();

        std::vector<float> a_vals(static_cast<size_t>(m * k), 1.0f);
        std::vector<float> b_vals(static_cast<size_t>(k * n), 1.0f);
        auto a_host = MakeHostTensor({static_cast<uint64_t>(m), static_cast<uint64_t>(k)}, a_vals, DataType::FLOAT16);
        auto b_host = MakeHostTensor({static_cast<uint64_t>(k), static_cast<uint64_t>(n)}, b_vals, DataType::FLOAT16);

        const auto main_stream_desc = GpuStreamDescriptors::Main;
        TraceTensor a_trace = TraceTensor::Create({static_cast<uint64_t>(m), static_cast<uint64_t>(k)},
                                                  DataType::FLOAT16, DEVICE_GPU, main_stream_desc);
        TraceTensor b_trace = TraceTensor::Create({static_cast<uint64_t>(k), static_cast<uint64_t>(n)},
                                                  DataType::FLOAT16, DEVICE_GPU, main_stream_desc);
        a_trace.markRetained();
        b_trace.markRetained();
        OpGraph graph({{.name = "a", .tensor = a_trace}, {.name = "b", .tensor = b_trace}}, {});

        TraceTensor output =
            graph.createTensor({static_cast<uint64_t>(m), static_cast<uint64_t>(n)}, DataType::FLOAT16,
                               DEVICE_GPU, main_stream_desc, false);
        FillConstant(graph, output, 1.0f, main_stream_desc);

        std::unordered_map<std::string, std::any> attrs{};
        attrs.emplace("accumulate_output", true);
        graph.recordOperation(
            OperationEntry{.type = OpType::MATMUL,
                           .inputs = {graph.getInputDescriptors()[0].tensor, graph.getInputDescriptors()[1].tensor},
                           .outputs = {output},
                           .attributes = attrs,
                           .gpu_stream_desc = main_stream_desc});

        graph.finalize();

        auto a_gpu = a_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
        auto b_gpu = b_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
        ExecutionPlan plan =
            ExecutionPlan::FromGraph(graph, {{.name = "a", .tensor = a_gpu}, {.name = "b", .tensor = b_gpu}}, {});
        ApplyPasses(plan);
        Executor executor{plan, backend, 0};
        executor.execute(allocator_registry);

        const auto output_rt = executor.getOutput(output);
        if (!output_rt)
        {
            throw std::runtime_error("Failed to retrieve matmul accumulate output");
        }

        // Expected = ones + A @ B
        auto expected =
            RealTensor::Allocate({static_cast<uint64_t>(m), static_cast<uint64_t>(n)}, DataType::FLOAT16, DEVICE_CPU);
        auto *exp_ptr = reinterpret_cast<uint16_t *>(expected->dataptr());
        const float expected_val = static_cast<float>(k + 1); // initial 1 + sum over k ones
        for (size_t i = 0; i < static_cast<size_t>(m * n); ++i)
        {
            exp_ptr[i] = utils::Fp16FromFp32(expected_val);
        }
        testing::AssertSimilar(expected, *output_rt, 5e-3f);

        unsetenv("FBAMTRAIN_DEBUG_GEMM");
        unsetenv("FBAMTRAIN_PREFER_GEMM_BACKEND");
    }

    void TestMatmulAccumulate()
    {
        // Triton path forced.
        RunMatmulAccumulateCase(/*m=*/8, /*k=*/8, /*n=*/8, /*disable_cutlass=*/true);
        // Cutlass path (alignment-friendly sizes).
        RunMatmulAccumulateCase(/*m=*/256, /*k=*/256, /*n=*/256, /*disable_cutlass=*/false);
    }

    void TestLinearBackwardAccumulate()
    {
        setenv("FBAMTRAIN_PREFER_GEMM_BACKEND", "triton", 1);
        ExecutionBackend &backend = ExecutionBackend::getInstance();
        const auto &allocator_registry = allocator::DefaultAllocatorRegistry::instance();

        const auto main_stream_desc = GpuStreamDescriptors::Main;
        TraceTensor x_trace = TraceTensor::Create({2, 8}, DataType::FLOAT16, DEVICE_GPU, main_stream_desc);
        TraceTensor upstream_trace = TraceTensor::Create({2, 8}, DataType::FLOAT16, DEVICE_GPU, main_stream_desc);
        x_trace.markRetained();
        upstream_trace.markRetained();
        OpGraph graph({{.name = "x", .tensor = x_trace}, {.name = "upstream", .tensor = upstream_trace}}, {});

        uint32_t seed = 0;
        Linear linear{"linear", 8, 8, DEVICE_GPU, DataType::FLOAT16, NONE, graph, seed, main_stream_desc};

        // Forward with saved input.
        TraceTensor logits = linear.buildForward(graph, {graph.getInputDescriptors()[0].tensor},
                                                 /*save_input_for_backward=*/true);
        (void)logits;

        // Upstream gradient already provided as input descriptor.
        std::unordered_map<std::string, TraceTensor> parameter_grads{};
        TraceTensor w_grad =
            graph.createTensor(linear.parameters()[0].tensor.shape().dims(), DataType::FLOAT16, DEVICE_GPU,
                               main_stream_desc, false);
        TraceTensor b_grad =
            graph.createTensor(linear.parameters()[1].tensor.shape().dims(), DataType::FLOAT16, DEVICE_GPU,
                               main_stream_desc, false);
        FillConstant(graph, w_grad, 1.0f, main_stream_desc);
        FillConstant(graph, b_grad, 1.0f, main_stream_desc);
        parameter_grads.emplace(linear.parameters()[0].name, w_grad);
        parameter_grads.emplace(linear.parameters()[1].name, b_grad);

        std::unordered_map<std::string, TraceTensor> operand_grads{};
        linear.buildBackward(graph, graph.getInputDescriptors()[1].tensor, parameter_grads, operand_grads);

        graph.finalize();

        // Inputs and upstream.
        // Inputs: all ones, upstream half 0.5 and half -0.5 to keep dw/db = 0.
        auto x_host = MakeHostTensor({2, 8}, std::vector<float>(16, 1.0f), DataType::FLOAT16);
        auto upstream_host = MakeHostTensor(
            {2, 8},
            {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f},
            DataType::FLOAT16);

        auto x_gpu = x_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
        auto upstream_gpu = upstream_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
        ExecutionPlan plan = ExecutionPlan::FromGraph(graph,
                                                      {
                                                          {.name = "x", .tensor = x_gpu},
                                                          {.name = "upstream", .tensor = upstream_gpu},
                                                      },
                                                      {});

        ApplyPasses(plan);
        Executor executor{plan, backend, 0};
        executor.execute(allocator_registry);

        const auto w_grad_rt = executor.getOutput(w_grad);
        const auto b_grad_rt = executor.getOutput(b_grad);
        if (!w_grad_rt || !b_grad_rt)
        {
            throw std::runtime_error("Failed to retrieve accumulated linear gradients");
        }

        // dw and db should be zeros; accumulation should leave the initial ones.
        std::vector<float> dw(64, 0.0f);
        std::array<float, 8> db{};

        auto expected_w = RealTensor::Allocate({8, 8}, DataType::FLOAT16, DEVICE_CPU);
        auto *exp_w = reinterpret_cast<uint16_t *>(expected_w->dataptr());
        for (size_t i = 0; i < dw.size(); ++i)
        {
            exp_w[i] = utils::Fp16FromFp32(dw[i] + 1.0f);
        }

        auto expected_b = RealTensor::Allocate({8}, DataType::FLOAT16, DEVICE_CPU);
        auto *exp_b = reinterpret_cast<uint16_t *>(expected_b->dataptr());
        for (size_t i = 0; i < db.size(); ++i)
        {
            exp_b[i] = utils::Fp16FromFp32(db[i] + 1.0f);
        }

        testing::AssertSimilar(expected_w, *w_grad_rt, 5e-3f);
        testing::AssertSimilar(expected_b, *b_grad_rt, 5e-3f);

        unsetenv("FBAMTRAIN_PREFER_GEMM_BACKEND");
    }
} // namespace

int main()
{
    TestMatmulAccumulate();
    TestLinearBackwardAccumulate();
    return 0;
}
