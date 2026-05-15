#include "testing.h"

#include <allocator.h>
#include <execution_backend.h>
#include <executor.h>
#include <functional.h>
#include <passes.h>
#include <tensorlib.h>
#include <utils.h>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace pi::tensorlib;

namespace
{
    constexpr Device DEVICE_CPU{DeviceType::CPU, 0};
    constexpr Device DEVICE_GPU{DeviceType::GPU, 0};

    std::shared_ptr<RealTensor> MakeHostFp16Tensor(const std::initializer_list<uint64_t> &shape,
                                                   const std::vector<float> &values)
    {
        auto tensor = RealTensor::Allocate(shape, DataType::FLOAT16, DEVICE_CPU);
        auto *dst = static_cast<uint16_t *>(tensor->dataptr());
        for (size_t i = 0; i < values.size(); ++i)
        {
            dst[i] = utils::Fp16FromFp32(values[i]);
        }
        return tensor;
    }

    std::shared_ptr<RealTensor> ComputeExpected(const std::vector<float> &a_vals, const std::vector<float> &b_vals,
                                                int m, int k, int n, float base_value, int repeat_count)
    {
        auto expected = RealTensor::Allocate({static_cast<uint64_t>(m), static_cast<uint64_t>(n)}, DataType::FLOAT32,
                                             DEVICE_CPU);
        auto *out_data = static_cast<float *>(expected->dataptr());
        for (int row = 0; row < m; ++row)
        {
            for (int col = 0; col < n; ++col)
            {
                float acc = 0.0f;
                for (int kk = 0; kk < k; ++kk)
                {
                    const float a_val = a_vals[static_cast<size_t>(kk * m + row)]; // A_base[k, m] transposed.
                    const float b_val = b_vals[static_cast<size_t>(kk * n + col)];
                    acc += a_val * b_val;
                }
                out_data[row * n + col] = base_value + acc * static_cast<float>(repeat_count);
            }
        }
        return expected;
    }
} // namespace

int main()
{
    constexpr int m = 7;
    constexpr int k = 11;
    constexpr int n = 13;
    constexpr int repeat_count = 4;
    constexpr int runs = 50;
    constexpr float base_value = 0.25f;

    std::vector<float> a_vals(static_cast<size_t>(m * k));
    std::vector<float> b_vals(static_cast<size_t>(k * n));
    for (int row = 0; row < k; ++row)
    {
        for (int col = 0; col < m; ++col)
        {
            const int idx = row * m + col;
            a_vals[static_cast<size_t>(idx)] = 0.01f * static_cast<float>(idx - 17);
        }
    }
    for (int row = 0; row < k; ++row)
    {
        for (int col = 0; col < n; ++col)
        {
            const int idx = row * n + col;
            b_vals[static_cast<size_t>(idx)] = 0.005f * static_cast<float>(idx - 29);
        }
    }

    auto a_host = MakeHostFp16Tensor({static_cast<uint64_t>(k), static_cast<uint64_t>(m)}, a_vals);
    auto b_host = MakeHostFp16Tensor({static_cast<uint64_t>(k), static_cast<uint64_t>(n)}, b_vals);
    auto a_gpu = a_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
    auto b_gpu = b_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);

    auto expected = ComputeExpected(a_vals, b_vals, m, k, n, base_value, repeat_count);

    const auto main_stream_desc = GpuStreamDescriptors::Main;
    TraceTensor a_base_trace =
        TraceTensor::Create({static_cast<uint64_t>(k), static_cast<uint64_t>(m)}, DataType::FLOAT16, DEVICE_GPU,
                            main_stream_desc);
    TraceTensor b_base_trace =
        TraceTensor::Create({static_cast<uint64_t>(k), static_cast<uint64_t>(n)}, DataType::FLOAT16, DEVICE_GPU,
                            main_stream_desc);
    a_base_trace.markRetained();
    b_base_trace.markRetained();
    OpGraph graph({{.name = "a_base", .tensor = a_base_trace}, {.name = "b_base", .tensor = b_base_trace}}, {});

    TraceTensor a_view = graph.getInputDescriptors()[0].tensor.transpose(graph, {1, 0});
    TraceTensor b_view = graph.getInputDescriptors()[1].tensor;

    TraceTensor out = graph.createTensor({static_cast<uint64_t>(m), static_cast<uint64_t>(n)}, DataType::FLOAT32,
                                         DEVICE_GPU, main_stream_desc, false);
    FillConstant(graph, out, base_value, main_stream_desc);

    std::unordered_map<std::string, std::any> attrs{};
    attrs.emplace("accumulate_output", true);
    for (int i = 0; i < repeat_count; ++i)
    {
        graph.recordOperation(
            OperationEntry{.type = OpType::MATMUL,
                           .inputs = {a_view, b_view},
                           .outputs = {out},
                           .attributes = attrs,
                           .gpu_stream_desc = main_stream_desc});
    }
    graph.finalize();

    std::vector<GraphExecutionInputDescriptor> inputs{
        {.name = "a_base", .tensor = a_gpu},
        {.name = "b_base", .tensor = b_gpu},
    };
    ExecutionPlan plan = ExecutionPlan::FromGraph(graph, inputs, {});

    std::vector<std::unique_ptr<pi::tensorlib::passes::CompilerPass>> passes{};
    passes.emplace_back(std::make_unique<FillConstantImplPass>());
    passes.emplace_back(std::make_unique<MatmulImplPass>());
    pi::tensorlib::passes::Transform(plan, passes);

    ExecutionBackend &backend = ExecutionBackend::getInstance();
    const auto &allocator_registry = allocator::CachingAllocatorRegistry::instance();

    setenv("FBAMTRAIN_PREFER_CONV2D_BACKEND", "triton", 1);
    setenv("FBAMTRAIN_PREFER_GEMM_BACKEND", "triton", 1);
    setenv("FBAMTRAIN_PREFER_MHA_BACKEND", "triton", 1);

    setenv("FBAMTRAIN_DEBUG_GEMM", "1", 1);
    std::stringstream debug_stream;
    auto *old_buf = std::clog.rdbuf(debug_stream.rdbuf());

    Executor executor{plan, backend, 0};
    executor.execute(allocator_registry);
    const auto out_rt = executor.getOutput(out);

    std::clog.rdbuf(old_buf);
    unsetenv("FBAMTRAIN_DEBUG_GEMM");

    const std::string debug_log = debug_stream.str();
    if (debug_log.find("matmul_fp16_out_fp32_unaligned_ta_cacc") == std::string::npos)
    {
        throw std::runtime_error("Expected matmul_fp16_out_fp32_unaligned_ta_cacc, got: " + debug_log);
    }

    if (!out_rt)
    {
        throw std::runtime_error("Failed to retrieve matmul output");
    }
    testing::AssertSimilar(expected, *out_rt, 1e-2);

    for (int run = 1; run < runs; ++run)
    {
        executor.execute(allocator_registry);
        const auto output = executor.getOutput(out);
        if (!output)
        {
            throw std::runtime_error("Failed to retrieve matmul output on run " + std::to_string(run));
        }
        try
        {
            testing::AssertSimilar(expected, *output, 1e-2);
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error("Run " + std::to_string(run) + " failed: " + e.what());
        }
    }

    return 0;
}
