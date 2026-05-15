#include "testing.h"

#include <allocator.h>
#include <execution_backend.h>
#include <executor.h>
#include <op_graph.h>
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
    constexpr uint64_t M = 128;
    constexpr uint64_t K = 256;
    constexpr uint64_t N = 64;

    constexpr Device DEVICE_CPU{DeviceType::CPU, 0};
    constexpr Device DEVICE_GPU{DeviceType::GPU, 0};

    bool PreferCutlassGemm()
    {
#if !PI_TENSORLIB_ENABLE_CUDA
        return false;
#endif
        const char *env = std::getenv("FBAMTRAIN_PREFER_GEMM_BACKEND");
        if (env == nullptr || env[0] == '\0')
        {
            return true;
        }
        const std::string value(env);
        if (value == "cutlass")
        {
            return true;
        }
        if (value == "triton")
        {
            return false;
        }
        throw std::runtime_error("FBAMTRAIN_PREFER_GEMM_BACKEND must be one of: cutlass, triton");
    }

    std::shared_ptr<RealTensor> MakeHostBf16Tensor(const std::initializer_list<uint64_t> &shape,
                                                   const std::vector<float> &values)
    {
        auto tensor = RealTensor::Allocate(shape, DataType::BFLOAT16, DEVICE_CPU);
        auto *dst = static_cast<uint16_t *>(tensor->dataptr());
        for (size_t i = 0; i < values.size(); ++i)
        {
            dst[i] = utils::Bf16FromFp32(values[i]);
        }
        return tensor;
    }

    std::shared_ptr<RealTensor> ComputeExpected(const std::shared_ptr<RealTensor> &a_host,
                                                const std::shared_ptr<RealTensor> &b_host)
    {
        auto expected = RealTensor::Allocate({M, N}, DataType::FLOAT32, DEVICE_CPU);
        auto *out_data = static_cast<float *>(expected->dataptr());

        const auto *a_data = static_cast<const uint16_t *>(a_host->dataptr());
        const auto *b_data = static_cast<const uint16_t *>(b_host->dataptr());

        for (uint64_t row = 0; row < M; ++row)
        {
            for (uint64_t col = 0; col < N; ++col)
            {
                float acc = 0.0f;
                for (uint64_t kk = 0; kk < K; ++kk)
                {
                    const float a_val = utils::Fp32FromBf16(a_data[row * K + kk]);
                    const float b_val = utils::Fp32FromBf16(b_data[kk * N + col]);
                    acc += a_val * b_val;
                }
                out_data[row * N + col] = acc;
            }
        }

        return expected;
    }
} // namespace

int main()
{
    std::vector<float> a_vals(static_cast<size_t>(M * K));
    std::vector<float> b_vals(static_cast<size_t>(K * N));
    for (uint64_t row = 0; row < M; ++row)
    {
        for (uint64_t col = 0; col < K; ++col)
        {
            const uint64_t idx = row * K + col;
            a_vals[static_cast<size_t>(idx)] = 0.01f * static_cast<float>(idx % 127) - 0.5f;
        }
    }
    for (uint64_t row = 0; row < K; ++row)
    {
        for (uint64_t col = 0; col < N; ++col)
        {
            const uint64_t idx = row * N + col;
            b_vals[static_cast<size_t>(idx)] = 0.005f * static_cast<float>(idx % 97) + 0.25f;
        }
    }

    const auto a_host = MakeHostBf16Tensor({M, K}, a_vals);
    const auto b_host = MakeHostBf16Tensor({K, N}, b_vals);
    const auto expected = ComputeExpected(a_host, b_host);

    const auto main_stream_desc = GpuStreamDescriptors::Main;
    TraceTensor a_cpu = TraceTensor::Create({M, K}, DataType::BFLOAT16, DEVICE_CPU, main_stream_desc);
    TraceTensor b_cpu = TraceTensor::Create({K, N}, DataType::BFLOAT16, DEVICE_CPU, main_stream_desc);
    a_cpu.markRetained();
    b_cpu.markRetained();

    OpGraph graph({{.name = "a", .tensor = a_cpu}, {.name = "b", .tensor = b_cpu}}, {});

    TraceTensor a_gpu = graph.createTensor({M, K}, DataType::BFLOAT16, DEVICE_GPU, main_stream_desc, false);
    TraceTensor b_gpu = graph.createTensor({K, N}, DataType::BFLOAT16, DEVICE_GPU, main_stream_desc, false);
    TraceTensor out_gpu = graph.createTensor({M, N}, DataType::FLOAT32, DEVICE_GPU, main_stream_desc, false);
    TraceTensor out_cpu = graph.createTensor({M, N}, DataType::FLOAT32, DEVICE_CPU, main_stream_desc, false);

    graph.recordOperation(OperationEntry{.type = OpType::DEVICE_COPY,
                                         .inputs = {a_cpu},
                                         .outputs = {a_gpu},
                                         .attributes = {},
                                         .gpu_stream_desc = main_stream_desc});
    graph.recordOperation(OperationEntry{.type = OpType::DEVICE_COPY,
                                         .inputs = {b_cpu},
                                         .outputs = {b_gpu},
                                         .attributes = {},
                                         .gpu_stream_desc = main_stream_desc});
    graph.recordOperation(OperationEntry{.type = OpType::MATMUL,
                                         .inputs = {a_gpu, b_gpu},
                                         .outputs = {out_gpu},
                                         .attributes = {},
                                         .gpu_stream_desc = main_stream_desc});
    graph.recordOperation(OperationEntry{.type = OpType::DEVICE_COPY,
                                         .inputs = {out_gpu},
                                         .outputs = {out_cpu},
                                         .attributes = {},
                                         .gpu_stream_desc = main_stream_desc});

    graph.finalize();

    ExecutionPlan plan = ExecutionPlan::FromGraph(graph, {{.name = "a", .tensor = a_host}, {.name = "b", .tensor = b_host}}, {});

    std::vector<std::unique_ptr<passes::CompilerPass>> passes{};
    passes.emplace_back(std::make_unique<MatmulImplPass>());
    pi::tensorlib::passes::Transform(plan, passes);

    ExecutionBackend &execution_backend = ExecutionBackend::getInstance();
    const auto &allocator_registry = allocator::CachingAllocatorRegistry::instance();

    setenv("FBAMTRAIN_DEBUG_GEMM", "1", 1);
    std::stringstream debug_stream;
    auto *old_buf = std::clog.rdbuf(debug_stream.rdbuf());

    Executor executor{plan, execution_backend, 0};
    executor.execute(allocator_registry);
    const auto out_rt = executor.getOutput(out_cpu);

    std::clog.rdbuf(old_buf);
    unsetenv("FBAMTRAIN_DEBUG_GEMM");

    if (!out_rt)
    {
        throw std::runtime_error("Failed to retrieve matmul output");
    }

    const std::string expected_kernel_substr = PreferCutlassGemm() ? "cutlass_matmul_bf16_out_fp32"
                                                                    : "matmul_bf16_out_fp32_cstore";
    const std::string debug_log = debug_stream.str();
    if (debug_log.find(expected_kernel_substr) == std::string::npos)
    {
        throw std::runtime_error("Expected kernel substring '" + expected_kernel_substr + "', got: " + debug_log);
    }

    testing::AssertSimilar(expected, *out_rt, 5e-3);
    return 0;
}
