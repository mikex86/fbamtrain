#include "testing.h"

#include <allocator.h>
#include <execution_backend.h>
#include <executor.h>
#include <functional.h>
#include <passes.h>
#include <safe_tensors.h>
#include <tensorlib.h>
#include <utils.h>

#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace pi::tensorlib;

namespace
{
    constexpr Device DEVICE_GPU{DeviceType::GPU, 0};
    constexpr auto DTYPE = DataType::FLOAT16;

    struct MatmulCase
    {
        std::string name;
        std::string a_key;
        std::string b_key;
        std::string expected_key;
        bool transpose_a{};
        bool transpose_b{};
        std::string expected_triton_kernel_substr;
        std::string expected_cutlass_kernel_substr;
    };

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

    TraceTensor TransposeIfNeeded(OpGraph &graph, const TraceTensor &base, const bool transpose_flag)
    {
        return transpose_flag ? base.transpose(graph, {1, 0}) : base;
    }

    void RunCase(const MatmulCase &cfg, const std::map<std::string, std::shared_ptr<RealTensor>> &tensors)
    {
        const auto &a_info = tensors.at(cfg.a_key);
        const auto &b_info = tensors.at(cfg.b_key);
        const auto &exp_info = tensors.at(cfg.expected_key);

        const std::shared_ptr<RealTensor> a_base = a_info->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
        const std::shared_ptr<RealTensor> b_base = b_info->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
        const std::shared_ptr<RealTensor> expected = exp_info;

        const auto main_stream_desc = GpuStreamDescriptors::Main;
        TraceTensor a_base_trace = TraceTensor::Create(a_base->shape().dims(), DTYPE, DEVICE_GPU, main_stream_desc);
        TraceTensor b_base_trace = TraceTensor::Create(b_base->shape().dims(), DTYPE, DEVICE_GPU, main_stream_desc);
        a_base_trace.markRetained();
        b_base_trace.markRetained();
        OpGraph graph({{.name = "a_base", .tensor = a_base_trace}, {.name = "b_base", .tensor = b_base_trace}}, {});

        TraceTensor a_view = TransposeIfNeeded(graph, graph.getInputDescriptors()[0].tensor, cfg.transpose_a);
        TraceTensor b_view = TransposeIfNeeded(graph, graph.getInputDescriptors()[1].tensor, cfg.transpose_b);

        TraceTensor out = graph.createTensor({static_cast<uint64_t>(expected->shape()[0]),
                                              static_cast<uint64_t>(expected->shape()[1])},
                                             DTYPE, DEVICE_GPU, main_stream_desc, false);
        graph.recordOperation(OperationEntry{.type = OpType::MATMUL,
                                             .inputs = {a_view, b_view},
                                             .outputs = {out},
                                             .gpu_stream_desc = main_stream_desc});
        graph.finalize();

        ExecutionPlan plan =
            ExecutionPlan::FromGraph(graph, {{.name = "a_base", .tensor = a_base}, {.name = "b_base", .tensor = b_base}}, {});

        std::vector<std::unique_ptr<pi::tensorlib::passes::CompilerPass>> passes{};
        passes.emplace_back(std::make_unique<MatmulImplPass>());
        pi::tensorlib::passes::Transform(plan, passes);

        ExecutionBackend &backend = ExecutionBackend::getInstance();
        const auto &allocator_registry = allocator::CachingAllocatorRegistry::instance();

        setenv("FBAMTRAIN_DEBUG_GEMM", "1", 1);
        std::stringstream debug_stream;
        auto *old_buf = std::clog.rdbuf(debug_stream.rdbuf());

        Executor executor{plan, backend, 0};
        executor.execute(allocator_registry);
        const auto out_rt = executor.getOutput(out);

        std::clog.rdbuf(old_buf);
        unsetenv("FBAMTRAIN_DEBUG_GEMM");

        if (!out_rt)
        {
            throw std::runtime_error("Failed to fetch matmul output");
        }

        const std::string expected_kernel_substr =
            PreferCutlassGemm() ? cfg.expected_cutlass_kernel_substr : cfg.expected_triton_kernel_substr;

        const std::string debug_log = debug_stream.str();
        if (debug_log.find(expected_kernel_substr) == std::string::npos)
        {
            std::cerr << "Expected kernel substring '" << expected_kernel_substr << "' in log, got: " << debug_log
                      << "\n";
            throw std::runtime_error("Incorrect kernel dispatched for transpose state");
        }

        pi::tensorlib::testing::AssertSimilar(expected, *out_rt, 1e-2);
        std::cout << "[transpose_matmul] case=" << cfg.name << " kernel=" << debug_log;
    }
} // namespace

int main()
{
    const auto tensor_map = safetensors::Load("reference.safetensors");

    const std::vector<MatmulCase> cases{
        {.name = "nn",
         .a_key = "a_nn",
         .b_key = "b_nn",
         .expected_key = "out_nn",
         .transpose_a = false,
         .transpose_b = false,
         .expected_triton_kernel_substr = "matmul_fp16_unaligned_cstore",
         .expected_cutlass_kernel_substr = "matmul_fp16_unaligned_cstore"},
        {.name = "ta",
         .a_key = "a_t",
         .b_key = "b_nn",
         .expected_key = "out_ta",
         .transpose_a = true,
         .transpose_b = false,
         .expected_triton_kernel_substr = "matmul_fp16_unaligned_ta_cstore",
         .expected_cutlass_kernel_substr = "cutlass_matmul_fp16_ta_kernel"},
        {.name = "tb",
         .a_key = "a_nn",
         .b_key = "b_t",
         .expected_key = "out_tb",
         .transpose_a = false,
         .transpose_b = true,
         .expected_triton_kernel_substr = "matmul_fp16_unaligned_tb_cstore",
         .expected_cutlass_kernel_substr = "cutlass_matmul_fp16_tb_kernel"},
        {.name = "tab",
         .a_key = "a_t",
         .b_key = "b_t",
         .expected_key = "out_tab",
         .transpose_a = true,
         .transpose_b = true,
         .expected_triton_kernel_substr = "matmul_fp16_unaligned_tab_cstore",
         .expected_cutlass_kernel_substr = "cutlass_matmul_fp16_tab_kernel"},
    };

    for (const auto &cfg : cases)
    {
        RunCase(cfg, tensor_map);
    }

    return 0;
}
