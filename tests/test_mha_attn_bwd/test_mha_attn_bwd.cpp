#include "testing.h"

#include "../common/test_dtype_utils.h"
#include <allocator.h>
#include <attention.h>
#include <execution_backend.h>
#include <executor.h>
#include <functional.h>
#include <passes.h>
#include <safe_tensors.h>
#include <tensorlib.h>
#include <utils.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{
    constexpr pi::tensorlib::Device DEVICE_GPU{pi::tensorlib::DeviceType::GPU, 0};

    constexpr uint64_t B = 2;
    constexpr uint64_t DEFAULT_H = 2;
    constexpr uint64_t DEFAULT_T = 128;
    constexpr uint64_t HS = 128;

    bool IsEnvFlagEnabled(const char *name)
    {
        const char *env = std::getenv(name);
        return env != nullptr && env[0] != '\0';
    }

    uint64_t GetTestHeads()
    {
        if (const char *env = std::getenv("MHA_TEST_HEADS"); env != nullptr && env[0] != '\0')
        {
            char *end = nullptr;
            const unsigned long parsed = std::strtoul(env, &end, 10);
            if (end == env || *end != '\0' || parsed == 0)
            {
                throw std::runtime_error("MHA_TEST_HEADS must be a positive integer");
            }
            return static_cast<uint64_t>(parsed);
        }
        return DEFAULT_H;
    }

    uint64_t GetTestSeqLen()
    {
        if (const char *env = std::getenv("MHA_TEST_SEQLEN"); env != nullptr && env[0] != '\0')
        {
            char *end = nullptr;
            const unsigned long parsed = std::strtoul(env, &end, 10);
            if (end == env || *end != '\0' || parsed == 0)
            {
                throw std::runtime_error("MHA_TEST_SEQLEN must be a positive integer");
            }
            return static_cast<uint64_t>(parsed);
        }
        return DEFAULT_T;
    }

    std::string ReferenceFileName(const pi::tensorlib::DataType dtype, const uint64_t heads, const uint64_t seq_len)
    {
        std::string file{"reference_"};
        file.append(test_utils::GetDtypeSuffix(dtype));
        if (heads != DEFAULT_H)
        {
            file.append("_h");
            file.append(std::to_string(heads));
        }
        if (seq_len != DEFAULT_T)
        {
            file.append("_t");
            file.append(std::to_string(seq_len));
        }
        file.append(".safetensors");
        return file;
    }

    bool PlanHasKernelSubstring(const pi::tensorlib::ExecutionPlan &plan, const std::string &needle)
    {
        for (const auto &entry : plan.entries)
        {
            if (!entry.kernel_descriptor)
            {
                continue;
            }
            const auto &kernel_name = entry.kernel_descriptor->kernel_name;
            const auto &function_name = entry.kernel_descriptor->function_name;
            if (kernel_name.find(needle) != std::string::npos || function_name.find(needle) != std::string::npos)
            {
                return true;
            }
        }
        return false;
    }

    bool PlanUsesTritonMhaKernels(const pi::tensorlib::ExecutionPlan &plan, const pi::tensorlib::DataType dtype)
    {
        const auto dtype_suffix = test_utils::GetDtypeSuffix(dtype);
        std::string fwd_kernel{"mha_full_attn_fwd_hs128_"};
        fwd_kernel.append(dtype_suffix);
        std::string bwd_kernel{"mha_full_attn_bwd_hs128_"};
        bwd_kernel.append(dtype_suffix);
        return PlanHasKernelSubstring(plan, fwd_kernel) || PlanHasKernelSubstring(plan, bwd_kernel);
    }

    bool PlanUsesCutlassMhaKernels(const pi::tensorlib::ExecutionPlan &plan, const pi::tensorlib::DataType dtype)
    {
        const auto dtype_suffix = test_utils::GetDtypeSuffix(dtype);
        std::string fwd_kernel{"mha_full_attn_cutlass_"};
        fwd_kernel.append(dtype_suffix);
        std::string bwd_kernel{"mha_full_attn_bwd_cutlass_"};
        bwd_kernel.append(dtype_suffix);
        return PlanHasKernelSubstring(plan, fwd_kernel) || PlanHasKernelSubstring(plan, bwd_kernel);
    }

    void DumpPlanKernels(const pi::tensorlib::ExecutionPlan &plan)
    {
        if (!IsEnvFlagEnabled("MHA_DEBUG_PLAN_KERNELS"))
        {
            return;
        }
        size_t index = 0;
        for (const auto &entry : plan.entries)
        {
            if (!entry.kernel_descriptor)
            {
                ++index;
                continue;
            }
            std::cerr << "[MHA_BWD] plan kernel " << index << " name=" << entry.kernel_descriptor->kernel_name
                      << " func=" << entry.kernel_descriptor->function_name << '\n';
            ++index;
        }
    }

    void ExpectCutlassMhaKernels(const pi::tensorlib::ExecutionPlan &plan, const pi::tensorlib::DataType dtype)
    {
#if !PI_TENSORLIB_ENABLE_CUDA
        (void)plan;
        (void)dtype;
        return;
#endif
        const bool expect_cutlass = IsEnvFlagEnabled("MHA_EXPECT_CUTLASS");
        const bool expect_bwd_only = IsEnvFlagEnabled("MHA_EXPECT_CUTLASS_BWD_ONLY");
        if (!expect_cutlass && !expect_bwd_only)
        {
            return;
        }
        const auto dtype_suffix = test_utils::GetDtypeSuffix(dtype);
        std::string fwd_kernel{"cutlass_mha_full_attn_"};
        fwd_kernel.append(dtype_suffix);
        fwd_kernel.append("_lse_kernel");
        std::string bwd_kernel{"cutlass_mha_full_attn_bwd_"};
        bwd_kernel.append(dtype_suffix);
        bwd_kernel.append("_kernel");

        const bool has_fwd = PlanHasKernelSubstring(plan, fwd_kernel);
        const bool has_bwd = PlanHasKernelSubstring(plan, bwd_kernel);
        if (!has_fwd || !has_bwd)
        {
            if (expect_bwd_only && !has_bwd)
            {
                throw std::runtime_error("Expected CUTLASS MHA bwd kernel was not selected");
            }
            if (!expect_bwd_only)
            {
                throw std::runtime_error("Expected CUTLASS MHA fwd/bwd kernels were not selected");
            }
        }
    }

    void ExpectFlashMhaKernels(const pi::tensorlib::ExecutionPlan &plan, const pi::tensorlib::DataType dtype)
    {
        if (!IsEnvFlagEnabled("MHA_EXPECT_FLASH"))
        {
            return;
        }
#if PI_TENSORLIB_ENABLE_CUDA
        const auto dtype_suffix = test_utils::GetDtypeSuffix(dtype);
        std::string fwd_kernel{"mha_full_attn_fa_fwd_hs128_"};
        fwd_kernel.append(dtype_suffix);
        std::string bwd_kernel{"mha_full_attn_fa_bwd_hs128_"};
        bwd_kernel.append(dtype_suffix);
        std::string dot_kernel{"mha_full_attn_fa_bwd_dot_do_o_hs128_"};
        dot_kernel.append(dtype_suffix);
        std::string convert_kernel{"mha_full_attn_fa_bwd_convert_dq_hs128_"};
        convert_kernel.append(dtype_suffix);

        if (!PlanHasKernelSubstring(plan, fwd_kernel) || !PlanHasKernelSubstring(plan, bwd_kernel) ||
            !PlanHasKernelSubstring(plan, dot_kernel) || !PlanHasKernelSubstring(plan, convert_kernel))
        {
            throw std::runtime_error("Expected FlashAttention MHA fwd/bwd kernels were not selected");
        }
#elif PI_TENSORLIB_ENABLE_HIP
        const auto dtype_suffix = test_utils::GetDtypeSuffix(dtype);
        std::string fwd_kernel{"mha_full_attn_fwd_hs128_"};
        fwd_kernel.append(dtype_suffix);
        std::string bwd_kernel{"mha_full_attn_bwd_hs128_"};
        bwd_kernel.append(dtype_suffix);
        if (!PlanHasKernelSubstring(plan, fwd_kernel) || !PlanHasKernelSubstring(plan, bwd_kernel))
        {
            throw std::runtime_error("Expected Triton MHA fwd/bwd kernels were not selected for Flash preference on HIP");
        }
#else
        (void)plan;
        (void)dtype;
        throw std::runtime_error("MHA_EXPECT_FLASH set but FlashAttention kernels are unavailable in this build");
#endif
    }

    void ExpectTritonMhaKernels(const pi::tensorlib::ExecutionPlan &plan, const pi::tensorlib::DataType dtype)
    {
        if (!IsEnvFlagEnabled("MHA_EXPECT_TRITON"))
        {
            return;
        }
        const auto dtype_suffix = test_utils::GetDtypeSuffix(dtype);
        std::string fwd_kernel{"mha_full_attn_fwd_hs128_"};
        fwd_kernel.append(dtype_suffix);
        std::string bwd_kernel{"mha_full_attn_bwd_hs128_"};
        bwd_kernel.append(dtype_suffix);

        if (!PlanHasKernelSubstring(plan, fwd_kernel) || !PlanHasKernelSubstring(plan, bwd_kernel))
        {
            throw std::runtime_error("Expected Triton MHA fwd/bwd kernels were not selected");
        }
    }

    void ApplyDefaultPasses(pi::tensorlib::ExecutionPlan &plan)
    {
        std::vector<std::unique_ptr<pi::tensorlib::passes::CompilerPass>> passes{};
        passes.emplace_back(std::make_unique<MhaAttentionImplPass>());
        passes.emplace_back(std::make_unique<MatmulFusePass>());
        passes.emplace_back(std::make_unique<MatmulImplPass>());
        passes.emplace_back(std::make_unique<TrailingBroadcastMulImplPass>());
        passes.emplace_back(std::make_unique<TrailingBroadcastAddImplPass>());
        passes.emplace_back(std::make_unique<ReduceSumImplPass>());
        passes.emplace_back(std::make_unique<ContiguousImplPass>());
        passes.emplace_back(std::make_unique<CastImplPass>());
        passes.emplace_back(std::make_unique<FillConstantImplPass>());
        passes.emplace_back(std::make_unique<FillZerosImplPass>());
        pi::tensorlib::passes::Transform(plan, passes);
    }

    template <typename MapT>
    std::shared_ptr<pi::tensorlib::RealTensor> FetchTensor(const MapT &ref, const std::string &name)
    {
        auto it = ref.find(name);
        if (it == ref.end())
        {
            throw std::runtime_error("Missing tensor in reference file: " + name);
        }
        return it->second;
    }

    void RunMhaAttentionBackwardCase(const pi::tensorlib::DataType dtype)
    {
        using namespace pi::tensorlib;

        ExecutionBackend &backend = ExecutionBackend::getInstance();
        auto &allocator_registry = allocator::DefaultAllocatorRegistry::instance();
        const auto main_stream_desc = GpuStreamDescriptors::Main;

        const uint64_t heads = GetTestHeads();
        const uint64_t seq_len = GetTestSeqLen();
        const uint64_t embed = heads * HS;

        const std::string reference_path = ReferenceFileName(dtype, heads, seq_len);
        const auto ref = safetensors::Load(reference_path);

        const auto x_host = FetchTensor(ref, "x");
        const auto w_qkv_host = FetchTensor(ref, "w_qkv");
        const auto b_qkv_host = FetchTensor(ref, "b_qkv");
        const auto w_proj_host = FetchTensor(ref, "w_proj");
        const auto b_proj_host = FetchTensor(ref, "b_proj");
        const auto upstream_host = FetchTensor(ref, "upstream");
        const auto expected_dx = FetchTensor(ref, "grad_x");
        const auto expected_dw_qkv = FetchTensor(ref, "grad_w_qkv");
        const auto expected_db_qkv = FetchTensor(ref, "grad_b_qkv");
        const auto expected_dw_proj = FetchTensor(ref, "grad_w_proj");
        const auto expected_db_proj = FetchTensor(ref, "grad_b_proj");

        if (x_host->shape().ndims() != 3 || x_host->shape()[0] != B || x_host->shape()[1] != seq_len ||
            x_host->shape()[2] != embed)
        {
            throw std::runtime_error("Unexpected input shape in reference data");
        }
        if (upstream_host->shape() != x_host->shape())
        {
            throw std::runtime_error("Upstream gradient shape mismatch in reference data");
        }

        OpGraph init_graph{{}, {}};
        uint32_t seed = 0;
        FullMhaAttention mha{"attn", embed, heads, DEVICE_GPU, dtype, init_graph, seed, main_stream_desc};

        TraceTensor x = TraceTensor::Create(x_host->shape().dims(), dtype, DEVICE_GPU, main_stream_desc);
        TraceTensor upstream = TraceTensor::Create(upstream_host->shape().dims(), dtype, DEVICE_GPU, main_stream_desc);
        x.markRetained();
        upstream.markRetained();
        const auto params = mha.parameters();
        auto retained_w_qkv = params[0].tensor;
        auto retained_b_qkv = params[1].tensor;
        auto retained_w_proj = params[2].tensor;
        auto retained_b_proj = params[3].tensor;
        retained_w_qkv.markRetained();
        retained_b_qkv.markRetained();
        retained_w_proj.markRetained();
        retained_b_proj.markRetained();

        OpGraph graph({
                          {.name = "x", .tensor = x},
                          {.name = "upstream", .tensor = upstream},
                          {.name = params[0].name, .tensor = retained_w_qkv},
                          {.name = params[1].name, .tensor = retained_b_qkv},
                          {.name = params[2].name, .tensor = retained_w_proj},
                          {.name = params[3].name, .tensor = retained_b_proj},
                      },
                      {});

        (void)mha.buildForward(graph, {x}, /*save_input_for_backward=*/true);

        std::unordered_map<std::string, TraceTensor> parameter_grads{};
        std::unordered_map<std::string, TraceTensor> operand_grads{};

        TraceTensor w_qkv_grad =
            graph.createTensor(w_qkv_host->shape().dims(), dtype, DEVICE_GPU, main_stream_desc, false);
        TraceTensor b_qkv_grad =
            graph.createTensor(b_qkv_host->shape().dims(), dtype, DEVICE_GPU, main_stream_desc, false);
        TraceTensor w_proj_grad =
            graph.createTensor(w_proj_host->shape().dims(), dtype, DEVICE_GPU, main_stream_desc, false);
        TraceTensor b_proj_grad =
            graph.createTensor(b_proj_host->shape().dims(), dtype, DEVICE_GPU, main_stream_desc, false);
        TraceTensor dx_grad = graph.createTensor(x_host->shape().dims(), dtype, DEVICE_GPU, main_stream_desc, false);
        FillZeros(graph, w_qkv_grad, main_stream_desc);
        FillZeros(graph, b_qkv_grad, main_stream_desc);
        FillZeros(graph, w_proj_grad, main_stream_desc);
        FillZeros(graph, b_proj_grad, main_stream_desc);
        FillZeros(graph, dx_grad, main_stream_desc);

        parameter_grads.emplace(params[0].name, w_qkv_grad);
        parameter_grads.emplace(params[1].name, b_qkv_grad);
        parameter_grads.emplace(params[2].name, w_proj_grad);
        parameter_grads.emplace(params[3].name, b_proj_grad);
        operand_grads.emplace("input", dx_grad);

        mha.buildBackward(graph, upstream, parameter_grads, operand_grads);
        graph.finalize();

        const auto x_gpu = x_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
        const auto upstream_gpu = upstream_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
        const auto w_qkv_gpu = w_qkv_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
        const auto b_qkv_gpu = b_qkv_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
        const auto w_proj_gpu = w_proj_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
        const auto b_proj_gpu = b_proj_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);

        std::vector<GraphExecutionInputDescriptor> inputs{
            {.name = "x", .tensor = x_gpu},
            {.name = "upstream", .tensor = upstream_gpu},
            {.name = params[0].name, .tensor = w_qkv_gpu},
            {.name = params[1].name, .tensor = b_qkv_gpu},
            {.name = params[2].name, .tensor = w_proj_gpu},
            {.name = params[3].name, .tensor = b_proj_gpu},
        };

        ExecutionPlan plan = ExecutionPlan::FromGraph(graph, inputs, {});
        ApplyDefaultPasses(plan);
        DumpPlanKernels(plan);
        ExpectCutlassMhaKernels(plan, dtype);
        ExpectFlashMhaKernels(plan, dtype);
        ExpectTritonMhaKernels(plan, dtype);
        Executor executor{plan, backend, 0};
        executor.execute(allocator_registry);

        const auto actual_w_qkv = executor.getOutput(w_qkv_grad);
        const auto actual_b_qkv = executor.getOutput(b_qkv_grad);
        const auto actual_w_proj = executor.getOutput(w_proj_grad);
        const auto actual_b_proj = executor.getOutput(b_proj_grad);
        const auto actual_dx = executor.getOutput(dx_grad);
        if (!actual_w_qkv || !actual_b_qkv || !actual_w_proj || !actual_b_proj || !actual_dx)
        {
            throw std::runtime_error("Failed to retrieve MHA backward gradients");
        }

        const float tolerance = test_utils::SelectTolerance(dtype, 4e-2f, 7e-3f);
        auto max_abs_diff = [](const std::shared_ptr<RealTensor> &expected,
                               const std::shared_ptr<RealTensor> &actual) -> double
        {
            auto expected_storage = expected->storage();
            auto actual_storage = actual->storage();
            if (expected_storage->device().device_type != DeviceType::CPU)
            {
                expected_storage = expected_storage->toCPU();
            }
            if (actual_storage->device().device_type != DeviceType::CPU)
            {
                actual_storage = actual_storage->toCPU();
            }

            const void *expected_data =
                static_cast<uint8_t *>(expected_storage->dataptr()) +
                expected->storageOffset() * GetDataTypeSize(expected->dtype());
            const void *actual_data =
                static_cast<uint8_t *>(actual_storage->dataptr()) +
                actual->storageOffset() * GetDataTypeSize(actual->dtype());

            double max_diff = 0.0;
            for (size_t i = 0; i < expected->shape().numel(); ++i)
            {
                size_t lin = i;
                size_t off_expected = 0;
                size_t off_actual = 0;
                for (auto d = static_cast<int64_t>(expected->shape().ndims()); d-- > 0;)
                {
                    const size_t idx_d = lin % expected->shape()[d];
                    lin /= expected->shape()[d];
                    off_expected += idx_d * expected->strides()[d];
                    off_actual += idx_d * actual->strides()[d];
                }

                double expected_value{};
                double actual_value{};
                switch (expected->dtype())
                {
                    case DataType::BFLOAT16:
                    {
                        const auto *e = static_cast<const uint16_t *>(expected_data);
                        const auto *a = static_cast<const uint16_t *>(actual_data);
                        expected_value = static_cast<double>(utils::Fp32FromBf16(e[off_expected]));
                        actual_value = static_cast<double>(utils::Fp32FromBf16(a[off_actual]));
                        break;
                    }
                    case DataType::FLOAT16:
                    {
                        const auto *e = static_cast<const uint16_t *>(expected_data);
                        const auto *a = static_cast<const uint16_t *>(actual_data);
                        expected_value = static_cast<double>(utils::Fp32FromFp16(e[off_expected]));
                        actual_value = static_cast<double>(utils::Fp32FromFp16(a[off_actual]));
                        break;
                    }
                    case DataType::FLOAT32:
                    {
                        const auto *e = static_cast<const float *>(expected_data);
                        const auto *a = static_cast<const float *>(actual_data);
                        expected_value = static_cast<double>(e[off_expected]);
                        actual_value = static_cast<double>(a[off_actual]);
                        break;
                    }
                    default:
                        break;
                }
                max_diff = std::max(max_diff, std::abs(expected_value - actual_value));
            }
            return max_diff;
        };

        auto assert_with_label = [&](const char *label,
                                     const std::shared_ptr<RealTensor> &expected,
                                     const std::shared_ptr<RealTensor> &actual,
                                     const float tol)
        {
            try
            {
                testing::AssertSimilar(expected, actual, tol);
            }
            catch (const std::exception &)
            {
                const double diff = max_abs_diff(expected, actual);
                std::cerr << "[MHA_BWD] Mismatch in " << label << ", max abs diff=" << diff << '\n';
                throw;
            }
        };

        if (const char *env = std::getenv("FBAMTRAIN_DEBUG_MHA_BWD_DIFF"); env != nullptr)
        {
            std::cerr << "[MHA_BWD] max abs diff w_qkv=" << max_abs_diff(expected_dw_qkv, actual_w_qkv.value()) << '\n';
            std::cerr << "[MHA_BWD] max abs diff b_qkv=" << max_abs_diff(expected_db_qkv, actual_b_qkv.value()) << '\n';
            std::cerr << "[MHA_BWD] max abs diff w_proj=" << max_abs_diff(expected_dw_proj, actual_w_proj.value()) << '\n';
            std::cerr << "[MHA_BWD] max abs diff b_proj=" << max_abs_diff(expected_db_proj, actual_b_proj.value()) << '\n';
            std::cerr << "[MHA_BWD] max abs diff dx=" << max_abs_diff(expected_dx, actual_dx.value()) << '\n';
        }

        float main_tolerance = tolerance;
        float bias_tolerance = (dtype == DataType::BFLOAT16 && tolerance < 2e-2f) ? 2e-2f : tolerance;
        const bool uses_triton = PlanUsesTritonMhaKernels(plan, dtype);
        const bool uses_cutlass = PlanUsesCutlassMhaKernels(plan, dtype);
        if (dtype == DataType::FLOAT16 && (IsEnvFlagEnabled("MHA_EXPECT_TRITON") || uses_triton))
        {
            bias_tolerance = std::max(bias_tolerance, 3e-2f);
        }
        if (dtype == DataType::FLOAT16 && (IsEnvFlagEnabled("MHA_EXPECT_CUTLASS") || uses_cutlass))
        {
            main_tolerance = std::max(main_tolerance, 1.3e-2f);
            bias_tolerance = std::max(bias_tolerance, 3.5e-2f);
        }
        assert_with_label("w_qkv", expected_dw_qkv, actual_w_qkv.value(), main_tolerance);
        assert_with_label("b_qkv", expected_db_qkv, actual_b_qkv.value(), bias_tolerance);
        assert_with_label("w_proj", expected_dw_proj, actual_w_proj.value(), main_tolerance);
        assert_with_label("b_proj", expected_db_proj, actual_b_proj.value(), main_tolerance);
        assert_with_label("dx", expected_dx, actual_dx.value(), main_tolerance);
    }
} // namespace

int main()
{
    const auto dtype = test_utils::GetTestDtype();
    RunMhaAttentionBackwardCase(dtype);
    return 0;
}
