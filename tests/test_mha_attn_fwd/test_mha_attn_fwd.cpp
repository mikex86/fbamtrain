#include "testing.h"

#include <allocator.h>

#include <cstdlib>
#include <iostream>

#include <cmath>
#include <execution_backend.h>
#include <executor.h>
#include <linear.h>
#include <optional>
#include <string>
#include <tensorlib.h>

#include "../common/test_dtype_utils.h"
#include <passes.h>
#include <safe_tensors.h>

static void ApplyDefaultPasses(pi::tensorlib::ExecutionPlan &execution_plan)
{
    std::vector<std::unique_ptr<pi::tensorlib::passes::CompilerPass>> passes{};
    passes.emplace_back(std::make_unique<MhaAttentionImplPass>());
    passes.emplace_back(std::make_unique<FillZerosImplPass>());
    passes.emplace_back(std::make_unique<FillConstantImplPass>());
    passes.emplace_back(std::make_unique<FillUniformImplPass>());
    pi::tensorlib::passes::Transform(execution_plan, passes);
}

namespace
{
    constexpr uint64_t DEFAULT_B = 32;
    constexpr uint64_t DEFAULT_H = 8;
    constexpr uint64_t DEFAULT_T = 4096;
    constexpr uint64_t HS = 128;
    constexpr float TOLERANCE_BF16 = 1e-3f;
    constexpr float TOLERANCE_FP16 = 3e-3f;

    bool IsEnvFlagEnabled(const char *name)
    {
        const char *env = std::getenv(name);
        return env != nullptr && env[0] != '\0';
    }

    uint64_t GetEnvPositiveU64(const char *name, const uint64_t fallback)
    {
        if (const char *env = std::getenv(name); env != nullptr && env[0] != '\0')
        {
            char *end = nullptr;
            const unsigned long parsed = std::strtoul(env, &end, 10);
            if (end == env || *end != '\0' || parsed == 0)
            {
                throw std::runtime_error(std::string(name) + " must be a positive integer");
            }
            return static_cast<uint64_t>(parsed);
        }
        return fallback;
    }

    uint64_t GetTestBatch()
    {
        return GetEnvPositiveU64("MHA_TEST_BATCH", DEFAULT_B);
    }

    uint64_t GetTestHeads()
    {
        return GetEnvPositiveU64("MHA_TEST_HEADS", DEFAULT_H);
    }

    uint64_t GetTestSeqLen()
    {
        return GetEnvPositiveU64("MHA_TEST_SEQLEN", DEFAULT_T);
    }

    std::string ReferenceFileName(const pi::tensorlib::DataType dtype,
                                  const uint64_t batch,
                                  const uint64_t heads,
                                  const uint64_t seq_len)
    {
        std::string file{"reference_"};
        file.append(test_utils::GetDtypeSuffix(dtype));
        if (batch != DEFAULT_B)
        {
            file.append("_b");
            file.append(std::to_string(batch));
        }
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

    void ExpectCutlassMhaKernel(const pi::tensorlib::ExecutionPlan &plan, const pi::tensorlib::DataType dtype)
    {
#if !PI_TENSORLIB_ENABLE_CUDA
        (void)plan;
        (void)dtype;
        return;
#endif
        if (!IsEnvFlagEnabled("MHA_EXPECT_CUTLASS"))
        {
            return;
        }
        std::string fwd_kernel{"cutlass_mha_full_attn_"};
        fwd_kernel.append(test_utils::GetDtypeSuffix(dtype));
        fwd_kernel.append("_lse_kernel");
        if (!PlanHasKernelSubstring(plan, fwd_kernel))
        {
            throw std::runtime_error("Expected CUTLASS MHA fwd kernel was not selected");
        }
    }

    void ExpectFlashMhaKernel(const pi::tensorlib::ExecutionPlan &plan, const pi::tensorlib::DataType dtype)
    {
        if (!IsEnvFlagEnabled("MHA_EXPECT_FLASH"))
        {
            return;
        }
#if PI_TENSORLIB_ENABLE_CUDA
        std::string fwd_kernel{"mha_full_attn_fa_fwd_hs128_"};
        fwd_kernel.append(test_utils::GetDtypeSuffix(dtype));
        if (!PlanHasKernelSubstring(plan, fwd_kernel))
        {
            throw std::runtime_error("Expected FlashAttention MHA fwd kernel was not selected");
        }
#elif PI_TENSORLIB_ENABLE_HIP
        std::string fwd_kernel{"mha_full_attn_fwd_hs128_"};
        fwd_kernel.append(test_utils::GetDtypeSuffix(dtype));
        if (!PlanHasKernelSubstring(plan, fwd_kernel))
        {
            throw std::runtime_error("Expected Triton MHA fwd kernel was not selected for Flash preference on HIP");
        }
#else
        (void)plan;
        (void)dtype;
        throw std::runtime_error("MHA_EXPECT_FLASH set but FlashAttention kernels are unavailable in this build");
#endif
    }

    void ExpectTritonMhaKernel(const pi::tensorlib::ExecutionPlan &plan, const pi::tensorlib::DataType dtype)
    {
        if (!IsEnvFlagEnabled("MHA_EXPECT_TRITON"))
        {
            return;
        }
        std::string fwd_kernel{"mha_full_attn_fwd_hs128_"};
        fwd_kernel.append(test_utils::GetDtypeSuffix(dtype));
        if (!PlanHasKernelSubstring(plan, fwd_kernel))
        {
            throw std::runtime_error("Expected Triton MHA fwd kernel was not selected");
        }
    }

    void RunMhaAttentionTest(const pi::tensorlib::DataType dtype)
    {
        pi::tensorlib::ExecutionBackend &execution_backend = pi::tensorlib::ExecutionBackend::getInstance();
        auto &allocator_registry = pi::tensorlib::allocator::DefaultAllocatorRegistry::instance();

        const uint64_t batch = GetTestBatch();
        const uint64_t heads = GetTestHeads();
        const uint64_t seq_len = GetTestSeqLen();

        const pi::tensorlib::Device device{
            .device_type = pi::tensorlib::DeviceType::GPU,
            .ordinal = 0,
        };
        const auto main_stream_desc = pi::tensorlib::GpuStreamDescriptors::Main;
        pi::tensorlib::OpGraph init_graph{{}, {}};

        pi::tensorlib::TraceTensor q =
            init_graph.createTensor({batch, seq_len, heads, HS}, dtype, device, main_stream_desc, false);
        pi::tensorlib::TraceTensor k =
            init_graph.createTensor({batch, seq_len, heads, HS}, dtype, device, main_stream_desc, false);
        pi::tensorlib::TraceTensor v =
            init_graph.createTensor({batch, seq_len, heads, HS}, dtype, device, main_stream_desc, false);

        uint32_t rng_seed = 42;

        pi::tensorlib::FillUniform(init_graph, q, -0.5f, 0.5f, rng_seed++, main_stream_desc);
        pi::tensorlib::FillUniform(init_graph, k, -0.5f, 0.5f, rng_seed++, main_stream_desc);
        pi::tensorlib::FillUniform(init_graph, v, -0.5f, 0.5f, rng_seed++, main_stream_desc);

        init_graph.finalize();

        pi::tensorlib::ExecutionPlan init_plan = pi::tensorlib::ExecutionPlan::FromGraph(init_graph, {}, {});
        ApplyDefaultPasses(init_plan);

        pi::tensorlib::Executor init_executor{init_plan, execution_backend, 0};
        init_executor.execute(allocator_registry);

        q.markRetained();
        k.markRetained();
        v.markRetained();
        pi::tensorlib::OpGraph graph{{
                                         pi::tensorlib::GraphInputDescriptor{.name = "q", .tensor = q},
                                         pi::tensorlib::GraphInputDescriptor{.name = "k", .tensor = k},
                                         pi::tensorlib::GraphInputDescriptor{.name = "v", .tensor = v},
                                     },
                                     {}};

        pi::tensorlib::TraceTensor output =
            pi::tensorlib::ScaledDotProductAttentionFwd(
                graph, q, k, v, static_cast<float>(1.0 / std::sqrt(HS)), false,
                /*use_fp16_flash_attn_acc=*/false,
                IsEnvFlagEnabled("MHA_TEST_USE_SCRATCH")
                    ? std::optional<pi::tensorlib::TraceTensor>(
                          graph.createTensor({batch, heads, seq_len}, pi::tensorlib::DataType::FLOAT32, device,
                                             main_stream_desc, false))
                    : std::nullopt,
                main_stream_desc);
        graph.finalize();

        std::shared_ptr<pi::tensorlib::RealTensor> q_real = *init_executor.getOutput(q);
        std::shared_ptr<pi::tensorlib::RealTensor> k_real = *init_executor.getOutput(k);
        std::shared_ptr<pi::tensorlib::RealTensor> v_real = *init_executor.getOutput(v);
        pi::tensorlib::ExecutionPlan plan = pi::tensorlib::ExecutionPlan::FromGraph(graph,
                                                                                    {
                                                                                        {.name = "q", .tensor = q_real},
                                                                                        {.name = "k", .tensor = k_real},
                                                                                        {.name = "v", .tensor = v_real},
                                                                                    },
                                                                                    {});
        ApplyDefaultPasses(plan);
        ExpectCutlassMhaKernel(plan, dtype);
        ExpectFlashMhaKernel(plan, dtype);
        ExpectTritonMhaKernel(plan, dtype);
        pi::tensorlib::Executor executor{plan, execution_backend, 0};
        executor.execute(allocator_registry);

        std::shared_ptr<pi::tensorlib::RealTensor> actual_output = *executor.getOutput(output);

        const std::string reference_path = ReferenceFileName(dtype, batch, heads, seq_len);

        const auto expected_tensors = pi::tensorlib::safetensors::Load(reference_path);
        const auto expected_output_it = expected_tensors.find("output");
        if (expected_output_it == expected_tensors.end())
        {
            throw std::runtime_error("Expected output tensor not found in reference file: " + reference_path);
        }
        const auto &expected_output = expected_output_it->second;

        const float tolerance = test_utils::SelectTolerance(dtype, TOLERANCE_BF16, TOLERANCE_FP16);
        pi::tensorlib::testing::AssertSimilar(expected_output, actual_output, tolerance);
    }
} // namespace

int main()
{
    const auto dtype = test_utils::GetTestDtype();
    RunMhaAttentionTest(dtype);
}
