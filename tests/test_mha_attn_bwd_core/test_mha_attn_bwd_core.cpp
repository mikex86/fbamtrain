#include "testing.h"

#include "../common/test_dtype_utils.h"
#include <allocator.h>
#include <execution_backend.h>
#include <executor.h>
#include <functional.h>
#include <passes.h>
#include <safe_tensors.h>
#include <tensorlib.h>

#include <algorithm>
#include <memory>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <cstring>

namespace
{
    constexpr pi::tensorlib::Device DEVICE_GPU{pi::tensorlib::DeviceType::GPU, 0};

    constexpr uint64_t B = 2;
    constexpr uint64_t H = 2;
    constexpr uint64_t DEFAULT_T = 128;
    constexpr uint64_t HS = 128;

    bool IsEnvFlagEnabled(const char *name)
    {
        const char *env = std::getenv(name);
        return env != nullptr && env[0] != '\0';
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
            return parsed;
        }
        return DEFAULT_T;
    }

    std::string ReferenceFileName(const pi::tensorlib::DataType dtype, const uint64_t seq_len)
    {
        std::string file{"reference_"};
        file.append(test_utils::GetDtypeSuffix(dtype));
        if (seq_len != DEFAULT_T)
        {
            file.append("_t");
            file.append(std::to_string(seq_len));
        }
        file.append(".safetensors");
        return file;
    }

    void ApplyDefaultPasses(pi::tensorlib::ExecutionPlan &plan)
    {
        std::vector<std::unique_ptr<pi::tensorlib::passes::CompilerPass>> passes{};
        passes.emplace_back(std::make_unique<MhaAttentionImplPass>());
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

    bool PlanHasKernelSubstring(const pi::tensorlib::ExecutionPlan &plan, const std::string &needle)
    {
        for (const auto &entry : plan.entries)
        {
            if (!entry.kernel_descriptor)
            {
                continue;
            }
            const auto &kernel_name = entry.kernel_descriptor->kernel_name;
            if (kernel_name.find(needle) != std::string::npos)
            {
                return true;
            }
        }
        return false;
    }

    bool PlanUsesTritonMhaKernel(const pi::tensorlib::ExecutionPlan &plan, const pi::tensorlib::DataType dtype)
    {
        std::string fwd_kernel{"mha_full_attn_fwd_hs128_"};
        fwd_kernel.append(test_utils::GetDtypeSuffix(dtype));
        std::string bwd_kernel{"mha_full_attn_bwd_hs128_"};
        bwd_kernel.append(test_utils::GetDtypeSuffix(dtype));
        return PlanHasKernelSubstring(plan, fwd_kernel) || PlanHasKernelSubstring(plan, bwd_kernel);
    }

    void ExpectFlashFwdKernel(const pi::tensorlib::ExecutionPlan &plan, const pi::tensorlib::DataType dtype)
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
        if (IsEnvFlagEnabled("MHA_TEST_FP16_ACC"))
        {
            if (!PlanHasKernelSubstring(plan, "mha_full_attn_fa_fwd_hs128_fp16_acc_fp16"))
            {
                throw std::runtime_error("Expected FlashAttention FP16-accumulation kernel was not selected");
            }
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

    void ExpectFlashBwdKernels(const pi::tensorlib::ExecutionPlan &plan, const pi::tensorlib::DataType dtype)
    {
        if (!IsEnvFlagEnabled("MHA_EXPECT_FLASH"))
        {
            return;
        }
#if PI_TENSORLIB_ENABLE_CUDA
        const auto dtype_suffix = test_utils::GetDtypeSuffix(dtype);
        std::string bwd_kernel{"mha_full_attn_fa_bwd_hs128_"};
        bwd_kernel.append(dtype_suffix);
        std::string dot_kernel{"mha_full_attn_fa_bwd_dot_do_o_hs128_"};
        dot_kernel.append(dtype_suffix);
        std::string convert_kernel{"mha_full_attn_fa_bwd_convert_dq_hs128_"};
        convert_kernel.append(dtype_suffix);
        if (!PlanHasKernelSubstring(plan, bwd_kernel) || !PlanHasKernelSubstring(plan, dot_kernel) ||
            !PlanHasKernelSubstring(plan, convert_kernel))
        {
            throw std::runtime_error("Expected FlashAttention MHA bwd kernels were not selected");
        }
#elif PI_TENSORLIB_ENABLE_HIP
        const auto dtype_suffix = test_utils::GetDtypeSuffix(dtype);
        std::string bwd_kernel{"mha_full_attn_bwd_hs128_"};
        bwd_kernel.append(dtype_suffix);
        if (!PlanHasKernelSubstring(plan, bwd_kernel))
        {
            throw std::runtime_error("Expected Triton MHA bwd kernel was not selected for Flash preference on HIP");
        }
#else
        (void)plan;
        (void)dtype;
        throw std::runtime_error("MHA_EXPECT_FLASH set but FlashAttention kernels are unavailable in this build");
#endif
    }

    void ExpectTritonFwdKernel(const pi::tensorlib::ExecutionPlan &plan, const pi::tensorlib::DataType dtype)
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

    void ExpectTritonBwdKernel(const pi::tensorlib::ExecutionPlan &plan, const pi::tensorlib::DataType dtype)
    {
        if (!IsEnvFlagEnabled("MHA_EXPECT_TRITON"))
        {
            return;
        }
        std::string bwd_kernel{"mha_full_attn_bwd_hs128_"};
        bwd_kernel.append(test_utils::GetDtypeSuffix(dtype));
        if (!PlanHasKernelSubstring(plan, bwd_kernel))
        {
            throw std::runtime_error("Expected Triton MHA bwd kernel was not selected");
        }
    }

    void ExpectNonEvenFlashKernel(const pi::tensorlib::ExecutionPlan &plan)
    {
        if (!IsEnvFlagEnabled("MHA_EXPECT_NON_EVEN_KERNEL"))
        {
            return;
        }
#if PI_TENSORLIB_ENABLE_CUDA
        for (const auto &entry : plan.entries)
        {
            if (!entry.kernel_descriptor)
            {
                continue;
            }
            const auto &kernel_name = entry.kernel_descriptor->kernel_name;
            if (kernel_name.find("mha_full_attn_fa_") != std::string::npos &&
                kernel_name.find("_even") != std::string::npos)
            {
                throw std::runtime_error("Expected FlashAttention non-even kernels, but even variant was selected");
            }
        }
#elif PI_TENSORLIB_ENABLE_HIP
        for (const auto &entry : plan.entries)
        {
            if (!entry.kernel_descriptor)
            {
                continue;
            }
            const auto &kernel_name = entry.kernel_descriptor->kernel_name;
            if (kernel_name.find("mha_full_attn_fwd_hs128_") != std::string::npos ||
                kernel_name.find("mha_full_attn_bwd_hs128_") != std::string::npos)
            {
                if (kernel_name.find("_uneven") == std::string::npos)
                {
                    throw std::runtime_error("Expected Triton non-even kernels, but even variant was selected");
                }
            }
        }
#else
        (void)plan;
        throw std::runtime_error("MHA_EXPECT_NON_EVEN_KERNEL set but FlashAttention kernels are unavailable in this build");
#endif
    }

    double MaxAbsFloat32(const std::shared_ptr<pi::tensorlib::RealTensor> &tensor)
    {
        using namespace pi::tensorlib;
        if (tensor->dtype() != DataType::FLOAT32)
        {
            throw std::runtime_error("MaxAbsFloat32 expects FLOAT32 tensor");
        }
        auto storage = tensor->storage();
        if (storage->device().device_type != DeviceType::CPU)
        {
            storage = storage->toCPU();
        }
        const auto *data = reinterpret_cast<const float *>(
            static_cast<const uint8_t *>(storage->dataptr()) +
            tensor->storageOffset() * GetDataTypeSize(tensor->dtype()));
        double max_abs = 0.0;
        const auto numel = tensor->shape().numel();
        for (size_t i = 0; i < numel; ++i)
        {
            max_abs = std::max(max_abs, std::abs(static_cast<double>(data[i])));
        }
        return max_abs;
    }

    std::shared_ptr<pi::tensorlib::RealTensor> ScaleScratch(const std::shared_ptr<pi::tensorlib::RealTensor> &src,
                                                            float scale)
    {
        using namespace pi::tensorlib;
        if (src->dtype() != DataType::FLOAT32)
        {
            throw std::runtime_error("ScaleScratch expects FLOAT32 tensor");
        }
        auto src_storage = src->storage();
        if (src_storage->device().device_type != DeviceType::CPU)
        {
            src_storage = src_storage->toCPU();
        }
        const auto &shape = src->shape();
        if (shape.ndims() != 3)
        {
            throw std::runtime_error("ScaleScratch expects a 3D scratch tensor");
        }
        auto dst = RealTensor::Allocate({shape[0], shape[1], shape[2]}, src->dtype(), Device{DeviceType::CPU, 0});
        auto dst_storage = dst->storage();

        const auto *src_base = reinterpret_cast<const float *>(
            static_cast<const uint8_t *>(src_storage->dataptr()) +
            src->storageOffset() * GetDataTypeSize(src->dtype()));
        auto *dst_base = reinterpret_cast<float *>(static_cast<uint8_t *>(dst_storage->dataptr()) +
                                                   dst->storageOffset() * GetDataTypeSize(dst->dtype()));

        const auto &src_strides = src->strides();
        const auto &dst_strides = dst->strides();

        const auto numel = shape.numel();
        for (size_t i = 0; i < numel; ++i)
        {
            size_t lin = i;
            size_t src_offset = 0;
            size_t dst_offset = 0;
            for (auto d = static_cast<int64_t>(shape.ndims()); d-- > 0;)
            {
                const size_t idx_d = lin % shape[d];
                lin /= shape[d];
                src_offset += idx_d * src_strides[d];
                dst_offset += idx_d * dst_strides[d];
            }
            dst_base[dst_offset] = src_base[src_offset] * scale;
        }

        return dst;
    }

    std::shared_ptr<pi::tensorlib::RealTensor>
    MaybeConvertScratchForFlash(const std::shared_ptr<pi::tensorlib::RealTensor> &scratch)
    {
#if !PI_TENSORLIB_ENABLE_CUDA
        return scratch;
#else
        if (!IsEnvFlagEnabled("MHA_EXPECT_FLASH"))
        {
            return scratch;
        }
        constexpr float kLn2 = 0.6931471805599453f;
        return ScaleScratch(scratch, kLn2);
#endif
    }

    void RunScaledDotProductAttentionBwdCase(const pi::tensorlib::DataType dtype)
    {
        using namespace pi::tensorlib;

        ExecutionBackend &backend = ExecutionBackend::getInstance();
        auto &allocator_registry = allocator::DefaultAllocatorRegistry::instance();
        const auto main_stream_desc = GpuStreamDescriptors::Main;

        const uint64_t t = GetTestSeqLen();
        const std::string reference_path = ReferenceFileName(dtype, t);
        const auto ref = safetensors::Load(reference_path);

        const auto q_host = FetchTensor(ref, "q");
        const auto k_host = FetchTensor(ref, "k");
        const auto v_host = FetchTensor(ref, "v");
        const auto output_host = FetchTensor(ref, "output");
        const auto scratch_host = FetchTensor(ref, "scratch");
        const auto upstream_host = FetchTensor(ref, "upstream");
        const auto expected_dq = FetchTensor(ref, "grad_q");
        const auto expected_dk = FetchTensor(ref, "grad_k");
        const auto expected_dv = FetchTensor(ref, "grad_v");

        if (q_host->shape().ndims() != 4 || q_host->shape()[0] != B || q_host->shape()[1] != t ||
            q_host->shape()[2] != H || q_host->shape()[3] != HS)
        {
            throw std::runtime_error("Unexpected q shape in reference data");
        }
        if (k_host->shape() != q_host->shape() || v_host->shape() != q_host->shape())
        {
            throw std::runtime_error("Reference k/v shapes do not match q");
        }
        if (output_host->shape() != q_host->shape())
        {
            throw std::runtime_error("Reference output shape does not match q");
        }
        if (upstream_host->shape() != q_host->shape())
        {
            throw std::runtime_error("Reference upstream shape does not match q");
        }
        if (scratch_host->dtype() != DataType::FLOAT32 || scratch_host->shape().ndims() != 3 ||
            scratch_host->shape()[0] != B || scratch_host->shape()[1] != H || scratch_host->shape()[2] != t)
        {
            throw std::runtime_error("Reference scratch must be FLOAT32 and shape (B,H,T)");
        }

        TraceTensor q = TraceTensor::Create(q_host->shape().dims(), dtype, DEVICE_GPU, main_stream_desc);
        TraceTensor k = TraceTensor::Create(k_host->shape().dims(), dtype, DEVICE_GPU, main_stream_desc);
        TraceTensor v = TraceTensor::Create(v_host->shape().dims(), dtype, DEVICE_GPU, main_stream_desc);
        TraceTensor output = TraceTensor::Create({B, t, H, HS}, dtype, DEVICE_GPU, main_stream_desc);
        TraceTensor scratch =
            TraceTensor::Create(scratch_host->shape().dims(), DataType::FLOAT32, DEVICE_GPU, main_stream_desc);
        TraceTensor upstream = TraceTensor::Create({B, t, H, HS}, dtype, DEVICE_GPU, main_stream_desc);
        q.markRetained();
        k.markRetained();
        v.markRetained();
        output.markRetained();
        scratch.markRetained();
        upstream.markRetained();

        OpGraph graph({
                          {.name = "q", .tensor = q},
                          {.name = "k", .tensor = k},
                          {.name = "v", .tensor = v},
                          {.name = "output", .tensor = output},
                          {.name = "scratch", .tensor = scratch},
                          {.name = "upstream", .tensor = upstream},
                      },
                      {});

        const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(HS));
        auto bwd_result =
            ScaledDotProductAttentionBwd(graph, q, k, v, output, scratch, upstream, softmax_scale, false,
                                         main_stream_desc);
        graph.finalize();

        const auto q_gpu = q_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
        const auto k_gpu = k_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
        const auto v_gpu = v_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
        const auto output_gpu = output_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
        const auto scratch_input = MaybeConvertScratchForFlash(scratch_host);
        const auto scratch_gpu = scratch_input->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
        const auto upstream_gpu = upstream_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);

        std::vector<GraphExecutionInputDescriptor> inputs{
            {.name = "q", .tensor = q_gpu},
            {.name = "k", .tensor = k_gpu},
            {.name = "v", .tensor = v_gpu},
            {.name = "output", .tensor = output_gpu},
            {.name = "scratch", .tensor = scratch_gpu},
            {.name = "upstream", .tensor = upstream_gpu},
        };

        ExecutionPlan plan = ExecutionPlan::FromGraph(graph, inputs, {});
        ApplyDefaultPasses(plan);
        ExpectFlashBwdKernels(plan, dtype);
        ExpectTritonBwdKernel(plan, dtype);
        ExpectNonEvenFlashKernel(plan);
        Executor executor{plan, backend, 0};
        executor.execute(allocator_registry);

        const auto actual_dq = executor.getOutput(bwd_result.grad_q);
        const auto actual_dk = executor.getOutput(bwd_result.grad_k);
        const auto actual_dv = executor.getOutput(bwd_result.grad_v);
        if (!actual_dq || !actual_dk || !actual_dv)
        {
            throw std::runtime_error("Failed to retrieve attention backward gradients");
        }

        float tolerance = test_utils::SelectTolerance(dtype, 1e-2f, 7e-3f);
        const bool uses_triton = PlanUsesTritonMhaKernel(plan, dtype);
        if (IsEnvFlagEnabled("MHA_EXPECT_TRITON") || uses_triton)
        {
            tolerance = std::max(tolerance, 3e-2f);
        }
        if ((IsEnvFlagEnabled("MHA_EXPECT_TRITON") || uses_triton) && t != DEFAULT_T)
        {
            tolerance = std::max(tolerance, 4e-2f);
        }
#if PI_TENSORLIB_ENABLE_CUDA
        if (dtype == DataType::FLOAT16 && IsEnvFlagEnabled("MHA_EXPECT_FLASH") &&
            IsEnvFlagEnabled("MHA_EXPECT_NON_EVEN_KERNEL"))
        {
            tolerance = 1e-2f;
        }
#endif
        testing::AssertSimilar(expected_dq, actual_dq.value(), tolerance);
        testing::AssertSimilar(expected_dk, actual_dk.value(), tolerance);
        testing::AssertSimilar(expected_dv, actual_dv.value(), tolerance);
    }

    void RunScaledDotProductAttentionFwdCase(const pi::tensorlib::DataType dtype)
    {
        using namespace pi::tensorlib;

        ExecutionBackend &backend = ExecutionBackend::getInstance();
        auto &allocator_registry = allocator::DefaultAllocatorRegistry::instance();
        const auto main_stream_desc = GpuStreamDescriptors::Main;

        const uint64_t t = GetTestSeqLen();
        const std::string reference_path = ReferenceFileName(dtype, t);
        const auto ref = safetensors::Load(reference_path);

        const auto q_host = FetchTensor(ref, "q");
        const auto k_host = FetchTensor(ref, "k");
        const auto v_host = FetchTensor(ref, "v");
        const auto output_host = FetchTensor(ref, "output");
        const auto scratch_host = FetchTensor(ref, "scratch");
        if (scratch_host->dtype() != DataType::FLOAT32 || scratch_host->shape().ndims() != 3 ||
            scratch_host->shape()[0] != B || scratch_host->shape()[1] != H || scratch_host->shape()[2] != t)
        {
            throw std::runtime_error("Reference scratch must be FLOAT32 and shape (B,H,T)");
        }

        TraceTensor q = TraceTensor::Create(q_host->shape().dims(), dtype, DEVICE_GPU, main_stream_desc);
        TraceTensor k = TraceTensor::Create(k_host->shape().dims(), dtype, DEVICE_GPU, main_stream_desc);
        TraceTensor v = TraceTensor::Create(v_host->shape().dims(), dtype, DEVICE_GPU, main_stream_desc);
        TraceTensor scratch =
            TraceTensor::Create(scratch_host->shape().dims(), DataType::FLOAT32, DEVICE_GPU, main_stream_desc);
        q.markRetained();
        k.markRetained();
        v.markRetained();

        OpGraph graph({
                          {.name = "q", .tensor = q},
                          {.name = "k", .tensor = k},
                          {.name = "v", .tensor = v},
                      },
                      {});

        const float softmax_scale = 1.0f / std::sqrt(static_cast<float>(HS));
        const bool use_fp16_acc = IsEnvFlagEnabled("MHA_TEST_FP16_ACC");
        TraceTensor output = ScaledDotProductAttentionFwd(graph, q, k, v, softmax_scale, false, use_fp16_acc, scratch,
                                                          main_stream_desc);
        graph.finalize();

        const auto q_gpu = q_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
        const auto k_gpu = k_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
        const auto v_gpu = v_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);

        std::vector<GraphExecutionInputDescriptor> inputs{
            {.name = "q", .tensor = q_gpu},
            {.name = "k", .tensor = k_gpu},
            {.name = "v", .tensor = v_gpu},
        };

        ExecutionPlan plan = ExecutionPlan::FromGraph(graph, inputs, {});
#if PI_TENSORLIB_ENABLE_HIP && !PI_TENSORLIB_ENABLE_CUDA
        if (use_fp16_acc)
        {
            try
            {
                ApplyDefaultPasses(plan);
            }
            catch (const std::runtime_error &err)
            {
                const std::string message = err.what();
                if (message.find("FP16 accumulation is not supported on AMD") != std::string::npos)
                {
                    return;
                }
                throw;
            }
            throw std::runtime_error("Expected FP16 accumulation request to fail on AMD");
        }
#endif
        ApplyDefaultPasses(plan);
        ExpectFlashFwdKernel(plan, dtype);
        ExpectTritonFwdKernel(plan, dtype);
        ExpectNonEvenFlashKernel(plan);
        Executor executor{plan, backend, 0};
        executor.execute(allocator_registry);

        const auto actual_output = executor.getOutput(output);
        const auto actual_scratch = executor.getOutput(scratch);
        if (!actual_output || !actual_scratch)
        {
            throw std::runtime_error("Failed to retrieve attention forward outputs");
        }

        float tolerance = test_utils::SelectTolerance(dtype, 1e-2f, 7e-3f);
        const bool uses_triton = PlanUsesTritonMhaKernel(plan, dtype);
        if (IsEnvFlagEnabled("MHA_EXPECT_TRITON") || uses_triton)
        {
            tolerance = std::max(tolerance, 3e-2f);
        }
        if ((IsEnvFlagEnabled("MHA_EXPECT_TRITON") || uses_triton) && t != DEFAULT_T)
        {
            tolerance = std::max(tolerance, 4e-2f);
        }
        testing::AssertSimilar(output_host, actual_output.value(), tolerance);
        const auto expected_scratch = MaybeConvertScratchForFlash(scratch_host);
        float scratch_tolerance = 1e-2f;
        if (IsEnvFlagEnabled("MHA_EXPECT_TRITON") || uses_triton)
        {
            scratch_tolerance = std::max(scratch_tolerance, 3e-2f);
        }
        if ((IsEnvFlagEnabled("MHA_EXPECT_TRITON") || uses_triton) && t != DEFAULT_T)
        {
            scratch_tolerance = std::max(scratch_tolerance, 4e-2f);
        }
        testing::AssertSimilar(expected_scratch, *actual_scratch, scratch_tolerance);
        const double scratch_max = MaxAbsFloat32(*actual_scratch);
        if (scratch_max <= 1e-4)
        {
            throw std::runtime_error("Attention forward scratch appears uninitialized (max abs ~ 0)");
        }
    }
} // namespace

int main()
{
    const auto dtype = test_utils::GetTestDtype();
    RunScaledDotProductAttentionFwdCase(dtype);
    RunScaledDotProductAttentionBwdCase(dtype);
    return 0;
}
