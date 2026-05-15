#include "testing.h"

#include <allocator.h>
#include <execution_backend.h>
#include <executor.h>
#include <functional.h>
#include <passes.h>
#include <tensorlib.h>
#include <utils.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace pi::tensorlib;

namespace
{
    constexpr Device DEVICE_CPU{DeviceType::CPU, 0};
    constexpr Device DEVICE_GPU{DeviceType::GPU, 0};

    struct MatmulPermutation
    {
        DataType output_dtype{};
        bool use_fp16_accumulation{};
        bool accumulate_output{};
        bool transpose_a{};
        bool transpose_b{};
        bool aligned{};
    };

    struct FusedPermutation
    {
        DataType output_dtype{};
        bool use_fp16_accumulation{};
        bool accumulate_output{};
        bool aligned{};
        bool write_out_preact{};
    };

    struct MatmulDims
    {
        uint64_t m{};
        uint64_t n{};
        uint64_t k{};
    };

    DataType InputDtypeForOutput(const DataType output_dtype)
    {
        return (output_dtype == DataType::BFLOAT16) ? DataType::BFLOAT16 : DataType::FLOAT16;
    }

    MatmulDims GetDims(const bool aligned)
    {
        if (aligned)
        {
            return MatmulDims{128, 128, 128};
        }
        return MatmulDims{130, 122, 126};
    }

    std::string TransposeSuffix(const bool transpose_a, const bool transpose_b)
    {
        if (transpose_a && transpose_b)
        {
            return "_tab";
        }
        if (transpose_a)
        {
            return "_ta";
        }
        if (transpose_b)
        {
            return "_tb";
        }
        return "";
    }

    std::string MakeTritonKernelName(const MatmulPermutation &cfg)
    {
        std::string name;
        if (cfg.output_dtype == DataType::FLOAT32)
        {
            name = "matmul_fp16_out_fp32";
            if (!cfg.aligned)
            {
                name += "_unaligned";
            }
        }
        else if (cfg.output_dtype == DataType::BFLOAT16)
        {
            name = cfg.aligned ? "matmul_bf16" : "matmul_bf16_unaligned";
        }
        else
        {
            if (cfg.aligned)
            {
                name = cfg.use_fp16_accumulation ? "matmul_fp16_acc_fp16" : "matmul_fp16";
            }
            else
            {
                name = cfg.use_fp16_accumulation ? "matmul_fp16_acc_fp16_unaligned" : "matmul_fp16_unaligned";
            }
        }
        name += TransposeSuffix(cfg.transpose_a, cfg.transpose_b);
        name += cfg.accumulate_output ? "_cacc" : "_cstore";
        return name;
    }

    std::string MakeCutlassKernelName(const MatmulPermutation &cfg)
    {
        std::string name;
        if (cfg.output_dtype == DataType::FLOAT32)
        {
            name = "cutlass_matmul_fp16_out_fp32";
        }
        else if (cfg.output_dtype == DataType::BFLOAT16)
        {
            name = "cutlass_matmul_bf16";
        }
        else
        {
            name = cfg.use_fp16_accumulation ? "cutlass_matmul_fp16_acc_fp16" : "cutlass_matmul_fp16";
        }
        name += TransposeSuffix(cfg.transpose_a, cfg.transpose_b);
        name += "_kernel";
        return name;
    }

    std::string MakeTritonAddmmGeluKernelName(const FusedPermutation &cfg)
    {
        std::string name = cfg.write_out_preact ? "addmm_gelu_preact" : "addmm_gelu";
        if (cfg.output_dtype == DataType::BFLOAT16)
        {
            name += "_bf16";
        }
        else
        {
            if (cfg.use_fp16_accumulation)
            {
                name += "_fp16_acc_fp16";
            }
            else
            {
                name += "_fp16";
            }
        }
        if (!cfg.aligned)
        {
            name += "_unaligned";
        }
        name += cfg.accumulate_output ? "_cacc" : "_cstore";
        return name;
    }

    std::string MakeCutlassAddmmGeluKernelName(const FusedPermutation &cfg)
    {
        std::string name = "cutlass_addmm_gelu";
        if (cfg.write_out_preact)
        {
            name += "_preact";
        }
        if (cfg.output_dtype == DataType::BFLOAT16)
        {
            name += "_bf16";
        }
        else
        {
            if (cfg.use_fp16_accumulation)
            {
                name += "_fp16_acc_fp16";
            }
            else
            {
                name += "_fp16";
            }
        }
        name += "_kernel";
        return name;
    }

    std::shared_ptr<RealTensor> AllocateAndFillHost(const uint64_t rows, const uint64_t cols, const DataType dtype,
                                                     std::mt19937 &rng)
    {
        auto tensor = RealTensor::Allocate({rows, cols}, dtype, DEVICE_CPU);
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
        const uint64_t total = rows * cols;

        if (dtype == DataType::FLOAT32)
        {
            auto *dst = reinterpret_cast<float *>(tensor->dataptr());
            for (uint64_t i = 0; i < total; ++i)
            {
                dst[i] = dist(rng);
            }
            return tensor;
        }

        auto *dst = reinterpret_cast<uint16_t *>(tensor->dataptr());
        for (uint64_t i = 0; i < total; ++i)
        {
            const float val = dist(rng);
            dst[i] = (dtype == DataType::BFLOAT16) ? utils::Bf16FromFp32(val) : utils::Fp16FromFp32(val);
        }
        return tensor;
    }

    std::shared_ptr<RealTensor> AllocateAndFillVector(const uint64_t size, const DataType dtype, std::mt19937 &rng)
    {
        auto tensor = RealTensor::Allocate({size}, dtype, DEVICE_CPU);
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

        if (dtype == DataType::FLOAT32)
        {
            auto *dst = reinterpret_cast<float *>(tensor->dataptr());
            for (uint64_t i = 0; i < size; ++i)
            {
                dst[i] = dist(rng);
            }
            return tensor;
        }

        auto *dst = reinterpret_cast<uint16_t *>(tensor->dataptr());
        for (uint64_t i = 0; i < size; ++i)
        {
            const float val = dist(rng);
            dst[i] = (dtype == DataType::BFLOAT16) ? utils::Bf16FromFp32(val) : utils::Fp16FromFp32(val);
        }
        return tensor;
    }

    float ReadVectorValue(const std::shared_ptr<RealTensor> &tensor, const uint64_t idx)
    {
        const auto &shape = tensor->shape();
        const auto &strides = tensor->strides();
        if (shape.ndims() != 1)
        {
            throw std::runtime_error("Expected 1D tensor for bias read");
        }
        const uint64_t offset = tensor->storageOffset() + idx * static_cast<uint64_t>(strides[0]);
        if (tensor->dtype() == DataType::FLOAT32)
        {
            const auto *data = static_cast<const float *>(tensor->storage()->dataptr());
            return data[offset];
        }
        if (tensor->dtype() == DataType::BFLOAT16)
        {
            const auto *data = static_cast<const uint16_t *>(tensor->storage()->dataptr());
            return utils::Fp32FromBf16(data[offset]);
        }
        const auto *data = static_cast<const uint16_t *>(tensor->storage()->dataptr());
        return utils::Fp32FromFp16(data[offset]);
    }

    float ReadValueAt(const std::shared_ptr<RealTensor> &tensor, const uint64_t row, const uint64_t col)
    {
        const auto &shape = tensor->shape();
        const auto &strides = tensor->strides();
        const uint64_t offset =
            tensor->storageOffset() + row * static_cast<uint64_t>(strides[0]) + col * static_cast<uint64_t>(strides[1]);

        if (tensor->dtype() == DataType::FLOAT32)
        {
            const auto *data = static_cast<const float *>(tensor->storage()->dataptr());
            return data[offset];
        }
        if (tensor->dtype() == DataType::BFLOAT16)
        {
            const auto *data = static_cast<const uint16_t *>(tensor->storage()->dataptr());
            return utils::Fp32FromBf16(data[offset]);
        }
        const auto *data = static_cast<const uint16_t *>(tensor->storage()->dataptr());
        return utils::Fp32FromFp16(data[offset]);
    }

    float Gelu(float x)
    {
        constexpr float kInvSqrt2 = 0.7071067811865475f;
        return 0.5f * x * (1.0f + std::erf(x * kInvSqrt2));
    }

    std::shared_ptr<RealTensor> ComputeReference(const std::shared_ptr<RealTensor> &a_base,
                                                 const std::shared_ptr<RealTensor> &b_base,
                                                 const MatmulPermutation &cfg, const MatmulDims &dims,
                                                 const float base_value)
    {
        auto expected = RealTensor::Allocate({dims.m, dims.n}, cfg.output_dtype, DEVICE_CPU);
        const auto &out_strides = expected->strides();
        const uint64_t out_offset = expected->storageOffset();
        auto *out_fp32 =
            cfg.output_dtype == DataType::FLOAT32 ? reinterpret_cast<float *>(expected->storage()->dataptr()) : nullptr;
        auto *out_u16 = cfg.output_dtype == DataType::FLOAT32
                            ? nullptr
                            : reinterpret_cast<uint16_t *>(expected->storage()->dataptr());
        const bool out_is_bf16 = cfg.output_dtype == DataType::BFLOAT16;

        for (uint64_t m = 0; m < dims.m; ++m)
        {
            for (uint64_t n = 0; n < dims.n; ++n)
            {
                float acc = cfg.accumulate_output ? base_value : 0.0f;
                for (uint64_t k = 0; k < dims.k; ++k)
                {
                    const float a_val = cfg.transpose_a ? ReadValueAt(a_base, k, m) : ReadValueAt(a_base, m, k);
                    const float b_val = cfg.transpose_b ? ReadValueAt(b_base, n, k) : ReadValueAt(b_base, k, n);
                    acc += a_val * b_val;
                }
                const uint64_t out_idx = out_offset + m * static_cast<uint64_t>(out_strides[0]) +
                                         n * static_cast<uint64_t>(out_strides[1]);
                if (out_fp32)
                {
                    out_fp32[out_idx] = acc;
                }
                else
                {
                    out_u16[out_idx] = out_is_bf16 ? utils::Bf16FromFp32(acc) : utils::Fp16FromFp32(acc);
                }
            }
        }
        return expected;
    }

    std::pair<std::shared_ptr<RealTensor>, std::shared_ptr<RealTensor>>
    ComputeFusedReference(const std::shared_ptr<RealTensor> &a_base, const std::shared_ptr<RealTensor> &b_base,
                          const std::shared_ptr<RealTensor> &bias, const FusedPermutation &cfg,
                          const MatmulDims &dims, const float base_value)
    {
        auto pre_act = RealTensor::Allocate({dims.m, dims.n}, cfg.output_dtype, DEVICE_CPU);
        auto output = RealTensor::Allocate({dims.m, dims.n}, cfg.output_dtype, DEVICE_CPU);

        const auto &pre_strides = pre_act->strides();
        const auto &out_strides = output->strides();
        const uint64_t pre_offset = pre_act->storageOffset();
        const uint64_t out_offset = output->storageOffset();
        auto *pre_fp32 =
            cfg.output_dtype == DataType::FLOAT32 ? reinterpret_cast<float *>(pre_act->storage()->dataptr()) : nullptr;
        auto *out_fp32 =
            cfg.output_dtype == DataType::FLOAT32 ? reinterpret_cast<float *>(output->storage()->dataptr()) : nullptr;
        auto *pre_u16 =
            cfg.output_dtype == DataType::FLOAT32 ? nullptr
                                                  : reinterpret_cast<uint16_t *>(pre_act->storage()->dataptr());
        auto *out_u16 =
            cfg.output_dtype == DataType::FLOAT32 ? nullptr
                                                  : reinterpret_cast<uint16_t *>(output->storage()->dataptr());
        const bool out_is_bf16 = cfg.output_dtype == DataType::BFLOAT16;

        for (uint64_t m = 0; m < dims.m; ++m)
        {
            for (uint64_t n = 0; n < dims.n; ++n)
            {
                float acc = 0.0f;
                for (uint64_t k = 0; k < dims.k; ++k)
                {
                    const float a_val = ReadValueAt(a_base, m, k);
                    const float b_val = ReadValueAt(b_base, k, n);
                    acc += a_val * b_val;
                }
                acc += ReadVectorValue(bias, n);
                const float pre_val = acc;
                float out_val = Gelu(pre_val);
                if (cfg.accumulate_output)
                {
                    out_val += base_value;
                }

                const uint64_t pre_idx = pre_offset + m * static_cast<uint64_t>(pre_strides[0]) +
                                         n * static_cast<uint64_t>(pre_strides[1]);
                const uint64_t out_idx = out_offset + m * static_cast<uint64_t>(out_strides[0]) +
                                         n * static_cast<uint64_t>(out_strides[1]);

                if (out_fp32)
                {
                    pre_fp32[pre_idx] = pre_val;
                    out_fp32[out_idx] = out_val;
                }
                else
                {
                    pre_u16[pre_idx] = out_is_bf16 ? utils::Bf16FromFp32(pre_val) : utils::Fp16FromFp32(pre_val);
                    out_u16[out_idx] = out_is_bf16 ? utils::Bf16FromFp32(out_val) : utils::Fp16FromFp32(out_val);
                }
            }
        }

        return {output, pre_act};
    }

    void ApplyPasses(ExecutionPlan &plan)
    {
        std::vector<std::unique_ptr<pi::tensorlib::passes::CompilerPass>> passes{};
        passes.emplace_back(std::make_unique<MatmulFusePass>());
        passes.emplace_back(std::make_unique<MatmulImplPass>());
        passes.emplace_back(std::make_unique<ActImplPass>());
        passes.emplace_back(std::make_unique<TrailingBroadcastAddImplPass>());
        passes.emplace_back(std::make_unique<FillConstantImplPass>());
        pi::tensorlib::passes::Transform(plan, passes);
    }

    std::string ExtractKernelName(const std::string &log, const std::string &prefix)
    {
        const auto pos = log.find(prefix);
        if (pos == std::string::npos)
        {
            return "";
        }
        const auto start = pos + prefix.size();
        const auto end = log.find('\n', start);
        return log.substr(start, end == std::string::npos ? std::string::npos : end - start);
    }

    std::string DescribePermutation(const MatmulPermutation &cfg)
    {
        std::string desc = "dtype=" + std::string(GetDataTypeName(cfg.output_dtype));
        desc += " fp16_acc=" + std::string(cfg.use_fp16_accumulation ? "1" : "0");
        desc += " cacc=" + std::string(cfg.accumulate_output ? "1" : "0");
        desc += " trans=" + TransposeSuffix(cfg.transpose_a, cfg.transpose_b);
        desc += " aligned=" + std::string(cfg.aligned ? "1" : "0");
        return desc;
    }

    std::string DescribeFusedPermutation(const FusedPermutation &cfg)
    {
        std::string desc = "dtype=" + std::string(GetDataTypeName(cfg.output_dtype));
        desc += " fp16_acc=" + std::string(cfg.use_fp16_accumulation ? "1" : "0");
        desc += " cacc=" + std::string(cfg.accumulate_output ? "1" : "0");
        desc += " aligned=" + std::string(cfg.aligned ? "1" : "0");
        desc += " preact=" + std::string(cfg.write_out_preact ? "1" : "0");
        return desc;
    }

    void RunPermutation(const MatmulPermutation &cfg, const MatmulDims &dims, const bool force_triton,
                        std::set<std::string> &seen_kernels)
    {
        if (force_triton)
        {
            setenv("FBAMTRAIN_PREFER_GEMM_BACKEND", "triton", 1);
        }
        else
        {
            unsetenv("FBAMTRAIN_PREFER_GEMM_BACKEND");
        }
        setenv("FBAMTRAIN_DEBUG_GEMM", "1", 1);

        std::mt19937 rng(static_cast<uint32_t>(dims.m + 131 * dims.n + 517 * dims.k +
                                               (cfg.transpose_a ? 17 : 0) + (cfg.transpose_b ? 23 : 0) +
                                               (cfg.accumulate_output ? 29 : 0) +
                                               static_cast<int>(cfg.output_dtype)));

        const DataType input_dtype = InputDtypeForOutput(cfg.output_dtype);
        const uint64_t a_rows = cfg.transpose_a ? dims.k : dims.m;
        const uint64_t a_cols = cfg.transpose_a ? dims.m : dims.k;
        const uint64_t b_rows = cfg.transpose_b ? dims.n : dims.k;
        const uint64_t b_cols = cfg.transpose_b ? dims.k : dims.n;

        auto a_base = AllocateAndFillHost(a_rows, a_cols, input_dtype, rng);
        auto b_base = AllocateAndFillHost(b_rows, b_cols, input_dtype, rng);

        const float base_value = cfg.accumulate_output ? 0.25f : 0.0f;
        auto expected = ComputeReference(a_base, b_base, cfg, dims, base_value);

        const auto main_stream_desc = GpuStreamDescriptors::Main;
        TraceTensor a_base_trace = TraceTensor::Create({a_rows, a_cols}, input_dtype, DEVICE_GPU, main_stream_desc);
        TraceTensor b_base_trace = TraceTensor::Create({b_rows, b_cols}, input_dtype, DEVICE_GPU, main_stream_desc);
        a_base_trace.markRetained();
        b_base_trace.markRetained();
        OpGraph graph({{.name = "a_base", .tensor = a_base_trace}, {.name = "b_base", .tensor = b_base_trace}}, {});

        TraceTensor a_view = cfg.transpose_a ? graph.getInputDescriptors()[0].tensor.transpose(graph, {1, 0})
                                             : graph.getInputDescriptors()[0].tensor;
        TraceTensor b_view = cfg.transpose_b ? graph.getInputDescriptors()[1].tensor.transpose(graph, {1, 0})
                                             : graph.getInputDescriptors()[1].tensor;

        TraceTensor out = graph.createTensor({dims.m, dims.n}, cfg.output_dtype, DEVICE_GPU, main_stream_desc, false);
        if (cfg.accumulate_output)
        {
            FillConstant(graph, out, base_value, main_stream_desc);
        }

        std::unordered_map<std::string, std::any> attrs{};
        if (cfg.use_fp16_accumulation)
        {
            attrs.emplace("use_fp16_matmul_acc", true);
        }
        if (cfg.accumulate_output)
        {
            attrs.emplace("accumulate_output", true);
        }
        graph.recordOperation(OperationEntry{.type = OpType::MATMUL,
                                             .inputs = {a_view, b_view},
                                             .outputs = {out},
                                             .attributes = std::move(attrs),
                                             .gpu_stream_desc = main_stream_desc});
        graph.finalize();

        auto a_gpu = a_base->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
        auto b_gpu = b_base->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
        std::stringstream debug_stream;
        auto *old_buf = std::clog.rdbuf(debug_stream.rdbuf());

        ExecutionPlan plan =
            ExecutionPlan::FromGraph(graph, {{.name = "a_base", .tensor = a_gpu}, {.name = "b_base", .tensor = b_gpu}}, {});
#if PI_TENSORLIB_ENABLE_HIP && !PI_TENSORLIB_ENABLE_CUDA
        if (cfg.use_fp16_accumulation)
        {
            try
            {
                ApplyPasses(plan);
            }
            catch (const std::runtime_error &err)
            {
                std::clog.rdbuf(old_buf);
                const std::string message = err.what();
                if (message.find("FP16 accumulation is not supported on AMD") != std::string::npos)
                {
                    return;
                }
                throw;
            }
            std::clog.rdbuf(old_buf);
            throw std::runtime_error("Expected FP16 accumulation request to fail on AMD");
        }
#endif
        ApplyPasses(plan);

        ExecutionBackend &backend = ExecutionBackend::getInstance();
        const auto &allocator_registry = allocator::CachingAllocatorRegistry::instance();

        Executor executor{plan, backend, 0};
        executor.execute(allocator_registry);
        std::clog.rdbuf(old_buf);

        const auto output_rt = executor.getOutput(out);
        if (!output_rt)
        {
            throw std::runtime_error("Failed to retrieve matmul output");
        }

        const float tol = (cfg.output_dtype == DataType::FLOAT32) ? 1e-3f : (cfg.output_dtype == DataType::BFLOAT16 ? 5e-2f : 1e-2f);
        try
        {
            testing::AssertSimilar(expected, *output_rt, tol);
        }
        catch (const std::exception &err)
        {
            throw std::runtime_error("Matmul permutation mismatch (" + DescribePermutation(cfg) + "): " +
                                     err.what());
        }

        const std::string debug_log = debug_stream.str();
        if (force_triton)
        {
            const std::string expected_kernel = MakeTritonKernelName(cfg);
            if (debug_log.find(expected_kernel) == std::string::npos)
            {
                throw std::runtime_error("Expected Triton kernel " + expected_kernel + ", got log: " + debug_log);
            }
            const auto selected =
                ExtractKernelName(debug_log, "[Gemm] Selecting Triton matmul kernel ");
            if (!selected.empty())
            {
                const auto name = selected.substr(0, selected.find(' '));
                seen_kernels.insert(name);
            }
        }
        else
        {
            const std::string expected_kernel = MakeCutlassKernelName(cfg);
            if (debug_log.find(expected_kernel) == std::string::npos)
            {
                throw std::runtime_error("Expected CUTLASS kernel " + expected_kernel + ", got log: " + debug_log);
            }
            const auto selected = ExtractKernelName(debug_log, "[Gemm] Selecting CUTLASS kernel ");
            if (!selected.empty())
            {
                seen_kernels.insert(selected);
            }
        }
    }

    void RunFusedPermutation(const FusedPermutation &cfg, const MatmulDims &dims, const bool force_triton,
                             std::set<std::string> &seen_kernels)
    {
        if (force_triton)
        {
            setenv("FBAMTRAIN_PREFER_GEMM_BACKEND", "triton", 1);
        }
        else
        {
            unsetenv("FBAMTRAIN_PREFER_GEMM_BACKEND");
        }
        setenv("FBAMTRAIN_DEBUG_GEMM", "1", 1);

        std::mt19937 rng(static_cast<uint32_t>(dims.m + 131 * dims.n + 517 * dims.k +
                                               (cfg.accumulate_output ? 29 : 0) +
                                               (cfg.write_out_preact ? 37 : 0) +
                                               static_cast<int>(cfg.output_dtype)));

        const DataType input_dtype = InputDtypeForOutput(cfg.output_dtype);
        auto a_base = AllocateAndFillHost(dims.m, dims.k, input_dtype, rng);
        auto b_base = AllocateAndFillHost(dims.k, dims.n, input_dtype, rng);
        auto bias_base = AllocateAndFillVector(dims.n, input_dtype, rng);

        const float base_value = cfg.accumulate_output ? 0.25f : 0.0f;
        auto [expected_out, expected_pre] = ComputeFusedReference(a_base, b_base, bias_base, cfg, dims, base_value);

        const auto main_stream_desc = GpuStreamDescriptors::Main;
        TraceTensor a_base_trace = TraceTensor::Create({dims.m, dims.k}, input_dtype, DEVICE_GPU, main_stream_desc);
        TraceTensor b_base_trace = TraceTensor::Create({dims.k, dims.n}, input_dtype, DEVICE_GPU, main_stream_desc);
        TraceTensor bias_trace = TraceTensor::Create({dims.n}, input_dtype, DEVICE_GPU, main_stream_desc);
        a_base_trace.markRetained();
        b_base_trace.markRetained();
        bias_trace.markRetained();
        OpGraph graph({{.name = "a_base", .tensor = a_base_trace},
                       {.name = "b_base", .tensor = b_base_trace},
                       {.name = "bias", .tensor = bias_trace}},
                      {});

        TraceTensor matmul_out =
            graph.createTensor({dims.m, dims.n}, cfg.output_dtype, DEVICE_GPU, main_stream_desc, false);
        TraceTensor act_out =
            graph.createTensor({dims.m, dims.n}, cfg.output_dtype, DEVICE_GPU, main_stream_desc, false);
        if (cfg.write_out_preact)
        {
            matmul_out.markRetained();
        }
        if (cfg.accumulate_output)
        {
            FillConstant(graph, act_out, base_value, main_stream_desc);
        }

        std::unordered_map<std::string, std::any> attrs{};
        if (cfg.use_fp16_accumulation)
        {
            attrs.emplace("use_fp16_matmul_acc", true);
        }
        if (cfg.accumulate_output)
        {
            attrs.emplace("accumulate_output", true);
        }
        if (cfg.write_out_preact)
        {
            attrs.emplace("write_out_preact", true);
        }
        graph.recordOperation(OperationEntry{.type = OpType::MATMUL,
                                             .inputs = {graph.getInputDescriptors()[0].tensor,
                                                        graph.getInputDescriptors()[1].tensor},
                                             .outputs = {matmul_out},
                                             .attributes = std::move(attrs),
                                             .gpu_stream_desc = main_stream_desc});

        const auto bias_view = graph.getInputDescriptors()[2].tensor;
        graph.recordOperation(
            OperationEntry{.type = OpType::PLUS,
                           .inputs = {matmul_out, bias_view},
                           .outputs = {matmul_out},
                           .gpu_stream_desc = main_stream_desc});
        std::unordered_map<std::string, std::any> act_attrs{};
        act_attrs.emplace("activation_function", pi::tensorlib::ActivationFunction::GELU);
        graph.recordOperation(
            OperationEntry{.type = OpType::ACT_FN,
                           .inputs = {matmul_out},
                           .outputs = {act_out},
                           .attributes = act_attrs,
                           .gpu_stream_desc = main_stream_desc});
        graph.finalize();

        auto a_gpu = a_base->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
        auto b_gpu = b_base->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
        auto bias_gpu = bias_base->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);

        std::stringstream debug_stream;
        auto *old_buf = std::clog.rdbuf(debug_stream.rdbuf());

        ExecutionPlan plan = ExecutionPlan::FromGraph(
            graph, {{.name = "a_base", .tensor = a_gpu}, {.name = "b_base", .tensor = b_gpu}, {.name = "bias", .tensor = bias_gpu}},
            {});
#if PI_TENSORLIB_ENABLE_HIP && !PI_TENSORLIB_ENABLE_CUDA
        if (cfg.use_fp16_accumulation)
        {
            try
            {
                ApplyPasses(plan);
            }
            catch (const std::runtime_error &err)
            {
                std::clog.rdbuf(old_buf);
                const std::string message = err.what();
                if (message.find("FP16 accumulation is not supported on AMD") != std::string::npos)
                {
                    return;
                }
                throw;
            }
            std::clog.rdbuf(old_buf);
            throw std::runtime_error("Expected FP16 accumulation request to fail on AMD");
        }
#endif
        ApplyPasses(plan);

        ExecutionBackend &backend = ExecutionBackend::getInstance();
        const auto &allocator_registry = allocator::CachingAllocatorRegistry::instance();

        Executor executor{plan, backend, 0};
        executor.execute(allocator_registry);
        std::clog.rdbuf(old_buf);

        const auto output_rt = executor.getOutput(act_out);
        if (!output_rt)
        {
            throw std::runtime_error("Failed to retrieve fused matmul output");
        }
        const auto pre_rt = executor.getOutput(matmul_out);
        if (cfg.write_out_preact && !pre_rt)
        {
            throw std::runtime_error("Failed to retrieve fused pre-activation output");
        }

        const float tol = (cfg.output_dtype == DataType::BFLOAT16) ? 5e-2f : 1e-2f;
        const std::string backend_label = force_triton ? "triton" : "cutlass";
        try
        {
            testing::AssertSimilar(expected_out, *output_rt, tol);
        }
        catch (const std::exception &err)
        {
            throw std::runtime_error("Fused matmul mismatch (" + DescribeFusedPermutation(cfg) + ", " + backend_label +
                                     "): " + err.what());
        }
        if (cfg.write_out_preact && pre_rt)
        {
            try
            {
                testing::AssertSimilar(expected_pre, *pre_rt, tol);
            }
            catch (const std::exception &err)
            {
                throw std::runtime_error("Fused pre-activation mismatch (" + DescribeFusedPermutation(cfg) + ", " +
                                         backend_label + "): " + err.what());
            }
        }

        const std::string debug_log = debug_stream.str();
        if (force_triton)
        {
            const std::string expected_kernel = MakeTritonAddmmGeluKernelName(cfg);
            if (debug_log.find(expected_kernel) == std::string::npos)
            {
                throw std::runtime_error("Expected Triton fused kernel " + expected_kernel + ", got log: " + debug_log);
            }
            seen_kernels.insert(expected_kernel);
        }
        else
        {
            const std::string expected_kernel = MakeCutlassAddmmGeluKernelName(cfg);
            if (debug_log.find(expected_kernel) == std::string::npos)
            {
                throw std::runtime_error("Expected CUTLASS fused kernel " + expected_kernel + ", got log: " + debug_log);
            }
            seen_kernels.insert(expected_kernel);
        }
    }

    std::vector<MatmulPermutation> BuildPermutations()
    {
        std::vector<MatmulPermutation> cases;
        const std::vector<bool> bools{false, true};
        const std::vector<std::pair<bool, bool>> transposes{
            {false, false}, {true, false}, {false, true}, {true, true}};

        for (const bool aligned : bools)
        {
            for (const auto &[ta, tb] : transposes)
            {
                for (const bool accumulate : bools)
                {
                    for (const bool fp16_acc : bools)
                    {
                        cases.push_back(MatmulPermutation{DataType::FLOAT16, fp16_acc, accumulate, ta, tb, aligned});
                    }
                    cases.push_back(MatmulPermutation{DataType::BFLOAT16, false, accumulate, ta, tb, aligned});
                    cases.push_back(MatmulPermutation{DataType::FLOAT32, false, accumulate, ta, tb, aligned});
                }
            }
        }
        return cases;
    }

    std::vector<FusedPermutation> BuildFusedPermutations()
    {
        std::vector<FusedPermutation> cases;
        const std::vector<bool> bools{false, true};

        for (const bool aligned : bools)
        {
            for (const bool accumulate : bools)
            {
                for (const bool write_out_preact : bools)
                {
                    cases.push_back(
                        FusedPermutation{DataType::BFLOAT16, false, accumulate, aligned, write_out_preact});
                    cases.push_back(
                        FusedPermutation{DataType::FLOAT16, false, accumulate, aligned, write_out_preact});
                    cases.push_back(
                        FusedPermutation{DataType::FLOAT16, true, accumulate, aligned, write_out_preact});
                }
            }
        }
        return cases;
    }
} // namespace

int main()
{
    const auto permutations = BuildPermutations();
    const auto fused_permutations = BuildFusedPermutations();
    std::set<std::string> triton_seen;
    std::set<std::string> cutlass_seen;
    std::set<std::string> fused_triton_seen;
    std::set<std::string> fused_cutlass_seen;

    std::set<std::string> triton_expected;
    std::set<std::string> cutlass_expected;
    for (const auto &cfg : permutations)
    {
#if PI_TENSORLIB_ENABLE_HIP && !PI_TENSORLIB_ENABLE_CUDA
        if (cfg.use_fp16_accumulation)
        {
            continue;
        }
#endif
        triton_expected.insert(MakeTritonKernelName(cfg));
#if PI_TENSORLIB_ENABLE_CUDA
        if (cfg.aligned)
        {
            cutlass_expected.insert(MakeCutlassKernelName(cfg));
        }
#endif
    }
    std::set<std::string> fused_triton_expected;
    std::set<std::string> fused_cutlass_expected;
    for (const auto &cfg : fused_permutations)
    {
#if PI_TENSORLIB_ENABLE_HIP && !PI_TENSORLIB_ENABLE_CUDA
        if (cfg.use_fp16_accumulation)
        {
            continue;
        }
#endif
        fused_triton_expected.insert(MakeTritonAddmmGeluKernelName(cfg));
#if PI_TENSORLIB_ENABLE_CUDA
        if (cfg.aligned)
        {
            fused_cutlass_expected.insert(MakeCutlassAddmmGeluKernelName(cfg));
        }
#endif
    }

    for (const auto &cfg : permutations)
    {
        const MatmulDims dims = GetDims(cfg.aligned);
        RunPermutation(cfg, dims, /*force_triton=*/true, triton_seen);
#if PI_TENSORLIB_ENABLE_CUDA
        if (cfg.aligned)
        {
            RunPermutation(cfg, dims, /*force_triton=*/false, cutlass_seen);
        }
#endif
    }

    for (const auto &cfg : fused_permutations)
    {
        const MatmulDims dims = GetDims(cfg.aligned);
        RunFusedPermutation(cfg, dims, /*force_triton=*/true, fused_triton_seen);
#if PI_TENSORLIB_ENABLE_CUDA
        if (cfg.aligned)
        {
            RunFusedPermutation(cfg, dims, /*force_triton=*/false, fused_cutlass_seen);
        }
#endif
    }

    if (triton_seen != triton_expected)
    {
        std::cerr << "Triton kernel coverage mismatch. Expected " << triton_expected.size() << " kernels, saw "
                  << triton_seen.size() << "\n";
        for (const auto &name : triton_expected)
        {
            if (triton_seen.find(name) == triton_seen.end())
            {
                std::cerr << "Missing Triton kernel: " << name << "\n";
            }
        }
        return 1;
    }
#if PI_TENSORLIB_ENABLE_CUDA
    if (cutlass_seen != cutlass_expected)
    {
        std::cerr << "CUTLASS kernel coverage mismatch. Expected " << cutlass_expected.size() << " kernels, saw "
                  << cutlass_seen.size() << "\n";
        for (const auto &name : cutlass_expected)
        {
            if (cutlass_seen.find(name) == cutlass_seen.end())
            {
                std::cerr << "Missing CUTLASS kernel: " << name << "\n";
            }
        }
        return 1;
    }
#endif
    if (fused_triton_seen != fused_triton_expected)
    {
        std::cerr << "Fused Triton kernel coverage mismatch. Expected " << fused_triton_expected.size()
                  << " kernels, saw " << fused_triton_seen.size() << "\n";
        for (const auto &name : fused_triton_expected)
        {
            if (fused_triton_seen.find(name) == fused_triton_seen.end())
            {
                std::cerr << "Missing fused Triton kernel: " << name << "\n";
            }
        }
        return 1;
    }
#if PI_TENSORLIB_ENABLE_CUDA
    if (fused_cutlass_seen != fused_cutlass_expected)
    {
        std::cerr << "Fused CUTLASS kernel coverage mismatch. Expected " << fused_cutlass_expected.size()
                  << " kernels, saw " << fused_cutlass_seen.size() << "\n";
        for (const auto &name : fused_cutlass_expected)
        {
            if (fused_cutlass_seen.find(name) == fused_cutlass_seen.end())
            {
                std::cerr << "Missing fused CUTLASS kernel: " << name << "\n";
            }
        }
        return 1;
    }
#endif

    std::cout << "[matmul_permutations] cases=" << permutations.size()
              << " triton_kernels=" << triton_seen.size() << " cutlass_kernels=" << cutlass_seen.size()
              << " fused_cases=" << fused_permutations.size() << " fused_triton_kernels=" << fused_triton_seen.size()
              << " fused_cutlass_kernels=" << fused_cutlass_seen.size() << "\n";
    return 0;
}
