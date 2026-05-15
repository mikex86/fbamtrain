#include "testing.h"

#include "../common/test_dtype_utils.h"
#include <allocator.h>
#include <conv2d.h>
#include <execution_backend.h>
#include <executor.h>
#include <functional.h>
#include <passes.h>
#include <safe_tensors.h>
#include <tensorlib.h>
#include <utils.h>

#include <array>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{
    constexpr pi::tensorlib::Device DEVICE_GPU{pi::tensorlib::DeviceType::GPU, 0};
    constexpr pi::tensorlib::Device DEVICE_CPU{pi::tensorlib::DeviceType::CPU, 0};
    constexpr uint32_t STRIDE = 1;
    constexpr uint32_t PADDING = 1;
    constexpr uint32_t DILATION = 1;

    void ApplyDefaultPasses(pi::tensorlib::ExecutionPlan &plan)
    {
        std::vector<std::unique_ptr<pi::tensorlib::passes::CompilerPass>> passes{};
        passes.emplace_back(std::make_unique<Conv2dImplPass>());
        passes.emplace_back(std::make_unique<ContiguousImplPass>());
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

    bool IsEnvFlagEnabled(const char *name)
    {
        const char *value = std::getenv(name);
        return value != nullptr && std::string(value) != "0" && std::string(value) != "false";
    }

    std::shared_ptr<pi::tensorlib::RealTensor>
    MakeShapeOnlyGpuTensor(const std::vector<uint64_t> &dims, const pi::tensorlib::DataType dtype)
    {
        pi::tensorlib::Shape shape(dims);
        return std::make_shared<pi::tensorlib::RealTensor>(shape, pi::tensorlib::Strides(shape), dtype,
                                                           DEVICE_GPU, false, 0);
    }

    std::string PlanKernelList(const pi::tensorlib::ExecutionPlan &plan)
    {
        std::stringstream ss;
        for (const auto &entry : plan.entries)
        {
            if (!entry.kernel_descriptor.has_value())
            {
                continue;
            }
            ss << "\n  kernel=" << entry.kernel_descriptor->kernel_name
               << " function=" << entry.kernel_descriptor->function_name;
        }
        return ss.str();
    }

    void RequirePlanFunctionContains(const pi::tensorlib::ExecutionPlan &plan, const std::string &needle)
    {
        for (const auto &entry : plan.entries)
        {
            if (!entry.kernel_descriptor.has_value())
            {
                continue;
            }
            if (entry.kernel_descriptor->function_name.find(needle) != std::string::npos ||
                entry.kernel_descriptor->kernel_name.find(needle) != std::string::npos)
            {
                return;
            }
        }
        throw std::runtime_error("Expected conv2d plan to contain kernel substring '" + needle +
                                 "'. Plan kernels:" + PlanKernelList(plan));
    }

    std::string Conv2dBinShapeSuffix(const uint32_t batch, const uint32_t height, const uint32_t width)
    {
        return "_stride1_pad2_dil2_ic1024_oc1024_b" + std::to_string(batch) +
               "_h" + std::to_string(height) + "_w" + std::to_string(width) + "_kernel";
    }

    const char *Conv2dDTypeSuffix(const pi::tensorlib::DataType dtype)
    {
        return dtype == pi::tensorlib::DataType::BFLOAT16 ? "bf16" : "fp16";
    }

    template <uint32_t Batch, uint32_t InH, uint32_t InW>
    pi::tensorlib::ExecutionPlan BuildConv2dFwdDispatchPlan(const pi::tensorlib::DataType dtype,
                                                            const bool use_fp16_accumulation)
    {
        using namespace pi::tensorlib;

        constexpr uint32_t IN_C = 1024;
        constexpr uint32_t OUT_C = 1024;
        constexpr uint32_t KERNEL_H = 3;
        constexpr uint32_t KERNEL_W = 3;
        constexpr uint32_t STRIDE_D = 1;
        constexpr uint32_t PAD_D = 2;
        constexpr uint32_t DILATION_D = 2;

        const auto main_stream_desc = GpuStreamDescriptors::Main;
        TraceTensor input = TraceTensor::Create({Batch, InH, InW, IN_C}, dtype, DEVICE_GPU, main_stream_desc);
        TraceTensor weight = TraceTensor::Create({OUT_C, KERNEL_H, KERNEL_W, IN_C}, dtype, DEVICE_GPU, main_stream_desc);
        input.markRetained();
        weight.markRetained();

        OpGraph graph({
                          {.name = "input", .tensor = input},
                          {.name = "weight", .tensor = weight},
                      },
                      {});
        (void)Conv2d(graph, input, weight, nullptr, {STRIDE_D, STRIDE_D}, {PAD_D, PAD_D}, main_stream_desc,
                     {DILATION_D, DILATION_D}, use_fp16_accumulation);
        graph.finalize();

        ExecutionPlan plan = ExecutionPlan::FromGraph(
            graph,
            {{.name = "input", .tensor = MakeShapeOnlyGpuTensor({Batch, InH, InW, IN_C}, dtype)},
             {.name = "weight", .tensor = MakeShapeOnlyGpuTensor({OUT_C, KERNEL_H, KERNEL_W, IN_C}, dtype)}},
            {});
        ApplyDefaultPasses(plan);
        return plan;
    }

    template <uint32_t Batch, uint32_t InH, uint32_t InW>
    pi::tensorlib::ExecutionPlan BuildConv2dWgradDispatchPlan(const pi::tensorlib::DataType dtype)
    {
        using namespace pi::tensorlib;

        constexpr uint32_t IN_C = 1024;
        constexpr uint32_t OUT_C = 1024;
        constexpr uint32_t KERNEL_H = 3;
        constexpr uint32_t KERNEL_W = 3;
        constexpr uint32_t STRIDE_D = 1;
        constexpr uint32_t PAD_D = 2;
        constexpr uint32_t DILATION_D = 2;

        const auto main_stream_desc = GpuStreamDescriptors::Main;
        TraceTensor input = TraceTensor::Create({Batch, InH, InW, IN_C}, dtype, DEVICE_GPU, main_stream_desc);
        TraceTensor upstream = TraceTensor::Create({Batch, InH, InW, OUT_C}, dtype, DEVICE_GPU, main_stream_desc);
        TraceTensor grad_weight = TraceTensor::Create({OUT_C, KERNEL_H, KERNEL_W, IN_C}, dtype, DEVICE_GPU,
                                                      main_stream_desc);
        input.markRetained();
        upstream.markRetained();
        grad_weight.markRetained();

        OpGraph graph({
                          {.name = "input", .tensor = input},
                          {.name = "upstream", .tensor = upstream},
                          {.name = "grad_weight", .tensor = grad_weight},
                      },
                      {});
        Conv2dWgradInto(graph, input, upstream, grad_weight, {STRIDE_D, STRIDE_D}, {PAD_D, PAD_D},
                        {DILATION_D, DILATION_D}, main_stream_desc, false);
        graph.finalize();

        ExecutionPlan plan = ExecutionPlan::FromGraph(
            graph,
            {{.name = "input", .tensor = MakeShapeOnlyGpuTensor({Batch, InH, InW, IN_C}, dtype)},
             {.name = "upstream", .tensor = MakeShapeOnlyGpuTensor({Batch, InH, InW, OUT_C}, dtype)},
             {.name = "grad_weight", .tensor = MakeShapeOnlyGpuTensor({OUT_C, KERNEL_H, KERNEL_W, IN_C}, dtype)}},
            {});
        ApplyDefaultPasses(plan);
        return plan;
    }

    template <uint32_t Batch, uint32_t InH, uint32_t InW>
    pi::tensorlib::ExecutionPlan BuildConv2dDgradDispatchPlan(const pi::tensorlib::DataType dtype)
    {
        using namespace pi::tensorlib;

        constexpr uint32_t IN_C = 1024;
        constexpr uint32_t OUT_C = 1024;
        constexpr uint32_t KERNEL_H = 3;
        constexpr uint32_t KERNEL_W = 3;
        constexpr uint32_t STRIDE_D = 1;
        constexpr uint32_t PAD_D = 2;
        constexpr uint32_t DILATION_D = 2;

        const auto main_stream_desc = GpuStreamDescriptors::Main;
        TraceTensor upstream = TraceTensor::Create({Batch, InH, InW, OUT_C}, dtype, DEVICE_GPU, main_stream_desc);
        TraceTensor weight = TraceTensor::Create({OUT_C, KERNEL_H, KERNEL_W, IN_C}, dtype, DEVICE_GPU, main_stream_desc);
        TraceTensor grad_input = TraceTensor::Create({Batch, InH, InW, IN_C}, dtype, DEVICE_GPU, main_stream_desc);
        upstream.markRetained();
        weight.markRetained();
        grad_input.markRetained();

        OpGraph graph({
                          {.name = "upstream", .tensor = upstream},
                          {.name = "weight", .tensor = weight},
                          {.name = "grad_input", .tensor = grad_input},
                      },
                      {});
        Conv2dDgradInto(graph, upstream, weight, grad_input, {STRIDE_D, STRIDE_D}, {PAD_D, PAD_D},
                        {DILATION_D, DILATION_D}, main_stream_desc);
        graph.finalize();

        ExecutionPlan plan = ExecutionPlan::FromGraph(
            graph,
            {{.name = "upstream", .tensor = MakeShapeOnlyGpuTensor({Batch, InH, InW, OUT_C}, dtype)},
             {.name = "weight", .tensor = MakeShapeOnlyGpuTensor({OUT_C, KERNEL_H, KERNEL_W, IN_C}, dtype)},
             {.name = "grad_input", .tensor = MakeShapeOnlyGpuTensor({Batch, InH, InW, IN_C}, dtype)}},
            {});
        ApplyDefaultPasses(plan);
        return plan;
    }

    template <uint32_t Batch, uint32_t InH, uint32_t InW>
    void RequireConv2dCutlass3DispatchForShape(const pi::tensorlib::DataType dtype)
    {
        const std::string shape_suffix = Conv2dBinShapeSuffix(Batch, InH, InW);
        const std::string dtype_suffix = Conv2dDTypeSuffix(dtype);

        RequirePlanFunctionContains(BuildConv2dFwdDispatchPlan<Batch, InH, InW>(dtype, false),
                                    "cutlass_conv2d_" + dtype_suffix + shape_suffix);
        if (dtype == pi::tensorlib::DataType::FLOAT16)
        {
            RequirePlanFunctionContains(BuildConv2dFwdDispatchPlan<Batch, InH, InW>(dtype, true),
                                        "cutlass_conv2d_fp16_acc_fp16" + shape_suffix);
        }
        RequirePlanFunctionContains(BuildConv2dWgradDispatchPlan<Batch, InH, InW>(dtype),
                                    "cutlass_conv2d_wgrad_" + dtype_suffix + shape_suffix);
        RequirePlanFunctionContains(BuildConv2dDgradDispatchPlan<Batch, InH, InW>(dtype),
                                    "cutlass_conv2d_dgrad_" + dtype_suffix + shape_suffix);
    }

    void RunConv2dCutlass3DispatchChecks(const pi::tensorlib::DataType dtype)
    {
        if (!IsEnvFlagEnabled("CONV2D_EXPECT_CUTLASS3"))
        {
            return;
        }

        RequireConv2dCutlass3DispatchForShape<32, 24, 80>(dtype);
        RequireConv2dCutlass3DispatchForShape<32, 48, 160>(dtype);
    }

    void RunConv2dBackwardCase(const pi::tensorlib::DataType dtype)
    {
        using namespace pi::tensorlib;

        ExecutionBackend &backend = ExecutionBackend::getInstance();
        auto &allocator_registry = allocator::DefaultAllocatorRegistry::instance();
        const auto main_stream_desc = GpuStreamDescriptors::Main;

        const auto ref = safetensors::Load(test_utils::ReferenceFileName(dtype));
        const auto input_host = FetchTensor(ref, "input");
        const auto weight_host = FetchTensor(ref, "weight");
        const auto upstream_host = FetchTensor(ref, "upstream");
        const auto expected_dx = FetchTensor(ref, "grad_input");
        const auto expected_dw = FetchTensor(ref, "grad_weight");

        if (input_host->shape().ndims() != 4 || weight_host->shape().ndims() != 4 ||
            upstream_host->shape().ndims() != 4)
        {
            throw std::runtime_error("Conv2d backward reference tensors must be 4D");
        }

        const auto in_channels = static_cast<uint32_t>(input_host->shape()[3]);
        const auto out_channels = static_cast<uint32_t>(weight_host->shape()[0]);
        const auto kernel_h = static_cast<uint32_t>(weight_host->shape()[1]);
        const auto kernel_w = static_cast<uint32_t>(weight_host->shape()[2]);

        OpGraph init_graph{{}, {}};
        uint32_t seed = 0;
        struct pi::tensorlib::Conv2d conv("conv", in_channels, out_channels,
                                          std::array<uint32_t, 2>{kernel_h, kernel_w},
                                          std::array<uint32_t, 2>{STRIDE, STRIDE},
                                          std::array<uint32_t, 2>{PADDING, PADDING},
                                          std::array<uint32_t, 2>{DILATION, DILATION},
                                          DEVICE_GPU, dtype, init_graph, seed, main_stream_desc, false);

        TraceTensor input = TraceTensor::Create(input_host->shape().dims(), dtype, DEVICE_GPU, main_stream_desc);
        TraceTensor upstream = TraceTensor::Create(upstream_host->shape().dims(), dtype, DEVICE_GPU, main_stream_desc);
        input.markRetained();
        upstream.markRetained();
        const auto params = conv.parameters();
        auto retained_weight = params[0].tensor;
        retained_weight.markRetained();

        OpGraph graph({
                          {.name = "input", .tensor = input},
                          {.name = "upstream", .tensor = upstream},
                          {.name = params[0].name, .tensor = retained_weight},
                      },
                      {});

        (void)conv.buildForward(graph, {input}, /*save_input_for_backward=*/true);

        std::unordered_map<std::string, TraceTensor> parameter_grads{};
        std::unordered_map<std::string, TraceTensor> operand_grads{};
        TraceTensor w_grad = graph.createTensor(weight_host->shape().dims(), dtype, DEVICE_GPU, main_stream_desc, false);
        TraceTensor dx_grad = graph.createTensor(input_host->shape().dims(), dtype, DEVICE_GPU, main_stream_desc, false);
        FillZeros(graph, w_grad, main_stream_desc);
        FillZeros(graph, dx_grad, main_stream_desc);
        parameter_grads.emplace(params[0].name, w_grad);
        operand_grads.emplace("input", dx_grad);

        conv.buildBackward(graph, upstream, parameter_grads, operand_grads);
        graph.finalize();

        const auto input_gpu = input_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
        const auto upstream_gpu = upstream_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
        const auto weight_gpu = weight_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);

        std::vector<GraphExecutionInputDescriptor> inputs{
            {.name = "input", .tensor = input_gpu},
            {.name = "upstream", .tensor = upstream_gpu},
            {.name = params[0].name, .tensor = weight_gpu},
        };

        ExecutionPlan plan = ExecutionPlan::FromGraph(graph, inputs, {});
        ApplyDefaultPasses(plan);

        Executor executor{plan, backend, 0};
        executor.execute(allocator_registry);

        const auto actual_w = executor.getOutput(w_grad);
        const auto actual_dx = executor.getOutput(dx_grad);
        if (!actual_w || !actual_dx)
        {
            throw std::runtime_error("Failed to retrieve conv2d backward gradients");
        }

        const float tolerance = test_utils::SelectTolerance(dtype, 2e-1f, 2e-1f);
        testing::AssertSimilar(expected_dw, *actual_w, tolerance);
        testing::AssertSimilar(expected_dx, *actual_dx, tolerance);
    }

    uint16_t EncodeValue(const float value, const pi::tensorlib::DataType dtype)
    {
        if (dtype == pi::tensorlib::DataType::BFLOAT16)
        {
            return pi::tensorlib::utils::Bf16FromFp32(value);
        }
        return pi::tensorlib::utils::Fp16FromFp32(value);
    }

    float DecodeValue(const uint16_t value, const pi::tensorlib::DataType dtype)
    {
        if (dtype == pi::tensorlib::DataType::BFLOAT16)
        {
            return pi::tensorlib::utils::Fp32FromBf16(value);
        }
        return pi::tensorlib::utils::Fp32FromFp16(value);
    }

    void RunConv2dForwardDilated1024DenseCase(const pi::tensorlib::DataType dtype)
    {
        using namespace pi::tensorlib;

        constexpr uint32_t BATCH = 32;
        constexpr uint32_t IN_H = 24;
        constexpr uint32_t IN_W = 80;
        constexpr uint32_t IN_C = 1024;
        constexpr uint32_t OUT_C = 1024;
        constexpr uint32_t KERNEL_H = 3;
        constexpr uint32_t KERNEL_W = 3;
        constexpr uint32_t STRIDE_D = 1;
        constexpr uint32_t PAD_D = 2;
        constexpr uint32_t DILATION_D = 2;

        const uint32_t effective = 1 + DILATION_D * (KERNEL_H - 1);
        const uint32_t OUT_H = (IN_H + 2 * PAD_D - effective) / STRIDE_D + 1;
        const uint32_t OUT_W = (IN_W + 2 * PAD_D - effective) / STRIDE_D + 1;

        ExecutionBackend &backend = ExecutionBackend::getInstance();
        auto &allocator_registry = allocator::DefaultAllocatorRegistry::instance();
        const auto main_stream_desc = GpuStreamDescriptors::Main;

        auto input_host = RealTensor::Allocate({BATCH, IN_H, IN_W, IN_C}, dtype, DEVICE_CPU);
        auto weight_host = RealTensor::Allocate({OUT_C, KERNEL_H, KERNEL_W, IN_C}, dtype, DEVICE_CPU);

        auto *input_data = static_cast<uint16_t *>(input_host->dataptr());
        auto *weight_data = static_cast<uint16_t *>(weight_host->dataptr());

        auto fill_values = [&](uint16_t *data, size_t total, uint32_t mul, uint32_t add, float scale)
        {
            for (size_t idx = 0; idx < total; ++idx)
            {
                const float base = static_cast<float>((idx * mul + add) % 257u) / 257.0f - 0.5f;
                data[idx] = EncodeValue(base * scale, dtype);
            }
        };

        fill_values(input_data, static_cast<size_t>(input_host->shape().numel()), 13u, 7u, 0.1f);
        fill_values(weight_data, static_cast<size_t>(weight_host->shape().numel()), 19u, 11u, 0.1f);

        auto input_idx = [](uint32_t n, uint32_t h, uint32_t w, uint32_t c) -> size_t
        {
            return static_cast<size_t>(((n * IN_H + h) * IN_W + w) * IN_C + c);
        };
        auto weight_idx = [](uint32_t oc, uint32_t kh, uint32_t kw, uint32_t ic) -> size_t
        {
            return static_cast<size_t>(((oc * KERNEL_H + kh) * KERNEL_W + kw) * IN_C + ic);
        };
        auto output_idx = [](uint32_t n, uint32_t h, uint32_t w, uint32_t c) -> size_t
        {
            return static_cast<size_t>(((n * OUT_H + h) * OUT_W + w) * OUT_C + c);
        };

        TraceTensor input = TraceTensor::Create(input_host->shape().dims(), dtype, DEVICE_GPU, main_stream_desc);
        TraceTensor weight = TraceTensor::Create(weight_host->shape().dims(), dtype, DEVICE_GPU, main_stream_desc);
        input.markRetained();
        weight.markRetained();

        OpGraph graph({
                          {.name = "input", .tensor = input},
                          {.name = "weight", .tensor = weight},
                      },
                      {});

        TraceTensor output =
            Conv2d(graph, input, weight, nullptr, {STRIDE_D, STRIDE_D}, {PAD_D, PAD_D}, main_stream_desc,
                   {DILATION_D, DILATION_D}, /*use_fp16_conv_acc=*/false);

        graph.finalize();

        const auto input_gpu = input_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
        const auto weight_gpu = weight_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);

        ExecutionPlan plan = ExecutionPlan::FromGraph(
            graph,
            {{.name = "input", .tensor = input_gpu},
             {.name = "weight", .tensor = weight_gpu}},
            {});
        ApplyDefaultPasses(plan);

        Executor executor{plan, backend, 0};
        executor.execute(allocator_registry);

        const auto actual_out = executor.getOutput(output);
        if (!actual_out)
        {
            throw std::runtime_error("Failed to retrieve dense dilated conv2d output");
        }

        const auto actual_out_cpu = (*actual_out)->device().device_type == DeviceType::CPU
                                        ? *actual_out
                                        : (*actual_out)->to(DEVICE_CPU, main_stream_desc);
        const auto *actual_out_data = static_cast<const uint16_t *>(actual_out_cpu->dataptr());

        struct OutIndex
        {
            uint32_t n;
            uint32_t h;
            uint32_t w;
            uint32_t oc;
        };
        const std::array<OutIndex, 8> out_samples{{
            {0, 0, 0, 0},
            {0, 3, 4, 13},
            {0, 7, 7, 127},
            {1, 2, 1, 511},
            {1, 4, 6, 19},
            {1, 7, 3, 937},
            {0, 5, 2, 256},
            {1, 6, 7, 1023},
        }};

        const float tolerance = test_utils::SelectTolerance(dtype, 4e-2f, 2e-2f);

        for (const auto &idx : out_samples)
        {
            double acc = 0.0;
            for (uint32_t kh = 0; kh < KERNEL_H; ++kh)
            {
                const int32_t ih =
                    static_cast<int32_t>(idx.h) * static_cast<int32_t>(STRIDE_D) -
                    static_cast<int32_t>(PAD_D) + static_cast<int32_t>(kh * DILATION_D);
                if (ih < 0 || ih >= static_cast<int32_t>(IN_H))
                {
                    continue;
                }
                for (uint32_t kw = 0; kw < KERNEL_W; ++kw)
                {
                    const int32_t iw =
                        static_cast<int32_t>(idx.w) * static_cast<int32_t>(STRIDE_D) -
                        static_cast<int32_t>(PAD_D) + static_cast<int32_t>(kw * DILATION_D);
                    if (iw < 0 || iw >= static_cast<int32_t>(IN_W))
                    {
                        continue;
                    }
                    for (uint32_t ic = 0; ic < IN_C; ++ic)
                    {
                        const float input_val =
                            DecodeValue(input_data[input_idx(idx.n, static_cast<uint32_t>(ih),
                                                              static_cast<uint32_t>(iw), ic)], dtype);
                        const float weight_val =
                            DecodeValue(weight_data[weight_idx(idx.oc, kh, kw, ic)], dtype);
                        acc += static_cast<double>(input_val) * static_cast<double>(weight_val);
                    }
                }
            }

            const float actual =
                DecodeValue(actual_out_data[output_idx(idx.n, idx.h, idx.w, idx.oc)], dtype);
            if (std::abs(actual - acc) > tolerance)
            {
                throw std::runtime_error("Dense conv2d forward mismatch at n=" + std::to_string(idx.n) +
                                         " h=" + std::to_string(idx.h) + " w=" + std::to_string(idx.w) +
                                         " oc=" + std::to_string(idx.oc) + ": expected " +
                                         std::to_string(acc) + ", got " + std::to_string(actual));
            }
        }
    }

    template <uint32_t Batch, uint32_t InH, uint32_t InW>
    void RunConv2dBackwardDilated1024DenseWgradCaseForShape(const pi::tensorlib::DataType dtype)
    {
        using namespace pi::tensorlib;

        constexpr uint32_t BATCH = Batch;
        constexpr uint32_t IN_H = InH;
        constexpr uint32_t IN_W = InW;
        constexpr uint32_t IN_C = 1024;
        constexpr uint32_t OUT_C = 1024;
        constexpr uint32_t KERNEL_H = 3;
        constexpr uint32_t KERNEL_W = 3;
        constexpr uint32_t STRIDE_D = 1;
        constexpr uint32_t PAD_D = 2;
        constexpr uint32_t DILATION_D = 2;

        const uint32_t effective = 1 + DILATION_D * (KERNEL_H - 1);
        const uint32_t OUT_H = (IN_H + 2 * PAD_D - effective) / STRIDE_D + 1;
        const uint32_t OUT_W = (IN_W + 2 * PAD_D - effective) / STRIDE_D + 1;

        ExecutionBackend &backend = ExecutionBackend::getInstance();
        auto &allocator_registry = allocator::DefaultAllocatorRegistry::instance();
        const auto main_stream_desc = GpuStreamDescriptors::Main;

        auto input_host = RealTensor::Allocate({BATCH, IN_H, IN_W, IN_C}, dtype, DEVICE_CPU);
        auto upstream_host = RealTensor::Allocate({BATCH, OUT_H, OUT_W, OUT_C}, dtype, DEVICE_CPU);
        auto weight_host = RealTensor::Allocate({OUT_C, KERNEL_H, KERNEL_W, IN_C}, dtype, DEVICE_CPU);

        auto *input_data = static_cast<uint16_t *>(input_host->dataptr());
        auto *upstream_data = static_cast<uint16_t *>(upstream_host->dataptr());
        auto *weight_data = static_cast<uint16_t *>(weight_host->dataptr());

        auto fill_values = [&](uint16_t *data, size_t total, uint32_t mul, uint32_t add, float scale)
        {
            for (size_t idx = 0; idx < total; ++idx)
            {
                const float base = static_cast<float>((idx * mul + add) % 257u) / 257.0f - 0.5f;
                data[idx] = EncodeValue(base * scale, dtype);
            }
        };

        fill_values(input_data, static_cast<size_t>(input_host->shape().numel()), 13u, 7u, 0.1f);
        fill_values(upstream_data, static_cast<size_t>(upstream_host->shape().numel()), 17u, 3u, 0.1f);
        fill_values(weight_data, static_cast<size_t>(weight_host->shape().numel()), 19u, 11u, 0.1f);

        auto input_idx = [](uint32_t n, uint32_t h, uint32_t w, uint32_t c) -> size_t
        {
            return static_cast<size_t>(((n * IN_H + h) * IN_W + w) * IN_C + c);
        };
        auto upstream_idx = [](uint32_t n, uint32_t h, uint32_t w, uint32_t c) -> size_t
        {
            return static_cast<size_t>(((n * OUT_H + h) * OUT_W + w) * OUT_C + c);
        };
        auto weight_idx = [](uint32_t oc, uint32_t kh, uint32_t kw, uint32_t ic) -> size_t
        {
            return static_cast<size_t>(((oc * KERNEL_H + kh) * KERNEL_W + kw) * IN_C + ic);
        };

        struct WgradIndex
        {
            uint32_t oc;
            uint32_t kh;
            uint32_t kw;
            uint32_t ic;
        };
        const std::array<WgradIndex, 9> wgrad_samples{{
            {0, 0, 0, 0},
            {2, 1, 1, 39},
            {7, 1, 1, 13},
            {31, 2, 0, 127},
            {127, 0, 2, 511},
            {255, 2, 2, 19},
            {511, 1, 0, 937},
            {768, 0, 1, 256},
            {1023, 2, 2, 1023},
        }};

        struct DxIndex
        {
            uint32_t n;
            uint32_t h;
            uint32_t w;
            uint32_t ic;
        };
        const std::array<DxIndex, 8> dx_samples{{
            {0, 0, 0, 0},
            {0, 3, 4, 13},
            {0, 7, 7, 127},
            {1, 2, 1, 511},
            {1, 4, 6, 19},
            {1, 7, 3, 937},
            {0, 5, 2, 256},
            {1, 6, 7, 1023},
        }};

        const float tolerance = test_utils::SelectTolerance(dtype, 4e-2f, 2e-2f);

        auto run_wgrad_case = [&](const bool accumulate_output)
        {
            TraceTensor input = TraceTensor::Create(input_host->shape().dims(), dtype, DEVICE_GPU, main_stream_desc);
            TraceTensor upstream = TraceTensor::Create(upstream_host->shape().dims(), dtype, DEVICE_GPU, main_stream_desc);
            TraceTensor weight = TraceTensor::Create(weight_host->shape().dims(), dtype, DEVICE_GPU, main_stream_desc);
            input.markRetained();
            upstream.markRetained();
            weight.markRetained();

            OpGraph graph({
                              {.name = "input", .tensor = input},
                              {.name = "upstream", .tensor = upstream},
                              {.name = "weight", .tensor = weight},
                          },
                          {});

            TraceTensor w_grad =
                graph.createTensor(weight_host->shape().dims(), dtype, DEVICE_GPU, main_stream_desc, false);
            if (accumulate_output)
            {
                FillConstant(graph, w_grad, 1.0f, main_stream_desc);
            }
            else
            {
                FillZeros(graph, w_grad, main_stream_desc);
            }

            Conv2dWgradInto(graph, input, upstream, w_grad, {STRIDE_D, STRIDE_D}, {PAD_D, PAD_D},
                            {DILATION_D, DILATION_D}, main_stream_desc, accumulate_output);

            graph.finalize();

            const auto input_gpu = input_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
            const auto upstream_gpu = upstream_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
            const auto weight_gpu = weight_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);

            ExecutionPlan plan = ExecutionPlan::FromGraph(
                graph,
                {{.name = "input", .tensor = input_gpu},
                 {.name = "upstream", .tensor = upstream_gpu},
                 {.name = "weight", .tensor = weight_gpu}},
                {});
            ApplyDefaultPasses(plan);

            Executor executor{plan, backend, 0};
            executor.execute(allocator_registry);

            const auto actual_w = executor.getOutput(w_grad);
            if (!actual_w)
            {
                throw std::runtime_error("Failed to retrieve dense dilated conv2d wgrad");
            }

            auto to_cpu = [&](const std::shared_ptr<RealTensor> &tensor) -> std::shared_ptr<RealTensor>
            {
                if (tensor->device().device_type == DeviceType::CPU)
                {
                    return tensor;
                }
                return tensor->to(DEVICE_CPU, main_stream_desc);
            };

            const auto actual_w_cpu = to_cpu(*actual_w);
            const auto *actual_w_data = static_cast<const uint16_t *>(actual_w_cpu->dataptr());

            for (const auto &idx : wgrad_samples)
            {
                double acc = 0.0;
                for (uint32_t n = 0; n < BATCH; ++n)
                {
                    for (uint32_t oh = 0; oh < OUT_H; ++oh)
                    {
                        for (uint32_t ow = 0; ow < OUT_W; ++ow)
                        {
                            const int32_t ih =
                                static_cast<int32_t>(oh * STRIDE_D) - static_cast<int32_t>(PAD_D) +
                                static_cast<int32_t>(idx.kh * DILATION_D);
                            const int32_t iw =
                                static_cast<int32_t>(ow * STRIDE_D) - static_cast<int32_t>(PAD_D) +
                                static_cast<int32_t>(idx.kw * DILATION_D);
                            if (ih < 0 || ih >= static_cast<int32_t>(IN_H) ||
                                iw < 0 || iw >= static_cast<int32_t>(IN_W))
                            {
                                continue;
                            }

                            const float input_val =
                                DecodeValue(input_data[input_idx(n, static_cast<uint32_t>(ih),
                                                                  static_cast<uint32_t>(iw), idx.ic)], dtype);
                            const float upstream_val =
                                DecodeValue(upstream_data[upstream_idx(n, oh, ow, idx.oc)], dtype);
                            acc += static_cast<double>(input_val) * static_cast<double>(upstream_val);
                        }
                    }
                }

                if (accumulate_output)
                {
                    acc += 1.0;
                }

                const float actual =
                    DecodeValue(actual_w_data[weight_idx(idx.oc, idx.kh, idx.kw, idx.ic)], dtype);
                if (std::abs(actual - acc) > tolerance)
                {
                    throw std::runtime_error(std::string("Dense conv2d wgrad") +
                                             (accumulate_output ? " accumulate" : "") +
                                             " mismatch at oc=" + std::to_string(idx.oc) +
                                             " kh=" + std::to_string(idx.kh) + " kw=" + std::to_string(idx.kw) +
                                             " ic=" + std::to_string(idx.ic) + ": expected " +
                                             std::to_string(acc) + ", got " + std::to_string(actual));
                }
            }
        };

        run_wgrad_case(false);
        run_wgrad_case(true);

    }

    void RunConv2dBackwardDilated1024DenseWgradCase(const pi::tensorlib::DataType dtype)
    {
        RunConv2dBackwardDilated1024DenseWgradCaseForShape<32, 24, 80>(dtype);
        RunConv2dBackwardDilated1024DenseWgradCaseForShape<32, 48, 160>(dtype);
    }

    void RunConv2dBackwardDilated1024DenseDgradCase(const pi::tensorlib::DataType dtype)
    {
        using namespace pi::tensorlib;

        constexpr uint32_t BATCH = 2;
        constexpr uint32_t IN_H = 8;
        constexpr uint32_t IN_W = 8;
        constexpr uint32_t IN_C = 1024;
        constexpr uint32_t OUT_C = 1024;
        constexpr uint32_t KERNEL_H = 3;
        constexpr uint32_t KERNEL_W = 3;
        constexpr uint32_t STRIDE_D = 1;
        constexpr uint32_t PAD_D = 2;
        constexpr uint32_t DILATION_D = 2;

        const uint32_t effective = 1 + DILATION_D * (KERNEL_H - 1);
        const uint32_t OUT_H = (IN_H + 2 * PAD_D - effective) / STRIDE_D + 1;
        const uint32_t OUT_W = (IN_W + 2 * PAD_D - effective) / STRIDE_D + 1;

        ExecutionBackend &backend = ExecutionBackend::getInstance();
        auto &allocator_registry = allocator::DefaultAllocatorRegistry::instance();
        const auto main_stream_desc = GpuStreamDescriptors::Main;

        auto input_host = RealTensor::Allocate({BATCH, IN_H, IN_W, IN_C}, dtype, DEVICE_CPU);
        auto upstream_host = RealTensor::Allocate({BATCH, OUT_H, OUT_W, OUT_C}, dtype, DEVICE_CPU);
        auto weight_host = RealTensor::Allocate({OUT_C, KERNEL_H, KERNEL_W, IN_C}, dtype, DEVICE_CPU);

        auto *input_data = static_cast<uint16_t *>(input_host->dataptr());
        auto *upstream_data = static_cast<uint16_t *>(upstream_host->dataptr());
        auto *weight_data = static_cast<uint16_t *>(weight_host->dataptr());

        auto fill_values = [&](uint16_t *data, size_t total, uint32_t mul, uint32_t add, float scale)
        {
            for (size_t idx = 0; idx < total; ++idx)
            {
                const float base = static_cast<float>((idx * mul + add) % 257u) / 257.0f - 0.5f;
                data[idx] = EncodeValue(base * scale, dtype);
            }
        };

        fill_values(input_data, static_cast<size_t>(input_host->shape().numel()), 13u, 7u, 0.1f);
        fill_values(upstream_data, static_cast<size_t>(upstream_host->shape().numel()), 17u, 3u, 0.1f);
        fill_values(weight_data, static_cast<size_t>(weight_host->shape().numel()), 19u, 11u, 0.1f);

        auto input_idx = [](uint32_t n, uint32_t h, uint32_t w, uint32_t c) -> size_t
        {
            return static_cast<size_t>(((n * IN_H + h) * IN_W + w) * IN_C + c);
        };
        auto upstream_idx = [](uint32_t n, uint32_t h, uint32_t w, uint32_t c) -> size_t
        {
            return static_cast<size_t>(((n * OUT_H + h) * OUT_W + w) * OUT_C + c);
        };
        auto weight_idx = [](uint32_t oc, uint32_t kh, uint32_t kw, uint32_t ic) -> size_t
        {
            return static_cast<size_t>(((oc * KERNEL_H + kh) * KERNEL_W + kw) * IN_C + ic);
        };

        TraceTensor input = TraceTensor::Create(input_host->shape().dims(), dtype, DEVICE_GPU, main_stream_desc);
        TraceTensor upstream = TraceTensor::Create(upstream_host->shape().dims(), dtype, DEVICE_GPU, main_stream_desc);
        TraceTensor weight = TraceTensor::Create(weight_host->shape().dims(), dtype, DEVICE_GPU, main_stream_desc);
        input.markRetained();
        upstream.markRetained();
        weight.markRetained();

        OpGraph graph({
                          {.name = "input", .tensor = input},
                          {.name = "upstream", .tensor = upstream},
                          {.name = "weight", .tensor = weight},
                      },
                      {});

        constexpr float kDgradSentinel = -17.0f;
        TraceTensor dx_grad = graph.createTensor(input_host->shape().dims(), dtype, DEVICE_GPU, main_stream_desc, false);
        FillConstant(graph, dx_grad, kDgradSentinel, main_stream_desc);

        Conv2dDgradInto(graph, upstream, weight, dx_grad, {STRIDE_D, STRIDE_D}, {PAD_D, PAD_D},
                        {DILATION_D, DILATION_D}, main_stream_desc);

        graph.finalize();

        const auto input_gpu = input_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
        const auto upstream_gpu = upstream_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
        const auto weight_gpu = weight_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);

        ExecutionPlan plan = ExecutionPlan::FromGraph(
            graph,
            {{.name = "input", .tensor = input_gpu},
             {.name = "upstream", .tensor = upstream_gpu},
             {.name = "weight", .tensor = weight_gpu}},
            {});
        ApplyDefaultPasses(plan);

        Executor executor{plan, backend, 0};
        executor.execute(allocator_registry);

        const auto actual_dx = executor.getOutput(dx_grad);
        if (!actual_dx)
        {
            throw std::runtime_error("Failed to retrieve dense dilated conv2d dgrad");
        }

        auto to_cpu = [&](const std::shared_ptr<RealTensor> &tensor) -> std::shared_ptr<RealTensor>
        {
            if (tensor->device().device_type == DeviceType::CPU)
            {
                return tensor;
            }
            return tensor->to(DEVICE_CPU, main_stream_desc);
        };

        const auto actual_dx_cpu = to_cpu(*actual_dx);
        const auto *actual_dx_data = static_cast<const uint16_t *>(actual_dx_cpu->dataptr());

        const uint16_t sentinel_bits = EncodeValue(kDgradSentinel, dtype);
        constexpr uint32_t BLOCK_K = 32;
        constexpr uint32_t IC_BLOCKS = (IN_C + BLOCK_K - 1) / BLOCK_K;
        std::array<uint8_t, IC_BLOCKS> block_written{};
        size_t sentinel_count = 0;
        for (uint32_t n = 0; n < BATCH; ++n)
        {
            for (uint32_t h = 0; h < IN_H; ++h)
            {
                for (uint32_t w = 0; w < IN_W; ++w)
                {
                    for (uint32_t c = 0; c < IN_C; ++c)
                    {
                        const size_t idx = input_idx(n, h, w, c);
                        if (actual_dx_data[idx] == sentinel_bits)
                        {
                            ++sentinel_count;
                        }
                        else
                        {
                            block_written[c / BLOCK_K] = 1;
                        }
                    }
                }
            }
        }

        if (sentinel_count > 0)
        {
            std::string missing_blocks;
            for (uint32_t block = 0; block < IC_BLOCKS; ++block)
            {
                if (block_written[block] == 0)
                {
                    if (!missing_blocks.empty())
                    {
                        missing_blocks += ",";
                    }
                    missing_blocks += std::to_string(block);
                }
            }
            throw std::runtime_error("Dense conv2d dgrad output not populated: sentinel_count=" +
                                     std::to_string(sentinel_count) + " missing_ic_blocks=[" + missing_blocks + "]");
        }

        struct DxIndex
        {
            uint32_t n;
            uint32_t h;
            uint32_t w;
            uint32_t ic;
        };
        const std::array<DxIndex, 8> dx_samples{{
            {0, 0, 0, 0},
            {0, 3, 4, 13},
            {0, 7, 7, 127},
            {1, 2, 1, 511},
            {1, 4, 6, 19},
            {1, 7, 3, 937},
            {0, 5, 2, 256},
            {1, 6, 7, 1023},
        }};

        const float tolerance = test_utils::SelectTolerance(dtype, 4e-2f, 2e-2f);

        for (const auto &idx : dx_samples)
        {
            double acc = 0.0;
            for (uint32_t oc = 0; oc < OUT_C; ++oc)
            {
                for (uint32_t kh = 0; kh < KERNEL_H; ++kh)
                {
                    const int32_t oh_raw =
                        static_cast<int32_t>(idx.h) + static_cast<int32_t>(PAD_D) -
                        static_cast<int32_t>(kh * DILATION_D);
                    if (oh_raw < 0 || oh_raw >= static_cast<int32_t>(OUT_H))
                    {
                        continue;
                    }
                    const uint32_t oh = static_cast<uint32_t>(oh_raw);
                    for (uint32_t kw = 0; kw < KERNEL_W; ++kw)
                    {
                        const int32_t ow_raw =
                            static_cast<int32_t>(idx.w) + static_cast<int32_t>(PAD_D) -
                            static_cast<int32_t>(kw * DILATION_D);
                        if (ow_raw < 0 || ow_raw >= static_cast<int32_t>(OUT_W))
                        {
                            continue;
                        }
                        const uint32_t ow = static_cast<uint32_t>(ow_raw);

                        const float upstream_val =
                            DecodeValue(upstream_data[upstream_idx(idx.n, oh, ow, oc)], dtype);
                        const float weight_val =
                            DecodeValue(weight_data[weight_idx(oc, kh, kw, idx.ic)], dtype);
                        acc += static_cast<double>(upstream_val) * static_cast<double>(weight_val);
                    }
                }
            }

            const float actual =
                DecodeValue(actual_dx_data[input_idx(idx.n, idx.h, idx.w, idx.ic)], dtype);
            if (std::abs(actual - acc) > tolerance)
            {
                throw std::runtime_error("Dense conv2d dgrad mismatch at n=" + std::to_string(idx.n) +
                                         " h=" + std::to_string(idx.h) + " w=" + std::to_string(idx.w) +
                                         " ic=" + std::to_string(idx.ic) + ": expected " +
                                         std::to_string(acc) + ", got " + std::to_string(actual));
            }
        }
    }
} // namespace

int main()
{
    const auto dtype = test_utils::GetTestDtype();
    RunConv2dBackwardCase(dtype);
    RunConv2dForwardDilated1024DenseCase(dtype);
    RunConv2dBackwardDilated1024DenseWgradCase(dtype);
    RunConv2dBackwardDilated1024DenseDgradCase(dtype);
    return 0;
}
