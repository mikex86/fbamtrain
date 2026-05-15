#include "testing.h"

#include "../common/test_dtype_utils.h"
#include <allocator.h>
#include <execution_backend.h>
#include <executor.h>
#include <functional.h>
#include <passes.h>
#include <tensorlib.h>
#include <utils.h>

#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace
{
    using namespace pi::tensorlib;

    constexpr Device DEVICE_CPU{DeviceType::CPU, 0};
    constexpr Device DEVICE_GPU{DeviceType::GPU, 0};

    constexpr uint32_t BATCH = 1;
    constexpr uint32_t IN_H = 8;
    constexpr uint32_t IN_W = 8;
    constexpr uint32_t IN_C = 32;
    constexpr uint32_t OUT_C = 64;
    constexpr uint32_t KERNEL_H = 3;
    constexpr uint32_t KERNEL_W = 3;
    constexpr uint32_t STRIDE = 1;
    constexpr uint32_t PAD = 1;
    constexpr uint32_t DILATION = 1;

    void ApplyPasses(ExecutionPlan &plan)
    {
        std::vector<std::unique_ptr<passes::CompilerPass>> passes{};
        passes.emplace_back(std::make_unique<Conv2dImplPass>());
        passes.emplace_back(std::make_unique<ContiguousImplPass>());
        passes.emplace_back(std::make_unique<FillConstantImplPass>());
        passes.emplace_back(std::make_unique<FillZerosImplPass>());
        passes::Transform(plan, passes);
    }

    uint32_t ComputeOutDim(const uint32_t in_size, const uint32_t kernel, const uint32_t pad, const uint32_t stride,
                           const uint32_t dilation)
    {
        const uint32_t effective = 1 + dilation * (kernel - 1);
        return (in_size + 2 * pad - effective) / stride + 1;
    }

    std::shared_ptr<RealTensor> MakeHostTensor(const std::initializer_list<uint64_t> shape,
                                               const std::vector<float> &values, const DataType dtype)
    {
        auto tensor = RealTensor::Allocate(shape, dtype, DEVICE_CPU);
        auto *dst = reinterpret_cast<uint16_t *>(tensor->dataptr());
        for (size_t i = 0; i < values.size(); ++i)
        {
            dst[i] = (dtype == DataType::BFLOAT16) ? utils::Bf16FromFp32(values[i]) : utils::Fp16FromFp32(values[i]);
        }
        return tensor;
    }

    std::vector<float> ComputeWgradReference(const std::vector<float> &input, const std::vector<float> &upstream,
                                             const uint32_t out_h, const uint32_t out_w)
    {
        std::vector<float> wgrad(static_cast<size_t>(OUT_C) * KERNEL_H * KERNEL_W * IN_C, 0.0f);

        auto input_idx = [](uint32_t n, uint32_t h, uint32_t w, uint32_t c) -> size_t
        {
            return static_cast<size_t>(((n * IN_H + h) * IN_W + w) * IN_C + c);
        };
        auto upstream_idx = [out_h, out_w](uint32_t n, uint32_t h, uint32_t w, uint32_t c) -> size_t
        {
            return static_cast<size_t>(((n * out_h + h) * out_w + w) * OUT_C + c);
        };
        auto weight_idx = [](uint32_t oc, uint32_t kh, uint32_t kw, uint32_t ic) -> size_t
        {
            return static_cast<size_t>(((oc * KERNEL_H + kh) * KERNEL_W + kw) * IN_C + ic);
        };

        for (uint32_t n = 0; n < BATCH; ++n)
        {
            for (uint32_t oh = 0; oh < out_h; ++oh)
            {
                for (uint32_t ow = 0; ow < out_w; ++ow)
                {
                    for (uint32_t oc = 0; oc < OUT_C; ++oc)
                    {
                        const float dout = upstream[upstream_idx(n, oh, ow, oc)];
                        for (uint32_t kh = 0; kh < KERNEL_H; ++kh)
                        {
                            const int32_t ih =
                                static_cast<int32_t>(oh * STRIDE) - static_cast<int32_t>(PAD) + static_cast<int32_t>(kh * DILATION);
                            if (ih < 0 || ih >= static_cast<int32_t>(IN_H))
                            {
                                continue;
                            }
                            for (uint32_t kw = 0; kw < KERNEL_W; ++kw)
                            {
                                const int32_t iw = static_cast<int32_t>(ow * STRIDE) - static_cast<int32_t>(PAD) +
                                                   static_cast<int32_t>(kw * DILATION);
                                if (iw < 0 || iw >= static_cast<int32_t>(IN_W))
                                {
                                    continue;
                                }
                                for (uint32_t ic = 0; ic < IN_C; ++ic)
                                {
                                    const float inp = input[input_idx(n, static_cast<uint32_t>(ih),
                                                                       static_cast<uint32_t>(iw), ic)];
                                    wgrad[weight_idx(oc, kh, kw, ic)] += inp * dout;
                                }
                            }
                        }
                    }
                }
            }
        }
        return wgrad;
    }

    void RunConv2dWgradAccumulateCase(const DataType dtype, const bool accumulate_output)
    {
        ExecutionBackend &backend = ExecutionBackend::getInstance();
        const auto &allocator_registry = allocator::DefaultAllocatorRegistry::instance();
        const auto main_stream_desc = GpuStreamDescriptors::Main;

        const uint32_t out_h = ComputeOutDim(IN_H, KERNEL_H, PAD, STRIDE, DILATION);
        const uint32_t out_w = ComputeOutDim(IN_W, KERNEL_W, PAD, STRIDE, DILATION);

        const size_t input_elems = static_cast<size_t>(BATCH) * IN_H * IN_W * IN_C;
        const size_t upstream_elems = static_cast<size_t>(BATCH) * out_h * out_w * OUT_C;
        const std::vector<float> input_vals(input_elems, 1.0f);
        const std::vector<float> upstream_vals(upstream_elems, 1.0f);

        auto input_host = MakeHostTensor({BATCH, IN_H, IN_W, IN_C}, input_vals, dtype);
        auto upstream_host = MakeHostTensor({BATCH, out_h, out_w, OUT_C}, upstream_vals, dtype);

        TraceTensor input_trace = TraceTensor::Create({BATCH, IN_H, IN_W, IN_C}, dtype, DEVICE_GPU, main_stream_desc);
        TraceTensor upstream_trace = TraceTensor::Create({BATCH, out_h, out_w, OUT_C}, dtype, DEVICE_GPU,
                                                         main_stream_desc);
        input_trace.markRetained();
        upstream_trace.markRetained();
        OpGraph graph({{.name = "input", .tensor = input_trace}, {.name = "upstream", .tensor = upstream_trace}}, {});

        TraceTensor grad_weight =
            graph.createTensor({OUT_C, KERNEL_H, KERNEL_W, IN_C}, dtype, DEVICE_GPU, main_stream_desc, false);
        FillConstant(graph, grad_weight, 1.0f, main_stream_desc);

        Conv2dWgradInto(graph, graph.getInputDescriptors()[0].tensor, graph.getInputDescriptors()[1].tensor,
                        grad_weight, {STRIDE, STRIDE}, {PAD, PAD}, {DILATION, DILATION},
                        main_stream_desc, accumulate_output);

        graph.finalize();

        auto input_gpu = input_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
        auto upstream_gpu = upstream_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);

        ExecutionPlan plan = ExecutionPlan::FromGraph(
            graph, {{.name = "input", .tensor = input_gpu}, {.name = "upstream", .tensor = upstream_gpu}}, {});
        ApplyPasses(plan);

        Executor executor{plan, backend, 0};
        executor.execute(allocator_registry);

        const auto actual = executor.getOutput(grad_weight);
        if (!actual)
        {
            throw std::runtime_error("Failed to retrieve conv2d wgrad accumulate output");
        }

        auto expected_vals = ComputeWgradReference(input_vals, upstream_vals, out_h, out_w);
        if (accumulate_output)
        {
            for (auto &val : expected_vals)
            {
                val += 1.0f;
            }
        }

        auto expected = RealTensor::Allocate({OUT_C, KERNEL_H, KERNEL_W, IN_C}, dtype, DEVICE_CPU);
        auto *dst = reinterpret_cast<uint16_t *>(expected->dataptr());
        for (size_t i = 0; i < expected_vals.size(); ++i)
        {
            dst[i] = (dtype == DataType::BFLOAT16) ? utils::Bf16FromFp32(expected_vals[i])
                                                   : utils::Fp16FromFp32(expected_vals[i]);
        }

        const float tolerance = test_utils::SelectTolerance(dtype, 2e-2f, 1e-2f);
        testing::AssertSimilar(expected, *actual, tolerance);
    }
} // namespace

int main()
{
    const auto dtype = test_utils::GetTestDtype();
    RunConv2dWgradAccumulateCase(dtype, /*accumulate_output=*/false);
    RunConv2dWgradAccumulateCase(dtype, /*accumulate_output=*/true);
    return 0;
}
