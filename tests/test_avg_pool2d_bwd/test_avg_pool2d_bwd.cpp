#include "testing.h"

#include "../common/test_dtype_utils.h"
#include <allocator.h>
#include <execution_backend.h>
#include <executor.h>
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

    constexpr Device DEVICE_GPU{DeviceType::GPU, 0};
    constexpr Device DEVICE_CPU{DeviceType::CPU, 0};
    const auto stream_desc = GpuStreamDescriptors::Main;

    constexpr uint32_t BATCH = 2;
    constexpr uint32_t CHANNELS = 4;
    constexpr uint32_t IN_H = 6;
    constexpr uint32_t IN_W = 8;
    constexpr uint32_t KERNEL_H = 2;
    constexpr uint32_t KERNEL_W = 2;
    constexpr uint32_t STRIDE_H = 2;
    constexpr uint32_t STRIDE_W = 2;
    constexpr uint32_t PAD_H = 0;
    constexpr uint32_t PAD_W = 0;

    constexpr uint32_t OUT_H = (IN_H + 2 * PAD_H - KERNEL_H) / STRIDE_H + 1;
    constexpr uint32_t OUT_W = (IN_W + 2 * PAD_W - KERNEL_W) / STRIDE_W + 1;

    uint16_t EncodeValue(const float value, const DataType dtype)
    {
        if (dtype == DataType::BFLOAT16)
        {
            return utils::Bf16FromFp32(value);
        }
        if (dtype == DataType::FLOAT16)
        {
            return utils::Fp16FromFp32(value);
        }
        throw std::runtime_error("Unsupported dtype in avg_pool2d_bwd test");
    }

    float DecodeValue(const uint16_t value, const DataType dtype)
    {
        if (dtype == DataType::BFLOAT16)
        {
            return utils::Fp32FromBf16(value);
        }
        if (dtype == DataType::FLOAT16)
        {
            return utils::Fp32FromFp16(value);
        }
        throw std::runtime_error("Unsupported dtype in avg_pool2d_bwd test");
    }

    void PopulateUpstream(const std::shared_ptr<RealTensor> &tensor, const DataType dtype)
    {
        auto *data = static_cast<uint16_t *>(tensor->dataptr());
        const uint64_t total = static_cast<uint64_t>(BATCH) * OUT_H * OUT_W * CHANNELS;
        for (uint64_t idx = 0; idx < total; ++idx)
        {
            const float value = static_cast<float>((idx + 1) % 257) / 257.0f - 0.5f;
            data[idx] = EncodeValue(value, dtype);
        }
    }

    std::shared_ptr<RealTensor> MakeExpectedGradInput(const std::shared_ptr<RealTensor> &upstream,
                                                      const DataType dtype)
    {
        auto expected =
            RealTensor::Allocate({BATCH, IN_H, IN_W, CHANNELS}, dtype, DEVICE_CPU);
        auto *expected_data = static_cast<uint16_t *>(expected->dataptr());
        const uint64_t total = static_cast<uint64_t>(BATCH) * IN_H * IN_W * CHANNELS;
        std::vector<float> accum(total, 0.0f);

        const auto *upstream_data = static_cast<const uint16_t *>(upstream->dataptr());
        const float scale = 1.0f / static_cast<float>(KERNEL_H * KERNEL_W);

        for (uint32_t b = 0; b < BATCH; ++b)
        {
            for (uint32_t oh = 0; oh < OUT_H; ++oh)
            {
                for (uint32_t ow = 0; ow < OUT_W; ++ow)
                {
                    for (uint32_t c = 0; c < CHANNELS; ++c)
                    {
                        const uint64_t out_index =
                            ((static_cast<uint64_t>(b) * OUT_H + oh) * OUT_W + ow) * CHANNELS + c;
                        const float upstream_val = DecodeValue(upstream_data[out_index], dtype);
                        const float scaled = upstream_val * scale;

                        const int32_t base_h = static_cast<int32_t>(oh * STRIDE_H) - static_cast<int32_t>(PAD_H);
                        const int32_t base_w = static_cast<int32_t>(ow * STRIDE_W) - static_cast<int32_t>(PAD_W);

                        for (uint32_t kh = 0; kh < KERNEL_H; ++kh)
                        {
                            const int32_t ih = base_h + static_cast<int32_t>(kh);
                            if (ih < 0 || ih >= static_cast<int32_t>(IN_H))
                            {
                                continue;
                            }
                            for (uint32_t kw = 0; kw < KERNEL_W; ++kw)
                            {
                                const int32_t iw = base_w + static_cast<int32_t>(kw);
                                if (iw < 0 || iw >= static_cast<int32_t>(IN_W))
                                {
                                    continue;
                                }
                                const uint64_t in_index =
                                    ((static_cast<uint64_t>(b) * IN_H + static_cast<uint32_t>(ih)) * IN_W +
                                     static_cast<uint32_t>(iw)) *
                                        CHANNELS +
                                    c;
                                accum[in_index] += scaled;
                            }
                        }
                    }
                }
            }
        }

        for (uint64_t idx = 0; idx < total; ++idx)
        {
            expected_data[idx] = EncodeValue(accum[idx], dtype);
        }
        return expected;
    }

    void ApplyPasses(ExecutionPlan &plan)
    {
        std::vector<std::unique_ptr<passes::CompilerPass>> passes{};
        passes.emplace_back(std::make_unique<AvgPool2dBwdImplPass>());
        passes.emplace_back(std::make_unique<ContiguousImplPass>());
        passes.emplace_back(std::make_unique<FillZerosImplPass>());
        passes.emplace_back(std::make_unique<FillConstantImplPass>());
        pi::tensorlib::passes::Transform(plan, passes);
    }

    bool PlanHasKernel(const ExecutionPlan &plan, const std::string &kernel_name)
    {
        for (const auto &entry : plan.entries)
        {
            if (entry.kernel_descriptor.has_value() && entry.kernel_descriptor->kernel_name == kernel_name)
            {
                return true;
            }
        }
        return false;
    }
} // namespace

int main()
{
    const auto dtype = test_utils::GetTestDtype();
    if (dtype != pi::tensorlib::DataType::BFLOAT16 && dtype != pi::tensorlib::DataType::FLOAT16)
    {
        throw std::runtime_error("avg_pool2d_bwd test requires BFLOAT16 or FLOAT16");
    }

    ExecutionBackend &execution_backend = ExecutionBackend::getInstance();
    auto &allocator_registry = allocator::DefaultAllocatorRegistry::instance();

    auto upstream_host = RealTensor::Allocate({BATCH, OUT_H, OUT_W, CHANNELS}, dtype, DEVICE_CPU);
    PopulateUpstream(upstream_host, dtype);

    auto expected_grad = MakeExpectedGradInput(upstream_host, dtype);

    TraceTensor upstream_cpu =
        TraceTensor::Create({BATCH, OUT_H, OUT_W, CHANNELS}, dtype, DEVICE_CPU, stream_desc);
    upstream_cpu.markRetained();

    OpGraph graph({{.name = "upstream", .tensor = upstream_cpu}}, {});

    TraceTensor upstream_gpu = upstream_cpu.to(graph, DEVICE_GPU, stream_desc);
    TraceTensor grad_input =
        graph.createTensor({BATCH, IN_H, IN_W, CHANNELS}, dtype, DEVICE_GPU, stream_desc, false);

    graph.recordOperation(OperationEntry{
        .type = OpType::AVG_POOL2D_BWD,
        .inputs = {upstream_gpu},
        .outputs = {grad_input},
        .attributes = {
            {"kernel_size", std::array<uint32_t, 2>{KERNEL_H, KERNEL_W}},
            {"stride", std::array<uint32_t, 2>{STRIDE_H, STRIDE_W}},
            {"padding", std::array<uint32_t, 2>{PAD_H, PAD_W}},
            {"channels_last", true},
            {"accumulate", false},
        },
        .gpu_stream_desc = stream_desc,
    });

    graph.finalize();

    ExecutionPlan plan = ExecutionPlan::FromGraph(graph, {{.name = "upstream", .tensor = upstream_host}}, {});
    ApplyPasses(plan);
#if PI_TENSORLIB_ENABLE_CUDA
    const std::string expected_kernel =
        std::string("avg_pool2d_bwd_noaccum_nhwc_2x2_s2_") + std::string(test_utils::GetDtypeSuffix(dtype));
    if (!PlanHasKernel(plan, expected_kernel))
    {
        throw std::runtime_error("Expected " + expected_kernel + " in execution plan");
    }
#endif

    Executor executor{plan, execution_backend, 0};
    executor.execute(allocator_registry);

    const auto actual_opt = executor.getOutput(grad_input);
    if (!actual_opt)
    {
        throw std::runtime_error("Missing avg_pool2d_bwd output");
    }

    const float tolerance = test_utils::SelectTolerance(dtype, 2e-2f, 2e-2f);
    testing::AssertSimilar(expected_grad, *actual_opt, tolerance);
    return 0;
}
