#include "testing.h"

#include <allocator.h>

#include <array>
#include <execution_backend.h>
#include <executor.h>
#include <functional.h>
#include <passes.h>
#include <safe_tensors.h>
#include <tensorlib.h>
#include <utils.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

static void ApplyDefaultPasses(pi::tensorlib::ExecutionPlan &execution_plan)
{
    std::vector<std::unique_ptr<pi::tensorlib::passes::CompilerPass>> passes{};
    passes.emplace_back(std::make_unique<AvgPool2dImplPass>());
    passes.emplace_back(std::make_unique<ContiguousImplPass>());
    passes.emplace_back(std::make_unique<FillZerosImplPass>());
    passes.emplace_back(std::make_unique<FillConstantImplPass>());
    passes.emplace_back(std::make_unique<FillUniformImplPass>());
    pi::tensorlib::passes::Transform(execution_plan, passes);
}

static bool PlanHasKernel(const pi::tensorlib::ExecutionPlan &plan, const std::string &kernel_name)
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

#define BATCH 4
#define CHANNELS 8
#define HEIGHT 32
#define WIDTH 28
#define KERNEL_H 3
#define KERNEL_W 5
#define STRIDE_H 2
#define STRIDE_W 3
#define PAD_H 1
#define PAD_W 2
#define SEED 2024

namespace
{
    constexpr uint32_t FAST_BATCH = 2;
    constexpr uint32_t FAST_HEIGHT = 8;
    constexpr uint32_t FAST_WIDTH = 10;
    constexpr uint32_t FAST_CHANNELS = 32;
    constexpr pi::tensorlib::Device DEVICE_CPU{
        .device_type = pi::tensorlib::DeviceType::CPU,
        .ordinal = 0,
    };

    float DecodeBf16(const uint16_t value)
    {
        return pi::tensorlib::utils::Fp32FromBf16(value);
    }

    uint16_t EncodeBf16(const float value)
    {
        return pi::tensorlib::utils::Bf16FromFp32(value);
    }

    void PopulateFastInput(const std::shared_ptr<pi::tensorlib::RealTensor> &tensor)
    {
        auto *data = static_cast<uint16_t *>(tensor->dataptr());
        const uint64_t total = static_cast<uint64_t>(FAST_BATCH) * FAST_HEIGHT * FAST_WIDTH * FAST_CHANNELS;
        for (uint64_t idx = 0; idx < total; ++idx)
        {
            const float value = static_cast<float>((idx * 17 + 11) % 251) / 83.0f - 1.5f;
            data[idx] = EncodeBf16(value);
        }
    }

    std::shared_ptr<pi::tensorlib::RealTensor> MakeFastExpected(
        const std::shared_ptr<pi::tensorlib::RealTensor> &input)
    {
        auto expected = pi::tensorlib::RealTensor::Allocate(
            {FAST_BATCH, FAST_HEIGHT / 2, FAST_WIDTH / 2, FAST_CHANNELS}, pi::tensorlib::DataType::BFLOAT16,
            DEVICE_CPU);
        const auto *input_data = static_cast<const uint16_t *>(input->dataptr());
        auto *expected_data = static_cast<uint16_t *>(expected->dataptr());

        for (uint32_t b = 0; b < FAST_BATCH; ++b)
        {
            for (uint32_t oh = 0; oh < FAST_HEIGHT / 2; ++oh)
            {
                for (uint32_t ow = 0; ow < FAST_WIDTH / 2; ++ow)
                {
                    for (uint32_t c = 0; c < FAST_CHANNELS; ++c)
                    {
                        const uint64_t out_idx =
                            ((static_cast<uint64_t>(b) * (FAST_HEIGHT / 2) + oh) * (FAST_WIDTH / 2) + ow) *
                                FAST_CHANNELS +
                            c;
                        const uint64_t in00 =
                            ((static_cast<uint64_t>(b) * FAST_HEIGHT + oh * 2) * FAST_WIDTH + ow * 2) *
                                FAST_CHANNELS +
                            c;
                        const uint64_t in01 = in00 + FAST_CHANNELS;
                        const uint64_t in10 = in00 + static_cast<uint64_t>(FAST_WIDTH) * FAST_CHANNELS;
                        const uint64_t in11 = in10 + FAST_CHANNELS;
                        const float avg = (DecodeBf16(input_data[in00]) + DecodeBf16(input_data[in01]) +
                                           DecodeBf16(input_data[in10]) + DecodeBf16(input_data[in11])) *
                                          0.25f;
                        expected_data[out_idx] = EncodeBf16(avg);
                    }
                }
            }
        }
        return expected;
    }
} // namespace

int main()
{
    pi::tensorlib::ExecutionBackend &execution_backend = pi::tensorlib::ExecutionBackend::getInstance();
    auto &allocator_registry = pi::tensorlib::allocator::DefaultAllocatorRegistry::instance();

    constexpr pi::tensorlib::Device device{
        .device_type = pi::tensorlib::DeviceType::GPU,
        .ordinal = 0,
    };
    const auto main_stream_desc = pi::tensorlib::GpuStreamDescriptors::Main;

    pi::tensorlib::OpGraph init_graph{{}, {}};
    pi::tensorlib::TraceTensor input =
        init_graph.createTensor({BATCH, CHANNELS, HEIGHT, WIDTH}, pi::tensorlib::DataType::BFLOAT16, device,
                                main_stream_desc, false);
    input.markRetained();

    pi::tensorlib::FillUniform(init_graph, input, -1.0f, 1.0f, SEED, main_stream_desc);
    init_graph.finalize();

    pi::tensorlib::ExecutionPlan init_plan = pi::tensorlib::ExecutionPlan::FromGraph(init_graph, {}, {});
    ApplyDefaultPasses(init_plan);
    pi::tensorlib::Executor init_executor{init_plan, execution_backend, 0};
    init_executor.execute(allocator_registry);

    pi::tensorlib::OpGraph graph{{
                                     {.name = "input", .tensor = input},
                                 },
                                 {}};

    const std::array<uint32_t, 2> kernel{KERNEL_H, KERNEL_W};
    const std::array<uint32_t, 2> stride{STRIDE_H, STRIDE_W};
    const std::array<uint32_t, 2> padding{PAD_H, PAD_W};

    pi::tensorlib::TraceTensor pooled = pi::tensorlib::AvgPool2d(graph, input, kernel, stride, main_stream_desc,
                                                                 padding);
    graph.finalize();

    std::shared_ptr<pi::tensorlib::RealTensor> input_real = *init_executor.getOutput(input);

    pi::tensorlib::ExecutionPlan plan =
        pi::tensorlib::ExecutionPlan::FromGraph(graph,
                                                {
                                                    {.name = "input", .tensor = input_real},
                                                },
                                                {});

    ApplyDefaultPasses(plan);
    pi::tensorlib::Executor executor{plan, execution_backend, 0};
    executor.execute(allocator_registry);

    std::shared_ptr<pi::tensorlib::RealTensor> actual_output = *executor.getOutput(pooled);

    const auto expected_tensors = pi::tensorlib::safetensors::Load("reference.safetensors");
    const auto expected_output_it = expected_tensors.find("output");
    if (expected_output_it == expected_tensors.end())
    {
        throw std::runtime_error("Expected output tensor not found in reference.safetensors");
    }
    const auto &expected_output = expected_output_it->second;

    pi::tensorlib::testing::AssertSimilar(expected_output, actual_output, 2e-3f);

    auto fast_input_host = pi::tensorlib::RealTensor::Allocate(
        {FAST_BATCH, FAST_HEIGHT, FAST_WIDTH, FAST_CHANNELS}, pi::tensorlib::DataType::BFLOAT16, DEVICE_CPU);
    PopulateFastInput(fast_input_host);
    auto fast_expected = MakeFastExpected(fast_input_host);

    pi::tensorlib::TraceTensor fast_input_cpu =
        pi::tensorlib::TraceTensor::Create({FAST_BATCH, FAST_HEIGHT, FAST_WIDTH, FAST_CHANNELS},
                                           pi::tensorlib::DataType::BFLOAT16, DEVICE_CPU, main_stream_desc);
    fast_input_cpu.markRetained();

    pi::tensorlib::OpGraph fast_graph({{.name = "input", .tensor = fast_input_cpu}}, {});
    pi::tensorlib::TraceTensor fast_input_gpu = fast_input_cpu.to(fast_graph, device, main_stream_desc);
    pi::tensorlib::TraceTensor fast_pooled =
        pi::tensorlib::AvgPool2d(fast_graph, fast_input_gpu, {2, 2}, {2, 2}, main_stream_desc, {0, 0}, true);
    fast_graph.finalize();

    pi::tensorlib::ExecutionPlan fast_plan =
        pi::tensorlib::ExecutionPlan::FromGraph(fast_graph, {{.name = "input", .tensor = fast_input_host}}, {});
    ApplyDefaultPasses(fast_plan);
#if PI_TENSORLIB_ENABLE_CUDA
    if (!PlanHasKernel(fast_plan, "avg_pool2d_nhwc_2x2_s2_bf16"))
    {
        throw std::runtime_error("Expected avg_pool2d_nhwc_2x2_s2_bf16 in execution plan");
    }
#endif

    pi::tensorlib::Executor fast_executor{fast_plan, execution_backend, 0};
    fast_executor.execute(allocator_registry);

    std::shared_ptr<pi::tensorlib::RealTensor> fast_actual_output = *fast_executor.getOutput(fast_pooled);
    pi::tensorlib::testing::AssertSimilar(fast_expected, fast_actual_output, 2e-3f);
}
