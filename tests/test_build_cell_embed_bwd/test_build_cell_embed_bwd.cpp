#include "testing.h"

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
    constexpr uint32_t BATCH_SIZE = 2;
    constexpr uint32_t ROWS = 2;
    constexpr uint32_t COLS = 2;
    constexpr uint32_t EMBED_DIM = 16;
    constexpr uint32_t NUM_CHANNELS_PER_CELL = 3;

    constexpr pi::tensorlib::Device DEVICE_GPU{pi::tensorlib::DeviceType::GPU, 0};
    constexpr pi::tensorlib::Device DEVICE_CPU{pi::tensorlib::DeviceType::CPU, 0};
    const auto stream_desc = pi::tensorlib::GpuStreamDescriptors::Main;

    uint32_t ComputeColor(const uint32_t index, const uint32_t r_mul, const uint32_t r_bias, const uint32_t g_mul,
                          const uint32_t g_bias, const uint32_t b_mul, const uint32_t b_bias)
    {
        const uint32_t r = (index * r_mul + r_bias) % 256u;
        const uint32_t g = (index * g_mul + g_bias) % 256u;
        const uint32_t b = (index * b_mul + b_bias) % 256u;
        return (r << 16u) | (g << 8u) | b;
    }

    void PopulateCellStates(const std::shared_ptr<pi::tensorlib::RealTensor> &tensor)
    {
        auto *data = static_cast<uint32_t *>(tensor->dataptr());
        uint32_t index = 0;
        for (uint32_t batch = 0; batch < BATCH_SIZE; ++batch)
        {
            for (uint32_t row = 0; row < ROWS; ++row)
            {
                for (uint32_t col = 0; col < COLS; ++col)
                {
                    const uint64_t base =
                        (static_cast<uint64_t>(batch) * ROWS * COLS + row * COLS + col) * NUM_CHANNELS_PER_CELL;
                    data[base + 0] = index;
                    data[base + 1] = ComputeColor(index, 17u, 5u, 19u, 11u, 23u, 13u);
                    data[base + 2] = ComputeColor(index, 29u, 17u, 31u, 19u, 37u, 23u);
                    ++index;
                }
            }
        }
    }

    void PopulateGradOut(const std::shared_ptr<pi::tensorlib::RealTensor> &tensor)
    {
        auto *data = static_cast<uint16_t *>(tensor->dataptr());
        const uint64_t rows = BATCH_SIZE * ROWS * COLS;
        for (uint64_t i = 0; i < rows; ++i)
        {
            for (uint64_t d = 0; d < EMBED_DIM; ++d)
            {
                const float value = static_cast<float>((i + 1) * (d + 1)) * 0.001f;
                data[i * EMBED_DIM + d] = pi::tensorlib::utils::Bf16FromFp32(value);
            }
        }
    }

    std::shared_ptr<pi::tensorlib::RealTensor> MakeExpectedColorGrad(
        const std::shared_ptr<pi::tensorlib::RealTensor> &grad_out_cpu,
        const std::shared_ptr<pi::tensorlib::RealTensor> &cell_states_cpu, const bool is_fg, const int channel)
    {
        auto expected =
            pi::tensorlib::RealTensor::Allocate({EMBED_DIM}, pi::tensorlib::DataType::BFLOAT16, DEVICE_CPU);
        auto *expected_data = static_cast<uint16_t *>(expected->dataptr());
        std::fill(expected_data, expected_data + EMBED_DIM, static_cast<uint16_t>(0));

        const auto *grad_data = static_cast<const uint16_t *>(grad_out_cpu->dataptr());
        const auto *cell_data = static_cast<const uint32_t *>(cell_states_cpu->dataptr());
        const uint64_t rows = BATCH_SIZE * ROWS * COLS;

        std::vector<float> accum(EMBED_DIM, 0.0f);
        const int state_channel = is_fg ? 1 : 2;
        for (uint64_t i = 0; i < rows; ++i)
        {
            const uint32_t rgb = cell_data[i * NUM_CHANNELS_PER_CELL + state_channel];
            const float inv255 = 1.0f / 255.0f;
            const float r = static_cast<float>((rgb >> 16) & 0xFFu) * inv255;
            const float g = static_cast<float>((rgb >> 8) & 0xFFu) * inv255;
            const float b = static_cast<float>(rgb & 0xFFu) * inv255;
            float channel_value = 0.0f;
            switch (channel)
            {
                case 0:
                    channel_value = r;
                    break;
                case 1:
                    channel_value = g;
                    break;
                case 2:
                    channel_value = b;
                    break;
                default:
                    throw std::runtime_error("Invalid color channel");
            }

            for (uint64_t d = 0; d < EMBED_DIM; ++d)
            {
                const float grad = pi::tensorlib::utils::Fp32FromBf16(grad_data[i * EMBED_DIM + d]);
                accum[d] += grad * channel_value;
            }
        }

        for (uint64_t d = 0; d < EMBED_DIM; ++d)
        {
            expected_data[d] = pi::tensorlib::utils::Bf16FromFp32(accum[d]);
        }
        return expected;
    }

    void ApplyPasses(pi::tensorlib::ExecutionPlan &plan)
    {
        std::vector<std::unique_ptr<pi::tensorlib::passes::CompilerPass>> passes;
        passes.emplace_back(std::make_unique<FillZerosImplPass>());
        passes.emplace_back(std::make_unique<BuildCellEmbedBwdPass>());
        passes.emplace_back(std::make_unique<CastImplPass>());
        pi::tensorlib::passes::Transform(plan, passes);
    }
} // namespace

int main()
{
    pi::tensorlib::ExecutionBackend &execution_backend = pi::tensorlib::ExecutionBackend::getInstance();
    auto &allocator_registry = pi::tensorlib::allocator::DefaultAllocatorRegistry::instance();

    auto cell_states_cpu = pi::tensorlib::RealTensor::Allocate(
        {BATCH_SIZE, ROWS, COLS, NUM_CHANNELS_PER_CELL}, pi::tensorlib::DataType::UINT32, DEVICE_CPU);
    PopulateCellStates(cell_states_cpu);

    auto grad_out_cpu = pi::tensorlib::RealTensor::Allocate({BATCH_SIZE * ROWS * COLS, EMBED_DIM},
                                                            pi::tensorlib::DataType::BFLOAT16, DEVICE_CPU);
    PopulateGradOut(grad_out_cpu);

    pi::tensorlib::TraceTensor cell_states_trace = pi::tensorlib::TraceTensor::Create(
        {BATCH_SIZE, ROWS, COLS, NUM_CHANNELS_PER_CELL}, pi::tensorlib::DataType::UINT32, DEVICE_CPU, stream_desc);
    pi::tensorlib::TraceTensor grad_out_trace = pi::tensorlib::TraceTensor::Create(
        {BATCH_SIZE * ROWS * COLS, EMBED_DIM}, pi::tensorlib::DataType::BFLOAT16, DEVICE_CPU, stream_desc);
    cell_states_trace.markRetained();
    grad_out_trace.markRetained();

    pi::tensorlib::OpGraph graph(
        {
            {.name = "cell_states", .tensor = cell_states_trace},
            {.name = "grad_out", .tensor = grad_out_trace},
        },
        {});

    const auto cell_states_gpu = cell_states_trace.to(graph, DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
    const auto grad_out_gpu = grad_out_trace.to(graph, DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);

    auto grad_out_flat = grad_out_gpu.viewInferred(graph, {-1, static_cast<int64_t>(EMBED_DIM)});
    auto cell_states_flat = cell_states_gpu.viewInferred(graph, {-1, static_cast<int64_t>(NUM_CHANNELS_PER_CELL)});

    auto fg_r = graph.createTensor({EMBED_DIM}, pi::tensorlib::DataType::BFLOAT16, DEVICE_GPU, stream_desc, false);
    auto fg_g = graph.createTensor({EMBED_DIM}, pi::tensorlib::DataType::BFLOAT16, DEVICE_GPU, stream_desc, false);
    auto fg_b = graph.createTensor({EMBED_DIM}, pi::tensorlib::DataType::BFLOAT16, DEVICE_GPU, stream_desc, false);
    auto bg_r = graph.createTensor({EMBED_DIM}, pi::tensorlib::DataType::BFLOAT16, DEVICE_GPU, stream_desc, false);
    auto bg_g = graph.createTensor({EMBED_DIM}, pi::tensorlib::DataType::BFLOAT16, DEVICE_GPU, stream_desc, false);
    auto bg_b = graph.createTensor({EMBED_DIM}, pi::tensorlib::DataType::BFLOAT16, DEVICE_GPU, stream_desc, false);

    FillZeros(graph, fg_r, stream_desc);
    FillZeros(graph, fg_g, stream_desc);
    FillZeros(graph, fg_b, stream_desc);
    FillZeros(graph, bg_r, stream_desc);
    FillZeros(graph, bg_g, stream_desc);
    FillZeros(graph, bg_b, stream_desc);

    graph.recordOperation(pi::tensorlib::OperationEntry{
        .type = pi::tensorlib::OpType::CUSTOM_OP,
        .inputs = {grad_out_flat, cell_states_flat},
        .outputs = {fg_r, fg_g, fg_b, bg_r, bg_g, bg_b},
        .attributes = {{"custom_op_name", "build_cell_embed_bwd_color"}},
        .gpu_stream_desc = stream_desc,
    });

    graph.finalize();

    pi::tensorlib::ExecutionPlan plan = pi::tensorlib::ExecutionPlan::FromGraph(
        graph, {{.name = "cell_states", .tensor = cell_states_cpu}, {.name = "grad_out", .tensor = grad_out_cpu}}, {});
    ApplyPasses(plan);

    pi::tensorlib::Executor executor{plan, execution_backend, 0};
    executor.execute(allocator_registry);

    const auto fg_r_out = executor.getOutput(fg_r);
    if (!fg_r_out)
    {
        throw std::runtime_error("Missing fg_r output");
    }
    const auto expected_fg_r = MakeExpectedColorGrad(grad_out_cpu, cell_states_cpu, true, 0);
    pi::tensorlib::testing::AssertSimilar(expected_fg_r, *fg_r_out, 2e-2f);
    return 0;
}
