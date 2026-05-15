#include "testing.h"

#include <allocator.h>

#include <execution_backend.h>
#include <executor.h>
#include <functional.h>
#include <passes.h>
#include <tensorlib.h>

#include <stdexcept>

using namespace pi::tensorlib;

namespace
{
    void ApplyDefaultPasses(ExecutionPlan &execution_plan)
    {
        std::vector<std::unique_ptr<passes::CompilerPass>> passes{};
        passes.emplace_back(std::make_unique<TrailingBroadcastAddImplPass>());
        passes.emplace_back(std::make_unique<FillZerosImplPass>());
        passes.emplace_back(std::make_unique<FillConstantImplPass>());
        passes.emplace_back(std::make_unique<FillUniformImplPass>());
        passes::Transform(execution_plan, passes);
    }

    std::shared_ptr<RealTensor> ComputeElementwiseReference(const std::shared_ptr<RealTensor> &lhs,
                                                            const std::shared_ptr<RealTensor> &rhs,
                                                            const Device device_cpu)
    {
        constexpr int64_t NUMEL = 256 * 128;

        auto expected = RealTensor::Allocate({NUMEL}, DataType::FLOAT32, device_cpu);

        const auto lhs_cpu = lhs->to(device_cpu, GpuStreamDescriptors::Main);
        const auto rhs_cpu = rhs->to(device_cpu, GpuStreamDescriptors::Main);

        const auto lhs_stride = lhs_cpu->strides()[0];
        const auto rhs_stride = rhs_cpu->strides()[0];
        const auto expected_stride = expected->strides()[0];

        const auto *lhs_ptr = static_cast<const float *>(lhs_cpu->dataptr());
        const auto *rhs_ptr = static_cast<const float *>(rhs_cpu->dataptr());
        auto *expected_ptr = static_cast<float *>(expected->dataptr());

        for (int64_t idx = 0; idx < NUMEL; ++idx)
        {
            expected_ptr[static_cast<size_t>(idx * expected_stride)] =
                lhs_ptr[static_cast<size_t>(idx * lhs_stride)] + rhs_ptr[static_cast<size_t>(idx * rhs_stride)];
        }

        return expected;
    }

    std::shared_ptr<RealTensor> ComputeTrailingBroadcastReference(const std::shared_ptr<RealTensor> &activation,
                                                                  const std::shared_ptr<RealTensor> &bias,
                                                                  const Device device_cpu)
    {
        constexpr int64_t ROWS = 256;
        constexpr int64_t COLS = 128;

        auto expected = RealTensor::Allocate({ROWS, COLS}, DataType::FLOAT32, device_cpu);

        const auto activation_cpu = activation->to(device_cpu, pi::tensorlib::GpuStreamDescriptors::Main);
        const auto bias_cpu = bias->to(device_cpu, pi::tensorlib::GpuStreamDescriptors::Main);

        const auto activation_strides = activation_cpu->strides();
        const auto bias_stride = bias_cpu->shape().ndims() == 1 ? bias_cpu->strides()[0] : bias_cpu->strides()[1];
        const auto expected_strides = expected->strides();

        const auto *activation_ptr = static_cast<const float *>(activation_cpu->dataptr());
        const auto *bias_ptr = static_cast<const float *>(bias_cpu->dataptr());
        auto *expected_ptr = static_cast<float *>(expected->dataptr());

        for (int64_t row = 0; row < ROWS; ++row)
        {
            for (int64_t col = 0; col < COLS; ++col)
            {
                const auto in_offset = static_cast<size_t>(row * activation_strides[0] + col * activation_strides[1]);
                const auto out_offset = static_cast<size_t>(row * expected_strides[0] + col * expected_strides[1]);
                expected_ptr[out_offset] = activation_ptr[in_offset] + bias_ptr[static_cast<size_t>(col * bias_stride)];
            }
        }

        return expected;
    }
} // namespace

int main()
{
    constexpr float TOLERANCE_FP32 = 2e-4f;
    constexpr int64_t ROWS = 256;
    constexpr int64_t COLS = 128;
    constexpr int64_t NUMEL = ROWS * COLS;

    ExecutionBackend &execution_backend = ExecutionBackend::getInstance();
    auto &allocator_registry = allocator::DefaultAllocatorRegistry::instance();

    constexpr Device DEVICE_GPU{DeviceType::GPU, 0};
    constexpr Device DEVICE_CPU{DeviceType::CPU, 0};
    const auto main_stream_desc = GpuStreamDescriptors::Main;

    OpGraph init_graph{{}, {}};

    TraceTensor activation = init_graph.createTensor({ROWS, COLS}, DataType::FLOAT32, DEVICE_GPU, main_stream_desc, false);
    TraceTensor bias = init_graph.createTensor({1, COLS}, DataType::FLOAT32, DEVICE_GPU, main_stream_desc, false);
    TraceTensor elem_lhs = init_graph.createTensor({NUMEL}, DataType::FLOAT32, DEVICE_GPU, main_stream_desc, false);
    TraceTensor elem_rhs = init_graph.createTensor({NUMEL}, DataType::FLOAT32, DEVICE_GPU, main_stream_desc, false);
    activation.markRetained();
    bias.markRetained();
    elem_lhs.markRetained();
    elem_rhs.markRetained();

    uint32_t seed = 123;
    FillUniform(init_graph, activation, -0.5f, 0.5f, seed++, main_stream_desc);
    FillUniform(init_graph, bias, -0.5f, 0.5f, seed++, main_stream_desc);
    FillUniform(init_graph, elem_lhs, -0.5f, 0.5f, seed++, main_stream_desc);
    FillUniform(init_graph, elem_rhs, -0.5f, 0.5f, seed++, main_stream_desc);

    init_graph.finalize();

    ExecutionPlan init_plan = ExecutionPlan::FromGraph(init_graph, {}, {});
    ApplyDefaultPasses(init_plan);
    Executor init_executor{init_plan, execution_backend, 0};
    init_executor.execute(allocator_registry);

    const auto activation_real = init_executor.getOutput(activation);
    const auto bias_real = init_executor.getOutput(bias);
    const auto elem_lhs_real = init_executor.getOutput(elem_lhs);
    const auto elem_rhs_real = init_executor.getOutput(elem_rhs);
    if (!activation_real || !bias_real || !elem_lhs_real || !elem_rhs_real)
    {
        throw std::runtime_error("Failed to retrieve initialization tensors");
    }

    const auto expected_trailing_ref = ComputeTrailingBroadcastReference(*activation_real, *bias_real, DEVICE_CPU);
    const auto expected_elementwise = ComputeElementwiseReference(*elem_lhs_real, *elem_rhs_real, DEVICE_CPU);

    OpGraph graph{{
                      {.name = "activation", .tensor = activation},
                      {.name = "bias", .tensor = bias},
                      {.name = "elem_lhs", .tensor = elem_lhs},
                      {.name = "elem_rhs", .tensor = elem_rhs},
                  },
                  {}};

    InplaceAdd(graph, activation, bias, main_stream_desc);
    InplaceAdd(graph, elem_lhs, elem_rhs, main_stream_desc);

    graph.finalize();

    ExecutionPlan plan = ExecutionPlan::FromGraph(graph,
                                                  {{.name = "activation", .tensor = *activation_real},
                                                   {.name = "bias", .tensor = *bias_real},
                                                   {.name = "elem_lhs", .tensor = *elem_lhs_real},
                                                   {.name = "elem_rhs", .tensor = *elem_rhs_real}},
                                                  {});

    ApplyDefaultPasses(plan);

    Executor executor{plan, execution_backend, 0};
    executor.execute(allocator_registry);

    const auto trailing_output = executor.getOutput(activation);
    const auto elementwise_output = executor.getOutput(elem_lhs);
    if (!trailing_output || !elementwise_output)
    {
        throw std::runtime_error("Expected output tensors not found");
    }

    testing::AssertSimilar(expected_trailing_ref, *trailing_output, TOLERANCE_FP32);
    testing::AssertSimilar(expected_elementwise, *elementwise_output, TOLERANCE_FP32);
    return 0;
}
