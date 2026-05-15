#include "testing.h"

#include <allocator.h>
#include <conv2d.h>
#include <execution_backend.h>
#include <executor.h>
#include <functional.h>
#include <passes.h>
#include <safe_tensors.h>
#include <tensorlib.h>

#include <array>
#include <memory>
#include <stdexcept>
#include <vector>

namespace
{
    constexpr uint32_t BATCH = 32;
    constexpr uint32_t IN_CHANNELS = 32;
    constexpr uint32_t OUT_CHANNELS = 64;
    constexpr uint32_t HEIGHT = 48;
    constexpr uint32_t WIDTH = 160;
    constexpr uint32_t KERNEL_SIZE = 3;
    constexpr uint32_t STRIDE = 1;
    constexpr uint32_t PADDING = 1;
    constexpr uint32_t DILATION = 1;
    constexpr float INPUT_LOW = -0.5f;
    constexpr float INPUT_HIGH = 0.5f;
    constexpr uint32_t INPUT_SEED = 2024;
    constexpr uint32_t INIT_SEED = 1337;
    constexpr float TOLERANCE = 2e-3f;

    void ApplyDefaultPasses(pi::tensorlib::ExecutionPlan &execution_plan)
    {
        std::vector<std::unique_ptr<pi::tensorlib::passes::CompilerPass>> passes{};
        passes.emplace_back(std::make_unique<Conv2dImplPass>());
        passes.emplace_back(std::make_unique<ContiguousImplPass>());
        passes.emplace_back(std::make_unique<FillZerosImplPass>());
        passes.emplace_back(std::make_unique<FillConstantImplPass>());
        passes.emplace_back(std::make_unique<FillUniformImplPass>());
        pi::tensorlib::passes::Transform(execution_plan, passes);
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
    pi::tensorlib::TraceTensor input = init_graph.createTensor({BATCH, IN_CHANNELS, HEIGHT, WIDTH},
                                                               pi::tensorlib::DataType::BFLOAT16, device,
                                                               main_stream_desc, false);
    pi::tensorlib::FillUniform(init_graph, input, INPUT_LOW, INPUT_HIGH, INPUT_SEED, main_stream_desc);
    input = input.transpose(init_graph, {0, 2, 3, 1}).contiguous(init_graph, main_stream_desc); // NCHW to NHWC

    uint32_t init_seed = INIT_SEED;
    struct pi::tensorlib::Conv2d conv("conv", IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, STRIDE, PADDING, DILATION, device,
                                      pi::tensorlib::DataType::BFLOAT16, init_graph, init_seed, main_stream_desc,
                                      false);

    init_graph.finalize();

    pi::tensorlib::ExecutionPlan init_plan = pi::tensorlib::ExecutionPlan::FromGraph(init_graph, {}, {});
    ApplyDefaultPasses(init_plan);

    pi::tensorlib::Executor init_executor{init_plan, execution_backend, 0};
    init_executor.execute(allocator_registry);

    const auto input_real_opt = init_executor.getOutput(input);
    if (!input_real_opt.has_value())
    {
        throw std::runtime_error("Failed to retrieve initialized input tensor");
    }
    const auto &input_real = *input_real_opt;

    std::vector<pi::tensorlib::GraphExecutionInputDescriptor> parameter_execution_inputs{};
    parameter_execution_inputs.reserve(conv.parameters().size());
    std::vector<pi::tensorlib::GraphInputDescriptor> parameter_descriptors{};
    parameter_descriptors.reserve(conv.parameters().size());

    for (const auto &[name, tensor] : conv.parameters())
    {
        const auto tensor_real_opt = init_executor.getOutput(tensor);
        if (!tensor_real_opt.has_value())
        {
            throw std::runtime_error("Failed to retrieve parameter tensor " + name);
        }
        parameter_execution_inputs.push_back(pi::tensorlib::GraphExecutionInputDescriptor{name, *tensor_real_opt});
        auto retained_tensor = tensor;
        retained_tensor.markRetained();
        parameter_descriptors.push_back(pi::tensorlib::GraphInputDescriptor{name, retained_tensor});
    }

    input.markRetained();
    pi::tensorlib::OpGraph graph{{
                                     {.name = "input", .tensor = input},
                                 },
                                 parameter_descriptors};

    pi::tensorlib::TraceTensor output = conv.buildForward(graph, {input});
    graph.finalize();

    std::vector<pi::tensorlib::GraphExecutionInputDescriptor> graph_inputs;
    graph_inputs.emplace_back(pi::tensorlib::GraphExecutionInputDescriptor{"input", input_real});

    pi::tensorlib::ExecutionPlan plan =
        pi::tensorlib::ExecutionPlan::FromGraph(graph, graph_inputs, parameter_execution_inputs);
    ApplyDefaultPasses(plan);

    pi::tensorlib::Executor executor{plan, execution_backend, 0};
    executor.execute(allocator_registry);

    const auto actual_output_opt = executor.getOutput(output);
    if (!actual_output_opt.has_value())
    {
        throw std::runtime_error("Failed to retrieve conv2d output tensor");
    }
    const auto &actual_output = *actual_output_opt;

    const auto expected_tensors = pi::tensorlib::safetensors::Load("reference.safetensors");
    const auto expected_it = expected_tensors.find("output");
    if (expected_it == expected_tensors.end())
    {
        throw std::runtime_error("Expected conv2d output tensor not found in reference.safetensors");
    }
    const auto &expected_output = expected_it->second;

    pi::tensorlib::testing::AssertSimilar(expected_output, actual_output, TOLERANCE);
    return 0;
}
