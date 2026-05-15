#include "testing.h"

#include "mlp.h"

#include <allocator.h>

#include <execution_backend.h>
#include <executor.h>
#include <mlp.h>
#include <passes.h>
#include <safe_tensors.h>
#include <tensorlib.h>

#include <memory>
#include <vector>

namespace
{
    constexpr uint32_t BATCH_SIZE = 32;
    constexpr uint32_t N_EMBD = 16;
    constexpr uint32_t HIDDEN_DIM = 4 * N_EMBD;
    constexpr uint32_t MODEL_INIT_SEED = 42u;

    constexpr float INPUT_LOW = -0.5f;
    constexpr float INPUT_HIGH = 0.5f;
    constexpr float TOLERANCE = 2e-3f;

    void ApplyDefaultPasses(pi::tensorlib::ExecutionPlan &execution_plan)
    {
        std::vector<std::unique_ptr<pi::tensorlib::passes::CompilerPass>> passes{};
        passes.emplace_back(std::make_unique<MatmulFusePass>());
        passes.emplace_back(std::make_unique<ActImplPass>());
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

    uint32_t rng_seed = MODEL_INIT_SEED;

    pi::tensorlib::MlpBlock mlp_block{"mlp",      N_EMBD,  HIDDEN_DIM, device, pi::tensorlib::DataType::BFLOAT16,
                                      init_graph, rng_seed, main_stream_desc};

    pi::tensorlib::TraceTensor input =
        init_graph.createTensor({BATCH_SIZE, N_EMBD}, pi::tensorlib::DataType::BFLOAT16, device, main_stream_desc, false);
    pi::tensorlib::FillUniform(init_graph, input, INPUT_LOW, INPUT_HIGH, rng_seed++, main_stream_desc);
    init_graph.finalize();

    pi::tensorlib::ExecutionPlan init_plan = pi::tensorlib::ExecutionPlan::FromGraph(init_graph, {}, {});
    ApplyDefaultPasses(init_plan);
    pi::tensorlib::Executor init_executor{init_plan, execution_backend, 0};
    init_executor.execute(allocator_registry);

    const auto block_parameters = mlp_block.parameters();

    std::vector<pi::tensorlib::GraphInputDescriptor> parameter_descriptors{};
    parameter_descriptors.reserve(block_parameters.size());
    for (const auto &[name, tensor] : block_parameters)
    {
        auto retained_tensor = tensor;
        retained_tensor.markRetained();
        parameter_descriptors.push_back(pi::tensorlib::GraphInputDescriptor{.name = name, .tensor = retained_tensor});
    }

    input.markRetained();
    pi::tensorlib::OpGraph main_graph(
        {
            {.name = "x", .tensor = input},
        },
        parameter_descriptors);

    const pi::tensorlib::TraceTensor output = mlp_block.buildForward(main_graph, {input}, false);
    main_graph.finalize();

    std::vector<pi::tensorlib::GraphExecutionInputDescriptor> parameter_real_descriptors{};
    parameter_real_descriptors.reserve(block_parameters.size());
    for (const auto &[name, tensor] : block_parameters)
    {
        const auto tensor_real_opt = init_executor.getOutput(tensor);
        if (!tensor_real_opt)
        {
            throw std::runtime_error("Parameter tensor not found after init graph execution");
        }
        parameter_real_descriptors.push_back({.name = name, .tensor = *tensor_real_opt});
    }

    const auto input_real_opt = init_executor.getOutput(input);
    if (!input_real_opt)
    {
        throw std::runtime_error("Input tensor not found after init graph execution");
    }

    pi::tensorlib::ExecutionPlan main_plan =
        pi::tensorlib::ExecutionPlan::FromGraph(main_graph,
                                                {
                                                    {.name = "x", .tensor = *input_real_opt},
                                                },
                                                parameter_real_descriptors);

    ApplyDefaultPasses(main_plan);

    pi::tensorlib::Executor executor{main_plan, execution_backend, 0};
    executor.execute(allocator_registry);

    const auto output_real_opt = executor.getOutput(output);
    if (!output_real_opt)
    {
        throw std::runtime_error("Output tensor not found after main graph execution");
    }
    const auto &actual_output = *output_real_opt;

    const auto expected_tensors = pi::tensorlib::safetensors::Load("reference.safetensors");
    const auto expected_output_it = expected_tensors.find("output");
    if (expected_output_it == expected_tensors.end())
    {
        throw std::runtime_error("Expected output tensor not found in reference.safetensors");
    }
    const auto &expected_output = expected_output_it->second;

    pi::tensorlib::testing::AssertSimilar(expected_output, actual_output, TOLERANCE);
}
