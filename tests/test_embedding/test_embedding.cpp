#include "testing.h"

#include <allocator.h>

#include <embedding.h>
#include <execution_backend.h>
#include <executor.h>
#include <passes.h>
#include <safe_tensors.h>
#include <tensorlib.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace
{
    constexpr uint32_t BATCH = 4;
    constexpr uint32_t SEQ_LEN = 64;
    constexpr uint32_t VOCAB_SIZE = 512;
    constexpr uint32_t EMBED_DIM = 128;

    constexpr float TOLERANCE = 1e-2f;

    constexpr uint32_t MODEL_INIT_SEED = 1337u;

    uint32_t ComputeIndex(const uint32_t linear_index) { return (linear_index * 17u + 11u) % VOCAB_SIZE; }

    void ApplyInitPasses(pi::tensorlib::ExecutionPlan &plan)
    {
        std::vector<std::unique_ptr<pi::tensorlib::passes::CompilerPass>> passes{};
        passes.emplace_back(std::make_unique<FillNormalImplPass>());
        pi::tensorlib::passes::Transform(plan, passes);
    }

    void ApplyMainPasses(pi::tensorlib::ExecutionPlan &plan)
    {
        std::vector<std::unique_ptr<pi::tensorlib::passes::CompilerPass>> passes{};
        passes.emplace_back(std::make_unique<GatherImplPass>());
        pi::tensorlib::passes::Transform(plan, passes);
    }
} // namespace

int main()
{
    pi::tensorlib::ExecutionBackend &execution_backend = pi::tensorlib::ExecutionBackend::getInstance();
    auto &allocator_registry = pi::tensorlib::allocator::DefaultAllocatorRegistry::instance();

    constexpr pi::tensorlib::Device device_gpu{
        .device_type = pi::tensorlib::DeviceType::GPU,
        .ordinal = 0,
    };
    constexpr pi::tensorlib::Device device_cpu{
        .device_type = pi::tensorlib::DeviceType::CPU,
        .ordinal = 0,
    };
    const auto main_stream_desc = pi::tensorlib::GpuStreamDescriptors::Main;

    pi::tensorlib::OpGraph init_graph{{}, {}};

    uint32_t init_seed = MODEL_INIT_SEED;
    pi::tensorlib::Embedding embedding_module{
        "embedding", VOCAB_SIZE, EMBED_DIM, device_gpu, pi::tensorlib::DataType::BFLOAT16, init_graph, init_seed,
        main_stream_desc};

    init_graph.finalize();

    pi::tensorlib::ExecutionPlan init_plan = pi::tensorlib::ExecutionPlan::FromGraph(init_graph, {}, {});
    ApplyInitPasses(init_plan);
    pi::tensorlib::Executor init_executor{init_plan, execution_backend, 0};
    init_executor.execute(allocator_registry);

    const auto embedding_params = embedding_module.parameters();

    std::vector<pi::tensorlib::GraphInputDescriptor> parameter_descriptors{};
    parameter_descriptors.reserve(embedding_params.size());
    for (const auto &[name, tensor] : embedding_params)
    {
        auto retained_tensor = tensor;
        retained_tensor.markRetained();
        parameter_descriptors.push_back(pi::tensorlib::GraphInputDescriptor{.name = name, .tensor = retained_tensor});
    }

    pi::tensorlib::TraceTensor indices_cpu =
        pi::tensorlib::TraceTensor::Create({BATCH, SEQ_LEN}, pi::tensorlib::DataType::UINT32, device_cpu,
                                            main_stream_desc);
    indices_cpu.markRetained();

    pi::tensorlib::OpGraph main_graph(
        {
            {.name = "indices", .tensor = indices_cpu},
        },
        parameter_descriptors);

    pi::tensorlib::TraceTensor indices_gpu =
        indices_cpu.to(main_graph, device_gpu, pi::tensorlib::GpuStreamDescriptors::Main);
    const pi::tensorlib::TraceTensor output = embedding_module.buildForward(main_graph, {indices_gpu});

    main_graph.finalize();

    std::vector<pi::tensorlib::GraphExecutionInputDescriptor> parameter_inputs{};
    parameter_inputs.reserve(embedding_params.size());
    for (const auto &[name, tensor] : embedding_params)
    {
        const auto real_opt = init_executor.getOutput(tensor);
        if (!real_opt)
        {
            throw std::runtime_error("Failed to retrieve parameter tensor after init pass");
        }
        parameter_inputs.push_back({.name = name, .tensor = *real_opt});
    }

    auto indices_real_cpu =
        pi::tensorlib::RealTensor::Allocate({BATCH, SEQ_LEN}, pi::tensorlib::DataType::UINT32, device_cpu);
    auto *indices_data = static_cast<uint32_t *>(indices_real_cpu->dataptr());
    constexpr uint64_t total_indices = static_cast<uint64_t>(BATCH) * static_cast<uint64_t>(SEQ_LEN);
    for (uint64_t i = 0; i < total_indices; ++i)
    {
        indices_data[i] = ComputeIndex(static_cast<uint32_t>(i));
    }

    pi::tensorlib::ExecutionPlan main_plan =
        pi::tensorlib::ExecutionPlan::FromGraph(main_graph,
                                                {
                                                    {.name = "indices", .tensor = indices_real_cpu},
                                                },
                                                parameter_inputs);

    ApplyMainPasses(main_plan);

    pi::tensorlib::Executor executor{main_plan, execution_backend, 0};
    executor.execute(allocator_registry);

    const auto output_real_opt = executor.getOutput(output);
    if (!output_real_opt)
    {
        throw std::runtime_error("Failed to retrieve embedding output tensor");
    }
    const auto &actual_output = *output_real_opt;

    const auto expected_tensors = pi::tensorlib::safetensors::Load("reference.safetensors");
    const auto expected_it = expected_tensors.find("output");
    if (expected_it == expected_tensors.end())
    {
        throw std::runtime_error("Expected tensor 'output' missing in reference.safetensors");
    }

    const auto &expected_output = expected_it->second;
    pi::tensorlib::testing::AssertSimilar(expected_output, actual_output, TOLERANCE);
}
