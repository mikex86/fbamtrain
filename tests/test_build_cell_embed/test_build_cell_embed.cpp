#include "testing.h"

#include <allocator.h>

#include <execution_backend.h>
#include <executor.h>
#include <functional.h>
#include <passes.h>
#include <safe_tensors.h>
#include <tensorlib.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    constexpr uint32_t BATCH_SIZE = 2;
    constexpr uint32_t FRAME_ROWS = 2;
    constexpr uint32_t FRAME_COLS = 3;
    constexpr uint32_t EMBED_DIM = 8;
    constexpr uint32_t MODEL_INIT_SEED = 123u;

    constexpr uint32_t NUM_CHANNELS_PER_CELL = 3;
    constexpr uint32_t MAX_CODEPOINT = 255;

    uint32_t ComputeCodepoint(const uint32_t index) { return (index * 13u + 7u) % MAX_CODEPOINT; }

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
            for (uint32_t row = 0; row < FRAME_ROWS; ++row)
            {
                for (uint32_t col = 0; col < FRAME_COLS; ++col)
                {
                    const uint64_t base =
                        (static_cast<uint64_t>(batch) * FRAME_ROWS * FRAME_COLS + row * FRAME_COLS + col) *
                        NUM_CHANNELS_PER_CELL;

                    data[base + 0] = ComputeCodepoint(index);
                    data[base + 1] = ComputeColor(index, 17u, 5u, 19u, 11u, 23u, 13u);
                    data[base + 2] = ComputeColor(index, 29u, 17u, 31u, 19u, 37u, 23u);
                    ++index;
                }
            }
        }
    }

    void ApplyInitPasses(pi::tensorlib::ExecutionPlan &plan)
    {
        std::vector<std::unique_ptr<pi::tensorlib::passes::CompilerPass>> passes;
        passes.emplace_back(std::make_unique<FillNormalImplPass>());
        pi::tensorlib::passes::Transform(plan, passes);
    }

    void ApplyMainPasses(pi::tensorlib::ExecutionPlan &plan)
    {
        std::vector<std::unique_ptr<pi::tensorlib::passes::CompilerPass>> passes;
        passes.emplace_back(std::make_unique<FillNormalImplPass>());
        passes.emplace_back(std::make_unique<BuildCellEmbedPass>());
        pi::tensorlib::passes::Transform(plan, passes);
    }
} // namespace

int main()
{
    pi::tensorlib::ExecutionBackend &execution_backend = pi::tensorlib::ExecutionBackend::getInstance();
    auto &allocator_registry = pi::tensorlib::allocator::DefaultAllocatorRegistry::instance();

    constexpr pi::tensorlib::Device device_gpu{.device_type = pi::tensorlib::DeviceType::GPU, .ordinal = 0};
    constexpr pi::tensorlib::Device device_cpu{.device_type = pi::tensorlib::DeviceType::CPU, .ordinal = 0};
    const auto stream_desc = pi::tensorlib::GpuStreamDescriptors::Main;

    pi::tensorlib::OpGraph init_graph{{}, {}};

    uint32_t init_seed = MODEL_INIT_SEED;

    auto codepoint_embed =
        init_graph.createTensor({MAX_CODEPOINT, EMBED_DIM}, pi::tensorlib::DataType::BFLOAT16, device_gpu, stream_desc,
                                false);
    auto position_embed = init_graph.createTensor({FRAME_ROWS * FRAME_COLS, EMBED_DIM},
                                                  pi::tensorlib::DataType::BFLOAT16, device_gpu, stream_desc, false);
    auto fg_r_embed = init_graph.createTensor({EMBED_DIM}, pi::tensorlib::DataType::BFLOAT16, device_gpu, stream_desc,
                                              false);
    auto fg_g_embed = init_graph.createTensor({EMBED_DIM}, pi::tensorlib::DataType::BFLOAT16, device_gpu, stream_desc,
                                              false);
    auto fg_b_embed = init_graph.createTensor({EMBED_DIM}, pi::tensorlib::DataType::BFLOAT16, device_gpu, stream_desc,
                                              false);
    auto bg_r_embed = init_graph.createTensor({EMBED_DIM}, pi::tensorlib::DataType::BFLOAT16, device_gpu, stream_desc,
                                              false);
    auto bg_g_embed = init_graph.createTensor({EMBED_DIM}, pi::tensorlib::DataType::BFLOAT16, device_gpu, stream_desc,
                                              false);
    auto bg_b_embed = init_graph.createTensor({EMBED_DIM}, pi::tensorlib::DataType::BFLOAT16, device_gpu, stream_desc,
                                              false);

    pi::tensorlib::FillNormal(init_graph, codepoint_embed, 0.0f, 1.0f, init_seed++, stream_desc);
    pi::tensorlib::FillNormal(init_graph, position_embed, 0.0f, 1.0f, init_seed++, stream_desc);
    pi::tensorlib::FillNormal(init_graph, bg_r_embed, 0.0f, 1.0f, init_seed++, stream_desc);
    pi::tensorlib::FillNormal(init_graph, bg_g_embed, 0.0f, 1.0f, init_seed++, stream_desc);
    pi::tensorlib::FillNormal(init_graph, bg_b_embed, 0.0f, 1.0f, init_seed++, stream_desc);
    pi::tensorlib::FillNormal(init_graph, fg_r_embed, 0.0f, 1.0f, init_seed++, stream_desc);
    pi::tensorlib::FillNormal(init_graph, fg_g_embed, 0.0f, 1.0f, init_seed++, stream_desc);
    pi::tensorlib::FillNormal(init_graph, fg_b_embed, 0.0f, 1.0f, init_seed++, stream_desc);

    init_graph.finalize();

    pi::tensorlib::ExecutionPlan init_plan = pi::tensorlib::ExecutionPlan::FromGraph(init_graph, {}, {});
    ApplyInitPasses(init_plan);
    pi::tensorlib::Executor init_executor{init_plan, execution_backend, 0};
    init_executor.execute(allocator_registry);

    std::vector<pi::tensorlib::GraphInputDescriptor> parameter_descriptors{};
    parameter_descriptors.reserve(8);
    const auto add_parameter_descriptor =
        [&parameter_descriptors](const std::string &name, const pi::tensorlib::TraceTensor &tensor)
    {
        auto retained_tensor = tensor;
        retained_tensor.markRetained();
        parameter_descriptors.push_back({.name = name, .tensor = retained_tensor});
    };
    add_parameter_descriptor("codepoint_embed", codepoint_embed);
    add_parameter_descriptor("position_embed", position_embed);
    add_parameter_descriptor("fg_r_embed", fg_r_embed);
    add_parameter_descriptor("fg_g_embed", fg_g_embed);
    add_parameter_descriptor("fg_b_embed", fg_b_embed);
    add_parameter_descriptor("bg_r_embed", bg_r_embed);
    add_parameter_descriptor("bg_g_embed", bg_g_embed);
    add_parameter_descriptor("bg_b_embed", bg_b_embed);

    pi::tensorlib::TraceTensor cell_states_cpu_trace = pi::tensorlib::TraceTensor::Create(
        {BATCH_SIZE, FRAME_ROWS, FRAME_COLS, NUM_CHANNELS_PER_CELL}, pi::tensorlib::DataType::UINT32, device_cpu,
        stream_desc);
    cell_states_cpu_trace.markRetained();

    pi::tensorlib::OpGraph main_graph(
        {
            {.name = "cell_states", .tensor = cell_states_cpu_trace},
        },
        parameter_descriptors);

    const pi::tensorlib::TraceTensor cell_states_gpu_trace =
        cell_states_cpu_trace.to(main_graph, device_gpu, pi::tensorlib::GpuStreamDescriptors::Main);
    const pi::tensorlib::TraceTensor cell_embeddings_trace = main_graph.createTensor(
        {BATCH_SIZE, FRAME_ROWS, FRAME_COLS, EMBED_DIM}, pi::tensorlib::DataType::BFLOAT16, device_gpu, stream_desc,
        false);

    const auto cell_embeddings_flat =
        cell_embeddings_trace.viewInferred(main_graph, {-1, static_cast<int64_t>(EMBED_DIM)});
    const auto cell_states_flat =
        cell_states_gpu_trace.viewInferred(main_graph, {-1, static_cast<int64_t>(NUM_CHANNELS_PER_CELL)});

    main_graph.recordOperation(pi::tensorlib::OperationEntry{
        .type = pi::tensorlib::OpType::CUSTOM_OP,
        .inputs = {cell_embeddings_flat, cell_states_flat, codepoint_embed, position_embed, fg_r_embed, fg_g_embed,
                   fg_b_embed, bg_r_embed, bg_g_embed, bg_b_embed},
        .outputs = {cell_embeddings_flat},
        .attributes = {{"custom_op_name", "build_cell_embed"}},
        .gpu_stream_desc = stream_desc,
    });
    main_graph.finalize();

    std::vector<pi::tensorlib::GraphExecutionInputDescriptor> parameter_inputs;
    parameter_inputs.reserve(parameter_descriptors.size());

    const auto add_parameter_input =
        [&parameter_inputs, &init_executor](const std::string &name, const pi::tensorlib::TraceTensor &tensor)
    {
        const auto tensor_real_opt = init_executor.getOutput(tensor);
        if (!tensor_real_opt)
        {
            throw std::runtime_error("Parameter tensor not found after init execution for " + name);
        }
        parameter_inputs.push_back({.name = name, .tensor = *tensor_real_opt});
    };

    add_parameter_input("codepoint_embed", codepoint_embed);
    add_parameter_input("position_embed", position_embed);
    add_parameter_input("fg_r_embed", fg_r_embed);
    add_parameter_input("fg_g_embed", fg_g_embed);
    add_parameter_input("fg_b_embed", fg_b_embed);
    add_parameter_input("bg_r_embed", bg_r_embed);
    add_parameter_input("bg_g_embed", bg_g_embed);
    add_parameter_input("bg_b_embed", bg_b_embed);

    auto cell_states_real = pi::tensorlib::RealTensor::Allocate(
        {BATCH_SIZE, FRAME_ROWS, FRAME_COLS, NUM_CHANNELS_PER_CELL}, pi::tensorlib::DataType::UINT32, device_cpu);
    PopulateCellStates(cell_states_real);

    pi::tensorlib::ExecutionPlan main_plan = pi::tensorlib::ExecutionPlan::FromGraph(
        main_graph, {{.name = "cell_states", .tensor = cell_states_real}}, parameter_inputs);
    ApplyMainPasses(main_plan);

    pi::tensorlib::Executor executor{main_plan, execution_backend, 0};
    executor.execute(allocator_registry);

    const auto output_real_opt = executor.getOutput(cell_embeddings_trace);
    if (!output_real_opt)
    {
        throw std::runtime_error("Failed to retrieve output tensor");
    }
    const auto &actual_output = *output_real_opt;

    const auto expected_tensors = pi::tensorlib::safetensors::Load("reference.safetensors");
    const auto expected_it = expected_tensors.find("output");
    if (expected_it == expected_tensors.end())
    {
        throw std::runtime_error("Expected tensor 'output' missing in reference.safetensors");
    }

    const auto &expected_output = expected_it->second;
    pi::tensorlib::testing::AssertSimilar(expected_output, actual_output, 2e-3f);
}
