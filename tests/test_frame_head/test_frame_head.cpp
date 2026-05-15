#include "testing.h"

#include "../common/test_dtype_utils.h"
#include <allocator.h>
#include <config.h>
#include <execution_backend.h>
#include <executor.h>
#include <framehead_model.h>
#include <functional.h>
#include <passes.h>
#include <safe_tensors.h>
#include <tensorlib.h>

#include <iostream>

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    using namespace pi::tensorlib;

    constexpr uint32_t BATCH_SIZE = 32;
    constexpr uint32_t ROWS = 48;
    constexpr uint32_t COLS = 128;
    constexpr uint32_t MAX_CODE_POINT = 128;
    constexpr uint32_t NUM_HEADS = 8;
    constexpr uint32_t NUM_LAYERS = 1;
    constexpr uint32_t NUM_DOWNSAMPLE_BLOCKS = 2;
    constexpr uint32_t N_EMBD = 1024;
    constexpr float RMS_EPS = 1e-5f;
    constexpr uint32_t MODEL_INIT_SEED = 1337u;
    constexpr float TOLERANCE_BF16 = 1e-2f;
    constexpr float TOLERANCE_FP16 = 2e-2f;
    constexpr uint32_t NUM_CHANNELS_PER_CELL = 3;

    uint32_t ComputeCodepoint(const uint32_t index) { return (index * 13u + 7u) % MAX_CODE_POINT; }

    uint32_t ComputeColor(const uint32_t index, const uint32_t r_mul, const uint32_t r_bias, const uint32_t g_mul,
                          const uint32_t g_bias, const uint32_t b_mul, const uint32_t b_bias)
    {
        const uint32_t r = (index * r_mul + r_bias) % 256u;
        const uint32_t g = (index * g_mul + g_bias) % 256u;
        const uint32_t b = (index * b_mul + b_bias) % 256u;
        return (r << 16u) | (g << 8u) | b;
    }

    void PopulateCellStates(const std::shared_ptr<RealTensor> &tensor)
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

                    data[base + 0] = ComputeCodepoint(index);
                    data[base + 1] = ComputeColor(index, 17u, 5u, 19u, 11u, 23u, 13u);
                    data[base + 2] = ComputeColor(index, 29u, 17u, 31u, 19u, 37u, 23u);
                    ++index;
                }
            }
        }
    }

    void ApplyDefaultPasses(ExecutionPlan &execution_plan)
    {
        std::vector<std::unique_ptr<passes::CompilerPass>> passes{};
        passes.emplace_back(std::make_unique<MatmulFusePass>());
        passes.emplace_back(std::make_unique<MatmulImplPass>());
        passes.emplace_back(std::make_unique<MhaAttentionImplPass>());
        passes.emplace_back(std::make_unique<RmsNormImplPass>());
        passes.emplace_back(std::make_unique<AvgPool1dImplPass>());
        passes.emplace_back(std::make_unique<AvgPool2dImplPass>());
        passes.emplace_back(std::make_unique<AvgPool2dBwdImplPass>());
        passes.emplace_back(std::make_unique<Conv2dImplPass>());
        passes.emplace_back(std::make_unique<MeanImplPass>());
        passes.emplace_back(std::make_unique<GatherImplPass>());
        passes.emplace_back(std::make_unique<ContiguousImplPass>());
        passes.emplace_back(std::make_unique<TrailingBroadcastAddImplPass>());
        passes.emplace_back(std::make_unique<TrailingBroadcastMulImplPass>());
        passes.emplace_back(std::make_unique<ActImplPass>());
        passes.emplace_back(std::make_unique<CastImplPass>());
        passes.emplace_back(std::make_unique<FillZerosImplPass>());
        passes.emplace_back(std::make_unique<FillConstantImplPass>());
        passes.emplace_back(std::make_unique<FillUniformImplPass>());
        passes.emplace_back(std::make_unique<FillNormalImplPass>());
        passes.emplace_back(std::make_unique<BuildCellEmbedPass>());
        passes.emplace_back(std::make_unique<BuildCellEmbedBwdPass>());
        pi::tensorlib::passes::Transform(execution_plan, passes);
    }

    fbamtrain::FbamModelConfiguration BuildModelConfig(const bool use_bias)
    {
        return fbamtrain::FbamModelConfiguration{.frame_rows = ROWS,
                                                 .frame_cols = COLS,
                                                 .n_embed = N_EMBD,
                                                 .n_layer = NUM_LAYERS,
                                                 .n_head = NUM_HEADS,
                                                 .downsample_blocks = NUM_DOWNSAMPLE_BLOCKS,
                                                 .max_code_point = MAX_CODE_POINT,
                                                 .rms_norm_eps = RMS_EPS,
                                                 .bias = use_bias,
                                                 .downsample_conv_mode = "dilated",
                                                 .downsample_conv_dilation = 2,
                                                 .model_init_seed = MODEL_INIT_SEED};
    }

    std::shared_ptr<RealTensor> ExecuteFrameHeadVariant(const bool use_bias, ExecutionBackend &execution_backend,
                                                        allocator::DefaultAllocatorRegistry &allocator_registry,
                                                        const Device &device, const Device &device_cpu,
                                                        const std::shared_ptr<RealTensor> &cell_states_real,
                                                        const DataType dtype)
    {
        const auto main_stream_desc = GpuStreamDescriptors::Main;
        OpGraph init_graph{{}, {}};
        const auto model_config = BuildModelConfig(use_bias);

        fbamtrain::FrameHeadModule frame_head_module(model_config, init_graph, device, dtype, main_stream_desc);

        init_graph.finalize();

        ExecutionPlan init_plan = ExecutionPlan::FromGraph(init_graph, {}, {});
        ApplyDefaultPasses(init_plan);
        Executor init_executor{init_plan, execution_backend, 0};
        init_executor.execute(allocator_registry);

        const auto parameters = frame_head_module.parameters();

        std::vector<GraphInputDescriptor> parameter_descriptors{};
        parameter_descriptors.reserve(parameters.size());
        for (const auto &[name, tensor] : parameters)
        {
            auto retained_tensor = tensor;
            retained_tensor.markRetained();
            parameter_descriptors.push_back(GraphInputDescriptor{.name = name, .tensor = retained_tensor});
        }

        TraceTensor cell_states_cpu =
            TraceTensor::Create({BATCH_SIZE, ROWS, COLS, NUM_CHANNELS_PER_CELL}, DataType::UINT32, device_cpu,
                                main_stream_desc);
        cell_states_cpu.markRetained();

        std::vector<GraphInputDescriptor> input_descriptors{
            {.name = "cell_states", .tensor = cell_states_cpu},
        };

        OpGraph main_graph(input_descriptors, parameter_descriptors);

        TraceTensor cell_states_gpu = cell_states_cpu.to(main_graph, device, GpuStreamDescriptors::Main);
        TraceTensor output = frame_head_module.buildForward(main_graph, {cell_states_gpu}, false);

        main_graph.deleteTensor(cell_states_gpu);

        main_graph.finalize();

        std::vector<GraphExecutionInputDescriptor> parameter_execution_inputs{};
        parameter_execution_inputs.reserve(parameters.size());
        for (const auto &[name, tensor] : parameters)
        {
            const auto tensor_real_opt = init_executor.getOutput(tensor);
            if (!tensor_real_opt)
            {
                throw std::runtime_error("Parameter tensor not found after init graph execution for " + name);
            }
            parameter_execution_inputs.push_back({.name = name, .tensor = *tensor_real_opt});
        }

        std::vector<GraphExecutionInputDescriptor> graph_inputs{
            {.name = "cell_states", .tensor = cell_states_real},
        };

        ExecutionPlan main_plan = ExecutionPlan::FromGraph(main_graph, graph_inputs, parameter_execution_inputs);
        ApplyDefaultPasses(main_plan);

        if (const char *debug_plan = std::getenv("DEBUG_FRAME_HEAD_PLAN"); debug_plan != nullptr)
        {
            std::cerr << "Execution plan entries (" << main_plan.entries.size() << "):\n";
            for (const auto &entry : main_plan.entries)
            {
                std::cerr << "  Entry " << entry.id << ": ";
                if (entry.kernel_descriptor.has_value())
                {
                    std::cerr << "kernel=" << entry.kernel_descriptor->kernel_name;
                }
                else if (entry.op_type.has_value())
                {
                    std::cerr << "op=" << static_cast<int>(*entry.op_type);
                }
                else
                {
                    std::cerr << "op=<none>";
                }
                std::cerr << " inputs=" << entry.inputs.size() << " outputs=" << entry.outputs.size() << '\n';
            }
        }

        Executor executor{main_plan, execution_backend, 0};
        executor.execute(allocator_registry);

        const auto output_real_opt = executor.getOutput(output);
        if (!output_real_opt)
        {
            throw std::runtime_error("Output tensor not found after main graph execution");
        }
        return *output_real_opt;
    }
} // namespace

int main()
{
    using namespace pi::tensorlib;

    ExecutionBackend &execution_backend = ExecutionBackend::getInstance();
    auto &allocator_registry = allocator::DefaultAllocatorRegistry::instance();

    const Device device{
        .device_type = DeviceType::GPU,
        .ordinal = 0,
    };

    constexpr Device device_cpu{
        .device_type = DeviceType::CPU,
        .ordinal = 0,
    };

    const auto dtype = test_utils::GetTestDtype();

    auto cell_states_real = RealTensor::Allocate(
        {BATCH_SIZE, ROWS, COLS, NUM_CHANNELS_PER_CELL}, DataType::UINT32, device_cpu);
    PopulateCellStates(cell_states_real);

    const std::string suffix(test_utils::GetDtypeSuffix(dtype));
    const std::string reference_path = test_utils::ReferenceFileName(dtype);
    const auto expected_tensors = safetensors::Load(reference_path);

    const float tolerance = test_utils::SelectTolerance(dtype, TOLERANCE_BF16, TOLERANCE_FP16);

    struct Variant
    {
        bool use_bias;
        const char *tensor_name;
    };
    constexpr std::array<Variant, 2> variants{{{true, "output_bias_true"}, {false, "output_bias_false"}}};

    std::map<std::string, std::shared_ptr<RealTensor>> actual_tensors{};

    for (const auto &variant : variants)
    {
        auto actual_output = ExecuteFrameHeadVariant(
            variant.use_bias, execution_backend, allocator_registry, device, device_cpu, cell_states_real, dtype);
        actual_tensors.emplace(variant.tensor_name, actual_output);

        const auto expected_it = expected_tensors.find(variant.tensor_name);
        if (expected_it == expected_tensors.end())
        {
            throw std::runtime_error("Expected output tensor not found in reference file: " + reference_path +
                                     " (missing key: " + std::string(variant.tensor_name) + ")");
        }

        try
        {
            testing::AssertSimilar(expected_it->second, actual_output, tolerance);
        }
        catch (const std::exception &ex)
        {
            std::cerr << "Comparison failed for variant '" << variant.tensor_name << "': " << ex.what() << '\n';
            std::map<std::string, std::shared_ptr<RealTensor>> debug_dump{{variant.tensor_name, actual_output}};
            const std::string debug_path = std::string("debug_") + variant.tensor_name + "_" + std::string(suffix) + ".safetensors";
            safetensors::SaveToFile(debug_path, debug_dump);
            throw;
        }
    }

    const std::string actual_path = "actual_" + suffix + ".safetensors";
    safetensors::SaveToFile(actual_path, actual_tensors);

    return 0;
}
