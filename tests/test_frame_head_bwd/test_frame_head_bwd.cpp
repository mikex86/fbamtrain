#include "testing.h"

#include "../common/test_dtype_utils.h"
#include <allocator.h>
#include <execution_backend.h>
#include <executor.h>
#include <framehead_model.h>
#include <functional.h>
#include <passes.h>
#include <safe_tensors.h>
#include <tensorlib.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <utils.h>

namespace
{
    using namespace pi::tensorlib;

    constexpr uint32_t BATCH_SIZE = 2;
    constexpr uint32_t ROWS = 8;
    constexpr uint32_t COLS = 8;
    constexpr uint32_t MAX_CODE_POINT = 128;
    constexpr uint32_t NUM_HEADS = 8;
    constexpr uint32_t NUM_LAYERS = 1;
    constexpr uint32_t NUM_DOWNSAMPLE_BLOCKS = 2;
    constexpr uint32_t N_EMBD = 1024;
    constexpr float RMS_EPS = 1e-5f;
    constexpr uint32_t MODEL_INIT_SEED = 1337u;
    constexpr float TOLERANCE_BF16 = 1e-1f;
    constexpr float TOLERANCE_FP16 = 3e-2f;
    constexpr uint32_t NUM_CHANNELS_PER_CELL = 3;

    constexpr Device DEVICE_GPU{DeviceType::GPU, 0};
    constexpr Device DEVICE_CPU{DeviceType::CPU, 0};
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
        passes.emplace_back(std::make_unique<FillZerosImplPass>());
        passes.emplace_back(std::make_unique<FillConstantImplPass>());
        passes.emplace_back(std::make_unique<FillUniformImplPass>());
        passes.emplace_back(std::make_unique<FillNormalImplPass>());
        passes.emplace_back(std::make_unique<LayerNormImplPass>());
        passes.emplace_back(std::make_unique<RmsNormImplPass>());
        passes.emplace_back(std::make_unique<AvgPool1dImplPass>());
        passes.emplace_back(std::make_unique<AvgPool2dImplPass>());
        passes.emplace_back(std::make_unique<AvgPool2dBwdImplPass>());
        passes.emplace_back(std::make_unique<Conv2dImplPass>());
        passes.emplace_back(std::make_unique<CastImplPass>());
        passes.emplace_back(std::make_unique<MeanImplPass>());
        passes.emplace_back(std::make_unique<FuseMulReducePass>());
        passes.emplace_back(std::make_unique<ReduceSumImplPass>());
        passes.emplace_back(std::make_unique<GatherImplPass>());
        passes.emplace_back(std::make_unique<DivAddFusePass>());
        passes.emplace_back(std::make_unique<TrailingBroadcastAddImplPass>());
        passes.emplace_back(std::make_unique<TrailingBroadcastMulImplPass>());
        passes.emplace_back(std::make_unique<DivImplPass>());
        passes.emplace_back(std::make_unique<ActImplPass>());
        passes.emplace_back(std::make_unique<LstmCellImplPass>());
        passes.emplace_back(std::make_unique<LstmCellBwdImplPass>());
        passes.emplace_back(std::make_unique<MhaAttentionImplPass>());
        passes.emplace_back(std::make_unique<BuildCellEmbedPass>());
        passes.emplace_back(std::make_unique<BuildCellEmbedBwdPass>());
        passes.emplace_back(std::make_unique<ContiguousImplPass>());
        passes.emplace_back(std::make_unique<ReduceSumPartialImplPass>());
        pi::tensorlib::passes::Transform(execution_plan, passes);
    }

    fbamtrain::FbamModelConfiguration BuildModelConfig()
    {
        return fbamtrain::FbamModelConfiguration{
            .frame_rows = ROWS,
            .frame_cols = COLS,
            .n_embed = N_EMBD,
            .n_layer = NUM_LAYERS,
            .n_head = NUM_HEADS,
            .downsample_blocks = NUM_DOWNSAMPLE_BLOCKS,
            .max_code_point = MAX_CODE_POINT,
            .rms_norm_eps = RMS_EPS,
            .bias = true,
            .downsample_conv_mode = "dilated",
            .downsample_conv_dilation = 2,
            .model_init_seed = MODEL_INIT_SEED,
            .use_fp16_accumulation = false,
            .streaming_chunk_size = 1,
            .recompute_interval = 1,
        };
    }

    template <typename MapT>
    std::shared_ptr<RealTensor> FetchTensor(const MapT &ref, const std::string &name)
    {
        auto it = ref.find(name);
        if (it == ref.end())
        {
            throw std::runtime_error("Missing tensor in reference file: " + name);
        }
        return it->second;
    }

    void RunFrameHeadBackwardCase(const DataType dtype)
    {
        ExecutionBackend &execution_backend = ExecutionBackend::getInstance();
        auto &allocator_registry = allocator::DefaultAllocatorRegistry::instance();
        const auto main_stream_desc = GpuStreamDescriptors::Main;

        const auto reference_path = test_utils::ReferenceFileName(dtype);
        const auto reference = safetensors::Load(reference_path);
        const auto upstream_host = FetchTensor(reference, "upstream");

        if (upstream_host->shape().ndims() != 2 || upstream_host->shape()[0] != BATCH_SIZE ||
            upstream_host->shape()[1] != N_EMBD)
        {
            throw std::runtime_error("Unexpected upstream gradient shape in reference data");
        }

        OpGraph init_graph{{}, {}};
        const auto model_config = BuildModelConfig();
        fbamtrain::FrameHeadModule frame_head_module(model_config, init_graph, DEVICE_GPU, dtype, main_stream_desc);
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
            TraceTensor::Create({BATCH_SIZE, ROWS, COLS, NUM_CHANNELS_PER_CELL}, DataType::UINT32, DEVICE_CPU,
                                main_stream_desc);
        TraceTensor upstream_cpu = TraceTensor::Create({BATCH_SIZE, N_EMBD}, dtype, DEVICE_CPU, main_stream_desc);
        cell_states_cpu.markRetained();
        upstream_cpu.markRetained();

        OpGraph graph({
                          {.name = "cell_states", .tensor = cell_states_cpu},
                          {.name = "upstream", .tensor = upstream_cpu},
                      },
                      parameter_descriptors);

        TraceTensor cell_states_gpu = cell_states_cpu.to(graph, DEVICE_GPU, GpuStreamDescriptors::Main);
        TraceTensor upstream_gpu = upstream_cpu.to(graph, DEVICE_GPU, GpuStreamDescriptors::Main);

        (void)frame_head_module.buildForward(graph, {cell_states_gpu}, true);

        std::unordered_map<std::string, TraceTensor> parameter_grads{};
        std::unordered_map<std::string, TraceTensor> operand_grads{};
        std::vector<std::pair<std::string, TraceTensor>> grad_entries{};
        parameter_grads.reserve(parameters.size());
        grad_entries.reserve(parameters.size());
        for (const auto &[name, tensor] : parameters)
        {
            TraceTensor grad = graph.createTensor(tensor.shape().dims(), dtype, DEVICE_GPU, main_stream_desc, false);
            FillZeros(graph, grad, main_stream_desc);
            parameter_grads.emplace(name, grad);
            grad_entries.emplace_back(name, grad);
        }

        frame_head_module.buildBackward(graph, upstream_gpu, parameter_grads, operand_grads);

        graph.finalize();

        std::vector<GraphExecutionInputDescriptor> parameter_execution_inputs{};
        parameter_execution_inputs.reserve(parameters.size());
        for (const auto &[name, tensor] : parameters)
        {
            const auto real_tensor_opt = init_executor.getOutput(tensor);
            if (!real_tensor_opt)
            {
                throw std::runtime_error("Parameter tensor not found after init graph execution for " + name);
            }
            parameter_execution_inputs.push_back({.name = name, .tensor = *real_tensor_opt});
        }

        auto cell_states_real = RealTensor::Allocate(
            {BATCH_SIZE, ROWS, COLS, NUM_CHANNELS_PER_CELL}, DataType::UINT32, DEVICE_CPU);
        PopulateCellStates(cell_states_real);

        std::vector<GraphExecutionInputDescriptor> graph_inputs{
            {.name = "cell_states", .tensor = cell_states_real},
            {.name = "upstream", .tensor = upstream_host},
        };

        ExecutionPlan plan = ExecutionPlan::FromGraph(graph, graph_inputs, parameter_execution_inputs);
        ApplyDefaultPasses(plan);

        Executor executor{plan, execution_backend, 0};
        executor.execute(allocator_registry);

        const float tolerance = test_utils::SelectTolerance(dtype, TOLERANCE_BF16, TOLERANCE_FP16);

        std::vector<std::string> mismatches{};
        for (const auto &[name, grad_trace] : grad_entries)
        {
            const auto expected = FetchTensor(reference, name);
            const auto actual_opt = executor.getOutput(grad_trace);
            if (!actual_opt)
            {
                throw std::runtime_error("Missing gradient output for " + name);
            }
            try
            {
                testing::AssertSimilar(expected, *actual_opt, tolerance);
            }
            catch (const std::runtime_error &err)
            {
                mismatches.emplace_back("Gradient mismatch for " + name + ": " + err.what());
            }
        }
        if (!mismatches.empty())
        {
            std::string summary = mismatches.front();
            const size_t report_count = std::min<size_t>(mismatches.size(), 8);
            for (size_t i = 1; i < report_count; ++i)
            {
                summary.append("\n");
                summary.append(mismatches[i]);
            }
            if (mismatches.size() > report_count)
            {
                summary.append("\n... and ");
                summary.append(std::to_string(mismatches.size() - report_count));
                summary.append(" more mismatches.");
            }
            throw std::runtime_error(summary);
        }
    }
} // namespace

int main()
{
    const auto dtype = test_utils::GetTestDtype();
    RunFrameHeadBackwardCase(dtype);
    return 0;
}
