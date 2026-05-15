#include "testing.h"

#include <allocator.h>

#include "../common/test_dtype_utils.h"
#include <attention.h>
#include <execution_backend.h>
#include <executor.h>
#include <functional.h>
#include <mlp.h>
#include <module.h>
#include <passes.h>
#include <rms_norm.h>
#include <safe_tensors.h>
#include <tensorlib.h>
#include <utils.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace
{
    using namespace pi::tensorlib;

    class TestAttentionBlock final : public Module<>
    {
        std::shared_ptr<RmsNorm> rms_norm_1_;
        std::shared_ptr<FullMhaAttention> attention_;
        std::shared_ptr<RmsNorm> rms_norm_2_;
        std::shared_ptr<MlpBlock> mlp_;
        Device device_;
        DataType dtype_;
        GpuStreamDescriptor compute_stream_descriptor_;

      public:
        TestAttentionBlock(const std::string &name, const uint32_t embed_dim, const uint32_t num_heads,
                           const uint32_t hidden_dim, const float rms_eps, const Device device, const DataType dtype,
                           OpGraph &graph, uint32_t &init_seed, const GpuStreamDescriptor compute_stream_descriptor)
            : Module(name), device_(device), dtype_(dtype), compute_stream_descriptor_(compute_stream_descriptor)
        {
            rms_norm_1_ = std::make_shared<RmsNorm>(name + ".ln1", embed_dim, device_, dtype_, rms_eps, graph,
                                                    false, compute_stream_descriptor_);
            attention_ = std::make_shared<FullMhaAttention>(name + ".attn", embed_dim, num_heads, device_, dtype_,
                                                            graph, init_seed, compute_stream_descriptor_);
            rms_norm_2_ = std::make_shared<RmsNorm>(name + ".ln2", embed_dim, device_, dtype_, rms_eps, graph,
                                                    false, compute_stream_descriptor_);
            mlp_ = std::make_shared<MlpBlock>(name + ".mlp", embed_dim, hidden_dim, device_, dtype_, graph, init_seed,
                                              compute_stream_descriptor_);
        }

        [[nodiscard]] TraceTensor buildForward(OpGraph &graph, const std::initializer_list<TraceTensor> inputs,
                                               const bool save_input_for_backward) override
        {
            const auto &input = *inputs.begin();

            TraceTensor norm1 = rms_norm_1_->buildForward(graph, {input}, save_input_for_backward);
            TraceTensor attn_out = attention_->buildForward(graph, {norm1}, save_input_for_backward);

            const auto &shape = input.shape();
            const uint64_t batch = shape[0];
            const uint64_t seq_len = shape[1];
            const uint64_t embed_dim = shape[2];

            // Residual add: attn_out += input
            InplaceAdd(graph, attn_out, input, compute_stream_descriptor_);
            TraceTensor residual_after_attn = attn_out;

            TraceTensor norm2 = rms_norm_2_->buildForward(graph, {residual_after_attn}, save_input_for_backward);
            TraceTensor mlp_out = mlp_->buildForward(graph, {norm2}, save_input_for_backward);

            TraceTensor mlp_out_reshaped = mlp_out.view(graph, {batch, seq_len, embed_dim});
            // Residual add: mlp_out += residual_after_attn
            InplaceAdd(graph, mlp_out_reshaped, residual_after_attn, compute_stream_descriptor_);

            return mlp_out_reshaped;
        }

        void buildBackward(OpGraph & /*graph*/, const TraceTensor & /*backward_input*/,
                           const std::unordered_map<std::string, TraceTensor> &,
                           const std::unordered_map<std::string, TraceTensor> &) override
        {
            throw std::runtime_error("TestAttentionBlock backward is not implemented.");
        }

        [[nodiscard]] std::vector<ParameterEntry> parameters() const override
        {
            std::vector<ParameterEntry> params{};
            const auto rms1_params = rms_norm_1_->parameters();
            const auto attn_params = attention_->parameters();
            const auto rms2_params = rms_norm_2_->parameters();
            const auto mlp_params = mlp_->parameters();
            params.reserve(rms1_params.size() + attn_params.size() + rms2_params.size() + mlp_params.size());
            for (const auto &entry : rms1_params)
            {
                params.push_back(entry);
            }
            for (const auto &entry : attn_params)
            {
                params.push_back(entry);
            }
            for (const auto &entry : rms2_params)
            {
                params.push_back(entry);
            }
            for (const auto &entry : mlp_params)
            {
                params.push_back(entry);
            }
            return params;
        }
    };

    constexpr uint32_t BATCH_SIZE = 2;
    constexpr uint32_t SEQ_LEN = 128;
    constexpr uint32_t NUM_HEADS = 8;
    constexpr uint32_t HEAD_DIM = 128;
    constexpr uint32_t N_EMBD = NUM_HEADS * HEAD_DIM;
    constexpr uint32_t HIDDEN_DIM = 4 * N_EMBD;

    constexpr float RMS_EPS = 1e-5f;
    constexpr float INPUT_LOW = -0.5f;
    constexpr float INPUT_HIGH = 0.5f;
    constexpr float TOLERANCE_BF16 = 1e-2f;
    constexpr float TOLERANCE_FP16 = 2e-2f;

    constexpr uint32_t MODEL_INIT_SEED = 1337u;

    void ApplyDefaultPasses(ExecutionPlan &execution_plan)
    {
        std::vector<std::unique_ptr<passes::CompilerPass>> passes{};
        passes.emplace_back(std::make_unique<MatmulFusePass>());
        passes.emplace_back(std::make_unique<MatmulImplPass>());
        passes.emplace_back(std::make_unique<MhaAttentionImplPass>());
        passes.emplace_back(std::make_unique<RmsNormImplPass>());
        passes.emplace_back(std::make_unique<FillZerosImplPass>());
        passes.emplace_back(std::make_unique<FillConstantImplPass>());
        passes.emplace_back(std::make_unique<FillUniformImplPass>());
        passes.emplace_back(std::make_unique<GatherImplPass>());
        passes.emplace_back(std::make_unique<ContiguousImplPass>());
        passes.emplace_back(std::make_unique<TrailingBroadcastAddImplPass>());
        passes.emplace_back(std::make_unique<ActImplPass>());
        pi::tensorlib::passes::Transform(execution_plan, passes);
    }

    bool PlanHasKernelSubstring(const ExecutionPlan &plan, const std::string &needle)
    {
        for (const auto &entry : plan.entries)
        {
            if (!entry.kernel_descriptor)
            {
                continue;
            }
            const auto &kernel_name = entry.kernel_descriptor->kernel_name;
            if (kernel_name.find(needle) != std::string::npos)
            {
                return true;
            }
        }
        return false;
    }

    bool PlanUsesTritonMhaKernel(const ExecutionPlan &plan, const DataType dtype)
    {
        std::string fwd_kernel{"mha_full_attn_fwd_hs128_"};
        fwd_kernel.append(test_utils::GetDtypeSuffix(dtype));
        return PlanHasKernelSubstring(plan, fwd_kernel);
    }
} // namespace

int main()
{
    ExecutionBackend &execution_backend = ExecutionBackend::getInstance();
    auto &allocator_registry = allocator::DefaultAllocatorRegistry::instance();

    constexpr Device device{
        .device_type = DeviceType::GPU,
        .ordinal = 0,
    };
    const auto main_stream_desc = GpuStreamDescriptors::Main;

    const auto dtype = test_utils::GetTestDtype();

    OpGraph init_graph{{}, {}};

    uint32_t rng_seed = MODEL_INIT_SEED;

    TestAttentionBlock attention_block{"block", N_EMBD, NUM_HEADS,  HIDDEN_DIM, RMS_EPS,
                                       device,  dtype,  init_graph, rng_seed, main_stream_desc};

    TraceTensor x = init_graph.createTensor({BATCH_SIZE, SEQ_LEN, N_EMBD}, dtype, device, main_stream_desc, false);
    FillUniform(init_graph, x, INPUT_LOW, INPUT_HIGH, rng_seed++, main_stream_desc);

    init_graph.finalize();

    ExecutionPlan init_plan = ExecutionPlan::FromGraph(init_graph, {}, {});
    ApplyDefaultPasses(init_plan);
    Executor init_executor{init_plan, execution_backend, 0};
    init_executor.execute(allocator_registry);

    const auto block_parameters = attention_block.parameters();

    std::vector<GraphInputDescriptor> parameter_descriptors{};
    parameter_descriptors.reserve(block_parameters.size());
    for (const auto &[name, tensor] : block_parameters)
    {
        auto retained_tensor = tensor;
        retained_tensor.markRetained();
        parameter_descriptors.push_back(GraphInputDescriptor{.name = name, .tensor = retained_tensor});
    }

    x.markRetained();
    OpGraph main_graph(
        {
            {.name = "x", .tensor = x},
        },
        parameter_descriptors);

    TraceTensor block_output = attention_block.buildForward(main_graph, {x}, false);

    main_graph.finalize();

    std::vector<GraphExecutionInputDescriptor> parameter_real_descriptors{};
    parameter_real_descriptors.reserve(block_parameters.size());
    for (const auto &[name, tensor] : block_parameters)
    {
        const auto tensor_real_opt = init_executor.getOutput(tensor);
        if (!tensor_real_opt)
        {
            throw std::runtime_error("Parameter tensor not found after init graph execution for " + name);
        }
        parameter_real_descriptors.push_back({.name = name, .tensor = *tensor_real_opt});
    }

    const auto input_real_opt = init_executor.getOutput(x);
    if (!input_real_opt)
    {
        throw std::runtime_error("Input tensor not found after init graph execution");
    }

    ExecutionPlan main_plan = ExecutionPlan::FromGraph(main_graph,
                                                       {
                                                           {.name = "x", .tensor = *input_real_opt},
                                                       },
                                                       parameter_real_descriptors);

    ApplyDefaultPasses(main_plan);

    Executor executor{main_plan, execution_backend, 0};
    executor.execute(allocator_registry);

    const auto output_real_opt = executor.getOutput(block_output);
    if (!output_real_opt)
    {
        throw std::runtime_error("Output tensor not found after main graph execution");
    }
    const auto &actual_output = *output_real_opt;

    const std::string reference_path = test_utils::ReferenceFileName(dtype);
    const auto expected_tensors = safetensors::Load(reference_path);
    const auto expected_output_it = expected_tensors.find("output");
    if (expected_output_it == expected_tensors.end())
    {
        throw std::runtime_error("Expected output tensor not found in reference file: " + reference_path);
    }
    const auto &expected_output = expected_output_it->second;
    float tolerance = test_utils::SelectTolerance(dtype, TOLERANCE_BF16, TOLERANCE_FP16);
    if (const char *backend_env = std::getenv("FBAMTRAIN_PREFER_MHA_BACKEND");
        (backend_env != nullptr && std::strcmp(backend_env, "triton") == 0) || PlanUsesTritonMhaKernel(main_plan, dtype))
    {
        if (dtype == DataType::BFLOAT16 && tolerance < 4e-2f)
        {
            tolerance = 4e-2f;
        }
        if (dtype == DataType::FLOAT16 && tolerance < 4e-2f)
        {
            tolerance = 4e-2f;
        }
    }
    testing::AssertSimilar(expected_output, actual_output, tolerance);
}
