#include "testing.h"

#include <allocator.h>
#include <execution_backend.h>
#include <executor.h>
#include <functional.h>
#include <linear.h>
#include <passes.h>
#include <safe_tensors.h>
#include <tensorlib.h>

#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    constexpr pi::tensorlib::Device DEVICE_CPU{pi::tensorlib::DeviceType::CPU, 0};
    constexpr pi::tensorlib::Device DEVICE_GPU{pi::tensorlib::DeviceType::GPU, 0};

    constexpr std::array<uint32_t, 6> TARGET_VALUES{1U, 0U, 5U, 3U, 2U, 6U};

    void ApplyDefaultPasses(pi::tensorlib::ExecutionPlan &plan)
    {
        std::vector<std::unique_ptr<pi::tensorlib::passes::CompilerPass>> passes{};
        passes.emplace_back(std::make_unique<CrossEntropyOnTargetsImplPass>());
        passes.emplace_back(std::make_unique<MeanImplPass>());
        passes.emplace_back(std::make_unique<ReduceSumPartialImplPass>());
        passes.emplace_back(std::make_unique<ReduceSumImplPass>());
        passes.emplace_back(std::make_unique<ContiguousImplPass>());
        passes.emplace_back(std::make_unique<MatmulFusePass>());
        passes.emplace_back(std::make_unique<MatmulImplPass>());
        passes.emplace_back(std::make_unique<TrailingBroadcastAddImplPass>());
        passes.emplace_back(std::make_unique<CastImplPass>());
        passes.emplace_back(std::make_unique<DivImplPass>());
        passes.emplace_back(std::make_unique<FillConstantImplPass>());
        passes.emplace_back(std::make_unique<FillUniformImplPass>());
        passes.emplace_back(std::make_unique<FillZerosImplPass>());
        pi::tensorlib::passes::Transform(plan, passes);
    }

    std::shared_ptr<pi::tensorlib::RealTensor> MakeTargets()
    {
        auto targets_cpu =
            pi::tensorlib::RealTensor::Allocate({TARGET_VALUES.size()}, pi::tensorlib::DataType::UINT32, DEVICE_CPU);
        auto *dst = static_cast<uint32_t *>(targets_cpu->dataptr());
        for (size_t i = 0; i < TARGET_VALUES.size(); ++i)
        {
            dst[i] = TARGET_VALUES[i];
        }
        return targets_cpu;
    }
} // namespace

int main(int argc, char **argv)
{
    using namespace pi::tensorlib;

    ExecutionBackend &backend = ExecutionBackend::getInstance();
    auto &allocator_registry = allocator::DefaultAllocatorRegistry::instance();

    const auto ref = safetensors::Load("reference.safetensors");
    auto fetch = [&ref](const char *name) -> std::shared_ptr<RealTensor>
    {
        auto it = ref.find(name);
        if (it == ref.end())
        {
            throw std::runtime_error(std::string("missing tensor: ") + name);
        }
        return it->second;
    };

    auto upstream_scalar = RealTensor::Allocate({1}, DataType::FLOAT32, DEVICE_CPU);
    *static_cast<float *>(upstream_scalar->dataptr()) = 1.0f;

    const auto run_case = [&](const char *suffix, const DataType dtype)
    {
        const auto main_stream_desc = GpuStreamDescriptors::Main;
        const auto x_host = fetch(std::string("x_").append(suffix).c_str());
        const auto weight_host = fetch(std::string("weight_").append(suffix).c_str());
        const auto bias_host = fetch(std::string("bias_").append(suffix).c_str());
        const auto expected_dx = fetch(std::string("grad_x_").append(suffix).c_str());
        const auto expected_dw = fetch(std::string("grad_w_").append(suffix).c_str());
        const auto expected_db = fetch(std::string("grad_b_").append(suffix).c_str());

        OpGraph init_graph{{}, {}};
        uint32_t seed = 0;
        Linear lm_head{"lm_head",
                       x_host->shape()[1],
                       weight_host->shape()[1],
                       DEVICE_GPU,
                       dtype,
                       ActivationFunction::NONE,
                       init_graph,
                       seed,
                       main_stream_desc};

        TraceTensor x = TraceTensor::Create(x_host->shape().dims(), dtype, DEVICE_GPU, main_stream_desc);
        TraceTensor targets = TraceTensor::Create({TARGET_VALUES.size()}, DataType::UINT32, DEVICE_GPU,
                                                  main_stream_desc);
        TraceTensor upstream = TraceTensor::Create({1}, DataType::FLOAT32, DEVICE_GPU, main_stream_desc);
        x.markRetained();
        targets.markRetained();
        upstream.markRetained();

        const auto params = lm_head.parameters();
        auto retained_weight = params[0].tensor;
        auto retained_bias = params[1].tensor;
        retained_weight.markRetained();
        retained_bias.markRetained();

        std::vector<GraphInputDescriptor> inputs{
            {.name = "x", .tensor = x},
            {.name = "targets", .tensor = targets},
            {.name = "upstream", .tensor = upstream},
            {.name = params[0].name, .tensor = retained_weight},
            {.name = params[1].name, .tensor = retained_bias},
        };

        OpGraph graph(inputs, {});

        const TraceTensor logits = lm_head.buildForward(graph, {x}, /*save_input_for_backward=*/true);
        const TraceTensor loss = CrossEntropyOnTargets(graph, logits, targets, Reduction::MEAN, main_stream_desc,
                                                       /*reduce_over_rows=*/true);
        (void)loss;
        const TraceTensor grad_logits =
            CrossEntropyOnTargetsBackward(graph, logits, targets, upstream, Reduction::MEAN,
                                          main_stream_desc, /*reduce_over_rows=*/true);

        std::unordered_map<std::string, TraceTensor> parameter_grads{};
        std::unordered_map<std::string, TraceTensor> operand_grads{};
        TraceTensor w_grad = graph.createTensor(weight_host->shape().dims(), dtype, DEVICE_GPU, main_stream_desc, false);
        TraceTensor b_grad = graph.createTensor(bias_host->shape().dims(), dtype, DEVICE_GPU, main_stream_desc, false);
        TraceTensor dx_grad = graph.createTensor(x_host->shape().dims(), dtype, DEVICE_GPU, main_stream_desc, false);
        FillZeros(graph, w_grad, main_stream_desc);
        FillZeros(graph, b_grad, main_stream_desc);
        FillZeros(graph, dx_grad, main_stream_desc);
        parameter_grads.emplace(params[0].name, w_grad);
        parameter_grads.emplace(params[1].name, b_grad);
        operand_grads.emplace("input", dx_grad);

        lm_head.buildBackward(graph, grad_logits, parameter_grads, operand_grads);

        graph.finalize();

        auto x_gpu = x_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
        auto targets_gpu = MakeTargets()->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
        auto upstream_gpu = upstream_scalar->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
        auto weight_gpu = weight_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);
        auto bias_gpu = bias_host->to(DEVICE_GPU, pi::tensorlib::GpuStreamDescriptors::Main);

        ExecutionPlan plan = ExecutionPlan::FromGraph(graph,
                                                      {
                                                          {.name = "x", .tensor = x_gpu},
                                                          {.name = "targets", .tensor = targets_gpu},
                                                          {.name = "upstream", .tensor = upstream_gpu},
                                                          {.name = params[0].name, .tensor = weight_gpu},
                                                          {.name = params[1].name, .tensor = bias_gpu},
                                                      },
                                                      {});

        ApplyDefaultPasses(plan);
        Executor executor{plan, backend, 0};
        executor.execute(allocator_registry);

        const float tol = (dtype == DataType::BFLOAT16) ? 8e-3f : 5e-3f;
        const auto actual_w = *executor.getOutput(w_grad);
        const auto actual_b = *executor.getOutput(b_grad);
        const auto actual_dx = *executor.getOutput(dx_grad);
        testing::AssertSimilar(expected_dw, actual_w, tol);
        testing::AssertSimilar(expected_db, actual_b, tol);
        testing::AssertSimilar(expected_dx, actual_dx, tol);
    };

    std::vector<std::string> cases{};
    if (argc > 1)
    {
        for (int i = 1; i < argc; ++i)
        {
            cases.emplace_back(argv[i]);
        }
    }
    else
    {
        throw std::runtime_error("Specify which cases to run (fp16, bf16) as arguments");
    }

    for (const auto &c : cases)
    {
        if (c == "fp16")
        {
            run_case("fp16", DataType::FLOAT16);
        }
        else if (c == "bf16")
        {
            run_case("bf16", DataType::BFLOAT16);
        }
        else
        {
            throw std::runtime_error("unknown case: " + c);
        }
    }

    return 0;
}
