#include "testing.h"

#include <allocator.h>

#include <execution_backend.h>
#include <executor.h>
#include <functional.h>
#include <passes.h>
#include <tensorlib.h>
#include <utils.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

using namespace pi::tensorlib;

namespace
{
    constexpr uint32_t NUM_ELEMENTS = 4096;

    void ApplyInitPasses(ExecutionPlan &plan)
    {
        std::vector<std::unique_ptr<pi::tensorlib::passes::CompilerPass>> passes{};
        passes.emplace_back(std::make_unique<FillZerosImplPass>());
        passes.emplace_back(std::make_unique<FillConstantImplPass>());
        passes.emplace_back(std::make_unique<FillUniformImplPass>());
        pi::tensorlib::passes::Transform(plan, passes);
    }

    void ApplyCastPass(ExecutionPlan &plan)
    {
        std::vector<std::unique_ptr<pi::tensorlib::passes::CompilerPass>> passes{};
        passes.emplace_back(std::make_unique<CastImplPass>());
        pi::tensorlib::passes::Transform(plan, passes);
    }

    void TestBf16ToFp32(ExecutionBackend &backend, const Device &device_gpu, const Device &device_cpu)
    {
        const auto main_stream_desc = GpuStreamDescriptors::Main;
        OpGraph init_graph{{}, {}};

        TraceTensor input = init_graph.createTensor({NUM_ELEMENTS}, DataType::BFLOAT16, device_gpu,
                                                    main_stream_desc, false);
        input.markRetained();

        uint32_t seed = 123u;
        FillUniform(init_graph, input, -0.5f, 0.5f, seed++, main_stream_desc);

        init_graph.finalize();

        ExecutionPlan init_plan = ExecutionPlan::FromGraph(init_graph, {}, {});
        ApplyInitPasses(init_plan);
        Executor init_executor{init_plan, backend, 0};
        init_executor.execute(pi::tensorlib::allocator::DefaultAllocatorRegistry::instance());

        const auto input_real_opt = init_executor.getOutput(input);
        if (!input_real_opt)
        {
            throw std::runtime_error("Failed to materialize BF16 input tensor");
        }
        const std::shared_ptr<RealTensor> &input_real = *input_real_opt;

        const auto input_storage_cpu = input_real->storage()->toCPU();
        const auto *input_data =
            reinterpret_cast<const uint16_t *>(static_cast<const uint8_t *>(input_storage_cpu->dataptr())) +
            input_real->storageOffset();

        auto expected = RealTensor::Allocate({NUM_ELEMENTS}, DataType::FLOAT32, device_cpu);
        auto *expected_data = static_cast<float *>(expected->dataptr());
        for (uint32_t i = 0; i < NUM_ELEMENTS; ++i)
        {
            expected_data[i] = utils::Fp32FromBf16(input_data[i]);
        }

        OpGraph graph({{.name = "input", .tensor = input}}, {});
        TraceTensor output = input.to(graph, DataType::FLOAT32, main_stream_desc);
        graph.finalize();

        ExecutionPlan plan = ExecutionPlan::FromGraph(graph,
                                                      {
                                                          {.name = "input", .tensor = input_real},
                                                      },
                                                      {});

        ApplyCastPass(plan);
        Executor executor{plan, backend, 0};
        executor.execute(pi::tensorlib::allocator::DefaultAllocatorRegistry::instance());

        const auto actual_output_opt = executor.getOutput(output);
        if (!actual_output_opt)
        {
            throw std::runtime_error("Failed to retrieve cast output tensor (BF16->FP32)");
        }
        const auto &actual_output = *actual_output_opt;

        testing::AssertSimilar(expected, actual_output, 1e-4);
    }

    void TestFp32ToBf16(ExecutionBackend &backend, const Device &device_gpu, const Device &device_cpu)
    {
        const auto main_stream_desc = GpuStreamDescriptors::Main;
        auto host_input = RealTensor::Allocate({NUM_ELEMENTS}, DataType::FLOAT32, device_cpu);
        auto *host_data = static_cast<float *>(host_input->dataptr());
        for (uint32_t i = 0; i < NUM_ELEMENTS; ++i)
        {
            host_data[i] = std::sin(static_cast<float>(i) * 0.01f);
        }

        auto expected = RealTensor::Allocate({NUM_ELEMENTS}, DataType::BFLOAT16, device_cpu);
        auto *expected_data = reinterpret_cast<uint16_t *>(expected->dataptr());
        for (uint32_t i = 0; i < NUM_ELEMENTS; ++i)
        {
            expected_data[i] = utils::Bf16FromFp32(host_data[i]);
        }

        TraceTensor input_cpu = TraceTensor::Create({NUM_ELEMENTS}, DataType::FLOAT32, device_cpu, main_stream_desc);
        input_cpu.markRetained();

        OpGraph graph({{.name = "input", .tensor = input_cpu}}, {});
        TraceTensor input_gpu = input_cpu.to(graph, device_gpu, GpuStreamDescriptors::Main);
        TraceTensor output = input_gpu.to(graph, DataType::BFLOAT16, main_stream_desc);
        graph.finalize();

        ExecutionPlan plan = ExecutionPlan::FromGraph(graph,
                                                      {
                                                          {.name = "input", .tensor = host_input},
                                                      },
                                                      {});

        ApplyCastPass(plan);
        Executor executor{plan, backend, 0};
        executor.execute(pi::tensorlib::allocator::DefaultAllocatorRegistry::instance());

        const auto actual_output_opt = executor.getOutput(output);
        if (!actual_output_opt)
        {
            throw std::runtime_error("Failed to retrieve cast output tensor (FP32->BF16)");
        }
        const auto &actual_output = *actual_output_opt;

        testing::AssertSimilar(expected, actual_output, 0.0);
    }
} // namespace

int main()
{
    ExecutionBackend &execution_backend = ExecutionBackend::getInstance();

    const Device device_gpu{
        .device_type = DeviceType::GPU,
        .ordinal = 0,
    };
    const Device device_cpu{
        .device_type = DeviceType::CPU,
        .ordinal = 0,
    };

    TestBf16ToFp32(execution_backend, device_gpu, device_cpu);
    TestFp32ToBf16(execution_backend, device_gpu, device_cpu);
    return 0;
}
