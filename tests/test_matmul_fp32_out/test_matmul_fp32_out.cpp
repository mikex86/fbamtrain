#include "testing.h"

#include <execution_backend.h>
#include <executor.h>
#include <op_graph.h>
#include <passes.h>
#include <safe_tensors.h>
#include <tensorlib.h>

#include <cstdint>
#include <stdexcept>
#include <string>

using namespace pi::tensorlib;

namespace
{
    constexpr uint64_t M = 128;
    constexpr uint64_t K = 256;
    constexpr uint64_t N = 64;

    constexpr Device DEVICE_CPU{
        .device_type = DeviceType::CPU,
        .ordinal = 0,
    };
    constexpr Device DEVICE_GPU{
        .device_type = DeviceType::GPU,
        .ordinal = 0,
    };
} // namespace

int main()
{
    ExecutionBackend &execution_backend = ExecutionBackend::getInstance();
    const auto &allocator_registry = allocator::DefaultAllocatorRegistry::instance();

    const auto tensors = safetensors::Load("reference.safetensors");
    const auto a_it = tensors.find("a");
    const auto b_it = tensors.find("b");
    const auto out_it = tensors.find("out");
    if (a_it == tensors.end() || b_it == tensors.end() || out_it == tensors.end())
    {
        throw std::runtime_error("reference.safetensors missing required tensors");
    }
    const auto &a_host = a_it->second;
    const auto &b_host = b_it->second;
    const auto &expected_out = out_it->second;

    // Trace tensors for inputs on host
    const auto main_stream_desc = GpuStreamDescriptors::Main;
    TraceTensor a_cpu = TraceTensor::Create({M, K}, DataType::FLOAT16, DEVICE_CPU, main_stream_desc);
    TraceTensor b_cpu = TraceTensor::Create({K, N}, DataType::FLOAT16, DEVICE_CPU, main_stream_desc);
    a_cpu.markRetained();
    b_cpu.markRetained();

    OpGraph graph({{.name = "a", .tensor = a_cpu}, {.name = "b", .tensor = b_cpu}}, {});

    // GPU copies
    TraceTensor a_gpu = graph.createTensor({M, K}, DataType::FLOAT16, DEVICE_GPU, main_stream_desc, false);
    TraceTensor b_gpu = graph.createTensor({K, N}, DataType::FLOAT16, DEVICE_GPU, main_stream_desc, false);
    TraceTensor out_gpu = graph.createTensor({M, N}, DataType::FLOAT32, DEVICE_GPU, main_stream_desc, false);
    TraceTensor out_cpu = graph.createTensor({M, N}, DataType::FLOAT32, DEVICE_CPU, main_stream_desc, false);

    const auto copy_stream_desc = GpuStreamDescriptors::Main;
    graph.recordOperation(
        OperationEntry{.type = OpType::DEVICE_COPY,
                       .inputs = {a_cpu},
                       .outputs = {a_gpu},
                       .attributes = {},
                       .gpu_stream_desc = copy_stream_desc});
    graph.recordOperation(
        OperationEntry{.type = OpType::DEVICE_COPY,
                       .inputs = {b_cpu},
                       .outputs = {b_gpu},
                       .attributes = {},
                       .gpu_stream_desc = copy_stream_desc});

    std::unordered_map<std::string, std::any> matmul_attr{};
    graph.recordOperation(OperationEntry{.type = OpType::MATMUL,
                                         .inputs = {a_gpu, b_gpu},
                                         .outputs = {out_gpu},
                                         .attributes = matmul_attr,
                                         .gpu_stream_desc = main_stream_desc});

    graph.recordOperation(OperationEntry{.type = OpType::DEVICE_COPY,
                                         .inputs = {out_gpu},
                                         .outputs = {out_cpu},
                                         .attributes = {},
                                         .gpu_stream_desc = copy_stream_desc});

    graph.finalize();

    ExecutionPlan plan = ExecutionPlan::FromGraph(graph,
                                                  {
                                                      {.name = "a", .tensor = a_host},
                                                      {.name = "b", .tensor = b_host},
                                                  },
                                                  {});

    std::vector<std::unique_ptr<passes::CompilerPass>> passes{};
    passes.emplace_back(std::make_unique<MatmulImplPass>());
    pi::tensorlib::passes::Transform(plan, passes);

    Executor executor{plan, execution_backend, 0};
    executor.execute(allocator_registry);

    const auto actual_out_opt = executor.getOutput(out_cpu);
    if (!actual_out_opt)
    {
        throw std::runtime_error("Failed to retrieve matmul output");
    }

    testing::AssertSimilar(expected_out, *actual_out_opt, 5e-3);
    return 0;
}
