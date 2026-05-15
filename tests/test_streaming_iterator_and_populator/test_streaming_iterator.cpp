#include "functional.h"
#include "testing.h"

#include <allocator.h>
#include <execution_backend.h>
#include <executor.h>
#include <op_graph.h>
#include <tensor_stream_iterator.h>
#include <tensor_stream_populator.h>
#include <tensorlib.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace pi::tensorlib;

namespace
{
    constexpr uint64_t SEQ_LEN = 4;
    constexpr uint64_t FEATURE_DIM = 6;

    constexpr Device DEVICE_CPU{
        .device_type = DeviceType::CPU,
        .ordinal = 0,
    };
    constexpr Device DEVICE_GPU{
        .device_type = DeviceType::GPU,
        .ordinal = 0,
    };

    std::shared_ptr<RealTensor> CreateHostTensor()
    {
        auto tensor = RealTensor::Allocate({SEQ_LEN, FEATURE_DIM}, DataType::FLOAT32, DEVICE_CPU);
        auto *host_data = static_cast<float *>(tensor->dataptr());
        for (uint64_t idx = 0; idx < SEQ_LEN * FEATURE_DIM; ++idx)
        {
            host_data[idx] = static_cast<float>(idx + 1);
        }
        return tensor;
    }

    void TestIterator(ExecutionBackend &execution_backend, const allocator::AllocatorRegistry &allocator_registry)
    {
        auto host_tensor = CreateHostTensor();
        constexpr int CHUNK = 1;
        const auto main_stream_desc = GpuStreamDescriptors::Main;

        TraceTensor source_cpu =
            TraceTensor::Create({SEQ_LEN, FEATURE_DIM}, DataType::FLOAT32, DEVICE_CPU, main_stream_desc);
        source_cpu.markRetained();
        OpGraph graph({{.name = "input", .tensor = source_cpu}}, {});

        TensorStreamIterator iterator(source_cpu, /*chunk_size=*/CHUNK, /*dim=*/0, DEVICE_GPU, TransferType::H2D,
                                      main_stream_desc);
        std::vector<TraceTensor> streamed_chunks;
        streamed_chunks.reserve(SEQ_LEN);

        while (auto chunk = iterator.next(graph))
        {
            const Device &chunk_device = chunk->device();
            if (chunk_device.device_type != DEVICE_GPU.device_type || chunk_device.ordinal != DEVICE_GPU.ordinal)
            {
                throw std::runtime_error("Streamed chunk was not materialized on the GPU");
            }
            streamed_chunks.push_back(*chunk);
        }
        AwaitComputeForTransfer(graph, TransferType::H2D, DEVICE_GPU, GpuStreamDescriptors::Main);
        AwaitAsyncTransfers(graph, TransferType::H2D, DEVICE_GPU, GpuStreamDescriptors::Main);

        if (streamed_chunks.size() != SEQ_LEN)
        {
            throw std::runtime_error("Iterator did not consume the full sequence length");
        }

        graph.finalize();

        ExecutionPlan plan = ExecutionPlan::FromGraph(graph,
                                                      {
                                                          {.name = "input", .tensor = host_tensor},
                                                      },
                                                      {});

        Executor executor{plan, execution_backend, 0};
        executor.execute(allocator_registry);

        for (uint64_t chunk_idx = 0; chunk_idx < streamed_chunks.size(); ++chunk_idx)
        {
            const auto actual_opt = executor.getOutput(streamed_chunks[chunk_idx]);
            if (!actual_opt)
            {
                throw std::runtime_error("Failed to retrieve streamed chunk output");
            }
            auto expected_chunk =
                RealTensor::Allocate({static_cast<uint64_t>(CHUNK), FEATURE_DIM}, DataType::FLOAT32, DEVICE_CPU);
            auto *dst = static_cast<float *>(expected_chunk->dataptr());
            const auto *src = static_cast<float *>(host_tensor->dataptr()) + chunk_idx * FEATURE_DIM;
            std::copy(src, src + FEATURE_DIM * CHUNK, dst);
            testing::AssertSimilar(expected_chunk, *actual_opt, 0.0);
        }
    }

    void TestPopulator(ExecutionBackend &execution_backend, const allocator::AllocatorRegistry &allocator_registry)
    {
        auto host_tensor = CreateHostTensor();
        const auto main_stream_desc = GpuStreamDescriptors::Main;

        TraceTensor source_cpu =
            TraceTensor::Create({SEQ_LEN, FEATURE_DIM}, DataType::FLOAT32, DEVICE_CPU, main_stream_desc);
        source_cpu.markRetained();
        OpGraph graph({{.name = "input", .tensor = source_cpu}}, {});
        TraceTensor destination_cpu =
            graph.createTensor({SEQ_LEN, FEATURE_DIM}, DataType::FLOAT32, DEVICE_CPU, main_stream_desc, false);

        constexpr int CHUNK = 1;
        TensorStreamIterator iterator(source_cpu, /*chunk_size=*/CHUNK, /*dim=*/0, DEVICE_GPU, TransferType::H2D,
                                      main_stream_desc);
        TensorStreamPopulator populator(destination_cpu, /*dim=*/0, TransferType::D2H);

        while (auto chunk = iterator.next(graph))
        {
            AwaitAsyncTransfers(graph, TransferType::H2D, DEVICE_GPU, GpuStreamDescriptors::Main);
            populator.populateNext(graph, *chunk, GpuStreamDescriptors::Main);
        }
        AwaitComputeForTransfer(graph, TransferType::D2H, DEVICE_GPU, GpuStreamDescriptors::Main);
        AwaitAsyncTransfers(graph, TransferType::D2H, DEVICE_GPU, GpuStreamDescriptors::Main);

        graph.finalize();

        ExecutionPlan plan = ExecutionPlan::FromGraph(graph,
                                                      {
                                                          {.name = "input", .tensor = host_tensor},
                                                      },
                                                      {});

        Executor executor{plan, execution_backend, 0};
        executor.execute(allocator_registry);

        const auto destination_opt = executor.getOutput(destination_cpu);
        if (!destination_opt)
        {
            throw std::runtime_error("Failed to retrieve populated destination tensor");
        }

        testing::AssertSimilar(host_tensor, *destination_opt, 0.0);
    }
} // namespace

int main()
{
    ExecutionBackend &execution_backend = ExecutionBackend::getInstance();
    const auto &allocator_registry = allocator::CachingAllocatorRegistry::instance();

    TestIterator(execution_backend, allocator_registry);
    TestPopulator(execution_backend, allocator_registry);
    return 0;
}
