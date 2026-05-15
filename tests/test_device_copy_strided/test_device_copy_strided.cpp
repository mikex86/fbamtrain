#include "testing.h"

#include <allocator.h>
#include <execution_backend.h>
#include <executor.h>
#include <functional.h>
#include <op_graph.h>
#include <tensorlib.h>

#include <cstdint>
#include <unordered_map>
#include <vector>

using namespace pi::tensorlib;

namespace
{
    constexpr Device DEVICE_CPU{DeviceType::CPU, 0};
    constexpr Device DEVICE_GPU{DeviceType::GPU, 0};
    const auto main_stream_desc = GpuStreamDescriptors::Main;

    std::shared_ptr<RealTensor> MakeHostTensor(uint64_t dim0, uint64_t dim1, float start, float step)
    {
        auto tensor = RealTensor::Allocate({dim0, dim1}, DataType::FLOAT32, DEVICE_CPU);
        auto *data = static_cast<float *>(tensor->dataptr());
        const uint64_t total = dim0 * dim1;
        for (uint64_t i = 0; i < total; ++i)
        {
            data[i] = start + step * static_cast<float>(i);
        }
        return tensor;
    }

    std::shared_ptr<RealTensor> MakeHostTensor3d(uint64_t d0, uint64_t d1, uint64_t d2, float start, float step)
    {
        auto tensor = RealTensor::Allocate({d0, d1, d2}, DataType::FLOAT32, DEVICE_CPU);
        auto *data = static_cast<float *>(tensor->dataptr());
        const uint64_t total = d0 * d1 * d2;
        for (uint64_t i = 0; i < total; ++i)
        {
            data[i] = start + step * static_cast<float>(i);
        }
        return tensor;
    }

    void Run2DCopyTest()
    {
        ExecutionBackend &execution_backend = ExecutionBackend::getInstance();
        const auto &allocator_registry = allocator::CachingAllocatorRegistry::instance();

        // Shape chosen so dst leading dimension differs from logical columns.
        const Shape src_shape({3, 4});
        const uint64_t padding = 2;
        const Shape dst_shape({src_shape[0], src_shape[1] + padding});

        auto src_host = MakeHostTensor(src_shape[0], src_shape[1], /*start=*/1.0f, /*step=*/1.0f);
        auto dst_init_host =
            MakeHostTensor(dst_shape[0], dst_shape[1], /*start=*/-1.0f, /*step=*/0.0f); // sentinel fill

        TraceTensor src_cpu = TraceTensor::Create(src_shape.dims(), DataType::FLOAT32, DEVICE_CPU, main_stream_desc);
        src_cpu.markRetained();
        TraceTensor dst_seed_cpu = TraceTensor::Create(dst_shape.dims(), DataType::FLOAT32, DEVICE_CPU,
                                                       main_stream_desc);
        dst_seed_cpu.markRetained();

        OpGraph graph({{.name = "src", .tensor = src_cpu}, {.name = "dst_seed", .tensor = dst_seed_cpu}}, {});

        TraceTensor src_gpu = graph.createTensor(src_shape.dims(), DataType::FLOAT32, DEVICE_GPU, main_stream_desc, false);
        TraceTensor dst_base_gpu = graph.createTensor(dst_shape.dims(), DataType::FLOAT32, DEVICE_GPU, main_stream_desc, false);
        TraceTensor dst_view_gpu = dst_base_gpu.slice(graph, /*dim=*/1, /*start=*/0, /*length=*/src_shape[1]);
        TraceTensor dst_out_cpu = graph.createTensor(dst_shape.dims(), DataType::FLOAT32, DEVICE_CPU, main_stream_desc,
                                                     false);

        // Seed destination with sentinel values then issue strided D2D copy into the view.
        DeviceCopy(graph, dst_seed_cpu, dst_base_gpu, GpuStreamDescriptors::Main);
        DeviceCopy(graph, src_cpu, src_gpu, GpuStreamDescriptors::Main);
        DeviceCopy(graph, src_gpu, dst_view_gpu, GpuStreamDescriptors::Main);
        DeviceCopy(graph, dst_base_gpu, dst_out_cpu, GpuStreamDescriptors::Main);

        graph.finalize();

        ExecutionPlan plan = ExecutionPlan::FromGraph(graph,
                                                      {
                                                          {.name = "src", .tensor = src_host},
                                                          {.name = "dst_seed", .tensor = dst_init_host},
                                                      },
                                                      {});

        Executor executor{plan, execution_backend, 0};
        executor.execute(allocator_registry);

        const auto dst_result = executor.getOutput(dst_out_cpu);
        if (!dst_result)
        {
            throw std::runtime_error("Failed to fetch destination output.");
        }

        // Build expected host buffer: first cols copied from src, remainder remains sentinel.
        auto expected = MakeHostTensor(dst_shape[0], dst_shape[1], /*start=*/-1.0f, /*step=*/0.0f);
        const auto *src_data = static_cast<float *>(src_host->dataptr());
        auto *expected_data = static_cast<float *>(expected->dataptr());
        for (uint64_t r = 0; r < src_shape[0]; ++r)
        {
            for (uint64_t c = 0; c < src_shape[1]; ++c)
            {
                expected_data[r * dst_shape[1] + c] = src_data[r * src_shape[1] + c];
            }
        }

        testing::AssertSimilar(expected, *dst_result, 0.0);
    }

    void Run2DH2DCopyTest()
    {
        ExecutionBackend &execution_backend = ExecutionBackend::getInstance();
        const auto &allocator_registry = allocator::CachingAllocatorRegistry::instance();

        const Shape src_shape({3, 4});
        const uint64_t padding = 3;
        const Shape dst_shape({src_shape[0], src_shape[1] + padding});

        auto src_host = MakeHostTensor(src_shape[0], src_shape[1], /*start=*/2.0f, /*step=*/1.0f);
        auto dst_init_host =
            MakeHostTensor(dst_shape[0], dst_shape[1], /*start=*/-5.0f, /*step=*/0.0f); // sentinel fill

        TraceTensor src_cpu = TraceTensor::Create(src_shape.dims(), DataType::FLOAT32, DEVICE_CPU, main_stream_desc);
        src_cpu.markRetained();
        TraceTensor dst_seed_cpu = TraceTensor::Create(dst_shape.dims(), DataType::FLOAT32, DEVICE_CPU,
                                                       main_stream_desc);
        dst_seed_cpu.markRetained();

        OpGraph graph({{.name = "src", .tensor = src_cpu}, {.name = "dst_seed", .tensor = dst_seed_cpu}}, {});

        TraceTensor dst_base_gpu =
            graph.createTensor(dst_shape.dims(), DataType::FLOAT32, DEVICE_GPU, main_stream_desc, false);
        TraceTensor dst_view_gpu = dst_base_gpu.slice(graph, /*dim=*/1, /*start=*/0, /*length=*/src_shape[1]);
        TraceTensor dst_out_cpu = graph.createTensor(dst_shape.dims(), DataType::FLOAT32, DEVICE_CPU, main_stream_desc,
                                                     false);

        DeviceCopy(graph, dst_seed_cpu, dst_base_gpu, GpuStreamDescriptors::Main);
        DeviceCopy(graph, src_cpu, dst_view_gpu, GpuStreamDescriptors::Main);
        DeviceCopy(graph, dst_base_gpu, dst_out_cpu, GpuStreamDescriptors::Main);

        graph.finalize();

        ExecutionPlan plan = ExecutionPlan::FromGraph(graph,
                                                      {
                                                          {.name = "src", .tensor = src_host},
                                                          {.name = "dst_seed", .tensor = dst_init_host},
                                                      },
                                                      {});

        Executor executor{plan, execution_backend, 0};
        executor.execute(allocator_registry);

        const auto dst_result = executor.getOutput(dst_out_cpu);
        if (!dst_result)
        {
            throw std::runtime_error("Failed to fetch destination output for H2D strided copy.");
        }

        auto expected = MakeHostTensor(dst_shape[0], dst_shape[1], /*start=*/-5.0f, /*step=*/0.0f);
        const auto *src_data = static_cast<float *>(src_host->dataptr());
        auto *expected_data = static_cast<float *>(expected->dataptr());
        for (uint64_t r = 0; r < src_shape[0]; ++r)
        {
            for (uint64_t c = 0; c < src_shape[1]; ++c)
            {
                expected_data[r * dst_shape[1] + c] = src_data[r * src_shape[1] + c];
            }
        }

        testing::AssertSimilar(expected, *dst_result, 0.0);
    }

    void Run2DD2HCopyTest()
    {
        ExecutionBackend &execution_backend = ExecutionBackend::getInstance();
        const auto &allocator_registry = allocator::CachingAllocatorRegistry::instance();

        const Shape src_shape({3, 4});
        const uint64_t padding = 1;
        const Shape dst_shape({src_shape[0], src_shape[1] + padding});

        auto src_host = MakeHostTensor(src_shape[0], src_shape[1], /*start=*/3.0f, /*step=*/1.0f);
        auto dst_init_host =
            MakeHostTensor(dst_shape[0], dst_shape[1], /*start=*/-2.0f, /*step=*/0.0f); // sentinel fill

        TraceTensor src_cpu = TraceTensor::Create(src_shape.dims(), DataType::FLOAT32, DEVICE_CPU, main_stream_desc);
        src_cpu.markRetained();
        TraceTensor dst_seed_cpu = TraceTensor::Create(dst_shape.dims(), DataType::FLOAT32, DEVICE_CPU,
                                                       main_stream_desc);
        dst_seed_cpu.markRetained();

        OpGraph graph({{.name = "src", .tensor = src_cpu}, {.name = "dst_seed", .tensor = dst_seed_cpu}}, {});

        TraceTensor src_gpu_base =
            graph.createTensor(src_shape.dims(), DataType::FLOAT32, DEVICE_GPU, main_stream_desc, false);
        TraceTensor src_gpu_view = src_gpu_base.slice(graph, /*dim=*/1, /*start=*/0, /*length=*/src_shape[1]);
        TraceTensor dst_base_cpu = graph.createTensor(dst_shape.dims(), DataType::FLOAT32, DEVICE_CPU,
                                                      main_stream_desc, false);
        TraceTensor dst_view_cpu = dst_base_cpu.slice(graph, /*dim=*/1, /*start=*/0, /*length=*/src_shape[1]);

        DeviceCopy(graph, dst_seed_cpu, dst_base_cpu, GpuStreamDescriptors::Main);
        DeviceCopy(graph, src_cpu, src_gpu_base, GpuStreamDescriptors::Main);
        DeviceCopy(graph, src_gpu_view, dst_view_cpu, GpuStreamDescriptors::Main);

        graph.finalize();

        ExecutionPlan plan = ExecutionPlan::FromGraph(graph,
                                                      {
                                                          {.name = "src", .tensor = src_host},
                                                          {.name = "dst_seed", .tensor = dst_init_host},
                                                      },
                                                      {});

        Executor executor{plan, execution_backend, 0};
        executor.execute(allocator_registry);

        const auto dst_result = executor.getOutput(dst_base_cpu);
        if (!dst_result)
        {
            throw std::runtime_error("Failed to fetch destination output for D2H strided copy.");
        }

        auto expected = MakeHostTensor(dst_shape[0], dst_shape[1], /*start=*/-2.0f, /*step=*/0.0f);
        const auto *src_data = static_cast<float *>(src_host->dataptr());
        auto *expected_data = static_cast<float *>(expected->dataptr());
        for (uint64_t r = 0; r < src_shape[0]; ++r)
        {
            for (uint64_t c = 0; c < src_shape[1]; ++c)
            {
                expected_data[r * dst_shape[1] + c] = src_data[r * src_shape[1] + c];
            }
        }

        testing::AssertSimilar(expected, *dst_result, 0.0);
    }

    void Run3DCopyTest()
    {
        ExecutionBackend &execution_backend = ExecutionBackend::getInstance();
        const auto &allocator_registry = allocator::CachingAllocatorRegistry::instance();

        const Shape src_shape({2, 3, 4});
        const uint64_t padding_y = 2;
        const Shape dst_shape({src_shape[0], src_shape[1] + padding_y, src_shape[2]});

        auto src_host = MakeHostTensor3d(src_shape[0], src_shape[1], src_shape[2], /*start=*/1.0f, /*step=*/1.0f);
        auto dst_init_host =
            MakeHostTensor3d(dst_shape[0], dst_shape[1], dst_shape[2], /*start=*/-1.0f, /*step=*/0.0f); // sentinel

        TraceTensor src_cpu = TraceTensor::Create(src_shape.dims(), DataType::FLOAT32, DEVICE_CPU, main_stream_desc);
        src_cpu.markRetained();
        TraceTensor dst_seed_cpu = TraceTensor::Create(dst_shape.dims(), DataType::FLOAT32, DEVICE_CPU,
                                                       main_stream_desc);
        dst_seed_cpu.markRetained();

        OpGraph graph({{.name = "src", .tensor = src_cpu}, {.name = "dst_seed", .tensor = dst_seed_cpu}}, {});

        TraceTensor src_gpu = graph.createTensor(src_shape.dims(), DataType::FLOAT32, DEVICE_GPU, main_stream_desc, false);
        TraceTensor dst_base_gpu = graph.createTensor(dst_shape.dims(), DataType::FLOAT32, DEVICE_GPU, main_stream_desc, false);
        TraceTensor dst_view_gpu = dst_base_gpu.slice(graph, /*dim=*/1, /*start=*/0, /*length=*/src_shape[1]);
        TraceTensor dst_out_cpu = graph.createTensor(dst_shape.dims(), DataType::FLOAT32, DEVICE_CPU, main_stream_desc,
                                                     false);

        DeviceCopy(graph, dst_seed_cpu, dst_base_gpu, GpuStreamDescriptors::Main);
        DeviceCopy(graph, src_cpu, src_gpu, GpuStreamDescriptors::Main);
        DeviceCopy(graph, src_gpu, dst_view_gpu, GpuStreamDescriptors::Main);
        DeviceCopy(graph, dst_base_gpu, dst_out_cpu, GpuStreamDescriptors::Main);

        graph.finalize();

        ExecutionPlan plan = ExecutionPlan::FromGraph(graph,
                                                      {
                                                          {.name = "src", .tensor = src_host},
                                                          {.name = "dst_seed", .tensor = dst_init_host},
                                                      },
                                                      {});

        Executor executor{plan, execution_backend, 0};
        executor.execute(allocator_registry);

        const auto dst_result = executor.getOutput(dst_out_cpu);
        if (!dst_result)
        {
            throw std::runtime_error("Failed to fetch destination output (3D).");
        }

        auto expected = MakeHostTensor3d(dst_shape[0], dst_shape[1], dst_shape[2], /*start=*/-1.0f, /*step=*/0.0f);
        const auto *src_data = static_cast<float *>(src_host->dataptr());
        auto *expected_data = static_cast<float *>(expected->dataptr());
        for (uint64_t d = 0; d < src_shape[0]; ++d)
        {
            for (uint64_t r = 0; r < src_shape[1]; ++r)
            {
                for (uint64_t c = 0; c < src_shape[2]; ++c)
                {
                    const uint64_t src_idx = d * src_shape[1] * src_shape[2] + r * src_shape[2] + c;
                    const uint64_t dst_idx = d * dst_shape[1] * dst_shape[2] + r * dst_shape[2] + c;
                    expected_data[dst_idx] = src_data[src_idx];
                }
            }
        }

        testing::AssertSimilar(expected, *dst_result, 0.0);
    }

    void Run3DH2DCopyTest()
    {
        ExecutionBackend &execution_backend = ExecutionBackend::getInstance();
        const auto &allocator_registry = allocator::CachingAllocatorRegistry::instance();

        const Shape src_shape({2, 3, 4});
        const uint64_t padding_y = 1;
        const Shape dst_shape({src_shape[0], src_shape[1] + padding_y, src_shape[2]});

        auto src_host = MakeHostTensor3d(src_shape[0], src_shape[1], src_shape[2], /*start=*/5.0f, /*step=*/1.0f);
        auto dst_init_host =
            MakeHostTensor3d(dst_shape[0], dst_shape[1], dst_shape[2], /*start=*/-7.0f, /*step=*/0.0f); // sentinel

        TraceTensor src_cpu = TraceTensor::Create(src_shape.dims(), DataType::FLOAT32, DEVICE_CPU, main_stream_desc);
        src_cpu.markRetained();
        TraceTensor dst_seed_cpu = TraceTensor::Create(dst_shape.dims(), DataType::FLOAT32, DEVICE_CPU,
                                                       main_stream_desc);
        dst_seed_cpu.markRetained();

        OpGraph graph({{.name = "src", .tensor = src_cpu}, {.name = "dst_seed", .tensor = dst_seed_cpu}}, {});

        TraceTensor dst_base_gpu =
            graph.createTensor(dst_shape.dims(), DataType::FLOAT32, DEVICE_GPU, main_stream_desc, false);
        TraceTensor dst_view_gpu = dst_base_gpu.slice(graph, /*dim=*/1, /*start=*/0, /*length=*/src_shape[1]);
        TraceTensor dst_out_cpu = graph.createTensor(dst_shape.dims(), DataType::FLOAT32, DEVICE_CPU, main_stream_desc,
                                                     false);

        DeviceCopy(graph, dst_seed_cpu, dst_base_gpu, GpuStreamDescriptors::Main);
        DeviceCopy(graph, src_cpu, dst_view_gpu, GpuStreamDescriptors::Main);
        DeviceCopy(graph, dst_base_gpu, dst_out_cpu, GpuStreamDescriptors::Main);

        graph.finalize();

        ExecutionPlan plan = ExecutionPlan::FromGraph(graph,
                                                      {
                                                          {.name = "src", .tensor = src_host},
                                                          {.name = "dst_seed", .tensor = dst_init_host},
                                                      },
                                                      {});

        Executor executor{plan, execution_backend, 0};
        executor.execute(allocator_registry);

        const auto dst_result = executor.getOutput(dst_out_cpu);
        if (!dst_result)
        {
            throw std::runtime_error("Failed to fetch destination output (3D H2D).");
        }

        auto expected = MakeHostTensor3d(dst_shape[0], dst_shape[1], dst_shape[2], /*start=*/-7.0f, /*step=*/0.0f);
        const auto *src_data = static_cast<float *>(src_host->dataptr());
        auto *expected_data = static_cast<float *>(expected->dataptr());
        for (uint64_t d = 0; d < src_shape[0]; ++d)
        {
            for (uint64_t r = 0; r < src_shape[1]; ++r)
            {
                for (uint64_t c = 0; c < src_shape[2]; ++c)
                {
                    const uint64_t src_idx = d * src_shape[1] * src_shape[2] + r * src_shape[2] + c;
                    const uint64_t dst_idx = d * dst_shape[1] * dst_shape[2] + r * dst_shape[2] + c;
                    expected_data[dst_idx] = src_data[src_idx];
                }
            }
        }

        testing::AssertSimilar(expected, *dst_result, 0.0);
    }

    void Run3DCopySingle3dMemcpyTest()
    {
        ExecutionBackend &execution_backend = ExecutionBackend::getInstance();
        const auto &allocator_registry = allocator::CachingAllocatorRegistry::instance();

        // Padding only on inner-most dimension so slow path can use a single 3D pitched copy.
        const Shape base_shape({2, 3, 6});
        const uint64_t slice_len = 4;

        auto src_host = MakeHostTensor3d(base_shape[0], base_shape[1], base_shape[2], /*start=*/10.0f, /*step=*/1.0f);
        auto dst_init_host =
            MakeHostTensor3d(base_shape[0], base_shape[1], base_shape[2], /*start=*/-4.0f, /*step=*/0.0f);

        TraceTensor src_cpu = TraceTensor::Create(base_shape.dims(), DataType::FLOAT32, DEVICE_CPU, main_stream_desc);
        src_cpu.markRetained();
        TraceTensor dst_seed_cpu = TraceTensor::Create(base_shape.dims(), DataType::FLOAT32, DEVICE_CPU,
                                                       main_stream_desc);
        dst_seed_cpu.markRetained();

        OpGraph graph({{.name = "src", .tensor = src_cpu}, {.name = "dst_seed", .tensor = dst_seed_cpu}}, {});

        TraceTensor src_gpu_base =
            graph.createTensor(base_shape.dims(), DataType::FLOAT32, DEVICE_GPU, main_stream_desc, false);
        TraceTensor src_gpu_view = src_gpu_base.slice(graph, /*dim=*/2, /*start=*/0, /*length=*/slice_len);
        TraceTensor dst_gpu_base =
            graph.createTensor(base_shape.dims(), DataType::FLOAT32, DEVICE_GPU, main_stream_desc, false);
        TraceTensor dst_gpu_view = dst_gpu_base.slice(graph, /*dim=*/2, /*start=*/0, /*length=*/slice_len);
        TraceTensor dst_out_cpu =
            graph.createTensor(base_shape.dims(), DataType::FLOAT32, DEVICE_CPU, main_stream_desc, false);

        DeviceCopy(graph, dst_seed_cpu, dst_gpu_base, GpuStreamDescriptors::Main);
        DeviceCopy(graph, src_cpu, src_gpu_base, GpuStreamDescriptors::Main);
        DeviceCopy(graph, src_gpu_view, dst_gpu_view, GpuStreamDescriptors::Main);
        DeviceCopy(graph, dst_gpu_base, dst_out_cpu, GpuStreamDescriptors::Main);

        graph.finalize();

        ExecutionPlan plan = ExecutionPlan::FromGraph(graph,
                                                      {
                                                          {.name = "src", .tensor = src_host},
                                                          {.name = "dst_seed", .tensor = dst_init_host},
                                                      },
                                                      {});

        Executor executor{plan, execution_backend, 0};
        executor.execute(allocator_registry);

        const auto dst_result = executor.getOutput(dst_out_cpu);
        if (!dst_result)
        {
            throw std::runtime_error("Failed to fetch destination output (3D single memcpy D2D).");
        }

        auto expected = MakeHostTensor3d(base_shape[0], base_shape[1], base_shape[2], /*start=*/-4.0f, /*step=*/0.0f);
        const auto *src_data = static_cast<float *>(src_host->dataptr());
        auto *expected_data = static_cast<float *>(expected->dataptr());
        for (uint64_t d = 0; d < base_shape[0]; ++d)
        {
            for (uint64_t r = 0; r < base_shape[1]; ++r)
            {
                for (uint64_t c = 0; c < slice_len; ++c)
                {
                    const uint64_t src_idx = d * base_shape[1] * base_shape[2] + r * base_shape[2] + c;
                    const uint64_t dst_idx = d * base_shape[1] * base_shape[2] + r * base_shape[2] + c;
                    expected_data[dst_idx] = src_data[src_idx];
                }
            }
        }

        testing::AssertSimilar(expected, *dst_result, 0.0);
    }

    void Run3DH2DSingle3dMemcpyTest()
    {
        ExecutionBackend &execution_backend = ExecutionBackend::getInstance();
        const auto &allocator_registry = allocator::CachingAllocatorRegistry::instance();

        const Shape base_shape({2, 3, 6});
        const uint64_t slice_len = 4;

        auto src_host = MakeHostTensor3d(base_shape[0], base_shape[1], base_shape[2], /*start=*/20.0f, /*step=*/1.0f);
        auto dst_init_host =
            MakeHostTensor3d(base_shape[0], base_shape[1], base_shape[2], /*start=*/-6.0f, /*step=*/0.0f);

        TraceTensor src_cpu_base =
            TraceTensor::Create(base_shape.dims(), DataType::FLOAT32, DEVICE_CPU, main_stream_desc);
        src_cpu_base.markRetained();
        TraceTensor dst_seed_cpu =
            TraceTensor::Create(base_shape.dims(), DataType::FLOAT32, DEVICE_CPU, main_stream_desc);
        dst_seed_cpu.markRetained();

        OpGraph graph({{.name = "src", .tensor = src_cpu_base}, {.name = "dst_seed", .tensor = dst_seed_cpu}}, {});

        TraceTensor src_cpu_view = src_cpu_base.slice(graph, /*dim=*/2, /*start=*/0, /*length=*/slice_len);
        TraceTensor dst_gpu_base =
            graph.createTensor(base_shape.dims(), DataType::FLOAT32, DEVICE_GPU, main_stream_desc, false);
        TraceTensor dst_gpu_view = dst_gpu_base.slice(graph, /*dim=*/2, /*start=*/0, /*length=*/slice_len);
        TraceTensor dst_out_cpu =
            graph.createTensor(base_shape.dims(), DataType::FLOAT32, DEVICE_CPU, main_stream_desc, false);

        DeviceCopy(graph, dst_seed_cpu, dst_gpu_base, GpuStreamDescriptors::Main);
        DeviceCopy(graph, src_cpu_view, dst_gpu_view, GpuStreamDescriptors::Main);
        DeviceCopy(graph, dst_gpu_base, dst_out_cpu, GpuStreamDescriptors::Main);

        graph.finalize();

        ExecutionPlan plan = ExecutionPlan::FromGraph(graph,
                                                      {
                                                          {.name = "src", .tensor = src_host},
                                                          {.name = "dst_seed", .tensor = dst_init_host},
                                                      },
                                                      {});

        Executor executor{plan, execution_backend, 0};
        executor.execute(allocator_registry);

        const auto dst_result = executor.getOutput(dst_out_cpu);
        if (!dst_result)
        {
            throw std::runtime_error("Failed to fetch destination output (H2D single memcpy).");
        }

        auto expected = MakeHostTensor3d(base_shape[0], base_shape[1], base_shape[2], /*start=*/-6.0f, /*step=*/0.0f);
        const auto *src_data = static_cast<float *>(src_host->dataptr());
        auto *expected_data = static_cast<float *>(expected->dataptr());
        for (uint64_t d = 0; d < base_shape[0]; ++d)
        {
            for (uint64_t r = 0; r < base_shape[1]; ++r)
            {
                for (uint64_t c = 0; c < slice_len; ++c)
                {
                    const uint64_t src_idx = d * base_shape[1] * base_shape[2] + r * base_shape[2] + c;
                    const uint64_t dst_idx = d * base_shape[1] * base_shape[2] + r * base_shape[2] + c;
                    expected_data[dst_idx] = src_data[src_idx];
                }
            }
        }

        testing::AssertSimilar(expected, *dst_result, 0.0);
    }

    void Run3DD2HSingle3dMemcpyTest()
    {
        ExecutionBackend &execution_backend = ExecutionBackend::getInstance();
        const auto &allocator_registry = allocator::CachingAllocatorRegistry::instance();

        const Shape base_shape({2, 3, 6});
        const uint64_t slice_len = 4;

        auto src_host = MakeHostTensor3d(base_shape[0], base_shape[1], base_shape[2], /*start=*/30.0f, /*step=*/1.0f);
        auto dst_init_host =
            MakeHostTensor3d(base_shape[0], base_shape[1], base_shape[2], /*start=*/-8.0f, /*step=*/0.0f);

        TraceTensor src_cpu_base =
            TraceTensor::Create(base_shape.dims(), DataType::FLOAT32, DEVICE_CPU, main_stream_desc);
        src_cpu_base.markRetained();
        TraceTensor dst_seed_cpu =
            TraceTensor::Create(base_shape.dims(), DataType::FLOAT32, DEVICE_CPU, main_stream_desc);
        dst_seed_cpu.markRetained();

        OpGraph graph({{.name = "src", .tensor = src_cpu_base}, {.name = "dst_seed", .tensor = dst_seed_cpu}}, {});

        TraceTensor src_gpu_base =
            graph.createTensor(base_shape.dims(), DataType::FLOAT32, DEVICE_GPU, main_stream_desc, false);
        TraceTensor src_gpu_view = src_gpu_base.slice(graph, /*dim=*/2, /*start=*/0, /*length=*/slice_len);
        TraceTensor dst_cpu_base =
            graph.createTensor(base_shape.dims(), DataType::FLOAT32, DEVICE_CPU, main_stream_desc, false);
        TraceTensor dst_cpu_view = dst_cpu_base.slice(graph, /*dim=*/2, /*start=*/0, /*length=*/slice_len);

        DeviceCopy(graph, dst_seed_cpu, dst_cpu_base, GpuStreamDescriptors::Main);
        DeviceCopy(graph, src_cpu_base, src_gpu_base, GpuStreamDescriptors::Main);
        DeviceCopy(graph, src_gpu_view, dst_cpu_view, GpuStreamDescriptors::Main);

        graph.finalize();

        ExecutionPlan plan = ExecutionPlan::FromGraph(graph,
                                                      {
                                                          {.name = "src", .tensor = src_host},
                                                          {.name = "dst_seed", .tensor = dst_init_host},
                                                      },
                                                      {});

        Executor executor{plan, execution_backend, 0};
        executor.execute(allocator_registry);

        const auto dst_result = executor.getOutput(dst_cpu_base);
        if (!dst_result)
        {
            throw std::runtime_error("Failed to fetch destination output (D2H single memcpy).");
        }

        auto expected = MakeHostTensor3d(base_shape[0], base_shape[1], base_shape[2], /*start=*/-8.0f, /*step=*/0.0f);
        const auto *src_data = static_cast<float *>(src_host->dataptr());
        auto *expected_data = static_cast<float *>(expected->dataptr());
        for (uint64_t d = 0; d < base_shape[0]; ++d)
        {
            for (uint64_t r = 0; r < base_shape[1]; ++r)
            {
                for (uint64_t c = 0; c < slice_len; ++c)
                {
                    const uint64_t src_idx = d * base_shape[1] * base_shape[2] + r * base_shape[2] + c;
                    const uint64_t dst_idx = d * base_shape[1] * base_shape[2] + r * base_shape[2] + c;
                    expected_data[dst_idx] = src_data[src_idx];
                }
            }
        }

        testing::AssertSimilar(expected, *dst_result, 0.0);
    }

    void Run3DH2DFallback3dAs2dTest()
    {
        ExecutionBackend &execution_backend = ExecutionBackend::getInstance();
        const auto &allocator_registry = allocator::CachingAllocatorRegistry::instance();

        // Destination stride product mismatch forces the 3D slow path to fall back to multiple 2D copies.
        const Shape src_base_shape({2, 3, 6});
        const Shape dst_base_shape({2, 4, 4});
        const uint64_t slice_len = 4;

        auto src_host =
            MakeHostTensor3d(src_base_shape[0], src_base_shape[1], src_base_shape[2], /*start=*/40.0f, /*step=*/1.0f);
        auto dst_init_host = MakeHostTensor3d(dst_base_shape[0], dst_base_shape[1], dst_base_shape[2],
                                              /*start=*/-10.0f, /*step=*/0.0f);

        TraceTensor src_cpu_base =
            TraceTensor::Create(src_base_shape.dims(), DataType::FLOAT32, DEVICE_CPU, main_stream_desc);
        src_cpu_base.markRetained();
        TraceTensor dst_seed_cpu =
            TraceTensor::Create(dst_base_shape.dims(), DataType::FLOAT32, DEVICE_CPU, main_stream_desc);
        dst_seed_cpu.markRetained();

        OpGraph graph({{.name = "src", .tensor = src_cpu_base}, {.name = "dst_seed", .tensor = dst_seed_cpu}}, {});

        TraceTensor src_cpu_view = src_cpu_base.slice(graph, /*dim=*/2, /*start=*/0, /*length=*/slice_len);
        TraceTensor dst_gpu_base =
            graph.createTensor(dst_base_shape.dims(), DataType::FLOAT32, DEVICE_GPU, main_stream_desc, false);
        TraceTensor dst_gpu_view = dst_gpu_base.slice(graph, /*dim=*/1, /*start=*/0, /*length=*/src_base_shape[1]);
        TraceTensor dst_out_cpu =
            graph.createTensor(dst_base_shape.dims(), DataType::FLOAT32, DEVICE_CPU, main_stream_desc, false);

        DeviceCopy(graph, dst_seed_cpu, dst_gpu_base, GpuStreamDescriptors::Main);
        DeviceCopy(graph, src_cpu_view, dst_gpu_view, GpuStreamDescriptors::Main);
        DeviceCopy(graph, dst_gpu_base, dst_out_cpu, GpuStreamDescriptors::Main);

        graph.finalize();

        ExecutionPlan plan = ExecutionPlan::FromGraph(graph,
                                                      {
                                                          {.name = "src", .tensor = src_host},
                                                          {.name = "dst_seed", .tensor = dst_init_host},
                                                      },
                                                      {});

        Executor executor{plan, execution_backend, 0};
        executor.execute(allocator_registry);

        const auto dst_result = executor.getOutput(dst_out_cpu);
        if (!dst_result)
        {
            throw std::runtime_error("Failed to fetch destination output (H2D 3D fallback -> 2D).");
        }

        auto expected = MakeHostTensor3d(dst_base_shape[0], dst_base_shape[1], dst_base_shape[2], /*start=*/-10.0f,
                                         /*step=*/0.0f);
        const auto *src_data = static_cast<float *>(src_host->dataptr());
        auto *expected_data = static_cast<float *>(expected->dataptr());
        for (uint64_t d = 0; d < src_base_shape[0]; ++d)
        {
            for (uint64_t r = 0; r < src_base_shape[1]; ++r)
            {
                for (uint64_t c = 0; c < slice_len; ++c)
                {
                    const uint64_t src_idx = d * src_base_shape[1] * src_base_shape[2] + r * src_base_shape[2] + c;
                    const uint64_t dst_idx = d * dst_base_shape[1] * dst_base_shape[2] + r * dst_base_shape[2] + c;
                    expected_data[dst_idx] = src_data[src_idx];
                }
            }
        }

        testing::AssertSimilar(expected, *dst_result, 0.0);
    }
    // Broadcast (zero-stride) copies should replicate the single source row across the destination.
    void Run2DBroadcastH2DCopyTest()
    {
        ExecutionBackend &execution_backend = ExecutionBackend::getInstance();
        const auto &allocator_registry = allocator::CachingAllocatorRegistry::instance();

        const Shape src_base_shape({1, 4});
        const Shape dst_shape({3, 4});

        auto src_host = MakeHostTensor(src_base_shape[0], src_base_shape[1], /*start=*/31.0f, /*step=*/1.0f);

        TraceTensor src_cpu_base =
            TraceTensor::Create(src_base_shape.dims(), DataType::FLOAT32, DEVICE_CPU, main_stream_desc);
        src_cpu_base.markRetained();
        OpGraph graph({{.name = "src", .tensor = src_cpu_base}}, {});

        TraceTensor src_cpu_broadcast = src_cpu_base.broadcast(graph, /*dim=*/0, /*new_size=*/dst_shape[0]);
        TraceTensor dst_gpu = graph.createTensor(dst_shape.dims(), DataType::FLOAT32, DEVICE_GPU, main_stream_desc, false);
        TraceTensor dst_cpu_out =
            graph.createTensor(dst_shape.dims(), DataType::FLOAT32, DEVICE_CPU, main_stream_desc, false);

        DeviceCopy(graph, src_cpu_broadcast, dst_gpu, GpuStreamDescriptors::Main);
        DeviceCopy(graph, dst_gpu, dst_cpu_out, GpuStreamDescriptors::Main);

        graph.finalize();

        ExecutionPlan plan = ExecutionPlan::FromGraph(graph, {{.name = "src", .tensor = src_host}}, {});

        Executor executor{plan, execution_backend, 0};
        executor.execute(allocator_registry);

        const auto dst_result = executor.getOutput(dst_cpu_out);
        if (!dst_result)
        {
            throw std::runtime_error("Failed to fetch destination output (H2D broadcast).");
        }

        auto expected = MakeHostTensor(dst_shape[0], dst_shape[1], /*start=*/0.0f, /*step=*/0.0f);
        const auto *src_data = static_cast<float *>(src_host->dataptr());
        auto *exp_data = static_cast<float *>(expected->dataptr());
        for (uint64_t r = 0; r < dst_shape[0]; ++r)
        {
            for (uint64_t c = 0; c < dst_shape[1]; ++c)
            {
                exp_data[r * dst_shape[1] + c] = src_data[c];
            }
        }

        testing::AssertSimilar(expected, *dst_result, 0.0);
    }

    void Run2DBroadcastD2DCopyTest()
    {
        ExecutionBackend &execution_backend = ExecutionBackend::getInstance();
        const auto &allocator_registry = allocator::CachingAllocatorRegistry::instance();

        const Shape src_base_shape({1, 4});
        const Shape dst_shape({3, 4});

        auto src_host = MakeHostTensor(src_base_shape[0], src_base_shape[1], /*start=*/41.0f, /*step=*/1.0f);

        TraceTensor src_cpu_base =
            TraceTensor::Create(src_base_shape.dims(), DataType::FLOAT32, DEVICE_CPU, main_stream_desc);
        src_cpu_base.markRetained();
        OpGraph graph({{.name = "src", .tensor = src_cpu_base}}, {});

        TraceTensor src_gpu_base =
            graph.createTensor(src_base_shape.dims(), DataType::FLOAT32, DEVICE_GPU, main_stream_desc, false);
        TraceTensor src_gpu_broadcast = src_gpu_base.broadcast(graph, /*dim=*/0, /*new_size=*/dst_shape[0]);
        TraceTensor dst_gpu = graph.createTensor(dst_shape.dims(), DataType::FLOAT32, DEVICE_GPU, main_stream_desc, false);
        TraceTensor dst_cpu_out =
            graph.createTensor(dst_shape.dims(), DataType::FLOAT32, DEVICE_CPU, main_stream_desc, false);

        DeviceCopy(graph, src_cpu_base, src_gpu_base, GpuStreamDescriptors::Main);
        DeviceCopy(graph, src_gpu_broadcast, dst_gpu, GpuStreamDescriptors::Main);
        DeviceCopy(graph, dst_gpu, dst_cpu_out, GpuStreamDescriptors::Main);

        graph.finalize();

        ExecutionPlan plan = ExecutionPlan::FromGraph(graph, {{.name = "src", .tensor = src_host}}, {});

        Executor executor{plan, execution_backend, 0};
        executor.execute(allocator_registry);

        const auto dst_result = executor.getOutput(dst_cpu_out);
        if (!dst_result)
        {
            throw std::runtime_error("Failed to fetch destination output (D2D broadcast).");
        }

        auto expected = MakeHostTensor(dst_shape[0], dst_shape[1], /*start=*/0.0f, /*step=*/0.0f);
        const auto *src_data = static_cast<float *>(src_host->dataptr());
        auto *exp_data = static_cast<float *>(expected->dataptr());
        for (uint64_t r = 0; r < dst_shape[0]; ++r)
        {
            for (uint64_t c = 0; c < dst_shape[1]; ++c)
            {
                exp_data[r * dst_shape[1] + c] = src_data[c];
            }
        }

        testing::AssertSimilar(expected, *dst_result, 0.0);
    }

    void Run2DBroadcastD2HCopyTest()
    {
        ExecutionBackend &execution_backend = ExecutionBackend::getInstance();
        const auto &allocator_registry = allocator::CachingAllocatorRegistry::instance();

        const Shape src_base_shape({1, 4});
        const Shape dst_shape({3, 4});

        auto src_host = MakeHostTensor(src_base_shape[0], src_base_shape[1], /*start=*/51.0f, /*step=*/1.0f);

        TraceTensor src_cpu_base =
            TraceTensor::Create(src_base_shape.dims(), DataType::FLOAT32, DEVICE_CPU, main_stream_desc);
        src_cpu_base.markRetained();
        OpGraph graph({{.name = "src", .tensor = src_cpu_base}}, {});

        TraceTensor src_gpu_base =
            graph.createTensor(src_base_shape.dims(), DataType::FLOAT32, DEVICE_GPU, main_stream_desc, false);
        TraceTensor src_gpu_broadcast = src_gpu_base.broadcast(graph, /*dim=*/0, /*new_size=*/dst_shape[0]);
        TraceTensor dst_cpu_out =
            graph.createTensor(dst_shape.dims(), DataType::FLOAT32, DEVICE_CPU, main_stream_desc, false);

        DeviceCopy(graph, src_cpu_base, src_gpu_base, GpuStreamDescriptors::Main);
        DeviceCopy(graph, src_gpu_broadcast, dst_cpu_out, GpuStreamDescriptors::Main);

        graph.finalize();

        ExecutionPlan plan = ExecutionPlan::FromGraph(graph, {{.name = "src", .tensor = src_host}}, {});

        Executor executor{plan, execution_backend, 0};
        executor.execute(allocator_registry);

        const auto dst_result = executor.getOutput(dst_cpu_out);
        if (!dst_result)
        {
            throw std::runtime_error("Failed to fetch destination output (D2H broadcast).");
        }

        auto expected = MakeHostTensor(dst_shape[0], dst_shape[1], /*start=*/0.0f, /*step=*/0.0f);
        const auto *src_data = static_cast<float *>(src_host->dataptr());
        auto *exp_data = static_cast<float *>(expected->dataptr());
        for (uint64_t r = 0; r < dst_shape[0]; ++r)
        {
            for (uint64_t c = 0; c < dst_shape[1]; ++c)
            {
                exp_data[r * dst_shape[1] + c] = src_data[c];
            }
        }

        testing::AssertSimilar(expected, *dst_result, 0.0);
    }
} // namespace

int main()
{
    Run2DCopyTest();
    Run2DH2DCopyTest();
    Run2DD2HCopyTest();
    Run3DCopyTest();
    Run3DCopySingle3dMemcpyTest();
    Run3DH2DSingle3dMemcpyTest();
    Run3DD2HSingle3dMemcpyTest();
    Run3DH2DFallback3dAs2dTest();
    Run3DH2DCopyTest();
    Run2DBroadcastH2DCopyTest();
    Run2DBroadcastD2DCopyTest();
    Run2DBroadcastD2HCopyTest();
    return 0;
}
