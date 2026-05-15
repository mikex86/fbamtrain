#include "testing.h"

#include <execution_backend.h>
#include <executor.h>
#include <functional.h>
#include <op_graph.h>
#include <passes.h>
#include <tensorlib.h>
#include <utils.h>

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace pi::tensorlib;

namespace
{
    constexpr uint64_t BATCH = 8;
    constexpr uint64_t HIDDEN = 16;
    constexpr Device DEVICE_CPU{DeviceType::CPU, 0};
    constexpr Device DEVICE_GPU{DeviceType::GPU, 0};

    size_t Numel(const std::initializer_list<uint64_t> &shape)
    {
        size_t total = 1;
        for (const auto dim : shape)
        {
            total *= static_cast<size_t>(dim);
        }
        return total;
    }

    std::vector<float> MakePatternValues(const size_t count, const float scale, const float offset, const int mod)
    {
        std::vector<float> values(count);
        for (size_t i = 0; i < count; ++i)
        {
            const float base = static_cast<float>(static_cast<int>(i % static_cast<size_t>(mod)));
            values[i] = scale * base + offset;
        }
        return values;
    }

    std::shared_ptr<RealTensor> MakeHostTensorFloat32(const std::initializer_list<uint64_t> &shape,
                                                      const std::vector<float> &values)
    {
        if (values.size() != Numel(shape))
        {
            throw std::runtime_error("MakeHostTensorFloat32: value count mismatch");
        }
        auto tensor = RealTensor::Allocate(shape, DataType::FLOAT32, DEVICE_CPU);
        auto *dst = static_cast<float *>(tensor->dataptr());
        std::copy(values.begin(), values.end(), dst);
        return tensor;
    }

    std::shared_ptr<RealTensor> MakeHostTensorBf16(const std::initializer_list<uint64_t> &shape,
                                                   const std::vector<float> &values)
    {
        if (values.size() != Numel(shape))
        {
            throw std::runtime_error("MakeHostTensorBf16: value count mismatch");
        }
        auto tensor = RealTensor::Allocate(shape, DataType::BFLOAT16, DEVICE_CPU);
        auto *dst = static_cast<uint16_t *>(tensor->dataptr());
        for (size_t i = 0; i < values.size(); ++i)
        {
            dst[i] = utils::Bf16FromFp32(values[i]);
        }
        return tensor;
    }

    inline float Sigmoid(const float x)
    {
        return 1.0f / (1.0f + std::exp(-x));
    }

    inline float TanhLike(const float x)
    {
        const float t = std::exp(-2.0f * x);
        return (1.0f - t) / (1.0f + t);
    }

    inline size_t GateIndex(const size_t batch, const size_t hidden, const size_t gate, const size_t hidden_size)
    {
        return batch * (4 * hidden_size) + gate * hidden_size + hidden;
    }

    struct LstmFwdRef
    {
        std::vector<float> h_out;
        std::vector<float> c_out;
    };

    struct LstmRecomputeRef
    {
        std::vector<float> gate_out;
        std::vector<float> h_out;
        std::vector<float> c_out;
    };

    struct LstmBwdRef
    {
        std::vector<float> dGates;
        std::vector<float> dc_prev;
    };

    LstmFwdRef ComputeFwd(const std::vector<float> &gates, const std::vector<float> &c_prev)
    {
        LstmFwdRef ref{};
        ref.h_out.resize(BATCH * HIDDEN);
        ref.c_out.resize(BATCH * HIDDEN);

        for (size_t b = 0; b < BATCH; ++b)
        {
            for (size_t h = 0; h < HIDDEN; ++h)
            {
                const float gate_i = gates[GateIndex(b, h, 0, HIDDEN)];
                const float gate_f = gates[GateIndex(b, h, 1, HIDDEN)];
                const float gate_g = gates[GateIndex(b, h, 2, HIDDEN)];
                const float gate_o = gates[GateIndex(b, h, 3, HIDDEN)];

                const float i = Sigmoid(gate_i);
                const float f = Sigmoid(gate_f);
                const float g = TanhLike(gate_g);
                const float o = Sigmoid(gate_o);

                const float c_new = f * c_prev[b * HIDDEN + h] + i * g;
                const float h_new = o * TanhLike(c_new);

                ref.c_out[b * HIDDEN + h] = c_new;
                ref.h_out[b * HIDDEN + h] = h_new;
            }
        }

        return ref;
    }

    LstmRecomputeRef ComputeRecompute(const std::vector<float> &gates, const std::vector<float> &bias,
                                      const std::vector<float> &c_prev)
    {
        LstmRecomputeRef ref{};
        ref.gate_out.resize(BATCH * HIDDEN * 4);
        ref.h_out.resize(BATCH * HIDDEN);
        ref.c_out.resize(BATCH * HIDDEN);

        for (size_t b = 0; b < BATCH; ++b)
        {
            for (size_t h = 0; h < HIDDEN; ++h)
            {
                const float gate_i = gates[GateIndex(b, h, 0, HIDDEN)] + bias[h];
                const float gate_f = gates[GateIndex(b, h, 1, HIDDEN)] + bias[HIDDEN + h];
                const float gate_g = gates[GateIndex(b, h, 2, HIDDEN)] + bias[2 * HIDDEN + h];
                const float gate_o = gates[GateIndex(b, h, 3, HIDDEN)] + bias[3 * HIDDEN + h];

                const float i = Sigmoid(gate_i);
                const float f = Sigmoid(gate_f);
                const float g = TanhLike(gate_g);
                const float o = Sigmoid(gate_o);

                const float c_new = f * c_prev[b * HIDDEN + h] + i * g;
                const float h_new = o * TanhLike(c_new);

                ref.gate_out[GateIndex(b, h, 0, HIDDEN)] = i;
                ref.gate_out[GateIndex(b, h, 1, HIDDEN)] = f;
                ref.gate_out[GateIndex(b, h, 2, HIDDEN)] = g;
                ref.gate_out[GateIndex(b, h, 3, HIDDEN)] = o;
                ref.c_out[b * HIDDEN + h] = c_new;
                ref.h_out[b * HIDDEN + h] = h_new;
            }
        }

        return ref;
    }

    LstmBwdRef ComputeBwd(const std::vector<float> &dY, const std::vector<float> &dh_next,
                          const std::vector<float> &dc_next, const std::vector<float> &gate_out,
                          const std::vector<float> &c_prev, const std::vector<float> &c_out)
    {
        LstmBwdRef ref{};
        ref.dGates.resize(BATCH * HIDDEN * 4);
        ref.dc_prev.resize(BATCH * HIDDEN);

        for (size_t b = 0; b < BATCH; ++b)
        {
            for (size_t h = 0; h < HIDDEN; ++h)
            {
                const float i_gate = gate_out[GateIndex(b, h, 0, HIDDEN)];
                const float f_gate = gate_out[GateIndex(b, h, 1, HIDDEN)];
                const float g_gate = gate_out[GateIndex(b, h, 2, HIDDEN)];
                const float o_gate = gate_out[GateIndex(b, h, 3, HIDDEN)];

                const float c_prev_val = c_prev[b * HIDDEN + h];
                const float c_t = c_out[b * HIDDEN + h];
                const float tanh_c = TanhLike(c_t);
                const float dh_total = dY[b * HIDDEN + h] + dh_next[b * HIDDEN + h];
                const float dc_total =
                    dc_next[b * HIDDEN + h] + dh_total * o_gate * (1.0f - tanh_c * tanh_c);

                const float di = dc_total * g_gate;
                const float df = dc_total * c_prev_val;
                const float dg = dc_total * i_gate;
                const float do_gate = dh_total * tanh_c;

                const float dai = di * i_gate * (1.0f - i_gate);
                const float daf = df * f_gate * (1.0f - f_gate);
                const float dag = dg * (1.0f - g_gate * g_gate);
                const float dao = do_gate * o_gate * (1.0f - o_gate);

                ref.dGates[GateIndex(b, h, 0, HIDDEN)] = dai;
                ref.dGates[GateIndex(b, h, 1, HIDDEN)] = daf;
                ref.dGates[GateIndex(b, h, 2, HIDDEN)] = dag;
                ref.dGates[GateIndex(b, h, 3, HIDDEN)] = dao;
                ref.dc_prev[b * HIDDEN + h] = dc_total * f_gate;
            }
        }

        return ref;
    }

    void RunFwdBf16Case(ExecutionBackend &execution_backend,
                        const allocator::AllocatorRegistry &allocator_registry, const std::vector<float> &gates,
                        const std::vector<float> &c_prev, const LstmFwdRef &ref)
    {
        const auto main_stream_desc = GpuStreamDescriptors::Main;

        auto gates_host = MakeHostTensorFloat32({BATCH, 4 * HIDDEN}, gates);
        auto cprev_host = MakeHostTensorFloat32({BATCH, HIDDEN}, c_prev);
        auto expected_h = MakeHostTensorBf16({BATCH, HIDDEN}, ref.h_out);
        auto expected_c = MakeHostTensorFloat32({BATCH, HIDDEN}, ref.c_out);

        TraceTensor gates_cpu = TraceTensor::Create({BATCH, 4 * HIDDEN}, DataType::FLOAT32, DEVICE_CPU, main_stream_desc);
        TraceTensor cprev_cpu = TraceTensor::Create({BATCH, HIDDEN}, DataType::FLOAT32, DEVICE_CPU, main_stream_desc);
        gates_cpu.markRetained();
        cprev_cpu.markRetained();

        OpGraph graph({{.name = "gates", .tensor = gates_cpu}, {.name = "cprev", .tensor = cprev_cpu}}, {});

        TraceTensor gates_gpu = graph.createTensor({BATCH, 4 * HIDDEN}, DataType::FLOAT32, DEVICE_GPU, main_stream_desc, false);
        TraceTensor cprev_gpu = graph.createTensor({BATCH, HIDDEN}, DataType::FLOAT32, DEVICE_GPU, main_stream_desc, false);
        DeviceCopy(graph, gates_cpu, gates_gpu, GpuStreamDescriptors::Main);
        DeviceCopy(graph, cprev_cpu, cprev_gpu, GpuStreamDescriptors::Main);

        TraceTensor h_gpu = graph.createTensor({BATCH, HIDDEN}, DataType::BFLOAT16, DEVICE_GPU, main_stream_desc, false);
        TraceTensor c_gpu = graph.createTensor({BATCH, HIDDEN}, DataType::FLOAT32, DEVICE_GPU, main_stream_desc, false);
        TraceTensor y_gpu = graph.createTensor({BATCH, HIDDEN}, DataType::BFLOAT16, DEVICE_GPU, main_stream_desc, false);
        LstmCellFwdInplace(graph, gates_gpu, cprev_gpu, h_gpu, c_gpu, y_gpu, main_stream_desc);

        TraceTensor h_cpu = graph.createTensor({BATCH, HIDDEN}, DataType::BFLOAT16, DEVICE_CPU, main_stream_desc, false);
        TraceTensor y_cpu = graph.createTensor({BATCH, HIDDEN}, DataType::BFLOAT16, DEVICE_CPU, main_stream_desc, false);
        TraceTensor c_cpu = graph.createTensor({BATCH, HIDDEN}, DataType::FLOAT32, DEVICE_CPU, main_stream_desc, false);
        DeviceCopy(graph, h_gpu, h_cpu, GpuStreamDescriptors::Main);
        DeviceCopy(graph, y_gpu, y_cpu, GpuStreamDescriptors::Main);
        DeviceCopy(graph, c_gpu, c_cpu, GpuStreamDescriptors::Main);

        graph.finalize();

        ExecutionPlan plan = ExecutionPlan::FromGraph(graph, {{.name = "gates", .tensor = gates_host},
                                                             {.name = "cprev", .tensor = cprev_host}},
                                                      {});

        std::vector<std::unique_ptr<passes::CompilerPass>> passes{};
        passes.emplace_back(std::make_unique<LstmCellImplPass>());
        pi::tensorlib::passes::Transform(plan, passes);

        Executor executor{plan, execution_backend, 0};
        executor.execute(allocator_registry);

        const auto actual_h = executor.getOutput(h_cpu);
        const auto actual_y = executor.getOutput(y_cpu);
        const auto actual_c = executor.getOutput(c_cpu);
        if (!actual_h || !actual_y || !actual_c)
        {
            throw std::runtime_error("Failed to retrieve LSTM cell forward outputs (bf16 case)");
        }

        testing::AssertSimilar(expected_h, *actual_h, 1e-2);
        testing::AssertSimilar(expected_h, *actual_y, 1e-2);
        testing::AssertSimilar(expected_c, *actual_c, 1e-3);
    }

    void RunRecomputeBf16Case(ExecutionBackend &execution_backend,
                              const allocator::AllocatorRegistry &allocator_registry,
                              const std::vector<float> &gates, const std::vector<float> &bias,
                              const std::vector<float> &c_prev, const LstmRecomputeRef &ref)
    {
        const auto main_stream_desc = GpuStreamDescriptors::Main;

        auto gates_host = MakeHostTensorFloat32({BATCH, 4 * HIDDEN}, gates);
        auto bias_host = MakeHostTensorFloat32({4 * HIDDEN}, bias);
        auto cprev_host = MakeHostTensorFloat32({BATCH, HIDDEN}, c_prev);
        auto expected_gate_out = MakeHostTensorFloat32({BATCH, 4 * HIDDEN}, ref.gate_out);
        auto expected_h = MakeHostTensorBf16({BATCH, HIDDEN}, ref.h_out);
        auto expected_c = MakeHostTensorFloat32({BATCH, HIDDEN}, ref.c_out);

        TraceTensor gates_cpu = TraceTensor::Create({BATCH, 4 * HIDDEN}, DataType::FLOAT32, DEVICE_CPU, main_stream_desc);
        TraceTensor bias_cpu = TraceTensor::Create({4 * HIDDEN}, DataType::FLOAT32, DEVICE_CPU, main_stream_desc);
        TraceTensor cprev_cpu = TraceTensor::Create({BATCH, HIDDEN}, DataType::FLOAT32, DEVICE_CPU, main_stream_desc);
        gates_cpu.markRetained();
        bias_cpu.markRetained();
        cprev_cpu.markRetained();

        OpGraph graph({{.name = "gates", .tensor = gates_cpu},
                       {.name = "bias", .tensor = bias_cpu},
                       {.name = "cprev", .tensor = cprev_cpu}},
                      {});

        TraceTensor gates_gpu = graph.createTensor({BATCH, 4 * HIDDEN}, DataType::FLOAT32, DEVICE_GPU, main_stream_desc, false);
        TraceTensor bias_gpu = graph.createTensor({4 * HIDDEN}, DataType::FLOAT32, DEVICE_GPU, main_stream_desc, false);
        TraceTensor cprev_gpu = graph.createTensor({BATCH, HIDDEN}, DataType::FLOAT32, DEVICE_GPU, main_stream_desc, false);
        DeviceCopy(graph, gates_cpu, gates_gpu, GpuStreamDescriptors::Main);
        DeviceCopy(graph, bias_cpu, bias_gpu, GpuStreamDescriptors::Main);
        DeviceCopy(graph, cprev_cpu, cprev_gpu, GpuStreamDescriptors::Main);

        TraceTensor gate_out_gpu =
            graph.createTensor({BATCH, 4 * HIDDEN}, DataType::FLOAT32, DEVICE_GPU, main_stream_desc, false);
        TraceTensor h_out_gpu = graph.createTensor({BATCH, HIDDEN}, DataType::BFLOAT16, DEVICE_GPU, main_stream_desc, false);
        TraceTensor c_out_gpu = graph.createTensor({BATCH, HIDDEN}, DataType::FLOAT32, DEVICE_GPU, main_stream_desc, false);

        graph.recordOperation(OperationEntry{.type = OpType::LSTM_CELL_RECOMPUTE,
                                             .inputs = {gates_gpu, bias_gpu, cprev_gpu},
                                             .outputs = {gate_out_gpu, h_out_gpu, c_out_gpu},
                                             .attributes = {},
                                             .gpu_stream_desc = main_stream_desc});

        TraceTensor gate_out_cpu =
            graph.createTensor({BATCH, 4 * HIDDEN}, DataType::FLOAT32, DEVICE_CPU, main_stream_desc, false);
        TraceTensor h_out_cpu = graph.createTensor({BATCH, HIDDEN}, DataType::BFLOAT16, DEVICE_CPU, main_stream_desc, false);
        TraceTensor c_out_cpu = graph.createTensor({BATCH, HIDDEN}, DataType::FLOAT32, DEVICE_CPU, main_stream_desc, false);
        DeviceCopy(graph, gate_out_gpu, gate_out_cpu, GpuStreamDescriptors::Main);
        DeviceCopy(graph, h_out_gpu, h_out_cpu, GpuStreamDescriptors::Main);
        DeviceCopy(graph, c_out_gpu, c_out_cpu, GpuStreamDescriptors::Main);

        graph.finalize();

        ExecutionPlan plan = ExecutionPlan::FromGraph(graph, {{.name = "gates", .tensor = gates_host},
                                                             {.name = "bias", .tensor = bias_host},
                                                             {.name = "cprev", .tensor = cprev_host}},
                                                      {});

        std::vector<std::unique_ptr<passes::CompilerPass>> passes{};
        passes.emplace_back(std::make_unique<LstmCellBwdImplPass>());
        pi::tensorlib::passes::Transform(plan, passes);

        Executor executor{plan, execution_backend, 0};
        executor.execute(allocator_registry);

        const auto actual_gate_out = executor.getOutput(gate_out_cpu);
        const auto actual_h_out = executor.getOutput(h_out_cpu);
        const auto actual_c_out = executor.getOutput(c_out_cpu);
        if (!actual_gate_out || !actual_h_out || !actual_c_out)
        {
            throw std::runtime_error("Failed to retrieve LSTM cell recompute outputs (bf16 case)");
        }

        testing::AssertSimilar(expected_gate_out, *actual_gate_out, 1e-4);
        testing::AssertSimilar(expected_h, *actual_h_out, 1e-2);
        testing::AssertSimilar(expected_c, *actual_c_out, 1e-3);
    }

    void RunBwdBf16Case(ExecutionBackend &execution_backend,
                        const allocator::AllocatorRegistry &allocator_registry, const std::vector<float> &dY,
                        const std::vector<float> &dh_next, const std::vector<float> &dc_next,
                        const std::vector<float> &gate_out, const std::vector<float> &c_prev,
                        const std::vector<float> &c_out, const LstmBwdRef &ref)
    {
        const auto main_stream_desc = GpuStreamDescriptors::Main;

        auto dY_host = MakeHostTensorFloat32({BATCH, HIDDEN}, dY);
        auto dh_next_host = MakeHostTensorFloat32({BATCH, HIDDEN}, dh_next);
        auto dc_next_host = MakeHostTensorFloat32({BATCH, HIDDEN}, dc_next);
        auto gate_out_host = MakeHostTensorFloat32({BATCH, 4 * HIDDEN}, gate_out);
        auto c_prev_host = MakeHostTensorFloat32({BATCH, HIDDEN}, c_prev);
        auto c_out_host = MakeHostTensorFloat32({BATCH, HIDDEN}, c_out);
        auto expected_dgates = MakeHostTensorFloat32({BATCH, 4 * HIDDEN}, ref.dGates);
        auto expected_dgates_half = MakeHostTensorBf16({BATCH, 4 * HIDDEN}, ref.dGates);
        auto expected_dc_prev = MakeHostTensorFloat32({BATCH, HIDDEN}, ref.dc_prev);

        TraceTensor dY_cpu = TraceTensor::Create({BATCH, HIDDEN}, DataType::FLOAT32, DEVICE_CPU, main_stream_desc);
        TraceTensor dh_next_cpu = TraceTensor::Create({BATCH, HIDDEN}, DataType::FLOAT32, DEVICE_CPU, main_stream_desc);
        TraceTensor dc_next_cpu = TraceTensor::Create({BATCH, HIDDEN}, DataType::FLOAT32, DEVICE_CPU, main_stream_desc);
        TraceTensor gate_out_cpu = TraceTensor::Create({BATCH, 4 * HIDDEN}, DataType::FLOAT32, DEVICE_CPU, main_stream_desc);
        TraceTensor c_prev_cpu = TraceTensor::Create({BATCH, HIDDEN}, DataType::FLOAT32, DEVICE_CPU, main_stream_desc);
        TraceTensor c_out_cpu = TraceTensor::Create({BATCH, HIDDEN}, DataType::FLOAT32, DEVICE_CPU, main_stream_desc);
        dY_cpu.markRetained();
        dh_next_cpu.markRetained();
        dc_next_cpu.markRetained();
        gate_out_cpu.markRetained();
        c_prev_cpu.markRetained();
        c_out_cpu.markRetained();

        OpGraph graph({{.name = "dY", .tensor = dY_cpu},
                       {.name = "dh_next", .tensor = dh_next_cpu},
                       {.name = "dc_next", .tensor = dc_next_cpu},
                       {.name = "gate_out", .tensor = gate_out_cpu},
                       {.name = "c_prev", .tensor = c_prev_cpu},
                       {.name = "c_out", .tensor = c_out_cpu}},
                      {});

        TraceTensor dY_gpu = graph.createTensor({BATCH, HIDDEN}, DataType::FLOAT32, DEVICE_GPU, main_stream_desc, false);
        TraceTensor dh_next_gpu = graph.createTensor({BATCH, HIDDEN}, DataType::FLOAT32, DEVICE_GPU, main_stream_desc, false);
        TraceTensor dc_next_gpu = graph.createTensor({BATCH, HIDDEN}, DataType::FLOAT32, DEVICE_GPU, main_stream_desc, false);
        TraceTensor gate_out_gpu =
            graph.createTensor({BATCH, 4 * HIDDEN}, DataType::FLOAT32, DEVICE_GPU, main_stream_desc, false);
        TraceTensor c_prev_gpu = graph.createTensor({BATCH, HIDDEN}, DataType::FLOAT32, DEVICE_GPU, main_stream_desc, false);
        TraceTensor c_out_gpu = graph.createTensor({BATCH, HIDDEN}, DataType::FLOAT32, DEVICE_GPU, main_stream_desc, false);
        DeviceCopy(graph, dY_cpu, dY_gpu, GpuStreamDescriptors::Main);
        DeviceCopy(graph, dh_next_cpu, dh_next_gpu, GpuStreamDescriptors::Main);
        DeviceCopy(graph, dc_next_cpu, dc_next_gpu, GpuStreamDescriptors::Main);
        DeviceCopy(graph, gate_out_cpu, gate_out_gpu, GpuStreamDescriptors::Main);
        DeviceCopy(graph, c_prev_cpu, c_prev_gpu, GpuStreamDescriptors::Main);
        DeviceCopy(graph, c_out_cpu, c_out_gpu, GpuStreamDescriptors::Main);

        TraceTensor dGates_gpu =
            graph.createTensor({BATCH, 4 * HIDDEN}, DataType::FLOAT32, DEVICE_GPU, main_stream_desc, false);
        TraceTensor dGates_half_gpu =
            graph.createTensor({BATCH, 4 * HIDDEN}, DataType::BFLOAT16, DEVICE_GPU, main_stream_desc, false);
        TraceTensor dc_prev_gpu = graph.createTensor({BATCH, HIDDEN}, DataType::FLOAT32, DEVICE_GPU, main_stream_desc, false);

        graph.recordOperation(OperationEntry{.type = OpType::LSTM_CELL_BWD,
                                             .inputs = {dY_gpu, dh_next_gpu, dc_next_gpu, gate_out_gpu, c_prev_gpu, c_out_gpu},
                                             .outputs = {dGates_gpu, dGates_half_gpu, dc_prev_gpu},
                                             .attributes = {},
                                             .gpu_stream_desc = main_stream_desc});

        TraceTensor dGates_cpu =
            graph.createTensor({BATCH, 4 * HIDDEN}, DataType::FLOAT32, DEVICE_CPU, main_stream_desc, false);
        TraceTensor dGates_half_cpu =
            graph.createTensor({BATCH, 4 * HIDDEN}, DataType::BFLOAT16, DEVICE_CPU, main_stream_desc, false);
        TraceTensor dc_prev_cpu = graph.createTensor({BATCH, HIDDEN}, DataType::FLOAT32, DEVICE_CPU, main_stream_desc, false);
        DeviceCopy(graph, dGates_gpu, dGates_cpu, GpuStreamDescriptors::Main);
        DeviceCopy(graph, dGates_half_gpu, dGates_half_cpu, GpuStreamDescriptors::Main);
        DeviceCopy(graph, dc_prev_gpu, dc_prev_cpu, GpuStreamDescriptors::Main);

        graph.finalize();

        ExecutionPlan plan = ExecutionPlan::FromGraph(
            graph, {{.name = "dY", .tensor = dY_host},
                    {.name = "dh_next", .tensor = dh_next_host},
                    {.name = "dc_next", .tensor = dc_next_host},
                    {.name = "gate_out", .tensor = gate_out_host},
                    {.name = "c_prev", .tensor = c_prev_host},
                    {.name = "c_out", .tensor = c_out_host}},
            {});

        std::vector<std::unique_ptr<passes::CompilerPass>> passes{};
        passes.emplace_back(std::make_unique<LstmCellBwdImplPass>());
        pi::tensorlib::passes::Transform(plan, passes);

        Executor executor{plan, execution_backend, 0};
        executor.execute(allocator_registry);

        const auto actual_dgates = executor.getOutput(dGates_cpu);
        const auto actual_dgates_half = executor.getOutput(dGates_half_cpu);
        const auto actual_dc_prev = executor.getOutput(dc_prev_cpu);
        if (!actual_dgates || !actual_dgates_half || !actual_dc_prev)
        {
            throw std::runtime_error("Failed to retrieve LSTM cell backward outputs (bf16 case)");
        }

        testing::AssertSimilar(expected_dgates, *actual_dgates, 1e-3);
        testing::AssertSimilar(expected_dgates_half, *actual_dgates_half, 1e-2);
        testing::AssertSimilar(expected_dc_prev, *actual_dc_prev, 1e-3);
    }
} // namespace

int main()
{
    ExecutionBackend &execution_backend = ExecutionBackend::getInstance();
    const auto &allocator_registry = allocator::DefaultAllocatorRegistry::instance();

    const auto gates = MakePatternValues(static_cast<size_t>(BATCH * 4 * HIDDEN), 0.01f, -0.7f, 97);
    const auto c_prev = MakePatternValues(static_cast<size_t>(BATCH * HIDDEN), 0.02f, -0.3f, 53);
    const auto bias = MakePatternValues(static_cast<size_t>(4 * HIDDEN), 0.005f, -0.02f, 29);

    const auto dY = MakePatternValues(static_cast<size_t>(BATCH * HIDDEN), 0.015f, -0.25f, 41);
    const auto dh_next = MakePatternValues(static_cast<size_t>(BATCH * HIDDEN), 0.01f, -0.1f, 37);
    const auto dc_next = MakePatternValues(static_cast<size_t>(BATCH * HIDDEN), 0.02f, -0.2f, 31);

    const auto fwd_ref = ComputeFwd(gates, c_prev);
    const auto recompute_ref = ComputeRecompute(gates, bias, c_prev);
    const auto bwd_ref = ComputeBwd(dY, dh_next, dc_next, recompute_ref.gate_out, c_prev, recompute_ref.c_out);

    RunFwdBf16Case(execution_backend, allocator_registry, gates, c_prev, fwd_ref);
    RunRecomputeBf16Case(execution_backend, allocator_registry, gates, bias, c_prev, recompute_ref);
    RunBwdBf16Case(execution_backend, allocator_registry, dY, dh_next, dc_next, recompute_ref.gate_out, c_prev,
                   recompute_ref.c_out, bwd_ref);

    return 0;
}
