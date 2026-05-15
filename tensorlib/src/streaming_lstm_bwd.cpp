#include "streaming_lstm.h"

#include "functional.h"
#include "op_graph.h"
#include "transfer.h"

#include <algorithm>
#include <any>
#include <array>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace pi::tensorlib
{
    LstmBackwardStreamingResult
    StreamingLstmBwd(OpGraph &graph, const TraceTensor &x, const TraceTensor &h0, const TraceTensor &c0,
                     const TraceTensor &w_ih, const TraceTensor &w_hh, const TraceTensor &b_ih, const TraceTensor &b_hh,
                     const TraceTensor & /*y_cache*/, const TraceTensor &h_cache_host, const TraceTensor &c_cache_host,
                     const TraceTensor &dy_host, const std::optional<TraceTensor> &dh_n,
                     const std::optional<TraceTensor> &dc_n, int recompute_interval, size_t streaming_chunk_size,
                     const std::optional<TraceTensor> &dx_host_opt, const GpuStreamDescriptor &compute_stream_descriptor)
    {
        const Device device = w_ih.device();
        if (recompute_interval <= 0)
        {
            throw std::runtime_error("StreamingLstm backward requires positive recompute_interval");
        }

        if (x.shape().ndims() != 3)
        {
            throw std::runtime_error("StreamingLstm backward expects x to have shape (T, B, I)");
        }
        if (x.device().device_type != DeviceType::CPU)
        {
            throw std::runtime_error("StreamingLstm backward expects x to reside on CPU");
        }
        const auto io_dtype = x.dtype();
        if (io_dtype != DataType::FLOAT16 && io_dtype != DataType::BFLOAT16)
        {
            throw std::runtime_error("StreamingLstm backward expects x to be FLOAT16 or BFLOAT16");
        }

        if (h_cache_host.device().device_type != DeviceType::CPU ||
            c_cache_host.device().device_type != DeviceType::CPU || dy_host.device().device_type != DeviceType::CPU)
        {
            throw std::runtime_error("StreamingLstm backward expects caches and dY to reside on CPU");
        }
        if (h_cache_host.dtype() != DataType::FLOAT16 && h_cache_host.dtype() != DataType::BFLOAT16)
        {
            throw std::runtime_error("StreamingLstm backward expects h_cache to be FLOAT16 or BFLOAT16");
        }
        if (h_cache_host.dtype() != io_dtype)
        {
            throw std::runtime_error("StreamingLstm backward expects h_cache dtype to match x dtype");
        }
        if (c_cache_host.dtype() != DataType::FLOAT16 && c_cache_host.dtype() != DataType::FLOAT32)
        {
            throw std::runtime_error("StreamingLstm backward expects c_cache to be FLOAT16 or FLOAT32");
        }
        if (dy_host.dtype() != DataType::FLOAT16 && dy_host.dtype() != DataType::BFLOAT16 &&
            dy_host.dtype() != DataType::FLOAT32)
        {
            throw std::runtime_error("StreamingLstm backward expects dy to be FLOAT16, BFLOAT16, or FLOAT32");
        }

        if (h0.dtype() != DataType::FLOAT32 && h0.dtype() != DataType::FLOAT16)
        {
            throw std::runtime_error("StreamingLstm backward expects h0 to be FLOAT16 or FLOAT32");
        }
        if (c0.dtype() != DataType::FLOAT32 && c0.dtype() != DataType::FLOAT16)
        {
            throw std::runtime_error("StreamingLstm backward expects c0 to be FLOAT16 or FLOAT32");
        }
        if (w_ih.dtype() != DataType::FLOAT32 || w_hh.dtype() != DataType::FLOAT32 ||
            b_ih.dtype() != DataType::FLOAT32 || b_hh.dtype() != DataType::FLOAT32)
        {
            throw std::runtime_error("StreamingLstm backward expects weights/bias to be FLOAT32");
        }

        if (w_ih.device().device_type != DeviceType::GPU || w_hh.device().device_type != DeviceType::GPU ||
            b_ih.device().device_type != DeviceType::GPU || b_hh.device().device_type != DeviceType::GPU)
        {
            throw std::runtime_error("StreamingLstm backward expects weights/bias to be on GPU");
        }
        if (h0.device().device_type != DeviceType::GPU || c0.device().device_type != DeviceType::GPU)
        {
            throw std::runtime_error("StreamingLstm backward expects h0/c0 to be on GPU");
        }
        if (w_ih.device() != w_hh.device() || w_ih.device() != b_ih.device() || w_ih.device() != b_hh.device())
        {
            throw std::runtime_error("StreamingLstm backward expects all weights/bias to be on the same device");
        }
        if (h0.device() != w_ih.device() || c0.device() != w_ih.device())
        {
            throw std::runtime_error("StreamingLstm backward expects h0/c0 on the same device as weights");
        }

        const size_t seq_len = x.shape()[0];
        const size_t batch = x.shape()[1];
        const size_t input_size = x.shape()[2];
        const size_t hidden_size = w_hh.shape()[0];
        const size_t gate_dim = 4 * hidden_size;

        if (h0.shape().ndims() != 1 || h0.shape()[0] != hidden_size)
        {
            throw std::runtime_error("StreamingLstm backward expects h0 to be a 1D tensor of size H");
        }
        if (c0.shape().ndims() != 1 || c0.shape()[0] != hidden_size)
        {
            throw std::runtime_error("StreamingLstm backward expects c0 to be a 1D tensor of size H");
        }

        TraceTensor h0_view = h0.view(graph, {1, hidden_size});
        TraceTensor c0_view = c0.view(graph, {1, hidden_size});

        const size_t expected_c_checkpoints =
            (seq_len + static_cast<size_t>(recompute_interval) - 1) / static_cast<size_t>(recompute_interval);
        if (c_cache_host.shape()[0] < expected_c_checkpoints)
        {
            throw std::runtime_error("StreamingLstm backward expects enough cell checkpoints for recomputation");
        }
        if (h_cache_host.shape()[0] < seq_len)
        {
            throw std::runtime_error("StreamingLstm backward expects h_cache to cover the full sequence");
        }

        OpGraphGpuTxRange range = graph.createGpuTxRange("pi::tensorlib::StreamingLstmBwd");

        TraceTensor w_ih_cast =
            graph.createTensor(w_ih.shape().dims(), io_dtype, device, compute_stream_descriptor, false);
        TraceTensor w_hh_cast =
            graph.createTensor(w_hh.shape().dims(), io_dtype, device, compute_stream_descriptor, false);
        graph.recordOperation(OperationEntry{.type = OpType::CAST,
                                             .inputs = {w_ih},
                                             .outputs = {w_ih_cast},
                                             .attributes = {},
                                             .gpu_stream_desc = compute_stream_descriptor});
        graph.recordOperation(OperationEntry{.type = OpType::CAST,
                                             .inputs = {w_hh},
                                             .outputs = {w_hh_cast},
                                             .attributes = {},
                                             .gpu_stream_desc = compute_stream_descriptor});

        TraceTensor w_concat = graph.createTensor({input_size + hidden_size, gate_dim}, io_dtype, device,
                                                  compute_stream_descriptor, false);
        TraceTensor w_concat_top = w_concat.slice(graph, 0, 0, input_size);
        TraceTensor w_concat_bottom = w_concat.slice(graph, 0, input_size, hidden_size);
        DeviceCopy(graph, w_ih_cast, w_concat_top, compute_stream_descriptor);
        DeviceCopy(graph, w_hh_cast, w_concat_bottom, compute_stream_descriptor);

        TraceTensor b_sum = graph.createTensor({gate_dim}, DataType::FLOAT32, device, compute_stream_descriptor, false);
        graph.recordOperation(OperationEntry{.type = OpType::PLUS,
                                             .inputs = {b_ih, b_hh},
                                             .outputs = {b_sum},
                                             .attributes = {},
                                             .gpu_stream_desc = compute_stream_descriptor});

        TraceTensor h0_cast = h0_view;
        if (h0_view.dtype() != io_dtype)
        {
            h0_cast =
                graph.createTensor(h0_view.shape().dims(), io_dtype, device, compute_stream_descriptor, false);
            graph.recordOperation(OperationEntry{.type = OpType::CAST,
                                                 .inputs = {h0_view},
                                                 .outputs = {h0_cast},
                                                 .attributes = {},
                                                 .gpu_stream_desc = compute_stream_descriptor});
        }

        bool casted_c0 = false;
        TraceTensor c0_fp32 = c0_view;
        if (c0_view.dtype() != DataType::FLOAT32)
        {
            casted_c0 = true;
            c0_fp32 =
                graph.createTensor(c0_view.shape().dims(), DataType::FLOAT32, device, compute_stream_descriptor, false);
            graph.recordOperation(OperationEntry{.type = OpType::CAST,
                                                 .inputs = {c0_view},
                                                 .outputs = {c0_fp32},
                                                 .attributes = {},
                                                 .gpu_stream_desc = compute_stream_descriptor});
        }
        TraceTensor h0_broadcast = (h0_cast.shape()[0] == batch) ? h0_cast : h0_cast.broadcast(graph, 0, batch);
        TraceTensor c0_broadcast = (c0_fp32.shape()[0] == batch) ? c0_fp32 : c0_fp32.broadcast(graph, 0, batch);

        TraceTensor grad_w_concat = graph.createTensor({input_size + hidden_size, gate_dim}, DataType::FLOAT32, device,
                                                       compute_stream_descriptor, false);
        TraceTensor grad_w_ih = grad_w_concat.slice(graph, 0, 0, input_size);
        TraceTensor grad_w_hh = grad_w_concat.slice(graph, 0, input_size, hidden_size);
        TraceTensor grad_b_ih =
            graph.createTensor(b_ih.shape().dims(), DataType::FLOAT32, device, compute_stream_descriptor, false);
        TraceTensor grad_b_hh =
            graph.createTensor(b_hh.shape().dims(), DataType::FLOAT32, device, compute_stream_descriptor, false);
        FillZeros(graph, grad_w_concat, GpuStreamDescriptors::Main);
        FillZeros(graph, grad_b_ih, GpuStreamDescriptors::Main);
        FillZeros(graph, grad_b_hh, GpuStreamDescriptors::Main);

        auto validate_dx_host = [&](const TraceTensor &dx_host_tensor)
        {
            if (dx_host_tensor.device().device_type != DeviceType::CPU)
            {
                throw std::runtime_error("StreamingLstm backward expects dx_host to reside on CPU");
            }
            if (dx_host_tensor.dtype() != DataType::FLOAT32)
            {
                throw std::runtime_error("StreamingLstm backward expects dx_host to be FLOAT32");
            }
            if (dx_host_tensor.shape() != x.shape())
            {
                throw std::runtime_error("StreamingLstm backward dx_host shape mismatch with x");
            }
            if (!dx_host_tensor.pinned())
            {
                throw std::runtime_error("StreamingLstm backward expects dx_host to be pinned host memory");
            }
        };

        TraceTensor dx_dev =
            graph.createTensor(x.shape().dims(), DataType::FLOAT32, device, compute_stream_descriptor, false);
        TraceTensor dx_host = dx_host_opt.has_value() ? *dx_host_opt
                                                      : graph.createTensor(x.shape().dims(), DataType::FLOAT32,
                                                                           Device{DeviceType::CPU, 0}, compute_stream_descriptor, true);
        if (dx_host_opt.has_value())
        {
            validate_dx_host(dx_host);
        }

        TraceTensor dh_next =
            graph.createTensor({batch, w_hh.shape()[0]}, DataType::FLOAT32, device, compute_stream_descriptor, false);
        TraceTensor dc_next =
            graph.createTensor({batch, w_hh.shape()[0]}, DataType::FLOAT32, device, compute_stream_descriptor, false);
        FillZeros(graph, dh_next, GpuStreamDescriptors::Main);
        FillZeros(graph, dc_next, GpuStreamDescriptors::Main);
        if (dh_n.has_value())
        {
            DeviceCopy(graph, *dh_n, dh_next, compute_stream_descriptor);
        }
        if (dc_n.has_value())
        {
            DeviceCopy(graph, *dc_n, dc_next, compute_stream_descriptor);
        }

        TraceTensor w_ih_T = w_ih_cast.transpose(graph, {1, 0});
        TraceTensor w_hh_T = w_hh_cast.transpose(graph, {1, 0});

        TraceTensor dGates_fp32 =
            graph.createTensor({batch, gate_dim}, DataType::FLOAT32, device, compute_stream_descriptor, false);
        TraceTensor dc_prev =
            graph.createTensor({batch, hidden_size}, DataType::FLOAT32, device, compute_stream_descriptor, false);
        TraceTensor dh_buffer =
            graph.createTensor({batch, hidden_size}, DataType::FLOAT32, device, compute_stream_descriptor, false);

        const size_t chunk_capacity = (streaming_chunk_size > 0) ? streaming_chunk_size : seq_len;
        const size_t recompute_capacity = chunk_capacity + static_cast<size_t>(recompute_interval);
        std::unordered_map<std::string, std::any> accumulate_attr{};
        accumulate_attr.emplace("accumulate_output", true);

        TraceTensor gates_buf = graph.createTensor({static_cast<uint64_t>(recompute_capacity), batch, gate_dim},
                                                   DataType::FLOAT32, device, compute_stream_descriptor, false);
        TraceTensor gate_out_buf = graph.createTensor({static_cast<uint64_t>(recompute_capacity), batch, gate_dim},
                                                      DataType::FLOAT32, device, compute_stream_descriptor, false);
        TraceTensor c_buf = graph.createTensor({static_cast<uint64_t>(recompute_capacity), batch, hidden_size},
                                               DataType::FLOAT32, device, compute_stream_descriptor, false);
        TraceTensor dGates_chunk = graph.createTensor({static_cast<uint64_t>(chunk_capacity), batch, gate_dim},
                                                      io_dtype, device, compute_stream_descriptor, false);

        const auto recompute_interval_size = static_cast<size_t>(recompute_interval);
        const size_t batch_size = batch;

        // Double-buffered combined input staging for x/h to keep a single GEMM per step.
        const uint64_t combined_capacity = recompute_capacity + 1;
        std::array combined_input_buffers{
            graph.createTensor({combined_capacity, batch, input_size + hidden_size}, io_dtype, device,
                               compute_stream_descriptor, false),
            graph.createTensor({combined_capacity, batch, input_size + hidden_size}, io_dtype, device,
                               compute_stream_descriptor, false)};
        std::array combined_x_buffers{combined_input_buffers[0].slice(graph, 2, 0, input_size),
                                      combined_input_buffers[1].slice(graph, 2, 0, input_size)};
        std::array combined_h_buffers{combined_input_buffers[0].slice(graph, 2, input_size, hidden_size),
                                      combined_input_buffers[1].slice(graph, 2, input_size, hidden_size)};
        std::array<TraceTensor, 2> dy_buffers_fp32{
            graph.createTensor({static_cast<uint64_t>(chunk_capacity), batch, hidden_size}, DataType::FLOAT32, device,
                               compute_stream_descriptor, false),
            graph.createTensor({static_cast<uint64_t>(chunk_capacity), batch, hidden_size}, DataType::FLOAT32, device,
                               compute_stream_descriptor, false)};
        std::array<TraceTensor, 2> c_checkpoint_buffers_fp32{
            graph.createTensor({batch, hidden_size}, DataType::FLOAT32, device, compute_stream_descriptor, false),
            graph.createTensor({batch, hidden_size}, DataType::FLOAT32, device, compute_stream_descriptor, false)};

        std::array<OpGraph::GpuEventHandle, 2> h2d_events{graph.createGpuEvent(device), graph.createGpuEvent(device)};
        std::array<bool, 2> h2d_events_recorded{false, false};
        std::array<OpGraph::GpuEventHandle, 2> compute_done_events{graph.createGpuEvent(device),
                                                                   graph.createGpuEvent(device)};
        std::array<bool, 2> compute_done_recorded{false, false};

        const bool dy_needs_cast = (dy_host.dtype() != DataType::FLOAT32);
        std::optional<std::array<TraceTensor, 2>> dy_buffers_lowp;
        if (dy_needs_cast)
        {
            dy_buffers_lowp = std::array<TraceTensor, 2>{
                graph.createTensor({static_cast<uint64_t>(chunk_capacity), batch, hidden_size}, dy_host.dtype(),
                                   device, compute_stream_descriptor, false),
                graph.createTensor({static_cast<uint64_t>(chunk_capacity), batch, hidden_size}, dy_host.dtype(),
                                   device, compute_stream_descriptor, false)};
        }

        const bool c_cache_needs_cast = (c_cache_host.dtype() != DataType::FLOAT32);
        std::optional<std::array<TraceTensor, 2>> c_checkpoint_buffers_fp16_opt;
        if (c_cache_needs_cast)
        {
            c_checkpoint_buffers_fp16_opt = std::array{
                graph.createTensor({batch, hidden_size}, DataType::FLOAT16, device, compute_stream_descriptor, false),
                graph.createTensor({batch, hidden_size}, DataType::FLOAT16, device, compute_stream_descriptor, false)};
        }

        struct ChunkMeta
        {
            uint64_t chunk_start{};
            uint64_t chunk_len{};
            uint64_t recompute_start{};
            uint64_t steps_to_recompute{};
            uint64_t prefix_steps{};
            bool has_checkpoint{};
            uint64_t c_cache_index{};
        };

        std::array<ChunkMeta, 2> chunk_meta{};
        std::array<bool, 2> chunk_valid{false, false};

        const size_t total_chunks = (seq_len + chunk_capacity - 1) / chunk_capacity;

        auto prefetch_chunk = [&](const uint64_t chunk_id, const int buffer_idx)
        {
            if (chunk_id >= total_chunks)
            {
                chunk_valid[buffer_idx] = false;
                return;
            }

            const uint64_t chunk_start = chunk_id * chunk_capacity;
            const uint64_t chunk_len = std::min<uint64_t>(chunk_capacity, seq_len - chunk_start);
            const uint64_t chunk_end = chunk_start + chunk_len;
            const bool has_checkpoint = chunk_start != 0;
            const uint64_t recompute_start =
                has_checkpoint ? ((chunk_start - 1) / recompute_interval_size) * recompute_interval_size + 1 : 0;
            const uint64_t steps_to_recompute = chunk_end - recompute_start;
            const uint64_t prefix_steps = chunk_start - recompute_start;

            if (steps_to_recompute > recompute_capacity)
            {
                throw std::runtime_error("StreamingLstm backward recompute span exceeds staging buffer");
            }

            ChunkMeta meta{};
            meta.chunk_start = chunk_start;
            meta.chunk_len = chunk_len;
            meta.recompute_start = recompute_start;
            meta.steps_to_recompute = steps_to_recompute;
            meta.prefix_steps = prefix_steps;
            meta.has_checkpoint = has_checkpoint;
            meta.c_cache_index = has_checkpoint ? (recompute_start - 1) / recompute_interval_size : 0;
            chunk_meta[buffer_idx] = meta;
            chunk_valid[buffer_idx] = true;

            TraceTensor x_src = x.slice(graph, 0, recompute_start, steps_to_recompute);
            TraceTensor x_dst = combined_x_buffers[buffer_idx].slice(graph, 0, 0, steps_to_recompute);
            DeviceCopy(graph, x_src, x_dst, GpuStreamDescriptors::H2D);

            TraceTensor dy_src = dy_host.slice(graph, 0, chunk_start, chunk_len);
            if (dy_needs_cast)
            {
                TraceTensor dy_dst = (*dy_buffers_lowp)[buffer_idx].slice(graph, 0, 0, chunk_len);
                DeviceCopy(graph, dy_src, dy_dst, GpuStreamDescriptors::H2D);
            }
            else
            {
                TraceTensor dy_dst = dy_buffers_fp32[buffer_idx].slice(graph, 0, 0, chunk_len);
                DeviceCopy(graph, dy_src, dy_dst, GpuStreamDescriptors::H2D);
            }

            if (has_checkpoint)
            {
                TraceTensor h_src = h_cache_host.at(graph, 0, recompute_start - 1);
                TraceTensor h_dst = combined_h_buffers[buffer_idx].at(graph, 0, 0);
                DeviceCopy(graph, h_src, h_dst, GpuStreamDescriptors::H2D);

                TraceTensor c_src = c_cache_host.at(graph, 0, meta.c_cache_index);
                if (c_cache_needs_cast)
                {
                    const auto &c_checkpoint_buffers_fp16 = *c_checkpoint_buffers_fp16_opt;
                    DeviceCopy(graph, c_src, c_checkpoint_buffers_fp16[buffer_idx], GpuStreamDescriptors::H2D);
                }
                else
                {
                    DeviceCopy(graph, c_src, c_checkpoint_buffers_fp32[buffer_idx], GpuStreamDescriptors::H2D);
                }
            }

            graph.recordGpuEvent(h2d_events[buffer_idx], GpuStreamDescriptors::H2D);
            h2d_events_recorded[buffer_idx] = true;
        };

        size_t processed_chunks = 0;
        int compute_idx = 0;
        int prefetch_idx = 1;

        if (total_chunks == 0)
        {
            throw std::runtime_error("StreamingLstm backward requires non-empty sequence");
        }

        // Ensure forward D2H transfers complete before we start H2D prefetching.
        AwaitAsyncTransfers(graph, TransferType::D2H, device, compute_stream_descriptor);
        AwaitAsyncTransfers(graph, TransferType::D2H, device, GpuStreamDescriptors::H2D);

        int64_t next_chunk_id = static_cast<int64_t>(total_chunks) - 1;
        prefetch_chunk(static_cast<uint64_t>(next_chunk_id), compute_idx);
        next_chunk_id--;

        while (processed_chunks < total_chunks)
        {
            if (!chunk_valid[compute_idx])
            {
                break;
            }

            const ChunkMeta &meta = chunk_meta[compute_idx];

            // Ensure the prefetched H2D transfers for this chunk are complete before use.
            if (h2d_events_recorded[compute_idx])
            {
                graph.awaitGpuEvent(h2d_events[compute_idx], compute_stream_descriptor);
                h2d_events_recorded[compute_idx] = false;
            }

            const bool has_next_chunk = next_chunk_id >= 0;
            const bool delay_prefetch = has_next_chunk && processed_chunks == 0;
            auto queue_next_chunk = [&]() -> bool
            {
                if (next_chunk_id < 0)
                {
                    chunk_valid[prefetch_idx] = false;
                    return false;
                }
                if (compute_done_recorded[prefetch_idx])
                {
                    graph.awaitGpuEvent(compute_done_events[prefetch_idx], GpuStreamDescriptors::H2D);
                    compute_done_recorded[prefetch_idx] = false;
                }
                prefetch_chunk(static_cast<uint64_t>(next_chunk_id), prefetch_idx);
                --next_chunk_id;
                return true;
            };

            if (!delay_prefetch)
            {
                queue_next_chunk();
            }

            // Prepare chunk views.
            TraceTensor combined_input_chunk_slice =
                combined_input_buffers[compute_idx].slice(graph, 0, 0, meta.steps_to_recompute);
            TraceTensor combined_h_chunk =
                combined_h_buffers[compute_idx].slice(graph, 0, 0, meta.steps_to_recompute + 1);
            TraceTensor dy_chunk_fp32 = dy_buffers_fp32[compute_idx].slice(graph, 0, 0, meta.chunk_len);
            if (dy_needs_cast)
            {
                TraceTensor dy_lowp = (*dy_buffers_lowp)[compute_idx].slice(graph, 0, 0, meta.chunk_len);
                graph.recordOperation(OperationEntry{.type = OpType::CAST,
                                                     .inputs = {dy_lowp},
                                                     .outputs = {dy_chunk_fp32},
                                                     .attributes = {},
                                                     .gpu_stream_desc = compute_stream_descriptor});
            }

            TraceTensor c_prev_checkpoint = c0_broadcast;
            if (meta.has_checkpoint)
            {
                if (c_cache_needs_cast)
                {
                    const auto &c_checkpoint_buffers_fp16 = *c_checkpoint_buffers_fp16_opt;
                    TraceTensor c_half = c_checkpoint_buffers_fp16[compute_idx];
                    TraceTensor c_fp32 = c_checkpoint_buffers_fp32[compute_idx];
                    graph.recordOperation(OperationEntry{.type = OpType::CAST,
                                                         .inputs = {c_half},
                                                         .outputs = {c_fp32},
                                                         .attributes = {},
                                                         .gpu_stream_desc = compute_stream_descriptor});
                    c_prev_checkpoint = c_fp32;
                }
                else
                {
                    c_prev_checkpoint = c_checkpoint_buffers_fp32[compute_idx];
                }
            }

            TraceTensor h_seed = combined_h_chunk.at(graph, 0, 0);
            if (!meta.has_checkpoint)
            {
                DeviceCopy(graph, h0_broadcast, h_seed, compute_stream_descriptor);
            }

            TraceTensor gates_recompute = gates_buf.slice(graph, 0, 0, meta.steps_to_recompute);
            TraceTensor gate_out_recompute = gate_out_buf.slice(graph, 0, 0, meta.steps_to_recompute);
            TraceTensor c_recompute = c_buf.slice(graph, 0, 0, meta.steps_to_recompute);

            TraceTensor dGates_chunk_slice = dGates_chunk.slice(graph, 0, 0, meta.chunk_len);

            // Recompute forward states for the span.
            TraceTensor c_prev = c_prev_checkpoint;
            bool prefetch_queued = !delay_prefetch;
            for (size_t idx = 0; idx < meta.steps_to_recompute; ++idx)
            {
                TraceTensor combined_input_step = combined_input_chunk_slice.at(graph, 0, idx);
                TraceTensor gates_step = gates_recompute.at(graph, 0, idx);
                TraceTensor gate_out_step = gate_out_recompute.at(graph, 0, idx);
                TraceTensor c_out_step = c_recompute.at(graph, 0, idx);
                TraceTensor h_out_step = combined_h_chunk.at(graph, 0, idx + 1);

                graph.recordOperation(OperationEntry{.type = OpType::MATMUL,
                                                     .inputs = {combined_input_step, w_concat},
                                                     .outputs = {gates_step},
                                                     .is_useful = false,
                                                     .attributes = {},
                                                     .gpu_stream_desc = compute_stream_descriptor});

                graph.recordOperation(OperationEntry{.type = OpType::LSTM_CELL_RECOMPUTE,
                                                     .inputs = {gates_step, b_sum, c_prev},
                                                     .outputs = {gate_out_step, h_out_step, c_out_step},
                                                     .is_useful = false,
                                                     .attributes = {},
                                                     .gpu_stream_desc = compute_stream_descriptor});
                c_prev = c_out_step;

                if (!prefetch_queued)
                {
                    prefetch_queued = queue_next_chunk();
                }
            }

            // Backward pass for this chunk.
            for (size_t local = meta.chunk_len; local-- > 0;)
            {
                const size_t recompute_idx = meta.prefix_steps + local;
                TraceTensor dY_t = dy_chunk_fp32.at(graph, 0, local);

                TraceTensor c_prev_step =
                    (recompute_idx == 0) ? c_prev_checkpoint : c_recompute.at(graph, 0, recompute_idx - 1);
                TraceTensor c_out_step = c_recompute.at(graph, 0, recompute_idx);
                TraceTensor gate_out_step = gate_out_recompute.at(graph, 0, recompute_idx);
                TraceTensor dGates_half = dGates_chunk_slice.at(graph, 0, local);

                graph.recordOperation(
                    OperationEntry{.type = OpType::LSTM_CELL_BWD,
                                   .inputs = {dY_t, dh_next, dc_next, gate_out_step, c_prev_step, c_out_step},
                                   .outputs = {dGates_fp32, dGates_half, dc_prev},
                                   .attributes = {},
                                   .gpu_stream_desc = compute_stream_descriptor});

                graph.recordOperation(OperationEntry{.type = OpType::MATMUL,
                                                     .inputs = {dGates_half, w_hh_T},
                                                     .outputs = {dh_buffer},
                                                     .attributes = {},
                                                     .gpu_stream_desc = compute_stream_descriptor});
                dh_next = dh_buffer;
                dc_next = dc_prev;
            }

            const size_t chunk_batch = meta.chunk_len * batch_size;
            TraceTensor dGates_chunk_flat = dGates_chunk_slice.view(graph, {chunk_batch, gate_dim});

            // dx and D2H transfer kick off as soon as dx is ready.
            TraceTensor dx_chunk = dx_dev.slice(graph, 0, meta.chunk_start, meta.chunk_len);
            TraceTensor dx_flat = dx_chunk.view(graph, {chunk_batch, input_size});
            graph.recordOperation(OperationEntry{.type = OpType::MATMUL,
                                                 .inputs = {dGates_chunk_flat, w_ih_T},
                                                 .outputs = {dx_flat},
                                                 .attributes = {},
                                                 .gpu_stream_desc = compute_stream_descriptor});

            TraceTensor dx_host_chunk = dx_host.slice(graph, 0, meta.chunk_start, meta.chunk_len);
            AwaitComputeForTransfer(graph, TransferType::D2H, device, compute_stream_descriptor);
            DeviceCopy(graph, dx_chunk, dx_host_chunk, GpuStreamDescriptors::D2H);

            // Weight/bias gradients.
            TraceTensor combined_input_chunk_grad =
                combined_input_buffers[compute_idx].slice(graph, 0, meta.prefix_steps, meta.chunk_len);
            TraceTensor combined_input_flat =
                combined_input_chunk_grad.view(graph, {chunk_batch, input_size + hidden_size});
            TraceTensor combined_input_T = combined_input_flat.transpose(graph, {1, 0});
            graph.recordOperation(OperationEntry{.type = OpType::MATMUL,
                                                 .inputs = {combined_input_T, dGates_chunk_flat},
                                                 .outputs = {grad_w_concat},
                                                 .attributes = accumulate_attr,
                                                 .gpu_stream_desc = compute_stream_descriptor});

            TraceTensor db_vec = ReduceSum(graph, dGates_chunk_flat, /*dim=*/0, /*keepdim=*/false,
                                           DataType::FLOAT32, compute_stream_descriptor);
            InplaceAdd(graph, grad_b_ih, db_vec, compute_stream_descriptor);
            InplaceAdd(graph, grad_b_hh, db_vec, compute_stream_descriptor);
            graph.deleteTensor(db_vec);

            graph.recordGpuEvent(compute_done_events[compute_idx], compute_stream_descriptor);
            compute_done_recorded[compute_idx] = true;

            processed_chunks++;
            if (processed_chunks < total_chunks)
            {
                std::swap(compute_idx, prefetch_idx);
            }
        }

        AwaitAsyncTransfers(graph, TransferType::D2H, device, compute_stream_descriptor);

        graph.deleteTensor(w_ih_cast);
        graph.deleteTensor(w_hh_cast);
        graph.deleteTensor(w_concat);
        graph.deleteTensor(b_sum);
        if (h0_cast.id() != h0_view.id())
        {
            graph.deleteTensor(h0_cast);
        }
        graph.deleteTensor(gates_buf);
        graph.deleteTensor(gate_out_buf);
        graph.deleteTensor(c_buf);
        graph.deleteTensor(dGates_chunk);
        graph.deleteTensor(dGates_fp32);

        for (auto &buf : combined_input_buffers)
        {
            graph.deleteTensor(buf);
        }
        for (auto &buf : dy_buffers_fp32)
        {
            graph.deleteTensor(buf);
        }
        for (auto &buf : c_checkpoint_buffers_fp32)
        {
            graph.deleteTensor(buf);
        }
        if (dy_buffers_lowp.has_value())
        {
            for (auto &buf : *dy_buffers_lowp)
            {
                graph.deleteTensor(buf);
            }
        }
        if (c_checkpoint_buffers_fp16_opt.has_value())
        {
            for (auto &buf : *c_checkpoint_buffers_fp16_opt)
            {
                graph.deleteTensor(buf);
            }
        }
        if (casted_c0)
        {
            graph.deleteTensor(c0_fp32);
        }
        graph.deleteTensor(dx_dev);

        TraceTensor grad_h0 =
            ReduceSum(graph, dh_next, /*dim=*/0, /*keepdim=*/false, DataType::FLOAT32, compute_stream_descriptor);
        TraceTensor grad_c0 =
            ReduceSum(graph, dc_next, /*dim=*/0, /*keepdim=*/false, DataType::FLOAT32, compute_stream_descriptor);

        return LstmBackwardStreamingResult{.grad_x = dx_host,
                                           .grad_h0 = grad_h0,
                                           .grad_c0 = grad_c0,
                                           .grad_w_ih = grad_w_ih,
                                           .grad_w_hh = grad_w_hh,
                                           .grad_b_ih = grad_b_ih,
                                           .grad_b_hh = grad_b_hh};
    }

    LstmBackwardStreamingResult StreamingLstmBwdStreamedDy(
        OpGraph &graph, const TraceTensor &x, const TraceTensor &h0, const TraceTensor &c0, const TraceTensor &w_ih,
        const TraceTensor &w_hh, const TraceTensor &b_ih, const TraceTensor &b_hh, const TraceTensor & /*y_cache*/,
        const TraceTensor &h_cache_host, const TraceTensor &c_cache_host, const StreamingDySupplier &dy_supplier,
        const GpuStreamDescriptor &dy_supplier_stream_desc, const DataType dy_dtype,
        const std::optional<TraceTensor> &dh_n, const std::optional<TraceTensor> &dc_n, const int recompute_interval,
        const size_t streaming_chunk_size, const std::optional<TraceTensor> &dx_host_opt,
        const GpuStreamDescriptor &compute_stream_descriptor)
    {
        const Device device = w_ih.device();
        if (recompute_interval <= 0)
        {
            throw std::runtime_error("StreamingLstm backward requires positive recompute_interval");
        }
        if (!dy_supplier)
        {
            throw std::runtime_error("StreamingLstm backward requires a valid dy_supplier");
        }
        if (!dy_supplier_stream_desc.isValid())
        {
            throw std::runtime_error("StreamingLstm backward requires a valid dy_supplier_stream_desc");
        }
        const int dy_supplier_stream_id = dy_supplier_stream_desc.getStreamId();
        if (dy_supplier_stream_id < 0)
        {
            throw std::runtime_error(
                "StreamingLstm backward requires dy_supplier_stream_desc to target a compute stream");
        }
        if (dy_dtype != DataType::FLOAT16 && dy_dtype != DataType::BFLOAT16 && dy_dtype != DataType::FLOAT32)
        {
            throw std::runtime_error("StreamingLstm backward expects dy_dtype to be FLOAT16, BFLOAT16, or FLOAT32");
        }

        if (x.shape().ndims() != 3)
        {
            throw std::runtime_error("StreamingLstm backward expects x to have shape (T, B, I)");
        }
        if (x.device().device_type != DeviceType::CPU)
        {
            throw std::runtime_error("StreamingLstm backward expects x to reside on CPU");
        }
        const auto io_dtype = x.dtype();
        if (io_dtype != DataType::FLOAT16 && io_dtype != DataType::BFLOAT16)
        {
            throw std::runtime_error("StreamingLstm backward expects x to be FLOAT16 or BFLOAT16");
        }

        if (h_cache_host.device().device_type != DeviceType::CPU ||
            c_cache_host.device().device_type != DeviceType::CPU)
        {
            throw std::runtime_error("StreamingLstm backward expects caches to reside on CPU");
        }
        if (h_cache_host.dtype() != DataType::FLOAT16 && h_cache_host.dtype() != DataType::BFLOAT16)
        {
            throw std::runtime_error("StreamingLstm backward expects h_cache to be FLOAT16 or BFLOAT16");
        }
        if (h_cache_host.dtype() != io_dtype)
        {
            throw std::runtime_error("StreamingLstm backward expects h_cache dtype to match x dtype");
        }
        if (c_cache_host.dtype() != DataType::FLOAT16 && c_cache_host.dtype() != DataType::FLOAT32)
        {
            throw std::runtime_error("StreamingLstm backward expects c_cache to be FLOAT16 or FLOAT32");
        }

        if (h0.dtype() != DataType::FLOAT32 && h0.dtype() != DataType::FLOAT16)
        {
            throw std::runtime_error("StreamingLstm backward expects h0 to be FLOAT16 or FLOAT32");
        }
        if (c0.dtype() != DataType::FLOAT32 && c0.dtype() != DataType::FLOAT16)
        {
            throw std::runtime_error("StreamingLstm backward expects c0 to be FLOAT16 or FLOAT32");
        }
        if (w_ih.dtype() != DataType::FLOAT32 || w_hh.dtype() != DataType::FLOAT32 ||
            b_ih.dtype() != DataType::FLOAT32 || b_hh.dtype() != DataType::FLOAT32)
        {
            throw std::runtime_error("StreamingLstm backward expects weights/bias to be FLOAT32");
        }

        if (w_ih.device().device_type != DeviceType::GPU || w_hh.device().device_type != DeviceType::GPU ||
            b_ih.device().device_type != DeviceType::GPU || b_hh.device().device_type != DeviceType::GPU)
        {
            throw std::runtime_error("StreamingLstm backward expects weights/bias to be on GPU");
        }
        if (h0.device().device_type != DeviceType::GPU || c0.device().device_type != DeviceType::GPU)
        {
            throw std::runtime_error("StreamingLstm backward expects h0/c0 to be on GPU");
        }
        if (w_ih.device() != w_hh.device() || w_ih.device() != b_ih.device() || w_ih.device() != b_hh.device())
        {
            throw std::runtime_error("StreamingLstm backward expects all weights/bias to be on the same device");
        }
        if (h0.device() != w_ih.device() || c0.device() != w_ih.device())
        {
            throw std::runtime_error("StreamingLstm backward expects h0/c0 on the same device as weights");
        }

        const size_t seq_len = x.shape()[0];
        const size_t batch = x.shape()[1];
        const size_t input_size = x.shape()[2];
        const size_t hidden_size = w_hh.shape()[0];
        const size_t gate_dim = 4 * hidden_size;

        if (h0.shape().ndims() != 1 || h0.shape()[0] != hidden_size)
        {
            throw std::runtime_error("StreamingLstm backward expects h0 to be a 1D tensor of size H");
        }
        if (c0.shape().ndims() != 1 || c0.shape()[0] != hidden_size)
        {
            throw std::runtime_error("StreamingLstm backward expects c0 to be a 1D tensor of size H");
        }

        TraceTensor h0_view = h0.view(graph, {1, hidden_size});
        TraceTensor c0_view = c0.view(graph, {1, hidden_size});

        const size_t expected_c_checkpoints =
            (seq_len + static_cast<size_t>(recompute_interval) - 1) / static_cast<size_t>(recompute_interval);
        if (c_cache_host.shape()[0] < expected_c_checkpoints)
        {
            throw std::runtime_error("StreamingLstm backward expects enough cell checkpoints for recomputation");
        }
        if (h_cache_host.shape()[0] < seq_len)
        {
            throw std::runtime_error("StreamingLstm backward expects h_cache to cover the full sequence");
        }

        TraceTensor w_ih_cast =
            graph.createTensor(w_ih.shape().dims(), io_dtype, device, compute_stream_descriptor, false);
        TraceTensor w_hh_cast =
            graph.createTensor(w_hh.shape().dims(), io_dtype, device, compute_stream_descriptor, false);
        graph.recordOperation(OperationEntry{.type = OpType::CAST,
                                             .inputs = {w_ih},
                                             .outputs = {w_ih_cast},
                                             .attributes = {},
                                             .gpu_stream_desc = compute_stream_descriptor});
        graph.recordOperation(OperationEntry{.type = OpType::CAST,
                                             .inputs = {w_hh},
                                             .outputs = {w_hh_cast},
                                             .attributes = {},
                                             .gpu_stream_desc = compute_stream_descriptor});

        TraceTensor w_concat = graph.createTensor({input_size + hidden_size, gate_dim}, io_dtype, device,
                                                  compute_stream_descriptor, false);
        TraceTensor w_concat_top = w_concat.slice(graph, 0, 0, input_size);
        TraceTensor w_concat_bottom = w_concat.slice(graph, 0, input_size, hidden_size);
        DeviceCopy(graph, w_ih_cast, w_concat_top, compute_stream_descriptor);
        DeviceCopy(graph, w_hh_cast, w_concat_bottom, compute_stream_descriptor);

        TraceTensor b_sum = graph.createTensor({gate_dim}, DataType::FLOAT32, device, compute_stream_descriptor, false);
        graph.recordOperation(OperationEntry{.type = OpType::PLUS,
                                             .inputs = {b_ih, b_hh},
                                             .outputs = {b_sum},
                                             .attributes = {},
                                             .gpu_stream_desc = compute_stream_descriptor});

        TraceTensor h0_cast = h0_view;
        if (h0_view.dtype() != io_dtype)
        {
            h0_cast =
                graph.createTensor(h0_view.shape().dims(), io_dtype, device, compute_stream_descriptor, false);
            graph.recordOperation(OperationEntry{.type = OpType::CAST,
                                                 .inputs = {h0_view},
                                                 .outputs = {h0_cast},
                                                 .attributes = {},
                                                 .gpu_stream_desc = compute_stream_descriptor});
        }

        bool casted_c0 = false;
        TraceTensor c0_fp32 = c0_view;
        if (c0_view.dtype() != DataType::FLOAT32)
        {
            casted_c0 = true;
            c0_fp32 =
                graph.createTensor(c0_view.shape().dims(), DataType::FLOAT32, device, compute_stream_descriptor, false);
            graph.recordOperation(OperationEntry{.type = OpType::CAST,
                                                 .inputs = {c0_view},
                                                 .outputs = {c0_fp32},
                                                 .attributes = {},
                                                 .gpu_stream_desc = compute_stream_descriptor});
        }
        TraceTensor h0_broadcast = (h0_cast.shape()[0] == batch) ? h0_cast : h0_cast.broadcast(graph, 0, batch);
        TraceTensor c0_broadcast = (c0_fp32.shape()[0] == batch) ? c0_fp32 : c0_fp32.broadcast(graph, 0, batch);

        TraceTensor grad_w_concat = graph.createTensor({input_size + hidden_size, gate_dim}, DataType::FLOAT32, device,
                                                       compute_stream_descriptor, false);
        TraceTensor grad_w_ih = grad_w_concat.slice(graph, 0, 0, input_size);
        TraceTensor grad_w_hh = grad_w_concat.slice(graph, 0, input_size, hidden_size);
        TraceTensor grad_b_ih =
            graph.createTensor(b_ih.shape().dims(), DataType::FLOAT32, device, compute_stream_descriptor, false);
        TraceTensor grad_b_hh =
            graph.createTensor(b_hh.shape().dims(), DataType::FLOAT32, device, compute_stream_descriptor, false);
        FillZeros(graph, grad_w_concat, compute_stream_descriptor);
        FillZeros(graph, grad_b_ih, compute_stream_descriptor);
        FillZeros(graph, grad_b_hh, compute_stream_descriptor);

        auto validate_dx_host = [&](const TraceTensor &dx_host_tensor)
        {
            if (dx_host_tensor.device().device_type != DeviceType::CPU)
            {
                throw std::runtime_error("StreamingLstm backward expects dx_host to reside on CPU");
            }
            if (dx_host_tensor.dtype() != DataType::FLOAT32)
            {
                throw std::runtime_error("StreamingLstm backward expects dx_host to be FLOAT32");
            }
            if (dx_host_tensor.shape() != x.shape())
            {
                throw std::runtime_error("StreamingLstm backward dx_host shape mismatch with x");
            }
            if (!dx_host_tensor.pinned())
            {
                throw std::runtime_error("StreamingLstm backward expects dx_host to be pinned host memory");
            }
        };

        TraceTensor dx_dev =
            graph.createTensor(x.shape().dims(), DataType::FLOAT32, device, compute_stream_descriptor, false);
        TraceTensor dx_host = dx_host_opt.has_value() ? *dx_host_opt
                                                      : graph.createTensor(x.shape().dims(), DataType::FLOAT32,
                                                                           Device{DeviceType::CPU, 0}, compute_stream_descriptor, true);
        if (dx_host_opt.has_value())
        {
            validate_dx_host(dx_host);
        }

        TraceTensor dh_next =
            graph.createTensor({batch, w_hh.shape()[0]}, DataType::FLOAT32, device, compute_stream_descriptor, false);
        TraceTensor dc_next =
            graph.createTensor({batch, w_hh.shape()[0]}, DataType::FLOAT32, device, compute_stream_descriptor, false);
        FillZeros(graph, dh_next, compute_stream_descriptor);
        FillZeros(graph, dc_next, compute_stream_descriptor);
        if (dh_n.has_value())
        {
            DeviceCopy(graph, *dh_n, dh_next, compute_stream_descriptor);
        }
        if (dc_n.has_value())
        {
            DeviceCopy(graph, *dc_n, dc_next, compute_stream_descriptor);
        }

        TraceTensor w_ih_T = w_ih_cast.transpose(graph, {1, 0});
        TraceTensor w_hh_T = w_hh_cast.transpose(graph, {1, 0});

        TraceTensor dGates_fp32 =
            graph.createTensor({batch, gate_dim}, DataType::FLOAT32, device, compute_stream_descriptor, false);
        TraceTensor dc_prev =
            graph.createTensor({batch, hidden_size}, DataType::FLOAT32, device, compute_stream_descriptor, false);
        TraceTensor dh_buffer =
            graph.createTensor({batch, hidden_size}, DataType::FLOAT32, device, compute_stream_descriptor, false);

        const size_t chunk_capacity = (streaming_chunk_size > 0) ? streaming_chunk_size : seq_len;
        const size_t recompute_capacity = chunk_capacity + static_cast<size_t>(recompute_interval);
        std::unordered_map<std::string, std::any> accumulate_attr{};
        accumulate_attr.emplace("accumulate_output", true);

        TraceTensor gates_buf = graph.createTensor({static_cast<uint64_t>(recompute_capacity), batch, gate_dim},
                                                   DataType::FLOAT32, device, compute_stream_descriptor, false);
        TraceTensor gate_out_buf = graph.createTensor({static_cast<uint64_t>(recompute_capacity), batch, gate_dim},
                                                      DataType::FLOAT32, device, compute_stream_descriptor, false);
        TraceTensor c_buf = graph.createTensor({static_cast<uint64_t>(recompute_capacity), batch, hidden_size},
                                               DataType::FLOAT32, device, compute_stream_descriptor, false);
        TraceTensor dGates_chunk = graph.createTensor({static_cast<uint64_t>(chunk_capacity), batch, gate_dim},
                                                      io_dtype, device, compute_stream_descriptor, false);

        const auto recompute_interval_size = static_cast<size_t>(recompute_interval);
        const size_t batch_size = batch;

        // Double-buffered combined input staging for x/h to keep a single GEMM per step.
        const uint64_t combined_capacity = recompute_capacity + 1;
        std::array combined_input_buffers{
            graph.createTensor({combined_capacity, batch, input_size + hidden_size}, io_dtype, device,
                               compute_stream_descriptor, false),
            graph.createTensor({combined_capacity, batch, input_size + hidden_size}, io_dtype, device,
                               compute_stream_descriptor, false)};
        std::array combined_x_buffers{combined_input_buffers[0].slice(graph, 2, 0, input_size),
                                      combined_input_buffers[1].slice(graph, 2, 0, input_size)};
        std::array combined_h_buffers{combined_input_buffers[0].slice(graph, 2, input_size, hidden_size),
                                      combined_input_buffers[1].slice(graph, 2, input_size, hidden_size)};

        std::array<TraceTensor, 2> dy_buffers_fp32{
            graph.createTensor({static_cast<uint64_t>(chunk_capacity), batch, hidden_size}, DataType::FLOAT32, device,
                               compute_stream_descriptor, false),
            graph.createTensor({static_cast<uint64_t>(chunk_capacity), batch, hidden_size}, DataType::FLOAT32, device,
                               compute_stream_descriptor, false)};
        std::array<TraceTensor, 2> c_checkpoint_buffers_fp32{
            graph.createTensor({batch, hidden_size}, DataType::FLOAT32, device, compute_stream_descriptor, false),
            graph.createTensor({batch, hidden_size}, DataType::FLOAT32, device, compute_stream_descriptor, false)};

        const bool dy_needs_cast = (dy_dtype != DataType::FLOAT32);
        std::optional<std::array<TraceTensor, 2>> dy_buffers_lowp;
        if (dy_needs_cast)
        {
            dy_buffers_lowp = std::array<TraceTensor, 2>{
                graph.createTensor({static_cast<uint64_t>(chunk_capacity), batch, hidden_size}, dy_dtype,
                                   device, compute_stream_descriptor, false),
                graph.createTensor({static_cast<uint64_t>(chunk_capacity), batch, hidden_size}, dy_dtype,
                                   device, compute_stream_descriptor, false)};
        }

        const bool c_cache_needs_cast = (c_cache_host.dtype() != DataType::FLOAT32);
        std::optional<std::array<TraceTensor, 2>> c_checkpoint_buffers_fp16_opt;
        if (c_cache_needs_cast)
        {
            c_checkpoint_buffers_fp16_opt = std::array{
                graph.createTensor({batch, hidden_size}, DataType::FLOAT16, device, compute_stream_descriptor, false),
                graph.createTensor({batch, hidden_size}, DataType::FLOAT16, device, compute_stream_descriptor, false)};
        }

        std::array<OpGraph::GpuEventHandle, 2> h2d_events{graph.createGpuEvent(device), graph.createGpuEvent(device)};
        std::array<bool, 2> h2d_events_recorded{false, false};
        std::array<OpGraph::GpuEventHandle, 2> compute_done_events{graph.createGpuEvent(device),
                                                                   graph.createGpuEvent(device)};
        std::array<bool, 2> compute_done_recorded{false, false};

        std::array dy_ready_events{graph.createGpuEvent(device), graph.createGpuEvent(device)};
        std::array<bool, 2> dy_ready_recorded{false, false};
        std::array dy_reuse_events{graph.createGpuEvent(device), graph.createGpuEvent(device)};
        std::array<bool, 2> dy_reuse_recorded{false, false};

        struct ChunkMeta
        {
            uint64_t chunk_start{};
            uint64_t chunk_len{};
            uint64_t recompute_start{};
            uint64_t steps_to_recompute{};
            uint64_t prefix_steps{};
            bool has_checkpoint{};
            uint64_t c_cache_index{};
        };

        std::array<ChunkMeta, 2> chunk_meta{};
        std::array<bool, 2> chunk_valid{false, false};

        const size_t total_chunks = (seq_len + chunk_capacity - 1) / chunk_capacity;

        auto prefetch_chunk = [&](const uint64_t chunk_id, const int buffer_idx)
        {
            if (chunk_id >= total_chunks)
            {
                chunk_valid[buffer_idx] = false;
                return;
            }

            const uint64_t chunk_start = chunk_id * chunk_capacity;
            const uint64_t chunk_len = std::min<uint64_t>(chunk_capacity, seq_len - chunk_start);
            const uint64_t chunk_end = chunk_start + chunk_len;
            const bool has_checkpoint = chunk_start != 0;
            const uint64_t recompute_start =
                has_checkpoint ? ((chunk_start - 1) / recompute_interval_size) * recompute_interval_size + 1 : 0;
            const uint64_t steps_to_recompute = chunk_end - recompute_start;
            const uint64_t prefix_steps = chunk_start - recompute_start;

            if (steps_to_recompute > recompute_capacity)
            {
                throw std::runtime_error("StreamingLstm backward recompute span exceeds staging buffer");
            }

            ChunkMeta meta{};
            meta.chunk_start = chunk_start;
            meta.chunk_len = chunk_len;
            meta.recompute_start = recompute_start;
            meta.steps_to_recompute = steps_to_recompute;
            meta.prefix_steps = prefix_steps;
            meta.has_checkpoint = has_checkpoint;
            meta.c_cache_index = has_checkpoint ? (recompute_start - 1) / recompute_interval_size : 0;
            chunk_meta[buffer_idx] = meta;
            chunk_valid[buffer_idx] = true;

            TraceTensor x_src = x.slice(graph, 0, recompute_start, steps_to_recompute);
            TraceTensor x_dst = combined_x_buffers[buffer_idx].slice(graph, 0, 0, steps_to_recompute);
            DeviceCopy(graph, x_src, x_dst, GpuStreamDescriptors::H2D);

            if (dy_supplier_stream_id != 0 && dy_reuse_recorded[buffer_idx])
            {
                graph.awaitGpuEvent(dy_reuse_events[buffer_idx], dy_supplier_stream_desc);
                dy_reuse_recorded[buffer_idx] = false;
            }

            TraceTensor dy_dst = (dy_needs_cast ? (*dy_buffers_lowp)[buffer_idx] : dy_buffers_fp32[buffer_idx])
                                     .slice(graph, 0, 0, chunk_len);
            const std::optional<GpuStreamDescriptor> supplier_stream =
                dy_supplier(graph, dy_dst, chunk_start, chunk_len);
            if (supplier_stream.has_value())
            {
                if (!supplier_stream->isValid())
                {
                    throw std::runtime_error("StreamingLstm backward supplier returned invalid stream descriptor");
                }
                if (supplier_stream->getStreamId() != dy_supplier_stream_id)
                {
                    throw std::runtime_error("StreamingLstm backward supplier returned unexpected stream");
                }
            }
            graph.recordGpuEvent(dy_ready_events[buffer_idx], dy_supplier_stream_desc);
            dy_ready_recorded[buffer_idx] = true;

            if (has_checkpoint)
            {
                TraceTensor h_src = h_cache_host.at(graph, 0, recompute_start - 1);
                TraceTensor h_dst = combined_h_buffers[buffer_idx].at(graph, 0, 0);
                DeviceCopy(graph, h_src, h_dst, GpuStreamDescriptors::H2D);

                TraceTensor c_src = c_cache_host.at(graph, 0, meta.c_cache_index);
                if (c_cache_needs_cast)
                {
                    const auto &c_checkpoint_buffers_fp16 = *c_checkpoint_buffers_fp16_opt;
                    DeviceCopy(graph, c_src, c_checkpoint_buffers_fp16[buffer_idx], GpuStreamDescriptors::H2D);
                }
                else
                {
                    DeviceCopy(graph, c_src, c_checkpoint_buffers_fp32[buffer_idx], GpuStreamDescriptors::H2D);
                }
            }

            graph.recordGpuEvent(h2d_events[buffer_idx], GpuStreamDescriptors::H2D);
            h2d_events_recorded[buffer_idx] = true;
        };

        size_t processed_chunks = 0;
        int compute_idx = 0;
        int prefetch_idx = 1;

        if (total_chunks == 0)
        {
            throw std::runtime_error("StreamingLstm backward requires non-empty sequence");
        }

        // Ensure forward D2H transfers complete before we start H2D prefetching.
        AwaitAsyncTransfers(graph, TransferType::D2H, device, compute_stream_descriptor);
        AwaitAsyncTransfers(graph, TransferType::D2H, device, GpuStreamDescriptors::H2D);

        if (dy_supplier_stream_id != 0)
        {
            // Ensure the supplier stream sees all earlier compute and D2H work before producing dY.
            AwaitAsyncTransfers(graph, TransferType::D2H, device, dy_supplier_stream_desc);
            auto supplier_sync = graph.createGpuEvent(device);
            graph.recordGpuEvent(supplier_sync, compute_stream_descriptor);
            graph.awaitGpuEvent(supplier_sync, dy_supplier_stream_desc);
            graph.deleteGpuEvent(supplier_sync);
        }

        int64_t next_chunk_id = static_cast<int64_t>(total_chunks) - 1;
        prefetch_chunk(static_cast<uint64_t>(next_chunk_id), compute_idx);
        next_chunk_id--;

        while (processed_chunks < total_chunks)
        {
            if (!chunk_valid[compute_idx])
            {
                break;
            }

            const ChunkMeta &meta = chunk_meta[compute_idx];

            // Ensure the prefetched H2D transfers for this chunk are complete before use.
            if (h2d_events_recorded[compute_idx])
            {
                graph.awaitGpuEvent(h2d_events[compute_idx], compute_stream_descriptor);
                h2d_events_recorded[compute_idx] = false;
            }
            if (dy_ready_recorded[compute_idx])
            {
                graph.awaitGpuEvent(dy_ready_events[compute_idx], compute_stream_descriptor);
                dy_ready_recorded[compute_idx] = false;
            }

            const bool has_next_chunk = next_chunk_id >= 0;
            const bool delay_prefetch = has_next_chunk && processed_chunks == 0;
            auto queue_next_chunk = [&]() -> bool
            {
                if (next_chunk_id < 0)
                {
                    chunk_valid[prefetch_idx] = false;
                    return false;
                }
                if (compute_done_recorded[prefetch_idx])
                {
                    graph.awaitGpuEvent(compute_done_events[prefetch_idx], GpuStreamDescriptors::H2D);
                    compute_done_recorded[prefetch_idx] = false;
                }
                prefetch_chunk(static_cast<uint64_t>(next_chunk_id), prefetch_idx);
                --next_chunk_id;
                return true;
            };

            if (!delay_prefetch)
            {
                queue_next_chunk();
            }

            // Prepare chunk views.
            TraceTensor combined_input_chunk_slice =
                combined_input_buffers[compute_idx].slice(graph, 0, 0, meta.steps_to_recompute);
            TraceTensor combined_h_chunk =
                combined_h_buffers[compute_idx].slice(graph, 0, 0, meta.steps_to_recompute + 1);
            TraceTensor dy_chunk_fp32 = dy_buffers_fp32[compute_idx].slice(graph, 0, 0, meta.chunk_len);
            if (dy_needs_cast)
            {
                TraceTensor dy_lowp = (*dy_buffers_lowp)[compute_idx].slice(graph, 0, 0, meta.chunk_len);
                graph.recordOperation(OperationEntry{.type = OpType::CAST,
                                                     .inputs = {dy_lowp},
                                                     .outputs = {dy_chunk_fp32},
                                                     .attributes = {},
                                                     .gpu_stream_desc = compute_stream_descriptor});
            }

            TraceTensor c_prev_checkpoint = c0_broadcast;
            if (meta.has_checkpoint)
            {
                if (c_cache_needs_cast)
                {
                    const auto &c_checkpoint_buffers_fp16 = *c_checkpoint_buffers_fp16_opt;
                    TraceTensor c_half = c_checkpoint_buffers_fp16[compute_idx];
                    TraceTensor c_fp32 = c_checkpoint_buffers_fp32[compute_idx];
                    graph.recordOperation(OperationEntry{.type = OpType::CAST,
                                                         .inputs = {c_half},
                                                         .outputs = {c_fp32},
                                                         .attributes = {},
                                                         .gpu_stream_desc = compute_stream_descriptor});
                    c_prev_checkpoint = c_fp32;
                }
                else
                {
                    c_prev_checkpoint = c_checkpoint_buffers_fp32[compute_idx];
                }
            }

            TraceTensor h_seed = combined_h_chunk.at(graph, 0, 0);
            if (!meta.has_checkpoint)
            {
                DeviceCopy(graph, h0_broadcast, h_seed, compute_stream_descriptor);
            }

            TraceTensor gates_recompute = gates_buf.slice(graph, 0, 0, meta.steps_to_recompute);
            TraceTensor gate_out_recompute = gate_out_buf.slice(graph, 0, 0, meta.steps_to_recompute);
            TraceTensor c_recompute = c_buf.slice(graph, 0, 0, meta.steps_to_recompute);

            TraceTensor dGates_chunk_slice = dGates_chunk.slice(graph, 0, 0, meta.chunk_len);

            // Recompute forward states for the span.
            TraceTensor c_prev = c_prev_checkpoint;
            bool prefetch_queued = !delay_prefetch;
            for (size_t idx = 0; idx < meta.steps_to_recompute; ++idx)
            {
                TraceTensor combined_input_step = combined_input_chunk_slice.at(graph, 0, idx);
                TraceTensor gates_step = gates_recompute.at(graph, 0, idx);
                TraceTensor gate_out_step = gate_out_recompute.at(graph, 0, idx);
                TraceTensor c_out_step = c_recompute.at(graph, 0, idx);
                TraceTensor h_out_step = combined_h_chunk.at(graph, 0, idx + 1);

                graph.recordOperation(OperationEntry{.type = OpType::MATMUL,
                                                     .inputs = {combined_input_step, w_concat},
                                                     .outputs = {gates_step},
                                                     .is_useful = false,
                                                     .attributes = {},
                                                     .gpu_stream_desc = compute_stream_descriptor});

                graph.recordOperation(OperationEntry{.type = OpType::LSTM_CELL_RECOMPUTE,
                                                     .inputs = {gates_step, b_sum, c_prev},
                                                     .outputs = {gate_out_step, h_out_step, c_out_step},
                                                     .is_useful = false,
                                                     .attributes = {},
                                                     .gpu_stream_desc = compute_stream_descriptor});
                c_prev = c_out_step;

                if (!prefetch_queued)
                {
                    prefetch_queued = queue_next_chunk();
                }
            }

            // Backward pass for this chunk.
            for (size_t local = meta.chunk_len; local-- > 0;)
            {
                const size_t recompute_idx = meta.prefix_steps + local;
                TraceTensor dY_t = dy_chunk_fp32.at(graph, 0, local);

                TraceTensor c_prev_step =
                    (recompute_idx == 0) ? c_prev_checkpoint : c_recompute.at(graph, 0, recompute_idx - 1);
                TraceTensor c_out_step = c_recompute.at(graph, 0, recompute_idx);
                TraceTensor gate_out_step = gate_out_recompute.at(graph, 0, recompute_idx);
                TraceTensor dGates_half = dGates_chunk_slice.at(graph, 0, local);

                graph.recordOperation(
                    OperationEntry{.type = OpType::LSTM_CELL_BWD,
                                   .inputs = {dY_t, dh_next, dc_next, gate_out_step, c_prev_step, c_out_step},
                                   .outputs = {dGates_fp32, dGates_half, dc_prev},
                                   .attributes = {},
                                   .gpu_stream_desc = compute_stream_descriptor});

                graph.recordOperation(OperationEntry{.type = OpType::MATMUL,
                                                     .inputs = {dGates_half, w_hh_T},
                                                     .outputs = {dh_buffer},
                                                     .attributes = {},
                                                     .gpu_stream_desc = compute_stream_descriptor});
                dh_next = dh_buffer;
                dc_next = dc_prev;
            }

            const size_t chunk_batch = meta.chunk_len * batch_size;
            TraceTensor dGates_chunk_flat = dGates_chunk_slice.view(graph, {chunk_batch, gate_dim});

            // dx and D2H transfer kick off as soon as dx is ready.
            TraceTensor dx_chunk = dx_dev.slice(graph, 0, meta.chunk_start, meta.chunk_len);
            TraceTensor dx_flat = dx_chunk.view(graph, {chunk_batch, input_size});
            graph.recordOperation(OperationEntry{.type = OpType::MATMUL,
                                                 .inputs = {dGates_chunk_flat, w_ih_T},
                                                 .outputs = {dx_flat},
                                                 .attributes = {},
                                                 .gpu_stream_desc = compute_stream_descriptor});

            TraceTensor dx_host_chunk = dx_host.slice(graph, 0, meta.chunk_start, meta.chunk_len);
            AwaitComputeForTransfer(graph, TransferType::D2H, device, compute_stream_descriptor);
            DeviceCopy(graph, dx_chunk, dx_host_chunk, GpuStreamDescriptors::D2H);

            // Weight/bias gradients.
            TraceTensor combined_input_chunk_grad =
                combined_input_buffers[compute_idx].slice(graph, 0, meta.prefix_steps, meta.chunk_len);
            TraceTensor combined_input_flat =
                combined_input_chunk_grad.view(graph, {chunk_batch, input_size + hidden_size});
            TraceTensor combined_input_T = combined_input_flat.transpose(graph, {1, 0});
            graph.recordOperation(OperationEntry{.type = OpType::MATMUL,
                                                 .inputs = {combined_input_T, dGates_chunk_flat},
                                                 .outputs = {grad_w_concat},
                                                 .attributes = accumulate_attr,
                                                 .gpu_stream_desc = compute_stream_descriptor});

            TraceTensor db_vec = ReduceSum(graph, dGates_chunk_flat, /*dim=*/0, /*keepdim=*/false,
                                           DataType::FLOAT32, compute_stream_descriptor);
            InplaceAdd(graph, grad_b_ih, db_vec, compute_stream_descriptor);
            InplaceAdd(graph, grad_b_hh, db_vec, compute_stream_descriptor);
            graph.deleteTensor(db_vec);

            if (dy_supplier_stream_id != 0)
            {
                graph.recordGpuEvent(dy_reuse_events[compute_idx], compute_stream_descriptor);
                dy_reuse_recorded[compute_idx] = true;
            }

            graph.recordGpuEvent(compute_done_events[compute_idx], compute_stream_descriptor);
            compute_done_recorded[compute_idx] = true;

            processed_chunks++;
            if (processed_chunks < total_chunks)
            {
                std::swap(compute_idx, prefetch_idx);
            }
        }

        AwaitAsyncTransfers(graph, TransferType::D2H, device, compute_stream_descriptor);

        graph.deleteTensor(w_ih_cast);
        graph.deleteTensor(w_hh_cast);
        graph.deleteTensor(w_concat);
        graph.deleteTensor(b_sum);
        if (h0_cast.id() != h0_view.id())
        {
            graph.deleteTensor(h0_cast);
        }
        graph.deleteTensor(gates_buf);
        graph.deleteTensor(gate_out_buf);
        graph.deleteTensor(c_buf);
        graph.deleteTensor(dGates_chunk);
        graph.deleteTensor(dGates_fp32);

        for (auto &buf : combined_input_buffers)
        {
            graph.deleteTensor(buf);
        }
        for (auto &buf : dy_buffers_fp32)
        {
            graph.deleteTensor(buf);
        }
        for (auto &buf : c_checkpoint_buffers_fp32)
        {
            graph.deleteTensor(buf);
        }
        if (dy_buffers_lowp.has_value())
        {
            for (auto &buf : *dy_buffers_lowp)
            {
                graph.deleteTensor(buf);
            }
        }
        if (c_checkpoint_buffers_fp16_opt.has_value())
        {
            for (auto &buf : *c_checkpoint_buffers_fp16_opt)
            {
                graph.deleteTensor(buf);
            }
        }
        if (casted_c0)
        {
            graph.deleteTensor(c0_fp32);
        }
        graph.deleteTensor(dx_dev);

        for (auto &event_handle : dy_ready_events)
        {
            graph.deleteGpuEvent(event_handle);
        }

        TraceTensor grad_h0 =
            ReduceSum(graph, dh_next, /*dim=*/0, /*keepdim=*/false, DataType::FLOAT32, compute_stream_descriptor);
        TraceTensor grad_c0 =
            ReduceSum(graph, dc_next, /*dim=*/0, /*keepdim=*/false, DataType::FLOAT32, compute_stream_descriptor);

        return LstmBackwardStreamingResult{.grad_x = dx_host,
                                           .grad_h0 = grad_h0,
                                           .grad_c0 = grad_c0,
                                           .grad_w_ih = grad_w_ih,
                                           .grad_w_hh = grad_w_hh,
                                           .grad_b_ih = grad_b_ih,
                                           .grad_b_hh = grad_b_hh};
    }
} // namespace pi::tensorlib
