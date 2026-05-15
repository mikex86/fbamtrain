#include "functional.h"
#include "tensor_stream_iterator.h"
#include "tensor_stream_populator.h"

#include <array>
#include <optional>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

static pi::tensorlib::TensorStreamIterator CreateStreamingIterator(const pi::tensorlib::TraceTensor &source_tensor,
                                                                   const size_t chunk_size,
                                                                   const pi::tensorlib::Device target_device)
{
    return {source_tensor, chunk_size, 0, target_device, pi::tensorlib::TransferType::H2D, std::nullopt};
}

static pi::tensorlib::TensorStreamPopulator CreatingStreamingPopulator(const pi::tensorlib::TraceTensor &dst_tensor)
{
    return {dst_tensor, 0, pi::tensorlib::TransferType::D2H};
}

pi::tensorlib::LstmForwardStreamingResult
pi::tensorlib::StreamingLstmFwd(OpGraph &graph, const TraceTensor &x_tensor_host, const TraceTensor &h_0_tensor,
                                const TraceTensor &c_0_tensor, const TraceTensor &w_ih_tensor,
                                const TraceTensor &w_hh_tensor, const TraceTensor &b_ih_tensor,
                                const TraceTensor &b_hh_tensor, DataType c_cache_dtype, const int recompute_interval,
                                const size_t streaming_chunk_size, const GpuStreamDescriptor &compute_stream_descriptor,
                                StreamingChunkHook chunk_hook)
{
    // assert w_hh_tensor has shape (H, 4 * H)
    if (w_hh_tensor.shape().ndims() != 2)
    {
        throw std::runtime_error("w_hh_tensor must have 2 dimensions (H, 4 * H)");
    }
    if (w_hh_tensor.shape()[1] % 4 != 0)
    {
        throw std::runtime_error("w_hh_tensor second dimension must be divisible by 4");
    }

    // assert w_ih_tensor has shape (I, 4 * H)
    if (w_ih_tensor.shape().ndims() != 2)
    {
        throw std::runtime_error("w_ih_tensor must have 2 dimensions (I, 4 * H)");
    }
    const auto &w_hh_shape = w_hh_tensor.shape();
    const auto hidden_size = w_hh_shape[0];
    const auto gate_size = w_hh_shape[1];
    if (gate_size != 4 * hidden_size)
    {
        throw std::runtime_error("w_hh_tensor second dimension must be exactly 4 * hidden_size");
    }

    // assert x_tensor_host is on CPU
    if (x_tensor_host.device().device_type != DeviceType::CPU)
    {
        throw std::runtime_error("x_tensor_host must be on CPU");
    }

    // assert x_tensor_host has shape (T, B, I)
    if (x_tensor_host.shape().ndims() != 3)
    {
        throw std::runtime_error("x_tensor_host must have 3 dimensions (T, B, I)");
    }

    const auto io_dtype = x_tensor_host.dtype();
    // assert x_tensor_host is dtype FLOAT16 or BFLOAT16
    if (io_dtype != DataType::FLOAT16 && io_dtype != DataType::BFLOAT16)
    {
        throw std::runtime_error("x_tensor_host must be of type FLOAT16 or BFLOAT16");
    }

    const auto &x_shape = x_tensor_host.shape();
    const uint64_t seq_len = x_shape[0];
    const uint64_t batch_size = x_shape[1];
    const uint64_t input_size = x_shape[2];

    if (h_0_tensor.shape().ndims() != 1 || h_0_tensor.shape()[0] != hidden_size)
    {
        throw std::runtime_error("h_0_tensor must be a 1D tensor of size H");
    }
    // assert h_0_tensor is dtype FLOAT32
    if (h_0_tensor.dtype() != DataType::FLOAT32)
    {
        throw std::runtime_error("h_0_tensor must be of type FLOAT32");
    }

    if (c_0_tensor.shape().ndims() != 1 || c_0_tensor.shape()[0] != hidden_size)
    {
        throw std::runtime_error("c_0_tensor_host must be a 1D tensor of size H");
    }
    // assert c_0_tensor_host is dtype FLOAT32
    if (c_0_tensor.dtype() != DataType::FLOAT32)
    {
        throw std::runtime_error("c_0_tensor_host must be of type FLOAT32");
    }

    // assert w_ih_tensor has shape (I, 4H)
    if (w_ih_tensor.shape().ndims() != 2)
    {
        throw std::runtime_error("w_ih_tensor must have 2 dimensions (I, 4 * H)");
    }
    if (w_ih_tensor.shape()[0] != input_size || w_ih_tensor.shape()[1] != gate_size)
    {
        throw std::runtime_error("w_ih_tensor shape mismatch with input size and hidden size");
    }
    // weights are stored in FP32
    if (w_hh_tensor.dtype() != DataType::FLOAT32 || w_ih_tensor.dtype() != DataType::FLOAT32)
    {
        throw std::runtime_error("w_hh_tensor and w_ih_tensor must be of type FLOAT32");
    }

    // assert b_ih_tensor has shape (4 * H)
    if (b_ih_tensor.shape().ndims() != 1)
    {
        throw std::runtime_error("b_ih_tensor must have 1 dimension (4 * H)");
    }
    if (b_ih_tensor.shape()[0] != gate_size)
    {
        throw std::runtime_error("b_ih_tensor shape mismatch with hidden size");
    }
    // assert b_hh_tensor is dtype FLOAT32
    if (b_hh_tensor.dtype() != DataType::FLOAT32)
    {
        throw std::runtime_error("b_hh_tensor must be of type FLOAT32");
    }

    // assert b_hh_tensor has shape (4 * H)
    if (b_hh_tensor.shape().ndims() != 1)
    {
        throw std::runtime_error("b_hh_tensor must have 1 dimension (4 * H)");
    }
    // assert b_hh_tensor is dtype FLOAT32
    if (b_hh_tensor.dtype() != DataType::FLOAT32)
    {
        throw std::runtime_error("b_hh_tensor must be of type FLOAT32");
    }

    // assert c_cache_dtype is FLOAT16 or FLOAT32
    if (c_cache_dtype != DataType::FLOAT16 && c_cache_dtype != DataType::FLOAT32)
    {
        throw std::runtime_error("c_cache_dtype must be FLOAT16 or FLOAT32");
    }

    // ensure all weights are on GPU before proceeding
    if (w_ih_tensor.device().device_type != DeviceType::GPU || w_hh_tensor.device().device_type != DeviceType::GPU ||
        b_ih_tensor.device().device_type != DeviceType::GPU || b_hh_tensor.device().device_type != DeviceType::GPU)
    {
        throw std::runtime_error("All weight tensors must be on GPU device");
    }

    // ensure all weights are on the same GPU device
    if (w_ih_tensor.device() != w_hh_tensor.device() || w_ih_tensor.device() != b_ih_tensor.device() ||
        w_ih_tensor.device() != b_hh_tensor.device())
    {
        throw std::runtime_error("All weight tensors must be on the same GPU device");
    }

    // ensure h0 and c0 are on GPU before proceeding
    if (h_0_tensor.device().device_type != DeviceType::GPU || c_0_tensor.device().device_type != DeviceType::GPU)
    {
        throw std::runtime_error("h_0_tensor and c_0_tensor must be on GPU device");
    }

    // ensure h0 and c0 are on the same GPU device as weights
    if (h_0_tensor.device() != w_ih_tensor.device() || c_0_tensor.device() != w_ih_tensor.device())
    {
        throw std::runtime_error("h_0_tensor and c_0_tensor must be on the same GPU device as weight tensors");
    }

    OpGraphGpuTxRange range = graph.createGpuTxRange("pi::tensorlib::StreamingLstmFwd");

    const size_t output_seqlen = seq_len;
    const size_t cache_seqlen = CEIL_DIV(seq_len, recompute_interval);

    const auto intermediate_datatype = (c_cache_dtype == DataType::FLOAT16) ? DataType::FLOAT16 : DataType::FLOAT32;

    TraceTensor output = graph.createTensor({output_seqlen, batch_size, hidden_size}, io_dtype,
                                            Device{DeviceType::CPU, 0}, compute_stream_descriptor, true);

    TraceTensor c_cache = graph.createTensor({cache_seqlen, batch_size, hidden_size}, intermediate_datatype,
                                             Device{DeviceType::CPU, 0}, compute_stream_descriptor, true);

    Device device = w_hh_tensor.device();

    // streaming iterator/populator over time dimension
    TensorStreamIterator x_iterator = CreateStreamingIterator(x_tensor_host, streaming_chunk_size, device);
    TensorStreamPopulator y_populator = CreatingStreamingPopulator(output);
    TensorStreamPopulator c_populator = CreatingStreamingPopulator(c_cache);

    // cast weights/bias to fp16 for compute
    TraceTensor w_ih_cast =
        graph.createTensor({input_size, gate_size}, io_dtype, device, compute_stream_descriptor, false);
    TraceTensor w_hh_cast =
        graph.createTensor({hidden_size, gate_size}, io_dtype, device, compute_stream_descriptor, false);
    TraceTensor b_ih_cast =
        graph.createTensor({gate_size}, io_dtype, device, compute_stream_descriptor, false);
    TraceTensor b_hh_cast =
        graph.createTensor({gate_size}, io_dtype, device, compute_stream_descriptor, false);

    TraceTensor w_concat = graph.createTensor({input_size + hidden_size, gate_size}, io_dtype, device,
                                              compute_stream_descriptor, false);

    graph.recordOperation(OperationEntry{.type = OpType::CAST,
                                         .inputs = {w_ih_tensor},
                                         .outputs = {w_ih_cast},
                                         .attributes = {},
                                         .gpu_stream_desc = compute_stream_descriptor});
    graph.recordOperation(OperationEntry{.type = OpType::CAST,
                                         .inputs = {w_hh_tensor},
                                         .outputs = {w_hh_cast},
                                         .attributes = {},
                                         .gpu_stream_desc = compute_stream_descriptor});
    graph.recordOperation(OperationEntry{.type = OpType::CAST,
                                         .inputs = {b_ih_tensor},
                                         .outputs = {b_ih_cast},
                                         .attributes = {},
                                         .gpu_stream_desc = compute_stream_descriptor});
    graph.recordOperation(OperationEntry{.type = OpType::CAST,
                                         .inputs = {b_hh_tensor},
                                         .outputs = {b_hh_cast},
                                         .attributes = {},
                                         .gpu_stream_desc = compute_stream_descriptor});

    TraceTensor b_sum =
        graph.createTensor({gate_size}, intermediate_datatype, device, compute_stream_descriptor, false);
    graph.recordOperation(OperationEntry{.type = OpType::PLUS,
                                         .inputs = {b_ih_tensor, b_hh_tensor},
                                         .outputs = {b_sum},
                                         .attributes = {},
                                         .gpu_stream_desc = compute_stream_descriptor});

    // lay out concatenated weights [w_ih; w_hh] so each step needs only one GEMM
    TraceTensor w_concat_top = w_concat.slice(graph, 0, 0, input_size);
    TraceTensor w_concat_bottom = w_concat.slice(graph, 0, input_size, hidden_size);
    graph.recordOperation(OperationEntry{.type = OpType::DEVICE_COPY,
                                         .inputs = {w_ih_cast},
                                         .outputs = {w_concat_top},
                                         .attributes = {},
                                         .gpu_stream_desc = compute_stream_descriptor});
    graph.recordOperation(OperationEntry{.type = OpType::DEVICE_COPY,
                                         .inputs = {w_hh_cast},
                                         .outputs = {w_concat_bottom},
                                         .attributes = {},
                                         .gpu_stream_desc = compute_stream_descriptor});

    // Double-buffered combined input to overlap H2D of the next chunk with compute of the current chunk.
    std::array combined_input_chunks{
        graph.createTensor({(streaming_chunk_size + 1), batch_size, input_size + hidden_size}, io_dtype,
                           device, compute_stream_descriptor, false),
        graph.createTensor({(streaming_chunk_size + 1), batch_size, input_size + hidden_size}, io_dtype,
                           device, compute_stream_descriptor, false)};
    std::array combined_x_chunks{combined_input_chunks[0].slice(graph, 2, 0, input_size),
                                 combined_input_chunks[1].slice(graph, 2, 0, input_size)};
    std::array combined_h_chunks{combined_input_chunks[0].slice(graph, 2, input_size, hidden_size),
                                 combined_input_chunks[1].slice(graph, 2, input_size, hidden_size)};

    TraceTensor h_0_view = h_0_tensor.view(graph, {1, hidden_size});
    TraceTensor c_0_view = c_0_tensor.view(graph, {1, hidden_size});

    TraceTensor h_0_cast =
        graph.createTensor(h_0_view.shape().dims(), io_dtype, device, compute_stream_descriptor, false);

    graph.recordOperation(OperationEntry{.type = OpType::CAST,
                                         .inputs = {h_0_view},
                                         .outputs = {h_0_cast},
                                         .attributes = {},
                                         .gpu_stream_desc = compute_stream_descriptor});

    TraceTensor h0_broadcast =
        (h_0_cast.shape()[0] == batch_size) ? h_0_cast : h_0_cast.broadcast(graph, 0, batch_size);

    const auto chunk_capacity = static_cast<uint64_t>(streaming_chunk_size);

    std::array<TraceTensor, 2> c_buffers{
        graph.createTensor({chunk_capacity, batch_size, hidden_size}, intermediate_datatype, device,
                           compute_stream_descriptor, false),
        graph.createTensor({chunk_capacity, batch_size, hidden_size}, intermediate_datatype, device,
                           compute_stream_descriptor, false)};

    int c_front_idx = 0; // destination for current chunk
    int c_back_idx = 1;  // source (previous chunk)

    // current hidden state buffer (used for the final output only)
    TraceTensor h_state =
        graph.createTensor({batch_size, hidden_size}, io_dtype, device, compute_stream_descriptor, false);

    // output chunk buffers (double-buffered)
    std::array<TraceTensor, 2> y_buffers{
        graph.createTensor({chunk_capacity, batch_size, hidden_size}, io_dtype, device,
                           compute_stream_descriptor, false),
        graph.createTensor({chunk_capacity, batch_size, hidden_size}, io_dtype, device,
                           compute_stream_descriptor, false)};

    TraceTensor c0_cast = c_0_view;
    if (c_0_view.dtype() != intermediate_datatype)
    {
        c0_cast = graph.createTensor(c_0_view.shape().dims(), intermediate_datatype, device, compute_stream_descriptor,
                                     false);
        graph.recordOperation(OperationEntry{.type = OpType::CAST,
                                             .inputs = {c_0_view},
                                             .outputs = {c0_cast},
                                             .attributes = {},
                                             .gpu_stream_desc = compute_stream_descriptor});
    }

    TraceTensor c0_broadcast = (c0_cast.shape()[0] == batch_size) ? c0_cast : c0_cast.broadcast(graph, 0, batch_size);
    graph.recordOperation(OperationEntry{.type = OpType::DEVICE_COPY,
                                         .inputs = {c0_broadcast},
                                         .outputs = {c_buffers[c_back_idx].at(graph, 0, 0)},
                                         .attributes = {},
                                         .gpu_stream_desc = compute_stream_descriptor});

    TraceTensor gates_sum1 =
        graph.createTensor({batch_size, gate_size}, intermediate_datatype, device, compute_stream_descriptor, false);
    TraceTensor gates_sum2 =
        graph.createTensor({batch_size, gate_size}, intermediate_datatype, device, compute_stream_descriptor, false);

    uint64_t steps_processed = 0;
    uint64_t back_chunk_steps = 1;

    std::array<std::optional<GpuStreamDescriptor>, 2> y_buffer_streams{std::nullopt, std::nullopt};

    // events to track completion of D2H transfers referencing y_buffers
    std::array<OpGraph::GpuEventHandle, 2> d2h_events{graph.createGpuEvent(device), graph.createGpuEvent(device)};
    std::array<bool, 2> d2h_events_recorded{false, false};

    // events to track completion of H2D transfers into combined_input_chunks
    std::array<OpGraph::GpuEventHandle, 2> h2d_events{graph.createGpuEvent(device), graph.createGpuEvent(device)};
    std::array<bool, 2> h2d_events_recorded{false, false};

    // events to track completion of chunk hook work referencing y_buffers
    std::array<OpGraph::GpuEventHandle, 2> hook_work_complete{graph.createGpuEvent(device),
                                                              graph.createGpuEvent(device)};
    std::array<bool, 2> hook_work_complete_recorded{false, false};

    // Seed initial hidden into the first compute buffer and into h_state for the final output.
    TraceTensor h_seed0 = combined_h_chunks[0].at(graph, 0, 0);
    graph.recordOperation(OperationEntry{.type = OpType::DEVICE_COPY,
                                         .inputs = {h0_broadcast},
                                         .outputs = {h_seed0},
                                         .attributes = {},
                                         .gpu_stream_desc = compute_stream_descriptor});
    graph.recordOperation(OperationEntry{.type = OpType::DEVICE_COPY,
                                         .inputs = {h0_broadcast},
                                         .outputs = {h_state},
                                         .attributes = {},
                                         .gpu_stream_desc = compute_stream_descriptor});

    int compute_idx = 0;
    int prefetch_idx = 1;

    // Prefetch the first chunk into the compute buffer.
    uint64_t current_chunk_steps = std::min<uint64_t>(static_cast<uint64_t>(streaming_chunk_size), seq_len);
    {
        // TODO: REVIEW IF THIS AWAIT IS NEEDED
        AwaitComputeForTransfer(graph, TransferType::H2D, device, compute_stream_descriptor);

        if (TraceTensor combined_x_first = combined_x_chunks[compute_idx].slice(graph, 0, 0, current_chunk_steps);
            !x_iterator.nextInplace(graph, combined_x_first))
        {
            throw std::runtime_error("TensorStreamIterator exhausted before expected steps");
        }
    }
    graph.recordGpuEvent(h2d_events[compute_idx], GpuStreamDescriptors::H2D);
    h2d_events_recorded[compute_idx] = true;

    while (steps_processed < seq_len)
    {
        TraceTensor &combined_input_chunk = combined_input_chunks[compute_idx];
        TraceTensor &combined_h_chunk = combined_h_chunks[compute_idx];
        TraceTensor combined_h_seed = combined_h_chunk.at(graph, 0, 0);

        if (h2d_events_recorded[compute_idx])
        {
            graph.awaitGpuEvent(h2d_events[compute_idx], compute_stream_descriptor);
            h2d_events_recorded[compute_idx] = false;
        }

        // Ensure the prefetched H2D for this chunk completed before use.
        AwaitAsyncTransfers(graph, TransferType::H2D, device, compute_stream_descriptor);

        // await D2H of the previous use of this buffer before overwriting it
        if (d2h_events_recorded[compute_idx])
        {
            graph.awaitGpuEvent(d2h_events[compute_idx], compute_stream_descriptor);
            d2h_events_recorded[compute_idx] = false;
        }

        // await chunk hook work completion before overwriting y buffer
        if (hook_work_complete_recorded[compute_idx])
        {
            graph.awaitGpuEvent(hook_work_complete[compute_idx], compute_stream_descriptor);
            hook_work_complete_recorded[compute_idx] = false;
        }

        const uint64_t steps_in_chunk = current_chunk_steps;
        TraceTensor y_chunk = y_buffers[compute_idx].slice(graph, 0, 0, steps_in_chunk);

        TraceTensor combined_input_chunk_slice = combined_input_chunk.slice(graph, 0, 0, steps_in_chunk + 1);
        TraceTensor combined_h_chunk_slice = combined_h_chunk.slice(graph, 0, 0, steps_in_chunk + 1);
        TraceTensor c_state_front_chunk = c_buffers[c_front_idx].slice(graph, 0, 0, steps_in_chunk);

        TraceTensor next_seed = h_state;

        // Prefetch the next chunk (if any) into the alternate buffer while we compute this one.
        if (steps_processed + steps_in_chunk < seq_len)
        {
            uint64_t next_steps = std::min<uint64_t>(static_cast<uint64_t>(streaming_chunk_size),
                                                     seq_len - (steps_processed + steps_in_chunk));
            TraceTensor &combined_x_next = combined_x_chunks[prefetch_idx];
            next_seed = combined_h_chunks[prefetch_idx].at(graph, 0, 0);

            // reuse guard: the next buffer was last used for compute one iteration ago
            if (steps_processed > 0)
            {
                AwaitComputeForTransfer(graph, TransferType::H2D, device, compute_stream_descriptor);
            }

            if (TraceTensor combined_x_chunk_slice_next = combined_x_next.slice(graph, 0, 0, next_steps);
                !x_iterator.nextInplace(graph, combined_x_chunk_slice_next))
            {
                throw std::runtime_error("TensorStreamIterator exhausted before expected steps");
            }
            graph.recordGpuEvent(h2d_events[prefetch_idx], GpuStreamDescriptors::H2D);
            h2d_events_recorded[prefetch_idx] = true;
            // Capture next chunk steps and swap roles after compute.
            current_chunk_steps = next_steps;
        }

        for (uint64_t local_step = 0; local_step < steps_in_chunk; ++local_step, ++steps_processed)
        {
            TraceTensor combined_input_step = combined_input_chunk_slice.at(graph, 0, local_step);
            TraceTensor combined_h_step = combined_h_chunk_slice.at(graph, 0, local_step);
            const bool is_last_step_overall = (steps_processed + 1 == seq_len);
            const bool is_last_step_in_chunk = (local_step + 1 == steps_in_chunk);
            TraceTensor combined_h_next =
                is_last_step_overall ? h_state                            // persist final hidden directly
                                     : (is_last_step_in_chunk ? next_seed // seed next chunk
                                                              : combined_h_chunk_slice.at(graph, 0, local_step + 1));

            TraceTensor c_prev_slice = (local_step == 0) ? c_buffers[c_back_idx].at(graph, 0, back_chunk_steps - 1)
                                                         : c_state_front_chunk.at(graph, 0, local_step - 1);

            graph.recordOperation(OperationEntry{.type = OpType::MATMUL,
                                                 .inputs = {combined_input_step, w_concat},
                                                 .outputs = {gates_sum1},
                                                 .attributes = {},
                                                 .gpu_stream_desc = compute_stream_descriptor});
            graph.recordOperation(OperationEntry{.type = OpType::PLUS,
                                                 .inputs = {gates_sum1, b_sum},
                                                 .outputs = {gates_sum2},
                                                 .attributes = {},
                                                 .gpu_stream_desc = compute_stream_descriptor});

            TraceTensor y_step = y_chunk.at(graph, 0, local_step);
            TraceTensor c_out_slice = c_state_front_chunk.at(graph, 0, local_step);

            // gates_sum2: (float32/float16)
            // c_prev_slice: (float32/float16)
            // combined_h_next: float16
            // c_out_slice: (float32/float16)
            // y_step: float16
            LstmCellFwdInplace(graph, gates_sum2, c_prev_slice, combined_h_next, c_out_slice, y_step,
                               compute_stream_descriptor);
        }

        if (y_buffer_streams[prefetch_idx].has_value())
        {
            const GpuStreamDescriptor stream_desc = *y_buffer_streams[prefetch_idx];
            AwaitComputeStream(graph, device, stream_desc);
            y_buffer_streams[prefetch_idx].reset();
        }

        back_chunk_steps = steps_in_chunk;

        std::optional<GpuStreamDescriptor> chunk_hook_stream_opt{};
        if (chunk_hook)
        {
            const uint64_t chunk_start = steps_processed - steps_in_chunk;
            chunk_hook_stream_opt = chunk_hook(graph, y_chunk, chunk_start, steps_in_chunk);

            // if the chunk hook returned a stream, we know it will keep using the y_chunk on that stream
            // until the current work completes. Therefore, we will record an event which we will later await
            // which signals that the chunk's work has completed.
            if (chunk_hook_stream_opt.has_value())
            {
                graph.recordGpuEvent(hook_work_complete[compute_idx], *chunk_hook_stream_opt);
                hook_work_complete_recorded[compute_idx] = true;
            }
        }

        // Track the stream that used this buffer (if any) so we can wait before reuse.
        y_buffer_streams[compute_idx] = chunk_hook_stream_opt;

        // Rotate once per chunk: the filled chunk becomes the back buffer.
        // We still read its tail (next chunk seed or final state), but it stays read-only after the swap.
        std::swap(c_front_idx, c_back_idx);

        // we cannot commence streaming out outputs to host until the current chunk's compute is done
        AwaitComputeForTransfer(graph, TransferType::D2H, device, compute_stream_descriptor);

        TraceTensor c_chunk_slice = c_buffers[c_back_idx].slice(graph, 0, 0, steps_in_chunk);
        const bool is_last_chunk = (steps_processed == seq_len);
        c_populator.populateNext(graph, c_chunk_slice, GpuStreamDescriptors::D2H, recompute_interval);
        y_populator.populateNext(graph, y_chunk, GpuStreamDescriptors::D2H);

        // record event that fires after the this chunk's d2h transfer is done
        graph.recordGpuEvent(d2h_events[compute_idx], GpuStreamDescriptors::D2H);
        d2h_events_recorded[compute_idx] = true;

        if (steps_processed < seq_len)
        {
            std::swap(compute_idx, prefetch_idx);
        }
    }

    graph.deleteTensor(gates_sum1);
    graph.deleteTensor(gates_sum2);

    TraceTensor c_n = graph.createTensor({batch_size, hidden_size}, io_dtype, Device{DeviceType::CPU, 0},
                                         compute_stream_descriptor, true);
    TraceTensor h_n = graph.createTensor({batch_size, hidden_size}, io_dtype, Device{DeviceType::CPU, 0},
                                         compute_stream_descriptor, true);

    graph.recordOperation(OperationEntry{.type = OpType::DEVICE_COPY,
                                         .inputs = {h_state},
                                         .outputs = {h_n},
                                         .attributes = {},
                                         .gpu_stream_desc = compute_stream_descriptor});
    // After the final rotation the last chunk lives in the back buffer; its tail holds the final cell state.
    TraceTensor c_final = c_buffers[c_back_idx].at(graph, 0, back_chunk_steps - 1);
    TraceTensor c_final_cast =
        graph.createTensor(c_final.shape().dims(), io_dtype, device, compute_stream_descriptor, false);
    graph.recordOperation(OperationEntry{.type = OpType::CAST,
                                         .inputs = {c_final},
                                         .outputs = {c_final_cast},
                                         .attributes = {},
                                         .gpu_stream_desc = compute_stream_descriptor});
    graph.recordOperation(OperationEntry{.type = OpType::DEVICE_COPY,
                                         .inputs = {c_final_cast},
                                         .outputs = {c_n},
                                         .attributes = {},
                                         .gpu_stream_desc = compute_stream_descriptor});
    AwaitAsyncTransfers(graph, TransferType::D2H, device, compute_stream_descriptor);

    for (auto &stream_opt : y_buffer_streams)
    {
        if (stream_opt.has_value())
        {
            const GpuStreamDescriptor stream_desc = *stream_opt;
            AwaitComputeStream(graph, device, stream_desc);
            stream_opt.reset();
        }
    }

    graph.deleteTensor(y_buffers[0]);
    graph.deleteTensor(y_buffers[1]);

    graph.deleteTensor(w_ih_cast);
    graph.deleteTensor(w_hh_cast);
    graph.deleteTensor(b_ih_cast);
    graph.deleteTensor(b_hh_cast);
    graph.deleteTensor(w_concat);
    graph.deleteTensor(b_sum);
    graph.deleteGpuEvent(d2h_events[0]);
    graph.deleteGpuEvent(d2h_events[1]);
    for (auto &buf : combined_input_chunks)
    {
        graph.deleteTensor(buf);
    }
    graph.deleteTensor(c_buffers[0]);
    graph.deleteTensor(c_buffers[1]);
    graph.deleteTensor(c_final_cast);
    graph.deleteTensor(h_0_cast);

    return LstmForwardStreamingResult{
        .output = output,
        .h_n = h_n,
        .c_n = c_n,
        .h_cache = output,
        .c_cache = c_cache,
    };
}
