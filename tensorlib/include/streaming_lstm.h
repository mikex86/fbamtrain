#pragma once

#include "functional.h"
#include "module.h"
#include "op_graph.h"

#include <functional>
#include <optional>

namespace pi::tensorlib
{
    struct StreamingLstmBackwardInput
    {
        /// Supplier to fill dY for [chunk_start, chunk_steps) into a device buffer owned by the LSTM.
        std::optional<StreamingDySupplier> upstream_supplier{};
        /// Stream descriptor the supplier uses for its compute work.
        GpuStreamDescriptor upstream_supplier_stream_desc;
        /// Dtype written into the supplier-provided dY buffers.
        DataType upstream_dy_dtype{DataType::FLOAT16};

        /// Optional full dY tensor if supplied materialised on host.
        std::optional<TraceTensor> dY_host{};

        /// Optional gradients for final states.
        std::optional<TraceTensor> d_hn{};
        std::optional<TraceTensor> d_cn{};

        /// Optional accumulate target for dx_host; if provided, backward will write into it.
        std::optional<TraceTensor> dx_host{};
    };

    class StreamingLstm final : public Module<LstmForwardStreamingResult, StreamingLstmBackwardInput>
    {
        size_t input_size_;
        size_t hidden_size_;

        int recompute_interval_{1};
        size_t streaming_chunk_size_{1};

        std::optional<StreamingChunkHook> chunk_hook_;

        GpuStreamDescriptor compute_stream_descriptor_;
        DataType io_dtype_{DataType::FLOAT16};

      public:
        TraceTensor weights_ih;
        TraceTensor weights_hh;
        TraceTensor bias_ih;
        TraceTensor bias_hh;

        TraceTensor h0;
        TraceTensor c0;

        StreamingLstm(const std::string &name, size_t input_size, size_t hidden_size, const int recompute_interval,
                      const size_t streaming_chunk_size, const Device device, OpGraph &graph, uint32_t &init_rng_seed,
                      const DataType io_dtype, const GpuStreamDescriptor &compute_stream_descriptor)
            : Module(name), input_size_(input_size), hidden_size_(hidden_size), recompute_interval_(recompute_interval),
              streaming_chunk_size_(streaming_chunk_size), compute_stream_descriptor_(compute_stream_descriptor),
              io_dtype_(io_dtype),
              weights_ih(graph.createTensor({input_size, 4 * hidden_size}, DataType::FLOAT32, device, compute_stream_descriptor, false)),
              weights_hh(graph.createTensor({hidden_size, 4 * hidden_size}, DataType::FLOAT32, device, compute_stream_descriptor, false)),
              bias_ih(graph.createTensor({4 * hidden_size}, DataType::FLOAT32, device, compute_stream_descriptor, false)),
              bias_hh(graph.createTensor({4 * hidden_size}, DataType::FLOAT32, device, compute_stream_descriptor, false)),

              h0(graph.createTensor({hidden_size}, DataType::FLOAT32, device, compute_stream_descriptor, false)),
              c0(graph.createTensor({hidden_size}, DataType::FLOAT32, device, compute_stream_descriptor, false))
        {
            KaimingUniformInit(graph, weights_ih, hidden_size, init_rng_seed++, compute_stream_descriptor);
            KaimingUniformInit(graph, weights_hh, hidden_size, init_rng_seed++, compute_stream_descriptor);
            KaimingUniformInit(graph, bias_ih, hidden_size, init_rng_seed++, compute_stream_descriptor);
            KaimingUniformInit(graph, bias_hh, hidden_size, init_rng_seed++, compute_stream_descriptor);

            FillZeros(graph, h0, compute_stream_descriptor);
            FillZeros(graph, c0, compute_stream_descriptor);
        }

        /**
         * Build the forward pass of the streaming LSTM.
         * @param graph the graph to build the operation in
         * @param inputs the input tensors: {x, h_0, c_0} ; The reason why we pass h_0 and c_0 as inputs
         * is to allow for pipeline-parallelism where different chunks of the input sequence are handled by different
         * workers. In this case, worker zero should default to the initial learned h0 and c0 states, while other
         * workers should receive the h and c states from the previous worker.
         * @return the result of the forward pass
         */
        [[nodiscard]] LstmForwardStreamingResult buildForward(OpGraph &graph, std::initializer_list<TraceTensor> inputs,
                                                              bool /*save_input_for_backward*/ = false) override
        {
            if (inputs.size() != 3)
            {
                throw std::runtime_error("StreamingLstm: expected 3 inputs {x, h_0, c_0}");
            }

            // input sequence
            TraceTensor x = inputs.begin()[0];

            // x: (T, B, I)
            if (x.shape().ndims() != 3)
            {
                throw std::runtime_error("StreamingLstm: input sequence must be a 3D tensor (T, B, I)");
            }
            if (x.shape()[2] != input_size_)
            {
                throw std::runtime_error("StreamingLstm: input size does not match weights_ih");
            }
            if (io_dtype_ != DataType::FLOAT16 && io_dtype_ != DataType::BFLOAT16)
            {
                throw std::runtime_error("StreamingLstm: io_dtype must be FLOAT16 or BFLOAT16");
            }
            // assert x is expected io dtype
            if (x.dtype() != io_dtype_)
            {
                throw std::runtime_error("StreamingLstm: input sequence dtype mismatch with configured io_dtype");
            }

            // make sure x is on CPU (-> host to device streaming)
            if (x.device().device_type != DeviceType::CPU)
            {
                throw std::runtime_error("StreamingLstm: input sequence must be on CPU");
            }

            // initial hidden state and cell state
            const auto &h_0 = inputs.begin()[1];
            const auto &c_0 = inputs.begin()[2];

            if (h_0.shape().ndims() != 1 || h_0.shape()[0] != hidden_size_)
            {
                throw std::runtime_error("StreamingLstm: initial hidden state must be a 1D tensor of size H");
            }
            // check h0 is fp32
            if (h_0.dtype() != DataType::FLOAT32)
            {
                throw std::runtime_error("StreamingLstm: initial hidden state must be of type FLOAT32");
            }
            // h0 must be on GPU
            if (h_0.device().device_type != DeviceType::GPU)
            {
                throw std::runtime_error("StreamingLstm: initial hidden state must be on GPU");
            }

            if (c_0.shape().ndims() != 1 || c_0.shape()[0] != hidden_size_)
            {
                throw std::runtime_error("StreamingLstm: initial cell state must be a 1D tensor of size H");
            }

            if (c_0.dtype() != DataType::FLOAT32)
            {
                throw std::runtime_error("StreamingLstm: initial cell state must be of type FLOAT32");
            }
            // c0 must be on GPU
            if (c_0.device().device_type != DeviceType::GPU)
            {
                throw std::runtime_error("StreamingLstm: initial cell state must be on GPU");
            }

            bw_context_.clear();

            // Keep gate/cache tensors in FP32 to avoid losing precision during recompute/save.
            LstmForwardStreamingResult fwd = StreamingLstmFwd(
                graph, x, h_0, c_0, weights_ih, weights_hh, bias_ih, bias_hh, DataType::FLOAT32, recompute_interval_,
                streaming_chunk_size_, compute_stream_descriptor_, chunk_hook_.value_or(StreamingChunkHook{}));

            // save h_cache and c_cache for backward
            bw_context_.saveForBackward("y", fwd.output);
            bw_context_.saveForBackward("h_cache", fwd.h_cache);
            bw_context_.saveForBackward("c_cache", fwd.c_cache);
            bw_context_.saveForBackward("x", x); // retain input sequence for backward

            return fwd;
        }

        void buildBackward(OpGraph &graph, const StreamingLstmBackwardInput &streaming_lstm_bwd_input,
                           const std::unordered_map<std::string, TraceTensor> &parameter_gradients,
                           const std::unordered_map<std::string, TraceTensor> &operand_gradients) override
        {
            const TraceTensor &x = bw_context_.getSaved("x");
            const TraceTensor &h_cache_host = bw_context_.getSaved("h_cache");
            const TraceTensor &c_cache_host = bw_context_.getSaved("c_cache");

            StreamingLstmBackwardInput bwd_input = streaming_lstm_bwd_input;
            auto dx_it = operand_gradients.find("input");
            if (dx_it != operand_gradients.end())
            {
                bwd_input.dx_host = dx_it->second;
            }

            std::optional<LstmBackwardStreamingResult> result_opt{};
            if (bwd_input.upstream_supplier.has_value())
            {
                result_opt = StreamingLstmBwdStreamedDy(
                    graph, x, h0, c0, weights_ih, weights_hh, bias_ih, bias_hh, bw_context_.getSaved("y"), h_cache_host,
                    c_cache_host, *bwd_input.upstream_supplier, bwd_input.upstream_supplier_stream_desc,
                    bwd_input.upstream_dy_dtype, bwd_input.d_hn, bwd_input.d_cn, recompute_interval_,
                    streaming_chunk_size_, bwd_input.dx_host, compute_stream_descriptor_);
            }
            else
            {
                if (!bwd_input.dY_host.has_value())
                {
                    throw std::runtime_error("StreamingLstm backward requires full upstream dY tensor for now");
                }
                result_opt = StreamingLstmBwd(graph, x, h0, c0, weights_ih, weights_hh, bias_ih, bias_hh,
                                              bw_context_.getSaved("y"), h_cache_host, c_cache_host, *bwd_input.dY_host,
                                              bwd_input.d_hn, bwd_input.d_cn, recompute_interval_,
                                              streaming_chunk_size_, bwd_input.dx_host, compute_stream_descriptor_);
            }

            const auto &[grad_x, grad_h0, grad_c0, grad_w_ih, grad_w_hh, grad_b_ih, grad_b_hh] = result_opt.value();

            const std::string w_ih_key = name_ + ".weights_ih";
            const std::string w_hh_key = name_ + ".weights_hh";
            const std::string b_ih_key = name_ + ".bias_ih";
            const std::string b_hh_key = name_ + ".bias_hh";
            const std::string h0_key = name_ + ".h0";
            const std::string c0_key = name_ + ".c0";

            InplaceAdd(graph, parameter_gradients.at(w_ih_key), grad_w_ih, GpuStreamDescriptors::Main);
            InplaceAdd(graph, parameter_gradients.at(w_hh_key), grad_w_hh, GpuStreamDescriptors::Main);
            InplaceAdd(graph, parameter_gradients.at(b_ih_key), grad_b_ih, GpuStreamDescriptors::Main);
            InplaceAdd(graph, parameter_gradients.at(b_hh_key), grad_b_hh, GpuStreamDescriptors::Main);

            const TraceTensor &h0_target = parameter_gradients.at(h0_key);
            if (grad_h0.shape() != h0_target.shape())
            {
                throw std::runtime_error("StreamingLstm backward grad_h0 shape mismatch with parameter gradient");
            }
            InplaceAdd(graph, h0_target, grad_h0, GpuStreamDescriptors::Main);

            const TraceTensor &c0_target = parameter_gradients.at(c0_key);
            if (grad_c0.shape() != c0_target.shape())
            {
                throw std::runtime_error("StreamingLstm backward grad_c0 shape mismatch with parameter gradient");
            }
            InplaceAdd(graph, c0_target, grad_c0, GpuStreamDescriptors::Main);

            if (dx_it != operand_gradients.end() && grad_x.id() != dx_it->second.id())
            {
                DeviceCopy(graph, grad_x, dx_it->second, GpuStreamDescriptors::Main);
            }

            bw_context_.release(graph);
        }

        void setChunkHook(StreamingChunkHook hook) { chunk_hook_ = std::move(hook); }

        void clearChunkHook() { chunk_hook_.reset(); }

        [[nodiscard]] size_t streamingChunkSize() const { return streaming_chunk_size_; }

        [[nodiscard]] size_t recomputeInterval() const { return recompute_interval_; }

        [[nodiscard]] DataType ioDtype() const { return io_dtype_; }

        [[nodiscard]] std::vector<ParameterEntry> parameters() const override
        {
            return {
                ParameterEntry{.name = name_ + '.' + "weights_ih", .tensor = weights_ih},
                ParameterEntry{.name = name_ + '.' + "weights_hh", .tensor = weights_hh},
                ParameterEntry{.name = name_ + '.' + "bias_ih", .tensor = bias_ih},
                ParameterEntry{.name = name_ + '.' + "bias_hh", .tensor = bias_hh},
                ParameterEntry{.name = name_ + '.' + "h0", .tensor = h0},
                ParameterEntry{.name = name_ + '.' + "c0", .tensor = c0},
            };
        }
    };

} // namespace pi::tensorlib
