#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>

#include "activation.h"
#include "op_graph.h"
#include "transfer.h"

#include "tensorlib.h"

namespace pi::tensorlib
{
    // Forward declarations
    class OpGraph;

    struct LstmCellResult
    {
        TraceTensor h;
        TraceTensor c;
    };

    /**
     * Result structure for LSTM forward pass
     * NOTE: The user is responsible for deleting all contained tensors!
     */
    struct LstmForwardStreamingResult
    {
        /// (T, B, H) outputs in pinned host memory (FP16)
        TraceTensor output;

        /// (B, H) final hidden & cell state in device memory (FP16)
        TraceTensor h_n, c_n;

        /// (T, B, H) hidden-state gate cache in pinned host
        TraceTensor h_cache;

        /// (T, B, H) cell-state gate cache in pinned host
        TraceTensor c_cache;
    };

    /**
     * Result structure for LSTM backward pass
     * NOTE: The user is responsible for deleting all contained tensors!
     */
    struct LstmBackwardStreamingResult
    {
        /// (T, B, I) input gradients in pinned host memory (FP32)
        TraceTensor grad_x;
        TraceTensor grad_h0;
        TraceTensor grad_c0;
        TraceTensor grad_w_ih;
        TraceTensor grad_w_hh;
        TraceTensor grad_b_ih;
        TraceTensor grad_b_hh;
    };

    enum class Reduction
    {
        MEAN,
        ADD
    };

    using StreamingChunkHook = std::function<std::optional<GpuStreamDescriptor>(
        OpGraph &graph, const TraceTensor &output_chunk, uint64_t chunk_start, uint64_t chunk_steps)>;
    using StreamingDySupplier = std::function<std::optional<GpuStreamDescriptor>(
        OpGraph &graph, const TraceTensor &dst, uint64_t chunk_start, uint64_t chunk_steps)>;

    /**
     * Non-streaming backward over a fully materialised upstream gradient. Intended for testing / validation.
     * Supports checkpointed cell caches produced by streaming forward.
     */
    [[nodiscard]] LstmBackwardStreamingResult
    StreamingLstmBwd(OpGraph &graph, const TraceTensor &x, const TraceTensor &h0, const TraceTensor &c0,
                     const TraceTensor &w_ih, const TraceTensor &w_hh, const TraceTensor &b_ih, const TraceTensor &b_hh,
                     const TraceTensor &y_cache, const TraceTensor &h_cache, const TraceTensor &c_cache,
                     const TraceTensor &dy, const std::optional<TraceTensor> &dh_n,
                     const std::optional<TraceTensor> &dc_n, int recompute_interval, size_t streaming_chunk_size,
                     const std::optional<TraceTensor> &dx_host, const GpuStreamDescriptor &compute_stream_descriptor);

    /**
     * Streaming backward over chunked upstream gradients supplied on device.
     * The supplier must write dY for each chunk into the provided device buffer using dy_supplier_stream_desc.
     */
    [[nodiscard]] LstmBackwardStreamingResult StreamingLstmBwdStreamedDy(
        OpGraph &graph, const TraceTensor &x, const TraceTensor &h0, const TraceTensor &c0, const TraceTensor &w_ih,
        const TraceTensor &w_hh, const TraceTensor &b_ih, const TraceTensor &b_hh, const TraceTensor &y_cache,
        const TraceTensor &h_cache, const TraceTensor &c_cache, const StreamingDySupplier &dy_supplier,
        const GpuStreamDescriptor &dy_supplier_stream_desc, DataType dy_dtype, const std::optional<TraceTensor> &dh_n,
        const std::optional<TraceTensor> &dc_n, int recompute_interval, size_t streaming_chunk_size,
        const std::optional<TraceTensor> &dx_host, const GpuStreamDescriptor &compute_stream_descriptor);

    void DeviceCopy(OpGraph &graph, const TraceTensor &input, const TraceTensor &output,
                    const GpuStreamDescriptor &stream_desc);

    void FillZeros(OpGraph &graph, TraceTensor &tensor, const GpuStreamDescriptor &compute_stream_descriptor);

    void FillUniform(OpGraph &graph, TraceTensor &tensor, float min, float max, uint32_t seed,
                     const GpuStreamDescriptor &compute_stream_descriptor);

    void FillConstant(OpGraph &graph, TraceTensor &tensor, float value, const GpuStreamDescriptor &compute_stream_descriptor);

    void FillNormal(OpGraph &graph, TraceTensor &tensor, float mean, float stddev, uint32_t seed,
                    const GpuStreamDescriptor &compute_stream_descriptor);

    void KaimingUniformInit(OpGraph &graph, TraceTensor &tensor, uint32_t features, uint32_t seed,
                            const GpuStreamDescriptor &compute_stream_descriptor);

    void OptimizerSgd(OpGraph &graph, TraceTensor &param, const TraceTensor &grad, TraceTensor &velocity,
                      float learning_rate, float momentum, float weight_decay, bool nesterov,
                      const GpuStreamDescriptor &compute_stream_descriptor);

    void OptimizerAdamw(OpGraph &graph, TraceTensor &param, const TraceTensor &grad, TraceTensor &m, TraceTensor &v,
                        const TraceTensor &bias_correction1, const TraceTensor &bias_correction2, float learning_rate,
                        float beta1, float beta2, float eps, float weight_decay,
                        const GpuStreamDescriptor &compute_stream_descriptor);

    [[nodiscard]] TraceTensor ScaledDotProductAttentionFwd(OpGraph &graph, const TraceTensor &query,
                                                           const TraceTensor &key, const TraceTensor &value,
                                                           float softmax_scale, bool causal,
                                                           bool use_fp16_flash_attn_acc,
                                                           std::optional<TraceTensor> scratch_out,
                                                           const GpuStreamDescriptor &compute_stream_descriptor);

    struct ScaledDotProductAttentionBwdResult
    {
        TraceTensor grad_q;
        TraceTensor grad_k;
        TraceTensor grad_v;
    };

    void ScaledDotProductAttentionBwdInto(OpGraph &graph, const TraceTensor &query, const TraceTensor &key,
                                          const TraceTensor &value, const TraceTensor &output,
                                          const TraceTensor &scratch, const TraceTensor &upstream,
                                          const TraceTensor &grad_q, const TraceTensor &grad_k,
                                          const TraceTensor &grad_v, float softmax_scale, bool causal,
                                          const GpuStreamDescriptor &compute_stream_descriptor);

    [[nodiscard]] ScaledDotProductAttentionBwdResult
    ScaledDotProductAttentionBwd(OpGraph &graph, const TraceTensor &query, const TraceTensor &key,
                                 const TraceTensor &value, const TraceTensor &output, const TraceTensor &scratch,
                                 const TraceTensor &upstream, float softmax_scale, bool causal,
                                 const GpuStreamDescriptor &compute_stream_descriptor);

    [[nodiscard]] TraceTensor LayerNormFwd(OpGraph &graph, const TraceTensor &input, const TraceTensor &weight,
                                           const TraceTensor &bias, float eps, const GpuStreamDescriptor &compute_stream_descriptor);

    [[nodiscard]] TraceTensor RmsNormFwd(OpGraph &graph, const TraceTensor &input, const TraceTensor &weight,
                                         float eps, const GpuStreamDescriptor &compute_stream_descriptor);

    void RmsNormFwdInplace(OpGraph &graph, const TraceTensor &input, const TraceTensor &weight, float eps, const GpuStreamDescriptor &compute_stream_descriptor);

    void RmsNormBwd(OpGraph &graph, const TraceTensor &input, const TraceTensor &weight, const TraceTensor &upstream,
                    const TraceTensor &grad_input, const TraceTensor &x_hat, float eps,
                    const GpuStreamDescriptor &compute_stream_descriptor);

    [[nodiscard]] TraceTensor AvgPool1d(OpGraph &graph, const TraceTensor &input, uint32_t kernel_size, uint32_t stride,
                                        const GpuStreamDescriptor &compute_stream_descriptor, int64_t pool_dim = 2);

    [[nodiscard]] TraceTensor AvgPool2d(OpGraph &graph, const TraceTensor &input,
                                        const std::array<uint32_t, 2> &kernel_size,
                                        const std::array<uint32_t, 2> &stride,
                                        const GpuStreamDescriptor &compute_stream_descriptor,
                                        const std::array<uint32_t, 2> &padding = {0, 0}, bool channels_last = false);

    [[nodiscard]] TraceTensor Conv2d(OpGraph &graph, const TraceTensor &input, const TraceTensor &weight,
                                     const TraceTensor *bias, const std::array<uint32_t, 2> &stride,
                                     const std::array<uint32_t, 2> &padding, const GpuStreamDescriptor &compute_stream_descriptor,
                                     const std::array<uint32_t, 2> &dilation = std::array<uint32_t, 2>{1, 1},
                                     bool use_fp16_conv_acc = false);

    void Conv2dDgradInto(OpGraph &graph, const TraceTensor &upstream, const TraceTensor &weight,
                         const TraceTensor &grad_input, const std::array<uint32_t, 2> &stride,
                         const std::array<uint32_t, 2> &padding, const std::array<uint32_t, 2> &dilation,
                         const GpuStreamDescriptor &compute_stream_descriptor);

    void Conv2dWgradInto(OpGraph &graph, const TraceTensor &input, const TraceTensor &upstream,
                         const TraceTensor &grad_weight, const std::array<uint32_t, 2> &stride,
                         const std::array<uint32_t, 2> &padding, const std::array<uint32_t, 2> &dilation,
                         const GpuStreamDescriptor &compute_stream_descriptor, bool accumulate_output = false);

    [[nodiscard]] TraceTensor Mean(OpGraph &graph, const TraceTensor &input, int64_t dim, bool keepdim,
                                   const GpuStreamDescriptor &compute_stream_descriptor);
    [[nodiscard]] TraceTensor ReduceSum(OpGraph &graph, const TraceTensor &input, int64_t dim, bool keepdim,
                                        DataType output_dtype, const GpuStreamDescriptor &compute_stream_descriptor);

    [[nodiscard]] TraceTensor Div(OpGraph &graph, const TraceTensor &lhs, const TraceTensor &rhs,
                                  const GpuStreamDescriptor &compute_stream_descriptor);
    void Div(OpGraph &graph, const TraceTensor &lhs, const TraceTensor &rhs, const TraceTensor &output,
             const GpuStreamDescriptor &compute_stream_descriptor);
    [[nodiscard]] TraceTensor Sqrt(OpGraph &graph, const TraceTensor &input,
                                   const GpuStreamDescriptor &compute_stream_descriptor);
    void Sqrt(OpGraph &graph, const TraceTensor &input, const TraceTensor &output,
              const GpuStreamDescriptor &compute_stream_descriptor);

    void InplaceAdd(OpGraph &graph, const TraceTensor &dst, const TraceTensor &rhs,
                    const GpuStreamDescriptor &compute_stream_descriptor);
    void InplaceMul(OpGraph &graph, const TraceTensor &dst, const TraceTensor &rhs,
                    const GpuStreamDescriptor &compute_stream_descriptor);
    void InplaceDiv(OpGraph &graph, const TraceTensor &dst, const TraceTensor &rhs,
                    const GpuStreamDescriptor &compute_stream_descriptor);

    [[nodiscard]] TraceTensor Gelu(OpGraph &graph, const TraceTensor &input, const GpuStreamDescriptor &compute_stream_descriptor);

    [[nodiscard]] TraceTensor GeluBackward(OpGraph &graph, const TraceTensor &input, const TraceTensor &upstream,
                                           const GpuStreamDescriptor &compute_stream_descriptor);

    void GeluInplace(OpGraph &graph, const TraceTensor &tensor, const GpuStreamDescriptor &compute_stream_descriptor);

    [[nodiscard]] TraceTensor Relu(OpGraph &graph, const TraceTensor &input, const GpuStreamDescriptor &compute_stream_descriptor);

    void ReluInplace(OpGraph &graph, const TraceTensor &tensor, const GpuStreamDescriptor &compute_stream_descriptor);

    [[nodiscard]] LstmCellResult LstmCellFwd(OpGraph &graph, const TraceTensor &gates, const TraceTensor &c_prev,
                                             const GpuStreamDescriptor &compute_stream_descriptor);

    void LstmCellFwdInplace(OpGraph &graph, const TraceTensor &gates, const TraceTensor &c_prev,
                            const TraceTensor &h_out, const TraceTensor &c_out,
                            const GpuStreamDescriptor &compute_stream_descriptor);

    void LstmCellFwdInplace(OpGraph &graph, const TraceTensor &gates, const TraceTensor &c_prev,
                            const TraceTensor &h_out, const TraceTensor &c_out, const TraceTensor &y_out,
                            const GpuStreamDescriptor &compute_stream_descriptor);

    [[nodiscard]] TraceTensor CrossEntropyOnTargets(OpGraph &graph, const TraceTensor &logits,
                                                    const TraceTensor &targets, Reduction reduction,
                                                    const GpuStreamDescriptor &compute_stream_descriptor,
                                                    bool reduce_over_rows = false);

    void CrossEntropyOnTargets(OpGraph &graph, const TraceTensor &logits, const TraceTensor &targets,
                               const TraceTensor &output, Reduction reduction,
                               const GpuStreamDescriptor &compute_stream_descriptor,
                               bool reduce_over_rows = false);

    [[nodiscard]] TraceTensor CrossEntropyOnTargetsBackward(OpGraph &graph, const TraceTensor &logits,
                                                            const TraceTensor &targets, const TraceTensor &upstream,
                                                            Reduction reduction,
                                                            const GpuStreamDescriptor &compute_stream_descriptor,
                                                            bool reduce_over_rows = false, bool is_useful = true);

    // 1D partial sum reduction: output length = ceil(input.numel() / block_size).
    [[nodiscard]] TraceTensor ReduceSumPartial(OpGraph &graph, const TraceTensor &input,
                                               const GpuStreamDescriptor &compute_stream_descriptor, int64_t block_size = 128);

    [[nodiscard]] LstmForwardStreamingResult
    StreamingLstmFwd(OpGraph &graph, const TraceTensor &x_tensor_host,
                     // (T, B, I) inputs in pinned host memory (FP16)
                     const TraceTensor &h_0_tensor,
                     // (B, H) initial hidden state in pinned host memory (FP16)
                     const TraceTensor &c_0_tensor,
                     // (B, H) initial cell state in pinned host memory (FP16)

                     const TraceTensor &w_ih_tensor,
                     // (4H, I) input-hidden weights in device memory (FP16)
                     const TraceTensor &w_hh_tensor,
                     // (4H, H) hidden-hidden weights in device memory (FP16)
                     const TraceTensor &b_ih_tensor,
                     // (4H) input-hidden bias in device memory (FP16) or nullptr
                     const TraceTensor &b_hh_tensor,
                     // data type for hidden state gate caches (FP16 or FP32)
                     DataType c_cache_dtype,
                     // data type for cell state gate caches (FP16 or FP32)

                     int recompute_interval,
                     // interval for recomputation of gate activations (1 = no recomputation,
                     // 2=ever 2nd step is retained, etc.)

                     size_t streaming_chunk_size, // size of streaming chunks (number of time steps processed at once)

                     const GpuStreamDescriptor &compute_stream_descriptor,

                     StreamingChunkHook chunk_hook = nullptr // optional per-chunk hook invoked while outputs are on GPU
    );

    /**
     * Instructs the main stream to wait for the completion of all async H2D/D2H transfers.
     * @param graph the operation graph
     * @param transfer_side the side of the transfers to await (H2D or D2H)
     * @param gpu_device the gpu device which the transfer is associated with
     * @param waiting_stream_desc the stream descriptor for the stream that should wait
     */
    void AwaitAsyncTransfers(OpGraph &graph, TransferType transfer_side, const Device &gpu_device,
                             const GpuStreamDescriptor &waiting_stream_desc);

    /**
     * Instructs a transfer stream (H2D/D2H) to wait for main-stream compute before reusing staging buffers.
     * @param graph the operation graph
     * @param transfer_side the transfer side to gate (H2D or D2H)
     * @param gpu_device the gpu device which the transfer is associated with
     * @param waiting_stream_desc the stream descriptor for the stream that should wait
     */
    void AwaitComputeForTransfer(OpGraph &graph, TransferType transfer_side, const Device &gpu_device,
                                 const GpuStreamDescriptor &waiting_stream_desc);

    /**
     * Instructs the main stream to wait for a given compute stream.
     * @param graph the operation graph
     * @param device
     * @param compute_stream_descriptor the compute stream to await
     */
    void AwaitComputeStream(OpGraph &graph, const Device &device, const GpuStreamDescriptor &compute_stream_descriptor);

    /**
     * Instructs a compute stream to wait for prior work on the main stream.
     * @param graph the operation graph
     * @param device
     * @param compute_stream_descriptor the compute stream that should wait on main
     */
    void AwaitMainStream(OpGraph &graph, const Device &device, const GpuStreamDescriptor &compute_stream_descriptor);

} // namespace pi::tensorlib
