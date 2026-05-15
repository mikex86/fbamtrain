#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace fbamtrain::ftccl
{
    enum class Result
    {
        Success,
        NotInitialized,
        InternalError,
        InvalidArgument,
        InvalidUsage,
        TooFewPeers,
        MasterConnectionFailed,
        RankConnectionFailed,
        RankConnectionLost,
        NoSharedStateAvailable,
        PendingAsyncOps,
        UpdateTopologyFailed,
        TopologyOptimizationFailed,
    };

    enum class DataType
    {
        UINT2,
        UINT4,
        UINT8,
        INT8,
        INT16,
        UINT16,
        UINT32,
        INT32,
        UINT64,
        INT64,
        FLOAT16,
        BFLOAT16,
        FLOAT32,
        FLOAT64,
    };

    enum class DeviceType
    {
        CPU,
        CUDA,
    };

    enum class RedOp
    {
        Sum,
        Avg,
        Prod,
        Max,
        Min,
    };

    enum class Attribute
    {
        GlobalWorldSize,
        PeerGroupWorldSize,
        NumDistinctPeerGroups,
        LargestPeerGroupWorldSize,
    };

    enum class SharedStateSyncStrategy
    {
        EnforcePopular,
        ReceiveOnly,
        SendOnly,
    };

    enum class DistributionHint
    {
        None,
        Normal,
        Uniform,
    };

    enum class QuantizationAlgorithm
    {
        None,
        MinMax,
        ZeroPointScale,
    };

    struct Ipv4Address
    {
        std::array<uint8_t, 4> bytes{};
    };

    struct Ipv6Address
    {
        std::array<uint8_t, 16> bytes{};
    };

    struct SocketAddress
    {
        enum class Protocol
        {
            Ipv4,
            Ipv6,
        };

        Protocol protocol{Protocol::Ipv4};
        Ipv4Address ipv4{};
        Ipv6Address ipv6{};
        uint16_t port{};
    };

    struct Communicator
    {
        void *handle{};
    };

    struct Master
    {
        void *handle{};
    };

    struct BuildInfo
    {
        bool has_cuda_support{};
    };

    struct CommCreateParams
    {
        SocketAddress master_address{};
        uint32_t peer_group{};
        uint32_t p2p_connection_pool_size{};
    };

    struct ReduceInfo
    {
        uint32_t local_world_size{};
        uint64_t tx_bytes{};
        uint64_t rx_bytes{};
    };

    struct ReduceOperandDescriptor
    {
        DataType datatype{};
        DistributionHint distribution_hint{DistributionHint::None};
    };

    struct QuantizationOptions
    {
        DataType quantized_datatype{};
        QuantizationAlgorithm algorithm{QuantizationAlgorithm::None};
    };

    struct ReduceDescriptor
    {
        size_t count{};
        RedOp op{RedOp::Sum};
        uint64_t tag{};
        ReduceOperandDescriptor src_descriptor{};
        QuantizationOptions quantization_options{};
    };

    struct ReduceOpDescriptor
    {
        const void *sendbuf{};
        void *recvbuf{};
        ReduceDescriptor descriptor{};
    };

    struct AsyncReduceOp
    {
        void *comm{};
        uint64_t tag{};
    };

    struct TensorInfo
    {
        const char *name{};
        void *data{};
        size_t count{};
        DataType datatype{};
        DeviceType device_type{};
        bool allow_content_inequality{};
    };

    struct SharedState
    {
        uint64_t revision{};
        size_t count{};
        TensorInfo *infos{};
    };

    struct SharedStateSyncInfo
    {
        uint64_t tx_bytes{};
        uint64_t rx_bytes{};
    };

    Result Init();
    Result GetBuildInfo(BuildInfo *info_out);
    [[nodiscard]] Result ParseSocketAddress(const char *ip, uint32_t port, SocketAddress *address_out);

    [[nodiscard]] Result CreateCommunicator(const CommCreateParams &params, Communicator *comm_out);
    [[nodiscard]] Result Connect(Communicator &comm);
    Result DestroyCommunicator(Communicator &comm);

    Result GetAttribute(const Communicator &comm, Attribute attribute, int *attribute_out);

    [[nodiscard]] Result UpdateTopology(Communicator &comm);
    [[nodiscard]] Result ArePeersPending(const Communicator &comm, bool *pending_out);
    [[nodiscard]] Result OptimizeTopology(const Communicator &comm);

    [[nodiscard]] Result SynchronizeSharedState(const Communicator &comm, SharedState &shared_state,
                                  SharedStateSyncStrategy strategy, SharedStateSyncInfo *sync_info_out);

    [[nodiscard]] Result AllReduce(const Communicator &comm, const void *send_buffer, void *recv_buffer,
                     const ReduceDescriptor &descriptor, ReduceInfo *reduce_info_out);
    [[nodiscard]] Result AllReduceAsync(const Communicator &comm, const void *send_buffer, void *recv_buffer,
                          const ReduceDescriptor &descriptor, AsyncReduceOp *op_out);
    [[nodiscard]] Result AwaitAsyncReduce(const AsyncReduceOp &op, ReduceInfo *reduce_info_out);
    [[nodiscard]] Result AllReduceMultipleWithRetry(const Communicator &comm, const ReduceOpDescriptor *descriptors,
                                      size_t descriptor_count, int max_in_flight, ReduceInfo *reduce_info_out);

    [[nodiscard]] Result CreateMaster(const SocketAddress &listen_address, Master *master_out);
    Result RunMaster(Master &master);
    Result InterruptMaster(Master &master);
    Result MasterAwaitTermination(Master &master);
    Result DestroyMaster(Master &master);

    const char *ResultString(Result result);
} // namespace fbamtrain::ftccl
