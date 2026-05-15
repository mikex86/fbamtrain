#include "ftccl.h"

#include <pccl.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <limits>
#include <vector>

namespace fbamtrain::ftccl
{
    namespace
    {
        pcclComm_t *AsPccl(const Communicator &comm)
        {
            return static_cast<pcclComm_t *>(comm.handle);
        }

        pcclMasterInstance_t *AsPccl(const Master &master)
        {
            return static_cast<pcclMasterInstance_t *>(master.handle);
        }

        Result FromPccl(const pcclResult_t result)
        {
            switch (result)
            {
            case pcclSuccess:
                return Result::Success;
            case pcclNotInitialized:
                return Result::NotInitialized;
            case pcclInternalError:
                return Result::InternalError;
            case pcclInvalidArgument:
                return Result::InvalidArgument;
            case pcclInvalidUsage:
                return Result::InvalidUsage;
            case pcclTooFewPeers:
                return Result::TooFewPeers;
            case pcclMasterConnectionFailed:
                return Result::MasterConnectionFailed;
            case pcclRankConnectionFailed:
                return Result::RankConnectionFailed;
            case pcclRankConnectionLost:
                return Result::RankConnectionLost;
            case pcclNoSharedStateAvailable:
                return Result::NoSharedStateAvailable;
            case pcclPendingAsyncOps:
                return Result::PendingAsyncOps;
            case pcclUpdateTopologyFailed:
                return Result::UpdateTopologyFailed;
            case pcclTopologyOptimizationFailed:
                return Result::TopologyOptimizationFailed;
            }
            return Result::InternalError;
        }

        pcclDataType_t ToPccl(const DataType datatype)
        {
            switch (datatype)
            {
            case DataType::UINT2:
                return pcclUint2;
            case DataType::UINT4:
                return pcclUint4;
            case DataType::UINT8:
                return pcclUint8;
            case DataType::INT8:
                return pcclInt8;
            case DataType::INT16:
                return pcclInt16;
            case DataType::UINT16:
                return pcclUint16;
            case DataType::UINT32:
                return pcclUint32;
            case DataType::INT32:
                return pcclInt32;
            case DataType::UINT64:
                return pcclUint64;
            case DataType::INT64:
                return pcclInt64;
            case DataType::FLOAT16:
                return pcclFloat16;
            case DataType::BFLOAT16:
                return pcclBFloat16;
            case DataType::FLOAT32:
                return pcclFloat;
            case DataType::FLOAT64:
                return pcclDouble;
            }
            return pcclFloat;
        }

        pcclDeviceType_t ToPccl(const DeviceType device_type)
        {
            switch (device_type)
            {
            case DeviceType::CPU:
                return pcclDeviceCpu;
            case DeviceType::CUDA:
                return pcclDeviceCuda;
            }
            return pcclDeviceCpu;
        }

        pcclRedOp_t ToPccl(const RedOp op)
        {
            switch (op)
            {
            case RedOp::Sum:
                return pcclSum;
            case RedOp::Avg:
                return pcclAvg;
            case RedOp::Prod:
                return pcclProd;
            case RedOp::Max:
                return pcclMax;
            case RedOp::Min:
                return pcclMin;
            }
            return pcclSum;
        }

        pcclAttribute_t ToPccl(const Attribute attribute)
        {
            switch (attribute)
            {
            case Attribute::GlobalWorldSize:
                return PCCL_ATTRIBUTE_GLOBAL_WORLD_SIZE;
            case Attribute::PeerGroupWorldSize:
                return PCCL_ATTRIBUTE_PEER_GROUP_WORLD_SIZE;
            case Attribute::NumDistinctPeerGroups:
                return PCCL_ATTRIBUTE_NUM_DISTINCT_PEER_GROUPS;
            case Attribute::LargestPeerGroupWorldSize:
                return PCCL_ATTRIBUTE_LARGEST_PEER_GROUP_WORLD_SIZE;
            }
            return PCCL_ATTRIBUTE_GLOBAL_WORLD_SIZE;
        }

        pcclSharedStateSyncStrategy_t ToPccl(const SharedStateSyncStrategy strategy)
        {
            switch (strategy)
            {
            case SharedStateSyncStrategy::EnforcePopular:
                return PCCL_SHARED_STATE_SYNC_STRATEGY_ENFORCE_POPULAR;
            case SharedStateSyncStrategy::ReceiveOnly:
                return PCCL_SHARED_STATE_SYNC_STRATEGY_RECEIVE_ONLY;
            case SharedStateSyncStrategy::SendOnly:
                return PCCL_SHARED_STATE_SYNC_STRATEGY_SEND_ONLY;
            }
            return PCCL_SHARED_STATE_SYNC_STRATEGY_ENFORCE_POPULAR;
        }

        pcclDistributionHint_t ToPccl(const DistributionHint hint)
        {
            switch (hint)
            {
            case DistributionHint::None:
                return pcclDistributionNone;
            case DistributionHint::Normal:
                return pcclDistributionNormal;
            case DistributionHint::Uniform:
                return pcclDistributionUniform;
            }
            return pcclDistributionNone;
        }

        pcclQuantizationAlgorithm_t ToPccl(const QuantizationAlgorithm algorithm)
        {
            switch (algorithm)
            {
            case QuantizationAlgorithm::None:
                return pcclQuantNone;
            case QuantizationAlgorithm::MinMax:
                return pcclQuantMinMax;
            case QuantizationAlgorithm::ZeroPointScale:
                return pcclQuantZeroPointScale;
            }
            return pcclQuantNone;
        }

        ccoip_socket_address_t ToPccl(const SocketAddress &address)
        {
            ccoip_socket_address_t pccl_address{};
            pccl_address.inet.protocol =
                address.protocol == SocketAddress::Protocol::Ipv4 ? inetIPv4 : inetIPv6;
            std::copy(address.ipv4.bytes.begin(), address.ipv4.bytes.end(), pccl_address.inet.ipv4.data);
            std::copy(address.ipv6.bytes.begin(), address.ipv6.bytes.end(), pccl_address.inet.ipv6.data);
            pccl_address.port = address.port;
            return pccl_address;
        }

        pcclReduceOperandDescriptor_t ToPccl(const ReduceOperandDescriptor &descriptor)
        {
            return pcclReduceOperandDescriptor_t{
                .datatype = ToPccl(descriptor.datatype),
                .distribution_hint = ToPccl(descriptor.distribution_hint),
            };
        }

        pcclQuantizationOptions_t ToPccl(const QuantizationOptions &options)
        {
            return pcclQuantizationOptions_t{
                .quantized_datatype = ToPccl(options.quantized_datatype),
                .algorithm = ToPccl(options.algorithm),
            };
        }

        pcclReduceDescriptor_t ToPccl(const ReduceDescriptor &descriptor)
        {
            return pcclReduceDescriptor_t{
                .count = descriptor.count,
                .op = ToPccl(descriptor.op),
                .tag = descriptor.tag,
                .src_descriptor = ToPccl(descriptor.src_descriptor),
                .quantization_options = ToPccl(descriptor.quantization_options),
            };
        }

        pcclReduceOpDescriptor_t ToPccl(const ReduceOpDescriptor &descriptor)
        {
            return pcclReduceOpDescriptor_t{
                .sendbuf = const_cast<void *>(descriptor.sendbuf),
                .recvbuf = descriptor.recvbuf,
                .descriptor = ToPccl(descriptor.descriptor),
            };
        }

        pcclTensorInfo_t ToPccl(const TensorInfo &info)
        {
            return pcclTensorInfo_t{
                .name = info.name,
                .data = info.data,
                .count = info.count,
                .datatype = ToPccl(info.datatype),
                .device_type = ToPccl(info.device_type),
                .allow_content_inequality = info.allow_content_inequality,
            };
        }

        void FromPccl(const pcclReduceInfo_t &pccl_info, ReduceInfo &info)
        {
            info.local_world_size = pccl_info.local_world_size;
            info.tx_bytes = pccl_info.tx_bytes;
            info.rx_bytes = pccl_info.rx_bytes;
        }

        void FromPccl(const pcclSharedStateSyncInfo_t &pccl_info, SharedStateSyncInfo &info)
        {
            info.tx_bytes = pccl_info.tx_bytes;
            info.rx_bytes = pccl_info.rx_bytes;
        }
    } // namespace

    Result Init()
    {
        return FromPccl(pcclInit());
    }

    Result GetBuildInfo(BuildInfo *info_out)
    {
        if (info_out == nullptr)
        {
            return Result::InvalidArgument;
        }

        pcclBuildInfo_t pccl_info{};
        const Result result = FromPccl(pcclGetBuildInfo(&pccl_info));
        if (result == Result::Success)
        {
            info_out->has_cuda_support = pccl_info.has_cuda_support;
        }
        return result;
    }

    Result ParseSocketAddress(const char *ip, const uint32_t port, SocketAddress *address_out)
    {
        if (ip == nullptr || address_out == nullptr || port > std::numeric_limits<uint16_t>::max())
        {
            return Result::InvalidArgument;
        }

        std::array<unsigned long, 4> octets{};
        char extra = '\0';
        const int parsed = std::sscanf(ip, "%lu.%lu.%lu.%lu%c", &octets[0], &octets[1], &octets[2], &octets[3],
                                       &extra);
        if (parsed != 4)
        {
            return Result::InvalidArgument;
        }

        SocketAddress address{};
        address.protocol = SocketAddress::Protocol::Ipv4;
        address.port = static_cast<uint16_t>(port);
        for (size_t idx = 0; idx < octets.size(); ++idx)
        {
            if (octets[idx] > 255)
            {
                return Result::InvalidArgument;
            }
            address.ipv4.bytes[idx] = static_cast<uint8_t>(octets[idx]);
        }

        *address_out = address;
        return Result::Success;
    }

    Result CreateCommunicator(const CommCreateParams &params, Communicator *comm_out)
    {
        if (comm_out == nullptr)
        {
            return Result::InvalidArgument;
        }

        pcclCommCreateParams_t pccl_params{
            .master_address = ToPccl(params.master_address),
            .peer_group = params.peer_group,
            .p2p_connection_pool_size = params.p2p_connection_pool_size,
        };

        pcclComm_t *pccl_comm{};
        const Result result = FromPccl(pcclCreateCommunicator(&pccl_params, &pccl_comm));
        if (result == Result::Success)
        {
            comm_out->handle = pccl_comm;
        }
        return result;
    }

    Result Connect(Communicator &comm)
    {
        return FromPccl(pcclConnect(AsPccl(comm)));
    }

    Result DestroyCommunicator(Communicator &comm)
    {
        if (comm.handle == nullptr)
        {
            return Result::Success;
        }
        const Result result = FromPccl(pcclDestroyCommunicator(AsPccl(comm)));
        if (result == Result::Success)
        {
            comm.handle = nullptr;
        }
        return result;
    }

    Result GetAttribute(const Communicator &comm, Attribute attribute, int *attribute_out)
    {
        return FromPccl(pcclGetAttribute(AsPccl(comm), ToPccl(attribute), attribute_out));
    }

    Result UpdateTopology(Communicator &comm)
    {
        return FromPccl(pcclUpdateTopology(AsPccl(comm)));
    }

    Result ArePeersPending(const Communicator &comm, bool *pending_out)
    {
        return FromPccl(pcclArePeersPending(AsPccl(comm), pending_out));
    }

    Result OptimizeTopology(const Communicator &comm)
    {
        return FromPccl(pcclOptimizeTopology(AsPccl(comm)));
    }

    Result SynchronizeSharedState(const Communicator &comm, SharedState &shared_state,
                                  SharedStateSyncStrategy strategy, SharedStateSyncInfo *sync_info_out)
    {
        if (shared_state.infos == nullptr && shared_state.count > 0)
        {
            return Result::InvalidArgument;
        }

        std::vector<pcclTensorInfo_t> infos;
        infos.reserve(shared_state.count);
        for (size_t i = 0; i < shared_state.count; ++i)
        {
            infos.push_back(ToPccl(shared_state.infos[i]));
        }

        pcclSharedState_t pccl_state{
            .revision = shared_state.revision,
            .count = shared_state.count,
            .infos = infos.data(),
        };

        pcclSharedStateSyncInfo_t pccl_sync_info{};
        const Result result = FromPccl(pcclSynchronizeSharedState(AsPccl(comm), &pccl_state, ToPccl(strategy),
                                                                  sync_info_out == nullptr ? nullptr
                                                                                           : &pccl_sync_info));
        if (result == Result::Success)
        {
            shared_state.revision = pccl_state.revision;
            if (sync_info_out != nullptr)
            {
                FromPccl(pccl_sync_info, *sync_info_out);
            }
        }
        return result;
    }

    Result AllReduce(const Communicator &comm, const void *send_buffer, void *recv_buffer,
                     const ReduceDescriptor &descriptor, ReduceInfo *reduce_info_out)
    {
        pcclReduceDescriptor_t pccl_descriptor = ToPccl(descriptor);
        pcclReduceInfo_t pccl_info{};
        const Result result = FromPccl(pcclAllReduce(send_buffer, recv_buffer, &pccl_descriptor, AsPccl(comm),
                                                     reduce_info_out == nullptr ? nullptr : &pccl_info));
        if (result == Result::Success && reduce_info_out != nullptr)
        {
            FromPccl(pccl_info, *reduce_info_out);
        }
        return result;
    }

    Result AllReduceAsync(const Communicator &comm, const void *send_buffer, void *recv_buffer,
                          const ReduceDescriptor &descriptor, AsyncReduceOp *op_out)
    {
        if (op_out == nullptr)
        {
            return Result::InvalidArgument;
        }

        pcclReduceDescriptor_t pccl_descriptor = ToPccl(descriptor);
        pcclAsyncReduceOp_t pccl_op{};
        const Result result =
            FromPccl(pcclAllReduceAsync(send_buffer, recv_buffer, &pccl_descriptor, AsPccl(comm), &pccl_op));
        if (result == Result::Success)
        {
            op_out->comm = pccl_op.comm;
            op_out->tag = pccl_op.tag;
        }
        return result;
    }

    Result AwaitAsyncReduce(const AsyncReduceOp &op, ReduceInfo *reduce_info_out)
    {
        pcclAsyncReduceOp_t pccl_op{
            .comm = static_cast<pcclComm_t *>(op.comm),
            .tag = op.tag,
        };
        pcclReduceInfo_t pccl_info{};
        const Result result =
            FromPccl(pcclAwaitAsyncReduce(&pccl_op, reduce_info_out == nullptr ? nullptr : &pccl_info));
        if (result == Result::Success && reduce_info_out != nullptr)
        {
            FromPccl(pccl_info, *reduce_info_out);
        }
        return result;
    }

    Result AllReduceMultipleWithRetry(const Communicator &comm, const ReduceOpDescriptor *descriptors,
                                      size_t descriptor_count, int max_in_flight, ReduceInfo *reduce_info_out)
    {
        if (descriptors == nullptr && descriptor_count > 0)
        {
            return Result::InvalidArgument;
        }

        std::vector<pcclReduceOpDescriptor_t> pccl_descriptors;
        pccl_descriptors.reserve(descriptor_count);
        for (size_t i = 0; i < descriptor_count; ++i)
        {
            pccl_descriptors.push_back(ToPccl(descriptors[i]));
        }

        pcclReduceInfo_t pccl_info{};
        const Result result = FromPccl(pcclAllReduceMultipleWithRetry(
            pccl_descriptors.data(), descriptor_count, AsPccl(comm), reduce_info_out == nullptr ? nullptr : &pccl_info,
            max_in_flight));
        if (result == Result::Success && reduce_info_out != nullptr)
        {
            FromPccl(pccl_info, *reduce_info_out);
        }
        return result;
    }

    Result CreateMaster(const SocketAddress &listen_address, Master *master_out)
    {
        if (master_out == nullptr)
        {
            return Result::InvalidArgument;
        }

        pcclMasterInstance_t *pccl_master{};
        const Result result = FromPccl(pcclCreateMaster(ToPccl(listen_address), &pccl_master));
        if (result == Result::Success)
        {
            master_out->handle = pccl_master;
        }
        return result;
    }

    Result RunMaster(Master &master)
    {
        return FromPccl(pcclRunMaster(AsPccl(master)));
    }

    Result InterruptMaster(Master &master)
    {
        return FromPccl(pcclInterruptMaster(AsPccl(master)));
    }

    Result MasterAwaitTermination(Master &master)
    {
        return FromPccl(pcclMasterAwaitTermination(AsPccl(master)));
    }

    Result DestroyMaster(Master &master)
    {
        if (master.handle == nullptr)
        {
            return Result::Success;
        }
        const Result result = FromPccl(pcclDestroyMaster(AsPccl(master)));
        if (result == Result::Success)
        {
            master.handle = nullptr;
        }
        return result;
    }

    const char *ResultString(const Result result)
    {
        switch (result)
        {
        case Result::Success:
            return "Success";
        case Result::NotInitialized:
            return "NotInitialized";
        case Result::InternalError:
            return "InternalError";
        case Result::InvalidArgument:
            return "InvalidArgument";
        case Result::InvalidUsage:
            return "InvalidUsage";
        case Result::TooFewPeers:
            return "TooFewPeers";
        case Result::MasterConnectionFailed:
            return "MasterConnectionFailed";
        case Result::RankConnectionFailed:
            return "RankConnectionFailed";
        case Result::RankConnectionLost:
            return "RankConnectionLost";
        case Result::NoSharedStateAvailable:
            return "NoSharedStateAvailable";
        case Result::PendingAsyncOps:
            return "PendingAsyncOps";
        case Result::UpdateTopologyFailed:
            return "UpdateTopologyFailed";
        case Result::TopologyOptimizationFailed:
            return "TopologyOptimizationFailed";
        }
        return "Unknown";
    }
} // namespace fbamtrain::ftccl
