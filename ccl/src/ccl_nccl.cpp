#include "ccl.h"

#include "ctx_management.h"

#include <cuda_runtime.h>
#include <nccl.h>

#include <cstring>
#include <stdexcept>
#include <string>

namespace fbamtrain::ccl
{
    namespace
    {
        constexpr int kMaxMessageBytes = 256;

        void CheckNccl(const ncclResult_t result, const char *message)
        {
            if (result != ncclSuccess)
            {
                throw std::runtime_error(std::string(message) + ": " + ncclGetErrorString(result));
            }
        }

        void CheckCuda(const cudaError_t result, const char *message)
        {
            if (result != cudaSuccess)
            {
                throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(result));
            }
        }

        ncclComm_t AsNccl(const Communicator &comm)
        {
            return static_cast<ncclComm_t>(comm.handle);
        }

        struct StreamBuffer
        {
            cudaStream_t stream{};
            char *device_buffer{};
        };

        StreamBuffer AllocateBuffer()
        {
            StreamBuffer buffer{};
            // Use the legacy default stream to avoid triggering NCCL strong-stream creation.
            buffer.stream = nullptr;
            CheckCuda(cudaMalloc(&buffer.device_buffer, kMaxMessageBytes), "Failed to allocate CUDA buffer for CCL");
            CheckCuda(cudaMemset(buffer.device_buffer, 0, kMaxMessageBytes), "Failed to clear CUDA buffer for CCL");
            return buffer;
        }

        void FreeBuffer(StreamBuffer &buffer)
        {
            if (buffer.device_buffer)
            {
                cudaFree(buffer.device_buffer);
                buffer.device_buffer = nullptr;
            }
            buffer.stream = nullptr;
        }

    }

    UniqueId GenerateUniqueId()
    {
        ncclUniqueId nccl_id{};
        CheckNccl(ncclGetUniqueId(&nccl_id), "Failed to get NCCL unique ID");
        UniqueId id{};
        static_assert(sizeof(nccl_id.internal) == kUniqueIdBytes, "NCCL unique ID size mismatch");
        std::memcpy(id.bytes.data(), nccl_id.internal, kUniqueIdBytes);
        return id;
    }

    Communicator InitCommunicator(const UniqueId &unique_id, const int world_size, const int rank, const int device_ordinal)
    {
        ncclUniqueId nccl_id{};
        static_assert(sizeof(nccl_id.internal) == kUniqueIdBytes, "NCCL unique ID size mismatch");
        std::memcpy(nccl_id.internal, unique_id.bytes.data(), kUniqueIdBytes);

        pi::tensorlib::internal::ctxmgmt::GpuSetCurrentDevice(device_ordinal);
        ncclComm_t comm{};
        CheckNccl(ncclCommInitRank(&comm, world_size, nccl_id, rank), "Failed to init NCCL communicator");
        return Communicator{.handle = reinterpret_cast<void *>(comm), .rank = rank, .world_size = world_size};
    }

    std::optional<Communicator> CreateSubsetCommunicator(Communicator &parent, const int color, const int key,
                                                         const int device_ordinal)
    {
        if (!parent.handle)
        {
            throw std::runtime_error("Cannot split an uninitialized CCL communicator.");
        }
        pi::tensorlib::internal::ctxmgmt::GpuSetCurrentDevice(device_ordinal);
        ncclComm_t child{};

        // NCCL uses the MPI-style "color" term for subset membership; keep that detail contained here.
        const int nccl_color = color == kSubsetNoColor ? NCCL_SPLIT_NOCOLOR : color;
        CheckNccl(ncclCommSplit(AsNccl(parent), nccl_color, key, &child, nullptr),
                  "Failed to create NCCL subset communicator");
        if (color == kSubsetNoColor || child == nullptr)
        {
            return std::nullopt;
        }

        int child_rank = 0;
        int child_world_size = 0;
        CheckNccl(ncclCommUserRank(child, &child_rank), "Failed to query split communicator rank");
        CheckNccl(ncclCommCount(child, &child_world_size), "Failed to query split communicator world size");
        return Communicator{.handle = reinterpret_cast<void *>(child), .rank = child_rank, .world_size = child_world_size};
    }

    void Destroy(Communicator &comm)
    {
        if (!comm.handle)
        {
            return;
        }
        CheckNccl(ncclCommDestroy(AsNccl(comm)), "Failed to destroy NCCL communicator");
        comm.handle = nullptr;
    }

    bool Abort(Communicator &comm) noexcept
    {
        if (!comm.handle)
        {
            return true;
        }
        const auto result = ncclCommAbort(AsNccl(comm));
        comm.handle = nullptr;
        return result == ncclSuccess;
    }

    void SendString(Communicator &comm, const int peer_rank, const int device_ordinal, const std::string &message)
    {
        pi::tensorlib::internal::ctxmgmt::GpuSetCurrentDevice(device_ordinal);
        StreamBuffer buffer = AllocateBuffer();
        std::string payload = message;
        if (payload.size() >= kMaxMessageBytes)
        {
            payload.resize(kMaxMessageBytes - 1);
        }
        payload.push_back('\0');
        CheckCuda(cudaMemcpy(buffer.device_buffer, payload.data(), payload.size(), cudaMemcpyHostToDevice),
                  "Failed to copy CCL message to device");
        const auto send_result =
            ncclSend(buffer.device_buffer, kMaxMessageBytes, ncclChar, peer_rank, AsNccl(comm), buffer.stream);
        if (send_result != ncclSuccess)
        {
            const auto cuda_result = cudaGetLastError();
            const auto async_result = [&comm]() {
                ncclResult_t async_error = ncclSuccess;
                ncclCommGetAsyncError(AsNccl(comm), &async_error);
                return async_error;
            }();
            throw std::runtime_error(std::string("Failed to send CCL message: ") + ncclGetErrorString(send_result) +
                                     " (cuda=" + cudaGetErrorString(cuda_result) +
                                     ", async=" + ncclGetErrorString(async_result) + ")");
        }
        CheckCuda(cudaStreamSynchronize(buffer.stream), "Failed to synchronize CCL send");
        FreeBuffer(buffer);
    }

    std::string RecvString(Communicator &comm, const int peer_rank, const int device_ordinal)
    {
        pi::tensorlib::internal::ctxmgmt::GpuSetCurrentDevice(device_ordinal);
        StreamBuffer buffer = AllocateBuffer();
        const auto recv_result =
            ncclRecv(buffer.device_buffer, kMaxMessageBytes, ncclChar, peer_rank, AsNccl(comm), buffer.stream);
        if (recv_result != ncclSuccess)
        {
            const auto cuda_result = cudaGetLastError();
            const auto async_result = [&comm]() {
                ncclResult_t async_error = ncclSuccess;
                ncclCommGetAsyncError(AsNccl(comm), &async_error);
                return async_error;
            }();
            throw std::runtime_error(std::string("Failed to receive CCL message: ") + ncclGetErrorString(recv_result) +
                                     " (cuda=" + cudaGetErrorString(cuda_result) +
                                     ", async=" + ncclGetErrorString(async_result) + ")");
        }
        CheckCuda(cudaStreamSynchronize(buffer.stream), "Failed to synchronize CCL receive");
        std::string message(kMaxMessageBytes, '\0');
        CheckCuda(cudaMemcpy(message.data(), buffer.device_buffer, kMaxMessageBytes, cudaMemcpyDeviceToHost),
                  "Failed to copy CCL message from device");
        const auto end = message.find('\0');
        if (end != std::string::npos)
        {
            message.resize(end);
        }
        FreeBuffer(buffer);
        return message;
    }

    void Barrier(Communicator &comm, const int device_ordinal)
    {
        pi::tensorlib::internal::ctxmgmt::GpuSetCurrentDevice(device_ordinal);
        constexpr uint32_t kBarrierValue = 1;
        uint32_t host_value = kBarrierValue;
        uint32_t host_result = 0;
        uint32_t *device_buffer = nullptr;
        CheckCuda(cudaMalloc(&device_buffer, sizeof(uint32_t)), "Failed to allocate CCL barrier buffer");
        CheckCuda(cudaMemcpy(device_buffer, &host_value, sizeof(uint32_t), cudaMemcpyHostToDevice),
                  "Failed to copy CCL barrier input");
        const auto reduce_result =
            ncclAllReduce(device_buffer, device_buffer, 1, ncclUint32, ncclSum, AsNccl(comm), nullptr);
        if (reduce_result != ncclSuccess)
        {
            const auto cuda_result = cudaGetLastError();
            const auto async_result = [&comm]() {
                ncclResult_t async_error = ncclSuccess;
                ncclCommGetAsyncError(AsNccl(comm), &async_error);
                return async_error;
            }();
            cudaFree(device_buffer);
            throw std::runtime_error(std::string("Failed to run CCL barrier allreduce: ") +
                                     ncclGetErrorString(reduce_result) + " (cuda=" +
                                     cudaGetErrorString(cuda_result) + ", async=" +
                                     ncclGetErrorString(async_result) + ")");
        }
        CheckCuda(cudaStreamSynchronize(nullptr), "Failed to synchronize CCL barrier");
        CheckCuda(cudaMemcpy(&host_result, device_buffer, sizeof(uint32_t), cudaMemcpyDeviceToHost),
                  "Failed to copy CCL barrier output");
        cudaFree(device_buffer);
        if (host_result != static_cast<uint32_t>(comm.world_size))
        {
            throw std::runtime_error("CCL barrier verification failed: expected sum " +
                                     std::to_string(comm.world_size) + ", got " + std::to_string(host_result));
        }
    }

    void SendBufferAsync(Communicator &comm, const int peer_rank, const int device_ordinal, const void *device_buffer,
                         const size_t num_bytes, CclGpuStream stream_handle)
    {
        if (num_bytes == 0)
        {
            return;
        }
        pi::tensorlib::internal::ctxmgmt::GpuSetCurrentDevice(device_ordinal);
        const cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);
        const auto send_result =
            ncclSend(device_buffer, num_bytes, ncclChar, peer_rank, AsNccl(comm), stream);
        if (send_result != ncclSuccess)
        {
            const auto cuda_result = cudaGetLastError();
            const auto async_result = [&comm]() {
                ncclResult_t async_error = ncclSuccess;
                ncclCommGetAsyncError(AsNccl(comm), &async_error);
                return async_error;
            }();
            throw std::runtime_error(std::string("Failed to send CCL buffer: ") + ncclGetErrorString(send_result) +
                                     " (cuda=" + cudaGetErrorString(cuda_result) +
                                     ", async=" + ncclGetErrorString(async_result) + ")");
        }
    }

    void RecvBufferAsync(Communicator &comm, const int peer_rank, const int device_ordinal, void *device_buffer,
                         const size_t num_bytes, CclGpuStream stream_handle)
    {
        if (num_bytes == 0)
        {
            return;
        }
        pi::tensorlib::internal::ctxmgmt::GpuSetCurrentDevice(device_ordinal);
        const cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);
        const auto recv_result =
            ncclRecv(device_buffer, num_bytes, ncclChar, peer_rank, AsNccl(comm), stream);
        if (recv_result != ncclSuccess)
        {
            const auto cuda_result = cudaGetLastError();
            const auto async_result = [&comm]() {
                ncclResult_t async_error = ncclSuccess;
                ncclCommGetAsyncError(AsNccl(comm), &async_error);
                return async_error;
            }();
            throw std::runtime_error(std::string("Failed to receive CCL buffer: ") + ncclGetErrorString(recv_result) +
                                     " (cuda=" + cudaGetErrorString(cuda_result) +
                                     ", async=" + ncclGetErrorString(async_result) + ")");
        }
    }

    void BroadcastAsync(Communicator &comm, const int root_rank, const int device_ordinal, void *device_buffer,
                        const size_t num_bytes, CclGpuStream stream_handle)
    {
        if (num_bytes == 0)
        {
            return;
        }
        pi::tensorlib::internal::ctxmgmt::GpuSetCurrentDevice(device_ordinal);
        const cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);
        const auto result = ncclBroadcast(device_buffer, device_buffer, num_bytes, ncclChar, root_rank, AsNccl(comm),
                                          stream);
        if (result != ncclSuccess)
        {
            const auto cuda_result = cudaGetLastError();
            const auto async_result = [&comm]() {
                ncclResult_t async_error = ncclSuccess;
                ncclCommGetAsyncError(AsNccl(comm), &async_error);
                return async_error;
            }();
            throw std::runtime_error(std::string("Failed to broadcast CCL buffer: ") + ncclGetErrorString(result) +
                                     " (cuda=" + cudaGetErrorString(cuda_result) +
                                     ", async=" + ncclGetErrorString(async_result) + ")");
        }
    }

    ncclDataType_t ToNcclDataType(const DataType data_type)
    {
        switch (data_type)
        {
            case DataType::FLOAT16:
                return ncclFloat16;
            case DataType::BFLOAT16:
                return ncclBfloat16;
            case DataType::FLOAT32:
                return ncclFloat32;
            case DataType::INT32:
                return ncclInt32;
            case DataType::UINT32:
                return ncclUint32;
        }
        throw std::runtime_error("Unsupported CCL data type.");
    }

    void AllReduceAsync(Communicator &comm, const int device_ordinal, const void *send_buffer, void *recv_buffer,
                        const size_t num_elements, const DataType data_type, CclGpuStream stream_handle)
    {
        if (num_elements == 0)
        {
            return;
        }
        pi::tensorlib::internal::ctxmgmt::GpuSetCurrentDevice(device_ordinal);
        const cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);
        const auto reduce_result =
            ncclAllReduce(send_buffer, recv_buffer, num_elements, ToNcclDataType(data_type), ncclSum, AsNccl(comm), stream);
        if (reduce_result != ncclSuccess)
        {
            const auto cuda_result = cudaGetLastError();
            const auto async_result = [&comm]() {
                ncclResult_t async_error = ncclSuccess;
                ncclCommGetAsyncError(AsNccl(comm), &async_error);
                return async_error;
            }();
            throw std::runtime_error(std::string("Failed to allreduce CCL buffer: ") +
                                     ncclGetErrorString(reduce_result) + " (cuda=" +
                                     cudaGetErrorString(cuda_result) + ", async=" +
                                     ncclGetErrorString(async_result) + ")");
        }
    }

    void AllGatherAsync(Communicator &comm, const int device_ordinal, const void *send_buffer, void *recv_buffer,
                        const size_t num_elements, const DataType data_type, CclGpuStream stream_handle)
    {
        if (num_elements == 0)
        {
            return;
        }
        pi::tensorlib::internal::ctxmgmt::GpuSetCurrentDevice(device_ordinal);
        const cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);
        const auto gather_result =
            ncclAllGather(send_buffer, recv_buffer, num_elements, ToNcclDataType(data_type), AsNccl(comm), stream);
        if (gather_result != ncclSuccess)
        {
            const auto cuda_result = cudaGetLastError();
            const auto async_result = [&comm]() {
                ncclResult_t async_error = ncclSuccess;
                ncclCommGetAsyncError(AsNccl(comm), &async_error);
                return async_error;
            }();
            throw std::runtime_error(std::string("Failed to allgather CCL buffer: ") +
                                     ncclGetErrorString(gather_result) + " (cuda=" +
                                     cudaGetErrorString(cuda_result) + ", async=" +
                                     ncclGetErrorString(async_result) + ")");
        }
    }
} // namespace fbamtrain::ccl
