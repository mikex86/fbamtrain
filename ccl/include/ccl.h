#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace fbamtrain::ccl
{
    constexpr size_t kUniqueIdBytes = 128;
    using CclGpuStream = void *;

    enum class DataType
    {
        FLOAT16,
        BFLOAT16,
        FLOAT32,
        INT32,
        UINT32,
    };

    struct UniqueId
    {
        std::array<unsigned char, kUniqueIdBytes> bytes{};
    };

    struct Communicator
    {
        void *handle{};
        int rank{};
        int world_size{};
    };

    // Sentinel for ranks that participate in the parent split call but should not receive a subset communicator.
    constexpr int kSubsetNoColor = -1;

    UniqueId GenerateUniqueId();

    Communicator InitCommunicator(const UniqueId &unique_id, int world_size, int rank, int device_ordinal);

    // Derive an addressable subset communicator from a parent communicator. Ranks with the same color form one
    // subset; kSubsetNoColor excludes the calling rank from all child communicators.
    [[nodiscard]] std::optional<Communicator> CreateSubsetCommunicator(Communicator &parent, int color, int key,
                                                                       int device_ordinal);

    void Destroy(Communicator &comm);

    bool Abort(Communicator &comm) noexcept;

    void SendString(Communicator &comm, int peer_rank, int device_ordinal, const std::string &message);

    [[nodiscard]] std::string RecvString(Communicator &comm, int peer_rank, int device_ordinal);

    void Barrier(Communicator &comm, int device_ordinal);

    void SendBufferAsync(Communicator &comm, int peer_rank, int device_ordinal, const void *device_buffer,
                         size_t num_bytes, CclGpuStream stream);

    void RecvBufferAsync(Communicator &comm, int peer_rank, int device_ordinal, void *device_buffer,
                         size_t num_bytes, CclGpuStream stream);

    void BroadcastAsync(Communicator &comm, int root_rank, int device_ordinal, void *device_buffer, size_t num_bytes,
                        CclGpuStream stream);

    void AllReduceAsync(Communicator &comm, int device_ordinal, const void *send_buffer, void *recv_buffer,
                        size_t num_elements, DataType data_type, CclGpuStream stream);

    void AllGatherAsync(Communicator &comm, int device_ordinal, const void *send_buffer, void *recv_buffer,
                        size_t num_elements, DataType data_type, CclGpuStream stream);
    
} // namespace fbamtrain::ccl
