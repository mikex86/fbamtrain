#include "ccl.h"

#include <stdexcept>
#include <string>

namespace fbamtrain::ccl
{
    namespace
    {
        std::runtime_error MakeDisabledError(const char *function)
        {
            return std::runtime_error(std::string("CCL/NCCL is disabled because CUDA is not enabled: ") + function);
        }
    }

    UniqueId GenerateUniqueId()
    {
        throw MakeDisabledError("GenerateUniqueId");
    }

    Communicator InitCommunicator(const UniqueId &, const int, const int, const int)
    {
        throw MakeDisabledError("InitCommunicator");
    }

    std::optional<Communicator> CreateSubsetCommunicator(Communicator &, const int, const int, const int)
    {
        throw MakeDisabledError("CreateSubsetCommunicator");
    }

    void Destroy(Communicator &comm)
    {
        comm.handle = nullptr;
    }

    bool Abort(Communicator &comm) noexcept
    {
        comm.handle = nullptr;
        return true;
    }

    void SendString(Communicator &, const int, const int, const std::string &)
    {
        throw MakeDisabledError("SendString");
    }

    std::string RecvString(Communicator &, const int, const int)
    {
        throw MakeDisabledError("RecvString");
    }

    void Barrier(Communicator &, const int)
    {
        throw MakeDisabledError("Barrier");
    }

    void SendBufferAsync(Communicator &, const int, const int, const void *, const size_t, CclGpuStream)
    {
        throw MakeDisabledError("SendBufferAsync");
    }

    void RecvBufferAsync(Communicator &, const int, const int, void *, const size_t, CclGpuStream)
    {
        throw MakeDisabledError("RecvBufferAsync");
    }

    void BroadcastAsync(Communicator &, const int, const int, void *, const size_t, CclGpuStream)
    {
        throw MakeDisabledError("BroadcastAsync");
    }

    void AllReduceAsync(Communicator &, const int, const void *, void *, const size_t, const DataType, CclGpuStream)
    {
        throw MakeDisabledError("AllReduceAsync");
    }

    void AllGatherAsync(Communicator &, const int, const void *, void *, const size_t, const DataType, CclGpuStream)
    {
        throw MakeDisabledError("AllGatherAsync");
    }
} // namespace fbamtrain::ccl
