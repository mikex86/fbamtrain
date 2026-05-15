#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace fbamtrain
{
    struct ParallelEndpointConfiguration;
    struct ParallelConfiguration;
    namespace ccl
    {
        struct Communicator;
    }

    namespace rdvz
    {
        void PerformRendezvous(const ParallelConfiguration &config, bool is_master,
                               const std::optional<uint32_t> &worker_id, uint32_t world_size);
        void PerformCclRendezvous(const ParallelEndpointConfiguration &endpoint, uint32_t rank, uint32_t world_size,
                                  int device_ordinal, const std::string &action_label);

        void PerformTeardown(const ParallelConfiguration &config, bool is_master,
                             const std::optional<uint32_t> &worker_id);

        [[nodiscard]] ccl::Communicator &GetCommunicator();
    }
} // namespace fbamtrain
