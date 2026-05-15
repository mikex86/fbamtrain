#include "rdvz.h"

#include "ccl.h"
#include "config.h"
#include "logger.h"

#include <stdexcept>
#include <string>
#include <unordered_set>
#include <arpa/inet.h>
#include <ctime>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

namespace
{
    void SendAll(const int fd, const void *data, const size_t size)
    {
        const auto *buffer = static_cast<const unsigned char *>(data);
        size_t total = 0;
        while (total < size)
        {
            const auto sent = ::send(fd, buffer + total, size - total, 0);
            if (sent <= 0)
            {
                throw std::runtime_error("Failed to send data on rendezvous socket.");
            }
            total += static_cast<size_t>(sent);
        }
    }

    void RecvAll(const int fd, void *data, const size_t size)
    {
        auto *buffer = static_cast<unsigned char *>(data);
        size_t total = 0;
        while (total < size)
        {
            const auto received = ::recv(fd, buffer + total, size - total, 0);
            if (received <= 0)
            {
                throw std::runtime_error("Failed to receive data on rendezvous socket.");
            }
            total += static_cast<size_t>(received);
        }
    }

    [[nodiscard]] int CreateServerSocket(const std::string &ip, const uint16_t port)
    {
        const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0)
        {
            throw std::runtime_error("Failed to create rendezvous server socket.");
        }
        int opt = 1;
        if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) != 0)
        {
            ::close(fd);
            throw std::runtime_error("Failed to set rendezvous socket options.");
        }
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        if (::inet_pton(AF_INET, ip.c_str(), &addr.sin_addr) != 1)
        {
            ::close(fd);
            throw std::runtime_error("Invalid rendezvous IP address: " + ip);
        }
        if (::bind(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0)
        {
            ::close(fd);
            throw std::runtime_error("Failed to bind rendezvous server socket.");
        }
        if (::listen(fd, 16) != 0)
        {
            ::close(fd);
            throw std::runtime_error("Failed to listen on rendezvous server socket.");
        }
        return fd;
    }

    [[nodiscard]] int CreateClientSocket(const std::string &ip, const uint16_t port)
    {
        constexpr int kMaxAttempts = 50;
        constexpr int kSleepMillis = 100;
        for (int attempt = 1; attempt <= kMaxAttempts; ++attempt)
        {
            const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
            if (fd < 0)
            {
                throw std::runtime_error("Failed to create rendezvous client socket.");
            }
            sockaddr_in addr{};
            addr.sin_family = AF_INET;
            addr.sin_port = htons(port);
            if (::inet_pton(AF_INET, ip.c_str(), &addr.sin_addr) != 1)
            {
                ::close(fd);
                throw std::runtime_error("Invalid rendezvous IP address: " + ip);
            }
            if (::connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0)
            {
                return fd;
            }
            ::close(fd);
            if (attempt == kMaxAttempts)
            {
                break;
            }
            timespec ts{};
            ts.tv_sec = 0;
            ts.tv_nsec = static_cast<long>(kSleepMillis) * 1000 * 1000;
            nanosleep(&ts, nullptr);
        }
        throw std::runtime_error("Failed to connect to rendezvous server after retries.");
    }

    fbamtrain::ccl::Communicator PerformCclHandshake(const fbamtrain::ParallelEndpointConfiguration &endpoint,
                                                     const uint32_t rank, const uint32_t world_size,
                                                     const int device_ordinal,
                                                     const std::string &action_label)
    {
        if (world_size < 2)
        {
            throw std::runtime_error("Parallel CCL world size must be at least 2.");
        }
        if (rank >= world_size)
        {
            throw std::runtime_error("Parallel CCL rank must be in [0, world_size).");
        }

        const int ccl_world_size = static_cast<int>(world_size);
        constexpr int master_rank = 0;

        fbamtrain::ccl::UniqueId unique_id{};

        if (rank == master_rank)
        {
            LOG(INFO) << "Starting " << action_label << " as rank 0 on " << endpoint.ip << ":" << endpoint.port;
            unique_id = fbamtrain::ccl::GenerateUniqueId();

            const int server_fd =
                CreateServerSocket(endpoint.ip, static_cast<uint16_t>(endpoint.port));
            std::unordered_set<uint32_t> connected_rank_set{};
            const size_t expected_connections = world_size - 1;
            for (size_t count = 0; count < expected_connections; ++count)
            {
                LOG(INFO) << action_label << ": waiting for rank connection (" << (count + 1) << "/"
                          << expected_connections << ")";
                const int client_fd = ::accept(server_fd, nullptr, nullptr);
                if (client_fd < 0)
                {
                    ::close(server_fd);
                    throw std::runtime_error("Failed to accept rendezvous connection.");
                }
                uint32_t rank_wire = 0;
                RecvAll(client_fd, &rank_wire, sizeof(rank_wire));
                LOG(INFO) << action_label << ": received rank " << rank_wire;

                if (rank_wire == 0 || rank_wire >= world_size)
                {
                    ::close(client_fd);
                    ::close(server_fd);
                    throw std::runtime_error("Rendezvous rank connected outside [1, world_size): " +
                                             std::to_string(rank_wire));
                }
                if (!connected_rank_set.insert(rank_wire).second)
                {
                    ::close(client_fd);
                    ::close(server_fd);
                    throw std::runtime_error("Duplicate rank connected to rendezvous server: " +
                                             std::to_string(rank_wire));
                }

                SendAll(client_fd, unique_id.bytes.data(), unique_id.bytes.size());
                LOG(INFO) << action_label << ": sent unique ID to rank " << rank_wire;
                ::close(client_fd);
            }
            ::close(server_fd);

            LOG(INFO) << action_label << ": initializing communicator for rank " << master_rank
                      << " (world_size=" << ccl_world_size << ")";
            auto comm = fbamtrain::ccl::InitCommunicator(unique_id, ccl_world_size, master_rank, device_ordinal);
            LOG(INFO) << action_label << ": entering NCCL barrier on rank 0";
            fbamtrain::ccl::Barrier(comm, device_ordinal);
            LOG(INFO) << action_label << " completed as rank 0.";
            return comm;
        }

        LOG(INFO) << action_label << ": connecting to rank 0 at " << endpoint.ip << ":" << endpoint.port;
        const int client_fd = CreateClientSocket(endpoint.ip, static_cast<uint16_t>(endpoint.port));
        const auto rank_wire = rank;
        SendAll(client_fd, &rank_wire, sizeof(rank_wire));
        LOG(INFO) << action_label << ": sent rank " << rank_wire;
        RecvAll(client_fd, unique_id.bytes.data(), unique_id.bytes.size());
        LOG(INFO) << action_label << ": received unique ID";
        ::close(client_fd);

        LOG(INFO) << action_label << ": initializing communicator for rank " << rank << " (world_size=" << ccl_world_size
                  << ")";
        auto comm = fbamtrain::ccl::InitCommunicator(unique_id, ccl_world_size, static_cast<int>(rank), device_ordinal);
        LOG(INFO) << action_label << ": entering NCCL barrier on rank " << rank;
        fbamtrain::ccl::Barrier(comm, device_ordinal);
        LOG(INFO) << action_label << " completed as rank " << rank << ".";
        return comm;
    }
    std::optional<fbamtrain::ccl::Communicator> g_comm{};
} // namespace

void fbamtrain::rdvz::PerformRendezvous(const ParallelConfiguration &config, const bool is_master,
                                        const std::optional<uint32_t> &worker_id, const uint32_t world_size)
{
    if (g_comm.has_value())
    {
        LOG(WARN) << "Rendezvous already performed; skipping.";
        return;
    }
    if (!config.frame_head_parallel.use_frame_head_parallel)
    {
        throw std::runtime_error("Frame-head parallel rendezvous requested, but frame_head_parallel is disabled.");
    }
    if (config.frame_head_parallel.transport != "ccl")
    {
        throw std::runtime_error("Frame-head recurrence rendezvous requires transport \"ccl\".");
    }
    if (is_master && worker_id.has_value())
    {
        throw std::runtime_error("Master role should not set worker_id.");
    }
    if (!is_master && !worker_id.has_value())
    {
        throw std::runtime_error("Worker role specified without worker_id.");
    }
    const uint32_t rank = is_master ? 0 : worker_id.value();
    const int device_ordinal = is_master ? 0 : static_cast<int>(worker_id.value());
    auto comm = PerformCclHandshake(config.frame_head_parallel.ccl_rendezvous, rank, world_size, device_ordinal,
                                    "Frame-head rendezvous");
    g_comm = comm;
}

void fbamtrain::rdvz::PerformCclRendezvous(const ParallelEndpointConfiguration &endpoint, const uint32_t rank,
                                           const uint32_t world_size, const int device_ordinal,
                                           const std::string &action_label)
{
    if (g_comm.has_value())
    {
        LOG(WARN) << "Rendezvous already performed; skipping.";
        return;
    }
    auto comm = PerformCclHandshake(endpoint, rank, world_size, device_ordinal, action_label);
    g_comm = comm;
}

void fbamtrain::rdvz::PerformTeardown(const ParallelConfiguration &config, const bool is_master,
                                      const std::optional<uint32_t> &worker_id)
{
    (void)config;
    (void)is_master;
    (void)worker_id;

    if (!g_comm.has_value())
    {
        LOG(WARN) << "Teardown requested, but no communicator is active.";
        return;
    }

    bool destroyed = false;
    try
    {
        fbamtrain::ccl::Destroy(g_comm.value());
        destroyed = true;
    }
    catch (const std::exception &ex)
    {
        LOG(WARN) << "Teardown destroy failed: " << ex.what() << ". Falling back to abort.";
    }

    if (!destroyed)
    {
        const bool aborted = fbamtrain::ccl::Abort(g_comm.value());
        if (!aborted)
        {
            LOG(WARN) << "Teardown abort reported failure.";
        }
    }

    g_comm.reset();
    LOG(INFO) << "Teardown completed.";
}

fbamtrain::ccl::Communicator &fbamtrain::rdvz::GetCommunicator()
{
    if (!g_comm.has_value())
    {
        throw std::runtime_error("CCL communicator is not initialized. Did you call PerformRendezvous?");
    }
    return g_comm.value();
}
