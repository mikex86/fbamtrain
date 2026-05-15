#include "async_step_logger.h"
#include "ctx_management.h"
#include "dataset_iterator.h"
#include "device_copy.h"

#include "ccl.h"
#include "checkpointing.h"
#include "dataset_cursor.h"
#include "format_utils.h"
#include "frame_utils.h"
#include "ftccl.h"
#include "functional.h"
#include "loss_accumulation_ring_buffer.h"
#include "main_action_model.h"
#include "optimizers.h"
#include "passes.h"
#include "rdvz.h"
#include "tokenizer.h"

#include <allocator.h>
#include <executor.h>
#include <framehead_model.h>
#include <gputx.h>
#include <logger.h>
#include <op_graph.h>
#include <safe_tensors.h>
#include <tensorlib.h>
#include <utils.h>

#include <cmath>

#include <algorithm>
#include <any>
#include <argparse/argparse.hpp>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <codecvt>
#include <config.h>
#include <cstdlib>
#include <functional>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>

#define PROJECT_VERSION "1.0.0"

static void run(const fbamtrain::RunConfiguration &run_config, bool validation_dump_mode,
                const std::string &validation_dump_file, uint64_t validation_dump_steps,
                const std::optional<fbamtrain::ParallelConfiguration> &parallel_config, bool is_master,
                const std::optional<uint32_t> &worker_id, const std::optional<uint32_t> &world_size,
                const std::optional<uint32_t> &ddp_world_size);

struct CheckpointLoadOutcome
{
    bool loaded{false};
    uint64_t step{0};
};

struct FtCclState
{
    fbamtrain::ftccl::Communicator communicator{};
    uint32_t expected_world_size{};
    uint64_t shared_state_revision{};
};

struct FtCclSharedStateView
{
    std::map<std::string, std::shared_ptr<pi::tensorlib::RealTensor>> tensors{};
    std::vector<fbamtrain::ftccl::TensorInfo> infos{};
};

struct FtCclSharedStateSyncOutcome
{
    uint64_t tx_bytes{};
    uint64_t rx_bytes{};
};

static CheckpointLoadOutcome LoadLatestCheckpointIfAvailable(
    const fbamtrain::RunConfiguration &run_config,
    const fbamtrain::checkpointing::CheckpointManager &checkpoint_manager,
    std::unordered_map<std::string, std::shared_ptr<pi::tensorlib::RealTensor>> &optimizer_params,
    fbamtrain::optim::Optimizer &frame_head_optimizer, fbamtrain::optim::Optimizer *action_model_optimizer,
    fbamtrain::RecordingDatasetIterator &active_iterator,
    const pi::tensorlib::GpuStreamDescriptor &compute_stream_descriptor)
{
    const auto latest_checkpoint = checkpoint_manager.findLatestCheckpoint();
    if (!latest_checkpoint.has_value())
    {
        return {};
    }

    LOG(INFO) << "Loading checkpoint from " << latest_checkpoint->path;
    auto loaded = checkpoint_manager.load(latest_checkpoint->path, run_config.micro_batch_size);
    const uint64_t step_count = loaded.step_count;

    for (const auto &[name, tensor] : optimizer_params)
    {
        if (!tensor)
        {
            throw std::runtime_error("Checkpoint target tensor is null for param: " + name);
        }
        const std::string key = "param/" + name;
        const auto it = loaded.tensors.find(key);
        if (it == loaded.tensors.end() || !it->second)
        {
            throw std::runtime_error("Missing checkpoint tensor: " + key);
        }
        const auto &source = it->second;
        if (source->dtype() != tensor->dtype())
        {
            throw std::runtime_error("Checkpoint dtype mismatch for '" + key + "': expected " +
                                     pi::tensorlib::GetDataTypeName(tensor->dtype()) + ", got " +
                                     pi::tensorlib::GetDataTypeName(source->dtype()));
        }
        if (source->shape() != tensor->shape())
        {
            throw std::runtime_error("Checkpoint shape mismatch for '" + key + "'.");
        }
        tensor->storage()->copyFrom(*source->storage(), compute_stream_descriptor);
    }

    frame_head_optimizer.loadState(loaded.tensors, "optim/frame_head");
    if (action_model_optimizer)
    {
        action_model_optimizer->loadState(loaded.tensors, "optim/action_model");
    }

    frame_head_optimizer.setStep(step_count);
    if (action_model_optimizer)
    {
        action_model_optimizer->setStep(step_count);
    }

    active_iterator.restore(loaded.dataset_state);

    for (const auto &[name, tensor] : loaded.tensors)
    {
        if (tensor)
        {
            tensor->free();
        }
    }
    loaded.tensors.clear();

    LOG(INFO) << "Resuming from checkpoint at step_count=" << step_count;
    return CheckpointLoadOutcome{.loaded = true, .step = step_count};
}

static void ApplyDefaultPasses(pi::tensorlib::ExecutionPlan &execution_plan)
{
    std::vector<std::unique_ptr<pi::tensorlib::passes::CompilerPass>> passes{};
    passes.emplace_back(std::make_unique<MatmulFusePass>());
    passes.emplace_back(std::make_unique<MatmulImplPass>());
    passes.emplace_back(std::make_unique<FillZerosImplPass>());
    passes.emplace_back(std::make_unique<FillConstantImplPass>());
    passes.emplace_back(std::make_unique<FillUniformImplPass>());
    passes.emplace_back(std::make_unique<FillNormalImplPass>());
    passes.emplace_back(std::make_unique<LayerNormImplPass>());
    passes.emplace_back(std::make_unique<RmsNormImplPass>());
    passes.emplace_back(std::make_unique<AvgPool1dImplPass>());
    passes.emplace_back(std::make_unique<AvgPool2dImplPass>());
    passes.emplace_back(std::make_unique<AvgPool2dBwdImplPass>());
    passes.emplace_back(std::make_unique<Conv2dImplPass>());
    passes.emplace_back(std::make_unique<CastImplPass>());
    passes.emplace_back(std::make_unique<MeanImplPass>());
    passes.emplace_back(std::make_unique<FuseMulReducePass>());
    passes.emplace_back(std::make_unique<ReduceSumImplPass>());
    passes.emplace_back(std::make_unique<CrossEntropyOnTargetsImplPass>());
    passes.emplace_back(std::make_unique<GatherImplPass>());
    passes.emplace_back(std::make_unique<DivAddFusePass>());
    passes.emplace_back(std::make_unique<TrailingBroadcastAddImplPass>());
    passes.emplace_back(std::make_unique<TrailingBroadcastMulImplPass>());
    passes.emplace_back(std::make_unique<DivImplPass>());
    passes.emplace_back(std::make_unique<ActImplPass>());
    passes.emplace_back(std::make_unique<LstmCellImplPass>());
    passes.emplace_back(std::make_unique<LstmCellBwdImplPass>());
    passes.emplace_back(std::make_unique<MhaAttentionImplPass>());
    passes.emplace_back(std::make_unique<BuildCellEmbedPass>());
    passes.emplace_back(std::make_unique<BuildCellEmbedBwdPass>());
    passes.emplace_back(std::make_unique<OptimizerImplPass>());
    passes.emplace_back(std::make_unique<ContiguousImplPass>());
    passes.emplace_back(std::make_unique<ReduceSumPartialImplPass>());
    pi::tensorlib::passes::Transform(execution_plan, passes);
}

static std::unordered_map<uint32_t, int> BuildWorkerIdToRankMap(const uint32_t world_size)
{
    std::unordered_map<uint32_t, int> worker_id_to_rank{};
    worker_id_to_rank.reserve(world_size);
    for (uint32_t rank = 0; rank < world_size; ++rank)
    {
        worker_id_to_rank.emplace(rank, static_cast<int>(rank));
    }
    return worker_id_to_rank;
}

static void ValidateFrameParallelWorldSize(const uint32_t world_size)
{
    if (world_size < 2)
    {
        throw std::runtime_error("Frame parallel world size must be at least 2.");
    }
}

static void CheckFtCclResult(const fbamtrain::ftccl::Result result, const std::string_view action)
{
    if (result != fbamtrain::ftccl::Result::Success)
    {
        throw std::runtime_error(std::string(action) + " failed: " + fbamtrain::ftccl::ResultString(result));
    }
}

static fbamtrain::ftccl::DataType ToFtCclDataType(const pi::tensorlib::DataType dtype)
{
    switch (dtype)
    {
        case pi::tensorlib::DataType::FLOAT16:
            return fbamtrain::ftccl::DataType::FLOAT16;
        case pi::tensorlib::DataType::BFLOAT16:
            return fbamtrain::ftccl::DataType::BFLOAT16;
        case pi::tensorlib::DataType::FLOAT32:
            return fbamtrain::ftccl::DataType::FLOAT32;
        case pi::tensorlib::DataType::UINT32:
            return fbamtrain::ftccl::DataType::UINT32;
        case pi::tensorlib::DataType::UINT64:
            return fbamtrain::ftccl::DataType::UINT64;
    }
    throw std::runtime_error("Unsupported data type for FTCCL shared state.");
}

static fbamtrain::ftccl::DeviceType ToFtCclDeviceType(const pi::tensorlib::DeviceType device_type)
{
    switch (device_type)
    {
        case pi::tensorlib::DeviceType::CPU:
            return fbamtrain::ftccl::DeviceType::CPU;
        case pi::tensorlib::DeviceType::GPU:
            return fbamtrain::ftccl::DeviceType::CUDA;
    }
    throw std::runtime_error("Unsupported device type for FTCCL shared state.");
}

static int GetFtCclPeerGroupWorldSize(const FtCclState &ftccl_state)
{
    int peer_group_world_size = 0;
    CheckFtCclResult(
        fbamtrain::ftccl::GetAttribute(ftccl_state.communicator, fbamtrain::ftccl::Attribute::PeerGroupWorldSize,
                                       &peer_group_world_size),
        "FTCCL peer-group world-size query");
    return peer_group_world_size;
}

static bool PrepareFtCclCollectivePhase(FtCclState &ftccl_state, const bool allow_topology_update,
                                        const std::string_view action)
{
    int peer_group_world_size = GetFtCclPeerGroupWorldSize(ftccl_state);
    if (allow_topology_update || peer_group_world_size < static_cast<int>(ftccl_state.expected_world_size))
    {
        // All recurrence masters reach this helper at the same loop boundary. Avoiding additional update call-sites
        // keeps the FTCCL operation sequence uniform across participants.
        for (;;)
        {
            const auto result = fbamtrain::ftccl::UpdateTopology(ftccl_state.communicator);
            if (result == fbamtrain::ftccl::Result::Success)
            {
                break;
            }
            if (result != fbamtrain::ftccl::Result::UpdateTopologyFailed)
            {
                CheckFtCclResult(result, "FTCCL topology update");
            }
            LOG(WARN) << "FTCCL topology update failed during " << action << "; retrying.";
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        peer_group_world_size = GetFtCclPeerGroupWorldSize(ftccl_state);
    }

    if (peer_group_world_size > 1)
    {
        for (;;)
        {
            const auto result = fbamtrain::ftccl::OptimizeTopology(ftccl_state.communicator);
            if (result == fbamtrain::ftccl::Result::Success)
            {
                break;
            }
            if (result != fbamtrain::ftccl::Result::TopologyOptimizationFailed)
            {
                CheckFtCclResult(result, "FTCCL topology optimization");
            }
            LOG(WARN) << "FTCCL topology optimization failed during " << action << "; retrying.";
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        peer_group_world_size = GetFtCclPeerGroupWorldSize(ftccl_state);
    }

    if (peer_group_world_size < static_cast<int>(ftccl_state.expected_world_size))
    {
        LOG(WARN) << "FTCCL " << action << " is waiting for expected_world_size="
                  << ftccl_state.expected_world_size << "; current peer-group world_size=" << peer_group_world_size
                  << ".";
        std::this_thread::sleep_for(std::chrono::seconds(1));
        return false;
    }

    return true;
}

static FtCclState CreateFtCclState(const fbamtrain::ParallelEndpointConfiguration &endpoint,
                                   const uint32_t expected_world_size)
{
    if (expected_world_size < 2)
    {
        throw std::runtime_error("FTCCL expected world size must be at least 2.");
    }

    CheckFtCclResult(fbamtrain::ftccl::Init(), "FTCCL initialization");

    // Keep FTCCL socket-address parsing behind the FTCCL abstraction boundary; fbamtrain config only owns endpoint
    // strings.
    fbamtrain::ftccl::SocketAddress master_address{};
    CheckFtCclResult(fbamtrain::ftccl::ParseSocketAddress(endpoint.ip.c_str(), endpoint.port, &master_address),
                     "FTCCL master endpoint parsing");

    fbamtrain::ftccl::Communicator communicator{};
    fbamtrain::ftccl::CommCreateParams params{
        .master_address = master_address,
        .peer_group = 0,
        .p2p_connection_pool_size = 16,
    };
    CheckFtCclResult(fbamtrain::ftccl::CreateCommunicator(params, &communicator), "FTCCL communicator creation");
    CheckFtCclResult(fbamtrain::ftccl::Connect(communicator), "FTCCL communicator connection");

    return FtCclState{.communicator = communicator, .expected_world_size = expected_world_size};
}

static FtCclSharedStateView BuildFtCclSharedStateView(
    const std::unordered_map<std::string, std::shared_ptr<pi::tensorlib::RealTensor>> &optimizer_params,
    const fbamtrain::FrameHeadModule &frame_head_module, const fbamtrain::ActionModelModule *action_model_module,
    fbamtrain::optim::Optimizer &frame_head_optimizer, fbamtrain::optim::Optimizer *action_model_optimizer)
{
    FtCclSharedStateView view{};
    frame_head_module.appendParamState(view.tensors, optimizer_params, "FTCCL frame-head shared state");
    if (action_model_module != nullptr)
    {
        action_model_module->appendParamState(view.tensors, optimizer_params, "FTCCL action-model shared state");
    }

    // Optimizer internals are part of the deterministic training state; omitting them would let peers agree on
    // parameters while silently diverging in the next update.
    frame_head_optimizer.appendOptimState(view.tensors, "optim/frame_head");
    if (action_model_optimizer != nullptr)
    {
        action_model_optimizer->appendOptimState(view.tensors, "optim/action_model");
    }

    view.infos.reserve(view.tensors.size());
    for (const auto &[name, tensor] : view.tensors)
    {
        if (!tensor)
        {
            throw std::runtime_error("FTCCL shared state tensor is null: " + name);
        }
        view.infos.push_back(fbamtrain::ftccl::TensorInfo{
            .name = name.c_str(),
            .data = tensor->dataptr(),
            .count = static_cast<size_t>(tensor->shape().numel()),
            .datatype = ToFtCclDataType(tensor->dtype()),
            .device_type = ToFtCclDeviceType(tensor->device().device_type),
            .allow_content_inequality = false,
        });
    }

    return view;
}

static void AppendFrameParallelCclState(
    std::map<std::string, std::shared_ptr<pi::tensorlib::RealTensor>> &state_view,
    const std::unordered_map<std::string, std::shared_ptr<pi::tensorlib::RealTensor>> &optimizer_params,
    const fbamtrain::FrameHeadModule &frame_head_module,
    fbamtrain::optim::Optimizer &frame_head_optimizer)
{
    frame_head_module.appendParamState(state_view, optimizer_params, "Frame-parallel CCL state");
    // Pure frame workers own frame-head optimizer state, but not action-model optimizer state.
    frame_head_optimizer.appendOptimState(state_view, "optim/frame_head");
}

static FtCclSharedStateSyncOutcome SynchronizeFtCclSharedState(FtCclState &ftccl_state,
                                                               FtCclSharedStateView &shared_state_view,
                                                               const std::string_view action)
{
    if (shared_state_view.infos.empty())
    {
        throw std::runtime_error("FTCCL shared state cannot be empty.");
    }

    fbamtrain::ftccl::SharedState shared_state{
        .revision = ftccl_state.shared_state_revision,
        .count = shared_state_view.infos.size(),
        .infos = shared_state_view.infos.data(),
    };
    fbamtrain::ftccl::SharedStateSyncInfo sync_info{};
    CheckFtCclResult(fbamtrain::ftccl::SynchronizeSharedState(ftccl_state.communicator, shared_state,
                                                              fbamtrain::ftccl::SharedStateSyncStrategy::EnforcePopular,
                                                              &sync_info),
                     action);
    ftccl_state.shared_state_revision = shared_state.revision;

    LOG(INFO) << action << " completed at revision " << ftccl_state.shared_state_revision
              << " (tx_bytes=" << sync_info.tx_bytes << ", rx_bytes=" << sync_info.rx_bytes << ").";
    return FtCclSharedStateSyncOutcome{.tx_bytes = sync_info.tx_bytes, .rx_bytes = sync_info.rx_bytes};
}

static bool AllReduceFtCclActionModelGradients(FtCclState &ftccl_state,
                                               const std::shared_ptr<pi::tensorlib::RealTensor> &gradient_buffer,
                                               const size_t element_count,
                                               const pi::tensorlib::DataType dtype)
{
    if (!gradient_buffer || element_count == 0)
    {
        throw std::runtime_error("Action-model gradient buffer missing for FTCCL-backed DDP allreduce.");
    }

    fbamtrain::ftccl::ReduceOpDescriptor descriptor{
        .sendbuf = gradient_buffer->dataptr(),
        .recvbuf = gradient_buffer->dataptr(),
        .descriptor = fbamtrain::ftccl::ReduceDescriptor{
            .count = element_count,
            .op = fbamtrain::ftccl::RedOp::Sum,
            .tag = (ftccl_state.shared_state_revision << 8U) | 1U,
            .src_descriptor =
                fbamtrain::ftccl::ReduceOperandDescriptor{
                    .datatype = ToFtCclDataType(dtype),
                    .distribution_hint = fbamtrain::ftccl::DistributionHint::Normal,
                },
            .quantization_options =
                fbamtrain::ftccl::QuantizationOptions{
                    .quantized_datatype = ToFtCclDataType(dtype),
                    .algorithm = fbamtrain::ftccl::QuantizationAlgorithm::None,
                },
        },
    };

    fbamtrain::ftccl::ReduceInfo reduce_info{};
    const auto result = fbamtrain::ftccl::AllReduceMultipleWithRetry(ftccl_state.communicator, &descriptor, 1, 1,
                                                                     &reduce_info);
    if (result == fbamtrain::ftccl::Result::TooFewPeers)
    {
        LOG(WARN) << "FTCCL action-model gradient allreduce dropped below two peers.";
        return false;
    }
    CheckFtCclResult(result, "FTCCL action-model gradient allreduce");
    const int peer_group_world_size = GetFtCclPeerGroupWorldSize(ftccl_state);
    if (peer_group_world_size < static_cast<int>(ftccl_state.expected_world_size))
    {
        LOG(WARN) << "FTCCL action-model gradient allreduce completed below expected_world_size="
                  << ftccl_state.expected_world_size << " (peer-group world_size=" << peer_group_world_size
                  << "); dropping step.";
        return false;
    }
    LOG(INFO) << "FTCCL action-model gradient allreduce completed at revision "
              << ftccl_state.shared_state_revision << " (local_world_size=" << reduce_info.local_world_size
              << ", tx_bytes=" << reduce_info.tx_bytes << ", rx_bytes=" << reduce_info.rx_bytes << ").";
    return true;
}

static void BroadcastCclTensorState(
    const std::map<std::string, std::shared_ptr<pi::tensorlib::RealTensor>> &state_view,
    fbamtrain::ccl::Communicator *comm, const int device_ordinal, const int root_rank,
    pi::tensorlib::gpustream::GpuStream main_stream, pi::tensorlib::gpustream::GpuStream ccl_stream,
    const std::string_view action)
{
    if (!comm || state_view.empty())
    {
        return;
    }

    GPUTX_RANGE("fbamtrain::ccl_state_broadcast");
    const pi::tensorlib::Device device_gpu{.device_type = pi::tensorlib::DeviceType::GPU, .ordinal = device_ordinal};
    for (const auto &[name, tensor] : state_view)
    {
        if (!tensor)
        {
            throw std::runtime_error("Tensor is null during " + std::string(action) + ": " + name);
        }

        const size_t num_bytes = tensor->shape().numel() * pi::tensorlib::GetDataTypeSize(tensor->dtype());
        if (tensor->device().device_type == pi::tensorlib::DeviceType::GPU)
        {
            pi::tensorlib::ExecutionBackend::GpuStreamWaitFor(main_stream, ccl_stream);
            fbamtrain::ccl::BroadcastAsync(*comm, root_rank, device_ordinal, tensor->dataptr(), num_bytes, ccl_stream);
            continue;
        }

        if (tensor->device().device_type != pi::tensorlib::DeviceType::CPU)
        {
            throw std::runtime_error("Unsupported tensor device during " + std::string(action) + ": " + name);
        }

        auto staging = pi::tensorlib::RealTensor::Allocate(tensor->shape().dims(), tensor->dtype(), device_gpu);
        if (comm->rank == root_rank)
        {
            pi::tensorlib::internal::device_copy::PerformDeviceCopy(tensor, staging, main_stream);
        }
        pi::tensorlib::ExecutionBackend::GpuStreamWaitFor(main_stream, ccl_stream);
        fbamtrain::ccl::BroadcastAsync(*comm, root_rank, device_ordinal, staging->dataptr(), num_bytes, ccl_stream);
        pi::tensorlib::ExecutionBackend::GpuStreamWaitFor(ccl_stream, main_stream);
        pi::tensorlib::internal::device_copy::PerformDeviceCopy(staging, tensor, main_stream);
        pi::tensorlib::ExecutionBackend::SynchronizeGpuStream(main_stream, device_ordinal);
        staging->free();
    }
    pi::tensorlib::ExecutionBackend::GpuStreamWaitFor(ccl_stream, main_stream);
    LOG(INFO) << action << " broadcast " << state_view.size() << " tensors.";
}

static void BroadcastParameters(const std::vector<pi::tensorlib::GraphExecutionInputDescriptor> &parameters,
                                fbamtrain::ccl::Communicator *comm, int device_ordinal, int root_rank,
                                fbamtrain::ccl::CclGpuStream stream)
{
    if (!comm || parameters.empty())
    {
        return;
    }
    GPUTX_RANGE("fbamtrain::ccl_broadcast");
    for (const auto &entry : parameters)
    {
        if (!entry.tensor)
        {
            throw std::runtime_error("Parameter tensor is null during broadcast.");
        }
        const size_t num_bytes = entry.tensor->shape().numel() * pi::tensorlib::GetDataTypeSize(entry.tensor->dtype());
        fbamtrain::ccl::BroadcastAsync(*comm, root_rank, device_ordinal, entry.tensor->dataptr(), num_bytes, stream);
    }
}

static bool BroadcastCclStepCommitDecision(fbamtrain::ccl::Communicator &comm, const int device_ordinal,
                                           const int root_rank, const bool root_should_commit,
                                           const std::shared_ptr<pi::tensorlib::RealTensor> &signal_cpu,
                                           const std::shared_ptr<pi::tensorlib::RealTensor> &signal_gpu,
                                           pi::tensorlib::gpustream::GpuStream main_stream,
                                           pi::tensorlib::gpustream::GpuStream ccl_stream)
{
    if (!signal_cpu || !signal_gpu)
    {
        throw std::runtime_error("Missing CCL step commit signal tensors.");
    }
    if (signal_cpu->shape().numel() != 1 || signal_gpu->shape().numel() != 1 ||
        signal_cpu->dtype() != pi::tensorlib::DataType::UINT32 ||
        signal_gpu->dtype() != pi::tensorlib::DataType::UINT32)
    {
        throw std::runtime_error("CCL step commit signal tensors must be uint32 scalars.");
    }

    if (comm.rank == root_rank)
    {
        *static_cast<uint32_t *>(signal_cpu->dataptr()) = root_should_commit ? 1U : 0U;
        pi::tensorlib::internal::device_copy::PerformDeviceCopy(signal_cpu, signal_gpu, main_stream);
    }

    // Pure frame workers do not participate in FTCCL, so the recurrence master broadcasts whether the just-computed
    // gradients are allowed to advance optimizer state.
    pi::tensorlib::ExecutionBackend::GpuStreamWaitFor(main_stream, ccl_stream);
    fbamtrain::ccl::BroadcastAsync(comm, root_rank, device_ordinal, signal_gpu->dataptr(), sizeof(uint32_t),
                                   ccl_stream);
    pi::tensorlib::ExecutionBackend::GpuStreamWaitFor(ccl_stream, main_stream);

    if (comm.rank != root_rank)
    {
        pi::tensorlib::internal::device_copy::PerformDeviceCopy(signal_gpu, signal_cpu, main_stream);
        pi::tensorlib::ExecutionBackend::SynchronizeGpuStream(main_stream, device_ordinal);
    }
    return *static_cast<const uint32_t *>(signal_cpu->dataptr()) != 0U;
}

static fbamtrain::ccl::DataType ToCclDataType(const pi::tensorlib::DataType dtype)
{
    switch (dtype)
    {
        case pi::tensorlib::DataType::FLOAT16:
            return fbamtrain::ccl::DataType::FLOAT16;
        case pi::tensorlib::DataType::BFLOAT16:
            return fbamtrain::ccl::DataType::BFLOAT16;
        case pi::tensorlib::DataType::FLOAT32:
            return fbamtrain::ccl::DataType::FLOAT32;
        case pi::tensorlib::DataType::UINT32:
            return fbamtrain::ccl::DataType::UINT32;
        case pi::tensorlib::DataType::UINT64:
            throw std::runtime_error("UINT64 is not supported for CCL allreduce.");
    }
    throw std::runtime_error("Unsupported data type for CCL allreduce.");
}

// returns true if a parameter should get a fp32 grad tensor despite not being fp32 itself.
// This is the actual whitelist for fp32-grad parameters for now...
static bool ParameterShouldHaveFp32Grad(const std::string_view parameter_name)
{
    auto ends_with = [parameter_name](const std::string_view suffix) -> bool
    {
        return parameter_name.size() >= suffix.size() &&
               parameter_name.substr(parameter_name.size() - suffix.size()) == suffix;
    };

    return ends_with(".codepoint_embedding") || ends_with(".position_embedding.weight") || ends_with(".fg_r_embed") ||
           ends_with(".fg_g_embed") || ends_with(".fg_b_embed") || ends_with(".bg_r_embed") ||
           ends_with(".bg_g_embed") || ends_with(".bg_b_embed");
}

static size_t RoundUpToMultiple(const size_t value, const size_t multiple)
{
    if (multiple == 0)
    {
        return value;
    }
    return ((value + multiple - 1) / multiple) * multiple;
}

struct FrameHeadGradInitResult
{
    pi::tensorlib::ExecutionPlan plan;
    std::shared_ptr<pi::tensorlib::RealTensor> allocation_space;
    pi::tensorlib::DataType dtype{};
    size_t allreduce_elements{};
    size_t allreduce_bytes{};
    size_t alignment_bytes{};
};

struct ContiguousGradAllreduceBuffer
{
    std::shared_ptr<pi::tensorlib::RealTensor> allocation_space;
    pi::tensorlib::DataType dtype{};
    size_t allreduce_elements{};
};

static void AllReduceCclFrameHeadGradients(
    fbamtrain::ccl::Communicator &comm, const int device_ordinal,
    const std::vector<ContiguousGradAllreduceBuffer> &gradient_buffers,
    pi::tensorlib::gpustream::GpuStream main_stream, fbamtrain::ccl::CclGpuStream ccl_stream)
{
    pi::tensorlib::ExecutionBackend::GpuStreamWaitFor(main_stream, ccl_stream);

    for (const auto &buffer : gradient_buffers)
    {
        if (!buffer.allocation_space || buffer.allreduce_elements == 0)
        {
            continue;
        }
        fbamtrain::ccl::AllReduceAsync(comm, device_ordinal, buffer.allocation_space->dataptr(),
                                       buffer.allocation_space->dataptr(), buffer.allreduce_elements,
                                       ToCclDataType(buffer.dtype), ccl_stream);
    }

    pi::tensorlib::ExecutionBackend::GpuStreamWaitFor(ccl_stream, main_stream);
}

static bool AllReduceFtCclFrameHeadGradients(
    FtCclState &ftccl_state, const std::vector<ContiguousGradAllreduceBuffer> &gradient_buffers)
{
    std::vector<fbamtrain::ftccl::ReduceOpDescriptor> descriptors{};
    descriptors.reserve(gradient_buffers.size());
    for (size_t idx = 0; idx < gradient_buffers.size(); ++idx)
    {
        const auto &buffer = gradient_buffers[idx];
        if (!buffer.allocation_space || buffer.allreduce_elements == 0)
        {
            continue;
        }

        descriptors.push_back(fbamtrain::ftccl::ReduceOpDescriptor{
            .sendbuf = buffer.allocation_space->dataptr(),
            .recvbuf = buffer.allocation_space->dataptr(),
            .descriptor =
                fbamtrain::ftccl::ReduceDescriptor{
                    .count = buffer.allreduce_elements,
                    .op = fbamtrain::ftccl::RedOp::Sum,
                    .tag = (ftccl_state.shared_state_revision << 8U) | (16U + idx),
                    .src_descriptor =
                        fbamtrain::ftccl::ReduceOperandDescriptor{
                            .datatype = ToFtCclDataType(buffer.dtype),
                            .distribution_hint = fbamtrain::ftccl::DistributionHint::Normal,
                        },
                    .quantization_options =
                        fbamtrain::ftccl::QuantizationOptions{
                            .quantized_datatype = ToFtCclDataType(buffer.dtype),
                            .algorithm = fbamtrain::ftccl::QuantizationAlgorithm::None,
                        },
                },
        });
    }

    if (descriptors.empty())
    {
        throw std::runtime_error("Frame-head gradient buffers missing for FTCCL-backed DDP allreduce.");
    }

    fbamtrain::ftccl::ReduceInfo reduce_info{};
    const auto result = fbamtrain::ftccl::AllReduceMultipleWithRetry(
        ftccl_state.communicator, descriptors.data(), descriptors.size(), std::max<size_t>(1, descriptors.size()),
        &reduce_info);
    if (result == fbamtrain::ftccl::Result::TooFewPeers)
    {
        LOG(WARN) << "FTCCL frame-head gradient allreduce dropped below two peers.";
        return false;
    }
    CheckFtCclResult(result, "FTCCL frame-head gradient allreduce");
    const int peer_group_world_size = GetFtCclPeerGroupWorldSize(ftccl_state);
    if (peer_group_world_size < static_cast<int>(ftccl_state.expected_world_size))
    {
        LOG(WARN) << "FTCCL frame-head gradient allreduce completed below expected_world_size="
                  << ftccl_state.expected_world_size << " (peer-group world_size=" << peer_group_world_size
                  << "); dropping step.";
        return false;
    }
    LOG(INFO) << "FTCCL frame-head gradient allreduce completed at revision " << ftccl_state.shared_state_revision
              << " (local_world_size=" << reduce_info.local_world_size << ", tx_bytes=" << reduce_info.tx_bytes
              << ", rx_bytes=" << reduce_info.rx_bytes << ").";
    return true;
}

struct FrameHeadUpstreamStaging
{
    size_t buffer_count{};
    std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> send_buffers{};
    std::vector<pi::tensorlib::GpuEvent> send_events{};
    std::vector<bool> send_event_recorded{};
    std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> recv_buffers{};
    std::vector<pi::tensorlib::GpuEvent> recv_events{};
    std::vector<bool> recv_event_recorded{};
};

// Initializes a set of gradients in a dedicated graph/executor so we can override allocation strategy.
// This allocates all grad tensors from a single contiguous GPU region via a bump allocator.
static FrameHeadGradInitResult InitializeContiguousGradients(
    pi::tensorlib::OpGraph &grad_init_graph, pi::tensorlib::ExecutionBackend &execution_backend,
    pi::tensorlib::allocator::AllocatorRegistry &allocator_registry, const pi::tensorlib::Device &device_gpu,
    const int device_ordinal, const std::vector<std::string> &grad_names,
    const std::unordered_map<std::string, pi::tensorlib::TraceTensor> &grad_traces,
    std::vector<std::shared_ptr<pi::tensorlib::allocator::LocalAllocatorRegistry>> &owned_local_allocator_registries)
{
    // Compute contiguous allocation requirements for gradient buffer.
    constexpr size_t kGradientBumpAlignmentBytes = 256;
    std::optional<pi::tensorlib::DataType> grad_dtype{};
    size_t alignment_bytes = kGradientBumpAlignmentBytes;
    size_t allocation_bytes = 0;
    for (const auto &name : grad_names)
    {
        const auto &trace_tensor = grad_traces.at(name);
        if (!grad_dtype.has_value())
        {
            grad_dtype = trace_tensor.dtype();
            alignment_bytes =
                RoundUpToMultiple(kGradientBumpAlignmentBytes, pi::tensorlib::GetDataTypeSize(grad_dtype.value()));
        }
        else if (trace_tensor.dtype() != grad_dtype.value())
        {
            throw std::runtime_error("Frame head gradients must share a single data type for contiguous allreduce.");
        }
        const size_t tensor_bytes = trace_tensor.shape().numel() * pi::tensorlib::GetDataTypeSize(trace_tensor.dtype());
        allocation_bytes += RoundUpToMultiple(tensor_bytes, alignment_bytes);
    }
    if (!grad_dtype.has_value())
    {
        throw std::runtime_error("Frame head gradients are empty; cannot allocate allreduce buffer.");
    }
    const size_t dtype_bytes = pi::tensorlib::GetDataTypeSize(grad_dtype.value());
    const size_t allreduce_elements = (allocation_bytes + dtype_bytes - 1) / dtype_bytes;
    allocation_bytes = allreduce_elements * dtype_bytes;

    // Backing storage for all gradients in this group.
    auto allocation_space = pi::tensorlib::RealTensor::Allocate({allreduce_elements}, grad_dtype.value(), device_gpu);
    auto bump_allocator = std::make_unique<pi::tensorlib::allocator::BumpAllocator>(
        allocation_space->dataptr(), allocation_bytes, device_ordinal, alignment_bytes);

    auto grad_allocator_registry = std::make_shared<pi::tensorlib::allocator::LocalAllocatorRegistry>();
    grad_allocator_registry->registerAllocator(pi::tensorlib::DeviceType::GPU, std::move(bump_allocator));
    grad_allocator_registry->registerAllocator(pi::tensorlib::DeviceType::CPU,
                                               std::make_unique<pi::tensorlib::allocator::CpuAllocator>());

    // Initialize grad tensors using the bump allocator; this wires their storage into one contiguous buffer.
    pi::tensorlib::ExecutionPlan plan = pi::tensorlib::ExecutionPlan::FromGraph(grad_init_graph, {}, {});
    ApplyDefaultPasses(plan);
    pi::tensorlib::Executor executor{plan, execution_backend, device_ordinal};
    executor.execute(*grad_allocator_registry);
    executor.await();

    // Keep this local allocator registry alive for the lifetime of plan-owned tensors.
    owned_local_allocator_registries.push_back(grad_allocator_registry);

    return FrameHeadGradInitResult{.plan = std::move(plan),
                                   .allocation_space = allocation_space,
                                   .dtype = grad_dtype.value(),
                                   .allreduce_elements = allreduce_elements,
                                   .allreduce_bytes = allocation_bytes,
                                   .alignment_bytes = alignment_bytes};
}

// Builds upstream gradient staging buffers for frame-head backward in frame-parallel mode.
// Master ranks allocate send buffers; worker ranks allocate receive buffers (including the primary input buffer).
// The buffers are sized so we can prefetch one upstream per worker without reusing an in-flight send/recv slot.
static FrameHeadUpstreamStaging
InitializeFrameHeadUpstreamStaging(const bool use_frame_parallel, const bool is_master, const int num_frame_workers,
                                   const uint32_t batch_size, const uint32_t n_embed,
                                   const pi::tensorlib::Device &device_gpu,
                                   const std::shared_ptr<pi::tensorlib::RealTensor> &primary_recv_buffer)
{
    FrameHeadUpstreamStaging staging{};
    if (!use_frame_parallel)
    {
        return staging;
    }

    staging.buffer_count = std::max<size_t>(2, static_cast<size_t>(num_frame_workers));
    if (is_master)
    {
        staging.send_buffers.reserve(staging.buffer_count);
        staging.send_events.reserve(staging.buffer_count);
        staging.send_event_recorded.assign(staging.buffer_count, false);
        for (size_t idx = 0; idx < staging.buffer_count; ++idx)
        {
            staging.send_buffers.push_back(pi::tensorlib::RealTensor::Allocate(
                {batch_size, n_embed}, pi::tensorlib::DataType::FLOAT32, device_gpu));
            staging.send_events.emplace_back(pi::tensorlib::ExecutionBackend::CreateEvent(device_gpu));
        }
    }
    else
    {
        staging.recv_buffers.reserve(staging.buffer_count);
        staging.recv_events.reserve(staging.buffer_count);
        staging.recv_event_recorded.assign(staging.buffer_count, false);
        staging.recv_buffers.push_back(primary_recv_buffer);
        staging.recv_events.emplace_back(pi::tensorlib::ExecutionBackend::CreateEvent(device_gpu));
        for (size_t idx = 1; idx < staging.buffer_count; ++idx)
        {
            staging.recv_buffers.push_back(pi::tensorlib::RealTensor::Allocate(
                {batch_size, n_embed}, pi::tensorlib::DataType::FLOAT32, device_gpu));
            staging.recv_events.emplace_back(pi::tensorlib::ExecutionBackend::CreateEvent(device_gpu));
        }
    }
    return staging;
}

static void SendUpstreamGradientForPos(const size_t pos, const int dst_rank, fbamtrain::ccl::Communicator *ccl_comm,
                                       const int device_ordinal, FrameHeadUpstreamStaging &frame_head_upstream,
                                       size_t &frame_head_upstream_send_sequence,
                                       const std::shared_ptr<pi::tensorlib::RealTensor> &grad_x_tensor,
                                       const size_t frame_head_upstream_bytes,
                                       const pi::tensorlib::gpustream::GpuStream &h2d_stream,
                                       const pi::tensorlib::gpustream::GpuStream &ccl_stream)
{
    // Master-side helper for frame-parallel backward: stage a single timestep's grad_x into a rotating
    // send buffer, then issue an async CCL send.
    if (!ccl_comm)
    {
        throw std::runtime_error("CCL communicator missing for frame head upstream send.");
    }
    if (frame_head_upstream.send_buffers.empty())
    {
        throw std::runtime_error("Frame head upstream send buffers are not initialized.");
    }
    if (!grad_x_tensor)
    {
        throw std::runtime_error("Missing grad_x tensor for frame head upstream send.");
    }

    const auto grad_x_step = grad_x_tensor->at(0, pos);
    const size_t send_buffer_index = frame_head_upstream_send_sequence % frame_head_upstream.send_buffers.size();
    ++frame_head_upstream_send_sequence;
    if (frame_head_upstream.send_event_recorded[send_buffer_index])
    {
        // Ensure the previous send using this buffer has completed before overwriting it.
        pi::tensorlib::ExecutionBackend::GpuStreamWaitForEvent(frame_head_upstream.send_events[send_buffer_index],
                                                               h2d_stream);
        frame_head_upstream.send_event_recorded[send_buffer_index] = false;
    }
    const auto &send_buffer = frame_head_upstream.send_buffers[send_buffer_index];
    // Stage the upstream on the H2D stream so it can overlap with main-stream compute.
    pi::tensorlib::internal::device_copy::PerformDeviceCopy(grad_x_step, send_buffer, h2d_stream);
    pi::tensorlib::ExecutionBackend::GpuStreamWaitFor(h2d_stream, ccl_stream);
    GPUTX_RANGE("fbamtrain::ccl_send_upstream");
    fbamtrain::ccl::SendBufferAsync(*ccl_comm, dst_rank, device_ordinal, send_buffer->dataptr(),
                                    frame_head_upstream_bytes, ccl_stream);
    frame_head_upstream.send_events[send_buffer_index].record(ccl_stream);
    frame_head_upstream.send_event_recorded[send_buffer_index] = true;
}

static bool ShouldEmitMetricsJson()
{
    static const bool enabled = []
    {
        const char *env = std::getenv("FBAMTRAIN_METRICS_JSON");
        return env != nullptr && env[0] != '\0';
    }();
    return enabled;
}

struct ValidationDumpTraceEntry
{
    std::string name;
    pi::tensorlib::TraceTensor tensor;
    bool required{false};
};

static void
CreateValidationDump(const pi::tensorlib::ExecutionPlan &execution_plan,
                     const std::vector<ValidationDumpTraceEntry> &trace_entries,
                     const std::unordered_map<std::string, std::shared_ptr<pi::tensorlib::RealTensor>> &extra_tensors,
                     const std::string &validation_dump_file)
{
    std::map<std::string, std::shared_ptr<pi::tensorlib::RealTensor>> dump_tensors{};

    for (const auto &entry : trace_entries)
    {
        if (dump_tensors.contains(entry.name))
        {
            throw std::runtime_error("Duplicate validation dump tensor name: " + entry.name);
        }
        const auto tensor_opt = execution_plan.getRealTensor(entry.tensor);
        if (!tensor_opt.has_value())
        {
            if (entry.required)
            {
                throw std::runtime_error("Required validation dump tensor not available: " + entry.name);
            }
            LOG(WARN) << "Validation tensor not available in execution plan: " << entry.name;
            continue;
        }
        dump_tensors.emplace(entry.name, tensor_opt.value());
    }

    for (const auto &[name, tensor] : extra_tensors)
    {
        if (dump_tensors.contains(name))
        {
            throw std::runtime_error("Duplicate validation dump tensor name: " + name);
        }
        if (tensor)
        {
            dump_tensors.emplace(name, tensor);
        }
    }

    pi::tensorlib::safetensors::SaveToFile(validation_dump_file, dump_tensors);
    LOG(INFO) << "Validation dump saved to " << validation_dump_file;
}

static std::string FormatValidationDumpPath(const std::string &base_path, const uint64_t step)
{
    const auto dot = base_path.rfind('.');
    if (dot == std::string::npos)
    {
        return base_path + ".step" + std::to_string(step);
    }
    return base_path.substr(0, dot) + ".step" + std::to_string(step) + base_path.substr(dot);
}

int main(const int argc, char *argv[])
{
    argparse::ArgumentParser program("fbamtrain");

    program.add_argument("-v", "--version")
        .action(
            [](const std::string &)
            {
                std::cout << "Version: " << PROJECT_VERSION << std::endl;
                exit(0);
            })
        .help("Show version information");

    program.add_argument("-c", "--config").required().help("Path to the training run configuration json file");
    program.add_argument("--validation-dump")
        .flag()
        .help("Run a single validation sequence and dump LSTM outputs instead of training");
    program.add_argument("--validation-dump-file")
        .default_value(std::string("validation_dump.safetensors"))
        .help("Path to write validation dump (safetensors)");
    program.add_argument("--validation-dump-steps")
        .default_value(std::string("1"))
        .help("Number of validation steps to run when --validation-dump is set (default: 1)");
    program.add_argument("--parallel-config")
        .default_value(std::string(""))
        .help("Path to the parallel run configuration json file");
    program.add_argument("--master").flag().help("Run as master process in parallel mode");
    program.add_argument("--worker_id")
        .default_value(std::string(""))
        .help("Worker id to use in parallel mode (unsigned integer)");
    program.add_argument("--world-size")
        .default_value(std::string(""))
        .help("Total CCL world size to use in parallel mode (master + workers)");
    program.add_argument("--ddp-world-size")
        .default_value(std::string(""))
        .help("Expected DDP recurrence-master world size for FTCCL-backed DDP");

    try
    {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error &err)
    {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    const auto config_path = program.get<std::string>("--config");
    const std::optional<fbamtrain::RunConfiguration> run_config =
        fbamtrain::RunConfiguration::FromJsonFile(config_path);
    if (!run_config.has_value())
    {
        LOG(ERR) << "Failed to load run configuration from " << config_path;
        return 1;
    }
    const bool validation_dump_mode = program.get<bool>("--validation-dump");
    const auto validation_dump_file = program.get<std::string>("--validation-dump-file");
    const auto validation_dump_steps =
        static_cast<uint64_t>(std::stoull(program.get<std::string>("--validation-dump-steps")));

    const auto parallel_config_path = program.get<std::string>("--parallel-config");
    std::optional<fbamtrain::ParallelConfiguration> parallel_config{};
    if (!parallel_config_path.empty())
    {
        parallel_config = fbamtrain::ParallelConfiguration::FromJsonFile(parallel_config_path);
        if (!parallel_config.has_value())
        {
            LOG(ERR) << "Failed to load parallel configuration from " << parallel_config_path;
            return 1;
        }
    }
    const bool is_master = program.get<bool>("--master");
    const auto worker_id_arg = program.get<std::string>("--worker_id");
    std::optional<uint32_t> worker_id{};
    if (!worker_id_arg.empty())
    {
        worker_id = static_cast<uint32_t>(std::stoul(worker_id_arg));
    }
    const auto world_size_arg = program.get<std::string>("--world-size");
    std::optional<uint32_t> world_size{};
    if (!world_size_arg.empty())
    {
        world_size = static_cast<uint32_t>(std::stoul(world_size_arg));
    }
    const auto ddp_world_size_arg = program.get<std::string>("--ddp-world-size");
    std::optional<uint32_t> ddp_world_size{};
    if (!ddp_world_size_arg.empty())
    {
        ddp_world_size = static_cast<uint32_t>(std::stoul(ddp_world_size_arg));
    }

    run(run_config.value(), validation_dump_mode, validation_dump_file, validation_dump_steps, parallel_config,
        is_master, worker_id, world_size, ddp_world_size);
    return 0;
}

struct StepGraphResult
{
    pi::tensorlib::OpGraph graph;
    pi::tensorlib::TraceTensor frame_head_output;
};

struct FrameHeadBackwardGraphResult
{
    pi::tensorlib::OpGraph graph;
    pi::tensorlib::TraceTensor upstream_grad_input;
};

static StepGraphResult BuildFrameEmbeddingStepGraph(uint32_t batch_size, uint32_t rows, uint32_t cols,
                                                    fbamtrain::FrameHeadModule &frame_head_module,
                                                    const pi::tensorlib::Device &device_gpu,
                                                    const pi::tensorlib::DataType output_dtype)
{
    pi::tensorlib::TraceTensor cell_states_gpu = pi::tensorlib::TraceTensor::Create(
        {batch_size, rows, cols, NUM_FRAME_CHANNELS}, pi::tensorlib::DataType::UINT32, device_gpu,
        pi::tensorlib::GpuStreamDescriptors::Main);
    cell_states_gpu.markRetained();

    std::vector<pi::tensorlib::GraphInputDescriptor> parameter_descriptors{};
    for (const auto &[name, tensor] : frame_head_module.parameters())
    {
        auto retained_tensor = tensor;
        retained_tensor.markRetained();
        parameter_descriptors.push_back(pi::tensorlib::GraphInputDescriptor{.name = name, .tensor = retained_tensor});
    }

    pi::tensorlib::OpGraph graph{
        {
            pi::tensorlib::GraphInputDescriptor{.name = "cell_states", .tensor = cell_states_gpu},
        },
        parameter_descriptors};

    pi::tensorlib::TraceTensor frame_head_output = frame_head_module.buildForward(graph, {cell_states_gpu}, false);
    if (frame_head_output.dtype() != output_dtype)
    {
        frame_head_output = frame_head_output.to(graph, output_dtype, pi::tensorlib::GpuStreamDescriptors::Main);
    }

    graph.finalize();

    return StepGraphResult{std::move(graph), frame_head_output};
}

static FrameHeadBackwardGraphResult BuildFrameHeadBackwardGraph(
    uint32_t batch_size, uint32_t rows, uint32_t cols, uint32_t n_embed, fbamtrain::FrameHeadModule &frame_head_module,
    const std::unordered_map<std::string, pi::tensorlib::TraceTensor> &frame_head_param_grads,
    const pi::tensorlib::Device &device_gpu, const pi::tensorlib::Device &device_cpu, const bool upstream_on_cpu)
{
    const auto &upstream_device = upstream_on_cpu ? device_cpu : device_gpu;
    const bool upstream_pinned = upstream_on_cpu;
    pi::tensorlib::TraceTensor upstream_grad_input =
        pi::tensorlib::TraceTensor::Create({batch_size, n_embed}, pi::tensorlib::DataType::FLOAT32, upstream_device,
                                           pi::tensorlib::GpuStreamDescriptors::Main, upstream_pinned);
    upstream_grad_input.markRetained();

    pi::tensorlib::TraceTensor cell_states_gpu = pi::tensorlib::TraceTensor::Create(
        {batch_size, rows, cols, NUM_FRAME_CHANNELS}, pi::tensorlib::DataType::UINT32, device_gpu,
        pi::tensorlib::GpuStreamDescriptors::Main);
    cell_states_gpu.markRetained();

    std::vector<pi::tensorlib::GraphInputDescriptor> input_descriptors{};
    input_descriptors.reserve(2);
    input_descriptors.push_back(pi::tensorlib::GraphInputDescriptor{.name = "cell_states", .tensor = cell_states_gpu});
    input_descriptors.push_back(
        pi::tensorlib::GraphInputDescriptor{.name = "frame_head_upstream", .tensor = upstream_grad_input});

    std::vector<pi::tensorlib::GraphInputDescriptor> parameter_descriptors{};
    for (const auto &[name, tensor] : frame_head_module.parameters())
    {
        parameter_descriptors.push_back(pi::tensorlib::GraphInputDescriptor{.name = name, .tensor = tensor});
    }

    for (const auto &[name, tensor] : frame_head_param_grads)
    {
        parameter_descriptors.push_back(pi::tensorlib::GraphInputDescriptor{.name = name + "_grad", .tensor = tensor});
    }

    pi::tensorlib::OpGraph graph{input_descriptors, parameter_descriptors};
    const auto frame_head_output = frame_head_module.buildForward(graph, {cell_states_gpu}, true);

    pi::tensorlib::TraceTensor upstream_grad_gpu_fp32 = upstream_grad_input;
    if (upstream_on_cpu)
    {
        upstream_grad_gpu_fp32 = graph.createTensor({batch_size, n_embed}, pi::tensorlib::DataType::FLOAT32, device_gpu,
                                                    pi::tensorlib::GpuStreamDescriptors::Main, false);
        pi::tensorlib::DeviceCopy(graph, upstream_grad_input, upstream_grad_gpu_fp32,
                                  pi::tensorlib::GpuStreamDescriptors::Main);
    }

    pi::tensorlib::TraceTensor upstream_grad_gpu =
        upstream_grad_gpu_fp32.to(graph, frame_head_output.dtype(), pi::tensorlib::GpuStreamDescriptors::Main);

    std::unordered_map<std::string, pi::tensorlib::TraceTensor> frame_head_operand_grads{};
    frame_head_module.buildBackward(graph, upstream_grad_gpu, frame_head_param_grads, frame_head_operand_grads);

    graph.finalize();
    return FrameHeadBackwardGraphResult{std::move(graph), upstream_grad_input};
}

static pi::tensorlib::ExecutionPlan
BuildContiguousGradZeroPlan(const std::shared_ptr<pi::tensorlib::RealTensor> &grad_allocation_space,
                            const std::string &input_name)
{
    if (!grad_allocation_space)
    {
        throw std::runtime_error("Contiguous grad allocation space is null for zeroing.");
    }

    auto grad_buffer_trace = pi::tensorlib::TraceTensor::Create(
        grad_allocation_space->shape().dims(), grad_allocation_space->dtype(), grad_allocation_space->device(),
        pi::tensorlib::GpuStreamDescriptors::Main, false);
    grad_buffer_trace.markRetained();

    pi::tensorlib::OpGraph graph{{pi::tensorlib::GraphInputDescriptor{.name = input_name, .tensor = grad_buffer_trace}},
                                 {}};
    pi::tensorlib::FillZeros(graph, grad_buffer_trace, pi::tensorlib::GpuStreamDescriptors::Main);
    graph.finalize();

    return pi::tensorlib::ExecutionPlan::FromGraph(
        graph, {pi::tensorlib::GraphExecutionInputDescriptor{.name = input_name, .tensor = grad_allocation_space}}, {});
}

static pi::tensorlib::ExecutionPlan
BuildContiguousGradZeroPlan(const std::vector<ContiguousGradAllreduceBuffer> &grad_buffers,
                            const std::string &input_name_prefix)
{
    if (grad_buffers.empty())
    {
        throw std::runtime_error("Contiguous grad buffer list is empty for zeroing.");
    }

    std::vector<pi::tensorlib::GraphInputDescriptor> input_descriptors{};
    input_descriptors.reserve(grad_buffers.size());
    std::vector<pi::tensorlib::GraphExecutionInputDescriptor> exec_inputs{};
    exec_inputs.reserve(grad_buffers.size());
    std::vector<pi::tensorlib::TraceTensor> traces{};
    traces.reserve(grad_buffers.size());

    for (size_t idx = 0; idx < grad_buffers.size(); ++idx)
    {
        const auto &buffer = grad_buffers[idx];
        if (!buffer.allocation_space)
        {
            throw std::runtime_error("Contiguous grad allocation space is null for zeroing.");
        }

        const std::string input_name = input_name_prefix + "_" + std::to_string(idx);
        auto grad_buffer_trace = pi::tensorlib::TraceTensor::Create(
            buffer.allocation_space->shape().dims(), buffer.allocation_space->dtype(),
            buffer.allocation_space->device(), pi::tensorlib::GpuStreamDescriptors::Main, false);
        grad_buffer_trace.markRetained();

        input_descriptors.push_back(
            pi::tensorlib::GraphInputDescriptor{.name = input_name, .tensor = grad_buffer_trace});
        exec_inputs.push_back(
            pi::tensorlib::GraphExecutionInputDescriptor{.name = input_name, .tensor = buffer.allocation_space});
        traces.push_back(grad_buffer_trace);
    }

    pi::tensorlib::OpGraph graph{input_descriptors, {}};
    for (auto &trace : traces)
    {
        pi::tensorlib::FillZeros(graph, trace, pi::tensorlib::GpuStreamDescriptors::Main);
    }
    graph.finalize();

    return pi::tensorlib::ExecutionPlan::FromGraph(graph, exec_inputs, {});
}

struct ActionModelStepGraphResult
{
    pi::tensorlib::OpGraph graph;
    pi::tensorlib::LstmForwardStreamingResult action_model_output;
    pi::tensorlib::TraceTensor loss_sum;
    pi::tensorlib::TraceTensor loss_mean;
    std::optional<pi::tensorlib::TraceTensor> projected_output;
    pi::tensorlib::TraceTensor frame_embeddings_host;
    pi::tensorlib::TraceTensor action_targets_host;
    pi::tensorlib::TraceTensor loss_denominator_host;
    std::optional<pi::tensorlib::TraceTensor> grad_x;
    std::optional<pi::tensorlib::TraceTensor> grad_w_ih;
    std::optional<pi::tensorlib::TraceTensor> grad_w_hh;
    std::optional<pi::tensorlib::TraceTensor> grad_b_ih;
    std::optional<pi::tensorlib::TraceTensor> grad_b_hh;
    std::optional<pi::tensorlib::TraceTensor> grad_h0;
    std::optional<pi::tensorlib::TraceTensor> grad_c0;
};

constexpr int CROSS_ENTROPY_STREAM_ID = 1;
constexpr int CCL_STREAM_ID = 2;
constexpr int LOSS_ACCUM_STREAM_ID = 3;
constexpr int LOSS_LOGGER_COPY_STREAM_ID = 4;
constexpr int CROSS_ENTROPY_STREAM_PRIORITY = 1000;

/**
 * Builds the action model step graph.
 * @param seq_len the sequence length
 * @param batch_size the batch size
 * @param n_embed the embedding size
 * @param vocab_size the vocabulary size
 * @param loss_scale scale applied to cross-entropy gradients
 * @param action_model_module the frame embedding module
 * @param action_model_grads the action model parameter gradients
 * @param capture_projected_output whether to capture the projected output logits. (used for validation dump)
 * @param device_gpu the gpu device
 * @param device_cpu the cpu device
 * @return the action model step graph and its output tensor & loss
 */
static ActionModelStepGraphResult
BuildActionModelStepGraph(const size_t seq_len, const size_t batch_size, const size_t n_embed, const size_t vocab_size,
                          float loss_scale, fbamtrain::ActionModelModule &action_model_module,
                          const std::unordered_map<std::string, pi::tensorlib::TraceTensor> &action_model_grads,
                          bool capture_projected_output, const pi::tensorlib::Device &device_gpu,
                          const pi::tensorlib::Device &device_cpu)
{
    const auto frame_embed_dtype = action_model_module.lstm_.ioDtype();
    pi::tensorlib::TraceTensor frame_embeddings_host = pi::tensorlib::TraceTensor::Create(
        {seq_len, batch_size, n_embed}, frame_embed_dtype, device_cpu, pi::tensorlib::GpuStreamDescriptors::Main, true);
    frame_embeddings_host.markRetained();

    pi::tensorlib::TraceTensor action_targets_host =
        pi::tensorlib::TraceTensor::Create({seq_len, batch_size}, pi::tensorlib::DataType::UINT32, device_cpu,
                                           pi::tensorlib::GpuStreamDescriptors::Main, true);
    action_targets_host.markRetained();

    pi::tensorlib::TraceTensor loss_denominator_host = pi::tensorlib::TraceTensor::Create(
        {1}, pi::tensorlib::DataType::FLOAT32, device_cpu, pi::tensorlib::GpuStreamDescriptors::Main, true);
    loss_denominator_host.markRetained();

    std::vector<pi::tensorlib::GraphInputDescriptor> parameter_descriptors{};
    for (const auto &parameter : action_model_module.parameters())
    {
        parameter_descriptors.push_back(
            pi::tensorlib::GraphInputDescriptor{.name = parameter.name, .tensor = parameter.tensor});
    }
    for (const auto &[name, tensor] : action_model_grads)
    {
        parameter_descriptors.push_back(pi::tensorlib::GraphInputDescriptor{.name = name + "_grad", .tensor = tensor});
    }

    pi::tensorlib::TraceTensor loss_mean = pi::tensorlib::TraceTensor::Create(
        {1}, pi::tensorlib::DataType::FLOAT32, device_gpu, pi::tensorlib::GpuStreamDescriptors::Main);
    loss_mean.markRetained();

    pi::tensorlib::TraceTensor loss_sum = pi::tensorlib::TraceTensor::Create(
        {1}, pi::tensorlib::DataType::FLOAT32, device_gpu, pi::tensorlib::GpuStreamDescriptors::Main);
    loss_sum.markRetained();

    pi::tensorlib::OpGraph graph{
        {// accept pre-allocated "loss" and "loss_sum" tensors
         pi::tensorlib::GraphInputDescriptor{.name = "loss_mean", .tensor = loss_mean},
         pi::tensorlib::GraphInputDescriptor{.name = "loss_sum", .tensor = loss_sum},

         // both the input frame embeddings and targets must be supplied as inputs to the graph
         pi::tensorlib::GraphInputDescriptor{.name = "frame_embeddings", .tensor = frame_embeddings_host},
         pi::tensorlib::GraphInputDescriptor{.name = "action_targets", .tensor = action_targets_host},
         pi::tensorlib::GraphInputDescriptor{.name = "loss_denominator", .tensor = loss_denominator_host}},
        parameter_descriptors};

    auto &lstm = action_model_module.lstm_;

    // TODO: consider pipeline parallelism, where this is passed between workers
    const auto h0 = lstm.h0;
    const auto c0 = lstm.c0;

    const auto streaming_chunk = lstm.streamingChunkSize();

    const auto &head_params = action_model_module.head().parameters();
    if (head_params.empty())
    {
        throw std::runtime_error("Action model head has no parameters");
    }
    const auto head_weight = head_params[0].tensor;
    const auto head_weight_dtype = head_weight.dtype();

    constexpr auto ce_stream_desc =
        pi::tensorlib::GpuStreamDescriptor{pi::tensorlib::StreamKind::Compute, CROSS_ENTROPY_STREAM_ID};

    // Targets staging for CE hook (double-buffered on GPU).
    std::array targets_chunk_gpu{graph.createTensor({streaming_chunk, batch_size}, pi::tensorlib::DataType::UINT32,
                                                    device_gpu, ce_stream_desc, false),
                                 graph.createTensor({streaming_chunk, batch_size}, pi::tensorlib::DataType::UINT32,
                                                    device_gpu, ce_stream_desc, false)};
    pi::tensorlib::FillZeros(graph, loss_sum, ce_stream_desc);

    pi::tensorlib::TraceTensor loss_denominator_gpu =
        graph.createTensor({1}, pi::tensorlib::DataType::FLOAT32, device_gpu, ce_stream_desc, false);
    pi::tensorlib::DeviceCopy(graph, loss_denominator_host, loss_denominator_gpu, ce_stream_desc);

    std::optional<pi::tensorlib::TraceTensor> projected_output{};
    if (capture_projected_output)
    {
        projected_output =
            graph.createTensor({seq_len, batch_size, vocab_size}, head_weight_dtype, device_cpu, ce_stream_desc, true);
        projected_output->markRetained();
    }

    struct TargetPrefetchState
    {
        std::array<pi::tensorlib::TraceTensor, 2> buffers;
        int current{0};
        int prefetched{-1};
        bool first_call{true};
    };
    auto target_state = std::make_shared<TargetPrefetchState>(
        TargetPrefetchState{.buffers = targets_chunk_gpu, .current = 0, .prefetched = -1, .first_call = true});

    // ---- Forward: streaming LSTM with CE hook ----
    pi::tensorlib::StreamingChunkHook ce_hook =
        [action_targets_host, loss_sum, target_state, seq_len, streaming_chunk, batch_size, n_embed, vocab_size,
         projected_output, ce_stream_desc, head_weight_dtype, &action_model_module, &device_gpu](
            pi::tensorlib::OpGraph &hook_graph, const pi::tensorlib::TraceTensor &output_chunk,
            const uint64_t chunk_start, const uint64_t chunk_steps) -> std::optional<pi::tensorlib::GpuStreamDescriptor>
    {
        if (chunk_start + chunk_steps > seq_len)
        {
            throw std::out_of_range("chunk hook slice: chunk_start=" + std::to_string(chunk_start) +
                                    " chunk_steps=" + std::to_string(chunk_steps) +
                                    " exceeds total_steps=" + std::to_string(streaming_chunk));
        }

        if (chunk_steps > streaming_chunk)
        {
            throw std::out_of_range("chunk_steps exceeds streaming_chunk size");
        }

        if (!target_state->first_call)
        {
            target_state->current = target_state->prefetched;
            target_state->prefetched = -1;
        }

        // Ensure CE stream waits for main-stream writes for this chunk before reading logits.
        pi::tensorlib::AwaitMainStream(hook_graph, device_gpu, ce_stream_desc);

        const auto &current_buffer = target_state->buffers[static_cast<size_t>(target_state->current)];

        if (target_state->first_call)
        {
            pi::tensorlib::TraceTensor targets_host_slice =
                action_targets_host.slice(hook_graph, 0, chunk_start, chunk_steps);
            pi::tensorlib::TraceTensor targets_chunk_slice = current_buffer.slice(hook_graph, 0, 0, chunk_steps);
            // copy on CE compute stream to keep dependency local
            pi::tensorlib::DeviceCopy(hook_graph, targets_host_slice, targets_chunk_slice, ce_stream_desc);
            target_state->first_call = false;
        }

        pi::tensorlib::TraceTensor hidden_slice = output_chunk.slice(hook_graph, 0, 0, chunk_steps);
        pi::tensorlib::TraceTensor targets_slice = current_buffer.slice(hook_graph, 0, 0, chunk_steps);

        // Project hidden->logits on the CE stream (cast if head dtype differs from LSTM output).
        pi::tensorlib::TraceTensor head_input = hidden_slice;
        if (head_input.dtype() != head_weight_dtype)
        {
            head_input = head_input.to(hook_graph, head_weight_dtype, ce_stream_desc);
        }
        pi::tensorlib::TraceTensor flat_hidden = head_input.view(hook_graph, {chunk_steps * batch_size, n_embed});

        pi::tensorlib::TraceTensor projected =
            action_model_module.head().buildForward(hook_graph, {flat_hidden}, false);

        pi::tensorlib::TraceTensor projected_reshaped =
            projected.view(hook_graph, {chunk_steps, batch_size, vocab_size});

        // Capture logits if requested
        if (projected_output.has_value())
        {
            pi::tensorlib::TraceTensor projected_host_slice =
                projected_output->slice(hook_graph, 0, chunk_start, chunk_steps);
            pi::tensorlib::DeviceCopy(hook_graph, projected_reshaped, projected_host_slice, ce_stream_desc);
        }

        // Cross entropy per-row then hierarchical partial sum to scalar and accumulate.
        pi::tensorlib::TraceTensor ce_sum =
            CrossEntropyOnTargets(hook_graph, projected_reshaped, targets_slice, pi::tensorlib::Reduction::ADD,
                                  ce_stream_desc, /*reduce_over_rows=*/true);
        InplaceAdd(hook_graph, loss_sum, ce_sum, ce_stream_desc);
        hook_graph.deleteTensor(ce_sum);
        hook_graph.deleteTensor(projected);

        if (const uint64_t next_start = chunk_start + chunk_steps; next_start < seq_len)
        {
            const uint64_t next_steps = std::min<uint64_t>(streaming_chunk, seq_len - next_start);
            const int next_buf = target_state->current ^ 1;
            pi::tensorlib::TraceTensor next_targets_host_slice =
                action_targets_host.slice(hook_graph, 0, next_start, next_steps);
            pi::tensorlib::TraceTensor next_targets_chunk_slice =
                target_state->buffers[static_cast<size_t>(next_buf)].slice(hook_graph, 0, 0, next_steps);
            // Copy targets for the next chunk on the CE compute stream to avoid queueing behind large H2D copies.
            pi::tensorlib::DeviceCopy(hook_graph, next_targets_host_slice, next_targets_chunk_slice, ce_stream_desc);
            target_state->prefetched = next_buf;
        }
        return ce_stream_desc;
    };
    lstm.setChunkHook(ce_hook);

    pi::tensorlib::LstmForwardStreamingResult result =
        action_model_module.buildForward(graph, {frame_embeddings_host, h0, c0}, false);

    if (capture_projected_output)
    {
        result.output.markRetained();
        result.h_n.markRetained();
        result.c_n.markRetained();
    }

    // Ensure any outstanding H2D copies on the cross-entropy stream complete before final reduction.
    pi::tensorlib::AwaitAsyncTransfers(graph, pi::tensorlib::TransferType::H2D, device_gpu, ce_stream_desc);
    pi::tensorlib::AwaitComputeForTransfer(graph, pi::tensorlib::TransferType::H2D, device_gpu, ce_stream_desc);
    pi::tensorlib::AwaitAsyncTransfers(graph, pi::tensorlib::TransferType::H2D, device_gpu,
                                       pi::tensorlib::GpuStreamDescriptors::Main);

    // ---- Loss normalization ----
    pi::tensorlib::Div(graph, loss_sum, loss_denominator_gpu, loss_mean, ce_stream_desc);

    // Ensure main-stream work (backward/optimizer) waits for CE stream completion.
    pi::tensorlib::AwaitComputeStream(graph, device_gpu, ce_stream_desc);

    // ---- Backward: streaming LSTM gradients ----

    // get LSTM parameters for gradient allocation
    const auto w_ih_param = lstm.getParameter(lstm.weights_ih);
    const auto w_hh_param = lstm.getParameter(lstm.weights_hh);
    const auto b_ih_param = lstm.getParameter(lstm.bias_ih);
    const auto b_hh_param = lstm.getParameter(lstm.bias_hh);

    const auto h0_param = lstm.getParameter(lstm.h0);
    const auto c0_param = lstm.getParameter(lstm.c0);

    // validate all parameters exist
    if (!w_ih_param || !w_hh_param || !b_ih_param || !b_hh_param || !h0_param || !c0_param)
    {
        throw std::runtime_error("Failed to find all LSTM parameters for gradient allocation");
    }

    // Build backward using saved forward context.
    std::unordered_map<std::string, pi::tensorlib::TraceTensor> lstm_operand_grads{};
    pi::tensorlib::TraceTensor grad_x_host =
        graph.createTensor(frame_embeddings_host.shape().dims(), pi::tensorlib::DataType::FLOAT32, device_cpu,
                           pi::tensorlib::GpuStreamDescriptors::Main, true);
    grad_x_host.markRetained();
    lstm_operand_grads.emplace("input", grad_x_host);

    fbamtrain::ActionModelBackwardInput backward_input{
        .action_targets_host = action_targets_host,
        .output_host = result.output,
        .cross_entropy_stream_desc = ce_stream_desc,
        .loss_scale = loss_scale,
    };
    action_model_module.buildBackward(graph, backward_input, action_model_grads, lstm_operand_grads);

    const auto &grad_w_ih = action_model_grads.at(w_ih_param->name);
    const auto &grad_w_hh = action_model_grads.at(w_hh_param->name);
    const auto &grad_b_ih = action_model_grads.at(b_ih_param->name);
    const auto &grad_b_hh = action_model_grads.at(b_hh_param->name);
    const auto &grad_h0 = action_model_grads.at(h0_param->name);
    const auto &grad_c0 = action_model_grads.at(c0_param->name);

    return ActionModelStepGraphResult{
        .graph = std::move(graph),
        .action_model_output = result,
        .loss_sum = loss_sum,
        .loss_mean = loss_mean,
        .projected_output = projected_output,
        .frame_embeddings_host = frame_embeddings_host,
        .action_targets_host = action_targets_host,
        .loss_denominator_host = loss_denominator_host,
        .grad_x = grad_x_host,
        .grad_w_ih = grad_w_ih,
        .grad_w_hh = grad_w_hh,
        .grad_b_ih = grad_b_ih,
        .grad_b_hh = grad_b_hh,
        .grad_h0 = grad_h0,
        .grad_c0 = grad_c0,
    };
}

struct TokenSearchResult
{
    uint32_t token_id{};
    TerminalFrame input_frame{};
    bool has_frame{true};
};

struct StreamTokenizationContext
{
    TermInflateStream *stream{};
    fbamtrain::TokenizerHandle handle;
    TerminalFrame start_frame{};
    bool exhausted{false};
};

static TokenSearchResult ConsumeNextToken(StreamTokenizationContext &context);

// Fixed replay window size: bounds backward replay memory independent of sequence length.
static constexpr size_t BACKWARD_FRAME_REPLAY_CHUNK_SIZE = 32;
// Sentinel for batch items that are exhausted before a replay chunk start position.
static constexpr uint64_t INVALID_REPLAY_FRAME_INDEX = std::numeric_limits<uint64_t>::max();

static std::vector<StreamTokenizationContext> BuildStreamContexts(const fbamtrain::BatchIterator &batch_iterator,
                                                                  const fbamtrain::Tokenizer &tokenizer)
{
    std::vector<StreamTokenizationContext> stream_contexts{};
    stream_contexts.reserve(batch_iterator.streams.size());
    for (auto *inflate_stream : batch_iterator.streams)
    {
        if (!inflate_stream->hasNextFrame())
        {
            StreamTokenizationContext context{
                .stream = inflate_stream,
                .handle = tokenizer.newHandle(),
                .start_frame = {},
                .exhausted = true,
            };
            stream_contexts.push_back(std::move(context));
            continue;
        }
        StreamTokenizationContext context{
            .stream = inflate_stream,
            .handle = tokenizer.newHandle(),
            .start_frame = inflate_stream->readFrame(),
            .exhausted = false,
        };
        stream_contexts.push_back(std::move(context));
    }
    return stream_contexts;
}

static std::vector<StreamTokenizationContext>
BuildReplayStreamContexts(const fbamtrain::BatchIterator &batch_iterator, const fbamtrain::Tokenizer &tokenizer,
                          const std::vector<uint64_t> &replay_start_frame_indices)
{
    // Reconstruct tokenizer/stream state from per-sample frame indices captured during forward.
    if (replay_start_frame_indices.size() != batch_iterator.streams.size())
    {
        throw std::runtime_error("Replay start frame index count does not match stream count.");
    }

    std::vector<StreamTokenizationContext> stream_contexts{};
    stream_contexts.reserve(batch_iterator.streams.size());
    for (size_t i = 0; i < batch_iterator.streams.size(); ++i)
    {
        auto *inflate_stream = batch_iterator.streams[i];
        const uint64_t replay_start = replay_start_frame_indices[i];
        if (replay_start == INVALID_REPLAY_FRAME_INDEX)
        {
            // This sample had no valid frame at the chunk boundary; replay emits padding frames.
            StreamTokenizationContext context{
                .stream = inflate_stream,
                .handle = tokenizer.newHandle(),
                .start_frame = {},
                .exhausted = true,
            };
            stream_contexts.push_back(std::move(context));
            continue;
        }

        inflate_stream->seek(replay_start);
        if (!inflate_stream->hasNextFrame())
        {
            StreamTokenizationContext context{
                .stream = inflate_stream,
                .handle = tokenizer.newHandle(),
                .start_frame = {},
                .exhausted = true,
            };
            stream_contexts.push_back(std::move(context));
            continue;
        }
        StreamTokenizationContext context{
            .stream = inflate_stream,
            .handle = tokenizer.newHandle(),
            .start_frame = inflate_stream->readFrame(),
            .exhausted = false,
        };
        stream_contexts.push_back(std::move(context));
    }
    return stream_contexts;
}

static uint64_t GetReplayStartFrameIndex(const StreamTokenizationContext &context)
{
    // Snapshot the exact decoder position to restart replay at a chunk boundary.
    if (context.exhausted)
    {
        return INVALID_REPLAY_FRAME_INDEX;
    }
    const size_t last_frame_index = context.stream->getLastFrameIndex();
    if (last_frame_index == static_cast<size_t>(-1))
    {
        return INVALID_REPLAY_FRAME_INDEX;
    }
    return static_cast<uint64_t>(last_frame_index);
}

static void RecordReplayChunkStartFrameIndices(const std::vector<StreamTokenizationContext> &stream_contexts,
                                               const size_t chunk_id, const size_t batch_size,
                                               std::vector<uint64_t> &replay_chunk_start_frame_indices)
{
    // Store per-sample restart points once per chunk during forward. Backward will reseek here.
    if ((chunk_id + 1) * batch_size > replay_chunk_start_frame_indices.size())
    {
        throw std::runtime_error("Replay chunk start index table is too small.");
    }
    for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx)
    {
        replay_chunk_start_frame_indices[chunk_id * batch_size + batch_idx] =
            GetReplayStartFrameIndex(stream_contexts[batch_idx]);
    }
}

static void PrepareFrameHeadStepInputs(std::vector<StreamTokenizationContext> &stream_contexts,
                                       uint32_t *cell_states_data, uint32_t *action_targets_data, size_t pos,
                                       size_t batch_size, uint32_t rows, uint32_t cols, size_t vocab_size_raw,
                                       uint32_t max_code_point, std::vector<fbamtrain::TerminalFrameCopy> *saved_frames,
                                       bool prepare_cell_states);

struct DecodedReplayChunk
{
    // Global sequence interval [chunk_start, chunk_start + chunk_len) represented by this decode.
    size_t chunk_start{};
    size_t chunk_len{};
    std::vector<fbamtrain::TerminalFrameCopy> frames{};
};

struct ReplayChunkDecodeContext
{
    // Immutable decode dependencies captured per micro-batch.
    const fbamtrain::BatchIterator *batch_iterator{};
    const fbamtrain::Tokenizer *tokenizer{};
    const std::vector<uint64_t> *replay_chunk_start_frame_indices{};
    size_t batch_size{};
    uint32_t rows{};
    uint32_t cols{};
    size_t vocab_size_raw{};
    uint32_t max_code_point{};
    bool use_frame_parallel{};
    uint32_t local_worker_id{};
    int num_frame_workers{};
};

static DecodedReplayChunk DecodeReplayChunk(const ReplayChunkDecodeContext &context, const size_t chunk_start,
                                            const size_t chunk_len)
{
    // Decode one replay chunk on CPU from saved chunk-boundary frame indices.
    if (!context.batch_iterator || !context.tokenizer || !context.replay_chunk_start_frame_indices)
    {
        throw std::runtime_error("Replay chunk decode context is not initialized.");
    }

    GPUTX_RANGE("fbamtrain::frame_replay_decode_chunk");
    std::vector<uint64_t> replay_chunk_start_indices(context.batch_size, INVALID_REPLAY_FRAME_INDEX);
    std::copy_n(context.replay_chunk_start_frame_indices->data() +
                    (chunk_start / BACKWARD_FRAME_REPLAY_CHUNK_SIZE) * context.batch_size,
                context.batch_size, replay_chunk_start_indices.begin());
    auto replay_contexts =
        BuildReplayStreamContexts(*context.batch_iterator, *context.tokenizer, replay_chunk_start_indices);

    std::vector<fbamtrain::TerminalFrameCopy> decoded_frames(chunk_len * context.batch_size);
    for (size_t chunk_pos = 0; chunk_pos < chunk_len; ++chunk_pos)
    {
        const size_t pos = chunk_start + chunk_pos;
        const uint32_t assigned_worker_id = context.use_frame_parallel
                                                ? static_cast<uint32_t>(pos % context.num_frame_workers)
                                                : context.local_worker_id;
        const bool materialize_locally = !context.use_frame_parallel || assigned_worker_id == context.local_worker_id;
        PrepareFrameHeadStepInputs(replay_contexts, nullptr, nullptr, chunk_pos, context.batch_size, context.rows,
                                   context.cols, context.vocab_size_raw, context.max_code_point,
                                   materialize_locally ? &decoded_frames : nullptr, false);
    }

    return DecodedReplayChunk{
        .chunk_start = chunk_start,
        .chunk_len = chunk_len,
        .frames = std::move(decoded_frames),
    };
}

class ReplayChunkDecodePipeline final
{
  public:
    ReplayChunkDecodePipeline(const ReplayChunkDecodeContext &context, const size_t sequence_length)
        : context_(context), next_chunk_end_(sequence_length)
    {
    }

    void start() { future_valid_ = launchNext(); }

    [[nodiscard]] bool hasPending() const { return future_valid_; }

    DecodedReplayChunk consumeAndLaunchNext()
    {
        // Consume the ready chunk, then immediately launch decode for the next older chunk.
        if (!future_valid_)
        {
            throw std::runtime_error("Replay chunk decode future is not available.");
        }
        DecodedReplayChunk chunk = future_.get();
        future_valid_ = launchNext();
        return chunk;
    }

  private:
    bool launchNext()
    {
        if (next_chunk_end_ == 0)
        {
            return false;
        }
        // Walk backward over the sequence in fixed-size decode windows.
        const size_t chunk_start = (next_chunk_end_ > BACKWARD_FRAME_REPLAY_CHUNK_SIZE)
                                       ? next_chunk_end_ - BACKWARD_FRAME_REPLAY_CHUNK_SIZE
                                       : 0;
        const size_t chunk_len = next_chunk_end_ - chunk_start;
        future_ = std::async(std::launch::async, DecodeReplayChunk, std::cref(context_), chunk_start, chunk_len);
        next_chunk_end_ = chunk_start;
        return true;
    }

    const ReplayChunkDecodeContext &context_;
    size_t next_chunk_end_{0};
    std::future<DecodedReplayChunk> future_{};
    bool future_valid_{false};
};

static void ZeroCellStatesForBatch(uint32_t *cell_states_data, const size_t batch_idx, const uint32_t rows,
                                   const uint32_t cols)
{
    const size_t frame_size = static_cast<size_t>(rows) * cols;
    const size_t stride = frame_size * NUM_FRAME_CHANNELS;
    auto *base = cell_states_data + batch_idx * stride;
    std::fill_n(base, stride, 0);
}

static fbamtrain::TerminalFrameCopy MakeEmptyFrameCopy(const uint32_t rows, const uint32_t cols)
{
    fbamtrain::TerminalFrameCopy copy{};
    copy.width = static_cast<int>(cols);
    copy.height = static_cast<int>(rows);
    copy.cells.assign(static_cast<size_t>(rows) * cols, Cell{});
    return copy;
}

static void PrepareFrameHeadStepInputs(std::vector<StreamTokenizationContext> &stream_contexts,
                                       uint32_t *cell_states_data, uint32_t *action_targets_data, const size_t pos,
                                       const size_t batch_size, const uint32_t rows, const uint32_t cols,
                                       const size_t vocab_size_raw, const uint32_t max_code_point,
                                       std::vector<fbamtrain::TerminalFrameCopy> *saved_frames,
                                       const bool prepare_cell_states)
{
    // Always advance tokenization so all ranks stay aligned. Only populate cell states when needed.
    size_t batch_idx = 0;
    for (auto &context : stream_contexts)
    {
        // if we run out of frames prematurely, pad with empty frames & target zero
        bool pad_empty_frame = false;
        if (context.exhausted)
        {
            pad_empty_frame = true;
        }

        TokenSearchResult result{};
        if (!context.exhausted)
        {
            result = ConsumeNextToken(context);
            if (!result.has_frame)
            {
                context.exhausted = true;
                pad_empty_frame = true;
            }
        }

        if (pad_empty_frame)
        {
            if (action_targets_data)
            {
                action_targets_data[pos * batch_size + batch_idx] = 0;
            }
            if (saved_frames)
            {
                (*saved_frames)[pos * batch_size + batch_idx] = MakeEmptyFrameCopy(rows, cols);
            }
            if (prepare_cell_states)
            {
                ZeroCellStatesForBatch(cell_states_data, batch_idx, rows, cols);
            }
            batch_idx++;
            continue;
        }

        const auto &input_frame = result.input_frame;
        const auto token_id = result.token_id;
        if (input_frame.height != rows || input_frame.width != cols)
        {
            throw std::runtime_error(
                "Obtained input frame from dataset with size different from configuration: Expected " +
                std::to_string(rows) + "x" + std::to_string(cols) + " according to configuration, got " +
                std::to_string(input_frame.height) + "x" + std::to_string(input_frame.width));
        }
        if (action_targets_data)
        {
            action_targets_data[pos * batch_size + batch_idx] = token_id % static_cast<uint32_t>(vocab_size_raw);
        }
        if (saved_frames)
        {
            (*saved_frames)[pos * batch_size + batch_idx] = fbamtrain::TerminalFrameCopy::FromFrame(input_frame);
        }

        if (prepare_cell_states)
        {
            fbamtrain::frameutils::PrepareCellStates(input_frame, batch_idx, rows, cols, max_code_point,
                                                     cell_states_data);
        }
        batch_idx++;
    }
}

static TerminalFrame MakeTerminalFrameView(const fbamtrain::TerminalFrameCopy &frame)
{
    return TerminalFrame{
        .width = frame.width,
        .height = frame.height,
        .cells = const_cast<Cell *>(frame.cells.data()),
        .meta_data = {},
    };
}

static void PrepareCellStatesFromSavedFrames(const std::vector<fbamtrain::TerminalFrameCopy> &saved_frames,
                                             uint32_t *cell_states_data, const size_t pos, const size_t batch_size,
                                             const uint32_t rows, const uint32_t cols, const uint32_t max_code_point)
{
    for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx)
    {
        const auto &saved_frame = saved_frames[pos * batch_size + batch_idx];
        if (saved_frame.width != static_cast<int>(cols) || saved_frame.height != static_cast<int>(rows))
        {
            throw std::runtime_error("Saved frame size mismatch during backward pass");
        }
        const TerminalFrame frame_view = MakeTerminalFrameView(saved_frame);
        fbamtrain::frameutils::PrepareCellStates(frame_view, batch_idx, rows, cols, max_code_point, cell_states_data);
    }
}

static TokenSearchResult ConsumeNextToken(StreamTokenizationContext &context)
{
    auto &handle = context.handle;
    handle.reset();

    if (context.exhausted)
    {
        context.exhausted = true;
        return TokenSearchResult{0, TerminalFrame{}, false};
    }

    TerminalFrame start_frame = context.start_frame;
    TerminalFrame frame = start_frame;
    const auto *frame_data = &frame.meta_data.user_data;

    std::optional<uint32_t> last_token_id{};
    std::optional<size_t> last_frame_index{};

    while (true)
    {
        const auto token_id_opt = handle.currentTokenId();

        if (frame_data->empty())
        {
            throw std::runtime_error("Frame is missing user_data for tokenization");
        }
        size_t data_len = frame_data->size();
        if (frame_data->back() == 0)
        {
            data_len--;
        }
        for (size_t idx = 0; idx < data_len; ++idx)
        {
            handle.addChar((*frame_data)[idx]);
        }

        auto prev_token_id = token_id_opt;
        if (token_id_opt.has_value())
        {
            last_token_id = static_cast<uint32_t>(*token_id_opt);
            last_frame_index = context.stream->getLastFrameIndex();
        }

        if (handle.isDead())
        {
            if (!prev_token_id.has_value())
            {
                if (last_token_id.has_value())
                {
                    context.stream->seek(*last_frame_index);
                    context.start_frame = context.stream->readFrame();
                    prev_token_id = last_token_id;
                }
                else
                {
                    throw std::runtime_error("Tokenizer handle is dead but no token was produced");
                }
            }
            return TokenSearchResult{static_cast<uint32_t>(*prev_token_id), start_frame, true};
        }

        if (!context.stream->hasNextFrame())
        {
            context.exhausted = true;
            return TokenSearchResult{0, TerminalFrame{}, false};
        }
        frame = context.start_frame = context.stream->readFrame();
        frame_data = &frame.meta_data.user_data;
    }
}

static float RunValidation(const fbamtrain::RunConfiguration &run_config, const size_t validation_seq_len,
                           const size_t grad_accum_steps, const uint32_t batch_size, const uint32_t rows,
                           const uint32_t cols, const size_t vocab_size_raw, const pi::tensorlib::Device &device_gpu,
                           const int device_ordinal, const fbamtrain::Tokenizer &tokenizer,
                           fbamtrain::RecordingDatasetIterator &validation_iterator,
                           pi::tensorlib::ExecutionBackend &execution_backend,
                           pi::tensorlib::allocator::AllocatorRegistry &allocator_registry,
                           pi::tensorlib::ExecutionPlan &frame_head_step_exec_plan,
                           const pi::tensorlib::TraceTensor &frame_head_output,
                           const std::shared_ptr<pi::tensorlib::RealTensor> &validation_frame_embeddings,
                           const std::shared_ptr<pi::tensorlib::RealTensor> &validation_action_targets,
                           pi::tensorlib::ExecutionPlan &validation_action_model_exec_plan,
                           const std::shared_ptr<pi::tensorlib::RealTensor> &validation_loss_mean_tensor,
                           const std::shared_ptr<pi::tensorlib::RealTensor> &validation_loss_mean_cpu)
{
    constexpr auto device_cpu = pi::tensorlib::Device{.device_type = pi::tensorlib::DeviceType::CPU, .ordinal = 0};

    auto cell_states_cpu = pi::tensorlib::RealTensor::Allocate({batch_size, rows, cols, NUM_FRAME_CHANNELS},
                                                               pi::tensorlib::DataType::UINT32, device_cpu, true);
    auto cell_states_gpu = pi::tensorlib::RealTensor::Allocate({batch_size, rows, cols, NUM_FRAME_CHANNELS},
                                                               pi::tensorlib::DataType::UINT32, device_gpu);

    const auto stream_bundle = pi::tensorlib::ExecutionBackend::GetStreamBundle(device_gpu);
    float validation_loss = 0.0f;
    auto *action_targets_data = static_cast<uint32_t *>(validation_action_targets->dataptr());

    for (size_t accum_idx = 0; accum_idx < grad_accum_steps; ++accum_idx)
    {
        const fbamtrain::BatchIterator batch_iterator = validation_iterator.nextBatch();
        auto stream_contexts = BuildStreamContexts(batch_iterator, tokenizer);
        auto *cell_states_data = static_cast<uint32_t *>(cell_states_cpu->dataptr());

        {
            GPUTX_RANGE("fbamtrain::validation");
            for (size_t pos = 0; pos < validation_seq_len; ++pos)
            {
                PrepareFrameHeadStepInputs(stream_contexts, cell_states_data, action_targets_data, pos, batch_size,
                                           rows, cols, vocab_size_raw, run_config.model_config.max_code_point, nullptr,
                                           true);

                pi::tensorlib::internal::device_copy::PerformDeviceCopy(cell_states_cpu, cell_states_gpu,
                                                                        stream_bundle->main_stream);

                frame_head_step_exec_plan.updateInputDescriptors(
                    {pi::tensorlib::GraphExecutionInputDescriptor{.name = "cell_states", .tensor = cell_states_gpu}});

                pi::tensorlib::Executor frame_head_executor{frame_head_step_exec_plan, execution_backend,
                                                            device_ordinal};
                {
                    GPUTX_RANGE("fbamtrain::frame_head_fwd");
                    frame_head_executor.execute(allocator_registry, false);
                }

                const auto output_tensor_opt = frame_head_executor.getOutput(frame_head_output, true);
                if (!output_tensor_opt.has_value())
                {
                    throw std::runtime_error("Failed to get frame head output tensor during validation.");
                }

                const auto output_slice = validation_frame_embeddings->at(0, pos);
                // Keep this copy on main stream so each frame's output is consumed before the next frame overwrites it.
                pi::tensorlib::internal::device_copy::PerformDeviceCopy(output_tensor_opt.value(), output_slice,
                                                                        stream_bundle->main_stream);
            }
        }

        pi::tensorlib::ExecutionBackend::SynchronizeStreamBundle(*stream_bundle);
        frame_head_step_exec_plan.releaseTensors();

        pi::tensorlib::Executor action_model_executor{validation_action_model_exec_plan, execution_backend,
                                                      device_ordinal};
        action_model_executor.execute(allocator_registry, false);
        action_model_executor.await();

        pi::tensorlib::internal::device_copy::PerformDeviceCopy(validation_loss_mean_tensor, validation_loss_mean_cpu,
                                                                stream_bundle->main_stream);
        pi::tensorlib::ExecutionBackend::SynchronizeStreamBundle(*stream_bundle);

        validation_loss += static_cast<const float *>(validation_loss_mean_cpu->dataptr())[0];
        validation_action_model_exec_plan.releaseTensors();
    }

    cell_states_cpu->free();
    cell_states_gpu->free();

    return validation_loss;
}

static void run(const fbamtrain::RunConfiguration &run_config, const bool validation_dump_mode,
                const std::string &validation_dump_file, const uint64_t validation_dump_steps,
                const std::optional<fbamtrain::ParallelConfiguration> &parallel_config, const bool is_master,
                const std::optional<uint32_t> &worker_id, const std::optional<uint32_t> &world_size,
                const std::optional<uint32_t> &ddp_world_size)
{
    LOG(INFO) << (validation_dump_mode ? "Starting validation dump run" : "Starting training run")
              << " with configuration:\n"
              << run_config.jsonstr();

    if (parallel_config.has_value())
    {
        LOG(INFO) << "Parallel configuration:\n" << parallel_config->jsonstr();
    }
    if (is_master)
    {
        LOG(INFO) << "Parallel role: master";
    }
    if (worker_id.has_value())
    {
        LOG(INFO) << "Parallel role: worker_id=" << worker_id.value();
    }
    if (world_size.has_value())
    {
        LOG(INFO) << "Parallel world_size=" << world_size.value();
    }
    if (ddp_world_size.has_value())
    {
        LOG(INFO) << "Parallel ddp_world_size=" << ddp_world_size.value();
    }
    const bool parallel_participant = parallel_config.has_value() && (is_master || worker_id.has_value());
    const bool frame_parallel_enabled =
        parallel_participant && parallel_config->frame_head_parallel.use_frame_head_parallel;
    const bool ddp_parallel_enabled = parallel_participant && parallel_config->ddp_parallel.use_ddp_parallel;
    const bool use_ccl_ddp = ddp_parallel_enabled && parallel_config->ddp_parallel.transport == "ccl";
    const bool use_ftccl_ddp = ddp_parallel_enabled && parallel_config->ddp_parallel.transport == "ftccl";
    const bool needs_ccl_communicator = frame_parallel_enabled || use_ccl_ddp;

    if (parallel_config.has_value() && !parallel_participant)
    {
        throw std::runtime_error("Parallel configuration provided but no master/worker role specified.");
    }

    int device_ordinal = 0;
    int frame_head_rank = 0;
    int frame_head_world_size = 1;
    uint32_t local_worker_id = 0;
    std::unordered_map<uint32_t, int> worker_id_to_rank{};
    if (parallel_participant)
    {
        if (needs_ccl_communicator && !world_size.has_value())
        {
            throw std::runtime_error(
                "Parallel participant needs a parent CCL communicator but --world-size is missing.");
        }
        if (needs_ccl_communicator)
        {
            ValidateFrameParallelWorldSize(world_size.value());
            frame_head_world_size = static_cast<int>(world_size.value());
        }
        if (is_master)
        {
            device_ordinal = 0;
            frame_head_rank = 0;
            local_worker_id = 0;
        }
        else
        {
            if (!worker_id.has_value())
            {
                throw std::runtime_error("Worker role specified without worker_id.");
            }
            if (needs_ccl_communicator && (worker_id.value() == 0 || worker_id.value() >= world_size.value()))
            {
                throw std::runtime_error("worker_id must be in [1, world_size) for parent CCL communicator.");
            }
            device_ordinal = static_cast<int>(worker_id.value());
            frame_head_rank = static_cast<int>(worker_id.value());
            local_worker_id = worker_id.value();
        }

        if (needs_ccl_communicator)
        {
            const auto &ccl_endpoint = frame_parallel_enabled ? parallel_config->frame_head_parallel.ccl_rendezvous
                                                              : parallel_config->ddp_parallel.ccl_rendezvous.value();
            fbamtrain::rdvz::PerformCclRendezvous(ccl_endpoint, static_cast<uint32_t>(frame_head_rank),
                                                  world_size.value(), device_ordinal, "Parent CCL rendezvous");
            LOG(INFO) << "Parent CCL rendezvous completed.";
            worker_id_to_rank = BuildWorkerIdToRankMap(world_size.value());
        }
    }
    const auto tokenizer = fbamtrain::Tokenizer::FromFile(run_config.tokenizer_file_path);
    const size_t vocab_size_raw = tokenizer.tokens().size();
    LOG(INFO) << "Loaded tokenizer with " << vocab_size_raw << " tokens from " << run_config.tokenizer_file_path;

    const auto seq_len =
        validation_dump_mode ? run_config.validation_sequence_length : run_config.train_sequence_length;

    fbamtrain::RecordingDatasetIterator train_iterator{run_config.train_data_config.recordings_directory_path,
                                                       run_config.micro_batch_size, run_config.train_sequence_length,
                                                       run_config.train_data_config.iteration_seed};

    fbamtrain::RecordingDatasetIterator valid_iterator{
        run_config.validation_data_config.recordings_directory_path, run_config.micro_batch_size,
        run_config.validation_sequence_length, run_config.validation_data_config.iteration_seed};

    fbamtrain::RecordingDatasetIterator &active_iterator = validation_dump_mode ? valid_iterator : train_iterator;
    std::optional<fbamtrain::DatasetCursorSnapshot> startup_dataset_cursor{};
    std::unique_ptr<fbamtrain::DatasetCursorCommitter> dataset_cursor_committer{};
    if (!validation_dump_mode && !run_config.checkpointing.dataset_cursor_path.empty())
    {
        startup_dataset_cursor = fbamtrain::DatasetCursorCommitter::Load(run_config.checkpointing.dataset_cursor_path);
        if (startup_dataset_cursor.has_value())
        {
            LOG(INFO) << "Found dataset cursor at step_count=" << startup_dataset_cursor->step_count
                      << " in " << run_config.checkpointing.dataset_cursor_path;
        }
        dataset_cursor_committer =
            std::make_unique<fbamtrain::DatasetCursorCommitter>(run_config.checkpointing.dataset_cursor_path);
    }

    constexpr auto device_cpu = pi::tensorlib::Device{.device_type = pi::tensorlib::DeviceType::CPU, .ordinal = 0};
    const auto device_gpu =
        pi::tensorlib::Device{.device_type = pi::tensorlib::DeviceType::GPU, .ordinal = device_ordinal};
    const auto compute_stream_descriptor = pi::tensorlib::GpuStreamDescriptors::Main;

    const auto model_config = run_config.model_config;
    const auto batch_size = run_config.micro_batch_size;
    const auto total_batch_size = run_config.total_batch_size;
    if (total_batch_size == 0 || total_batch_size % batch_size != 0)
    {
        throw std::runtime_error("total_batch_size must be a positive multiple of micro_batch_size.");
    }
    // Gradient accumulation: run multiple micro batches per optimizer step.
    const size_t grad_accum_steps = total_batch_size / batch_size;

    const auto model_dtype = model_config.dtype;
    pi::tensorlib::DataType data_type{};
    switch (model_dtype)
    {
        case fbamtrain::ModelDType::Float16:
            data_type = pi::tensorlib::DataType::FLOAT16;
            break;
        case fbamtrain::ModelDType::BFloat16:
            data_type = pi::tensorlib::DataType::BFLOAT16;
            break;
        default:
            throw std::runtime_error("Unsupported model dtype enum.");
    }

    // Streaming LSTM uses the configured IO dtype; keep head and LSTM in sync with model dtype.
    auto action_model_head_dtype = data_type;
    if (data_type == pi::tensorlib::DataType::BFLOAT16)
    {
        LOG(INFO) << "Using BF16 for frame head + action model head + streaming LSTM.";
    }

    pi::tensorlib::OpGraph init_graph{{}, {}};
    pi::tensorlib::OpGraph frame_head_grad_init_graph{{}, {}};
    pi::tensorlib::OpGraph action_model_grad_init_graph{{}, {}};
    constexpr size_t kVocabAlignment = 8;
    const size_t vocab_size = RoundUpToMultiple(vocab_size_raw, kVocabAlignment);
    if (vocab_size != vocab_size_raw)
    {
        LOG(INFO) << "Padding vocab size from " << vocab_size_raw << " to " << vocab_size << " for aligned GEMM";
    }
    fbamtrain::FrameHeadModule frame_head_module(model_config, init_graph, device_gpu, data_type,
                                                 compute_stream_descriptor);

    // mark all parameters of the frame head as retained
    for (const auto &parameter : frame_head_module.parameters())
    {
        parameter.tensor.markRetained();
    }

    // Classify frame-head gradient tensors by dtype group for contiguous allreduce buffers.
    std::unordered_map<std::string, pi::tensorlib::TraceTensor> frame_head_param_grads{};
    frame_head_param_grads.reserve(frame_head_module.parameters().size());
    std::unordered_map<std::string, pi::tensorlib::TraceTensor> frame_head_param_tensors{};
    frame_head_param_tensors.reserve(frame_head_module.parameters().size());
    std::vector<std::string> frame_head_param_names{};
    std::vector<std::string> frame_head_fp32_grad_names{};
    std::vector<std::string> frame_head_non_fp32_grad_names{};
    frame_head_param_names.reserve(frame_head_module.parameters().size());
    frame_head_fp32_grad_names.reserve(frame_head_module.parameters().size());
    frame_head_non_fp32_grad_names.reserve(frame_head_module.parameters().size());
    for (const auto &parameter : frame_head_module.parameters())
    {
        frame_head_param_tensors.emplace(parameter.name, parameter.tensor);
        frame_head_param_names.emplace_back(parameter.name);
        if (ParameterShouldHaveFp32Grad(parameter.name))
        {
            frame_head_fp32_grad_names.emplace_back(parameter.name);
        }
        else
        {
            frame_head_non_fp32_grad_names.emplace_back(parameter.name);
        }
    }

    constexpr auto ce_stream_desc =
        pi::tensorlib::GpuStreamDescriptor{pi::tensorlib::StreamKind::Compute, CROSS_ENTROPY_STREAM_ID};
    // Set CE stream priority once before any CE stream handle is retrieved; resetting it later destroys the stream.
    pi::tensorlib::ExecutionBackend::SetComputeStreamPriority(device_gpu, ce_stream_desc,
                                                              CROSS_ENTROPY_STREAM_PRIORITY);

    // Action model head runs on the CE stream, async from the main recurrent computation.
    fbamtrain::ActionModelModule action_model_module(model_config, batch_size, vocab_size, init_graph, device_gpu,
                                                     action_model_head_dtype, compute_stream_descriptor,
                                                     ce_stream_desc);

    // mark all parameters of the action model as retained
    for (const auto &parameter : action_model_module.parameters())
    {
        parameter.tensor.markRetained();
    }

    // Create action model gradients in a dedicated init graph so they can be packed contiguously.
    std::unordered_map<std::string, pi::tensorlib::TraceTensor> action_model_grads{};
    std::vector<std::string> action_model_param_names{};
    action_model_param_names.reserve(action_model_module.parameters().size());
    for (const auto &parameter : action_model_module.parameters())
    {
        auto grad_dtype = parameter.tensor.dtype();
        if (grad_dtype == pi::tensorlib::DataType::FLOAT16 || grad_dtype == pi::tensorlib::DataType::BFLOAT16)
        {
            // Accumulate action model grads in FP32 for stability.
            grad_dtype = pi::tensorlib::DataType::FLOAT32;
        }
        pi::tensorlib::TraceTensor grad_tensor = action_model_grad_init_graph.createTensor(
            parameter.tensor.shape().dims(), grad_dtype, device_gpu, compute_stream_descriptor, false);
        pi::tensorlib::FillZeros(action_model_grad_init_graph, grad_tensor, compute_stream_descriptor);
        grad_tensor.markRetained();

        action_model_grads.emplace(parameter.name, grad_tensor);
        action_model_param_names.emplace_back(parameter.name);
    }

    action_model_grad_init_graph.finalize();

    // create frame-embedding output tensor
    pi::tensorlib::TraceTensor frame_embedding_output =
        init_graph.createTensor({seq_len, run_config.micro_batch_size, model_config.n_embed}, data_type, device_cpu,
                                compute_stream_descriptor, true);
    frame_embedding_output.markRetained();

    // create action-target tensor
    pi::tensorlib::TraceTensor action_targets =
        init_graph.createTensor({seq_len, run_config.micro_batch_size}, pi::tensorlib::DataType::UINT32, device_cpu,
                                compute_stream_descriptor, true);
    action_targets.markRetained();

    init_graph.finalize();

    pi::tensorlib::ExecutionBackend &execution_backend = pi::tensorlib::ExecutionBackend::getInstance();
    pi::tensorlib::allocator::AllocatorRegistry &allocator_registry =
        pi::tensorlib::allocator::CachingAllocatorRegistry::instance();

    std::vector<std::shared_ptr<pi::tensorlib::allocator::LocalAllocatorRegistry>> owned_local_allocator_registries{};
    owned_local_allocator_registries.reserve(3);

    // execute model init graph
    pi::tensorlib::ExecutionPlan init_exec_plan = pi::tensorlib::ExecutionPlan::FromGraph(init_graph, {}, {});
    ApplyDefaultPasses(init_exec_plan);

    pi::tensorlib::Executor init_executor{init_exec_plan, execution_backend, device_ordinal};
    init_executor.execute(allocator_registry);
    init_executor.await();

    const auto initialize_frame_head_grad_group = [&](const std::vector<std::string> &grad_names,
                                                      const bool force_fp32) -> std::optional<FrameHeadGradInitResult>
    {
        if (grad_names.empty())
        {
            return std::nullopt;
        }

        frame_head_grad_init_graph = pi::tensorlib::OpGraph{{}, {}};
        for (const auto &name : grad_names)
        {
            const auto parameter_it = frame_head_param_tensors.find(name);
            if (parameter_it == frame_head_param_tensors.end())
            {
                throw std::runtime_error("Failed to resolve frame head parameter tensor for grad init: " + name);
            }

            const auto &parameter_tensor = parameter_it->second;
            const auto grad_dtype = force_fp32 ? pi::tensorlib::DataType::FLOAT32 : parameter_tensor.dtype();
            pi::tensorlib::TraceTensor grad_tensor = frame_head_grad_init_graph.createTensor(
                parameter_tensor.shape().dims(), grad_dtype, device_gpu, compute_stream_descriptor, false);
            pi::tensorlib::FillZeros(frame_head_grad_init_graph, grad_tensor, compute_stream_descriptor);
            grad_tensor.markRetained();
            frame_head_param_grads.insert_or_assign(name, grad_tensor);
        }

        frame_head_grad_init_graph.finalize();
        return InitializeContiguousGradients(frame_head_grad_init_graph, execution_backend, allocator_registry,
                                             device_gpu, device_ordinal, grad_names, frame_head_param_grads,
                                             owned_local_allocator_registries);
    };

    const std::optional<FrameHeadGradInitResult> frame_head_grad_init_non_fp32 =
        initialize_frame_head_grad_group(frame_head_non_fp32_grad_names, false);
    const std::optional<FrameHeadGradInitResult> frame_head_grad_init_fp32 =
        initialize_frame_head_grad_group(frame_head_fp32_grad_names, true);

    std::vector<ContiguousGradAllreduceBuffer> frame_head_grad_allreduce_buffers{};
    frame_head_grad_allreduce_buffers.reserve(2);
    if (frame_head_grad_init_non_fp32.has_value())
    {
        frame_head_grad_allreduce_buffers.push_back(
            ContiguousGradAllreduceBuffer{.allocation_space = frame_head_grad_init_non_fp32->allocation_space,
                                          .dtype = frame_head_grad_init_non_fp32->dtype,
                                          .allreduce_elements = frame_head_grad_init_non_fp32->allreduce_elements});
    }
    if (frame_head_grad_init_fp32.has_value())
    {
        frame_head_grad_allreduce_buffers.push_back(
            ContiguousGradAllreduceBuffer{.allocation_space = frame_head_grad_init_fp32->allocation_space,
                                          .dtype = frame_head_grad_init_fp32->dtype,
                                          .allreduce_elements = frame_head_grad_init_fp32->allreduce_elements});
    }

    const auto action_model_grad_init = InitializeContiguousGradients(
        action_model_grad_init_graph, execution_backend, allocator_registry, device_gpu, device_ordinal,
        action_model_param_names, action_model_grads, owned_local_allocator_registries);
    const auto &action_model_gradient_allocation_space = action_model_grad_init.allocation_space;

    // get frame embedding output tensor
    std::shared_ptr<pi::tensorlib::RealTensor> frame_embedding_output_tensor{};
    {
        const auto output_tensor_opt = init_executor.getOutput(frame_embedding_output);
        if (!output_tensor_opt.has_value())
        {
            throw std::runtime_error("Failed to get frame embedding output tensor from init executor");
        }
        frame_embedding_output_tensor = output_tensor_opt.value();
    }

    // get action targets tensor
    std::shared_ptr<pi::tensorlib::RealTensor> action_targets_tensor{};
    {
        const auto output_tensor_opt = init_executor.getOutput(action_targets);
        if (!output_tensor_opt.has_value())
        {
            throw std::runtime_error("Failed to get action targets tensor from init executor");
        }
        action_targets_tensor = output_tensor_opt.value();
    }

    // prepare loss-related tensors
    auto loss_mean_tensor = pi::tensorlib::RealTensor::AllocateOnStream(
        {1}, pi::tensorlib::DataType::FLOAT32, device_gpu, pi::tensorlib::GpuStreamDescriptors::Main, false);
    auto loss_sum_tensor = pi::tensorlib::RealTensor::AllocateOnStream(
        {1}, pi::tensorlib::DataType::FLOAT32, device_gpu, pi::tensorlib::GpuStreamDescriptors::Main, false);
    auto loss_denominator_tensor =
        pi::tensorlib::RealTensor::Allocate({1}, pi::tensorlib::DataType::FLOAT32, device_cpu, true);
    static_cast<float *>(loss_denominator_tensor->dataptr())[0] = static_cast<float>(seq_len * total_batch_size);

    const auto rows = run_config.frame_rows;
    const auto cols = run_config.frame_cols;
    const auto n_embed = model_config.n_embed;
    const int master_rank = 0;
    const bool use_frame_parallel = frame_parallel_enabled && frame_head_world_size > 1;

    // In frame-parallel mode, only the master runs the recurrence + optimizer.
    // Workers still advance tokenization and compute assigned frame embeddings.
    const bool runs_recurrence = !use_frame_parallel || is_master;
    const bool runs_frame_head_backward = runs_recurrence || use_frame_parallel;
    if (use_ftccl_ddp && runs_recurrence && !ddp_world_size.has_value())
    {
        throw std::runtime_error("FTCCL-backed DDP recurrence masters require --ddp-world-size.");
    }
    const int num_frame_workers = frame_head_world_size;
    if (use_frame_parallel)
    {
        ValidateFrameParallelWorldSize(static_cast<uint32_t>(num_frame_workers));
    }

    // build step graph
    auto [frame_head_step_graph, frame_head_output] =
        BuildFrameEmbeddingStepGraph(batch_size, rows, cols, frame_head_module, device_gpu, data_type);
    const size_t frame_head_output_bytes =
        frame_head_output.shape().numel() * pi::tensorlib::GetDataTypeSize(frame_head_output.dtype());
    const size_t frame_head_upstream_bytes =
        static_cast<size_t>(batch_size) * n_embed * pi::tensorlib::GetDataTypeSize(pi::tensorlib::DataType::FLOAT32);

    std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> frame_head_recv_buffers{};
    std::vector<pi::tensorlib::GpuEvent> frame_head_recv_events{};
    std::vector<bool> frame_head_recv_event_recorded{};
    if (use_frame_parallel && is_master)
    {
        const size_t kRecvBufferCount = std::max<size_t>(2, static_cast<size_t>(num_frame_workers));
        frame_head_recv_buffers.reserve(kRecvBufferCount);
        frame_head_recv_events.reserve(kRecvBufferCount);
        frame_head_recv_event_recorded.assign(kRecvBufferCount, false);
        for (size_t idx = 0; idx < kRecvBufferCount; ++idx)
        {
            frame_head_recv_buffers.push_back(
                pi::tensorlib::RealTensor::Allocate({batch_size, n_embed}, frame_head_output.dtype(), device_gpu));
            frame_head_recv_events.emplace_back(pi::tensorlib::ExecutionBackend::CreateEvent(device_gpu));
        }
    }

    // Double-buffered staging for async sends (prevents overwriting the frame head output before send completes).
    std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> frame_head_send_buffers{};
    std::vector<pi::tensorlib::GpuEvent> frame_head_send_events{};
    std::vector<bool> frame_head_send_event_recorded{};
    if (use_frame_parallel && !is_master)
    {
        const size_t kSendBufferCount = std::max<size_t>(2, static_cast<size_t>(num_frame_workers));
        frame_head_send_buffers.reserve(kSendBufferCount);
        frame_head_send_events.reserve(kSendBufferCount);
        frame_head_send_event_recorded.assign(kSendBufferCount, false);
        for (size_t idx = 0; idx < kSendBufferCount; ++idx)
        {
            frame_head_send_buffers.push_back(
                pi::tensorlib::RealTensor::Allocate({batch_size, n_embed}, frame_head_output.dtype(), device_gpu));
            frame_head_send_events.emplace_back(pi::tensorlib::ExecutionBackend::CreateEvent(device_gpu));
        }
    }

    // The recurrence owner emits upstream grads on CPU; workers receive them on GPU to avoid extra copies.
    const bool frame_head_upstream_on_cpu = runs_recurrence;
    auto frame_head_bwd_graph =
        BuildFrameHeadBackwardGraph(batch_size, rows, cols, n_embed, frame_head_module, frame_head_param_grads,
                                    device_gpu, device_cpu, frame_head_upstream_on_cpu);

    // build action model step graph
    const bool capture_projected_output = validation_dump_mode;
    const float loss_scale =
        static_cast<float>(1.0 / (static_cast<double>(seq_len) * static_cast<double>(total_batch_size)));
    auto action_model_step_graph =
        BuildActionModelStepGraph(seq_len, batch_size, n_embed, vocab_size, loss_scale, action_model_module,
                                  action_model_grads, capture_projected_output, device_gpu, device_cpu);

    std::shared_ptr<pi::tensorlib::RealTensor> cell_states_cpu = pi::tensorlib::RealTensor::Allocate(
        {batch_size, rows, cols, NUM_FRAME_CHANNELS}, pi::tensorlib::DataType::UINT32, device_cpu, true);

    // Create cell states on gpu (double-buffered)
    // current_dst_cell_states_buffer indicates which of the two buffers is currently written to by the data loader.
    // The respective other is read by the step executor.
    std::shared_ptr<pi::tensorlib::RealTensor> cell_states_gpu[2] = {
        pi::tensorlib::RealTensor::Allocate({batch_size, rows, cols, NUM_FRAME_CHANNELS},
                                            pi::tensorlib::DataType::UINT32, device_gpu),
        pi::tensorlib::RealTensor::Allocate({batch_size, rows, cols, NUM_FRAME_CHANNELS},
                                            pi::tensorlib::DataType::UINT32, device_gpu)};

    size_t current_dst_cell_states_buffer = 0;

    // -------- Lambdas for double-buffered cell states management --------

    // lambda to get *read* cell states buffer: This is the buffer read by the step executor.
    const auto get_read_cell_states_buffer = [&cell_states_gpu, &current_dst_cell_states_buffer]
    { return cell_states_gpu[1 - current_dst_cell_states_buffer]; };

    // lambda to get *write* cell states buffer: This is the buffer written to by the data loader.
    const auto get_write_cell_states_buffer = [&cell_states_gpu, &current_dst_cell_states_buffer]
    { return cell_states_gpu[current_dst_cell_states_buffer]; };

    // lambda to swap cell states buffers
    const auto swap_cell_states_buffer = [&current_dst_cell_states_buffer]
    { current_dst_cell_states_buffer = 1 - current_dst_cell_states_buffer; };

    // --------------------------------------------------------------------

    // get stream bundle for gpu device
    const auto stream_bundle = pi::tensorlib::ExecutionBackend::GetStreamBundle(device_gpu);
    pi::tensorlib::ExecutionBackend::GpuStreamWaitFor(stream_bundle->main_stream, stream_bundle->h2d_stream);

    // Use a dedicated compute stream for CCL to avoid blocking the main stream.
    const auto ccl_stream =
        use_frame_parallel ? stream_bundle->getComputeStream(CCL_STREAM_ID) : stream_bundle->main_stream;

    std::shared_ptr<pi::tensorlib::RealTensor> ftccl_step_commit_signal_cpu{};
    std::shared_ptr<pi::tensorlib::RealTensor> ftccl_step_commit_signal_gpu{};
    if (use_frame_parallel && use_ftccl_ddp)
    {
        ftccl_step_commit_signal_cpu =
            pi::tensorlib::RealTensor::Allocate({1}, pi::tensorlib::DataType::UINT32, device_cpu, true);
        ftccl_step_commit_signal_gpu =
            pi::tensorlib::RealTensor::Allocate({1}, pi::tensorlib::DataType::UINT32, device_gpu);
    }

    // Ensure CE stream sees any main-stream initialization before first use.
    const auto ce_stream = stream_bundle->getComputeStream(CROSS_ENTROPY_STREAM_ID);
    pi::tensorlib::ExecutionBackend::GpuStreamWaitFor(stream_bundle->main_stream, ce_stream);

    std::optional<std::reference_wrapper<fbamtrain::ccl::Communicator>> ccl_comm{};
    std::optional<fbamtrain::ccl::Communicator> recurrence_master_subset_comm{};
    std::optional<FtCclState> ftccl_state{};

    if (needs_ccl_communicator)
    {
        ccl_comm.emplace(fbamtrain::rdvz::GetCommunicator());

        // Recurrence-only gradients must not involve pure frame workers. All ranks enter the split call, but only
        // ranks that run recurrence receive the derived subset communicator used by CCL-backed DDP reductions.
        constexpr int kRecurrenceMasterSubsetColor = 0;
        const int subset_color = runs_recurrence ? kRecurrenceMasterSubsetColor : fbamtrain::ccl::kSubsetNoColor;

        recurrence_master_subset_comm =
            fbamtrain::ccl::CreateSubsetCommunicator(ccl_comm->get(), subset_color, frame_head_rank, device_ordinal);

        if (recurrence_master_subset_comm.has_value())
        {
            LOG(INFO) << "Joined recurrence-master subset communicator as rank " << recurrence_master_subset_comm->rank
                      << " of " << recurrence_master_subset_comm->world_size << ".";
        }
        else
        {
            LOG(INFO) << "This rank is not part of the recurrence-master subset communicator.";
        }
    }
    if (use_ccl_ddp && runs_recurrence && !recurrence_master_subset_comm.has_value())
    {
        throw std::runtime_error("CCL-backed DDP requires a recurrence-master subset communicator.");
    }
    if (use_ftccl_ddp && runs_recurrence)
    {
        if (!parallel_config->ddp_parallel.ftccl_master.has_value())
        {
            throw std::runtime_error("FTCCL-backed DDP requires ddp_parallel.ftccl_master.");
        }
        ftccl_state = CreateFtCclState(parallel_config->ddp_parallel.ftccl_master.value(), ddp_world_size.value());
        LOG(INFO) << "FTCCL communicator connected for recurrence master; expected_world_size="
                  << ftccl_state->expected_world_size << ".";
    }

    std::array<pi::tensorlib::GpuEvent, 2> cell_state_ready_events{
        pi::tensorlib::ExecutionBackend::CreateEvent(device_gpu),
        pi::tensorlib::ExecutionBackend::CreateEvent(device_gpu)};

    std::array<bool, 2> cell_state_ready_recorded{false, false};

    // create frame head parameter descriptors
    std::vector<pi::tensorlib::GraphExecutionInputDescriptor> frame_head_parameter_descriptors{};
    for (const auto &[name, trace_tensor] : frame_head_module.parameters())
    {
        const auto real_tensor_opt = init_executor.getOutput(trace_tensor);
        if (!real_tensor_opt.has_value())
        {
            throw std::runtime_error("Failed to get parameter tensor " + name + " from init executor");
        }
        const auto &real_tensor = real_tensor_opt.value();
        frame_head_parameter_descriptors.push_back(
            pi::tensorlib::GraphExecutionInputDescriptor{.name = name, .tensor = real_tensor});
    }

    // create action model parameter descriptors
    std::vector<pi::tensorlib::GraphExecutionInputDescriptor> action_model_parameter_descriptors{};
    for (const auto &[name, trace_tensor] : action_model_module.parameters())
    {
        const auto real_tensor_opt = init_executor.getOutput(trace_tensor);
        if (!real_tensor_opt.has_value())
        {
            throw std::runtime_error("Failed to get parameter tensor " + name + " from init executor");
        }
        const auto &real_tensor = real_tensor_opt.value();
        action_model_parameter_descriptors.push_back(
            pi::tensorlib::GraphExecutionInputDescriptor{.name = name, .tensor = real_tensor});
    }

    // Parameter broadcast is performed after checkpoint load so that all ranks see identical weights.

    // create frame head gradient descriptors
    std::vector<pi::tensorlib::GraphExecutionInputDescriptor> frame_head_grad_descriptors{};
    frame_head_grad_descriptors.reserve(frame_head_param_grads.size());
    for (const auto &name : frame_head_param_names)
    {
        const auto &trace_tensor = frame_head_param_grads.at(name);
        const bool is_fp32 = trace_tensor.dtype() == pi::tensorlib::DataType::FLOAT32;
        if (is_fp32 && !frame_head_grad_init_fp32.has_value())
        {
            throw std::runtime_error("Missing FP32 frame head grad init plan for tensor: " + name);
        }
        if (!is_fp32 && !frame_head_grad_init_non_fp32.has_value())
        {
            throw std::runtime_error("Missing non-FP32 frame head grad init plan for tensor: " + name);
        }
        const auto &init_plan = is_fp32 ? frame_head_grad_init_fp32->plan : frame_head_grad_init_non_fp32->plan;
        const auto real_tensor_opt = init_plan.getRealTensor(trace_tensor);
        if (!real_tensor_opt.has_value())
        {
            throw std::runtime_error("Failed to get frame head gradient tensor " + name + " from grad init plan");
        }
        const auto &grad_tensor = real_tensor_opt.value();
        frame_head_grad_descriptors.push_back(
            pi::tensorlib::GraphExecutionInputDescriptor{.name = name + "_grad", .tensor = grad_tensor});
    }

    // create action model gradient descriptors
    std::vector<pi::tensorlib::GraphExecutionInputDescriptor> action_model_grad_descriptors{};
    action_model_grad_descriptors.reserve(action_model_grads.size());
    for (const auto &name : action_model_param_names)
    {
        const auto &trace_tensor = action_model_grads.at(name);
        const auto real_tensor_opt = action_model_grad_init.plan.getRealTensor(trace_tensor);
        if (!real_tensor_opt.has_value())
        {
            throw std::runtime_error("Failed to get action model gradient tensor " + name + " from grad init plan");
        }
        const auto &grad_tensor = real_tensor_opt.value();
        action_model_grad_descriptors.push_back(
            pi::tensorlib::GraphExecutionInputDescriptor{.name = name + "_grad", .tensor = grad_tensor});
    }

    // create frame head step exec plan
    pi::tensorlib::ExecutionPlan frame_head_step_exec_plan = pi::tensorlib::ExecutionPlan::FromGraph(
        frame_head_step_graph,
        {pi::tensorlib::GraphExecutionInputDescriptor{.name = "cell_states", .tensor = get_read_cell_states_buffer()}},
        frame_head_parameter_descriptors);
    ApplyDefaultPasses(frame_head_step_exec_plan);

    // create frame head bwd graph inputs
    std::vector<pi::tensorlib::GraphExecutionInputDescriptor> frame_head_bwd_inputs{};
    frame_head_bwd_inputs.push_back(
        pi::tensorlib::GraphExecutionInputDescriptor{.name = "cell_states", .tensor = get_read_cell_states_buffer()});
    const auto frame_head_upstream_device = frame_head_upstream_on_cpu ? device_cpu : device_gpu;
    const bool frame_head_upstream_pinned = frame_head_upstream_on_cpu;
    auto frame_head_upstream_input_buffer =
        pi::tensorlib::RealTensor::Allocate({batch_size, n_embed}, pi::tensorlib::DataType::FLOAT32,
                                            frame_head_upstream_device, frame_head_upstream_pinned);
    frame_head_bwd_inputs.push_back(pi::tensorlib::GraphExecutionInputDescriptor{
        .name = "frame_head_upstream", .tensor = frame_head_upstream_input_buffer});

    // concat frame head parameter and gradient descriptors
    std::vector<pi::tensorlib::GraphExecutionInputDescriptor> frame_head_param_and_grad_descriptors{};
    {
        frame_head_param_and_grad_descriptors.reserve(frame_head_parameter_descriptors.size() +
                                                      frame_head_grad_descriptors.size());
        // concat
        frame_head_param_and_grad_descriptors.insert(frame_head_param_and_grad_descriptors.end(),
                                                     frame_head_parameter_descriptors.begin(),
                                                     frame_head_parameter_descriptors.end());

        frame_head_param_and_grad_descriptors.insert(frame_head_param_and_grad_descriptors.end(),
                                                     frame_head_grad_descriptors.begin(),
                                                     frame_head_grad_descriptors.end());
    }

    // create frame head bwd exec plan
    pi::tensorlib::ExecutionPlan frame_head_bwd_exec_plan = pi::tensorlib::ExecutionPlan::FromGraph(
        frame_head_bwd_graph.graph, frame_head_bwd_inputs, frame_head_param_and_grad_descriptors);
    ApplyDefaultPasses(frame_head_bwd_exec_plan);

    FrameHeadUpstreamStaging frame_head_upstream =
        InitializeFrameHeadUpstreamStaging(use_frame_parallel, is_master, num_frame_workers, batch_size, n_embed,
                                           device_gpu, frame_head_upstream_input_buffer);

    std::unordered_map<std::string, std::shared_ptr<pi::tensorlib::RealTensor>> frame_head_grad_real_tensors{};
    frame_head_grad_real_tensors.reserve(frame_head_param_names.size());
    for (const auto &name : frame_head_param_names)
    {
        const auto &trace_tensor = frame_head_param_grads.at(name);
        const auto real_tensor_opt = frame_head_bwd_exec_plan.getRealTensor(trace_tensor);
        if (!real_tensor_opt.has_value())
        {
            throw std::runtime_error("Failed to resolve frame head gradient tensor for reduction: " + name);
        }
        const auto &real_tensor = real_tensor_opt.value();
        frame_head_grad_real_tensors.emplace(name, real_tensor);
    }

    // concat action model parameter and gradient descriptors
    std::vector<pi::tensorlib::GraphExecutionInputDescriptor> action_model_param_and_grad_descriptors{};
    {
        action_model_param_and_grad_descriptors.reserve(action_model_parameter_descriptors.size() +
                                                        action_model_grads.size());
        // concat
        action_model_param_and_grad_descriptors.insert(action_model_param_and_grad_descriptors.end(),
                                                       action_model_parameter_descriptors.begin(),
                                                       action_model_parameter_descriptors.end());
        action_model_param_and_grad_descriptors.insert(action_model_param_and_grad_descriptors.end(),
                                                       action_model_grad_descriptors.begin(),
                                                       action_model_grad_descriptors.end());
    }

    // create action model exec plan (fwd & bwd)
    pi::tensorlib::ExecutionPlan action_model_exec_plan = pi::tensorlib::ExecutionPlan::FromGraph(
        action_model_step_graph.graph,
        {
            // supply pre-allocated "loss_mean" and "loss_sum" tensors
            pi::tensorlib::GraphExecutionInputDescriptor{.name = "loss_mean", .tensor = loss_mean_tensor},
            pi::tensorlib::GraphExecutionInputDescriptor{.name = "loss_sum", .tensor = loss_sum_tensor},

            // both the input frame embeddings and targets must be supplied as inputs to the execution plan
            pi::tensorlib::GraphExecutionInputDescriptor{.name = "frame_embeddings",
                                                         .tensor = frame_embedding_output_tensor},
            pi::tensorlib::GraphExecutionInputDescriptor{.name = "action_targets", .tensor = action_targets_tensor},
            pi::tensorlib::GraphExecutionInputDescriptor{.name = "loss_denominator", .tensor = loss_denominator_tensor},
        },
        action_model_param_and_grad_descriptors);
    ApplyDefaultPasses(action_model_exec_plan);

    const bool periodic_validation_enabled =
        !validation_dump_mode && run_config.enable_validation && run_config.validation_interval > 0;
    const bool report_validation_metrics = !parallel_participant || runs_recurrence;

    std::shared_ptr<pi::tensorlib::RealTensor> validation_frame_embedding_output_tensor{};
    std::shared_ptr<pi::tensorlib::RealTensor> validation_action_targets_tensor{};
    std::shared_ptr<pi::tensorlib::RealTensor> validation_loss_mean_tensor{};
    std::shared_ptr<pi::tensorlib::RealTensor> validation_loss_sum_tensor{};
    std::shared_ptr<pi::tensorlib::RealTensor> validation_loss_mean_cpu{};
    std::shared_ptr<pi::tensorlib::RealTensor> validation_loss_denominator_tensor{};
    std::optional<ActionModelStepGraphResult> validation_action_model_step_graph{};
    std::optional<pi::tensorlib::ExecutionPlan> validation_action_model_exec_plan{};

    if (periodic_validation_enabled)
    {
        const size_t validation_seq_len = run_config.validation_sequence_length;

        validation_frame_embedding_output_tensor =
            pi::tensorlib::RealTensor::Allocate({validation_seq_len, batch_size, n_embed}, data_type, device_cpu, true);
        validation_action_targets_tensor = pi::tensorlib::RealTensor::Allocate(
            {validation_seq_len, batch_size}, pi::tensorlib::DataType::UINT32, device_cpu, true);

        validation_loss_mean_tensor = pi::tensorlib::RealTensor::AllocateOnStream(
            {1}, pi::tensorlib::DataType::FLOAT32, device_gpu, pi::tensorlib::GpuStreamDescriptors::Main, false);
        validation_loss_sum_tensor = pi::tensorlib::RealTensor::AllocateOnStream(
            {1}, pi::tensorlib::DataType::FLOAT32, device_gpu, pi::tensorlib::GpuStreamDescriptors::Main, false);
        validation_loss_mean_cpu =
            pi::tensorlib::RealTensor::Allocate({1}, pi::tensorlib::DataType::FLOAT32, device_cpu, true);
        validation_loss_denominator_tensor =
            pi::tensorlib::RealTensor::Allocate({1}, pi::tensorlib::DataType::FLOAT32, device_cpu, true);
        static_cast<float *>(validation_loss_denominator_tensor->dataptr())[0] =
            static_cast<float>(validation_seq_len * total_batch_size);

        const float validation_loss_scale =
            static_cast<float>(1.0 / (static_cast<double>(validation_seq_len) * static_cast<double>(total_batch_size)));
        validation_action_model_step_graph =
            BuildActionModelStepGraph(validation_seq_len, batch_size, n_embed, vocab_size, validation_loss_scale,
                                      action_model_module, action_model_grads, false, device_gpu, device_cpu);

        validation_action_model_exec_plan =
            pi::tensorlib::ExecutionPlan::FromGraph(validation_action_model_step_graph->graph,
                                                    {pi::tensorlib::GraphExecutionInputDescriptor{
                                                         .name = "loss_mean",
                                                         .tensor = validation_loss_mean_tensor,
                                                     },
                                                     pi::tensorlib::GraphExecutionInputDescriptor{
                                                         .name = "loss_sum",
                                                         .tensor = validation_loss_sum_tensor,
                                                     },
                                                     pi::tensorlib::GraphExecutionInputDescriptor{
                                                         .name = "frame_embeddings",
                                                         .tensor = validation_frame_embedding_output_tensor,
                                                     },
                                                     pi::tensorlib::GraphExecutionInputDescriptor{
                                                         .name = "action_targets",
                                                         .tensor = validation_action_targets_tensor,
                                                     },
                                                     pi::tensorlib::GraphExecutionInputDescriptor{
                                                         .name = "loss_denominator",
                                                         .tensor = validation_loss_denominator_tensor,
                                                     }},
                                                    action_model_param_and_grad_descriptors);
        ApplyDefaultPasses(*validation_action_model_exec_plan);
    }

    // Zero all frame-head grad buffers (bf16/fp16 and fp32) in one plan.
    pi::tensorlib::ExecutionPlan frame_head_grad_zero_plan =
        BuildContiguousGradZeroPlan(frame_head_grad_allreduce_buffers, "frame_head_grad_buffer");
    ApplyDefaultPasses(frame_head_grad_zero_plan);
    pi::tensorlib::Executor frame_head_grad_zero_executor{frame_head_grad_zero_plan, execution_backend, device_ordinal};

    pi::tensorlib::ExecutionPlan action_model_grad_zero_plan =
        BuildContiguousGradZeroPlan(action_model_gradient_allocation_space, "action_model_grad_buffer");

    ApplyDefaultPasses(action_model_grad_zero_plan);

    pi::tensorlib::Executor action_model_grad_zero_executor{action_model_grad_zero_plan, execution_backend,
                                                            device_ordinal};

    // create named map of all parameters (frame head + action model)
    std::unordered_map<std::string, std::shared_ptr<pi::tensorlib::RealTensor>> optimizer_params{};
    {
        optimizer_params.reserve(frame_head_parameter_descriptors.size() + action_model_parameter_descriptors.size());
        for (const auto &entry : frame_head_parameter_descriptors)
        {
            optimizer_params.emplace(entry.name, entry.tensor);
        }
        for (const auto &entry : action_model_parameter_descriptors)
        {
            optimizer_params.emplace(entry.name, entry.tensor);
        }
    }

    // Frame head optimizer parameters (all ranks).
    std::vector<fbamtrain::optim::ParameterGrad> frame_head_optimizer_parameters{};
    frame_head_optimizer_parameters.reserve(frame_head_param_names.size());
    for (const auto &name : frame_head_param_names)
    {
        const auto param_it = optimizer_params.find(name);
        if (param_it == optimizer_params.end())
        {
            throw std::runtime_error("Failed to resolve frame head parameter for optimizer: " + name);
        }
        const auto grad_it = frame_head_grad_real_tensors.find(name);
        if (grad_it == frame_head_grad_real_tensors.end())
        {
            throw std::runtime_error("Failed to resolve frame head gradient for optimizer: " + name);
        }
        frame_head_optimizer_parameters.push_back(
            fbamtrain::optim::ParameterGrad{.name = name, .param = param_it->second, .grad = grad_it->second});
    }

    // Action model optimizer parameters (master only).
    std::vector<fbamtrain::optim::ParameterGrad> action_model_optimizer_parameters{};
    action_model_optimizer_parameters.reserve(action_model_grads.size());
    for (const auto &[name, trace_tensor] : action_model_grads)
    {
        const auto param_it = optimizer_params.find(name);
        if (param_it == optimizer_params.end())
        {
            throw std::runtime_error("Failed to resolve action model parameter for optimizer: " + name);
        }
        const auto grad_opt = action_model_exec_plan.getRealTensor(trace_tensor);
        if (!grad_opt.has_value())
        {
            throw std::runtime_error("Failed to resolve action model gradient for optimizer: " + name);
        }
        action_model_optimizer_parameters.push_back(
            fbamtrain::optim::ParameterGrad{.name = name, .param = param_it->second, .grad = grad_opt.value()});
    }

    std::unique_ptr<fbamtrain::optim::Optimizer> frame_head_optimizer = fbamtrain::optim::CreateOptimizer(
        run_config.optimizer_config, frame_head_optimizer_parameters, device_gpu, compute_stream_descriptor);
    std::unique_ptr<fbamtrain::optim::Optimizer> action_model_optimizer{};
    if (runs_recurrence)
    {
        action_model_optimizer = fbamtrain::optim::CreateOptimizer(
            run_config.optimizer_config, action_model_optimizer_parameters, device_gpu, compute_stream_descriptor);
    }

    fbamtrain::checkpointing::CheckpointManager checkpoint_manager{run_config.checkpointing,
                                                                   run_config.max_training_steps};
    const bool checkpointing_enabled = !validation_dump_mode && checkpoint_manager.enabled();

    uint64_t step = 0;

    // load from checkpoint if applicable
    if (checkpointing_enabled &&
        run_config.checkpointing.resume_behavior == fbamtrain::CheckpointResumeBehavior::LoadLatest)
    {
        const auto outcome =
            LoadLatestCheckpointIfAvailable(run_config, checkpoint_manager, optimizer_params, *frame_head_optimizer,
                                            action_model_optimizer.get(), active_iterator, compute_stream_descriptor);
        if (outcome.loaded)
        {
            step = outcome.step;
        }
    }
    if (startup_dataset_cursor.has_value())
    {
        if (startup_dataset_cursor->batch_size != run_config.micro_batch_size)
        {
            throw std::runtime_error("Dataset cursor batch_size does not match run configuration.");
        }
        if (startup_dataset_cursor->sequence_length != run_config.train_sequence_length)
        {
            throw std::runtime_error("Dataset cursor sequence_length does not match run configuration.");
        }
        if (startup_dataset_cursor->step_count != step)
        {
            LOG(WARN) << "Dataset cursor step_count=" << startup_dataset_cursor->step_count
                      << " differs from restored training step=" << step
                      << "; restoring dataset cursor without changing model step.";
        }
        active_iterator.restore(startup_dataset_cursor->toDatasetIteratorState());
        LOG(INFO) << "Restored dataset cursor from " << run_config.checkpointing.dataset_cursor_path
                  << " at step_count=" << startup_dataset_cursor->step_count;
    }

    std::optional<FtCclSharedStateView> ftccl_shared_state_view{};
    std::optional<std::map<std::string, std::shared_ptr<pi::tensorlib::RealTensor>>>
        frame_parallel_ccl_state_view{};
    if (use_frame_parallel)
    {
        frame_parallel_ccl_state_view.emplace();
        AppendFrameParallelCclState(*frame_parallel_ccl_state_view, optimizer_params, frame_head_module,
                                    *frame_head_optimizer);
    }

    if (ftccl_state.has_value())
    {
        assert(runs_recurrence);
        if (!action_model_optimizer)
        {
            throw std::runtime_error("FTCCL-backed DDP requires an action-model optimizer on recurrence masters.");
        }

        // In the FTCCL path, the shared-state revision is the durable training progress counter. A fresh FTCCL
        // master accepts non-zero initial revisions, which is what lets checkpoint resumes start at checkpoint step.
        ftccl_state->shared_state_revision = step;

        // FTCCL hashes and may transmit shared-state tensors directly. Ensure all tensorlib streams are quiescent
        // before exposing GPU-backed parameters and optimizer buffers to FTCCL.
        pi::tensorlib::ExecutionBackend::SynchronizeStreamBundle(*stream_bundle);
        ftccl_shared_state_view =
            BuildFtCclSharedStateView(optimizer_params, frame_head_module, &action_model_module, *frame_head_optimizer,
                                      action_model_optimizer.get());
        while (!PrepareFtCclCollectivePhase(*ftccl_state, false, "initial shared-state synchronization"))
        {
        }
        const auto initial_sync = SynchronizeFtCclSharedState(*ftccl_state, *ftccl_shared_state_view,
                                                              "FTCCL initial shared-state synchronization");
        if (initial_sync.rx_bytes != 0)
        {
            LOG(INFO) << "FTCCL initial shared-state sync received state; local frame-parallel state will be "
                         "redistributed before training.";
        }
        step = ftccl_state->shared_state_revision;
    }

    // Broadcast frame-local state exactly once after any checkpoint load and FTCCL initial sync decision.
    if (use_frame_parallel)
    {
        if (!ccl_comm.has_value())
        {
            throw std::runtime_error("CCL communicator missing for initial frame-parallel state broadcast.");
        }
        if (!frame_parallel_ccl_state_view.has_value())
        {
            throw std::runtime_error("Frame-parallel CCL state view missing for initial broadcast.");
        }
        BroadcastCclTensorState(*frame_parallel_ccl_state_view, &ccl_comm->get(), device_ordinal, master_rank,
                                stream_bundle->main_stream, ccl_stream, "Initial frame-parallel state");
    }

    const uint64_t max_steps = validation_dump_mode ? validation_dump_steps : run_config.max_training_steps;
    const bool dump_validation_like = validation_dump_mode;

    // event to signal completion of cell states copy to gpu
    auto copy_complete_event = pi::tensorlib::ExecutionBackend::CreateEvent(device_gpu);
    const bool emit_train_metrics_json = runs_recurrence && ShouldEmitMetricsJson();
    std::unique_ptr<LossAccumulationRingBuffer> loss_accumulation_ring{};
    if (runs_recurrence)
    {
        loss_accumulation_ring = std::make_unique<LossAccumulationRingBuffer>(
            device_gpu, device_cpu, device_ordinal, LOSS_ACCUM_STREAM_ID, LOSS_LOGGER_COPY_STREAM_ID, execution_backend,
            allocator_registry);
    }
    AsyncStepLogger async_step_logger{seq_len, loss_accumulation_ring.get()};

    while (step < max_steps)
    {
        auto step_start_event = pi::tensorlib::ExecutionBackend::CreateEvent(device_gpu, true);
        step_start_event.record(stream_bundle->main_stream);
        if (rows != model_config.frame_rows || cols != model_config.frame_cols)
        {
            throw std::runtime_error("Model frame size does not match training configuration: Model is " +
                                     std::to_string(model_config.frame_rows) + "x" +
                                     std::to_string(model_config.frame_cols) + ", training configuration is " +
                                     std::to_string(rows) + "x" + std::to_string(cols));
        }

        LOG(INFO) << (validation_dump_mode ? "Validation dump step " : "Training step ") << step;

        if (loss_accumulation_ring)
        {
            loss_accumulation_ring->BeginStep();
        }

        // Zero grads once per optimizer step; micro batches accumulate into these buffers.
        {
            GPUTX_RANGE("fbamtrain::zero_grad");
            if (runs_recurrence)
            {
                action_model_grad_zero_executor.execute(allocator_registry, false);
            }

            if (runs_frame_head_backward)
            {
                frame_head_grad_zero_executor.execute(allocator_registry, false);
            }
            // Intentionally no host-side await() here: both zero-grad graphs enqueue to streams that are naturally
            // ordered with subsequent work, so explicit cuStreamSynchronize() would only add idle time.
        }

        for (size_t accum_idx = 0; accum_idx < grad_accum_steps; ++accum_idx)
        {
            if (grad_accum_steps > 1)
            {
                LOG(INFO) << "Micro batch " << (accum_idx + 1) << "/" << grad_accum_steps;
            }
            fbamtrain::BatchIterator batch_iterator{};
            {
                GPUTX_RANGE("fbamtrain::next_batch_wait");
                batch_iterator = active_iterator.nextBatch();
            }

            auto stream_contexts = BuildStreamContexts(batch_iterator, tokenizer);

            auto cell_states_data = static_cast<uint32_t *>(cell_states_cpu->dataptr());
            auto action_targets_data =
                runs_recurrence ? static_cast<uint32_t *>(action_targets_tensor->dataptr()) : nullptr;
            const size_t replay_chunk_count =
                (seq_len + BACKWARD_FRAME_REPLAY_CHUNK_SIZE - 1) / BACKWARD_FRAME_REPLAY_CHUNK_SIZE;

            std::vector<uint64_t> replay_chunk_start_frame_indices{};
            if (runs_frame_head_backward)
            {
                replay_chunk_start_frame_indices.resize(replay_chunk_count * batch_size, INVALID_REPLAY_FRAME_INDEX);
            }

            ReplayChunkDecodeContext replay_decode_context{
                // References remain valid for the whole micro-batch; decode tasks only run in that scope.
                .batch_iterator = &batch_iterator,
                .tokenizer = &tokenizer,
                .replay_chunk_start_frame_indices = &replay_chunk_start_frame_indices,
                .batch_size = batch_size,
                .rows = static_cast<uint32_t>(rows),
                .cols = static_cast<uint32_t>(cols),
                .vocab_size_raw = vocab_size_raw,
                .max_code_point = model_config.max_code_point,
                .use_frame_parallel = use_frame_parallel,
                .local_worker_id = local_worker_id,
                .num_frame_workers = num_frame_workers,
            };

            ReplayChunkDecodePipeline replay_decode_pipeline{replay_decode_context, seq_len};

            auto forward_start_event = pi::tensorlib::ExecutionBackend::CreateEvent(device_gpu, true);
            forward_start_event.record(stream_bundle->main_stream);
            async_step_logger.enqueueFramePhaseStart(
                FramePhaseStartLogRequest{false, seq_len, batch_size, std::move(forward_start_event)});

            // forward
            {
                GPUTX_RANGE("fbamtrain::forward");

                bool copy_complete_recorded = false;
                size_t forward_completed_idx = 0;
                size_t frame_head_fwd_send_sequence = 0;
                size_t frame_head_fwd_recv_sequence = 0;

                for (size_t pos = 0; pos < seq_len; pos++)
                {
                    if (runs_frame_head_backward && (pos % BACKWARD_FRAME_REPLAY_CHUNK_SIZE == 0))
                    {
                        // Capture per-sample replay restart points at chunk boundaries during forward.
                        const size_t chunk_id = pos / BACKWARD_FRAME_REPLAY_CHUNK_SIZE;
                        RecordReplayChunkStartFrameIndices(stream_contexts, chunk_id, batch_size,
                                                           replay_chunk_start_frame_indices);
                    }

                    // Frame head parallelism: shard frames by position across worker ids.
                    const uint32_t assigned_worker_id =
                        use_frame_parallel ? static_cast<uint32_t>(pos % num_frame_workers) : local_worker_id;
                    const bool compute_locally = !use_frame_parallel || assigned_worker_id == local_worker_id;
                    const int assigned_rank =
                        use_frame_parallel ? worker_id_to_rank.at(assigned_worker_id) : frame_head_rank;

                    if (compute_locally && copy_complete_recorded)
                    {
                        // Wait for previous HtoD copy to complete.
                        // Only after this is it safe to overwrite the cpu cell states buffer.
                        copy_complete_event.synchronize();
                    }

                    PrepareFrameHeadStepInputs(stream_contexts, cell_states_data, action_targets_data, pos, batch_size,
                                               rows, cols, vocab_size_raw, model_config.max_code_point, nullptr,
                                               compute_locally);

                    std::unique_ptr<pi::tensorlib::Executor> step_executor{};

                    if (compute_locally)
                    {
                        // copy cell states to gpu
                        size_t read_buffer_index = 0;
                        {
                            // copy to the *write* cell states buffer
                            {
                                const auto write_buffer_index = current_dst_cell_states_buffer;
                                if (cell_state_ready_recorded[write_buffer_index])
                                {
                                    pi::tensorlib::ExecutionBackend::GpuStreamWaitForEvent(
                                        cell_state_ready_events[write_buffer_index], stream_bundle->h2d_stream);
                                    cell_state_ready_recorded[write_buffer_index] = false;
                                }
                                const auto write_cell_states_buffer = get_write_cell_states_buffer();
                                pi::tensorlib::internal::device_copy::PerformDeviceCopy(
                                    cell_states_cpu, write_cell_states_buffer, stream_bundle->h2d_stream);
                                copy_complete_event.record(stream_bundle->h2d_stream); // record an event after the copy
                                copy_complete_recorded = true;

                                swap_cell_states_buffer(); // swap buffers
                                read_buffer_index = write_buffer_index;
                            }

                            // update step execution plan with the new read cell states buffer:
                            // get_read_cell_states_buffer() now returns the buffer that was just copied to.
                            frame_head_step_exec_plan.updateInputDescriptors(
                                {pi::tensorlib::GraphExecutionInputDescriptor{
                                    .name = "cell_states", .tensor = get_read_cell_states_buffer()}});

                            // NOTE: step_exec_plan is intentionally copied here.
                            // This allows safe mutation of the step_exec_plan local variable, which we need to mutate
                            // to implement the double-buffered data-loader transfer scheme.
                            // It is also intentional that the step_executor is created *after* we update the input
                            // descriptors of the step_exec_plan such that the step_executor uses the correct cell
                            // states buffer.
                            step_executor = std::make_unique<pi::tensorlib::Executor>(
                                frame_head_step_exec_plan, execution_backend, device_ordinal);
                            step_executor->gpuWaitFor(
                                copy_complete_event); // make the step executor wait for the copy to complete
                        }

                        // there is no need to await the previous step executor, as all executors issue commands into
                        // the same streams, retaining sequential consistency.

                        // execute step
                        {
                            GPUTX_RANGE("fbamtrain::frame_head_fwd");
                            step_executor->execute(allocator_registry, false);
                        }

                        // Recurrence owner writes embeddings locally; workers stage + send theirs to the master.
                        {
                            const auto output_tensor_opt = step_executor->getOutput(frame_head_output, true);
                            if (!output_tensor_opt.has_value())
                            {
                                throw std::runtime_error("Failed to get frame head output tensor from executor");
                            }
                            const auto &output_tensor = output_tensor_opt.value();
                            if (runs_recurrence)
                            {
                                const auto output_slice = frame_embedding_output_tensor->at(0, pos);
                                pi::tensorlib::internal::device_copy::PerformDeviceCopy(output_tensor, output_slice,
                                                                                        stream_bundle->main_stream);
                            }
                            // if we are a worker, and frame head parallel is enabled, send the output to the master
                            else if (use_frame_parallel && !is_master)
                            {
                                if (!ccl_comm.has_value())
                                {
                                    throw std::runtime_error("CCL communicator missing for frame head send.");
                                }
                                if (frame_head_send_buffers.empty())
                                {
                                    throw std::runtime_error("Frame head send buffers are not initialized.");
                                }
                                const size_t send_buffer_index =
                                    frame_head_fwd_send_sequence % frame_head_send_buffers.size();
                                ++frame_head_fwd_send_sequence;
                                if (frame_head_send_event_recorded[send_buffer_index])
                                {
                                    pi::tensorlib::ExecutionBackend::GpuStreamWaitForEvent(
                                        frame_head_send_events[send_buffer_index], stream_bundle->main_stream);
                                    frame_head_send_event_recorded[send_buffer_index] = false;
                                }
                                const auto &send_buffer = frame_head_send_buffers[send_buffer_index];
                                pi::tensorlib::internal::device_copy::PerformDeviceCopy(output_tensor, send_buffer,
                                                                                        stream_bundle->main_stream);
                                pi::tensorlib::ExecutionBackend::GpuStreamWaitFor(stream_bundle->main_stream,
                                                                                  ccl_stream);
                                GPUTX_RANGE("fbamtrain::ccl_send");
                                fbamtrain::ccl::SendBufferAsync(ccl_comm->get(), master_rank, device_ordinal,
                                                                send_buffer->dataptr(), frame_head_output_bytes,
                                                                ccl_stream);
                                frame_head_send_events[send_buffer_index].record(ccl_stream);
                                frame_head_send_event_recorded[send_buffer_index] = true;
                            }
                        }

                        cell_state_ready_events[read_buffer_index].record(stream_bundle->main_stream);
                        cell_state_ready_recorded[read_buffer_index] = true;
                    }
                    else
                    {
                        if (!runs_recurrence)
                        {
                            // Non-recurrence workers skip frames they do not own.
                        }
                        else if (frame_head_recv_buffers.empty() || !ccl_comm.has_value())
                        {
                            throw std::runtime_error("Frame head receive buffer or communicator is not initialized.");
                        }
                        else
                        {
                            const size_t recv_buffer_index =
                                frame_head_fwd_recv_sequence % frame_head_recv_buffers.size();
                            ++frame_head_fwd_recv_sequence;
                            if (frame_head_recv_event_recorded[recv_buffer_index])
                            {
                                pi::tensorlib::ExecutionBackend::GpuStreamWaitForEvent(
                                    frame_head_recv_events[recv_buffer_index], ccl_stream);
                                frame_head_recv_event_recorded[recv_buffer_index] = false;
                            }
                            const auto &recv_buffer = frame_head_recv_buffers[recv_buffer_index];
                            // Receive frame embeddings computed by another rank.
                            GPUTX_RANGE("fbamtrain::ccl_recv");
                            fbamtrain::ccl::RecvBufferAsync(ccl_comm->get(), assigned_rank, device_ordinal,
                                                            recv_buffer->dataptr(), frame_head_output_bytes,
                                                            ccl_stream);
                            pi::tensorlib::ExecutionBackend::GpuStreamWaitFor(ccl_stream, stream_bundle->main_stream);
                            const auto output_slice = frame_embedding_output_tensor->at(0, pos);
                            pi::tensorlib::internal::device_copy::PerformDeviceCopy(recv_buffer, output_slice,
                                                                                    stream_bundle->main_stream);
                            frame_head_recv_events[recv_buffer_index].record(stream_bundle->main_stream);
                            frame_head_recv_event_recorded[recv_buffer_index] = true;
                        }
                    }

                    auto completion_event = pi::tensorlib::ExecutionBackend::CreateEvent(device_gpu, true);
                    completion_event.record(stream_bundle->main_stream);
                    async_step_logger.enqueueFrameProgressEvent(
                        FrameProgressEventLogRequest{forward_completed_idx, std::move(completion_event)});
                    ++forward_completed_idx;
                }

                async_step_logger.enqueueFramePhaseEnd();

                // Start decoding the first backward replay chunk early so it can overlap with action-model work.
                if (runs_frame_head_backward)
                {
                    replay_decode_pipeline.start();
                }

                frame_head_step_exec_plan.releaseTensors();

                if (runs_recurrence)
                {
                    // Only the recurrence owner runs the action model forward/backward.
                    // run forward & backward of main seq model
                    GPUTX_RANGE("fbamtrain::action_model_fwd_bwd");

                    // run action model step graph (fwd & bwd)
                    pi::tensorlib::Executor action_model_executor{action_model_exec_plan, execution_backend,
                                                                  device_ordinal};
                    action_model_executor.execute(allocator_registry, false);
                }
            }

            if (runs_frame_head_backward)
            {
                // Frame head backward (distributed when frame parallel is enabled).
                GPUTX_RANGE("fbamtrain::backward");
                std::shared_ptr<pi::tensorlib::RealTensor> grad_x_tensor{};
                if (runs_recurrence)
                {
                    const auto grad_x_opt = action_model_exec_plan.getRealTensor(*action_model_step_graph.grad_x);
                    if (!grad_x_opt.has_value())
                    {
                        throw std::runtime_error("Failed to get grad_x tensor for frame head backward");
                    }
                    grad_x_tensor = grad_x_opt.value();

                    if (use_frame_parallel)
                    {
                        // Upstream staging copies read grad_x from host memory on the H2D stream.
                        // Ensure those reads start only after action-model writes on main stream are visible.
                        pi::tensorlib::ExecutionBackend::GpuStreamWaitFor(stream_bundle->main_stream,
                                                                          stream_bundle->h2d_stream);
                    }
                }

                auto bwd_copy_complete_event = pi::tensorlib::ExecutionBackend::CreateEvent(device_gpu);
                bool bwd_copy_complete_recorded = false;
                size_t frame_head_upstream_send_sequence = 0;
                size_t frame_head_upstream_recv_sequence = 0;
                size_t bwd_completed_positions = 0;
                auto bwd_start_event = pi::tensorlib::ExecutionBackend::CreateEvent(device_gpu, true);
                bwd_start_event.record(stream_bundle->main_stream);
                async_step_logger.enqueueFramePhaseStart(
                    FramePhaseStartLogRequest{true, seq_len, batch_size, std::move(bwd_start_event)});

                const auto log_backward_progress = [&]
                {
                    if (bwd_completed_positions >= seq_len)
                    {
                        return;
                    }
                    auto completion_event = pi::tensorlib::ExecutionBackend::CreateEvent(device_gpu, true);
                    completion_event.record(stream_bundle->main_stream);
                    async_step_logger.enqueueFrameProgressEvent(
                        FrameProgressEventLogRequest{bwd_completed_positions, std::move(completion_event)});
                    ++bwd_completed_positions;
                };

                // Consume decoded replay chunks. Decoding of the next chunk is launched before compute on the current
                // chunk to overlap CPU decode with queued GPU work.
                while (replay_decode_pipeline.hasPending())
                {
                    DecodedReplayChunk decoded_chunk = replay_decode_pipeline.consumeAndLaunchNext();
                    const size_t chunk_start = decoded_chunk.chunk_start;
                    const size_t chunk_len = decoded_chunk.chunk_len;
                    const auto &replay_chunk_frames = decoded_chunk.frames;

                    for (size_t rel = chunk_len; rel > 0; --rel)
                    {
                        const size_t pos = chunk_start + rel - 1;
                        const size_t chunk_pos = rel - 1;
                        const uint32_t assigned_worker_id =
                            use_frame_parallel ? static_cast<uint32_t>(pos % num_frame_workers) : local_worker_id;

                        if (use_frame_parallel && runs_recurrence && assigned_worker_id != local_worker_id)
                        {
                            // Master rank ships upstream for non-local positions in reverse order.
                            SendUpstreamGradientForPos(
                                pos, worker_id_to_rank.at(assigned_worker_id), &ccl_comm->get(), device_ordinal,
                                frame_head_upstream, frame_head_upstream_send_sequence, grad_x_tensor,
                                frame_head_upstream_bytes, stream_bundle->h2d_stream, ccl_stream);
                            log_backward_progress();
                            continue;
                        }

                        const bool compute_locally = !use_frame_parallel || assigned_worker_id == local_worker_id;
                        if (!compute_locally)
                        {
                            log_backward_progress();
                            continue;
                        }

                        std::shared_ptr<pi::tensorlib::RealTensor> upstream_buffer{};
                        std::optional<size_t> upstream_buffer_index{};
                        if (use_frame_parallel && !runs_recurrence)
                        {
                            // Workers receive upstream on GPU (double-buffered).
                            if (!ccl_comm.has_value())
                            {
                                throw std::runtime_error("CCL communicator missing for frame head upstream recv.");
                            }
                            if (frame_head_upstream.recv_buffers.empty())
                            {
                                throw std::runtime_error("Frame head upstream receive buffers are not initialized.");
                            }
                            upstream_buffer_index =
                                frame_head_upstream_recv_sequence % frame_head_upstream.recv_buffers.size();
                            ++frame_head_upstream_recv_sequence;
                            if (frame_head_upstream.recv_event_recorded[*upstream_buffer_index])
                            {
                                pi::tensorlib::ExecutionBackend::GpuStreamWaitForEvent(
                                    frame_head_upstream.recv_events[*upstream_buffer_index], ccl_stream);
                                frame_head_upstream.recv_event_recorded[*upstream_buffer_index] = false;
                            }
                            upstream_buffer = frame_head_upstream.recv_buffers[*upstream_buffer_index];
                            GPUTX_RANGE("fbamtrain::ccl_recv_upstream");
                            fbamtrain::ccl::RecvBufferAsync(ccl_comm->get(), master_rank, device_ordinal,
                                                            upstream_buffer->dataptr(), frame_head_upstream_bytes,
                                                            ccl_stream);
                            pi::tensorlib::ExecutionBackend::GpuStreamWaitFor(ccl_stream, stream_bundle->main_stream);
                        }

                        if (bwd_copy_complete_recorded)
                        {
                            bwd_copy_complete_event.synchronize();
                        }

                        PrepareCellStatesFromSavedFrames(replay_chunk_frames, cell_states_data, chunk_pos, batch_size,
                                                         rows, cols, model_config.max_code_point);

                        size_t read_buffer_index = 0;
                        {
                            const auto write_buffer_index = current_dst_cell_states_buffer;
                            if (cell_state_ready_recorded[write_buffer_index])
                            {
                                pi::tensorlib::ExecutionBackend::GpuStreamWaitForEvent(
                                    cell_state_ready_events[write_buffer_index], stream_bundle->h2d_stream);
                                cell_state_ready_recorded[write_buffer_index] = false;
                            }
                            const auto write_cell_states_buffer = get_write_cell_states_buffer();
                            pi::tensorlib::internal::device_copy::PerformDeviceCopy(
                                cell_states_cpu, write_cell_states_buffer, stream_bundle->h2d_stream);
                            bwd_copy_complete_event.record(stream_bundle->h2d_stream);
                            bwd_copy_complete_recorded = true;
                            swap_cell_states_buffer();
                            read_buffer_index = write_buffer_index;
                        }

                        pi::tensorlib::GraphExecutionInputDescriptor upstream_input{};
                        if (use_frame_parallel && !runs_recurrence)
                        {
                            upstream_input = pi::tensorlib::GraphExecutionInputDescriptor{
                                .name = "frame_head_upstream",
                                .tensor = upstream_buffer,
                            };
                        }
                        else
                        {
                            auto grad_x_step = grad_x_tensor->at(0, pos);
                            upstream_input = pi::tensorlib::GraphExecutionInputDescriptor{
                                .name = "frame_head_upstream",
                                .tensor = grad_x_step,
                            };
                        }

                        frame_head_bwd_exec_plan.updateInputDescriptors(
                            {pi::tensorlib::GraphExecutionInputDescriptor{.name = "cell_states",
                                                                          .tensor = get_read_cell_states_buffer()},
                             upstream_input});

                        auto frame_head_bwd_executor = std::make_unique<pi::tensorlib::Executor>(
                            frame_head_bwd_exec_plan, execution_backend, device_ordinal);
                        frame_head_bwd_executor->gpuWaitFor(bwd_copy_complete_event);
                        {
                            GPUTX_RANGE("fbamtrain::frame_head_bwd");
                            frame_head_bwd_executor->execute(allocator_registry, false);
                        }

                        cell_state_ready_events[read_buffer_index].record(stream_bundle->main_stream);
                        cell_state_ready_recorded[read_buffer_index] = true;

                        if (upstream_buffer_index.has_value())
                        {
                            frame_head_upstream.recv_events[*upstream_buffer_index].record(stream_bundle->main_stream);
                            frame_head_upstream.recv_event_recorded[*upstream_buffer_index] = true;
                        }

                        log_backward_progress();
                    }
                }

                async_step_logger.enqueueFramePhaseEnd();

                // Do not host-await here: allreduce launch is ordered by a main->CCL stream dependency below.
            }

            if (runs_recurrence)
            {
                const bool do_dump = dump_validation_like && (accum_idx + 1 == grad_accum_steps);
                if (do_dump)
                {
                    // Validation dumps are emitted only by the recurrence owner.
                    std::unordered_map<std::string, std::shared_ptr<pi::tensorlib::RealTensor>>
                        frame_head_grad_tensors{};
                    frame_head_grad_tensors.reserve(frame_head_param_grads.size());
                    for (const auto &[name, trace_tensor] : frame_head_param_grads)
                    {
                        if (const auto real_tensor_opt = frame_head_bwd_exec_plan.getRealTensor(trace_tensor);
                            real_tensor_opt.has_value())
                        {
                            frame_head_grad_tensors.emplace("grad_" + name, real_tensor_opt.value());
                        }
                    }
                    std::unordered_map<std::string, std::shared_ptr<pi::tensorlib::RealTensor>> param_tensors{};
                    param_tensors.reserve(frame_head_parameter_descriptors.size() +
                                          action_model_parameter_descriptors.size());
                    for (const auto &entry : frame_head_parameter_descriptors)
                    {
                        param_tensors.emplace("param_" + entry.name, entry.tensor);
                    }
                    for (const auto &entry : action_model_parameter_descriptors)
                    {
                        param_tensors.emplace("param_" + entry.name, entry.tensor);
                    }
                    for (const auto &[name, tensor] : frame_head_grad_tensors)
                    {
                        param_tensors.emplace(name, tensor);
                    }
                    std::vector<ValidationDumpTraceEntry> trace_entries{
                        {
                            .name = "output",
                            .tensor = action_model_step_graph.action_model_output.output,
                            .required = true,
                        },
                        {
                            .name = "h_n",
                            .tensor = action_model_step_graph.action_model_output.h_n,
                        },
                        {
                            .name = "c_n",
                            .tensor = action_model_step_graph.action_model_output.c_n,
                        },
                        {
                            .name = "loss",
                            .tensor = action_model_step_graph.loss_mean,
                        },
                        {
                            .name = "frame_embeddings",
                            .tensor = action_model_step_graph.frame_embeddings_host,
                        },
                    };
                    trace_entries.reserve(trace_entries.size() + 8);
                    const auto add_optional =
                        [&trace_entries](const char *name, const std::optional<pi::tensorlib::TraceTensor> &tensor)
                    {
                        if (tensor.has_value())
                        {
                            trace_entries.emplace_back(ValidationDumpTraceEntry{.name = name, .tensor = *tensor});
                        }
                    };
                    add_optional("logits", action_model_step_graph.projected_output);
                    add_optional("grad_x", action_model_step_graph.grad_x);
                    add_optional("grad_w_ih", action_model_step_graph.grad_w_ih);
                    add_optional("grad_w_hh", action_model_step_graph.grad_w_hh);
                    add_optional("grad_b_ih", action_model_step_graph.grad_b_ih);
                    add_optional("grad_b_hh", action_model_step_graph.grad_b_hh);
                    add_optional("grad_h0", action_model_step_graph.grad_h0);
                    add_optional("grad_c0", action_model_step_graph.grad_c0);
                    CreateValidationDump(action_model_exec_plan, trace_entries, param_tensors,
                                         FormatValidationDumpPath(validation_dump_file, step));
                }

                {
                    if (!loss_accumulation_ring)
                    {
                        throw std::runtime_error("Loss accumulation ring buffer is not initialized.");
                    }
                    loss_accumulation_ring->AccumulateMicroStepLoss(loss_mean_tensor, ce_stream);
                }

                action_model_exec_plan.releaseTensors();
            }
        }

        if (loss_accumulation_ring)
        {
            loss_accumulation_ring->FinalizeStep();
        }

        bool commit_current_step = true;

        if (runs_frame_head_backward)
        {
            GPUTX_RANGE("fbamtrain::frame_head_grad_allreduce");
            if (use_frame_parallel)
            {
                if (!ccl_comm.has_value())
                {
                    throw std::runtime_error("CCL communicator missing for frame-head gradient allreduce.");
                }
                // The parent frame-head communicator spans the whole addressable training set in CCL-backed
                // configurations, so this single reduction covers both the frame axis and the DDP axis.
                AllReduceCclFrameHeadGradients(ccl_comm->get(), device_ordinal, frame_head_grad_allreduce_buffers,
                                               stream_bundle->main_stream, ccl_stream);
            }
            else if (use_ccl_ddp)
            {
                if (!recurrence_master_subset_comm.has_value())
                {
                    throw std::runtime_error(
                        "Recurrence-master subset communicator missing for frame-head DDP allreduce.");
                }
                // DDP without frame-head parallel still needs frame-head gradients summed across recurrence masters.
                AllReduceCclFrameHeadGradients(recurrence_master_subset_comm.value(), device_ordinal,
                                               frame_head_grad_allreduce_buffers, stream_bundle->main_stream,
                                               ccl_stream);
            }
            else if (use_ftccl_ddp && runs_recurrence)
            {
                assert(ftccl_state.has_value());

                // With no CCL frame axis, FTCCL must reduce the frame-head gradients before the optimizer can advance
                // deterministic shared state.
                pi::tensorlib::ExecutionBackend::SynchronizeStreamBundle(*stream_bundle);
                commit_current_step =
                    AllReduceFtCclFrameHeadGradients(*ftccl_state, frame_head_grad_allreduce_buffers);
                pi::tensorlib::ExecutionBackend::SynchronizeStreamBundle(*stream_bundle);
            }
        }

        if (use_ccl_ddp && runs_recurrence)
        {
            GPUTX_RANGE("fbamtrain::action_model_grad_allreduce");
            if (!recurrence_master_subset_comm.has_value())
            {
                throw std::runtime_error("Recurrence-master subset communicator missing for action-model allreduce.");
            }
            if (!action_model_gradient_allocation_space || action_model_grad_init.allreduce_elements == 0)
            {
                throw std::runtime_error("Action-model gradient buffer missing for CCL-backed DDP allreduce.");
            }

            // The action-model gradients only exist on recurrence masters, so they use the derived subset
            // communicator rather than the parent frame-head communicator.
            pi::tensorlib::ExecutionBackend::GpuStreamWaitFor(stream_bundle->main_stream, ccl_stream);
            fbamtrain::ccl::AllReduceAsync(
                recurrence_master_subset_comm.value(), device_ordinal,
                action_model_gradient_allocation_space->dataptr(), action_model_gradient_allocation_space->dataptr(),
                action_model_grad_init.allreduce_elements, ToCclDataType(action_model_grad_init.dtype), ccl_stream);
            pi::tensorlib::ExecutionBackend::GpuStreamWaitFor(ccl_stream, stream_bundle->main_stream);
        }

        if (commit_current_step && use_ftccl_ddp && runs_recurrence)
        {
            GPUTX_RANGE("fbamtrain::ftccl_action_model_grad_allreduce");
            assert(ftccl_state.has_value());

            // FTCCL operates directly on the CUDA buffer. Make all queued gradient writes visible before entering the
            // retrying collective and keep optimizer execution behind the completed reduce.
            pi::tensorlib::ExecutionBackend::SynchronizeStreamBundle(*stream_bundle);
            const bool reduce_completed = AllReduceFtCclActionModelGradients(
                *ftccl_state, action_model_gradient_allocation_space, action_model_grad_init.allreduce_elements,
                action_model_grad_init.dtype);
            commit_current_step = reduce_completed;
            pi::tensorlib::ExecutionBackend::SynchronizeStreamBundle(*stream_bundle);
        }

        if (use_frame_parallel && use_ftccl_ddp)
        {
            if (!ccl_comm.has_value())
            {
                throw std::runtime_error("CCL communicator missing for FTCCL step commit decision.");
            }
            commit_current_step = BroadcastCclStepCommitDecision(
                ccl_comm->get(), device_ordinal, master_rank, commit_current_step, ftccl_step_commit_signal_cpu,
                ftccl_step_commit_signal_gpu, stream_bundle->main_stream, ccl_stream);
        }

        if (!commit_current_step)
        {
            LOG(WARN) << "Dropping training step " << step
                      << " because FTCCL DDP gradient allreduce lost quorum.";

            auto dropped_step_end_event = pi::tensorlib::ExecutionBackend::CreateEvent(device_gpu, true);
            dropped_step_end_event.record(stream_bundle->main_stream);
            async_step_logger.enqueue(StepTimingLogRequest{step, false, std::move(step_start_event),
                                                           std::move(dropped_step_end_event), true});

            if (ftccl_state.has_value())
            {
                if (!ftccl_shared_state_view.has_value())
                {
                    throw std::runtime_error("FTCCL shared state view missing for dropped-step recovery.");
                }
                // The optimizer has not run, so the current shared-state revision still names the last committed
                // state. Resynchronize that revision before retrying with a repaired peer group.
                while (!PrepareFtCclCollectivePhase(*ftccl_state, true, "dropped-step recovery synchronization"))
                {
                }
                const auto recovery_sync = SynchronizeFtCclSharedState(*ftccl_state, *ftccl_shared_state_view,
                                                                        "FTCCL dropped-step recovery synchronization");
                if (recovery_sync.rx_bytes != 0)
                {
                    LOG(INFO) << "FTCCL dropped-step recovery received shared state; local frame-parallel state will "
                                 "be redistributed.";
                }
            }

            if (use_frame_parallel && use_ftccl_ddp)
            {
                if (!frame_parallel_ccl_state_view.has_value())
                {
                    throw std::runtime_error("Frame-parallel CCL state view missing for dropped-step recovery.");
                }
                BroadcastCclTensorState(*frame_parallel_ccl_state_view, &ccl_comm->get(), device_ordinal, master_rank,
                                        stream_bundle->main_stream, ccl_stream,
                                        "Dropped-step frame-parallel state");
            }
            continue;
        }

        if (runs_recurrence && action_model_optimizer)
        {
            GPUTX_RANGE("fbamtrain::action_model_optim_step");
            action_model_optimizer->step();
        }

        // After the action model optimizer step, broadcast updated action model weights from the master.
        pi::tensorlib::gputx::ScopedRange *gpu_tx_range = nullptr;
        if (use_frame_parallel)
        {
            // Broadcast updated recurrent weights only after action-model optimizer writes are visible.
            pi::tensorlib::ExecutionBackend::GpuStreamWaitFor(stream_bundle->main_stream, ccl_stream);
            gpu_tx_range = MAKE_GPUTX_RANGE("fbamtrain::action_model_param_broadcast");
            BroadcastParameters(action_model_parameter_descriptors, &ccl_comm->get(), device_ordinal, master_rank,
                                ccl_stream);
        }

        {
            GPUTX_RANGE("fbamtrain::frame_head_optim_step");
            frame_head_optimizer->step();
        }

        if (use_frame_parallel)
        {
            assert(gpu_tx_range != nullptr);

            // Ensure updated recurrent weights are visible before starting the next step.
            pi::tensorlib::ExecutionBackend::GpuStreamWaitFor(ccl_stream, stream_bundle->main_stream);

            delete gpu_tx_range; // end of "fbamtrain::action_model_param_broadcast" range
        }

        if (ftccl_state.has_value())
        {
            assert(runs_recurrence);

            // Parameters and optimizer state have advanced deterministically from the last synchronized revision.
            // FTCCL expects the next shared-state sync to advertise exactly this +1 revision.
            ++ftccl_state->shared_state_revision;
            step = ftccl_state->shared_state_revision;
            pi::tensorlib::ExecutionBackend::SynchronizeStreamBundle(*stream_bundle);
            while (!PrepareFtCclCollectivePhase(*ftccl_state, true, "post-step shared-state synchronization"))
            {
            }
            const auto sync_outcome = SynchronizeFtCclSharedState(*ftccl_state, *ftccl_shared_state_view,
                                                                  "FTCCL post-step shared-state synchronization");
            if (sync_outcome.rx_bytes != 0)
            {
                LOG(WARN) << "FTCCL post-step shared-state synchronization received bytes after local deterministic "
                             "advancement; local frame-parallel state will be redistributed.";
            }
            step = ftccl_state->shared_state_revision;
        }
        else
        {
            step++;
        }

        if (use_frame_parallel && use_ftccl_ddp)
        {
            if (!ccl_comm.has_value())
            {
                throw std::runtime_error("CCL communicator missing for post-step frame-parallel state broadcast.");
            }
            if (!frame_parallel_ccl_state_view.has_value())
            {
                throw std::runtime_error("Frame-parallel CCL state view missing for post-step broadcast.");
            }
            BroadcastCclTensorState(*frame_parallel_ccl_state_view, &ccl_comm->get(), device_ordinal, master_rank,
                                    stream_bundle->main_stream, ccl_stream, "Post-step frame-parallel state");
        }
        if (dataset_cursor_committer)
        {
            dataset_cursor_committer->commit(
                fbamtrain::DatasetCursorSnapshot::FromIterator(step, active_iterator, run_config.train_sequence_length));
        }

        // NOTE: ALLOC_100 is deallocated implicitly here, as for the next step all previous CREATE_TENSOR results will
        // be free-ed

        // Record an async step-end marker on main stream.
        {
            auto step_end_event = pi::tensorlib::ExecutionBackend::CreateEvent(device_gpu, true);
            step_end_event.record(stream_bundle->main_stream);
            async_step_logger.enqueue(StepTimingLogRequest{step, emit_train_metrics_json, std::move(step_start_event),
                                                           std::move(step_end_event)});
        }

        const auto &gpu_allocator = allocator_registry.getAllocator(pi::tensorlib::DeviceType::GPU);
        const auto gpu_metrics = gpu_allocator.getMetrics();
        LOG(INFO) << "Allocator metrics (allocator=GPU): {Peak in-use: "
                  << fbamtrain::formatutils::FormatBytes(gpu_metrics.peak_in_use_bytes)
                  << ", Current in-use: " << fbamtrain::formatutils::FormatBytes(gpu_metrics.in_use_bytes)
                  << ", Cached Bytes: " << fbamtrain::formatutils::FormatBytes(gpu_metrics.cached_bytes)
                  << ", Reserved Bytes: " << fbamtrain::formatutils::FormatBytes(gpu_metrics.reserved_bytes) << "}";

        const auto &cpu_allocator = allocator_registry.getAllocator(pi::tensorlib::DeviceType::CPU);
        const auto cpu_metrics = cpu_allocator.getMetrics();
        LOG(INFO) << "Allocator metrics (allocator=CPU): {Peak in-use: "
                  << fbamtrain::formatutils::FormatBytes(cpu_metrics.peak_in_use_bytes)
                  << ", Current in-use: " << fbamtrain::formatutils::FormatBytes(cpu_metrics.in_use_bytes)
                  << ", Cached Bytes: " << fbamtrain::formatutils::FormatBytes(cpu_metrics.cached_bytes)
                  << ", Reserved Bytes: " << fbamtrain::formatutils::FormatBytes(cpu_metrics.reserved_bytes) << "}";

        if (periodic_validation_enabled && step % run_config.validation_interval == 0)
        {
            if (!validation_action_model_exec_plan.has_value())
            {
                throw std::runtime_error("Validation execution plan is not initialized.");
            }

            LOG(INFO) << "Running validation at step " << step;
            const float validation_loss = RunValidation(
                run_config, run_config.validation_sequence_length, grad_accum_steps, batch_size, rows, cols,
                vocab_size_raw, device_gpu, device_ordinal, tokenizer, valid_iterator, execution_backend,
                allocator_registry, frame_head_step_exec_plan, frame_head_output,
                validation_frame_embedding_output_tensor, validation_action_targets_tensor,
                *validation_action_model_exec_plan, validation_loss_mean_tensor, validation_loss_mean_cpu);

            if (report_validation_metrics)
            {
                LOG(INFO) << "Step " << step << "; loss/validation = " << validation_loss;
                if (ShouldEmitMetricsJson())
                {
                    std::cout << std::setprecision(10) << "{\"step\":" << step
                              << ",\"loss/validation\":" << validation_loss << "}\n"
                              << std::flush;
                }
            }
        }

        // save checkpoint if enabled
        if (checkpointing_enabled && (!parallel_participant || is_master) &&
            step % run_config.checkpointing.checkpoint_interval == 0)
        {
            if (device_gpu.device_type == pi::tensorlib::DeviceType::GPU)
            {
                pi::tensorlib::ExecutionBackend::SynchronizeStreamBundle(*stream_bundle);
            }

            std::map<std::string, std::shared_ptr<pi::tensorlib::RealTensor>> checkpoint_tensors{};

            frame_head_module.appendParamState(checkpoint_tensors, optimizer_params, "Checkpoint frame-head state");
            action_model_module.appendParamState(checkpoint_tensors, optimizer_params, "Checkpoint action-model state");

            frame_head_optimizer->appendOptimState(checkpoint_tensors, "optim/frame_head");
            if (action_model_optimizer)
            {
                action_model_optimizer->appendOptimState(checkpoint_tensors, "optim/action_model");
            }

            const auto dataset_state = active_iterator.checkpoint();
            const auto checkpoint_path = checkpoint_manager.save(step, checkpoint_tensors, dataset_state);
            LOG(INFO) << "Saved checkpoint at step_count=" << step << " to " << checkpoint_path;
        }
    }

    async_step_logger.stop();
    cell_states_cpu->free();

    if (ftccl_state.has_value())
    {
        CheckFtCclResult(fbamtrain::ftccl::DestroyCommunicator(ftccl_state->communicator),
                         "FTCCL communicator destruction");
        ftccl_state.reset();
    }

    // perform teardown
    if (needs_ccl_communicator)
    {
        if (is_master || worker_id.has_value())
        {
            // Ensure all async compute/CCL work has finished before communicator teardown.
            pi::tensorlib::ExecutionBackend::SynchronizeStreamBundle(*stream_bundle);

            if (recurrence_master_subset_comm.has_value())
            {
                // Derived communicators are independent NCCL communicators and must be torn down before the parent.
                fbamtrain::ccl::Destroy(recurrence_master_subset_comm.value());
                recurrence_master_subset_comm.reset();
            }
            fbamtrain::rdvz::PerformTeardown(parallel_config.value(), is_master, worker_id);
        }
    }
}
