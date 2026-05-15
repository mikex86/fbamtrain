#include "passes.h"

#include "passes/pass_utils.h"
#include <algorithm>
#include <any>
#include <execution_backend.h>
#include <handles.h>
#include <iostream>
#include <kernels/kernel_binaries.h>
#include <limits>
#include <optional>
#include <stream_descriptor.h>
#include <stream_utils.h>
#include <string>
#include <unordered_map>
#include <vector>

struct CutlassMhaBwdWorkspaceLayout
{
    uint64_t bytes{0};
    uint64_t stride_bh{0};
    uint64_t offset_gq{0};
    uint64_t gq_entries_per_bh{0};
    uint64_t gradq_temp_bytes{0};
    uint32_t num_splits_key{1};
};

struct CutlassMhaBwdWorkspaceCacheEntry
{
    std::shared_ptr<pi::tensorlib::RealTensor> workspace;
    std::shared_ptr<pi::tensorlib::RealTensor> template_tensor;
    CutlassMhaBwdWorkspaceLayout layout;
    int device_ordinal{-1};
    int stream_id{0};
};

struct CutlassMhaBwdWorkspaceState
{
    std::shared_ptr<CutlassMhaBwdWorkspaceCacheEntry> cache_entry;
    pi::tensorlib::GpuStreamDescriptor cleanup_stream_desc{};
};

struct CutlassMhaKernelSet
{
    const kernel_bin_t<kernel_meta_mha_cutlass_t> *bf16_kernel;
    const kernel_bin_t<kernel_meta_mha_cutlass_t> *fp16_kernel;
};

struct CutlassMhaBwdKernelSet
{
    const kernel_bin_t<kernel_meta_mha_cutlass_bwd_t> *bf16_kernel;
    const kernel_bin_t<kernel_meta_mha_cutlass_bwd_t> *fp16_kernel;
};

struct FlashScratchState
{
    std::shared_ptr<pi::tensorlib::RealTensor> lse{};
    std::shared_ptr<pi::tensorlib::RealTensor> global{};
};

struct TritonMhaScratchState
{
    std::shared_ptr<pi::tensorlib::RealTensor> global{};
};

static bool IsEnvSet(const char *name) { return std::getenv(name) != nullptr; }

static bool IsMhaKernelLogEnabled() { return IsEnvFlagEnabled("FBAMTRAIN_LOG_MHA_KERNEL"); }
static bool IsMhaShapeLogEnabled() { return IsEnvFlagEnabled("FBAMTRAIN_LOG_MHA_SHAPES"); }

static std::shared_ptr<pi::tensorlib::RealTensor>
AllocateMhaScratch(const std::initializer_list<uint64_t> &dims, const pi::tensorlib::DataType dtype,
                   const pi::tensorlib::Device &device,
                   const pi::tensorlib::GpuStreamDescriptor &compute_stream_descriptor, const bool pinned)
{
    return pi::tensorlib::RealTensor::AllocateOnStream(dims, dtype, device, compute_stream_descriptor, pinned);
}

static pi::tensorlib::ComputeKernelDescriptor CreateFillZerosKernelDescriptor()
{
    kernel_bin_t<kernel_meta_elementwise_t> kernel = kfill_zeros;

    return pi::tensorlib::ComputeKernelDescriptor{
        .kernel_name = "fill_zeros",
        .function_name = kfill_zeros.function_name,
        .expected_arg_count = kernel.arg_count,
        .argument_provider = [kernel](const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                                      const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
            -> pi::tensorlib::KernelLaunchArguments
        {
            if (inputs.size() != 1)
            {
                throw std::runtime_error("fill_zeros expects exactly one input tensor");
            }
            if (outputs.size() != 1)
            {
                throw std::runtime_error("fill_zeros expects exactly one output tensor");
            }
            if (inputs[0]->id() != outputs[0]->id())
            {
                throw std::runtime_error("fill_zeros is inplace and requires identical input/output tensors");
            }

            const auto &out = outputs[0];
            if (out->device().device_type != pi::tensorlib::DeviceType::GPU)
            {
                throw std::runtime_error("fill_zeros currently supports GPU tensors only");
            }

            const auto device_ordinal = ValidateSameDeviceOrdinal("fill_zeros", {out});
            const size_t bytes_per_element = pi::tensorlib::GetDataTypeSize(out->dtype());
            const uint64_t total_bytes_64 = out->shape().numel() * bytes_per_element;
            if (total_bytes_64 > std::numeric_limits<uint32_t>::max())
            {
                throw std::runtime_error("fill_zeros tensor is too large for the current kernel implementation");
            }
            const auto total_bytes = static_cast<uint32_t>(total_bytes_64);

            void *out_ptr = out->dataptr();
            pi::tensorlib::KernelLaunchArguments arguments{.args = {out_ptr, total_bytes, static_cast<void *>(nullptr)},
                                                           .grid_dim_x = (total_bytes + kernel.meta.block_size - 1) /
                                                                         kernel.meta.block_size,
                                                           .grid_dim_y = 1,
                                                           .grid_dim_z = 1,
                                                           .block_dim_x = TRITON_WARP_SIZE * kernel.num_warps,
                                                           .block_dim_y = 1,
                                                           .block_dim_z = 1,
                                                           .shared_mem_bytes = kernel.shared_mem_bytes,
                                                           .device_ordinal = device_ordinal};
            return arguments;
        },
        .cuda_descriptor = pi::tensorlib::MakeCudaKernelDescriptor(kernel.data, kernel.size),
        .hip_descriptor = pi::tensorlib::MakeHipKernelDescriptor(kernel.data, kernel.size),
    };
}

static bool IsMhaDebugEnabled() { return IsEnvSet("FBAMTRAIN_DEBUG_MHA"); }

static void LogMhaKernelSelection(const char *op, const char *backend, const std::string &kernel_name)
{
    if (!IsMhaKernelLogEnabled())
    {
        return;
    }
    std::cout << "[MHA] Selected " << op << " kernel (" << backend << "): " << kernel_name << '\n';
}

static void LogMhaShape(const char *op, const char *backend, const uint32_t batch_size, const uint32_t n_head,
                        const uint32_t n_ctx, const uint32_t head_size)
{
    if (!IsMhaShapeLogEnabled())
    {
        return;
    }
    std::cout << "[MHA] " << op << " shape (" << backend << "): B=" << batch_size << " H=" << n_head << " T=" << n_ctx
              << " HS=" << head_size << '\n';
}

enum class MhaBackendPreference
{
    Flash,
    Cutlass,
    Triton,
};

static constexpr float kSoftmaxScaleLog2e = 1.4426950408889634f;

namespace
{
    struct CutlassMhaBwdWorkspaceKey
    {
        int device_ordinal{};
        int stream_id{};
        CutlassMhaBwdWorkspaceLayout layout{};
    };

    bool operator==(const CutlassMhaBwdWorkspaceKey &lhs, const CutlassMhaBwdWorkspaceKey &rhs)
    {
        return lhs.device_ordinal == rhs.device_ordinal && lhs.stream_id == rhs.stream_id &&
               lhs.layout.bytes == rhs.layout.bytes && lhs.layout.stride_bh == rhs.layout.stride_bh &&
               lhs.layout.offset_gq == rhs.layout.offset_gq &&
               lhs.layout.gq_entries_per_bh == rhs.layout.gq_entries_per_bh &&
               lhs.layout.gradq_temp_bytes == rhs.layout.gradq_temp_bytes &&
               lhs.layout.num_splits_key == rhs.layout.num_splits_key;
    }

    struct CutlassMhaBwdWorkspaceKeyHash
    {
        size_t operator()(const CutlassMhaBwdWorkspaceKey &key) const
        {
            auto combine = [](const size_t seed, const size_t value)
            { return seed ^ (value + 0x9e3779b9u + (seed << 6u) + (seed >> 2u)); };
            size_t h = 0;
            h = combine(h, std::hash<int>{}(key.device_ordinal));
            h = combine(h, std::hash<int>{}(key.stream_id));
            h = combine(h, std::hash<uint64_t>{}(key.layout.bytes));
            h = combine(h, std::hash<uint64_t>{}(key.layout.stride_bh));
            h = combine(h, std::hash<uint64_t>{}(key.layout.offset_gq));
            h = combine(h, std::hash<uint64_t>{}(key.layout.gq_entries_per_bh));
            h = combine(h, std::hash<uint64_t>{}(key.layout.gradq_temp_bytes));
            h = combine(h, std::hash<uint32_t>{}(key.layout.num_splits_key));
            return h;
        }
    };

    std::unordered_map<CutlassMhaBwdWorkspaceKey, std::shared_ptr<CutlassMhaBwdWorkspaceCacheEntry>,
                       CutlassMhaBwdWorkspaceKeyHash> &
    GetCutlassMhaBwdWorkspaceCache()
    {
        static std::unordered_map<CutlassMhaBwdWorkspaceKey, std::shared_ptr<CutlassMhaBwdWorkspaceCacheEntry>,
                                  CutlassMhaBwdWorkspaceKeyHash>
            cache_holder;
        return cache_holder;
    }

    void FillCutlassBwdWorkspaceTemplate(uint8_t *host_bytes, const CutlassMhaBwdWorkspaceLayout &layout);

    std::shared_ptr<CutlassMhaBwdWorkspaceCacheEntry>
    GetCutlassMhaBwdWorkspace(const pi::tensorlib::Device &device,
                              const pi::tensorlib::GpuStreamDescriptor &compute_stream_descriptor,
                              const CutlassMhaBwdWorkspaceLayout &layout)
    {
        if (layout.bytes == 0)
        {
            return nullptr;
        }
        const int compute_stream_id = compute_stream_descriptor.getStreamId();
        CutlassMhaBwdWorkspaceKey key{device.ordinal, compute_stream_id, layout};
        auto &cache = GetCutlassMhaBwdWorkspaceCache();
        if (const auto it = cache.find(key); it != cache.end())
        {
            return it->second;
        }

        const uint64_t workspace_elems = (layout.bytes + 3) / 4;
        const auto workspace_tensor = pi::tensorlib::RealTensor::AllocateOnStream(
            {workspace_elems}, pi::tensorlib::DataType::FLOAT32, device, compute_stream_descriptor, false);

        const auto host_template =
            pi::tensorlib::RealTensor::Allocate({workspace_elems}, pi::tensorlib::DataType::FLOAT32,
                                                pi::tensorlib::Device{pi::tensorlib::DeviceType::CPU, 0}, false);
        auto *host_bytes = static_cast<uint8_t *>(host_template->dataptr());
        std::fill_n(host_bytes, layout.bytes, 0);

        // Cutlass GradQTempStorage expects lock=0 and counter=-1 for each tile.
        FillCutlassBwdWorkspaceTemplate(host_bytes, layout);

        const auto template_tensor = pi::tensorlib::RealTensor::AllocateOnStream(
            {workspace_elems}, pi::tensorlib::DataType::FLOAT32, device, compute_stream_descriptor, false);
        template_tensor->storage()->copyFrom(*host_template->storage(), compute_stream_descriptor);
        workspace_tensor->storage()->copyFrom(*template_tensor->storage(), compute_stream_descriptor);

        auto entry = std::make_shared<CutlassMhaBwdWorkspaceCacheEntry>(CutlassMhaBwdWorkspaceCacheEntry{
            .workspace = workspace_tensor,
            .template_tensor = template_tensor,
            .layout = layout,
            .device_ordinal = device.ordinal,
            .stream_id = compute_stream_id,
        });
        cache.emplace(key, entry);
        return entry;
    }

    void FillCutlassBwdWorkspaceTemplate(uint8_t *host_bytes, const CutlassMhaBwdWorkspaceLayout &layout)
    {
        if (layout.gq_entries_per_bh == 0 || layout.gradq_temp_bytes == 0 || layout.stride_bh == 0)
        {
            return;
        }
        const uint64_t total_bh = layout.bytes / (layout.stride_bh * 4);
        for (uint64_t bh = 0; bh < total_bh; ++bh)
        {
            const uint64_t bh_base = (layout.stride_bh * bh + layout.offset_gq) * sizeof(float);
            for (uint64_t entry = 0; entry < layout.gq_entries_per_bh; ++entry)
            {
                const uint64_t entry_base = bh_base + entry * layout.gradq_temp_bytes;
                if (entry_base + sizeof(int32_t) * 2 <= layout.bytes)
                {
                    const uint64_t counter_offset = entry_base + sizeof(int32_t);
                    host_bytes[counter_offset + 0] = 0xFF;
                    host_bytes[counter_offset + 1] = 0xFF;
                    host_bytes[counter_offset + 2] = 0xFF;
                    host_bytes[counter_offset + 3] = 0xFF;
                }
            }
        }
    }
} // namespace

static MhaBackendPreference GetMhaBackendPreference()
{
    const auto env = GetEnvValue("FBAMTRAIN_PREFER_MHA_BACKEND");
#if !PI_TENSORLIB_ENABLE_CUDA
    (void)env;
    return MhaBackendPreference::Triton;
#else
    if (!env.has_value())
    {
        return MhaBackendPreference::Flash;
    }
    if (*env == "flash")
    {
        return MhaBackendPreference::Flash;
    }
    if (*env == "cutlass")
    {
        return MhaBackendPreference::Cutlass;
    }
    if (*env == "triton")
    {
        return MhaBackendPreference::Triton;
    }
    throw std::runtime_error("FBAMTRAIN_PREFER_MHA_BACKEND must be one of: flash, cutlass, triton");
#endif
}

static uint64_t AlignUp(const uint64_t value, const uint64_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

static void RequireContiguousBTHS(const std::shared_ptr<pi::tensorlib::RealTensor> &tensor, const std::string &name,
                                  const uint32_t n_ctx, const uint32_t n_head, const uint32_t head_size,
                                  const std::string &stride_error)
{
    if (tensor->shape().ndims() != 4)
    {
        throw std::runtime_error(name + " tensor must have 4 dimensions");
    }
    const auto &strides = tensor->strides();
    const uint64_t expected_t = static_cast<uint64_t>(n_head) * head_size;
    const uint64_t expected_h = head_size;
    const uint64_t expected_z = static_cast<uint64_t>(n_ctx) * n_head * head_size;
    if (strides[3] != 1 || strides[2] != expected_h || strides[1] != expected_t || strides[0] != expected_z)
    {
        throw std::runtime_error(stride_error);
    }
}

static uint32_t RequirePackedBTHS(const std::shared_ptr<pi::tensorlib::RealTensor> &tensor, const std::string &name,
                                  const uint32_t n_ctx, const uint32_t head_size, const std::string &stride_error)
{
    if (tensor->shape().ndims() != 4)
    {
        throw std::runtime_error(name + " tensor must have 4 dimensions");
    }
    const auto &strides = tensor->strides();
    if (strides[3] != 1 || strides[2] != head_size)
    {
        throw std::runtime_error(stride_error);
    }
    const uint64_t expected_z = static_cast<uint64_t>(n_ctx) * strides[1];
    if (strides[0] != expected_z)
    {
        throw std::runtime_error(stride_error);
    }
    return static_cast<uint32_t>(strides[1]);
}

static void RequireContiguousBHT(const std::shared_ptr<pi::tensorlib::RealTensor> &tensor, const std::string &name,
                                 const uint32_t n_ctx, const uint32_t n_head, const std::string &stride_error)
{
    if (tensor->shape().ndims() != 3)
    {
        throw std::runtime_error(name + " tensor must have 3 dimensions");
    }
    const auto &strides = tensor->strides();
    const uint64_t expected_t = 1;
    const uint64_t expected_h = n_ctx;
    const uint64_t expected_z = static_cast<uint64_t>(n_head) * n_ctx;
    if (strides[2] != expected_t || strides[1] != expected_h || strides[0] != expected_z)
    {
        throw std::runtime_error(stride_error);
    }
}

static void RequireContiguousHeadDim(const std::shared_ptr<pi::tensorlib::RealTensor> &tensor, const std::string &name)
{
    if (tensor->shape().ndims() != 4)
    {
        throw std::runtime_error(name + " tensor must have 4 dimensions");
    }
    if (tensor->strides()[3] != 1)
    {
        throw std::runtime_error(name + " tensor must have contiguous head dimension");
    }
}

// Workspace layout matches Cutlass MHA bwd params in:
// - kernel_gen/kernel_gen/third_party/cutlass/examples/41_fused_multi_head_attention/kernel_backward.h
// - kernel_gen/kernel_gen/kernels/cutlass_codegen/mha_cutlass_bwd_kernel.cu
// The buffer stores [gk | gv | gq] per (batch, head), aligned to 128 bits (4 floats).
static CutlassMhaBwdWorkspaceLayout ComputeCutlassMhaBwdWorkspaceLayout(const uint32_t batch_size,
                                                                        const uint32_t n_head, const uint32_t n_ctx,
                                                                        const uint32_t head_size,
                                                                        const kernel_meta_mha_cutlass_bwd_t &meta)
{
    const uint64_t block_i = meta.block_size_i;
    const uint64_t block_j = meta.block_size_j;
    const uint64_t gradq_temp_bytes = meta.gradq_temp_bytes;
    const uint64_t num_blocks = (n_ctx + block_i - 1) / block_i;
    const uint64_t num_cols = (head_size + block_j - 1) / block_j;
    const uint64_t num_key_blocks = (n_ctx + block_j - 1) / block_j;
    
    // Cutlass clamps split-K to at most 8 (see kernel_backward.h getNumSplitsKey()).
    uint32_t num_splits_key = static_cast<uint32_t>(std::max<uint64_t>(1, std::min<uint64_t>(num_key_blocks, 8)));
#if PI_TENSORLIB_ENABLE_CUDA && (NV_KERNEL_ARCH == 90)
    // Hopper path in this CUTLASS backward integration is currently numerically unstable with split-K>1.
    // Keep CUTLASS enabled, but force single-split accumulation for correctness.
    num_splits_key = 1;
#endif
    const uint64_t gradq_temp_elems = (gradq_temp_bytes + 3) / 4;

    const bool output_in_rf = head_size <= block_i;
    uint64_t workspace_elements_gk = 0;
    uint64_t workspace_elements_gv = 0;
    if (!output_in_rf)
    {
        const uint64_t aligned_keys = AlignUp(n_ctx, block_j);
        const uint64_t aligned_head = AlignUp(head_size, block_i);
        workspace_elements_gk = num_splits_key * aligned_keys * aligned_head;
        workspace_elements_gv = num_splits_key * aligned_keys * aligned_head;
    }

    const uint64_t workspace_elements_gq = num_blocks * num_cols * gradq_temp_elems;
    const uint64_t workspace_stride_bh =
        AlignUp(workspace_elements_gk + workspace_elements_gv + workspace_elements_gq, 4);
    const uint64_t workspace_bytes =
        workspace_stride_bh * static_cast<uint64_t>(batch_size) * static_cast<uint64_t>(n_head) * 4;

    return CutlassMhaBwdWorkspaceLayout{.bytes = workspace_bytes,
                                        .stride_bh = workspace_stride_bh,
                                        .offset_gq = workspace_elements_gk + workspace_elements_gv,
                                        .gq_entries_per_bh = num_blocks * num_cols,
                                        .gradq_temp_bytes = gradq_temp_bytes,
                                        .num_splits_key = num_splits_key};
}

static bool RequiresCutlassBwdWorkspaceReset(const CutlassMhaBwdWorkspaceLayout &layout)
{
    return layout.bytes > 0 && layout.num_splits_key > 1 && layout.gradq_temp_bytes > 0;
}

enum class FlashBwdKernelKind
{
    DotDoO,
    Main,
    ConvertDQ,
};

struct FlashMhaBwdScratchState
{
    std::shared_ptr<pi::tensorlib::RealTensor> dq_accum;
    std::shared_ptr<pi::tensorlib::RealTensor> dsoftmax_sum;
};

struct FlashMhaBwdScratchKey
{
    int device_ordinal{};
    int stream_id{};
};

struct FlashMhaBwdScratchKeyHash
{
    size_t operator()(const FlashMhaBwdScratchKey &key) const
    {
        return (static_cast<size_t>(key.device_ordinal) << 32) ^ static_cast<size_t>(key.stream_id);
    }
};

static bool operator==(const FlashMhaBwdScratchKey &lhs, const FlashMhaBwdScratchKey &rhs)
{
    return lhs.device_ordinal == rhs.device_ordinal && lhs.stream_id == rhs.stream_id;
}

using FlashMhaBwdScratchPool =
    std::unordered_map<FlashMhaBwdScratchKey, std::shared_ptr<FlashMhaBwdScratchState>, FlashMhaBwdScratchKeyHash>;

static std::shared_ptr<FlashMhaBwdScratchState>
GetFlashBwdScratch(FlashMhaBwdScratchPool &scratch_pool, const pi::tensorlib::Device &device,
                   const pi::tensorlib::GpuStreamDescriptor &compute_stream_descriptor)
{
    const int compute_stream_id = compute_stream_descriptor.getStreamId();
    FlashMhaBwdScratchKey key{device.ordinal, compute_stream_id};
    if (const auto it = scratch_pool.find(key); it != scratch_pool.end())
    {
        return it->second;
    }
    auto state = std::make_shared<FlashMhaBwdScratchState>();
    scratch_pool.emplace(key, state);
    return state;
}

static std::shared_ptr<pi::tensorlib::RealTensor>
EnsureFlashMhaBwdScratch(FlashMhaBwdScratchState &state, uint32_t batch_size, uint32_t n_ctx, uint32_t n_head,
                         uint32_t head_size, const pi::tensorlib::Device &device,
                         const pi::tensorlib::GpuStreamDescriptor &compute_stream_descriptor)
{
    const auto n_ctx_rounded = static_cast<uint32_t>(AlignUp(n_ctx, 128));
    const uint64_t required_elements = static_cast<uint64_t>(batch_size) * n_ctx_rounded * n_head * head_size;
    bool needs_alloc = true;
    if (state.dq_accum && state.dq_accum->dtype() == pi::tensorlib::DataType::FLOAT32)
    {
        if (state.dq_accum->storage() && state.dq_accum->storage()->isFreed())
        {
            needs_alloc = true;
        }
        else
        {
            needs_alloc = state.dq_accum->shape().numel() < required_elements;
        }
    }
    if (needs_alloc)
    {
        state.dq_accum = AllocateMhaScratch({batch_size, n_ctx_rounded, n_head, head_size},
                                            pi::tensorlib::DataType::FLOAT32, device, compute_stream_descriptor, false);
    }
    state.dq_accum->storage()->setLastStreamId(compute_stream_descriptor.getStreamId());
    return state.dq_accum;
}

static std::shared_ptr<pi::tensorlib::RealTensor>
EnsureFlashMhaBwdSoftmaxSum(FlashMhaBwdScratchState &state, uint32_t batch_size, uint32_t n_ctx, uint32_t n_head,
                            const pi::tensorlib::Device &device,
                            const pi::tensorlib::GpuStreamDescriptor &compute_stream_descriptor)
{
    const uint32_t n_ctx_rounded = static_cast<uint32_t>(AlignUp(n_ctx, 128));
    const uint64_t required_elements = static_cast<uint64_t>(batch_size) * n_head * n_ctx_rounded;
    bool needs_alloc = true;
    if (state.dsoftmax_sum && state.dsoftmax_sum->dtype() == pi::tensorlib::DataType::FLOAT32)
    {
        if (state.dsoftmax_sum->storage() && state.dsoftmax_sum->storage()->isFreed())
        {
            needs_alloc = true;
        }
        else
        {
            needs_alloc = state.dsoftmax_sum->shape().numel() < required_elements;
        }
    }
    if (needs_alloc)
    {
        state.dsoftmax_sum = AllocateMhaScratch({batch_size, n_head, n_ctx_rounded}, pi::tensorlib::DataType::FLOAT32,
                                                device, compute_stream_descriptor, false);
    }
    state.dsoftmax_sum->storage()->setLastStreamId(compute_stream_descriptor.getStreamId());
    return state.dsoftmax_sum;
}

static void RemoveFlashBwdPreEntry(pi::tensorlib::ExecutionPlan &execution_plan, size_t &entry_idx,
                                   const std::shared_ptr<pi::tensorlib::RealTensor> &delta_tensor)
{
    if (!delta_tensor)
    {
        return;
    }
    auto it = std::find_if(execution_plan.entries.begin(),
                           execution_plan.entries.begin() + static_cast<std::ptrdiff_t>(entry_idx),
                           [&](const pi::tensorlib::ExecutionEntry &entry)
                           {
                               return entry.op_type == pi::tensorlib::OpType::MHA_ATTN_BWD_PRE &&
                                      entry.outputs.size() == 1 && entry.outputs[0] == delta_tensor;
                           });
    if (it != execution_plan.entries.begin() + static_cast<std::ptrdiff_t>(entry_idx))
    {
        const auto removed_index = static_cast<size_t>(std::distance(execution_plan.entries.begin(), it));
        execution_plan.entries.erase(it);
        if (removed_index < entry_idx)
        {
            --entry_idx;
        }
    }
}

#if PI_TENSORLIB_ENABLE_CUDA
static constexpr CutlassMhaKernelSet kCutlassMhaKernelsNoLse{
    &kmha_full_attn_cutlass_bf16,
    &kmha_full_attn_cutlass_fp16,
};

static constexpr CutlassMhaKernelSet kCutlassMhaKernelsLse{
    &kmha_full_attn_cutlass_bf16_lse,
    &kmha_full_attn_cutlass_fp16_lse,
};

static constexpr CutlassMhaBwdKernelSet kCutlassMhaBwdKernels{
    &kmha_full_attn_bwd_cutlass_bf16,
    &kmha_full_attn_bwd_cutlass_fp16,
};
#else
static constexpr CutlassMhaKernelSet kCutlassMhaKernelsNoLse{nullptr, nullptr};
static constexpr CutlassMhaKernelSet kCutlassMhaKernelsLse{nullptr, nullptr};
static constexpr CutlassMhaBwdKernelSet kCutlassMhaBwdKernels{nullptr, nullptr};
#endif

static constexpr uint32_t kCutlassLseAlign = 32;

static bool IsFlashEvenMn(const uint32_t n_ctx, const uint32_t block_m, const uint32_t block_n)
{
    return (n_ctx % block_m) == 0 && (n_ctx % block_n) == 0;
}

static const kernel_bin_t<kernel_meta_mha_cutlass_t> *SelectCutlassMhaKernel(const CutlassMhaKernelSet &kernels,
                                                                             const pi::tensorlib::DataType dtype)
{
    switch (dtype)
    {
        case pi::tensorlib::DataType::BFLOAT16:
            return kernels.bf16_kernel;
        case pi::tensorlib::DataType::FLOAT16:
            return kernels.fp16_kernel;
        default:
            return nullptr;
    }
}

static const kernel_bin_t<kernel_meta_mha_cutlass_bwd_t> *
SelectCutlassMhaBwdKernel(const CutlassMhaBwdKernelSet &kernels, const pi::tensorlib::DataType dtype)
{
    switch (dtype)
    {
        case pi::tensorlib::DataType::BFLOAT16:
            return kernels.bf16_kernel;
        case pi::tensorlib::DataType::FLOAT16:
            return kernels.fp16_kernel;
        default:
            return nullptr;
    }
}

static pi::tensorlib::ComputeKernelDescriptor
CreateCutlassMhaKernelDescriptor(const float softmax_scale, const pi::tensorlib::DataType dtype,
                                 const kernel_bin_t<kernel_meta_mha_cutlass_t> &kernel_bin)
{
    return pi::tensorlib::ComputeKernelDescriptor{
        .kernel_name = kernel_bin.kernel_name,
        .function_name = kernel_bin.function_name,
        .expected_arg_count = kernel_bin.arg_count,
        .argument_provider = [softmax_scale, dtype,
                              &kernel_bin](const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                                           const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
            -> pi::tensorlib::KernelLaunchArguments
        {
            if (inputs.size() != 3)
            {
                throw std::runtime_error("Invalid number of inputs for mha_full_attention kernel");
            }
            if (outputs.size() != 2)
            {
                throw std::runtime_error("Invalid number of outputs for mha_full_attention kernel");
            }

            const auto &q = inputs[0];
            const auto &k = inputs[1];
            const auto &v = inputs[2];
            const auto &o = outputs[0];
            const auto &scratch_m = outputs[1];

            if (q->dtype() != dtype || k->dtype() != dtype || v->dtype() != dtype)
            {
                throw std::runtime_error("q, k and v tensors must match attention kernel dtype");
            }
            if (o->dtype() != dtype)
            {
                throw std::runtime_error("output tensor must match attention kernel dtype");
            }
            if (scratch_m->dtype() != pi::tensorlib::DataType::FLOAT32)
            {
                throw std::runtime_error("scratch tensor must be of type FLOAT32");
            }

            const auto row_stride_q_t =
                RequirePackedBTHS(q, "q", static_cast<uint32_t>(q->shape()[1]), static_cast<uint32_t>(q->shape()[3]),
                                  "q tensor must be packed (B,T,H,HS)");
            const auto row_stride_k_t =
                RequirePackedBTHS(k, "k", static_cast<uint32_t>(k->shape()[1]), static_cast<uint32_t>(k->shape()[3]),
                                  "k tensor must be packed (B,T,H,HS)");
            const auto row_stride_v_t =
                RequirePackedBTHS(v, "v", static_cast<uint32_t>(v->shape()[1]), static_cast<uint32_t>(v->shape()[3]),
                                  "v tensor must be packed (B,T,H,HS)");
            if (row_stride_k_t != row_stride_q_t || row_stride_v_t != row_stride_q_t)
            {
                throw std::runtime_error("q/k/v row strides must match for MHA kernel");
            }

            if (scratch_m->shape().ndims() != 3)
            {
                throw std::runtime_error("scratch tensor must have shape (B, H, T)");
            }

            const auto device_ordinal = ValidateSameDeviceOrdinal("mha_full_attention", {q, k, v, scratch_m, o});

            const auto batch_size = static_cast<uint32_t>(q->shape()[0]);
            const auto n_ctx = static_cast<uint32_t>(q->shape()[1]);
            const auto n_head = static_cast<uint32_t>(q->shape()[2]);
            const auto head_size = static_cast<uint32_t>(q->shape()[3]);

            if (head_size != kernel_bin.meta.head_dim)
            {
                throw std::runtime_error("Cutlass MHA kernel expects head_size=" +
                                         std::to_string(kernel_bin.meta.head_dim));
            }
            if (o->shape()[0] != batch_size || o->shape()[1] != n_ctx || o->shape()[2] != n_head ||
                o->shape()[3] != head_size)
            {
                throw std::runtime_error("output tensor shape must be (B, T, H, HS)");
            }
            RequireContiguousBTHS(o, "output", n_ctx, n_head, head_size, "output tensor must be contiguous (B,T,H,HS)");

            const auto stride_q_z = static_cast<uint32_t>(q->strides()[0]);
            const auto stride_q_h = static_cast<uint32_t>(q->strides()[2]);
            const auto stride_q_t = row_stride_q_t;

            const auto stride_k_z = static_cast<uint32_t>(k->strides()[0]);
            const auto stride_k_h = static_cast<uint32_t>(k->strides()[2]);
            const auto stride_k_t = row_stride_k_t;

            const auto stride_v_z = static_cast<uint32_t>(v->strides()[0]);
            const auto stride_v_h = static_cast<uint32_t>(v->strides()[2]);
            const auto stride_v_t = row_stride_v_t;

            const auto stride_o_z = static_cast<uint32_t>(o->strides()[0]);
            const auto stride_o_h = static_cast<uint32_t>(o->strides()[2]);
            const auto stride_o_t = static_cast<uint32_t>(o->strides()[1]);

            const auto stride_m_z = static_cast<uint32_t>(scratch_m->strides()[0]);
            const auto stride_m_h = static_cast<uint32_t>(scratch_m->strides()[1]);
            const auto stride_m_t = static_cast<uint32_t>(scratch_m->strides()[2]);

            void *q_ptr = q->dataptr();
            void *k_ptr = k->dataptr();
            void *v_ptr = v->dataptr();
            void *scratch_m_ptr = scratch_m->dataptr();
            void *o_ptr = o->dataptr();

            if (IsMhaDebugEnabled())
            {
                std::cout << "[MHA] softmax_scale=" << softmax_scale << '\n';
                auto ptr_align = [](void *ptr) -> uint64_t { return reinterpret_cast<uintptr_t>(ptr) % 16; };
                std::cout << "[MHA] Tensor ptr alignment (q,k,v,o,scratch)=" << ptr_align(q_ptr) << '/'
                          << ptr_align(k_ptr) << '/' << ptr_align(v_ptr) << '/' << ptr_align(o_ptr) << '/'
                          << ptr_align(scratch_m_ptr) << '\n';
            }

            pi::tensorlib::KernelLaunchArguments arguments{};
            arguments.args = {softmax_scale, scratch_m_ptr, batch_size, n_head,     q_ptr,
                              stride_q_z,    stride_q_h,    stride_q_t, k_ptr,      stride_k_z,
                              stride_k_h,    stride_k_t,    v_ptr,      stride_v_z, stride_v_h,
                              stride_v_t,    o_ptr,         stride_o_z, stride_o_h, stride_o_t,
                              n_ctx,         stride_m_z,    stride_m_h, stride_m_t, static_cast<void *>(nullptr)};

            const auto grid_x = CEIL_DIV(n_ctx, kernel_bin.meta.block_size_x);
            arguments.grid_dim_x = grid_x;
// Newer CUTLASS kernels (sm100+) map grid.y -> batch and grid.z -> heads.
#if PI_TENSORLIB_ENABLE_CUDA && (NV_KERNEL_ARCH >= 100)
            arguments.grid_dim_y = batch_size;
            arguments.grid_dim_z = n_head;
#else
            arguments.grid_dim_y = n_head;
            arguments.grid_dim_z = batch_size;
#endif

// Newer CUTLASS kernels (sm100+) use 1D threadblock layout.
#if PI_TENSORLIB_ENABLE_CUDA && (NV_KERNEL_ARCH >= 100)
            arguments.block_dim_x = CUDA_WARP_SIZE * kernel_bin.num_warps;
            arguments.block_dim_y = 1;
#else
            arguments.block_dim_x = CUDA_WARP_SIZE;
            arguments.block_dim_y = kernel_bin.num_warps;
#endif
            arguments.block_dim_z = 1;

            arguments.shared_mem_bytes = kernel_bin.shared_mem_bytes;
            arguments.device_ordinal = device_ordinal;

            if (IsMhaDebugEnabled())
            {
                std::cout << "[MHA] CUTLASS launch params: B=" << batch_size << ", H=" << n_head << ", T=" << n_ctx
                          << ", head_dim=" << head_size << ", block_m=" << kernel_bin.meta.block_size_x
                          << ", block_n=" << kernel_bin.meta.block_size_y << ", num_warps=" << kernel_bin.num_warps
                          << '\n';
            }

            return arguments;
        },
        .cuda_descriptor = pi::tensorlib::MakeCudaKernelDescriptor(kernel_bin.data, kernel_bin.size),
        .hip_descriptor = pi::tensorlib::MakeHipKernelDescriptor(kernel_bin.data, kernel_bin.size),
    };
}

static pi::tensorlib::ComputeKernelDescriptor
CreateCutlassMhaBwdKernelDescriptor(const float softmax_scale, const pi::tensorlib::DataType dtype,
                                    const kernel_bin_t<kernel_meta_mha_cutlass_bwd_t> &kernel_bin,
                                    const pi::tensorlib::GpuStreamDescriptor &compute_stream_descriptor,
                                    const std::shared_ptr<CutlassMhaBwdWorkspaceState> &workspace_state)
{
    const int compute_stream_id = compute_stream_descriptor.getStreamId();
    const auto compute_stream_descriptor_copy = compute_stream_descriptor;
    pi::tensorlib::ComputeKernelDescriptor descriptor{
        .kernel_name = kernel_bin.kernel_name,
        .function_name = kernel_bin.function_name,
        .expected_arg_count = kernel_bin.arg_count,
        .argument_provider = nullptr,
        .cuda_descriptor = pi::tensorlib::MakeCudaKernelDescriptor(kernel_bin.data, kernel_bin.size),
        .hip_descriptor = pi::tensorlib::MakeHipKernelDescriptor(kernel_bin.data, kernel_bin.size),
    };

    descriptor.argument_provider =
        [softmax_scale, dtype, compute_stream_id, compute_stream_descriptor_copy, &kernel_bin,
         workspace_state](const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                          const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
        -> pi::tensorlib::KernelLaunchArguments
    {
        if (inputs.size() != 7)
        {
            throw std::runtime_error("Invalid number of inputs for mha_full_attention backward kernel");
        }
        if (outputs.size() != 3)
        {
            throw std::runtime_error("Invalid number of outputs for mha_full_attention backward kernel");
        }

        const auto &q = inputs[0];
        const auto &k = inputs[1];
        const auto &v = inputs[2];
        const auto &o = inputs[3];
        const auto &do_tensor = inputs[4];
        const auto &m_tensor = inputs[5];
        const auto &delta = inputs[6];

        const auto &dq = outputs[0];
        const auto &dk = outputs[1];
        const auto &dv = outputs[2];

        if (q->dtype() != dtype || k->dtype() != dtype || v->dtype() != dtype || o->dtype() != dtype ||
            do_tensor->dtype() != dtype)
        {
            throw std::runtime_error("mha_full_attention backward tensors must match kernel dtype");
        }
        if (m_tensor->dtype() != pi::tensorlib::DataType::FLOAT32 || delta->dtype() != pi::tensorlib::DataType::FLOAT32)
        {
            throw std::runtime_error("mha_full_attention backward M/delta tensors must be FLOAT32");
        }
        if (dq->dtype() != dtype || dk->dtype() != dtype || dv->dtype() != dtype)
        {
            throw std::runtime_error("mha_full_attention backward outputs must match kernel dtype");
        }

        RequireContiguousHeadDim(q, "q");
        RequireContiguousHeadDim(k, "k");
        RequireContiguousHeadDim(v, "v");
        RequireContiguousHeadDim(dq, "dq");
        RequireContiguousHeadDim(dk, "dk");
        RequireContiguousHeadDim(dv, "dv");

        if (m_tensor->shape().ndims() != 3 || delta->shape().ndims() != 3)
        {
            throw std::runtime_error("mha_full_attention backward M/delta tensors must have shape (B, H, T)");
        }

        const auto batch_size = static_cast<uint32_t>(q->shape()[0]);
        const auto n_ctx = static_cast<uint32_t>(q->shape()[1]);
        const auto n_head = static_cast<uint32_t>(q->shape()[2]);
        const auto head_size = static_cast<uint32_t>(q->shape()[3]);

        if (head_size != kernel_bin.meta.head_dim)
        {
            throw std::runtime_error("Cutlass MHA bwd kernel expects head_size=" +
                                     std::to_string(kernel_bin.meta.head_dim));
        }

        if (k->shape() != q->shape() || v->shape() != q->shape())
        {
            throw std::runtime_error("mha_full_attention backward Q/K/V tensor shapes must match");
        }
        if (dq->shape().ndims() != 4 || dq->shape()[0] != batch_size || dq->shape()[1] != n_ctx ||
            dq->shape()[2] != n_head || dq->shape()[3] != head_size || dk->shape() != dq->shape() ||
            dv->shape() != dq->shape())
        {
            throw std::runtime_error("mha_full_attention backward gradient shapes must be (B, T, H, HS)");
        }
        if (o->shape()[0] != batch_size || o->shape()[1] != n_ctx || o->shape()[2] != n_head ||
            o->shape()[3] != head_size)
        {
            throw std::runtime_error("mha_full_attention backward output shape must be (B, T, H, HS)");
        }
        if (do_tensor->shape() != o->shape())
        {
            throw std::runtime_error("mha_full_attention backward upstream must match output shape");
        }
        if (m_tensor->shape()[0] != batch_size || m_tensor->shape()[1] != n_head || m_tensor->shape()[2] != n_ctx ||
            delta->shape()[0] != batch_size || delta->shape()[1] != n_head || delta->shape()[2] != n_ctx)
        {
            throw std::runtime_error("mha_full_attention backward M/delta shapes must be (B, H, T)");
        }

        const auto device_ordinal =
            ValidateSameDeviceOrdinal("mha_full_attention_bwd", {q, k, v, o, do_tensor, m_tensor, delta, dq, dk, dv});

        const auto row_stride_q = RequirePackedBTHS(q, "q", n_ctx, head_size,
                                                    "mha_full_attention backward requires packed q tensor (B,T,H,HS)");
        const auto row_stride_k = RequirePackedBTHS(k, "k", n_ctx, head_size,
                                                    "mha_full_attention backward requires packed k tensor (B,T,H,HS)");
        const auto row_stride_v = RequirePackedBTHS(v, "v", n_ctx, head_size,
                                                    "mha_full_attention backward requires packed v tensor (B,T,H,HS)");
        if (row_stride_k != row_stride_q || row_stride_v != row_stride_q)
        {
            throw std::runtime_error("mha_full_attention backward requires matching q/k/v row strides");
        }

        const auto row_stride_dq = RequirePackedBTHS(
            dq, "grad_q", n_ctx, head_size, "mha_full_attention backward requires packed grad_q tensor (B,T,H,HS)");
        const auto row_stride_dk = RequirePackedBTHS(
            dk, "grad_k", n_ctx, head_size, "mha_full_attention backward requires packed grad_k tensor (B,T,H,HS)");
        const auto row_stride_dv = RequirePackedBTHS(
            dv, "grad_v", n_ctx, head_size, "mha_full_attention backward requires packed grad_v tensor (B,T,H,HS)");
        if (row_stride_dq != row_stride_q || row_stride_dk != row_stride_q || row_stride_dv != row_stride_q)
        {
            throw std::runtime_error("mha_full_attention backward requires grad row strides to match q/k/v");
        }
        const uint32_t expected_row_stride = n_head * head_size;
        if (expected_row_stride == 0 || row_stride_q % expected_row_stride != 0)
        {
            throw std::runtime_error("mha_full_attention backward requires q row stride to be divisible by H*HS");
        }
        const uint32_t gqkv_stride_multiplier = row_stride_q / expected_row_stride;
        if (gqkv_stride_multiplier > 127)
        {
            throw std::runtime_error("mha_full_attention backward requires q row stride multiplier <= 127");
        }
        RequireContiguousBTHS(o, "output", n_ctx, n_head, head_size,
                              "mha_full_attention backward requires contiguous output tensor (B,T,H,HS)");
        RequireContiguousBTHS(do_tensor, "upstream", n_ctx, n_head, head_size,
                              "mha_full_attention backward requires contiguous upstream tensor (B,T,H,HS)");

        RequireContiguousBHT(m_tensor, "scratch M", n_ctx, n_head,
                             "mha_full_attention backward requires contiguous scratch M tensor");
        RequireContiguousBHT(delta, "delta", n_ctx, n_head,
                             "mha_full_attention backward requires contiguous delta tensor");

        const auto layout = ComputeCutlassMhaBwdWorkspaceLayout(batch_size, n_head, n_ctx, head_size, kernel_bin.meta);
        const uint64_t workspace_bytes = layout.bytes;

        void *workspace_ptr = nullptr;
        if (workspace_state && workspace_state->cache_entry && workspace_state->cache_entry->workspace)
        {
            if (workspace_state->cleanup_stream_desc.isValid() &&
                !(workspace_state->cleanup_stream_desc == compute_stream_descriptor_copy))
            {
                // Ensure the cleanup stream finished resetting the workspace before launching on the compute stream.
                const auto stream_bundle = pi::tensorlib::ExecutionBackend::GetStreamBundle(q->device());
                if (!stream_bundle)
                {
                    throw std::runtime_error("Failed to acquire GPU stream bundle for MHA workspace wait");
                }
                pi::tensorlib::ExecutionBackend::GpuStreamWaitFor(
                    pi::tensorlib::streamutils::GetStream(stream_bundle, workspace_state->cleanup_stream_desc),
                    pi::tensorlib::streamutils::GetStream(stream_bundle, compute_stream_descriptor_copy));
            }
            workspace_state->cache_entry->workspace->storage()->setLastStreamId(compute_stream_id);
            workspace_ptr = workspace_state->cache_entry->workspace->dataptr();
        }

        if (IsMhaDebugEnabled())
        {
            std::cout << "[MHA] CUTLASS bwd workspace_bytes=" << workspace_bytes
                      << " num_splits_key=" << layout.num_splits_key << '\n';
        }

        pi::tensorlib::KernelLaunchArguments arguments{};
        const auto stride_q_z = static_cast<uint32_t>(q->strides()[0]);
        const auto stride_q_h = static_cast<uint32_t>(q->strides()[2]);
        const auto stride_k_z = static_cast<uint32_t>(k->strides()[0]);
        const auto stride_k_h = static_cast<uint32_t>(k->strides()[2]);
        const auto stride_v_z = static_cast<uint32_t>(v->strides()[0]);
        const auto stride_v_h = static_cast<uint32_t>(v->strides()[2]);
        const auto stride_o_z = static_cast<uint32_t>(o->strides()[0]);
        const auto stride_o_h = static_cast<uint32_t>(o->strides()[2]);
        const auto stride_o_t = static_cast<uint32_t>(o->strides()[1]);
        const auto stride_do_z = static_cast<uint32_t>(do_tensor->strides()[0]);
        const auto stride_do_h = static_cast<uint32_t>(do_tensor->strides()[2]);
        const auto stride_do_t = static_cast<uint32_t>(do_tensor->strides()[1]);

        arguments.args = {
            q->dataptr(),        stride_q_z,       stride_q_h,   row_stride_q,  k->dataptr(),  stride_k_z,
            stride_k_h,          row_stride_k,     v->dataptr(), stride_v_z,    stride_v_h,    row_stride_v,
            o->dataptr(),        stride_o_z,       stride_o_h,   stride_o_t,    softmax_scale, do_tensor->dataptr(),
            stride_do_z,         stride_do_h,      stride_do_t,  dq->dataptr(), dk->dataptr(), dv->dataptr(),
            m_tensor->dataptr(), delta->dataptr(), batch_size,   n_head,        n_ctx,         workspace_ptr};

        arguments.grid_dim_x = layout.num_splits_key;
        arguments.grid_dim_y = n_head;
        arguments.grid_dim_z = batch_size;

        arguments.block_dim_x = CUDA_WARP_SIZE * kernel_bin.num_warps;
        arguments.block_dim_y = 1;
        arguments.block_dim_z = 1;

        arguments.shared_mem_bytes = kernel_bin.shared_mem_bytes;
        arguments.device_ordinal = device_ordinal;

        return arguments;
    };
    return descriptor;
}

static kernel_bin_t<kernel_meta_2d_t> SelectMhaFwdKernel(const pi::tensorlib::DataType dtype,
                                                         const bool use_fp16_accumulation, const bool write_lse,
                                                         const bool even_mn, std::string &kernel_name)
{
    kernel_bin_t<kernel_meta_2d_t> kernel;
    switch (dtype)
    {
        case pi::tensorlib::DataType::BFLOAT16:
            if (even_mn)
            {
                kernel = write_lse ? kmha_full_attn_fwd_hs128_bf16 : kmha_full_attn_fwd_hs128_bf16_nolse;
                kernel_name = write_lse ? "mha_full_attn_fwd_hs128_bf16" : "mha_full_attn_fwd_hs128_bf16_nolse";
            }
            else
            {
                kernel = write_lse ? kmha_full_attn_fwd_hs128_bf16_uneven : kmha_full_attn_fwd_hs128_bf16_uneven_nolse;
                kernel_name =
                    write_lse ? "mha_full_attn_fwd_hs128_bf16_uneven" : "mha_full_attn_fwd_hs128_bf16_uneven_nolse";
            }
            break;
        case pi::tensorlib::DataType::FLOAT16:
            if (use_fp16_accumulation)
            {
#if PI_TENSORLIB_ENABLE_CUDA
                if (even_mn)
                {
                    kernel = write_lse ? kmha_full_attn_fwd_hs128_fp16_acc_fp16
                                       : kmha_full_attn_fwd_hs128_fp16_acc_fp16_nolse;
                    kernel_name = write_lse ? "mha_full_attn_fwd_hs128_fp16_acc_fp16"
                                            : "mha_full_attn_fwd_hs128_fp16_acc_fp16_nolse";
                }
                else
                {
                    kernel = write_lse ? kmha_full_attn_fwd_hs128_fp16_acc_fp16_uneven
                                       : kmha_full_attn_fwd_hs128_fp16_acc_fp16_uneven_nolse;
                    kernel_name = write_lse ? "mha_full_attn_fwd_hs128_fp16_acc_fp16_uneven"
                                            : "mha_full_attn_fwd_hs128_fp16_acc_fp16_uneven_nolse";
                }
#else
                RequireFp16AccumulationSupported(true);
#endif
            }
            else
            {
                if (even_mn)
                {
                    kernel = write_lse ? kmha_full_attn_fwd_hs128_fp16 : kmha_full_attn_fwd_hs128_fp16_nolse;
                    kernel_name = write_lse ? "mha_full_attn_fwd_hs128_fp16" : "mha_full_attn_fwd_hs128_fp16_nolse";
                }
                else
                {
                    kernel =
                        write_lse ? kmha_full_attn_fwd_hs128_fp16_uneven : kmha_full_attn_fwd_hs128_fp16_uneven_nolse;
                    kernel_name =
                        write_lse ? "mha_full_attn_fwd_hs128_fp16_uneven" : "mha_full_attn_fwd_hs128_fp16_uneven_nolse";
                }
            }
            break;
        default:
            throw std::runtime_error("Unsupported data type for mha_full_attention kernel");
    }
    return kernel;
}

static kernel_bin_t<kernel_meta_2d_t> SelectFlashMhaFwdKernel(const pi::tensorlib::DataType dtype,
                                                              const bool use_fp16_accumulation, const bool write_lse,
                                                              const bool even_mn, std::string &kernel_name)
{
#if PI_TENSORLIB_ENABLE_CUDA
    kernel_bin_t<kernel_meta_2d_t> kernel;
    switch (dtype)
    {
        case pi::tensorlib::DataType::BFLOAT16:
            if (even_mn)
            {
                kernel =
                    write_lse ? kmha_full_attn_fa_fwd_hs128_bf16_even : kmha_full_attn_fa_fwd_hs128_bf16_even_nolse;
                kernel_name =
                    write_lse ? "mha_full_attn_fa_fwd_hs128_bf16_even" : "mha_full_attn_fa_fwd_hs128_bf16_even_nolse";
            }
            else
            {
                kernel = write_lse ? kmha_full_attn_fa_fwd_hs128_bf16 : kmha_full_attn_fa_fwd_hs128_bf16_nolse;
                kernel_name = write_lse ? "mha_full_attn_fa_fwd_hs128_bf16" : "mha_full_attn_fa_fwd_hs128_bf16_nolse";
            }
            break;
        case pi::tensorlib::DataType::FLOAT16:
            if (use_fp16_accumulation)
            {
                if (even_mn)
                {
                    kernel = write_lse ? kmha_full_attn_fa_fwd_hs128_fp16_acc_fp16_even
                                       : kmha_full_attn_fa_fwd_hs128_fp16_acc_fp16_even_nolse;
                    kernel_name = write_lse ? "mha_full_attn_fa_fwd_hs128_fp16_acc_fp16_even"
                                            : "mha_full_attn_fa_fwd_hs128_fp16_acc_fp16_even_nolse";
                }
                else
                {
                    kernel = write_lse ? kmha_full_attn_fa_fwd_hs128_fp16_acc_fp16
                                       : kmha_full_attn_fa_fwd_hs128_fp16_acc_fp16_nolse;
                    kernel_name = write_lse ? "mha_full_attn_fa_fwd_hs128_fp16_acc_fp16"
                                            : "mha_full_attn_fa_fwd_hs128_fp16_acc_fp16_nolse";
                }
            }
            else
            {
                if (even_mn)
                {
                    kernel =
                        write_lse ? kmha_full_attn_fa_fwd_hs128_fp16_even : kmha_full_attn_fa_fwd_hs128_fp16_even_nolse;
                    kernel_name = write_lse ? "mha_full_attn_fa_fwd_hs128_fp16_even"
                                            : "mha_full_attn_fa_fwd_hs128_fp16_even_nolse";
                }
                else
                {
                    kernel = write_lse ? kmha_full_attn_fa_fwd_hs128_fp16 : kmha_full_attn_fa_fwd_hs128_fp16_nolse;
                    kernel_name =
                        write_lse ? "mha_full_attn_fa_fwd_hs128_fp16" : "mha_full_attn_fa_fwd_hs128_fp16_nolse";
                }
            }
            break;
        default:
            throw std::runtime_error("Unsupported data type for flash attention kernel");
    }
    return kernel;
#else
    (void)dtype;
    (void)use_fp16_accumulation;
    (void)write_lse;
    (void)even_mn;
    (void)kernel_name;
    throw std::runtime_error("Flash attention kernels require CUDA support.");
#endif
}

static pi::tensorlib::KernelArgumentProvider
MakeMhaFullAttentionArgumentProvider(float softmax_scale, const pi::tensorlib::DataType dtype,
                                     const kernel_bin_t<kernel_meta_2d_t> &kernel,
                                     const pi::tensorlib::GpuStreamDescriptor &compute_stream_descriptor,
                                     const std::shared_ptr<TritonMhaScratchState> &scratch_state)
{
    const int compute_stream_id = compute_stream_descriptor.getStreamId();
    const auto compute_stream_descriptor_copy = compute_stream_descriptor;
    return [kernel, softmax_scale, dtype, compute_stream_id, compute_stream_descriptor_copy,
            scratch_state](const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                               const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
               -> pi::tensorlib::KernelLaunchArguments
    {
        if (inputs.size() != 3)
        {
            throw std::runtime_error("Invalid number of inputs for mha_full_attention kernel");
        }
        if (outputs.size() != 1 && outputs.size() != 2)
        {
            throw std::runtime_error("Invalid number of outputs for mha_full_attention kernel");
        }

        const auto &q = inputs[0];
        const auto &k = inputs[1];
        const auto &v = inputs[2];
        const auto &o = outputs[0];

        if (q->dtype() != dtype || k->dtype() != dtype || v->dtype() != dtype)
        {
            throw std::runtime_error("q, k and v tensors must match attention kernel dtype");
        }
        if (o->dtype() != dtype)
        {
            throw std::runtime_error("output tensor must match attention kernel dtype");
        }

        const auto device_ordinal = outputs.size() == 2
                                        ? ValidateSameDeviceOrdinal("mha_full_attention", {q, k, v, outputs[1], o})
                                        : ValidateSameDeviceOrdinal("mha_full_attention", {q, k, v, o});

        void *q_ptr = q->dataptr();
        void *k_ptr = k->dataptr();
        void *v_ptr = v->dataptr();
        void *o_ptr = o->dataptr();
        void *scratch_m_ptr = nullptr;

        const auto batch_size = static_cast<uint32_t>(q->shape()[0]);
        const auto n_ctx = static_cast<uint32_t>(q->shape()[1]);
        const auto n_head = static_cast<uint32_t>(q->shape()[2]);
        const auto head_size = static_cast<uint32_t>(q->shape()[3]);

        const auto row_stride_q = RequirePackedBTHS(q, "q", n_ctx, head_size, "q tensor must be packed (B,T,H,HS)");
        const auto row_stride_k = RequirePackedBTHS(k, "k", n_ctx, head_size, "k tensor must be packed (B,T,H,HS)");
        const auto row_stride_v = RequirePackedBTHS(v, "v", n_ctx, head_size, "v tensor must be packed (B,T,H,HS)");
        if (row_stride_k != row_stride_q || row_stride_v != row_stride_q)
        {
            throw std::runtime_error("q/k/v row strides must match for MHA kernel");
        }

        if (head_size != 128)
        {
            throw std::runtime_error("head_size must be 128 for mha_full_attention kernel; got " +
                                     std::to_string(head_size) + " instead");
        }
        if (o->shape()[0] != batch_size || o->shape()[1] != n_ctx || o->shape()[2] != n_head ||
            o->shape()[3] != head_size)
        {
            throw std::runtime_error("output tensor shape must be (B, T, H, HS)");
        }
        RequireContiguousBTHS(o, "output", n_ctx, n_head, head_size, "output tensor must be contiguous (B,T,H,HS)");

        void *global_scratch_ptr = nullptr;

        if (outputs.size() == 2)
        {
            const auto &scratch_m = outputs[1];
            if (scratch_m->dtype() != pi::tensorlib::DataType::FLOAT32)
            {
                throw std::runtime_error("scratch tensor must be of type FLOAT32");
            }
            if (scratch_m->shape().ndims() != 3)
            {
                throw std::runtime_error("scratch tensor must have shape (B, H, T)");
            }
            const auto &strides = scratch_m->strides();
            const uint64_t expected_t = 1;
            const uint64_t expected_h = n_ctx;
            const uint64_t expected_z = static_cast<uint64_t>(n_head) * n_ctx;
            if (strides[2] != expected_t || strides[1] != expected_h || strides[0] != expected_z)
            {
                throw std::runtime_error("scratch tensor must be contiguous (B,H,T)");
            }
            scratch_m_ptr = scratch_m->dataptr();
        }

        std::vector<std::any> args = {softmax_scale, scratch_m_ptr, batch_size, n_head,       q_ptr,
                                      k_ptr,         v_ptr,         o_ptr,      row_stride_q, n_ctx};

        pi::tensorlib::KernelLaunchArguments arguments{.args = {},
                                                       .grid_dim_x = (n_ctx + kernel.meta.block_size_x - 1) /
                                                                     kernel.meta.block_size_x,
                                                       .grid_dim_y = batch_size * n_head,
                                                       .grid_dim_z = 1,
                                                       .block_dim_x = TRITON_WARP_SIZE * kernel.num_warps,
                                                       .block_dim_y = 1,
                                                       .block_dim_z = 1,
                                                       .shared_mem_bytes = kernel.shared_mem_bytes,
                                                       .device_ordinal = device_ordinal};

        if (kernel.global_scratch_size > 0)
        {
            const uint64_t grid_x = (n_ctx + kernel.meta.block_size_x - 1) / kernel.meta.block_size_x;
            const uint64_t grid = grid_x * n_head * batch_size;
            uint64_t scratch_bytes = static_cast<uint64_t>(kernel.global_scratch_size) * grid;
            const uint64_t scratch_elems = (scratch_bytes + 3) / 4;
            auto scratch_tensor = scratch_state->global;
            if (scratch_tensor && scratch_tensor->shape().numel() < scratch_elems)
            {
                scratch_tensor.reset();
            }
            if (!scratch_tensor)
            {
                scratch_tensor = AllocateMhaScratch({scratch_elems}, pi::tensorlib::DataType::UINT32, q->device(),
                                                    compute_stream_descriptor_copy, false);
                scratch_state->global = scratch_tensor;
            }
            scratch_tensor->storage()->setLastStreamId(compute_stream_id);
            global_scratch_ptr = scratch_tensor->dataptr();
        }

        args.emplace_back(global_scratch_ptr);
        arguments.args = std::move(args);
        return arguments;
    };
}

static pi::tensorlib::ComputeKernelDescriptor CreateMhaFullAttentionComputeKernelDescriptor(
    const float softmax_scale, const pi::tensorlib::DataType dtype, const kernel_bin_t<kernel_meta_2d_t> &kernel,
    const std::string &kernel_name, const pi::tensorlib::GpuStreamDescriptor &compute_stream_descriptor)
{
    auto scratch_state = std::make_shared<TritonMhaScratchState>();
    pi::tensorlib::ComputeKernelDescriptor descriptor{
        .kernel_name = kernel_name,
        .function_name = kernel.function_name,
        .expected_arg_count = kernel.arg_count,
        .argument_provider = nullptr,
        .cuda_descriptor = pi::tensorlib::MakeCudaKernelDescriptor(kernel.data, kernel.size),
        .hip_descriptor = pi::tensorlib::MakeHipKernelDescriptor(kernel.data, kernel.size),
    };
    descriptor.argument_provider =
        MakeMhaFullAttentionArgumentProvider(softmax_scale, dtype, kernel, compute_stream_descriptor, scratch_state);
    return descriptor;
}

static pi::tensorlib::KernelArgumentProvider
MakeFlashMhaFullAttentionArgumentProvider(float softmax_scale, const pi::tensorlib::DataType dtype,
                                          const kernel_bin_t<kernel_meta_2d_t> &kernel,
                                          const pi::tensorlib::GpuStreamDescriptor &compute_stream_descriptor)
{
    auto scratch_state = std::make_shared<FlashScratchState>();
    const int compute_stream_id = compute_stream_descriptor.getStreamId();
    const auto compute_stream_descriptor_copy = compute_stream_descriptor;
    return [kernel, softmax_scale, dtype, compute_stream_id, compute_stream_descriptor_copy,
            scratch_state](const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                           const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
               -> pi::tensorlib::KernelLaunchArguments
    {
        if (inputs.size() != 3)
        {
            throw std::runtime_error("Invalid number of inputs for mha_full_attention kernel");
        }
        if (outputs.size() != 1 && outputs.size() != 2)
        {
            throw std::runtime_error("Invalid number of outputs for mha_full_attention kernel");
        }

        const auto &q = inputs[0];
        const auto &k = inputs[1];
        const auto &v = inputs[2];
        const auto &o = outputs[0];

        if (q->dtype() != dtype || k->dtype() != dtype || v->dtype() != dtype)
        {
            throw std::runtime_error("q, k and v tensors must match attention kernel dtype");
        }
        if (o->dtype() != dtype)
        {
            throw std::runtime_error("output tensor must match attention kernel dtype");
        }

        const auto device_ordinal = outputs.size() == 2
                                        ? ValidateSameDeviceOrdinal("mha_full_attention", {q, k, v, outputs[1], o})
                                        : ValidateSameDeviceOrdinal("mha_full_attention", {q, k, v, o});

        void *q_ptr = q->dataptr();
        void *k_ptr = k->dataptr();
        void *v_ptr = v->dataptr();
        void *o_ptr = o->dataptr();

        const auto batch_size = static_cast<uint32_t>(q->shape()[0]);
        const auto n_ctx = static_cast<uint32_t>(q->shape()[1]);
        const auto n_head = static_cast<uint32_t>(q->shape()[2]);
        const auto head_size = static_cast<uint32_t>(q->shape()[3]);

        const auto row_stride_q = RequirePackedBTHS(q, "q", n_ctx, head_size, "q tensor must be packed (B,T,H,HS)");
        const auto row_stride_k = RequirePackedBTHS(k, "k", n_ctx, head_size, "k tensor must be packed (B,T,H,HS)");
        const auto row_stride_v = RequirePackedBTHS(v, "v", n_ctx, head_size, "v tensor must be packed (B,T,H,HS)");
        if (row_stride_k != row_stride_q || row_stride_v != row_stride_q)
        {
            throw std::runtime_error("q/k/v row strides must match for MHA kernel");
        }

        if (head_size != 128)
        {
            throw std::runtime_error("head_size must be 128 for mha_full_attention kernel; got " +
                                     std::to_string(head_size) + " instead");
        }
        if (o->shape()[0] != batch_size || o->shape()[1] != n_ctx || o->shape()[2] != n_head ||
            o->shape()[3] != head_size)
        {
            throw std::runtime_error("output tensor shape must be (B, T, H, HS)");
        }
        RequireContiguousBTHS(o, "output", n_ctx, n_head, head_size, "output tensor must be contiguous (B,T,H,HS)");

        std::shared_ptr<pi::tensorlib::RealTensor> scratch_tensor{};
        if (outputs.size() == 2)
        {
            const auto &scratch_m = outputs[1];
            if (scratch_m->dtype() != pi::tensorlib::DataType::FLOAT32)
            {
                throw std::runtime_error("scratch tensor must be of type FLOAT32");
            }
            if (scratch_m->shape().ndims() != 3)
            {
                throw std::runtime_error("scratch tensor must have shape (B, H, T)");
            }
            const auto &strides = scratch_m->strides();
            const uint64_t expected_t = 1;
            const uint64_t expected_h = n_ctx;
            const uint64_t expected_z = static_cast<uint64_t>(n_head) * n_ctx;
            if (strides[2] != expected_t || strides[1] != expected_h || strides[0] != expected_z)
            {
                throw std::runtime_error("scratch tensor must be contiguous (B,H,T)");
            }
            scratch_tensor = scratch_m;
        }
        else
        {
            const uint64_t scratch_elems = static_cast<uint64_t>(batch_size) * n_head * n_ctx;
            const auto &cached = scratch_state->lse;
            if (cached && cached->shape().numel() >= scratch_elems)
            {
                scratch_tensor = cached;
            }
            if (!scratch_tensor)
            {
                scratch_tensor = AllocateMhaScratch({batch_size, n_head, n_ctx}, pi::tensorlib::DataType::FLOAT32,
                                                    q->device(), compute_stream_descriptor_copy, false);
                scratch_state->lse = scratch_tensor;
            }
            scratch_tensor->storage()->setLastStreamId(compute_stream_id);
        }

        void *scratch_m_ptr = scratch_tensor ? scratch_tensor->dataptr() : nullptr;
        if (!scratch_m_ptr)
        {
            throw std::runtime_error("flash attention requires a valid scratch buffer for LSE");
        }

        std::vector<std::any> args = {softmax_scale, scratch_m_ptr, batch_size, n_head,       q_ptr,
                                      k_ptr,         v_ptr,         o_ptr,      row_stride_q, n_ctx};

        pi::tensorlib::KernelLaunchArguments arguments{.args = {},
                                                       .grid_dim_x = (n_ctx + kernel.meta.block_size_x - 1) /
                                                                     kernel.meta.block_size_x,
                                                       .grid_dim_y = batch_size,
                                                       .grid_dim_z = n_head,
                                                       .block_dim_x = CUDA_WARP_SIZE * kernel.num_warps,
                                                       .block_dim_y = 1,
                                                       .block_dim_z = 1,
                                                       .shared_mem_bytes = kernel.shared_mem_bytes,
                                                       .device_ordinal = device_ordinal};

        void *global_scratch_ptr = nullptr;
        if (kernel.global_scratch_size > 0)
        {
            const uint64_t grid_x = (n_ctx + kernel.meta.block_size_x - 1) / kernel.meta.block_size_x;
            const uint64_t grid = grid_x * n_head * batch_size;
            uint64_t scratch_bytes = static_cast<uint64_t>(kernel.global_scratch_size) * grid;
            const uint64_t scratch_elems = (scratch_bytes + 3) / 4;
            std::shared_ptr<pi::tensorlib::RealTensor> scratch_global = scratch_state->global;
            if (!scratch_global || scratch_global->shape().numel() < scratch_elems)
            {
                scratch_global = AllocateMhaScratch({scratch_elems}, pi::tensorlib::DataType::UINT32, q->device(),
                                                    compute_stream_descriptor_copy, false);
                scratch_state->global = scratch_global;
            }
            scratch_global->storage()->setLastStreamId(compute_stream_id);
            global_scratch_ptr = scratch_global->dataptr();
        }

        args.emplace_back(global_scratch_ptr);
        arguments.args = std::move(args);
        return arguments;
    };
}

static pi::tensorlib::ComputeKernelDescriptor CreateFlashMhaFullAttentionComputeKernelDescriptor(
    float softmax_scale, const pi::tensorlib::DataType dtype, const kernel_bin_t<kernel_meta_2d_t> &kernel,
    const std::string &kernel_name, const pi::tensorlib::GpuStreamDescriptor &compute_stream_descriptor)
{
    (void)compute_stream_descriptor;
    pi::tensorlib::ComputeKernelDescriptor descriptor{
        .kernel_name = kernel_name,
        .function_name = kernel.function_name,
        .expected_arg_count = kernel.arg_count,
        .argument_provider = nullptr,
        .cuda_descriptor = pi::tensorlib::MakeCudaKernelDescriptor(kernel.data, kernel.size),
        .hip_descriptor = pi::tensorlib::MakeHipKernelDescriptor(kernel.data, kernel.size),
    };
    return descriptor;
}

static kernel_bin_t<kernel_meta_2d_t> SelectMhaBwdPreKernel(const pi::tensorlib::DataType dtype,
                                                            std::string &kernel_name)
{
    kernel_bin_t<kernel_meta_2d_t> kernel;
    switch (dtype)
    {
        case pi::tensorlib::DataType::BFLOAT16:
            kernel = kmha_full_attn_bwd_pre_hs128_bf16;
            kernel_name = "mha_full_attn_bwd_pre_hs128_bf16";
            break;
        case pi::tensorlib::DataType::FLOAT16:
            kernel = kmha_full_attn_bwd_pre_hs128_fp16;
            kernel_name = "mha_full_attn_bwd_pre_hs128_fp16";
            break;
        default:
            throw std::runtime_error("Unsupported data type for mha_full_attention backward preprocess kernel");
    }
    return kernel;
}

static pi::tensorlib::KernelArgumentProvider
MakeMhaAttentionBwdPreArgumentProvider(const pi::tensorlib::DataType dtype,
                                       const kernel_bin_t<kernel_meta_2d_t> &kernel,
                                       const pi::tensorlib::GpuStreamDescriptor &compute_stream_descriptor,
                                       const std::shared_ptr<TritonMhaScratchState> &scratch_state)
{
    const int compute_stream_id = compute_stream_descriptor.getStreamId();
    const auto compute_stream_descriptor_copy = compute_stream_descriptor;
    return [kernel, dtype, compute_stream_id, compute_stream_descriptor_copy,
            scratch_state](const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                               const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
               -> pi::tensorlib::KernelLaunchArguments
    {
        if (inputs.size() != 2)
        {
            throw std::runtime_error("Invalid number of inputs for mha_full_attention bwd preprocess kernel");
        }
        if (outputs.size() != 1)
        {
            throw std::runtime_error("Invalid number of outputs for mha_full_attention bwd preprocess kernel");
        }

        const auto &o = inputs[0];
        const auto &do_tensor = inputs[1];
        const auto &delta = outputs[0];

        if (o->dtype() != dtype || do_tensor->dtype() != dtype)
        {
            throw std::runtime_error("mha_full_attention bwd preprocess tensors must match kernel dtype");
        }
        if (delta->dtype() != pi::tensorlib::DataType::FLOAT32)
        {
            throw std::runtime_error("delta tensor must be FLOAT32");
        }

        if (delta->shape().ndims() != 3)
        {
            throw std::runtime_error("delta tensor must have shape (B, H, T)");
        }

        const auto device_ordinal = ValidateSameDeviceOrdinal("mha_full_attention_bwd_pre", {o, do_tensor, delta});

        const auto batch_size = static_cast<uint32_t>(o->shape()[0]);
        const auto n_ctx = static_cast<uint32_t>(o->shape()[1]);
        const auto n_head = static_cast<uint32_t>(o->shape()[2]);
        const auto head_size = static_cast<uint32_t>(o->shape()[3]);

        if (o->shape().ndims() != 4 || do_tensor->shape().ndims() != 4)
        {
            throw std::runtime_error("output/upstream tensor must have 4 dimensions");
        }
        if (o->shape()[1] != n_ctx || o->shape()[2] != n_head || o->shape()[3] != head_size)
        {
            throw std::runtime_error("output tensor shape must be (B, T, H, HS)");
        }
        if (do_tensor->shape()[1] != n_ctx || do_tensor->shape()[2] != n_head || do_tensor->shape()[3] != head_size)
        {
            throw std::runtime_error("upstream tensor shape must be (B, T, H, HS)");
        }

        RequireContiguousBTHS(o, "output", n_ctx, n_head, head_size, "output tensor must be contiguous (B,T,H,HS)");
        RequireContiguousBTHS(do_tensor, "upstream", n_ctx, n_head, head_size,
                              "upstream tensor must be contiguous (B,T,H,HS)");

        if (do_tensor->shape() != o->shape())
        {
            throw std::runtime_error("upstream tensor shape must match output tensor shape");
        }
        if (delta->shape()[0] != batch_size || delta->shape()[1] != n_head || delta->shape()[2] != n_ctx)
        {
            throw std::runtime_error("delta tensor shape must be (B, H, T)");
        }

        void *o_ptr = o->dataptr();
        void *do_ptr = do_tensor->dataptr();
        void *delta_ptr = delta->dataptr();

        const auto stride_z = static_cast<uint32_t>(o->strides()[0]);
        const auto stride_h = static_cast<uint32_t>(o->strides()[2]);
        const auto stride_tok = static_cast<uint32_t>(o->strides()[1]);
        const auto stride_d = static_cast<uint32_t>(o->strides()[3]);

        const auto stride_delta_z = static_cast<uint32_t>(delta->strides()[0]);
        const auto stride_delta_h = static_cast<uint32_t>(delta->strides()[1]);
        const auto stride_delta_t = static_cast<uint32_t>(delta->strides()[2]);

        if (IsMhaDebugEnabled())
        {
            std::cout << "[MHA] BWD_PRE strides: o=(" << stride_z << ',' << stride_h << ',' << stride_tok << ','
                      << stride_d << ") delta=(" << stride_delta_z << ',' << stride_delta_h << ',' << stride_delta_t
                      << ")\n";
        }

        void *global_scratch_ptr = nullptr;
        std::vector<std::any> args = {o_ptr,          do_ptr,         delta_ptr,     batch_size, n_head,
                                      n_ctx,          stride_z,       stride_h,      stride_tok, stride_d,
                                      stride_delta_z, stride_delta_h, stride_delta_t};

        pi::tensorlib::KernelLaunchArguments arguments{.args = {},
                                                       .grid_dim_x = (n_ctx + kernel.meta.block_size_x - 1) /
                                                                     kernel.meta.block_size_x,
                                                       .grid_dim_y = batch_size * n_head,
                                                       .grid_dim_z = 1,
                                                       .block_dim_x = TRITON_WARP_SIZE * kernel.num_warps,
                                                       .block_dim_y = 1,
                                                       .block_dim_z = 1,
                                                       .shared_mem_bytes = kernel.shared_mem_bytes,
                                                       .device_ordinal = device_ordinal};

        if (kernel.global_scratch_size > 0)
        {
            const uint64_t grid_x = (n_ctx + kernel.meta.block_size_x - 1) / kernel.meta.block_size_x;
            const uint64_t grid = grid_x * n_head * batch_size;
            uint64_t scratch_bytes = static_cast<uint64_t>(kernel.global_scratch_size) * grid;
            const uint64_t scratch_elems = (scratch_bytes + 3) / 4;
            auto scratch_tensor = scratch_state->global;
            if (scratch_tensor && scratch_tensor->shape().numel() < scratch_elems)
            {
                scratch_tensor.reset();
            }
            if (!scratch_tensor)
            {
                scratch_tensor = AllocateMhaScratch({scratch_elems}, pi::tensorlib::DataType::UINT32, o->device(),
                                                    compute_stream_descriptor_copy, false);
                scratch_state->global = scratch_tensor;
            }
            scratch_tensor->storage()->setLastStreamId(compute_stream_id);
            global_scratch_ptr = scratch_tensor->dataptr();
        }

        args.emplace_back(global_scratch_ptr);
        arguments.args = std::move(args);
        return arguments;
    };
}

static pi::tensorlib::ComputeKernelDescriptor
CreateMhaAttentionBwdPreKernelDescriptor(const pi::tensorlib::DataType dtype,
                                         const kernel_bin_t<kernel_meta_2d_t> &kernel, const std::string &kernel_name,
                                         const pi::tensorlib::GpuStreamDescriptor &compute_stream_descriptor)
{
    auto scratch_state = std::make_shared<TritonMhaScratchState>();
    pi::tensorlib::ComputeKernelDescriptor descriptor{
        .kernel_name = kernel_name,
        .function_name = kernel.function_name,
        .expected_arg_count = kernel.arg_count,
        .argument_provider = nullptr,
        .cuda_descriptor = pi::tensorlib::MakeCudaKernelDescriptor(kernel.data, kernel.size),
        .hip_descriptor = pi::tensorlib::MakeHipKernelDescriptor(kernel.data, kernel.size),
    };
    descriptor.argument_provider =
        MakeMhaAttentionBwdPreArgumentProvider(dtype, kernel, compute_stream_descriptor, scratch_state);
    return descriptor;
}

static kernel_bin_t<kernel_meta_2d_t> SelectMhaBwdKernel(const pi::tensorlib::DataType dtype, const bool even_mn,
                                                         std::string &kernel_name)
{
    kernel_bin_t<kernel_meta_2d_t> kernel;
    switch (dtype)
    {
        case pi::tensorlib::DataType::BFLOAT16:
            if (even_mn)
            {
                kernel = kmha_full_attn_bwd_hs128_bf16;
                kernel_name = "mha_full_attn_bwd_hs128_bf16";
            }
            else
            {
                kernel = kmha_full_attn_bwd_hs128_bf16_uneven;
                kernel_name = "mha_full_attn_bwd_hs128_bf16_uneven";
            }
            break;
        case pi::tensorlib::DataType::FLOAT16:
            if (even_mn)
            {
                kernel = kmha_full_attn_bwd_hs128_fp16;
                kernel_name = "mha_full_attn_bwd_hs128_fp16";
            }
            else
            {
                kernel = kmha_full_attn_bwd_hs128_fp16_uneven;
                kernel_name = "mha_full_attn_bwd_hs128_fp16_uneven";
            }
            break;
        default:
            throw std::runtime_error("Unsupported data type for mha_full_attention backward kernel");
    }
    return kernel;
}

static kernel_bin_t<kernel_meta_2d_t> SelectFlashMhaBwdKernel(const pi::tensorlib::DataType dtype, const bool even_mn,
                                                              std::string &kernel_name)
{
#if PI_TENSORLIB_ENABLE_CUDA
    kernel_bin_t<kernel_meta_2d_t> kernel;
    switch (dtype)
    {
        case pi::tensorlib::DataType::BFLOAT16:
            if (even_mn)
            {
                kernel = kmha_full_attn_fa_bwd_hs128_bf16_even;
                kernel_name = "mha_full_attn_fa_bwd_hs128_bf16_even";
            }
            else
            {
                kernel = kmha_full_attn_fa_bwd_hs128_bf16;
                kernel_name = "mha_full_attn_fa_bwd_hs128_bf16";
            }
            break;
        case pi::tensorlib::DataType::FLOAT16:
            if (even_mn)
            {
                kernel = kmha_full_attn_fa_bwd_hs128_fp16_even;
                kernel_name = "mha_full_attn_fa_bwd_hs128_fp16_even";
            }
            else
            {
                kernel = kmha_full_attn_fa_bwd_hs128_fp16;
                kernel_name = "mha_full_attn_fa_bwd_hs128_fp16";
            }
            break;
        default:
            throw std::runtime_error("Unsupported data type for flash attention backward kernel");
    }
    return kernel;
#else
    (void)dtype;
    (void)even_mn;
    (void)kernel_name;
    throw std::runtime_error("Flash attention kernels require CUDA support.");
#endif
}

static kernel_bin_t<kernel_meta_2d_t> SelectFlashMhaBwdDotKernel(const pi::tensorlib::DataType dtype,
                                                                 std::string &kernel_name)
{
#if PI_TENSORLIB_ENABLE_CUDA
    kernel_bin_t<kernel_meta_2d_t> kernel;
    switch (dtype)
    {
        case pi::tensorlib::DataType::BFLOAT16:
            kernel = kmha_full_attn_fa_bwd_dot_do_o_hs128_bf16;
            kernel_name = "mha_full_attn_fa_bwd_dot_do_o_hs128_bf16";
            break;
        case pi::tensorlib::DataType::FLOAT16:
            kernel = kmha_full_attn_fa_bwd_dot_do_o_hs128_fp16;
            kernel_name = "mha_full_attn_fa_bwd_dot_do_o_hs128_fp16";
            break;
        default:
            throw std::runtime_error("Unsupported data type for flash attention backward dot kernel");
    }
    return kernel;
#else
    (void)dtype;
    (void)kernel_name;
    throw std::runtime_error("Flash attention kernels require CUDA support.");
#endif
}

static kernel_bin_t<kernel_meta_2d_t> SelectFlashMhaBwdConvertKernel(const pi::tensorlib::DataType dtype,
                                                                     std::string &kernel_name)
{
#if PI_TENSORLIB_ENABLE_CUDA
    kernel_bin_t<kernel_meta_2d_t> kernel;
    switch (dtype)
    {
        case pi::tensorlib::DataType::BFLOAT16:
            kernel = kmha_full_attn_fa_bwd_convert_dq_hs128_bf16;
            kernel_name = "mha_full_attn_fa_bwd_convert_dq_hs128_bf16";
            break;
        case pi::tensorlib::DataType::FLOAT16:
            kernel = kmha_full_attn_fa_bwd_convert_dq_hs128_fp16;
            kernel_name = "mha_full_attn_fa_bwd_convert_dq_hs128_fp16";
            break;
        default:
            throw std::runtime_error("Unsupported data type for flash attention backward convert kernel");
    }
    return kernel;
#else
    (void)dtype;
    (void)kernel_name;
    throw std::runtime_error("Flash attention kernels require CUDA support.");
#endif
}

static pi::tensorlib::KernelArgumentProvider MakeMhaAttentionBwdArgumentProvider(
    float softmax_scale, const pi::tensorlib::DataType dtype, const kernel_bin_t<kernel_meta_2d_t> &kernel,
    const pi::tensorlib::GpuStreamDescriptor &compute_stream_descriptor, const bool allow_non_even,
    const std::shared_ptr<TritonMhaScratchState> &scratch_state)
{
    const int compute_stream_id = compute_stream_descriptor.getStreamId();
    const auto compute_stream_descriptor_copy = compute_stream_descriptor;
    return [kernel, dtype, softmax_scale, compute_stream_id, compute_stream_descriptor_copy, allow_non_even,
            scratch_state](const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                               const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
               -> pi::tensorlib::KernelLaunchArguments
    {
        if (inputs.size() != 7)
        {
            throw std::runtime_error("Invalid number of inputs for mha_full_attention backward kernel");
        }
        if (outputs.size() != 3)
        {
            throw std::runtime_error("Invalid number of outputs for mha_full_attention backward kernel");
        }

        const auto &q = inputs[0];
        const auto &k = inputs[1];
        const auto &v = inputs[2];
        const auto &o_tensor = inputs[3];
        const auto &do_tensor = inputs[4];
        const auto &m_tensor = inputs[5];
        const auto &delta = inputs[6];

        const auto &dq = outputs[0];
        const auto &dk = outputs[1];
        const auto &dv = outputs[2];

        if (q->dtype() != dtype || k->dtype() != dtype || v->dtype() != dtype || o_tensor->dtype() != dtype ||
            do_tensor->dtype() != dtype)
        {
            throw std::runtime_error("mha_full_attention backward tensors must match kernel dtype");
        }
        if (m_tensor->dtype() != pi::tensorlib::DataType::FLOAT32 || delta->dtype() != pi::tensorlib::DataType::FLOAT32)
        {
            throw std::runtime_error("mha_full_attention backward M/delta tensors must be FLOAT32");
        }
        if (dq->dtype() != dtype || dk->dtype() != dtype || dv->dtype() != dtype)
        {
            throw std::runtime_error("mha_full_attention backward outputs must match kernel dtype");
        }

        RequireContiguousHeadDim(q, "q");
        RequireContiguousHeadDim(k, "k");
        RequireContiguousHeadDim(v, "v");
        RequireContiguousHeadDim(dq, "dq");
        RequireContiguousHeadDim(dk, "dk");
        RequireContiguousHeadDim(dv, "dv");

        if (m_tensor->shape().ndims() != 3 || delta->shape().ndims() != 3)
        {
            throw std::runtime_error("mha_full_attention backward M/delta tensors must have shape (B, H, T)");
        }

        const auto batch_size = static_cast<uint32_t>(q->shape()[0]);
        const auto n_ctx = static_cast<uint32_t>(q->shape()[1]);
        const auto n_head = static_cast<uint32_t>(q->shape()[2]);
        const auto head_size = static_cast<uint32_t>(q->shape()[3]);

        if (k->shape() != q->shape() || v->shape() != q->shape())
        {
            throw std::runtime_error("mha_full_attention backward Q/K/V tensor shapes must match");
        }
        if (dq->shape().ndims() != 4 || dq->shape()[0] != batch_size || dq->shape()[1] != n_ctx ||
            dq->shape()[2] != n_head || dq->shape()[3] != head_size || dk->shape() != dq->shape() ||
            dv->shape() != dq->shape())
        {
            throw std::runtime_error("mha_full_attention backward gradient shapes must be (B, T, H, HS)");
        }
        if (o_tensor->shape()[0] != batch_size || o_tensor->shape()[1] != n_ctx || o_tensor->shape()[2] != n_head ||
            o_tensor->shape()[3] != head_size)
        {
            throw std::runtime_error("mha_full_attention backward output shape must be (B, T, H, HS)");
        }
        if (do_tensor->shape() != o_tensor->shape())
        {
            throw std::runtime_error("mha_full_attention backward upstream must match output shape");
        }
        if (m_tensor->shape()[0] != batch_size || m_tensor->shape()[1] != n_head || m_tensor->shape()[2] != n_ctx ||
            delta->shape()[0] != batch_size || delta->shape()[1] != n_head || delta->shape()[2] != n_ctx)
        {
            throw std::runtime_error("mha_full_attention backward M/delta shapes must be (B, H, T)");
        }

        const auto device_ordinal = ValidateSameDeviceOrdinal(
            "mha_full_attention_bwd", {q, k, v, o_tensor, do_tensor, m_tensor, delta, dq, dk, dv});

        if (!allow_non_even && (n_ctx % kernel.meta.block_size_x != 0))
        {
            throw std::runtime_error("mha_full_attention backward requires N_CTX divisible by " +
                                     std::to_string(kernel.meta.block_size_x));
        }

        const auto row_stride_q = RequirePackedBTHS(q, "q", n_ctx, head_size,
                                                    "mha_full_attention backward requires packed q tensor (B,T,H,HS)");
        const auto row_stride_k = RequirePackedBTHS(k, "k", n_ctx, head_size,
                                                    "mha_full_attention backward requires packed k tensor (B,T,H,HS)");
        const auto row_stride_v = RequirePackedBTHS(v, "v", n_ctx, head_size,
                                                    "mha_full_attention backward requires packed v tensor (B,T,H,HS)");
        if (row_stride_k != row_stride_q || row_stride_v != row_stride_q)
        {
            throw std::runtime_error("mha_full_attention backward requires matching q/k/v row strides");
        }
        const auto row_stride_o =
            RequirePackedBTHS(o_tensor, "output", n_ctx, head_size,
                              "mha_full_attention backward requires packed output tensor (B,T,H,HS)");
        const auto row_stride_do =
            RequirePackedBTHS(do_tensor, "upstream", n_ctx, head_size,
                              "mha_full_attention backward requires packed upstream tensor (B,T,H,HS)");
        const auto row_stride_dq = RequirePackedBTHS(
            dq, "grad_q", n_ctx, head_size, "mha_full_attention backward requires packed grad_q tensor (B,T,H,HS)");
        const auto row_stride_dk = RequirePackedBTHS(
            dk, "grad_k", n_ctx, head_size, "mha_full_attention backward requires packed grad_k tensor (B,T,H,HS)");
        const auto row_stride_dv = RequirePackedBTHS(
            dv, "grad_v", n_ctx, head_size, "mha_full_attention backward requires packed grad_v tensor (B,T,H,HS)");
        if (row_stride_o != row_stride_do || row_stride_dq != row_stride_q || row_stride_dk != row_stride_q ||
            row_stride_dv != row_stride_q)
        {
            throw std::runtime_error("mha_full_attention backward requires output/upstream row strides to match "
                                     "and grad row strides to match q/k/v");
        }
        RequireContiguousBHT(m_tensor, "scratch M", n_ctx, n_head,
                             "mha_full_attention backward requires contiguous scratch M tensor");
        RequireContiguousBHT(delta, "delta", n_ctx, n_head,
                             "mha_full_attention backward requires contiguous delta tensor");

        void *global_scratch_ptr = nullptr;
        std::vector<std::any> args = {q->dataptr(),  k->dataptr(),         v->dataptr(),     o_tensor->dataptr(),
                                      softmax_scale, do_tensor->dataptr(), dq->dataptr(),    dk->dataptr(),
                                      dv->dataptr(), m_tensor->dataptr(),  delta->dataptr(), row_stride_q,
                                      row_stride_do, batch_size,           n_head,           n_ctx};

        const uint32_t grid_x = (n_ctx + kernel.meta.block_size_x - 1) / kernel.meta.block_size_x;
        pi::tensorlib::KernelLaunchArguments arguments{.args = {},
                                                       .grid_dim_x = grid_x,
                                                       .grid_dim_y = 1,
                                                       .grid_dim_z = batch_size * n_head,
                                                       .block_dim_x = TRITON_WARP_SIZE * kernel.num_warps,
                                                       .block_dim_y = 1,
                                                       .block_dim_z = 1,
                                                       .shared_mem_bytes = kernel.shared_mem_bytes,
                                                       .device_ordinal = device_ordinal};

        if (kernel.global_scratch_size > 0)
        {
            const uint64_t grid_x = (n_ctx + kernel.meta.block_size_x - 1) / kernel.meta.block_size_x;
            const uint64_t grid = grid_x * n_head * batch_size;
            uint64_t scratch_bytes = static_cast<uint64_t>(kernel.global_scratch_size) * grid;
            const uint64_t scratch_elems = (scratch_bytes + 3) / 4;
            auto scratch_tensor = scratch_state->global;
            if (scratch_tensor && scratch_tensor->shape().numel() < scratch_elems)
            {
                scratch_tensor.reset();
            }
            if (!scratch_tensor)
            {
                scratch_tensor = AllocateMhaScratch({scratch_elems}, pi::tensorlib::DataType::UINT32, q->device(),
                                                    compute_stream_descriptor_copy, false);
                scratch_state->global = scratch_tensor;
            }
            scratch_tensor->storage()->setLastStreamId(compute_stream_id);
            global_scratch_ptr = scratch_tensor->dataptr();
        }

        args.emplace_back(global_scratch_ptr);
        arguments.args = std::move(args);
        return arguments;
    };
}

static pi::tensorlib::ComputeKernelDescriptor
CreateMhaAttentionBwdKernelDescriptor(const float softmax_scale, const pi::tensorlib::DataType dtype,
                                      const kernel_bin_t<kernel_meta_2d_t> &kernel, const std::string &kernel_name,
                                      const pi::tensorlib::GpuStreamDescriptor &compute_stream_descriptor,
                                      const bool allow_non_even)
{
    auto scratch_state = std::make_shared<TritonMhaScratchState>();
    pi::tensorlib::ComputeKernelDescriptor descriptor{
        .kernel_name = kernel_name,
        .function_name = kernel.function_name,
        .expected_arg_count = kernel.arg_count,
        .argument_provider = nullptr,
        .cuda_descriptor = pi::tensorlib::MakeCudaKernelDescriptor(kernel.data, kernel.size),
        .hip_descriptor = pi::tensorlib::MakeHipKernelDescriptor(kernel.data, kernel.size),
    };
    descriptor.argument_provider = MakeMhaAttentionBwdArgumentProvider(
        softmax_scale, dtype, kernel, compute_stream_descriptor, allow_non_even, scratch_state);
    return descriptor;
}

static pi::tensorlib::KernelArgumentProvider MakeFlashMhaAttentionBwdArgumentProvider(
    float softmax_scale, const pi::tensorlib::DataType dtype, const kernel_bin_t<kernel_meta_2d_t> &kernel,
    const pi::tensorlib::GpuStreamDescriptor &compute_stream_descriptor, FlashBwdKernelKind kind,
    const std::shared_ptr<FlashMhaBwdScratchState> &scratch_state)
{
    const auto compute_stream_descriptor_copy = compute_stream_descriptor;
    return [kernel, softmax_scale, dtype, compute_stream_descriptor_copy, kind,
            scratch_state](const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &inputs,
                           const std::vector<std::shared_ptr<pi::tensorlib::RealTensor>> &outputs)
               -> pi::tensorlib::KernelLaunchArguments
    {
        if (inputs.size() != 7)
        {
            throw std::runtime_error("Invalid number of inputs for mha_full_attention backward kernel");
        }
        if (outputs.size() != 3)
        {
            throw std::runtime_error("Invalid number of outputs for mha_full_attention backward kernel");
        }

        const auto &q = inputs[0];
        const auto &k = inputs[1];
        const auto &v = inputs[2];
        const auto &o_tensor = inputs[3];
        const auto &do_tensor = inputs[4];
        const auto &m_tensor = inputs[5];
        const auto &delta = inputs[6];

        const auto &dq = outputs[0];
        const auto &dk = outputs[1];
        const auto &dv = outputs[2];

        if (q->dtype() != dtype || k->dtype() != dtype || v->dtype() != dtype || o_tensor->dtype() != dtype ||
            do_tensor->dtype() != dtype)
        {
            throw std::runtime_error("mha_full_attention backward tensors must match kernel dtype");
        }
        if (m_tensor->dtype() != pi::tensorlib::DataType::FLOAT32 || delta->dtype() != pi::tensorlib::DataType::FLOAT32)
        {
            throw std::runtime_error("mha_full_attention backward expects FLOAT32 scratch tensors");
        }
        if (dq->dtype() != dtype || dk->dtype() != dtype || dv->dtype() != dtype)
        {
            throw std::runtime_error("mha_full_attention backward outputs must match kernel dtype");
        }

        const auto batch_size = static_cast<uint32_t>(q->shape()[0]);
        const auto n_ctx = static_cast<uint32_t>(q->shape()[1]);
        const auto n_head = static_cast<uint32_t>(q->shape()[2]);
        const auto head_size = static_cast<uint32_t>(q->shape()[3]);

        if (head_size != 128)
        {
            throw std::runtime_error("head_size must be 128 for mha_full_attention backward; got " +
                                     std::to_string(head_size) + " instead");
        }

        if (m_tensor->shape()[0] != batch_size || m_tensor->shape()[1] != n_head || m_tensor->shape()[2] != n_ctx ||
            delta->shape()[0] != batch_size || delta->shape()[1] != n_head || delta->shape()[2] != n_ctx)
        {
            throw std::runtime_error("mha_full_attention backward M/delta shapes must be (B, H, T)");
        }

        const auto device_ordinal = ValidateSameDeviceOrdinal(
            "mha_full_attention_bwd", {q, k, v, o_tensor, do_tensor, m_tensor, delta, dq, dk, dv});

        const auto row_stride_q = RequirePackedBTHS(q, "q", n_ctx, head_size,
                                                    "mha_full_attention backward requires packed q tensor (B,T,H,HS)");
        const auto row_stride_k = RequirePackedBTHS(k, "k", n_ctx, head_size,
                                                    "mha_full_attention backward requires packed k tensor (B,T,H,HS)");
        const auto row_stride_v = RequirePackedBTHS(v, "v", n_ctx, head_size,
                                                    "mha_full_attention backward requires packed v tensor (B,T,H,HS)");
        if (row_stride_k != row_stride_q || row_stride_v != row_stride_q)
        {
            throw std::runtime_error("mha_full_attention backward requires matching q/k/v row strides");
        }

        const auto row_stride_o =
            RequirePackedBTHS(o_tensor, "output", n_ctx, head_size,
                              "mha_full_attention backward requires packed output tensor (B,T,H,HS)");
        const auto row_stride_do =
            RequirePackedBTHS(do_tensor, "upstream", n_ctx, head_size,
                              "mha_full_attention backward requires packed upstream tensor (B,T,H,HS)");
        const auto row_stride_dq = RequirePackedBTHS(
            dq, "grad_q", n_ctx, head_size, "mha_full_attention backward requires packed grad_q tensor (B,T,H,HS)");
        const auto row_stride_dk = RequirePackedBTHS(
            dk, "grad_k", n_ctx, head_size, "mha_full_attention backward requires packed grad_k tensor (B,T,H,HS)");
        const auto row_stride_dv = RequirePackedBTHS(
            dv, "grad_v", n_ctx, head_size, "mha_full_attention backward requires packed grad_v tensor (B,T,H,HS)");
        if (row_stride_o != row_stride_do || row_stride_dq != row_stride_q || row_stride_dk != row_stride_q ||
            row_stride_dv != row_stride_q)
        {
            throw std::runtime_error("mha_full_attention backward requires output/upstream row strides to match "
                                     "and grad row strides to match q/k/v");
        }

        RequireContiguousBHT(m_tensor, "scratch M", n_ctx, n_head,
                             "mha_full_attention backward requires contiguous scratch M tensor");
        RequireContiguousBHT(delta, "delta", n_ctx, n_head,
                             "mha_full_attention backward requires contiguous delta tensor");

        auto softmax_sum_tensor = EnsureFlashMhaBwdSoftmaxSum(*scratch_state, batch_size, n_ctx, n_head, q->device(),
                                                              compute_stream_descriptor_copy);

        std::vector<std::any> args = {q->dataptr(),
                                      k->dataptr(),
                                      v->dataptr(),
                                      o_tensor->dataptr(),
                                      softmax_scale,
                                      do_tensor->dataptr(),
                                      dq->dataptr(),
                                      dk->dataptr(),
                                      dv->dataptr(),
                                      m_tensor->dataptr(),
                                      softmax_sum_tensor->dataptr(),
                                      batch_size,
                                      n_head,
                                      n_ctx,
                                      row_stride_q,
                                      row_stride_do};

        uint64_t grid_x = 0;
        switch (kind)
        {
            case FlashBwdKernelKind::Main:
                grid_x = (n_ctx + kernel.meta.block_size_y - 1) / kernel.meta.block_size_y;
                break;
            case FlashBwdKernelKind::DotDoO:
            case FlashBwdKernelKind::ConvertDQ:
                grid_x = (n_ctx + kernel.meta.block_size_x - 1) / kernel.meta.block_size_x;
                break;
        }

        pi::tensorlib::KernelLaunchArguments arguments{.args = {},
                                                       .grid_dim_x = static_cast<uint32_t>(grid_x),
                                                       .grid_dim_y = batch_size,
                                                       .grid_dim_z = n_head,
                                                       .block_dim_x = CUDA_WARP_SIZE * kernel.num_warps,
                                                       .block_dim_y = 1,
                                                       .block_dim_z = 1,
                                                       .shared_mem_bytes = kernel.shared_mem_bytes,
                                                       .device_ordinal = device_ordinal};

        auto scratch_tensor = EnsureFlashMhaBwdScratch(*scratch_state, batch_size, n_ctx, n_head, head_size,
                                                       q->device(), compute_stream_descriptor_copy);

        args.emplace_back(scratch_tensor->dataptr());
        arguments.args = std::move(args);
        return arguments;
    };
}

static pi::tensorlib::ComputeKernelDescriptor
CreateFlashMhaAttentionBwdKernelDescriptor(const kernel_bin_t<kernel_meta_2d_t> &kernel, const std::string &kernel_name,
                                           const std::string &function_name)
{
    pi::tensorlib::ComputeKernelDescriptor descriptor{
        .kernel_name = kernel_name,
        .function_name = function_name,
        .expected_arg_count = kernel.arg_count,
        .argument_provider = nullptr,
        .cuda_descriptor = pi::tensorlib::MakeCudaKernelDescriptor(kernel.data, kernel.size),
        .hip_descriptor = pi::tensorlib::MakeHipKernelDescriptor(kernel.data, kernel.size),
    };
    return descriptor;
}

void MhaAttentionImplPass::transform(pi::tensorlib::ExecutionPlan &execution_plan)
{
    FlashMhaBwdScratchPool flash_bwd_scratch{};

    for (size_t entry_idx = 0; entry_idx < execution_plan.entries.size(); ++entry_idx)
    {
        auto &mha_op = execution_plan.entries[entry_idx];
        if (mha_op.op_type == pi::tensorlib::OpType::MHA_ATTN_FWD)
        {
            if (mha_op.inputs.size() != 3)
            {
                continue;
            }
            if (mha_op.outputs.size() != 1 && mha_op.outputs.size() != 2)
            {
                continue;
            }
            const auto &input = mha_op.inputs[0];
            const auto dtype = input->dtype();
            if (!IsSupportedHalfType(dtype))
            {
                continue;
            }

            const auto &attributes = mha_op.attributes;

            const auto softmax_scale = std::any_cast<float>(attributes.at("softmax_scale"));
            const auto compute_stream_descriptor = mha_op.gpu_stream_desc;
            if (!compute_stream_descriptor.isValid())
            {
                throw std::runtime_error("MHA attention forward requires a valid compute stream descriptor");
            }

            if (const auto causal = std::any_cast<bool>(attributes.at("causal")); !causal) // full attention
            {
                bool requested_fp16_accumulation = false;
                if (const auto attr_it = attributes.find("use_fp16_flash_attn_acc");
                    attr_it != attributes.end() && attr_it->second.type() == typeid(bool))
                {
                    requested_fp16_accumulation = std::any_cast<bool>(attr_it->second);
                }
                if (dtype == pi::tensorlib::DataType::FLOAT16)
                {
                    RequireFp16AccumulationSupported(requested_fp16_accumulation);
                }
                const bool use_fp16_accumulation =
                    (dtype == pi::tensorlib::DataType::FLOAT16) ? requested_fp16_accumulation : false;
                const auto &q = mha_op.inputs[0];
                const auto &k = mha_op.inputs[1];
                const auto &v = mha_op.inputs[2];
                const auto &o = mha_op.outputs[0];

                if (q->dtype() != dtype || k->dtype() != dtype || v->dtype() != dtype || o->dtype() != dtype)
                {
                    throw std::runtime_error("mha_full_attention tensors must match kernel dtype");
                }

                const bool write_lse = (mha_op.outputs.size() == 2);
                std::shared_ptr<pi::tensorlib::RealTensor> scratch_m{};
                if (write_lse)
                {
                    scratch_m = mha_op.outputs[1];
                    if (scratch_m->shape().ndims() != 3)
                    {
                        throw std::runtime_error("scratch tensor must have shape (B, H, T)");
                    }
                }

                const auto batch_size = static_cast<uint32_t>(q->shape()[0]);
                const auto n_ctx = static_cast<uint32_t>(q->shape()[1]);
                const auto n_head = static_cast<uint32_t>(q->shape()[2]);
                const auto head_size = static_cast<uint32_t>(q->shape()[3]);

                if (write_lse && scratch_m)
                {
                    RequireContiguousBHT(scratch_m, "scratch M", n_ctx, n_head,
                                         "scratch tensor must be contiguous (B,H,T)");
                }

                const auto row_stride_q =
                    RequirePackedBTHS(q, "q", n_ctx, head_size, "q tensor must be packed (B,T,H,HS)");
                const auto row_stride_k =
                    RequirePackedBTHS(k, "k", n_ctx, head_size, "k tensor must be packed (B,T,H,HS)");
                const auto row_stride_v =
                    RequirePackedBTHS(v, "v", n_ctx, head_size, "v tensor must be packed (B,T,H,HS)");
                if (row_stride_k != row_stride_q || row_stride_v != row_stride_q)
                {
                    throw std::runtime_error("q/k/v row strides must match for MHA kernel");
                }
                {
                    if (o->shape().ndims() != 4)
                    {
                        throw std::runtime_error("output tensor must have 4 dimensions");
                    }
                    if (o->shape()[0] != batch_size || o->shape()[1] != n_ctx || o->shape()[2] != n_head ||
                        o->shape()[3] != head_size)
                    {
                        throw std::runtime_error("output tensor shape must be (B, T, H, HS)");
                    }
                    RequireContiguousBTHS(o, "output", n_ctx, n_head, head_size,
                                          "output tensor must be contiguous (B,T,H,HS)");
                }

                const uint64_t b = batch_size;
                const uint64_t h = n_head;
                const uint64_t t = n_ctx;
                const uint64_t hs = head_size;
                const uint64_t qk_flops = 2 * b * h * t * t * hs;
                const uint64_t pv_flops = 2 * b * h * t * t * hs;
                const uint64_t softmax_flops = 3 * b * h * t * t; // exp + sum + scale
                const uint64_t flop_estimate = qk_flops + pv_flops + softmax_flops;

                mha_op.op_type = std::nullopt; // mark as kernel

                const auto prefer_backend = GetMhaBackendPreference();
                kernel_bin_t<kernel_meta_2d_t> flash_kernel{};
                std::string flash_kernel_name{};
                bool flash_available = false;
#if PI_TENSORLIB_ENABLE_CUDA
                flash_kernel =
                    SelectFlashMhaFwdKernel(dtype, use_fp16_accumulation, write_lse, false, flash_kernel_name);
                const bool flash_even_mn =
                    IsFlashEvenMn(n_ctx, flash_kernel.meta.block_size_x, flash_kernel.meta.block_size_y);
                flash_kernel =
                    SelectFlashMhaFwdKernel(dtype, use_fp16_accumulation, write_lse, flash_even_mn, flash_kernel_name);
                flash_available = true;
#endif

                const auto *candidate_cutlass =
                    SelectCutlassMhaKernel(write_lse ? kCutlassMhaKernelsLse : kCutlassMhaKernelsNoLse, dtype);
                const kernel_bin_t<kernel_meta_mha_cutlass_t> *cutlass_kernel = nullptr;
                bool cutlass_available = false;
                std::string cutlass_skip_reason{};

                if (candidate_cutlass != nullptr && write_lse)
                {
                    if ((n_ctx % kCutlassLseAlign) != 0)
                    {
                        cutlass_skip_reason = "sequence length must be aligned to " + std::to_string(kCutlassLseAlign);
                    }
                    else if ((n_head % 4u) != 0)
                    {
                        cutlass_skip_reason =
                            "num_heads=" + std::to_string(n_head) + " not supported by CUTLASS kernel";
                    }
                    else if (head_size == candidate_cutlass->meta.head_dim)
                    {
                        cutlass_kernel = candidate_cutlass;
                        cutlass_available = true;
                    }
                    else
                    {
                        cutlass_skip_reason = "head_dim mismatch (tensor=" + std::to_string(head_size) +
                                              ", kernel=" + std::to_string(candidate_cutlass->meta.head_dim) + ")";
                    }
                }
                else
                {
                    if (use_fp16_accumulation && dtype == pi::tensorlib::DataType::FLOAT16)
                    {
                        cutlass_skip_reason = "requested FP16 accumulation not supported by CUTLASS variant";
                    }
                    else
                    {
                        cutlass_skip_reason = write_lse ? "no CUTLASS kernel available for requested dtype"
                                                        : "no CUTLASS kernel available for attention without LSE";
                    }
                }

                if (IsMhaDebugEnabled() && prefer_backend == MhaBackendPreference::Cutlass && !cutlass_available &&
                    !cutlass_skip_reason.empty())
                {
                    std::cout << "[MHA] CUTLASS fwd unavailable: " << cutlass_skip_reason << '\n';
                }

                const auto select_backend = [&]() -> MhaBackendPreference
                {
                    switch (prefer_backend)
                    {
                        case MhaBackendPreference::Triton:
                            return MhaBackendPreference::Triton;
                        case MhaBackendPreference::Cutlass:
                            if (cutlass_available)
                            {
                                return MhaBackendPreference::Cutlass;
                            }
                            if (flash_available)
                            {
                                return MhaBackendPreference::Flash;
                            }
                            return MhaBackendPreference::Triton;
                        case MhaBackendPreference::Flash:
                        default:
                            if (flash_available)
                            {
                                return MhaBackendPreference::Flash;
                            }
                            if (cutlass_available)
                            {
                                return MhaBackendPreference::Cutlass;
                            }
                            return MhaBackendPreference::Triton;
                    }
                }();
                const float kernel_softmax_scale = (select_backend == MhaBackendPreference::Triton)
                                                       ? softmax_scale * kSoftmaxScaleLog2e
                                                       : softmax_scale;

                if (select_backend == MhaBackendPreference::Cutlass)
                {
                    if (cutlass_kernel == nullptr)
                    {
                        throw std::runtime_error("Internal error: cutlass_kernel is null despite cutlass_available");
                    }
                    LogMhaShape("fwd", "cutlass", batch_size, n_head, n_ctx, head_size);
                    if (IsMhaDebugEnabled())
                    {
                        if (use_fp16_accumulation && dtype == pi::tensorlib::DataType::FLOAT16)
                        {
                            std::cout << "[MHA] Using CUTLASS kernel despite requested FP16 accumulation flag\n";
                        }
                        std::cout << "[MHA] Strides: q=(" << q->strides()[0] << ',' << q->strides()[1] << ','
                                  << q->strides()[2] << ',' << q->strides()[3] << ") k=(" << k->strides()[0] << ','
                                  << k->strides()[1] << ',' << k->strides()[2] << ',' << k->strides()[3] << ") v=("
                                  << v->strides()[0] << ',' << v->strides()[1] << ',' << v->strides()[2] << ','
                                  << v->strides()[3] << ") o=(" << o->strides()[0] << ',' << o->strides()[1] << ','
                                  << o->strides()[2] << ',' << o->strides()[3] << ")" << '\n';
                    }
                    LogMhaKernelSelection("fwd", "cutlass", cutlass_kernel->function_name);
                    mha_op.kernel_descriptor = CreateCutlassMhaKernelDescriptor(softmax_scale, dtype, *cutlass_kernel);
                }
                else if (select_backend == MhaBackendPreference::Flash)
                {
                    LogMhaShape("fwd", "flash", batch_size, n_head, n_ctx, head_size);
                    LogMhaKernelSelection("fwd", "flash", flash_kernel_name);
                    mha_op.kernel_descriptor = CreateFlashMhaFullAttentionComputeKernelDescriptor(
                        softmax_scale, dtype, flash_kernel, flash_kernel_name, compute_stream_descriptor);
                    auto &flash_descriptor = *mha_op.kernel_descriptor;
                    flash_descriptor.argument_provider = MakeFlashMhaFullAttentionArgumentProvider(
                        softmax_scale, dtype, flash_kernel, compute_stream_descriptor);
                }
                else
                {
                    LogMhaShape("fwd", "triton", batch_size, n_head, n_ctx, head_size);
                    if (IsMhaDebugEnabled())
                    {
                        if (!cutlass_skip_reason.empty())
                        {
                            std::cout << "[MHA] Skipping CUTLASS kernel: " << cutlass_skip_reason << '\n';
                        }
                    }
                    std::string triton_kernel_name{};
                    auto triton_kernel =
                        SelectMhaFwdKernel(dtype, use_fp16_accumulation, write_lse, true, triton_kernel_name);
                    const bool triton_even_mn =
                        IsFlashEvenMn(n_ctx, triton_kernel.meta.block_size_x, triton_kernel.meta.block_size_y);
                    triton_kernel =
                        SelectMhaFwdKernel(dtype, use_fp16_accumulation, write_lse, triton_even_mn, triton_kernel_name);
                    LogMhaKernelSelection("fwd", "triton", triton_kernel_name);
                    mha_op.kernel_descriptor = CreateMhaFullAttentionComputeKernelDescriptor(
                        kernel_softmax_scale, dtype, triton_kernel, triton_kernel_name, compute_stream_descriptor);
                }
                mha_op.flop_estimate = flop_estimate;
            }
            continue;
        }
        if (mha_op.op_type == pi::tensorlib::OpType::MHA_ATTN_BWD_PRE)
        {
            if (mha_op.inputs.size() != 2 || mha_op.outputs.size() != 1)
            {
                continue;
            }

            const auto dtype = mha_op.inputs[0]->dtype();
            if (!IsSupportedHalfType(dtype))
            {
                continue;
            }

            std::string kernel_name{};
            auto kernel = SelectMhaBwdPreKernel(dtype, kernel_name);
            const auto compute_stream_descriptor = mha_op.gpu_stream_desc;
            if (!compute_stream_descriptor.isValid())
            {
                throw std::runtime_error("MHA attention backward prepass requires a valid compute stream descriptor");
            }
            mha_op.op_type = std::nullopt;
            mha_op.kernel_descriptor =
                CreateMhaAttentionBwdPreKernelDescriptor(dtype, kernel, kernel_name, compute_stream_descriptor);
        }
        if (mha_op.op_type == pi::tensorlib::OpType::MHA_ATTN_BWD)
        {
            if (mha_op.inputs.size() != 7 || mha_op.outputs.size() != 3)
            {
                continue;
            }
            const auto dtype = mha_op.inputs[0]->dtype();
            if (!IsSupportedHalfType(dtype))
            {
                continue;
            }

            const auto &attributes = mha_op.attributes;
            const auto softmax_scale = std::any_cast<float>(attributes.at("softmax_scale"));
            const auto causal = std::any_cast<bool>(attributes.at("causal"));
            if (causal)
            {
                throw std::runtime_error("mha_full_attention backward does not support causal attention");
            }

            const auto compute_stream_descriptor = mha_op.gpu_stream_desc;
            if (!compute_stream_descriptor.isValid())
            {
                throw std::runtime_error("MHA attention backward requires a valid compute stream descriptor");
            }
            mha_op.op_type = std::nullopt;

            const auto &q = mha_op.inputs[0];
            const auto batch_size = static_cast<uint32_t>(q->shape()[0]);
            const auto n_ctx = static_cast<uint32_t>(q->shape()[1]);
            const auto n_head = static_cast<uint32_t>(q->shape()[2]);
            const auto head_size = static_cast<uint32_t>(q->shape()[3]);

            const auto prefer_backend = GetMhaBackendPreference();
            kernel_bin_t<kernel_meta_2d_t> flash_kernel{};
            std::string flash_kernel_name{};
            bool flash_available = false;
#if PI_TENSORLIB_ENABLE_CUDA
            flash_kernel = SelectFlashMhaBwdKernel(dtype, false, flash_kernel_name);
            const bool flash_even_mn =
                IsFlashEvenMn(n_ctx, flash_kernel.meta.block_size_x, flash_kernel.meta.block_size_y);
            flash_kernel = SelectFlashMhaBwdKernel(dtype, flash_even_mn, flash_kernel_name);
            flash_available = true;
#endif

            const auto *candidate_cutlass = SelectCutlassMhaBwdKernel(kCutlassMhaBwdKernels, dtype);
            const kernel_bin_t<kernel_meta_mha_cutlass_bwd_t> *cutlass_kernel = nullptr;
            bool cutlass_available = false;
            std::string cutlass_skip_reason{};
            if (candidate_cutlass != nullptr)
            {
                if ((n_head % 4u) != 0)
                {
                    cutlass_skip_reason = "num_heads=" + std::to_string(n_head) + " not supported by CUTLASS kernel";
                }
                else if (head_size == candidate_cutlass->meta.head_dim)
                {
                    cutlass_kernel = candidate_cutlass;
                    cutlass_available = true;
                }
                else
                {
                    cutlass_skip_reason = "head_dim mismatch (tensor=" + std::to_string(head_size) +
                                          ", kernel=" + std::to_string(candidate_cutlass->meta.head_dim) + ")";
                }
            }
            else
            {
                cutlass_skip_reason = "no CUTLASS kernel available for requested dtype";
            }

            if (IsMhaDebugEnabled() && prefer_backend == MhaBackendPreference::Cutlass && !cutlass_available &&
                !cutlass_skip_reason.empty())
            {
                std::cout << "[MHA] CUTLASS bwd unavailable: " << cutlass_skip_reason << '\n';
            }

            const auto select_backend = [&]() -> MhaBackendPreference
            {
                switch (prefer_backend)
                {
                    case MhaBackendPreference::Triton:
                        return MhaBackendPreference::Triton;
                    case MhaBackendPreference::Cutlass:
                        if (cutlass_available)
                        {
                            return MhaBackendPreference::Cutlass;
                        }
                        if (flash_available)
                        {
                            return MhaBackendPreference::Flash;
                        }
                        return MhaBackendPreference::Triton;
                    case MhaBackendPreference::Flash:
                    default:
                        if (flash_available)
                        {
                            return MhaBackendPreference::Flash;
                        }
                        if (cutlass_available)
                        {
                            return MhaBackendPreference::Cutlass;
                        }
                        return MhaBackendPreference::Triton;
                }
            }();
            if (select_backend == MhaBackendPreference::Cutlass)
            {
                if (cutlass_kernel == nullptr)
                {
                    throw std::runtime_error("Internal error: cutlass_kernel is null despite cutlass_available");
                }
                LogMhaShape("bwd", "cutlass", batch_size, n_head, n_ctx, head_size);
                LogMhaKernelSelection("bwd", "cutlass", cutlass_kernel->function_name);
                const auto layout =
                    ComputeCutlassMhaBwdWorkspaceLayout(batch_size, n_head, n_ctx, head_size, cutlass_kernel->meta);
                const uint64_t workspace_bytes = layout.bytes;
                const bool reset_workspace = RequiresCutlassBwdWorkspaceReset(layout);

                std::shared_ptr<CutlassMhaBwdWorkspaceState> workspace_state{};
                if (workspace_bytes > 0)
                {
                    auto cache_entry = GetCutlassMhaBwdWorkspace(q->device(), compute_stream_descriptor, layout);
                    if (!cache_entry)
                    {
                        throw std::runtime_error("Failed to allocate CUTLASS MHA bwd workspace.");
                    }
                    workspace_state = std::make_shared<CutlassMhaBwdWorkspaceState>(CutlassMhaBwdWorkspaceState{
                        .cache_entry = cache_entry,
                        .cleanup_stream_desc = compute_stream_descriptor,
                    });
                }

                mha_op.kernel_descriptor = CreateCutlassMhaBwdKernelDescriptor(
                    softmax_scale, dtype, *cutlass_kernel, compute_stream_descriptor, workspace_state);

                if (workspace_state && workspace_state->cache_entry && reset_workspace)
                {
                    pi::tensorlib::ExecutionEntry copy_template{
                        .op_type = pi::tensorlib::OpType::DEVICE_COPY,
                        .inputs = {workspace_state->cache_entry->template_tensor},
                        .outputs = {workspace_state->cache_entry->workspace},
                        .attributes = {},
                        .gpu_stream_desc = compute_stream_descriptor};

                    const auto insert_index = static_cast<std::ptrdiff_t>(entry_idx + 1);
                    execution_plan.entries.insert(execution_plan.entries.begin() + insert_index, copy_template);
                    entry_idx += 1;
                }
            }
            else if (select_backend == MhaBackendPreference::Flash)
            {
                LogMhaShape("bwd", "flash", batch_size, n_head, n_ctx, head_size);
                LogMhaKernelSelection("bwd", "flash", flash_kernel_name);

                // Flash bwd kernels compute delta internally, so the prepass becomes redundant.
                RemoveFlashBwdPreEntry(execution_plan, entry_idx, mha_op.inputs[6]);

                std::string dot_kernel_name{};
                auto dot_kernel = SelectFlashMhaBwdDotKernel(dtype, dot_kernel_name);
                std::string convert_kernel_name{};
                auto convert_kernel = SelectFlashMhaBwdConvertKernel(dtype, convert_kernel_name);
                auto scratch_state =
                    GetFlashBwdScratch(flash_bwd_scratch, mha_op.inputs[0]->device(), compute_stream_descriptor);
                auto softmax_sum = EnsureFlashMhaBwdSoftmaxSum(*scratch_state, batch_size, n_ctx, n_head,
                                                               mha_op.inputs[0]->device(), compute_stream_descriptor);
                auto dq_accum = EnsureFlashMhaBwdScratch(*scratch_state, batch_size, n_ctx, n_head, head_size,
                                                         mha_op.inputs[0]->device(), compute_stream_descriptor);

                const float kernel_softmax_scale = softmax_scale;
                mha_op.kernel_descriptor =
                    CreateFlashMhaAttentionBwdKernelDescriptor(flash_kernel, flash_kernel_name, "mha_attn_bwd");
                auto &flash_descriptor = *mha_op.kernel_descriptor;
                flash_descriptor.argument_provider = MakeFlashMhaAttentionBwdArgumentProvider(
                    kernel_softmax_scale, dtype, flash_kernel, compute_stream_descriptor, FlashBwdKernelKind::Main,
                    scratch_state);
                const auto inputs = mha_op.inputs;
                const auto outputs = mha_op.outputs;
                const auto map = mha_op.attributes;

                pi::tensorlib::ExecutionEntry dot_entry{.op_type = std::nullopt,
                                                        .kernel_descriptor = CreateFlashMhaAttentionBwdKernelDescriptor(
                                                            dot_kernel, dot_kernel_name, "mha_attn_bwd_dot_do_o"),
                                                        .inputs = inputs,
                                                        .outputs = outputs,
                                                        .attributes = map,
                                                        .gpu_stream_desc = mha_op.gpu_stream_desc};
                if (dot_entry.kernel_descriptor)
                {
                    auto &dot_desc = *dot_entry.kernel_descriptor;
                    dot_desc.argument_provider = MakeFlashMhaAttentionBwdArgumentProvider(
                        kernel_softmax_scale, dtype, dot_kernel, compute_stream_descriptor, FlashBwdKernelKind::DotDoO,
                        scratch_state);
                }

                pi::tensorlib::ExecutionEntry convert_entry{
                    .op_type = std::nullopt,
                    .kernel_descriptor = CreateFlashMhaAttentionBwdKernelDescriptor(convert_kernel, convert_kernel_name,
                                                                                    "mha_attn_bwd_convert_dq"),
                    .inputs = inputs,
                    .outputs = outputs,
                    .attributes = map,
                    .gpu_stream_desc = mha_op.gpu_stream_desc};
                if (convert_entry.kernel_descriptor)
                {
                    auto &convert_desc = *convert_entry.kernel_descriptor;
                    convert_desc.argument_provider = MakeFlashMhaAttentionBwdArgumentProvider(
                        kernel_softmax_scale, dtype, convert_kernel, compute_stream_descriptor,
                        FlashBwdKernelKind::ConvertDQ, scratch_state);
                }

                pi::tensorlib::ExecutionEntry zero_softmax_entry{.op_type = std::nullopt,
                                                                 .kernel_descriptor = CreateFillZerosKernelDescriptor(),
                                                                 .inputs = {softmax_sum},
                                                                 .outputs = {softmax_sum},
                                                                 .attributes = {},
                                                                 .gpu_stream_desc = compute_stream_descriptor};

                pi::tensorlib::ExecutionEntry zero_entry{.op_type = std::nullopt,
                                                         .kernel_descriptor = CreateFillZerosKernelDescriptor(),
                                                         .inputs = {dq_accum},
                                                         .outputs = {dq_accum},
                                                         .attributes = {},
                                                         .gpu_stream_desc = compute_stream_descriptor};

                const auto insert_index = static_cast<std::ptrdiff_t>(entry_idx);
                execution_plan.entries.insert(execution_plan.entries.begin() + insert_index, zero_softmax_entry);
                execution_plan.entries.insert(execution_plan.entries.begin() + insert_index + 1, zero_entry);
                execution_plan.entries.insert(execution_plan.entries.begin() + insert_index + 2, dot_entry);
                execution_plan.entries.insert(execution_plan.entries.begin() + insert_index + 4, convert_entry);
                entry_idx += 4;
            }
            else
            {
                LogMhaShape("bwd", "triton", batch_size, n_head, n_ctx, head_size);
                if (IsMhaDebugEnabled())
                {
                    if (!cutlass_skip_reason.empty())
                    {
                        std::cout << "[MHA] Skipping CUTLASS bwd kernel: " << cutlass_skip_reason << '\n';
                    }
                }
                std::string kernel_name{};
                auto kernel = SelectMhaBwdKernel(dtype, true, kernel_name);
                const bool even_mn = (n_ctx % kernel.meta.block_size_x) == 0;
                if (!even_mn)
                {
                    kernel = SelectMhaBwdKernel(dtype, false, kernel_name);
                }
                LogMhaKernelSelection("bwd", "triton", kernel_name);
                mha_op.kernel_descriptor = CreateMhaAttentionBwdKernelDescriptor(
                    softmax_scale, dtype, kernel, kernel_name, compute_stream_descriptor, !even_mn);
            }
        }
    }
}
