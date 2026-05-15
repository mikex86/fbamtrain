#pragma once

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C"
{
#endif

    // Matches labels produced by ASM acro EMBED_ASSET(name, "path", align)
#define EMBED_ASSET_DECLARE(name)                                                                                      \
    extern const char name##_function_name[];                                                                          \
    extern const unsigned char name##_start[];                                                                         \
    extern const unsigned char name##_end[];

#define EMBED_KINFO_DECLARE(name)                                                                                      \
    extern const uint32_t name##_smem_bytes;                                                                           \
    extern const uint32_t name##_num_warps;                                                                            \
    extern const uint32_t name##_arg_count;                                                                            \
    extern const uint32_t name##_global_scratch_size;

#define EMBED_ELEMWISE_KINFO_DECLARE(name) extern const uint32_t name##_block_size;

#define EMBED_GEMM_KINFO_DECLARE(name)                                                                                 \
    extern const uint32_t name##_block_size_m;                                                                         \
    extern const uint32_t name##_block_size_n;                                                                         \
    extern const uint32_t name##_block_size_k;                                                                         \
    extern const uint32_t name##_swizzle_size;

#define EMBED_2D_KINFO_DECLARE(name)                                                                                   \
    extern const uint32_t name##_block_size_x;                                                                         \
    extern const uint32_t name##_block_size_y;

#define EMBED_3D_KINFO_DECLARE(name)                                                                                   \
    extern const uint32_t name##_block_size_x;                                                                         \
    extern const uint32_t name##_block_size_y;                                                                         \
    extern const uint32_t name##_block_size_z;

#define EMBED_4D_KINFO_DECLARE(name)                                                                                   \
    extern const uint32_t name##_block_size_x;                                                                         \
    extern const uint32_t name##_block_size_y;                                                                         \
    extern const uint32_t name##_block_size_z;                                                                         \
    extern const uint32_t name##_block_size_w;

#define EMBED_CONV2D_KINFO_DECLARE(name)                                                                               \
    extern const uint32_t name##_block_pixels;                                                                         \
    extern const uint32_t name##_block_oc;                                                                             \
    extern const uint32_t name##_block_k;                                                                              \
    extern const uint32_t name##_in_channels;                                                                          \
    extern const uint32_t name##_kernel_h;                                                                             \
    extern const uint32_t name##_kernel_w;

#define EMBED_CONV2D_CUTLASS_KINFO_DECLARE(name)                                                                       \
    extern const uint32_t name##_block_pixels;                                                                         \
    extern const uint32_t name##_block_oc;                                                                             \
    extern const uint32_t name##_block_k;                                                                              \
    extern const uint32_t name##_in_channels;                                                                          \
    extern const uint32_t name##_kernel_h;                                                                             \
    extern const uint32_t name##_kernel_w;                                                                             \
    extern const uint32_t name##_stride_h;                                                                             \
    extern const uint32_t name##_stride_w;                                                                             \
    extern const uint32_t name##_padding_h;                                                                            \
    extern const uint32_t name##_padding_w;                                                                            \
    extern const uint32_t name##_dilation_h;                                                                           \
    extern const uint32_t name##_dilation_w;                                                                           \
    extern const uint32_t name##_groups;

#define EMBED_MHA_CUTLASS_KINFO_DECLARE(name)                                                                          \
    extern const uint32_t name##_block_size_x;                                                                         \
    extern const uint32_t name##_block_size_y;                                                                         \
    extern const uint32_t name##_head_dim;

#define EMBED_MHA_CUTLASS_BWD_KINFO_DECLARE(name)                                                                      \
    extern const uint32_t name##_block_size_i;                                                                         \
    extern const uint32_t name##_block_size_j;                                                                         \
    extern const uint32_t name##_head_dim;                                                                             \
    extern const uint32_t name##_gradq_tile_elements;                                                                  \
    extern const uint32_t name##_gradq_temp_bytes;

#ifdef __cplusplus
} // extern "C"
#endif

struct kernel_meta_gemm_t
{
    uint32_t block_size_m{};
    uint32_t block_size_n{};
    uint32_t block_size_k{};
    uint32_t swizzle_size{};
};

struct kernel_meta_2d_t
{
    uint32_t block_size_x{};
    uint32_t block_size_y{};
};

struct kernel_meta_3d_t
{
    uint32_t block_size_x{};
    uint32_t block_size_y{};
    uint32_t block_size_z{};
};

struct kernel_meta_4d_t
{
    uint32_t block_size_x{};
    uint32_t block_size_y{};
    uint32_t block_size_z{};
    uint32_t block_size_w{};
};

struct kernel_meta_elementwise_t
{
    uint32_t block_size{};
};

struct kernel_meta_pool1d_t
{
    uint32_t block_size{};
    uint32_t max_kernel_size{};
};

struct kernel_meta_pool2d_t
{
    uint32_t block_size_h{};
    uint32_t block_size_w{};
    uint32_t max_kernel_h{};
    uint32_t max_kernel_w{};
};

struct kernel_meta_conv2d_t
{
    uint32_t block_pixels{};
    uint32_t block_oc{};
    uint32_t block_k{};
    uint32_t in_channels{};
    uint32_t kernel_h{};
    uint32_t kernel_w{};
};

struct kernel_meta_conv2d_cutlass_t
{
    uint32_t block_pixels{};
    uint32_t block_oc{};
    uint32_t block_k{};
    uint32_t in_channels{};
    uint32_t kernel_h{};
    uint32_t kernel_w{};
    uint32_t stride_h{};
    uint32_t stride_w{};
    uint32_t padding_h{};
    uint32_t padding_w{};
    uint32_t dilation_h{};
    uint32_t dilation_w{};
    uint32_t groups{};
};

struct kernel_meta_mha_cutlass_t
{
    uint32_t block_size_x{};
    uint32_t block_size_y{};
    uint32_t head_dim{};
};

struct kernel_meta_mha_cutlass_bwd_t
{
    uint32_t block_size_i{};
    uint32_t block_size_j{};
    uint32_t head_dim{};
    uint32_t gradq_tile_elements{};
    uint32_t gradq_temp_bytes{};
};

template <typename Meta> struct kernel_bin_t
{
    char *kernel_name{};
    char *function_name{};
    unsigned char *data{};
    size_t size{};
    uint32_t shared_mem_bytes{};
    uint32_t num_warps{};
    uint32_t arg_count{};
    uint32_t global_scratch_size{};
    Meta meta{};
};

#ifndef KERNEL_BINARY_PROTOTYPES_ONLY
#define DECLARE_GEMM_KERNEL_BINARY(name)                                                                               \
    EMBED_ASSET_DECLARE(name)                                                                                          \
    EMBED_KINFO_DECLARE(name)                                                                                          \
    EMBED_GEMM_KINFO_DECLARE(name)                                                                                     \
    struct kernel_bin_t<kernel_meta_gemm_t> k##name{                                                                   \
        (char *)#name,                                                                                                 \
        (char *)name##_function_name,                                                                                  \
        (unsigned char *)name##_start,                                                                                 \
        (size_t)(name##_end - name##_start),                                                                           \
        name##_smem_bytes,                                                                                             \
        name##_num_warps,                                                                                              \
        name##_arg_count,                                                                                              \
        name##_global_scratch_size,                                                                                    \
        kernel_meta_gemm_t{name##_block_size_m, name##_block_size_n, name##_block_size_k, name##_swizzle_size}};
#else
#define DECLARE_GEMM_KERNEL_BINARY(name) extern struct kernel_bin_t<kernel_meta_gemm_t> k##name;
#endif

#ifndef KERNEL_BINARY_PROTOTYPES_ONLY
#define DECLARE_ELEMWISE_KERNEL_BINARY(name)                                                                           \
    EMBED_ASSET_DECLARE(name)                                                                                          \
    EMBED_KINFO_DECLARE(name)                                                                                          \
    EMBED_ELEMWISE_KINFO_DECLARE(name)                                                                                 \
    struct kernel_bin_t<kernel_meta_elementwise_t> k##name{(char *)#name,                                               \
                                                           (char *)name##_function_name,                               \
                                                           (unsigned char *)name##_start,                              \
                                                           (size_t)(name##_end - name##_start),                        \
                                                           name##_smem_bytes,                                          \
                                                           name##_num_warps,                                           \
                                                           name##_arg_count,                                           \
                                                           name##_global_scratch_size,                                 \
                                                           kernel_meta_elementwise_t{name##_block_size}};
#else
#define DECLARE_ELEMWISE_KERNEL_BINARY(name) extern struct kernel_bin_t<kernel_meta_elementwise_t> k##name;
#endif

#define EMBED_POOL1D_KINFO_DECLARE(name)                                                                               \
    extern const uint32_t name##_block_size;                                                                           \
    extern const uint32_t name##_max_kernel_size;

#ifndef KERNEL_BINARY_PROTOTYPES_ONLY
#define DECLARE_POOL1D_KERNEL_BINARY(name)                                                                             \
    EMBED_ASSET_DECLARE(name)                                                                                          \
    EMBED_KINFO_DECLARE(name)                                                                                          \
    EMBED_POOL1D_KINFO_DECLARE(name)                                                                                   \
    struct kernel_bin_t<kernel_meta_pool1d_t> k##name{                                                                 \
        (char *)#name,                                                                                                 \
        (char *)name##_function_name,                                                                                  \
        (unsigned char *)name##_start,                                                                                 \
        (size_t)(name##_end - name##_start),                                                                           \
        name##_smem_bytes,                                                                                             \
        name##_num_warps,                                                                                              \
        name##_arg_count,                                                                                              \
        name##_global_scratch_size,                                                                                    \
        kernel_meta_pool1d_t{name##_block_size, name##_max_kernel_size}};
#else
#define DECLARE_POOL1D_KERNEL_BINARY(name) extern struct kernel_bin_t<kernel_meta_pool1d_t> k##name;
#endif

#define EMBED_POOL2D_KINFO_DECLARE(name)                                                                               \
    extern const uint32_t name##_block_size_h;                                                                         \
    extern const uint32_t name##_block_size_w;                                                                         \
    extern const uint32_t name##_max_kernel_h;                                                                         \
    extern const uint32_t name##_max_kernel_w;

#ifndef KERNEL_BINARY_PROTOTYPES_ONLY
#define DECLARE_POOL2D_KERNEL_BINARY(name)                                                                             \
    EMBED_ASSET_DECLARE(name)                                                                                          \
    EMBED_KINFO_DECLARE(name)                                                                                          \
    EMBED_POOL2D_KINFO_DECLARE(name)                                                                                   \
    struct kernel_bin_t<kernel_meta_pool2d_t> k##name{                                                                 \
        (char *)#name,                                                                                                 \
        (char *)name##_function_name,                                                                                  \
        (unsigned char *)name##_start,                                                                                 \
        (size_t)(name##_end - name##_start),                                                                           \
        name##_smem_bytes,                                                                                             \
        name##_num_warps,                                                                                              \
        name##_arg_count,                                                                                              \
        name##_global_scratch_size,                                                                                    \
        kernel_meta_pool2d_t{name##_block_size_h, name##_block_size_w, name##_max_kernel_h, name##_max_kernel_w}};
#else
#define DECLARE_POOL2D_KERNEL_BINARY(name) extern struct kernel_bin_t<kernel_meta_pool2d_t> k##name;
#endif

#ifndef KERNEL_BINARY_PROTOTYPES_ONLY
#define DECLARE_CONV2D_KERNEL_BINARY(name)                                                                             \
    EMBED_ASSET_DECLARE(name)                                                                                          \
    EMBED_KINFO_DECLARE(name)                                                                                          \
    EMBED_CONV2D_KINFO_DECLARE(name)                                                                                   \
    struct kernel_bin_t<kernel_meta_conv2d_t> k##name{(char *)#name,                                                    \
                                                      (char *)name##_function_name,                                    \
                                                      (unsigned char *)name##_start,                                   \
                                                      (size_t)(name##_end - name##_start),                             \
                                                      name##_smem_bytes,                                               \
                                                      name##_num_warps,                                                \
                                                      name##_arg_count,                                                \
                                                      name##_global_scratch_size,                                      \
                                                      kernel_meta_conv2d_t{name##_block_pixels, name##_block_oc,       \
                                                                           name##_block_k, name##_in_channels,         \
                                                                           name##_kernel_h, name##_kernel_w}};
#else
#define DECLARE_CONV2D_KERNEL_BINARY(name) extern struct kernel_bin_t<kernel_meta_conv2d_t> k##name;
#endif

#ifndef KERNEL_BINARY_PROTOTYPES_ONLY
#define DECLARE_CONV2D_CUTLASS_KERNEL_BINARY(name)                                                                     \
    EMBED_ASSET_DECLARE(name)                                                                                          \
    EMBED_KINFO_DECLARE(name)                                                                                          \
    EMBED_CONV2D_CUTLASS_KINFO_DECLARE(name)                                                                           \
    struct kernel_bin_t<kernel_meta_conv2d_cutlass_t> k##name{                                                         \
        (char *)#name,                                                                                                 \
        (char *)name##_function_name,                                                                                  \
        (unsigned char *)name##_start,                                                                                 \
        (size_t)(name##_end - name##_start),                                                                           \
        name##_smem_bytes,                                                                                             \
        name##_num_warps,                                                                                              \
        name##_arg_count,                                                                                              \
        name##_global_scratch_size,                                                                                    \
        kernel_meta_conv2d_cutlass_t{name##_block_pixels, name##_block_oc, name##_block_k, name##_in_channels,         \
                                     name##_kernel_h, name##_kernel_w, name##_stride_h, name##_stride_w,               \
                                     name##_padding_h, name##_padding_w, name##_dilation_h, name##_dilation_w,         \
                                     name##_groups}};
#else
#define DECLARE_CONV2D_CUTLASS_KERNEL_BINARY(name) extern struct kernel_bin_t<kernel_meta_conv2d_cutlass_t> k##name;
#endif

#ifndef KERNEL_BINARY_PROTOTYPES_ONLY
#define DECLARE_MHA_CUTLASS_KERNEL_BINARY(name)                                                                        \
    EMBED_ASSET_DECLARE(name)                                                                                          \
    EMBED_KINFO_DECLARE(name)                                                                                          \
    EMBED_MHA_CUTLASS_KINFO_DECLARE(name)                                                                              \
    struct kernel_bin_t<kernel_meta_mha_cutlass_t> k##name{(char *)#name,                                               \
                                                          (char *)name##_function_name,                               \
                                                          (unsigned char *)name##_start,                              \
                                                          (size_t)(name##_end - name##_start),                        \
                                                          name##_smem_bytes,                                          \
                                                          name##_num_warps,                                           \
                                                          name##_arg_count,                                           \
                                                          name##_global_scratch_size,                                 \
                                                          kernel_meta_mha_cutlass_t{name##_block_size_x,              \
                                                                                    name##_block_size_y,              \
                                                                                    name##_head_dim}};
#else
#define DECLARE_MHA_CUTLASS_KERNEL_BINARY(name) extern struct kernel_bin_t<kernel_meta_mha_cutlass_t> k##name;
#endif

#ifndef KERNEL_BINARY_PROTOTYPES_ONLY
#define DECLARE_MHA_CUTLASS_BWD_KERNEL_BINARY(name)                                                                    \
    EMBED_ASSET_DECLARE(name)                                                                                          \
    EMBED_KINFO_DECLARE(name)                                                                                          \
    EMBED_MHA_CUTLASS_BWD_KINFO_DECLARE(name)                                                                          \
    struct kernel_bin_t<kernel_meta_mha_cutlass_bwd_t> k##name{(char *)#name,                                           \
                                                              (char *)name##_function_name,                           \
                                                              (unsigned char *)name##_start,                          \
                                                              (size_t)(name##_end - name##_start),                    \
                                                              name##_smem_bytes,                                      \
                                                              name##_num_warps,                                       \
                                                              name##_arg_count,                                       \
                                                              name##_global_scratch_size,                             \
                                                              kernel_meta_mha_cutlass_bwd_t{                           \
                                                                  name##_block_size_i,                                 \
                                                                  name##_block_size_j,                                 \
                                                                  name##_head_dim,                                     \
                                                                  name##_gradq_tile_elements,                          \
                                                                  name##_gradq_temp_bytes}};
#else
#define DECLARE_MHA_CUTLASS_BWD_KERNEL_BINARY(name)                                                                    \
    extern struct kernel_bin_t<kernel_meta_mha_cutlass_bwd_t> k##name;
#endif

#ifndef KERNEL_BINARY_PROTOTYPES_ONLY
#define DECLARE_GEMM_CUTLASS_KERNEL_BINARY(name)                                                                       \
    EMBED_ASSET_DECLARE(name)                                                                                          \
    EMBED_KINFO_DECLARE(name)                                                                                          \
    EMBED_GEMM_CUTLASS_KINFO_DECLARE(name)                                                                             \
    struct kernel_bin_t<kernel_meta_gemm_cutlass_t> k##name{                                                           \
        (char *)#name,                                                                                                 \
        (char *)name##_function_name,                                                                                  \
        (unsigned char *)name##_start,                                                                                 \
        (size_t)(name##_end - name##_start),                                                                           \
        name##_smem_bytes,                                                                                             \
        name##_num_warps,                                                                                              \
        name##_arg_count,                                                                                              \
        name##_global_scratch_size,                                                                                    \
        kernel_meta_gemm_cutlass_t{name##_block_m, name##_block_n, name##_block_k}};
#else
#define DECLARE_GEMM_CUTLASS_KERNEL_BINARY(name) extern struct kernel_bin_t<kernel_meta_gemm_cutlass_t> k##name;
#endif

#ifndef KERNEL_BINARY_PROTOTYPES_ONLY
#define DECLARE_2D_KERNEL_BINARY(name)                                                                                 \
    EMBED_ASSET_DECLARE(name)                                                                                          \
    EMBED_KINFO_DECLARE(name)                                                                                          \
    EMBED_2D_KINFO_DECLARE(name)                                                                                       \
    struct kernel_bin_t<kernel_meta_2d_t> k##name{(char *)#name,                                                        \
                                                  (char *)name##_function_name,                                        \
                                                  (unsigned char *)name##_start,                                       \
                                                  (size_t)(name##_end - name##_start),                                 \
                                                  name##_smem_bytes,                                                   \
                                                  name##_num_warps,                                                    \
                                                  name##_arg_count,                                                    \
                                                  name##_global_scratch_size,                                          \
                                                  kernel_meta_2d_t{name##_block_size_x, name##_block_size_y}};
#else
#define DECLARE_2D_KERNEL_BINARY(name) extern struct kernel_bin_t<kernel_meta_2d_t> k##name;
#endif

#ifndef KERNEL_BINARY_PROTOTYPES_ONLY
#define DECLARE_3D_KERNEL_BINARY(name)                                                                                 \
    EMBED_ASSET_DECLARE(name)                                                                                          \
    EMBED_KINFO_DECLARE(name)                                                                                          \
    EMBED_3D_KINFO_DECLARE(name)                                                                                       \
    struct kernel_bin_t<kernel_meta_3d_t> k##name{                                                                     \
        (char *)#name,                                                                                                 \
        (char *)name##_function_name,                                                                                  \
        (unsigned char *)name##_start,                                                                                 \
        (size_t)(name##_end - name##_start),                                                                           \
        name##_smem_bytes,                                                                                             \
        name##_num_warps,                                                                                              \
        name##_arg_count,                                                                                              \
        name##_global_scratch_size,                                                                                    \
        kernel_meta_3d_t{name##_block_size_x, name##_block_size_y, name##_block_size_z}};
#else
#define DECLARE_3D_KERNEL_BINARY(name) extern struct kernel_bin_t<kernel_meta_3d_t> k##name;
#endif

#ifndef KERNEL_BINARY_PROTOTYPES_ONLY
#define DECLARE_4D_KERNEL_BINARY(name)                                                                                 \
    EMBED_ASSET_DECLARE(name)                                                                                          \
    EMBED_KINFO_DECLARE(name)                                                                                          \
    EMBED_4D_KINFO_DECLARE(name)                                                                                       \
    struct kernel_bin_t<kernel_meta_4d_t> k##name{                                                                     \
        (char *)#name,                                                                                                 \
        (char *)name##_function_name,                                                                                  \
        (unsigned char *)name##_start,                                                                                 \
        (size_t)(name##_end - name##_start),                                                                           \
        name##_smem_bytes,                                                                                             \
        name##_num_warps,                                                                                              \
        name##_arg_count,                                                                                              \
        name##_global_scratch_size,                                                                                    \
        kernel_meta_4d_t{name##_block_size_x, name##_block_size_y, name##_block_size_z, name##_block_size_w}};
#else
#define DECLARE_4D_KERNEL_BINARY(name) extern struct kernel_bin_t<kernel_meta_4d_t> k##name;
#endif
