cmake_minimum_required(VERSION 3.28.3)
project(fbamtrain)

set(CMAKE_CXX_STANDARD 20)

option(USE_AVX2 "Enable AVX2+FMA fast path" OFF)
option(USE_OPENMP "Enable OpenMP parallel loops" OFF)
option(USE_MARCH_NATIVE "Use -march=native instead of explicit -mavx2 -mfma" OFF)
option(GPUTX_ENABLE_NVTX "Enable NVTX profiling ranges when CUDA is available" ON)
option(GPUTX_ENABLE_ROCTX "Enable ROCTX profiling ranges when HIP is available" ON)
option(ENABLE_ASAN "Enable AddressSanitizer for C++ targets" OFF)

set(NV_KERNEL_ARCH 90 CACHE STRING "NVIDIA kernel arch for embedded kernels")
set(AMD_KERNEL_ARCH 1101 CACHE STRING "AMD kernel arch for embedded kernels")

set(CMAKE_CUDA_ARCHITECTURES "${NV_KERNEL_ARCH}" CACHE STRING "CUDA architectures")

enable_testing()

set(FBAMTRAIN_ENV_PREFER_CUTLASS "FBAMTRAIN_PREFER_GEMM_BACKEND=cutlass;FBAMTRAIN_PREFER_CONV2D_BACKEND=cutlass")
set(FBAMTRAIN_ENV_PREFER_TRITON "FBAMTRAIN_PREFER_GEMM_BACKEND=triton;FBAMTRAIN_PREFER_CONV2D_BACKEND=triton")
set(FBAMTRAIN_ENV_PREFER_MHA_FLASH "FBAMTRAIN_PREFER_MHA_BACKEND=flash")
set(FBAMTRAIN_ENV_PREFER_MHA_CUTLASS "FBAMTRAIN_PREFER_MHA_BACKEND=cutlass")
set(FBAMTRAIN_ENV_PREFER_MHA_TRITON "FBAMTRAIN_PREFER_MHA_BACKEND=triton")

add_subdirectory(third_party/json)
add_subdirectory(third_party/termstreamxz)
add_subdirectory(third_party/argparse)
add_subdirectory(third_party/threadpark)

add_subdirectory(tensorlib)
add_subdirectory(kernel_gen)
add_subdirectory(ccl)
set(PCCL_BUILD_STATIC_LIB ON CACHE BOOL "Build PCCL as a static library" FORCE)
add_subdirectory(third_party/pccl EXCLUDE_FROM_ALL)
add_subdirectory(ftccl)

set(FBAMTRAIN_TRITON_WARP_SIZE 32)
if (TENSORLIB_ENABLE_HIP)
    set(_amd_arch_raw "${AMD_KERNEL_ARCH}")
    if (_amd_arch_raw MATCHES "^gfx")
        string(REGEX REPLACE "^gfx" "" _amd_arch_num "${_amd_arch_raw}")
    else ()
        set(_amd_arch_num "${_amd_arch_raw}")
    endif ()
    if (_amd_arch_num MATCHES "^[0-9]+$")
        math(EXPR _amd_arch_major "${_amd_arch_num} / 100")
        if (_amd_arch_major LESS 10)
            set(FBAMTRAIN_TRITON_WARP_SIZE 64)
        else ()
            set(FBAMTRAIN_TRITON_WARP_SIZE 32)
        endif ()
    else ()
        message(WARNING "AMD_KERNEL_ARCH (${AMD_KERNEL_ARCH}) is not numeric; defaulting TRITON_WARP_SIZE=32")
    endif ()
endif ()

add_library(fbamtrain_lib
        src/passes/matmul_pass.cpp
        src/passes/fill_passes.cpp
        src/passes/norm_passes.cpp
        src/passes/pool_passes.cpp
        src/passes/conv2d_pass.cpp
        src/passes/mean_pass.cpp
        src/passes/reduce_sum_pass.cpp
        src/passes/mul_reduce_fuse_pass.cpp
        src/passes/cross_entropy_pass.cpp
        src/passes/build_cell_embed_pass.cpp
        src/passes/build_cell_embed_bwd_pass.cpp
        src/passes/kernel_binaries.cpp
        src/passes/mha_attention_pass.cpp
        src/passes/gather_pass.cpp
        src/passes/cast_pass.cpp
        src/passes/contiguous_pass.cpp
        src/passes/trailing_broadcast_passes.cpp
        src/passes/act_pass.cpp
        src/passes/lstm_pass.cpp
        src/passes/lstm_bwd_pass.cpp
        src/passes/reduce_sum_partial_pass.cpp
        src/passes/optimizer_pass.cpp
        src/config.cpp
        src/checkpointing.cpp
        src/dataset_cursor.cpp
        src/logger.cpp
        src/async_step_logger.cpp
        src/loss_accumulation_ring_buffer.cpp
        src/dataset_iterator.cpp
        src/framehead_model.cpp
        src/main_action_model.cpp
        src/frame_utils.cpp
        src/random.cpp
        src/optimizers.cpp
        src/tokenizer.cpp
        src/rdvz.cpp)

target_include_directories(fbamtrain_lib PUBLIC include)
target_include_directories(fbamtrain_lib PUBLIC kernel_gen/include)
target_link_libraries(fbamtrain_lib PUBLIC nlohmann_json::nlohmann_json termstreamxz)
target_link_libraries(fbamtrain_lib PUBLIC argparse::argparse)
target_link_libraries(fbamtrain_lib PUBLIC threadpark)
target_link_libraries(fbamtrain_lib PUBLIC tensorlib)
target_link_libraries(fbamtrain_lib PUBLIC ccl)
target_link_libraries(fbamtrain_lib PUBLIC ftccl)

target_compile_definitions(fbamtrain_lib PRIVATE
        PI_TENSORLIB_ENABLE_CUDA=$<IF:$<BOOL:${TENSORLIB_ENABLE_CUDA}>,1,0>
        PI_TENSORLIB_ENABLE_HIP=$<IF:$<BOOL:${TENSORLIB_ENABLE_HIP}>,1,0>
        GPUTX_ENABLE_NVTX=$<IF:$<BOOL:${GPUTX_ENABLE_NVTX}>,1,0>
        GPUTX_ENABLE_ROCTX=$<IF:$<BOOL:${GPUTX_ENABLE_ROCTX}>,1,0>
        TRITON_WARP_SIZE=${FBAMTRAIN_TRITON_WARP_SIZE})

target_compile_definitions(tensorlib PRIVATE
        GPUTX_ENABLE_NVTX=$<IF:$<BOOL:${GPUTX_ENABLE_NVTX}>,1,0>
        GPUTX_ENABLE_ROCTX=$<IF:$<BOOL:${GPUTX_ENABLE_ROCTX}>,1,0>
        TRITON_WARP_SIZE=${FBAMTRAIN_TRITON_WARP_SIZE})

# OpenMP
if (USE_OPENMP)
    find_package(OpenMP)
    if (OpenMP_CXX_FOUND)
        target_link_libraries(fbamtrain_lib PRIVATE OpenMP::OpenMP_CXX)
    else ()
        message(WARNING "OpenMP not found; continuing without it")
    endif ()
endif ()

if (MSVC)
    target_compile_options(fbamtrain_lib PRIVATE /O2 /fp:fast)
else ()
    target_compile_options(fbamtrain_lib PRIVATE -O3 -fopenmp-simd)
endif ()

# AVX2/FMA flags
if (USE_AVX2)
    if (MSVC)
        target_compile_options(fbamtrain_lib PRIVATE /arch:AVX2)
        # MSVC doesn't define them by default:
        target_compile_definitions(fbamtrain_lib PRIVATE __AVX2__ __FMA__)
    else ()
        if (USE_MARCH_NATIVE)
            target_compile_options(fbamtrain_lib PRIVATE -march=native)
        else ()
            target_compile_options(fbamtrain_lib PRIVATE -mavx2 -mfma)
        endif ()
    endif ()
endif ()

# if release, enable IPO
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_ok OUTPUT ipo_msg)
    if (ipo_ok)
        set_property(TARGET fbamtrain_lib PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif ()
endif ()

add_executable(fbamtrain src/main.cpp)
target_link_libraries(fbamtrain PRIVATE fbamtrain_lib)

add_executable(tenscmp src/tenscmp.cpp)
target_link_libraries(tenscmp PRIVATE tensorlib argparse::argparse)

add_subdirectory(tests)
add_subdirectory(benchmarks)

set(GPUTX_HAS_NVTX 0)
if (TENSORLIB_ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED COMPONENTS cudart cuda_driver)

    target_link_libraries(fbamtrain_lib PRIVATE kernels_nv)
    target_link_libraries(tensorlib PRIVATE kernels_nv)
    target_link_libraries(tensorlib PRIVATE CUDA::cudart_static CUDA::cuda_driver)

    if (GPUTX_ENABLE_NVTX)
        find_package(CUDAToolkit QUIET COMPONENTS nvtx3)
        if (TARGET CUDA::nvtx3)
            target_link_libraries(tensorlib PRIVATE CUDA::nvtx3)
            set(GPUTX_HAS_NVTX 1)
        else ()
            message(WARNING "nvtx3 library not found; CUDA profiling ranges will be disabled at runtime.")
        endif ()
    endif ()
endif ()

set(GPUTX_HAS_ROCTX 0)
if (TENSORLIB_ENABLE_HIP)
    find_path(HIP_INCLUDE_DIR hip/hip_runtime.h HINTS /opt/rocm/include)
    if (HIP_INCLUDE_DIR)
        target_include_directories(fbamtrain_lib PUBLIC ${HIP_INCLUDE_DIR})
        target_include_directories(tensorlib PUBLIC ${HIP_INCLUDE_DIR})
    else ()
        message(WARNING "HIP headers not found; HIP targets may fail to build.")
    endif ()

    target_link_libraries(fbamtrain_lib PRIVATE kernels_amd)
    target_link_libraries(tensorlib PRIVATE kernels_amd)

    if (GPUTX_ENABLE_ROCTX)
        find_library(ROCTX_LIBRARY roctx64 HINTS /opt/rocm/lib /opt/rocm/lib64)
        if (ROCTX_LIBRARY)
            target_link_libraries(tensorlib PRIVATE ${ROCTX_LIBRARY})
            set(GPUTX_HAS_ROCTX 1)
        else ()
            message(WARNING "roctx64 library not found; HIP profiling ranges will be disabled at runtime.")
        endif ()
    endif ()
endif ()

target_compile_definitions(fbamtrain_lib PRIVATE
        GPUTX_HAS_NVTX=${GPUTX_HAS_NVTX}
        GPUTX_HAS_ROCTX=${GPUTX_HAS_ROCTX})
target_compile_definitions(tensorlib PRIVATE
        GPUTX_HAS_NVTX=${GPUTX_HAS_NVTX}
        GPUTX_HAS_ROCTX=${GPUTX_HAS_ROCTX})
