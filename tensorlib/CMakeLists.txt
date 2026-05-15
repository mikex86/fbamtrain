option(TENSORLIB_ENABLE_CUDA "Enable CUDA support" ON)
option(TENSORLIB_ENABLE_HIP "Enable HIP support" OFF)

add_library(tensorlib src/tensorlib.cpp
        src/shape_utils.cpp
        src/op_graph.cpp
        src/execution_backend.cpp
        src/allocator.cpp
        src/module.cpp
        src/mfu.cpp
        src/execution_plan.cpp
        src/compute_kernel.cpp
        src/executor.cpp
        src/kernel_cache.cpp
        src/functional.cpp
        src/safe_tensors.cpp
        src/testing.cpp
        src/device_copy.cpp
        src/cpudbg.cpp
        src/ctx_management.cpp
        src/gputx.cpp
        src/gpu_stream.cpp
        src/streaming_lstm_bwd.cpp
        src/streaming_lstm_fwd.cpp
        src/tensor_stream_iterator.cpp
        src/tensor_stream_populator.cpp
)
target_include_directories(tensorlib PUBLIC include)

target_link_libraries(tensorlib PRIVATE nlohmann_json::nlohmann_json)
target_include_directories(tensorlib PRIVATE "${CMAKE_SOURCE_DIR}/kernel_gen/include")

if (TENSORLIB_ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(tensorlib PUBLIC CUDA::cudart_static CUDA::cuda_driver)
    target_include_directories(tensorlib PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
endif ()

if (TENSORLIB_ENABLE_HIP)
    list(APPEND CMAKE_PREFIX_PATH "/opt/rocm")
    find_package(hip REQUIRED)
    target_link_libraries(tensorlib PRIVATE hip::host)
    target_compile_definitions(tensorlib PUBLIC __HIP_PLATFORM_AMD__)
endif ()

target_compile_definitions(tensorlib PUBLIC PI_TENSORLIB_ENABLE_CUDA=$<IF:$<BOOL:${TENSORLIB_ENABLE_CUDA}>,1,0>)
target_compile_definitions(tensorlib PUBLIC PI_TENSORLIB_ENABLE_HIP=$<IF:$<BOOL:${TENSORLIB_ENABLE_HIP}>,1,0>)
