include_guard(GLOBAL)

if (NOT TARGET CUDA::cudart)
    find_package(CUDAToolkit REQUIRED COMPONENTS cudart)
endif ()

find_path(FBAMTRAIN_NCCL_INCLUDE_DIR
    NAMES nccl.h
    REQUIRED)

find_library(FBAMTRAIN_NCCL_SHARED_LIBRARY
    NAMES nccl
    REQUIRED)

# Enforce dynamic linkage to reduce final link times.
if (FBAMTRAIN_NCCL_SHARED_LIBRARY MATCHES "\\.a$")
    message(FATAL_ERROR
        "Found static NCCL library at ${FBAMTRAIN_NCCL_SHARED_LIBRARY}, but a shared libnccl.so is required.")
endif ()

if (NOT TARGET fbamtrain_nccl_shared)
    add_library(fbamtrain_nccl_shared SHARED IMPORTED GLOBAL)
    set_target_properties(fbamtrain_nccl_shared PROPERTIES
        IMPORTED_LOCATION "${FBAMTRAIN_NCCL_SHARED_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${FBAMTRAIN_NCCL_INCLUDE_DIR}")
endif ()

target_link_libraries(ccl PUBLIC CUDA::cudart)
target_link_libraries(ccl PUBLIC fbamtrain_nccl_shared)
