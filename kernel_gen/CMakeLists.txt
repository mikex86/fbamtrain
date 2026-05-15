enable_language(ASM)

set_source_files_properties(
        src/embed_kernels_nv.asm
        src/embed_kernels_amd.asm
        PROPERTIES LANGUAGE ASM
)

set(ASSET_DIR "${CMAKE_CURRENT_SOURCE_DIR}/kernel_gen/output")

if (TENSORLIB_ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED COMPONENTS cudart cuda_driver)

    set(GENERATED_CONV2D_LAUNCH_OBJ
            "${CMAKE_CURRENT_BINARY_DIR}/generated_conv2d_launch.cu.o")
    add_custom_command(
            OUTPUT "${GENERATED_CONV2D_LAUNCH_OBJ}"
            COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}"
                    -std=c++17
                    -O3
                    -DNDEBUG
                    --expt-relaxed-constexpr
                    --expt-extended-lambda
                    -Xcompiler=-fPIC
                    -I"${CMAKE_CURRENT_LIST_DIR}/include"
                    -I"${CMAKE_CURRENT_LIST_DIR}/include/nv"
                    -I"${CMAKE_SOURCE_DIR}/tensorlib/include"
                    -I"${CMAKE_CURRENT_LIST_DIR}/kernel_gen/kernels/cutlass_codegen"
                    -I"${CMAKE_CURRENT_LIST_DIR}/kernel_gen/kernels/cutlass_codegen/third_party/cutlass/include"
                    -I"${CMAKE_CURRENT_LIST_DIR}/kernel_gen/kernels/cutlass_codegen/third_party/cutlass/tools/util/include"
                    -c "${CMAKE_CURRENT_LIST_DIR}/src/generated_conv2d_launch.cu"
                    -o "${GENERATED_CONV2D_LAUNCH_OBJ}"
            DEPENDS "${CMAKE_CURRENT_LIST_DIR}/src/generated_conv2d_launch.cu"
                    "${CMAKE_CURRENT_LIST_DIR}/include/nv/generated_conv2d_launch.h"
                    "${CMAKE_CURRENT_LIST_DIR}/kernel_gen/kernels/cutlass_codegen/conv2d_cutlass3_wgrad_traits.cuh"
            VERBATIM)
    set_source_files_properties("${GENERATED_CONV2D_LAUNCH_OBJ}"
            PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)

    add_library(kernels_nv_obj OBJECT src/embed_kernels_nv.asm)
    target_compile_options(kernels_nv_obj PRIVATE -x assembler-with-cpp)
    target_compile_options(kernels_nv_obj PRIVATE
            $<$<COMPILE_LANGUAGE:ASM>:-Wa,-I,${ASSET_DIR}>
    )
    target_compile_definitions(kernels_nv_obj PUBLIC NV_KERNEL_ARCH=${NV_KERNEL_ARCH})

    add_library(kernels_nv STATIC
            $<TARGET_OBJECTS:kernels_nv_obj>
            "${GENERATED_CONV2D_LAUNCH_OBJ}")
    target_include_directories(kernels_nv PUBLIC
            ${CMAKE_CURRENT_LIST_DIR}/include
            ${CMAKE_CURRENT_LIST_DIR}/include/nv)
    target_include_directories(kernels_nv PRIVATE
            ${CMAKE_SOURCE_DIR}/tensorlib/include
            ${CUDAToolkit_INCLUDE_DIRS}
            ${CMAKE_CURRENT_LIST_DIR}/kernel_gen/kernels/cutlass_codegen
            ${CMAKE_CURRENT_LIST_DIR}/kernel_gen/kernels/cutlass_codegen/third_party/cutlass/include
            ${CMAKE_CURRENT_LIST_DIR}/kernel_gen/kernels/cutlass_codegen/third_party/cutlass/tools/util/include)
    target_link_libraries(kernels_nv PUBLIC CUDA::cudart CUDA::cuda_driver)
    target_compile_definitions(kernels_nv PUBLIC NV_KERNEL_ARCH=${NV_KERNEL_ARCH})
endif ()

if (TENSORLIB_ENABLE_HIP)
    add_library(kernels_amd_obj OBJECT src/embed_kernels_amd.asm)
    target_compile_options(kernels_amd_obj PRIVATE -x assembler-with-cpp)
    target_compile_options(kernels_amd_obj PRIVATE
            $<$<COMPILE_LANGUAGE:ASM>:-Wa,-I,${ASSET_DIR}>
    )
    target_compile_definitions(kernels_amd_obj PUBLIC AMD_KERNEL_ARCH=${AMD_KERNEL_ARCH})

    add_library(kernels_amd STATIC $<TARGET_OBJECTS:kernels_amd_obj>)
    target_include_directories(kernels_amd PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include/amd)
    target_compile_definitions(kernels_amd PUBLIC AMD_KERNEL_ARCH=${AMD_KERNEL_ARCH})
endif ()
