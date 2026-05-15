add_executable(test_conv2d_bwd test_conv2d_bwd.cpp)

target_link_libraries(test_conv2d_bwd PRIVATE tensorlib)
target_link_libraries(test_conv2d_bwd PRIVATE fbamtrain_lib)

set(FBAMTRAIN_ENV_EXPECT_CONV2D_CUTLASS3 "")
if(NV_KERNEL_ARCH STREQUAL "90")
    set(FBAMTRAIN_ENV_EXPECT_CONV2D_CUTLASS3 ";CONV2D_EXPECT_CUTLASS3=1")
endif()

add_test(NAME test_conv2d_bwd_bf16 COMMAND test_conv2d_bwd)
set_tests_properties(test_conv2d_bwd_bf16 PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_CUTLASS};TEST_DTYPE=bf16${FBAMTRAIN_ENV_EXPECT_CONV2D_CUTLASS3}")
add_test(NAME test_conv2d_bwd_bf16_no_cutlass COMMAND test_conv2d_bwd)
set_tests_properties(test_conv2d_bwd_bf16_no_cutlass PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_TRITON};TEST_DTYPE=bf16")

add_test(NAME test_conv2d_bwd_fp16 COMMAND test_conv2d_bwd)
set_tests_properties(test_conv2d_bwd_fp16 PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_CUTLASS};TEST_DTYPE=fp16${FBAMTRAIN_ENV_EXPECT_CONV2D_CUTLASS3}")
add_test(NAME test_conv2d_bwd_fp16_no_cutlass COMMAND test_conv2d_bwd)
set_tests_properties(test_conv2d_bwd_fp16_no_cutlass PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_TRITON};TEST_DTYPE=fp16")

add_custom_command(TARGET test_conv2d_bwd POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/reference_bf16.safetensors
        $<TARGET_FILE_DIR:test_conv2d_bwd>/reference_bf16.safetensors
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/reference_fp16.safetensors
        $<TARGET_FILE_DIR:test_conv2d_bwd>/reference_fp16.safetensors)
