add_executable(test_matmul_large_shape test_matmul_large_shape.cpp)

target_link_libraries(test_matmul_large_shape PRIVATE tensorlib)
target_link_libraries(test_matmul_large_shape PRIVATE fbamtrain_lib)

add_test(NAME test_matmul_large_shape_bf16 COMMAND test_matmul_large_shape)
set_tests_properties(test_matmul_large_shape_bf16 PROPERTIES ENVIRONMENT
        "${FBAMTRAIN_ENV_PREFER_TRITON};TEST_DTYPE=bf16")
add_test(NAME test_matmul_large_shape_fp16 COMMAND test_matmul_large_shape)
set_tests_properties(test_matmul_large_shape_fp16 PROPERTIES ENVIRONMENT
        "${FBAMTRAIN_ENV_PREFER_TRITON};TEST_DTYPE=fp16")

add_custom_command(TARGET test_matmul_large_shape POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/reference_bf16.safetensors
        $<TARGET_FILE_DIR:test_matmul_large_shape>/reference_bf16.safetensors
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/reference_fp16.safetensors
        $<TARGET_FILE_DIR:test_matmul_large_shape>/reference_fp16.safetensors)
