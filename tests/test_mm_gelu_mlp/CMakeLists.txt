add_executable(test_mm_gelu_mlp test_mm_gelu_mlp.cpp)

target_link_libraries(test_mm_gelu_mlp PRIVATE tensorlib)
target_link_libraries(test_mm_gelu_mlp PRIVATE fbamtrain_lib)

add_test(NAME test_mm_gelu_mlp_bf16 COMMAND test_mm_gelu_mlp)
set_tests_properties(test_mm_gelu_mlp_bf16 PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_CUTLASS};TEST_DTYPE=bf16")
add_test(NAME test_mm_gelu_mlp_bf16_no_cutlass COMMAND test_mm_gelu_mlp)
set_tests_properties(test_mm_gelu_mlp_bf16_no_cutlass PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_TRITON};TEST_DTYPE=bf16")

add_test(NAME test_mm_gelu_mlp_fp16 COMMAND test_mm_gelu_mlp)
set_tests_properties(test_mm_gelu_mlp_fp16 PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_CUTLASS};TEST_DTYPE=fp16")
add_test(NAME test_mm_gelu_mlp_fp16_no_cutlass COMMAND test_mm_gelu_mlp)
set_tests_properties(test_mm_gelu_mlp_fp16_no_cutlass PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_TRITON};TEST_DTYPE=fp16")

add_custom_command(TARGET test_mm_gelu_mlp POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/reference_bf16.safetensors
        $<TARGET_FILE_DIR:test_mm_gelu_mlp>/reference_bf16.safetensors
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/reference_fp16.safetensors
        $<TARGET_FILE_DIR:test_mm_gelu_mlp>/reference_fp16.safetensors)
