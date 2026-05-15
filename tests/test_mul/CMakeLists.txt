add_executable(test_mul test_mul.cpp)

target_link_libraries(test_mul PRIVATE tensorlib)
target_link_libraries(test_mul PRIVATE fbamtrain_lib)

add_test(NAME test_mul_bf16 COMMAND test_mul)
set_tests_properties(test_mul_bf16 PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_CUTLASS};TEST_DTYPE=bf16")
add_test(NAME test_mul_bf16_no_cutlass COMMAND test_mul)
set_tests_properties(test_mul_bf16_no_cutlass PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_TRITON};TEST_DTYPE=bf16")

add_test(NAME test_mul_fp16 COMMAND test_mul)
set_tests_properties(test_mul_fp16 PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_CUTLASS};TEST_DTYPE=fp16")
add_test(NAME test_mul_fp16_no_cutlass COMMAND test_mul)
set_tests_properties(test_mul_fp16_no_cutlass PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_TRITON};TEST_DTYPE=fp16")

add_custom_command(TARGET test_mul POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/reference_bf16.safetensors
        $<TARGET_FILE_DIR:test_mul>/reference_bf16.safetensors
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/reference_fp16.safetensors
        $<TARGET_FILE_DIR:test_mul>/reference_fp16.safetensors)
