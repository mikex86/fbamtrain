add_executable(test_mha_attn_and_reshape_fwd test_mha_attn_and_reshape_fwd.cpp)

target_link_libraries(test_mha_attn_and_reshape_fwd PRIVATE tensorlib)
target_link_libraries(test_mha_attn_and_reshape_fwd PRIVATE fbamtrain_lib)

add_test(NAME test_mha_attn_and_reshape_fwd_bf16 COMMAND test_mha_attn_and_reshape_fwd)
set_tests_properties(test_mha_attn_and_reshape_fwd_bf16 PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_CUTLASS};${FBAMTRAIN_ENV_PREFER_MHA_FLASH};TEST_DTYPE=bf16")
add_test(NAME test_mha_attn_and_reshape_fwd_bf16_no_fa COMMAND test_mha_attn_and_reshape_fwd)
set_tests_properties(test_mha_attn_and_reshape_fwd_bf16_no_fa PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_CUTLASS};${FBAMTRAIN_ENV_PREFER_MHA_TRITON};TEST_DTYPE=bf16")
add_test(NAME test_mha_attn_and_reshape_fwd_bf16_no_cutlass COMMAND test_mha_attn_and_reshape_fwd)
set_tests_properties(test_mha_attn_and_reshape_fwd_bf16_no_cutlass PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_TRITON};${FBAMTRAIN_ENV_PREFER_MHA_TRITON};TEST_DTYPE=bf16")

add_test(NAME test_mha_attn_and_reshape_fwd_fp16 COMMAND test_mha_attn_and_reshape_fwd)
set_tests_properties(test_mha_attn_and_reshape_fwd_fp16 PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_CUTLASS};${FBAMTRAIN_ENV_PREFER_MHA_FLASH};TEST_DTYPE=fp16")
add_test(NAME test_mha_attn_and_reshape_fwd_fp16_no_fa COMMAND test_mha_attn_and_reshape_fwd)
set_tests_properties(test_mha_attn_and_reshape_fwd_fp16_no_fa PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_CUTLASS};${FBAMTRAIN_ENV_PREFER_MHA_TRITON};TEST_DTYPE=fp16")
add_test(NAME test_mha_attn_and_reshape_fwd_fp16_no_cutlass COMMAND test_mha_attn_and_reshape_fwd)
set_tests_properties(test_mha_attn_and_reshape_fwd_fp16_no_cutlass PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_TRITON};${FBAMTRAIN_ENV_PREFER_MHA_TRITON};TEST_DTYPE=fp16")

# make sure reference safetensors are available in the build directory
add_custom_command(TARGET test_mha_attn_and_reshape_fwd POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/reference_bf16.safetensors
        $<TARGET_FILE_DIR:test_mha_attn_and_reshape_fwd>/reference_bf16.safetensors
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/reference_fp16.safetensors
        $<TARGET_FILE_DIR:test_mha_attn_and_reshape_fwd>/reference_fp16.safetensors)
