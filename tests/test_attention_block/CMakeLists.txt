add_executable(test_attention_block test_attention_block.cpp)

target_link_libraries(test_attention_block PRIVATE tensorlib)
target_link_libraries(test_attention_block PRIVATE fbamtrain_lib)

add_test(NAME test_attention_block_bf16 COMMAND test_attention_block)
set_tests_properties(test_attention_block_bf16 PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_CUTLASS};${FBAMTRAIN_ENV_PREFER_MHA_FLASH};TEST_DTYPE=bf16")
add_test(NAME test_attention_block_bf16_no_fa COMMAND test_attention_block)
set_tests_properties(test_attention_block_bf16_no_fa PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_CUTLASS};${FBAMTRAIN_ENV_PREFER_MHA_TRITON};TEST_DTYPE=bf16")
add_test(NAME test_attention_block_bf16_no_cutlass COMMAND test_attention_block)
set_tests_properties(test_attention_block_bf16_no_cutlass PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_TRITON};${FBAMTRAIN_ENV_PREFER_MHA_TRITON};TEST_DTYPE=bf16")

add_test(NAME test_attention_block_fp16 COMMAND test_attention_block)
set_tests_properties(test_attention_block_fp16 PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_CUTLASS};${FBAMTRAIN_ENV_PREFER_MHA_FLASH};TEST_DTYPE=fp16")
add_test(NAME test_attention_block_fp16_no_fa COMMAND test_attention_block)
set_tests_properties(test_attention_block_fp16_no_fa PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_CUTLASS};${FBAMTRAIN_ENV_PREFER_MHA_TRITON};TEST_DTYPE=fp16")
add_test(NAME test_attention_block_fp16_no_cutlass COMMAND test_attention_block)
set_tests_properties(test_attention_block_fp16_no_cutlass PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_TRITON};${FBAMTRAIN_ENV_PREFER_MHA_TRITON};TEST_DTYPE=fp16")

add_custom_command(TARGET test_attention_block POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/reference_bf16.safetensors
        $<TARGET_FILE_DIR:test_attention_block>/reference_bf16.safetensors
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/reference_fp16.safetensors
        $<TARGET_FILE_DIR:test_attention_block>/reference_fp16.safetensors)
