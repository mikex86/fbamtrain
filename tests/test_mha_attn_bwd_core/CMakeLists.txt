add_executable(test_mha_attn_bwd_core test_mha_attn_bwd_core.cpp)
target_link_libraries(test_mha_attn_bwd_core PRIVATE fbamtrain_lib)
target_compile_features(test_mha_attn_bwd_core PRIVATE cxx_std_20)
add_test(NAME test_mha_attn_bwd_core_bf16 COMMAND test_mha_attn_bwd_core)
set_tests_properties(test_mha_attn_bwd_core_bf16 PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_MHA_TRITON};MHA_EXPECT_TRITON=1;TEST_DTYPE=bf16")
add_test(NAME test_mha_attn_bwd_core_fp16 COMMAND test_mha_attn_bwd_core)
set_tests_properties(test_mha_attn_bwd_core_fp16 PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_MHA_TRITON};MHA_EXPECT_TRITON=1;TEST_DTYPE=fp16")

add_test(NAME test_mha_attn_bwd_core_bf16_flash COMMAND test_mha_attn_bwd_core)
set_tests_properties(test_mha_attn_bwd_core_bf16_flash PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_MHA_FLASH};MHA_EXPECT_FLASH=1;TEST_DTYPE=bf16")
add_test(NAME test_mha_attn_bwd_core_fp16_flash COMMAND test_mha_attn_bwd_core)
set_tests_properties(test_mha_attn_bwd_core_fp16_flash PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_MHA_FLASH};MHA_EXPECT_FLASH=1;TEST_DTYPE=fp16")
add_test(NAME test_mha_attn_bwd_core_fp16_flash_acc COMMAND test_mha_attn_bwd_core)
set_tests_properties(test_mha_attn_bwd_core_fp16_flash_acc PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_MHA_FLASH};MHA_EXPECT_FLASH=1;MHA_TEST_FP16_ACC=1;TEST_DTYPE=fp16")
add_test(NAME test_mha_attn_bwd_core_bf16_flash_t127 COMMAND test_mha_attn_bwd_core)
set_tests_properties(test_mha_attn_bwd_core_bf16_flash_t127 PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_MHA_FLASH};MHA_EXPECT_FLASH=1;MHA_EXPECT_NON_EVEN_KERNEL=1;MHA_TEST_SEQLEN=127;TEST_DTYPE=bf16")
add_test(NAME test_mha_attn_bwd_core_fp16_flash_t127 COMMAND test_mha_attn_bwd_core)
set_tests_properties(test_mha_attn_bwd_core_fp16_flash_t127 PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_MHA_FLASH};MHA_EXPECT_FLASH=1;MHA_EXPECT_NON_EVEN_KERNEL=1;MHA_TEST_SEQLEN=127;TEST_DTYPE=fp16")

add_custom_command(TARGET test_mha_attn_bwd_core POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/reference_bf16.safetensors
        $<TARGET_FILE_DIR:test_mha_attn_bwd_core>/reference_bf16.safetensors
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/reference_fp16.safetensors
        $<TARGET_FILE_DIR:test_mha_attn_bwd_core>/reference_fp16.safetensors
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/reference_bf16_t127.safetensors
        $<TARGET_FILE_DIR:test_mha_attn_bwd_core>/reference_bf16_t127.safetensors
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/reference_fp16_t127.safetensors
        $<TARGET_FILE_DIR:test_mha_attn_bwd_core>/reference_fp16_t127.safetensors)
