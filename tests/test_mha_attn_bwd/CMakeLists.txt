add_executable(test_mha_attn_bwd test_mha_attn_bwd.cpp)

target_link_libraries(test_mha_attn_bwd PRIVATE tensorlib)
target_link_libraries(test_mha_attn_bwd PRIVATE fbamtrain_lib)

add_test(NAME test_mha_attn_bwd_bf16 COMMAND test_mha_attn_bwd)
set_tests_properties(test_mha_attn_bwd_bf16 PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_MHA_FLASH};MHA_EXPECT_FLASH=1;TEST_DTYPE=bf16")
add_test(NAME test_mha_attn_bwd_bf16_no_fa COMMAND test_mha_attn_bwd)
set_tests_properties(test_mha_attn_bwd_bf16_no_fa PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_MHA_TRITON};MHA_EXPECT_TRITON=1;TEST_DTYPE=bf16")
add_test(NAME test_mha_attn_bwd_bf16_no_fa_heads4 COMMAND test_mha_attn_bwd)
set_tests_properties(test_mha_attn_bwd_bf16_no_fa_heads4 PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_MHA_CUTLASS};MHA_EXPECT_CUTLASS=1;MHA_TEST_HEADS=4;TEST_DTYPE=bf16")
add_test(NAME test_mha_attn_bwd_bf16_no_cutlass COMMAND test_mha_attn_bwd)
set_tests_properties(test_mha_attn_bwd_bf16_no_cutlass PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_MHA_TRITON};MHA_EXPECT_TRITON=1;TEST_DTYPE=bf16")
add_test(NAME test_mha_attn_bwd_bf16_prefer_cutlass_heads4 COMMAND test_mha_attn_bwd)
set_tests_properties(test_mha_attn_bwd_bf16_prefer_cutlass_heads4 PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_MHA_CUTLASS};MHA_EXPECT_CUTLASS=1;MHA_TEST_HEADS=4;TEST_DTYPE=bf16")

add_test(NAME test_mha_attn_bwd_bf16_train_flash COMMAND test_mha_attn_bwd)
set_tests_properties(test_mha_attn_bwd_bf16_train_flash PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_MHA_FLASH};MHA_EXPECT_FLASH=1;TEST_DTYPE=bf16;MHA_TEST_HEADS=8;MHA_TEST_SEQLEN=480")
add_test(NAME test_mha_attn_bwd_bf16_train_cutlass COMMAND test_mha_attn_bwd)
set_tests_properties(test_mha_attn_bwd_bf16_train_cutlass PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_MHA_CUTLASS};MHA_EXPECT_CUTLASS=1;TEST_DTYPE=bf16;MHA_TEST_HEADS=8;MHA_TEST_SEQLEN=480")

add_test(NAME test_mha_attn_bwd_fp16 COMMAND test_mha_attn_bwd)
set_tests_properties(test_mha_attn_bwd_fp16 PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_MHA_FLASH};MHA_EXPECT_FLASH=1;TEST_DTYPE=fp16")
add_test(NAME test_mha_attn_bwd_fp16_no_fa COMMAND test_mha_attn_bwd)
set_tests_properties(test_mha_attn_bwd_fp16_no_fa PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_MHA_TRITON};MHA_EXPECT_TRITON=1;TEST_DTYPE=fp16")
add_test(NAME test_mha_attn_bwd_fp16_no_fa_heads4 COMMAND test_mha_attn_bwd)
set_tests_properties(test_mha_attn_bwd_fp16_no_fa_heads4 PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_MHA_CUTLASS};MHA_EXPECT_CUTLASS=1;MHA_TEST_HEADS=4;TEST_DTYPE=fp16")
add_test(NAME test_mha_attn_bwd_fp16_no_cutlass COMMAND test_mha_attn_bwd)
set_tests_properties(test_mha_attn_bwd_fp16_no_cutlass PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_MHA_TRITON};MHA_EXPECT_TRITON=1;TEST_DTYPE=fp16")
add_test(NAME test_mha_attn_bwd_fp16_prefer_cutlass_heads4 COMMAND test_mha_attn_bwd)
set_tests_properties(test_mha_attn_bwd_fp16_prefer_cutlass_heads4 PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_MHA_CUTLASS};MHA_EXPECT_CUTLASS=1;MHA_TEST_HEADS=4;TEST_DTYPE=fp16")

add_test(NAME test_mha_attn_bwd_fp16_train_flash COMMAND test_mha_attn_bwd)
set_tests_properties(test_mha_attn_bwd_fp16_train_flash PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_MHA_FLASH};MHA_EXPECT_FLASH=1;TEST_DTYPE=fp16;MHA_TEST_HEADS=8;MHA_TEST_SEQLEN=480")
add_test(NAME test_mha_attn_bwd_fp16_train_cutlass COMMAND test_mha_attn_bwd)
set_tests_properties(test_mha_attn_bwd_fp16_train_cutlass PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_MHA_CUTLASS};MHA_EXPECT_CUTLASS=1;TEST_DTYPE=fp16;MHA_TEST_HEADS=8;MHA_TEST_SEQLEN=480")

add_custom_command(TARGET test_mha_attn_bwd POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/reference_bf16.safetensors
        $<TARGET_FILE_DIR:test_mha_attn_bwd>/reference_bf16.safetensors
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/reference_fp16.safetensors
        $<TARGET_FILE_DIR:test_mha_attn_bwd>/reference_fp16.safetensors
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/reference_bf16_h4.safetensors
        $<TARGET_FILE_DIR:test_mha_attn_bwd>/reference_bf16_h4.safetensors
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/reference_fp16_h4.safetensors
        $<TARGET_FILE_DIR:test_mha_attn_bwd>/reference_fp16_h4.safetensors
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/reference_bf16_h8_t480.safetensors
        $<TARGET_FILE_DIR:test_mha_attn_bwd>/reference_bf16_h8_t480.safetensors
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/reference_fp16_h8_t480.safetensors
        $<TARGET_FILE_DIR:test_mha_attn_bwd>/reference_fp16_h8_t480.safetensors)
