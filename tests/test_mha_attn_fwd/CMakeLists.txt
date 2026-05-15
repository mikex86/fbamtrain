add_executable(test_mha_attn_fwd test_mha_attn_fwd.cpp)

target_link_libraries(test_mha_attn_fwd PRIVATE tensorlib)
target_link_libraries(test_mha_attn_fwd PRIVATE fbamtrain_lib)

add_test(NAME test_mha_attn_fwd_bf16 COMMAND test_mha_attn_fwd)
set_tests_properties(test_mha_attn_fwd_bf16 PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_MHA_FLASH};MHA_EXPECT_FLASH=1;TEST_DTYPE=bf16")
add_test(NAME test_mha_attn_fwd_bf16_no_fa COMMAND test_mha_attn_fwd)
set_tests_properties(test_mha_attn_fwd_bf16_no_fa PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_MHA_TRITON};MHA_EXPECT_TRITON=1;TEST_DTYPE=bf16")
add_test(NAME test_mha_attn_fwd_bf16_no_cutlass COMMAND test_mha_attn_fwd)
set_tests_properties(test_mha_attn_fwd_bf16_no_cutlass PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_MHA_TRITON};MHA_EXPECT_TRITON=1;TEST_DTYPE=bf16")

add_test(NAME test_mha_attn_fwd_fp16 COMMAND test_mha_attn_fwd)
set_tests_properties(test_mha_attn_fwd_fp16 PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_MHA_FLASH};MHA_EXPECT_FLASH=1;TEST_DTYPE=fp16")
add_test(NAME test_mha_attn_fwd_fp16_no_fa COMMAND test_mha_attn_fwd)
set_tests_properties(test_mha_attn_fwd_fp16_no_fa PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_MHA_TRITON};MHA_EXPECT_TRITON=1;TEST_DTYPE=fp16")
add_test(NAME test_mha_attn_fwd_fp16_no_cutlass COMMAND test_mha_attn_fwd)
set_tests_properties(test_mha_attn_fwd_fp16_no_cutlass PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_MHA_TRITON};MHA_EXPECT_TRITON=1;TEST_DTYPE=fp16")

add_test(NAME test_mha_attn_fwd_train_bf16_flash COMMAND test_mha_attn_fwd)
set_tests_properties(test_mha_attn_fwd_train_bf16_flash PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_MHA_FLASH};MHA_EXPECT_FLASH=1;TEST_DTYPE=bf16;MHA_TEST_BATCH=2;MHA_TEST_HEADS=8;MHA_TEST_SEQLEN=480;MHA_TEST_USE_SCRATCH=1")
add_test(NAME test_mha_attn_fwd_train_fp16_flash COMMAND test_mha_attn_fwd)
set_tests_properties(test_mha_attn_fwd_train_fp16_flash PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_MHA_FLASH};MHA_EXPECT_FLASH=1;TEST_DTYPE=fp16;MHA_TEST_BATCH=2;MHA_TEST_HEADS=8;MHA_TEST_SEQLEN=480;MHA_TEST_USE_SCRATCH=1")
add_test(NAME test_mha_attn_fwd_train_bf16_cutlass COMMAND test_mha_attn_fwd)
set_tests_properties(test_mha_attn_fwd_train_bf16_cutlass PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_MHA_CUTLASS};MHA_EXPECT_CUTLASS=1;TEST_DTYPE=bf16;MHA_TEST_BATCH=2;MHA_TEST_HEADS=8;MHA_TEST_SEQLEN=480;MHA_TEST_USE_SCRATCH=1")
add_test(NAME test_mha_attn_fwd_train_fp16_cutlass COMMAND test_mha_attn_fwd)
set_tests_properties(test_mha_attn_fwd_train_fp16_cutlass PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_MHA_CUTLASS};MHA_EXPECT_CUTLASS=1;TEST_DTYPE=fp16;MHA_TEST_BATCH=2;MHA_TEST_HEADS=8;MHA_TEST_SEQLEN=480;MHA_TEST_USE_SCRATCH=1")

# make sure reference safetensors are available in the build directory
add_custom_command(TARGET test_mha_attn_fwd POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/reference_bf16.safetensors
        $<TARGET_FILE_DIR:test_mha_attn_fwd>/reference_bf16.safetensors
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/reference_fp16.safetensors
        $<TARGET_FILE_DIR:test_mha_attn_fwd>/reference_fp16.safetensors
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/reference_bf16_b2_t480.safetensors
        $<TARGET_FILE_DIR:test_mha_attn_fwd>/reference_bf16_b2_t480.safetensors
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/reference_fp16_b2_t480.safetensors
        $<TARGET_FILE_DIR:test_mha_attn_fwd>/reference_fp16_b2_t480.safetensors)
