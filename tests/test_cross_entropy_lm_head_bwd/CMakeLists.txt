add_executable(test_cross_entropy_lm_head_bwd test_cross_entropy_lm_head_bwd.cpp)

target_link_libraries(test_cross_entropy_lm_head_bwd PRIVATE tensorlib)
target_link_libraries(test_cross_entropy_lm_head_bwd PRIVATE fbamtrain_lib)

add_test(NAME test_cross_entropy_lm_head_bwd_fp16 COMMAND test_cross_entropy_lm_head_bwd fp16)
set_tests_properties(test_cross_entropy_lm_head_bwd_fp16 PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_CUTLASS}")
add_test(NAME test_cross_entropy_lm_head_bwd_bf16 COMMAND test_cross_entropy_lm_head_bwd bf16)
set_tests_properties(test_cross_entropy_lm_head_bwd_bf16 PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_CUTLASS}")
add_test(NAME test_cross_entropy_lm_head_bwd_fp16_no_cutlass COMMAND test_cross_entropy_lm_head_bwd fp16)
set_tests_properties(test_cross_entropy_lm_head_bwd_fp16_no_cutlass PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_TRITON}")
add_test(NAME test_cross_entropy_lm_head_bwd_bf16_no_cutlass COMMAND test_cross_entropy_lm_head_bwd bf16)
set_tests_properties(test_cross_entropy_lm_head_bwd_bf16_no_cutlass PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_TRITON}")

add_custom_command(TARGET test_cross_entropy_lm_head_bwd POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/reference.safetensors
        $<TARGET_FILE_DIR:test_cross_entropy_lm_head_bwd>)
