add_executable(test_frame_head test_frame_head.cpp)

target_link_libraries(test_frame_head PRIVATE tensorlib)
target_link_libraries(test_frame_head PRIVATE fbamtrain_lib)

add_test(NAME test_frame_head_bf16 COMMAND test_frame_head)
set_tests_properties(test_frame_head_bf16 PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_CUTLASS};TEST_DTYPE=bf16")
add_test(NAME test_frame_head_bf16_no_cutlass COMMAND test_frame_head)
set_tests_properties(test_frame_head_bf16_no_cutlass PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_TRITON};TEST_DTYPE=bf16")

add_test(NAME test_frame_head_fp16 COMMAND test_frame_head)
set_tests_properties(test_frame_head_fp16 PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_CUTLASS};TEST_DTYPE=fp16")
add_test(NAME test_frame_head_fp16_no_cutlass COMMAND test_frame_head)
set_tests_properties(test_frame_head_fp16_no_cutlass PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_TRITON};TEST_DTYPE=fp16")

add_custom_command(TARGET test_frame_head POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/reference_bf16.safetensors
        $<TARGET_FILE_DIR:test_frame_head>/reference_bf16.safetensors
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/reference_fp16.safetensors
        $<TARGET_FILE_DIR:test_frame_head>/reference_fp16.safetensors)
