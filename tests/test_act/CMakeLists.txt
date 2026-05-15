add_executable(test_act test_act.cpp)

target_link_libraries(test_act PRIVATE tensorlib)
target_link_libraries(test_act PRIVATE fbamtrain_lib)

add_test(NAME test_act_bf16 COMMAND test_act)
set_tests_properties(test_act_bf16 PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_CUTLASS};TEST_DTYPE=bf16")
add_test(NAME test_act_bf16_no_cutlass COMMAND test_act)
set_tests_properties(test_act_bf16_no_cutlass PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_TRITON};TEST_DTYPE=bf16")

add_test(NAME test_act_fp16 COMMAND test_act)
set_tests_properties(test_act_fp16 PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_CUTLASS};TEST_DTYPE=fp16")
add_test(NAME test_act_fp16_no_cutlass COMMAND test_act)
set_tests_properties(test_act_fp16_no_cutlass PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_TRITON};TEST_DTYPE=fp16")

add_custom_command(TARGET test_act POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/reference_bf16.safetensors
        $<TARGET_FILE_DIR:test_act>/reference_bf16.safetensors
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/reference_fp16.safetensors
        $<TARGET_FILE_DIR:test_act>/reference_fp16.safetensors)
