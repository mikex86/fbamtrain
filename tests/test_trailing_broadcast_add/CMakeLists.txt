add_executable(test_trailing_broadcast_add test_trailing_broadcast_add.cpp)
add_executable(test_trailing_broadcast_add_fp32 test_trailing_broadcast_add_fp32.cpp)

target_link_libraries(test_trailing_broadcast_add PRIVATE tensorlib)
target_link_libraries(test_trailing_broadcast_add PRIVATE fbamtrain_lib)
target_link_libraries(test_trailing_broadcast_add_fp32 PRIVATE tensorlib)
target_link_libraries(test_trailing_broadcast_add_fp32 PRIVATE fbamtrain_lib)

add_test(NAME test_trailing_broadcast_add_bf16 COMMAND test_trailing_broadcast_add)
set_tests_properties(test_trailing_broadcast_add_bf16 PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_CUTLASS};TEST_DTYPE=bf16")
add_test(NAME test_trailing_broadcast_add_bf16_no_cutlass COMMAND test_trailing_broadcast_add)
set_tests_properties(test_trailing_broadcast_add_bf16_no_cutlass PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_TRITON};TEST_DTYPE=bf16")

add_test(NAME test_trailing_broadcast_add_fp16 COMMAND test_trailing_broadcast_add)
set_tests_properties(test_trailing_broadcast_add_fp16 PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_CUTLASS};TEST_DTYPE=fp16")
add_test(NAME test_trailing_broadcast_add_fp16_no_cutlass COMMAND test_trailing_broadcast_add)
set_tests_properties(test_trailing_broadcast_add_fp16_no_cutlass PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_TRITON};TEST_DTYPE=fp16")

add_test(NAME test_trailing_broadcast_add_fp32 COMMAND test_trailing_broadcast_add_fp32)

add_custom_command(TARGET test_trailing_broadcast_add POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/reference_bf16.safetensors
        $<TARGET_FILE_DIR:test_trailing_broadcast_add>/reference_bf16.safetensors
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/reference_fp16.safetensors
        $<TARGET_FILE_DIR:test_trailing_broadcast_add>/reference_fp16.safetensors)
