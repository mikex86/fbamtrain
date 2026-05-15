add_executable(test_matmul_fp32_out test_matmul_fp32_out.cpp)

target_link_libraries(test_matmul_fp32_out PRIVATE tensorlib)
target_link_libraries(test_matmul_fp32_out PRIVATE fbamtrain_lib)

add_test(NAME test_matmul_fp32_out COMMAND test_matmul_fp32_out)
set_tests_properties(test_matmul_fp32_out PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_CUTLASS}")
add_test(NAME test_matmul_fp32_out_no_cutlass COMMAND test_matmul_fp32_out)
set_tests_properties(test_matmul_fp32_out_no_cutlass PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_TRITON}")

add_custom_command(TARGET test_matmul_fp32_out POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/reference.safetensors
        $<TARGET_FILE_DIR:test_matmul_fp32_out>)
