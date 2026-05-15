add_executable(test_matmul_transpose test_matmul_transpose.cpp)
target_link_libraries(test_matmul_transpose PRIVATE tensorlib)
target_link_libraries(test_matmul_transpose PRIVATE fbamtrain_lib)
add_test(NAME test_matmul_transpose COMMAND test_matmul_transpose)
set_tests_properties(test_matmul_transpose PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_CUTLASS}")
add_test(NAME test_matmul_transpose_no_cutlass COMMAND test_matmul_transpose)
set_tests_properties(test_matmul_transpose_no_cutlass PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_TRITON}")
add_custom_command(TARGET test_matmul_transpose POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/reference.safetensors
        $<TARGET_FILE_DIR:test_matmul_transpose>)
