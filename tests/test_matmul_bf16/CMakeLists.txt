add_executable(test_matmul_bf16 test_matmul_bf16.cpp)
target_link_libraries(test_matmul_bf16 PRIVATE tensorlib)
target_link_libraries(test_matmul_bf16 PRIVATE fbamtrain_lib)

add_test(NAME test_matmul_bf16 COMMAND test_matmul_bf16)
set_tests_properties(test_matmul_bf16 PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_CUTLASS}")

add_test(NAME test_matmul_bf16_no_cutlass COMMAND test_matmul_bf16)
set_tests_properties(test_matmul_bf16_no_cutlass PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_TRITON}")
