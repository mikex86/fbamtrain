add_executable(test_matmul_accumulate test_matmul_accumulate.cpp)

target_link_libraries(test_matmul_accumulate PRIVATE tensorlib)
target_link_libraries(test_matmul_accumulate PRIVATE fbamtrain_lib)

add_test(NAME test_matmul_accumulate COMMAND test_matmul_accumulate)
set_tests_properties(test_matmul_accumulate PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_CUTLASS}")

add_test(NAME test_matmul_accumulate_no_cutlass COMMAND test_matmul_accumulate)
set_tests_properties(
        test_matmul_accumulate_no_cutlass
        PROPERTIES
        ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_TRITON}")
