add_executable(test_fill_zeros test_fill_zeros.cpp)

target_link_libraries(test_fill_zeros PRIVATE tensorlib)
target_link_libraries(test_fill_zeros PRIVATE fbamtrain_lib)

add_test(NAME test_fill_zeros_bf16 COMMAND test_fill_zeros)
set_tests_properties(test_fill_zeros_bf16 PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_CUTLASS};TEST_DTYPE=bf16")
add_test(NAME test_fill_zeros_bf16_no_cutlass COMMAND test_fill_zeros)
set_tests_properties(test_fill_zeros_bf16_no_cutlass PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_TRITON};TEST_DTYPE=bf16")

add_test(NAME test_fill_zeros_fp16 COMMAND test_fill_zeros)
set_tests_properties(test_fill_zeros_fp16 PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_CUTLASS};TEST_DTYPE=fp16")
add_test(NAME test_fill_zeros_fp16_no_cutlass COMMAND test_fill_zeros)
set_tests_properties(test_fill_zeros_fp16_no_cutlass PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_TRITON};TEST_DTYPE=fp16")
