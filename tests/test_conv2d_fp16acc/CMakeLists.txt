add_executable(test_conv2d_fp16acc test_conv2d_fp16acc.cpp)

target_link_libraries(test_conv2d_fp16acc PRIVATE tensorlib)
target_link_libraries(test_conv2d_fp16acc PRIVATE fbamtrain_lib)

add_test(NAME test_conv2d_fp16acc COMMAND test_conv2d_fp16acc)
set_tests_properties(test_conv2d_fp16acc PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_CUTLASS}")
add_test(NAME test_conv2d_fp16acc_no_cutlass COMMAND test_conv2d_fp16acc)
set_tests_properties(test_conv2d_fp16acc_no_cutlass PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_TRITON}")

add_custom_command(TARGET test_conv2d_fp16acc POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/reference.safetensors
        $<TARGET_FILE_DIR:test_conv2d_fp16acc>)
