add_executable(test_conv2d test_conv2d.cpp)

target_link_libraries(test_conv2d PRIVATE tensorlib)
target_link_libraries(test_conv2d PRIVATE fbamtrain_lib)

add_test(NAME test_conv2d COMMAND test_conv2d)
set_tests_properties(test_conv2d PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_CUTLASS}")
add_test(NAME test_conv2d_no_cutlass COMMAND test_conv2d)
set_tests_properties(test_conv2d_no_cutlass PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_TRITON}")

add_custom_command(TARGET test_conv2d POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/reference.safetensors
        $<TARGET_FILE_DIR:test_conv2d>)
