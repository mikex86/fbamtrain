add_executable(test_avg_pool1d test_avg_pool1d.cpp)

target_link_libraries(test_avg_pool1d PRIVATE tensorlib)
target_link_libraries(test_avg_pool1d PRIVATE fbamtrain_lib)

add_test(NAME test_avg_pool1d COMMAND test_avg_pool1d)
set_tests_properties(test_avg_pool1d PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_CUTLASS}")
add_test(NAME test_avg_pool1d_no_cutlass COMMAND test_avg_pool1d)
set_tests_properties(test_avg_pool1d_no_cutlass PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_TRITON}")

add_custom_command(TARGET test_avg_pool1d POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/reference.safetensors
        $<TARGET_FILE_DIR:test_avg_pool1d>)
