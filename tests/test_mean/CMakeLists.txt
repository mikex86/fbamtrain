add_executable(test_mean test_mean.cpp)

target_link_libraries(test_mean PRIVATE tensorlib)
target_link_libraries(test_mean PRIVATE fbamtrain_lib)

add_test(NAME test_mean COMMAND test_mean)
set_tests_properties(test_mean PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_CUTLASS}")
add_test(NAME test_mean_no_cutlass COMMAND test_mean)
set_tests_properties(test_mean_no_cutlass PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_TRITON}")

add_custom_command(TARGET test_mean POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/reference.safetensors
        $<TARGET_FILE_DIR:test_mean>)
