add_executable(test_mean_bwd test_mean_bwd.cpp)

target_link_libraries(test_mean_bwd PRIVATE tensorlib)
target_link_libraries(test_mean_bwd PRIVATE fbamtrain_lib)

add_test(NAME test_mean_bwd COMMAND test_mean_bwd)
set_tests_properties(test_mean_bwd PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_CUTLASS}")
add_test(NAME test_mean_bwd_no_cutlass COMMAND test_mean_bwd)
set_tests_properties(test_mean_bwd_no_cutlass PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_TRITON}")

add_custom_command(TARGET test_mean_bwd POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/reference.safetensors
        $<TARGET_FILE_DIR:test_mean_bwd>)
