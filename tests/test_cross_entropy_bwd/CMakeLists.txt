add_executable(test_cross_entropy_bwd test_cross_entropy_bwd.cpp)

target_link_libraries(test_cross_entropy_bwd PRIVATE tensorlib)
target_link_libraries(test_cross_entropy_bwd PRIVATE fbamtrain_lib)

add_test(NAME test_cross_entropy_bwd COMMAND test_cross_entropy_bwd)
set_tests_properties(test_cross_entropy_bwd PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_CUTLASS}")
add_test(NAME test_cross_entropy_bwd_no_cutlass COMMAND test_cross_entropy_bwd)
set_tests_properties(test_cross_entropy_bwd_no_cutlass PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_TRITON}")

add_custom_command(TARGET test_cross_entropy_bwd POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/reference.safetensors
        $<TARGET_FILE_DIR:test_cross_entropy_bwd>)
