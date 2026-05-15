add_executable(test_cross_entropy test_cross_entropy.cpp)

target_link_libraries(test_cross_entropy PRIVATE tensorlib)
target_link_libraries(test_cross_entropy PRIVATE fbamtrain_lib)

add_test(NAME test_cross_entropy COMMAND test_cross_entropy)
set_tests_properties(test_cross_entropy PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_CUTLASS}")
add_test(NAME test_cross_entropy_no_cutlass COMMAND test_cross_entropy)
set_tests_properties(test_cross_entropy_no_cutlass PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_TRITON}")

add_custom_command(TARGET test_cross_entropy POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/reference.safetensors
        $<TARGET_FILE_DIR:test_cross_entropy>)
