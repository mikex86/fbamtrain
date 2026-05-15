add_executable(test_div_add test_div_add.cpp)

target_link_libraries(test_div_add PRIVATE tensorlib)
target_link_libraries(test_div_add PRIVATE fbamtrain_lib)

add_test(NAME test_div_add COMMAND test_div_add)
set_tests_properties(test_div_add PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_CUTLASS}")
add_test(NAME test_div_add_no_cutlass COMMAND test_div_add)
set_tests_properties(test_div_add_no_cutlass PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_TRITON}")

add_custom_command(TARGET test_div_add POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/reference.safetensors
        $<TARGET_FILE_DIR:test_div_add>)
