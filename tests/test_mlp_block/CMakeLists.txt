add_executable(test_mlp_block test_mlp_block.cpp)

target_link_libraries(test_mlp_block PRIVATE tensorlib)
target_link_libraries(test_mlp_block PRIVATE fbamtrain_lib)

add_test(NAME test_mlp_block COMMAND test_mlp_block)
set_tests_properties(test_mlp_block PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_CUTLASS}")
add_test(NAME test_mlp_block_no_cutlass COMMAND test_mlp_block)
set_tests_properties(test_mlp_block_no_cutlass PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_TRITON}")

add_custom_command(TARGET test_mlp_block POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/reference.safetensors
        $<TARGET_FILE_DIR:test_mlp_block>)
