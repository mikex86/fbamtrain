add_executable(test_build_cell_embed test_build_cell_embed.cpp)

target_link_libraries(test_build_cell_embed PRIVATE tensorlib)
target_link_libraries(test_build_cell_embed PRIVATE fbamtrain_lib)

add_test(NAME test_build_cell_embed COMMAND test_build_cell_embed)
set_tests_properties(test_build_cell_embed PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_CUTLASS}")
add_test(NAME test_build_cell_embed_no_cutlass COMMAND test_build_cell_embed)
set_tests_properties(test_build_cell_embed_no_cutlass PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_TRITON}")

add_custom_command(TARGET test_build_cell_embed POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/reference.safetensors
        $<TARGET_FILE_DIR:test_build_cell_embed>)
