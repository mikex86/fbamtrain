add_executable(test_embedding test_embedding.cpp)

target_link_libraries(test_embedding PRIVATE tensorlib)
target_link_libraries(test_embedding PRIVATE fbamtrain_lib)

add_test(NAME test_embedding COMMAND test_embedding)
set_tests_properties(test_embedding PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_CUTLASS}")
add_test(NAME test_embedding_no_cutlass COMMAND test_embedding)
set_tests_properties(test_embedding_no_cutlass PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_TRITON}")

add_custom_command(TARGET test_embedding POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/reference.safetensors
        $<TARGET_FILE_DIR:test_embedding>)
