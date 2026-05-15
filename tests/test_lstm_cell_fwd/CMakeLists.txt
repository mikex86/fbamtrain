add_executable(test_lstm_cell_fwd test_lstm_cell_fwd.cpp)

target_link_libraries(test_lstm_cell_fwd PRIVATE tensorlib)
target_link_libraries(test_lstm_cell_fwd PRIVATE fbamtrain_lib)

add_test(NAME test_lstm_cell_fwd COMMAND test_lstm_cell_fwd)
set_tests_properties(test_lstm_cell_fwd PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_CUTLASS}")

add_custom_command(TARGET test_lstm_cell_fwd POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/reference.safetensors
        $<TARGET_FILE_DIR:test_lstm_cell_fwd>)
