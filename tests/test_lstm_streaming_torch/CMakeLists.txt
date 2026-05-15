add_executable(test_lstm_streaming_torch test_lstm_streaming_torch.cpp)

target_link_libraries(test_lstm_streaming_torch PRIVATE tensorlib)
target_link_libraries(test_lstm_streaming_torch PRIVATE fbamtrain_lib)

add_test(NAME test_lstm_streaming_torch COMMAND test_lstm_streaming_torch)
set_tests_properties(test_lstm_streaming_torch PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_CUTLASS}")

add_custom_command(TARGET test_lstm_streaming_torch POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/reference.safetensors
        $<TARGET_FILE_DIR:test_lstm_streaming_torch>)
