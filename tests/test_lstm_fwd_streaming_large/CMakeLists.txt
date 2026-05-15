add_executable(test_lstm_fwd_streaming_large test_lstm_fwd_streaming_large.cpp)

target_link_libraries(test_lstm_fwd_streaming_large PRIVATE tensorlib)
target_link_libraries(test_lstm_fwd_streaming_large PRIVATE fbamtrain_lib)

add_test(NAME test_lstm_fwd_streaming_large COMMAND test_lstm_fwd_streaming_large)
set_tests_properties(test_lstm_fwd_streaming_large PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_CUTLASS}")

add_custom_command(TARGET test_lstm_fwd_streaming_large POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/reference.safetensors
        $<TARGET_FILE_DIR:test_lstm_fwd_streaming_large>)
