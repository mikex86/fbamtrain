add_executable(test_lstm_streaming_ce test_lstm_streaming_ce.cpp)

target_link_libraries(test_lstm_streaming_ce PRIVATE tensorlib)
target_link_libraries(test_lstm_streaming_ce PRIVATE fbamtrain_lib)

add_test(NAME test_lstm_streaming_ce COMMAND test_lstm_streaming_ce)
set_tests_properties(test_lstm_streaming_ce PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_CUTLASS}")

add_custom_command(TARGET test_lstm_streaming_ce POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/reference.safetensors
        $<TARGET_FILE_DIR:test_lstm_streaming_ce>/reference.safetensors)
