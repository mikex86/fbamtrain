add_executable(test_lstm_fwd_simple test_lstm_fwd_simple.cpp)

target_link_libraries(test_lstm_fwd_simple PRIVATE tensorlib)
target_link_libraries(test_lstm_fwd_simple PRIVATE fbamtrain_lib)

add_test(NAME test_lstm_fwd_simple COMMAND test_lstm_fwd_simple)
set_tests_properties(test_lstm_fwd_simple PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_CUTLASS}")

add_custom_command(TARGET test_lstm_fwd_simple POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/reference.safetensors
        $<TARGET_FILE_DIR:test_lstm_fwd_simple>)

add_executable(test_lstm_fwd_downcast_plus test_lstm_fwd_downcast_plus.cpp)

target_link_libraries(test_lstm_fwd_downcast_plus PRIVATE tensorlib)
target_link_libraries(test_lstm_fwd_downcast_plus PRIVATE fbamtrain_lib)

add_test(NAME test_lstm_fwd_downcast_plus COMMAND test_lstm_fwd_downcast_plus)
set_tests_properties(test_lstm_fwd_downcast_plus PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_CUTLASS}")

add_custom_command(TARGET test_lstm_fwd_downcast_plus POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/reference.safetensors
        $<TARGET_FILE_DIR:test_lstm_fwd_downcast_plus>)
