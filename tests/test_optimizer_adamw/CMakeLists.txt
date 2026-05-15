add_executable(test_optimizer_adamw test_optimizer_adamw.cpp)

target_link_libraries(test_optimizer_adamw PRIVATE tensorlib)
target_link_libraries(test_optimizer_adamw PRIVATE fbamtrain_lib)

add_test(NAME test_optimizer_adamw COMMAND test_optimizer_adamw)

add_custom_command(TARGET test_optimizer_adamw POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/reference.safetensors
        $<TARGET_FILE_DIR:test_optimizer_adamw>/reference.safetensors)
