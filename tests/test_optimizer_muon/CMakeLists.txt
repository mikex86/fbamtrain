add_executable(test_optimizer_muon test_optimizer_muon.cpp)

target_link_libraries(test_optimizer_muon PRIVATE tensorlib)
target_link_libraries(test_optimizer_muon PRIVATE fbamtrain_lib)

add_test(NAME test_optimizer_muon COMMAND test_optimizer_muon)

add_custom_command(TARGET test_optimizer_muon POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/reference.safetensors
        $<TARGET_FILE_DIR:test_optimizer_muon>/reference.safetensors)
