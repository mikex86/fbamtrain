add_executable(test_mfu test_mfu.cpp)
target_link_libraries(test_mfu PRIVATE tensorlib fbamtrain_lib)
add_test(NAME test_mfu COMMAND test_mfu)
