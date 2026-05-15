add_executable(test_reduce_sum test_reduce_sum.cpp)

target_link_libraries(test_reduce_sum PRIVATE tensorlib)
target_link_libraries(test_reduce_sum PRIVATE fbamtrain_lib)

add_test(NAME test_reduce_sum COMMAND test_reduce_sum)
