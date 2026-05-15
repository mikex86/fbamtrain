add_executable(bench_mha_mfu bench_mha_mfu.cpp)
target_link_libraries(bench_mha_mfu PRIVATE tensorlib fbamtrain_lib)
add_test(NAME bench_mha_mfu COMMAND bench_mha_mfu)
