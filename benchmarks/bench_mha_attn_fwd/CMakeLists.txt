add_executable(bench_mha_attn_fwd bench_mha_attn_fwd.cpp)

target_link_libraries(bench_mha_attn_fwd PRIVATE tensorlib)
target_link_libraries(bench_mha_attn_fwd PRIVATE fbamtrain_lib)
target_include_directories(bench_mha_attn_fwd PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)

add_test(NAME bench_mha_attn_fwd COMMAND bench_mha_attn_fwd)
