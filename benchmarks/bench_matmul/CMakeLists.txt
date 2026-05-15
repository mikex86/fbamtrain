add_executable(bench_matmul bench_matmul.cpp)

target_link_libraries(bench_matmul PRIVATE tensorlib)
target_link_libraries(bench_matmul PRIVATE fbamtrain_lib)
target_include_directories(bench_matmul PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)

add_test(NAME bench_matmul COMMAND bench_matmul)
