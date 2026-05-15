add_executable(bench_conv2d bench_conv2d.cpp)

target_link_libraries(bench_conv2d PRIVATE tensorlib)
target_link_libraries(bench_conv2d PRIVATE fbamtrain_lib)
target_include_directories(bench_conv2d PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)

add_test(NAME bench_conv2d COMMAND bench_conv2d)
