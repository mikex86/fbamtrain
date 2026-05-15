add_library(ftccl STATIC src/ftccl_pccl.cpp)

target_include_directories(ftccl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(ftccl PUBLIC pccl)
