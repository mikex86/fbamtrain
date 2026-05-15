add_library(ccl)

target_include_directories(ccl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if (TENSORLIB_ENABLE_CUDA)
    target_sources(ccl PRIVATE src/ccl_nccl.cpp)
    target_compile_definitions(ccl PUBLIC PI_TENSORLIB_ENABLE_CUDA=1)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/nccl.cmake)
else ()
    target_sources(ccl PRIVATE src/ccl_stub.cpp)
    target_compile_definitions(ccl PUBLIC PI_TENSORLIB_ENABLE_CUDA=0)
endif ()

target_link_libraries(ccl PUBLIC tensorlib)
