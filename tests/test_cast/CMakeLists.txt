add_executable(test_cast test_cast.cpp)

target_link_libraries(test_cast PRIVATE tensorlib)
target_link_libraries(test_cast PRIVATE fbamtrain_lib)

add_test(NAME test_cast COMMAND test_cast)
set_tests_properties(test_cast PROPERTIES ENVIRONMENT "${FBAMTRAIN_ENV_PREFER_CUTLASS}")
