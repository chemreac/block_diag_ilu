set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
add_definitions(-DUNIT_TEST)
add_executable (test_block_diag_ilu test_block_diag_ilu.cpp)
target_link_libraries(test_block_diag_ilu lapack)