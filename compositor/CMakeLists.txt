cmake_minimum_required(VERSION 2.6)
project(Compositor)

set(CMAKE_CXX_COMPILER clang++)

# C++14 is rad
set(CMAKE_CXX_FLAGS "-g -std=c++14 -Wall ${CMAKE_CXX_FLAGS}")

include_directories(.)

add_executable(compositor compositor.cpp)
target_link_libraries(compositor boost_system boost_program_options boost_filesystem IlmImf Imath Half Iex IexMath IlmThread pthread)
