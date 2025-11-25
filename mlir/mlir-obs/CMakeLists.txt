cmake_minimum_required(VERSION 3.13)
project(MLIROBF LANGUAGES CXX C)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# MLIR is already part of the parent build — DO NOT use find_package(MLIR)

# Add include path for headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Add the library directory
add_subdirectory(lib)
