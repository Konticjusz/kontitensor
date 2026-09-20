cmake_minimum_required(VERSION 3.20)

project(tinytensor
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Library
add_library(tinytensor
    src/tensor.cpp
    include/tinytensor/tensor.hpp
)

target_include_directories(tinytensor
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(tinytensor
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

# Tests
add_executable(test_tensor
    tests/test_tensor.cpp
)

target_link_libraries(test_tensor
    PRIVATE
        tinytensor
)

target_compile_options(test_tensor
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)