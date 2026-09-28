cmake_minimum_required(VERSION 3.20)

project(kontitensor
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Library
add_library(kontitensor
    src/tensor.cpp
    src/shape.cpp
    src/optimizer.cpp
)

target_include_directories(kontitensor
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(kontitensor
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
        kontitensor
)


target_compile_options(test_tensor
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

# Tests
enable_testing()


target_link_libraries(test_tensor PRIVATE kontitensor)

add_test(
    NAME tensor_tests
    COMMAND test_tensor
)

add_executable(mnist
    examples/mnist.cpp
)

target_link_libraries(mnist PRIVATE kontitensor)