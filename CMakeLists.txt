# CMakeLists.txt (root)
cmake_minimum_required(VERSION 3.16)
project(tangent_matmul_tests CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra -Wpedantic -Wconversion -Wdouble-promotion)
# Optional but very helpful:
add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
link_libraries(asan ubsan)

# Library under test
add_library(tangent INTERFACE)
# target_include_directories(tangent INTERFACE ${CMAKE_SOURCE_DIR}/include)

include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.6.0 # pick a stable release
)
FetchContent_MakeAvailable(Catch2)

# Tests
add_executable(tests
  tests/matrix_tests.cpp
)
find_package(Catch2)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)

# Fetch Catch2 header (option A): vendor it
# Put catch_amalgamated.hpp in third_party/catch2/ and add that include path:
target_include_directories(tests PRIVATE ${CMAKE_SOURCE_DIR}/third_party/catch2)

target_include_directories(tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party   # for catch2/catch_amalgamated.hpp
)
