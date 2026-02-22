# CMake configuration for 01_sample_tests

cmake_minimum_required(VERSION 3.18)
project(sample_tests LANGUAGES CUDA)

enable_language(CUDA)

# Set build output directory to /build
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build/lib")

# Add executable target for test._compile.cu
add_executable(sample_tests_compile test._compile.cu)

# Optionally, set CUDA as required language if needed
