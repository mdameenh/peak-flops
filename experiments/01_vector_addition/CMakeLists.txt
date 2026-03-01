enable_language(CUDA)

# CMake configuration for 01_vector_addition experiment
cmake_minimum_required(VERSION 3.18)
project(vector_addition LANGUAGES CUDA)

# Ensure CUDA language is enabled
enable_language(CUDA)

# Set build output directory to /build
if(NOT CMAKE_BINARY_DIR STREQUAL "${CMAKE_SOURCE_DIR}/build")
	set(CMAKE_BINARY_DIR "${CMAKE_SOURCE_DIR}/build")
endif()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# Add executable target for test_compile.cu
add_executable(sample_tests_compile test_compile.cu)

# Optionally, set CUDA properties or flags here
# set_target_properties(sample_tests_compile PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
