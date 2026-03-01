

cmake_minimum_required(VERSION 3.18)
project(peak_flops)

# Set build output directory to /build
set(CMAKE_BINARY_DIR "${CMAKE_SOURCE_DIR}/build")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# Add subdirectories for experiments and projects
add_subdirectory(experiments/01_vector_addition)
add_subdirectory(experiments/02_axpy)
