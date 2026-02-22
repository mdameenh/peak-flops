
cmake_minimum_required(VERSION 3.18)
project(peak_flops)

# Set build output directory to /build
set(CMAKE_BINARY_DIR "${CMAKE_SOURCE_DIR}/build")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# Add subdirectories for projects
add_subdirectory(experiments/01_sample_tests)

# Optionally add benchmarks and experiments if they have CMakeLists.txt
# add_subdirectory(benchmarks)
# add_subdirectory(experiments)

# You can add more subdirectories as needed

# Optionally, add custom targets for documentation or scripts
# add_custom_target(docs ALL COMMAND ${CMAKE_COMMAND} -E echo "Build docs")
