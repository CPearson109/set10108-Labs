cmake_minimum_required(VERSION 3.17)

# Set the C++ standard to 17
set(CMAKE_CXX_STANDARD 17)

# Include the directories for the project
include_directories(../contrib)

# Create a CPU target using main_cpu.cpp
add_executable(cw1_cpu main_cpu.cpp)

# Enable CUDA for GPU project
enable_language(CUDA)

# Create a GPU target using main_gpu.cu
add_executable(cw1_gpu main_gpu.cu)

# Create an optimized GPU target using main_gpu_optimised.cu
add_executable(cw1_gpu_optimised main_gpu_optimised.cu)

# Link required libraries for CUDA for both GPU targets
find_package(CUDA REQUIRED)
target_link_libraries(cw1_gpu cuda cudart)
target_link_libraries(cw1_gpu_optimised cuda cudart)
