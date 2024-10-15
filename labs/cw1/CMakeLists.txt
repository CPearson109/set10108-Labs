cmake_minimum_required(VERSION 3.17)
set(CMAKE_CXX_STANDARD 17)

# Include the directories for the project
include_directories(../contrib)

# Create a CPU target using main-cpu.cpp
add_executable(cw1_cpu main_cpu.cpp)

# Enable CUDA for GPU project
enable_language(CUDA)

# Create a GPU target using main-gpu.cu
add_executable(cw1_gpu main_gpu.cu)

# Add any required libraries for CUDA here, for example:
# target_link_libraries(cw1-gpu cuda cudart)
