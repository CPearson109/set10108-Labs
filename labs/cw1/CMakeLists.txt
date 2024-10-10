# Project name and version
project(set10108-cw1 LANGUAGES CXX CUDA)
cmake_minimum_required(VERSION 3.17)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Enable CUDA support
find_package(CUDA REQUIRED)

# Include directories (you can modify the path if necessary)
include_directories(${CUDA_INCLUDE_DIRS} ../contrib)

# Add the executable, using CUDA
add_executable(cw1 main.cu)

# Ensure that CUDA is enabled for the target
set_target_properties(cw1 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
