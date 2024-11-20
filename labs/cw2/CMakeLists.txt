# CMakeLists.txt
cmake_minimum_required(VERSION 3.18)
project(set10108-cw2 CUDA CXX)

# Enable C++17 and CUDA
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_language(CUDA)

# Specify CUDA architectures to target
# Adjust these based on your GPU. Common architectures:
# 60: Pascal, 61: Pascal, 75: Turing, 86: Ampere, 89: Hopper
set(CMAKE_CUDA_ARCHITECTURES 75 86)

# Include directories
include_directories("${CMAKE_SOURCE_DIR}/../contrib")
include_directories("${CMAKE_SOURCE_DIR}/../contrib/sfml/include")
include_directories("${CMAKE_SOURCE_DIR}/../contrib/stb") # Assuming stb_image.h is here

# Link directories for SFML
link_directories("${CMAKE_SOURCE_DIR}/../contrib/sfml/lib/Debug")
link_directories("${CMAKE_SOURCE_DIR}/../contrib/sfml/lib/Release")

# Find SFML for non-Windows systems
if (NOT WIN32)
    find_package(SFML 2.5 COMPONENTS system window graphics REQUIRED)
    if(NOT SFML_FOUND)
        message(FATAL_ERROR "SFML not found. Please install SFML 2.5 or later.")
    endif()
endif()

# Add executable
add_executable(cw2_main cw2_main.cpp colour_temperature.cu)

# Link libraries
if (WIN32)
    target_link_libraries(cw2_main
        optimized sfml-system
        optimized sfml-window
        optimized sfml-graphics
        debug sfml-system-d
        debug sfml-window-d
        debug sfml-graphics-d
        cudart
    )
else()
    target_link_libraries(cw2_main
        sfml-system
        sfml-window
        sfml-graphics
        cudart
    )
endif()

# Post-build commands to copy SFML DLLs for Windows
if (WIN32)
    # Copy Debug DLLs when building Debug configuration
    add_custom_command(TARGET cw2_main POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/../contrib/sfml/lib/Debug"
        "$<TARGET_FILE_DIR:cw2_main>"
        COMMENT "Copying SFML Debug DLLs to executable directory"
        VERBATIM
    )

    # Copy Release DLLs when building Release configuration
    add_custom_command(TARGET cw2_main POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/../contrib/sfml/lib/Release"
        "$<TARGET_FILE_DIR:cw2_main>"
        COMMENT "Copying SFML Release DLLs to executable directory"
        VERBATIM
    )
endif()
