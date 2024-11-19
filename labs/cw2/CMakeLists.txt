project(set10108-cw2)
cmake_minimum_required(VERSION 3.17)
set(CMAKE_CXX_STANDARD 17)

# Include directories
include_directories("${CMAKE_SOURCE_DIR}/../contrib")
if (WIN32)
    include_directories("${CMAKE_SOURCE_DIR}/../contrib/sfml/include")
    link_directories("${CMAKE_SOURCE_DIR}/../contrib/sfml/lib/Debug")
    link_directories("${CMAKE_SOURCE_DIR}/../contrib/sfml/lib/Release")
else()
    find_package(SFML 2.5 COMPONENTS window graphics REQUIRED)
    if(NOT SFML_FOUND)
        message(FATAL_ERROR "SFML not found. Please install SFML 2.5 or later.")
    endif()
endif()

# Add executables
add_executable(cw2_unoptimised cw2_unoptimised.cpp)
add_executable(cw2_optimised cw2_optimised.cpp)
add_executable(cw2_cpu cw2_cpu.cpp) # Add the new CPU-only executable

# Link libraries
target_link_libraries(cw2_unoptimised 
    optimized sfml-system optimized sfml-window optimized sfml-graphics 
    debug sfml-system-d debug sfml-window-d debug sfml-graphics-d
)

target_link_libraries(cw2_optimised 
    optimized sfml-system optimized sfml-window optimized sfml-graphics 
    debug sfml-system-d debug sfml-window-d debug sfml-graphics-d
)

target_link_libraries(cw2_cpu 
    optimized sfml-system optimized sfml-window optimized sfml-graphics 
    debug sfml-system-d debug sfml-window-d debug sfml-graphics-d
)

# Post-build commands to copy SFML DLLs
add_custom_command(TARGET cw2_unoptimised POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_SOURCE_DIR}/../contrib/sfml/lib/Debug"
    "$<TARGET_FILE_DIR:cw2_unoptimised>"
)

add_custom_command(TARGET cw2_unoptimised POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_SOURCE_DIR}/../contrib/sfml/lib/Release"
    "$<TARGET_FILE_DIR:cw2_unoptimised>"
)

add_custom_command(TARGET cw2_optimised POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_SOURCE_DIR}/../contrib/sfml/lib/Debug"
    "$<TARGET_FILE_DIR:cw2_optimised>"
)

add_custom_command(TARGET cw2_optimised POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_SOURCE_DIR}/../contrib/sfml/lib/Release"
    "$<TARGET_FILE_DIR:cw2_optimised>"
)

add_custom_command(TARGET cw2_cpu POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_SOURCE_DIR}/../contrib/sfml/lib/Debug"
    "$<TARGET_FILE_DIR:cw2_cpu>"
)

add_custom_command(TARGET cw2_cpu POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_SOURCE_DIR}/../contrib/sfml/lib/Release"
    "$<TARGET_FILE_DIR:cw2_cpu>"
)
