cmake_minimum_required(VERSION 3.11.3)

project(SPH LANGUAGES CUDA CXX)
set(CMAKE_CUDA_STANDARD 17)


# Enable CUDA language
enable_language(CUDA)

# Set CUDA architectures and flags
set(CUDA_ARCHITECTURES "75")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}")

# Enable separable compilation
set(CUDA_SEPARABLE_COMPILATION ON)

# Find CUDA package
find_package(CUDA QUIET REQUIRED)

# Source files
file(GLOB project_SRCS 
    src/*.cpp 
    src/Benchmarks/*.cpp 
    src/Benchmarks/*.cu 
    src/*.cu
)

# Include and link directories for CUDA
include_directories(/usr/local/cuda/targets/x86_64-linux/include)
link_directories(/usr/local/cuda/targets/x86_64-linux/lib)

# Define the executable
add_executable(SPH ${project_SRCS})

# Set properties and compile options for the target
set_property(TARGET SPH PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(SPH PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCHITECTURES}")

target_compile_options(SPH PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:
        --generate-line-info
        --use_fast_math
        --relocatable-device-code=true
    >
)
