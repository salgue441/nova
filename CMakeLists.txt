cmake_minimum_required(VERSION 3.20)

# First check if CUDA is available
include(CheckLanguage)
check_language(CUDA)

# Initialize project with conditionally enabled CUDA
if(CMAKE_CUDA_COMPILER)
  project(Nova VERSION 0.1.0 LANGUAGES CXX CUDA)
else()
  project(Nova VERSION 0.1.0 LANGUAGES CXX)
endif()

# Add cmake modules path
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(Dependencies)
include(Functions)

# Build options
option(BUILD_TESTING "Build tests" ON)
option(BUILD_EXAMPLES "Build examples" ON)
option(USE_CUDA "Enable CUDA support" OFF)

# Global settings
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Validate CUDA option
if(USE_CUDA AND NOT CMAKE_CUDA_COMPILER)
  message(WARNING "CUDA support requested but no CUDA compiler found. Disabling CUDA support.")
  set(USE_CUDA OFF)
endif()

# Dependencies
find_package(Threads REQUIRED)

if(USE_CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

if(BUILD_TESTING)
  find_package(GTest REQUIRED)
endif()

# Components
add_subdirectory(src)

if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()

# if(BUILD_EXAMPLES)
# add_subdirectory(examples)
# endif()

# Print configuration summary
message(STATUS "")
message(STATUS "Nova Configuration:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA support: ${USE_CUDA}")
message(STATUS "  Testing enabled: ${BUILD_TESTING}")
message(STATUS "  Examples enabled: ${BUILD_EXAMPLES}")
message(STATUS "")