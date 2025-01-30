cmake_minimum_required(VERSION 3.20)

# Setup vcpkg if available
if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
    CACHE STRING "Vcpkg toolchain file")
endif()

# Check for CUDA availability
include(CheckLanguage)
check_language(CUDA)

# Project initialization (CUDA conditional)
if(CMAKE_CUDA_COMPILER)
  project(Brezel VERSION 0.1.0 LANGUAGES CXX CUDA)
else()
  project(Brezel VERSION 0.1.0 LANGUAGES CXX)
endif()

# Cmake modules
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(Options)
include(Functions)
include(Dependencies)

# Validate CUDA Settings
if(USE_CUDA AND NOT CMAKE_CUDA_COMPILER)
  message(WARNING "CUDA support requested but CUDA compiler not found. Disabling CUDA support.")
  set(USE_CUDA OFF)
endif()

# Components
add_subdirectory(src)

if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()

# Configuration summary
message(STATUS "\nNova Configuration:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA support: ${USE_CUDA}")
message(STATUS "  Testing enabled: ${BUILD_TESTING}")
message(STATUS "  Examples enabled: ${BUILD_EXAMPLES}")
message(STATUS "  Using vcpkg toolchain: ${CMAKE_TOOLCHAIN_FILE}\n")