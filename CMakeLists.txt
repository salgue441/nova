cmake_minimum_required(VERSION 3.20)

project(BrezelTensor
  VERSION 1.0.0
  DESCRIPTION "Modern C++20 Header-only Tensor Framework with CUDA support"
  HOMEPAGE_URL "https://github.com/salgue441/brezel-framework"
  LANGUAGES CXX)

# Set modern CMake policies
cmake_policy(SET CMP0048 NEW)
cmake_policy(SET CMP0077 NEW)
cmake_policy(SET CMP0079 NEW)

# Add CMake modules path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Include core modules
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(Options)
include(Functions)
include(Dependencies)

# Set global compilation flags
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add the main header-only library
add_subdirectory(include)

# Optional components
if(BUILD_TESTING)
  include(CTest)
  enable_testing()
  add_subdirectory(tests)
endif()

if(BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if(BUILD_DOCS)
  add_subdirectory(docs)
endif()

# Print configuration summary
message(STATUS "\nBrezel Configuration:")
message(STATUS "  Version:          ${PROJECT_VERSION}")
message(STATUS "  Build type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "  Install prefix:   ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  Testing enabled:  ${BUILD_TESTING}")
message(STATUS "  Examples enabled: ${BUILD_EXAMPLES}")
message(STATUS "  Docs enabled:     ${BUILD_DOCS}\n")