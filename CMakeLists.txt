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

# Setup vcpkg early if available
if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
    CACHE STRING "Vcpkg toolchain file")
endif()

# Include core modules
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(Options)
include(Functions)

# Set global compilation flags
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find required packages provided by vcpkg
find_package(fmt CONFIG REQUIRED)
find_package(TBB CONFIG REQUIRED)
find_package(Boost 1.80.0 COMPONENTS container system filesystem REQUIRED)
find_package(tl-expected CONFIG REQUIRED)
find_package(Threads REQUIRED)

# CUDA setup and validation if requested
if(USE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)

  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES native)
    add_definitions(-DBREZEL_WITH_CUDA)

    # Find CUDA-specific dependencies
    find_package(CUDAToolkit REQUIRED)

    # CUB for CUDA primitives (might be bundled with CUDA 11.0+)
    if(CUDAToolkit_VERSION VERSION_LESS 11.0)
      find_package(CUB CONFIG REQUIRED)
    endif()

    # CUTLASS for CUDA templates
    find_package(CUTLASS CONFIG REQUIRED)
  else()
    message(FATAL_ERROR "CUDA requested but no CUDA compiler found")
  endif()
endif()

# OpenMP support if requested
if(USE_OPENMP)
  find_package(OpenMP)

  if(OpenMP_CXX_FOUND)
    add_compile_options(${OpenMP_CXX_FLAGS})
    link_libraries(${OpenMP_CXX_LIBRARIES})
  else()
    message(WARNING "OpenMP requested but not found. Disabling.")
    set(USE_OPENMP OFF)
  endif()
endif()

# LTO support if requested
if(ENABLE_LTO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT LTO_SUPPORTED)

  if(LTO_SUPPORTED)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
  else()
    message(WARNING "LTO requested but not supported. Disabling.")
    set(ENABLE_LTO OFF)
  endif()
endif()

# Add compiler flags based on compiler type
include(CompilerFlags)

# Add the main header-only library
add_subdirectory(include)

# Optional components
if(BUILD_TESTING)
  include(CTest)
  enable_testing()
  find_package(GTest CONFIG REQUIRED)
  add_subdirectory(tests)
endif()

if(BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if(BUILD_BENCHMARKS)
  find_package(benchmark CONFIG REQUIRED)
  add_subdirectory(benchmarks)
endif()

if(BUILD_DOCS)
  find_package(Doxygen REQUIRED)
  add_subdirectory(docs)
endif()

# Installation configuration
set(BREZEL_INSTALL_CONFIGDIR "${CMAKE_INSTALL_LIBDIR}/cmake/BrezelTensor")

# Install headers
install(
  DIRECTORY ${PROJECT_SOURCE_DIR}/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.hpp"
)

# Generate and install version file
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/BrezelTensorConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

# Configure and install package config
configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/BrezelTensorConfig.cmake"
  INSTALL_DESTINATION ${BREZEL_INSTALL_CONFIGDIR}
)

# Install CMake files
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/BrezelTensorConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/BrezelTensorConfigVersion.cmake"
  DESTINATION ${BREZEL_INSTALL_CONFIGDIR}
)

# Export targets
install(
  TARGETS brezel
  EXPORT BrezelTargets
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
  EXPORT BrezelTargets
  FILE BrezelTargets.cmake
  NAMESPACE Brezel::
  DESTINATION ${BREZEL_INSTALL_CONFIGDIR}
)

# Package configuration
include(CPack)
set(CPACK_PACKAGE_VENDOR "BrezelFramework")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "${PROJECT_DESCRIPTION}")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")

# Print configuration summary
message(STATUS "\nBrezelTensor Configuration:")
message(STATUS "  Version:          ${PROJECT_VERSION}")
message(STATUS "  Build type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "  Install prefix:   ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  CUDA support:     ${USE_CUDA}")
message(STATUS "  OpenMP support:   ${USE_OPENMP}")
message(STATUS "  Testing enabled:  ${BUILD_TESTING}")
message(STATUS "  Examples enabled: ${BUILD_EXAMPLES}")
message(STATUS "  Docs enabled:     ${BUILD_DOCS}")
message(STATUS "  LTO enabled:      ${ENABLE_LTO}\n")