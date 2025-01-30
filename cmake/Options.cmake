# Build options with descriptions
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_TESTING "Build test suite" ON)
option(BUILD_EXAMPLES "Build example programs" ON)
option(BUILD_BENCHMARKS "Build benchmarking suite" OFF)
option(BUILD_DOCS "Build documentation" OFF)
option(USE_CUDA "Enable CUDA support" OFF)
option(USE_BLAS "Enable BLAS support" OFF)
option(USE_MKL "Enable Intel MKL support" OFF)
option(USE_OPENMP "Enable OpenMP support" ON)
option(USE_AVX "Enable AVX instructions" ON)
option(ENABLE_LTO "Enable Link Time Optimization" OFF)
option(ENABLE_SANITIZER "Enable sanitizer instrumentation" OFF)
option(ENABLE_COVERAGE "Enable coverage instrumentation" OFF)

# Advanced options
option(ENABLE_PROFILING "Enable profiling instrumentation" OFF)
option(ENABLE_LOGGING "Enable debug logging" OFF)
mark_as_advanced(ENABLE_PROFILING ENABLE_LOGGING)

# Validation of mutually exclusive options
if(USE_MKL AND USE_BLAS)
  message(FATAL_ERROR "Cannot enable both MKL and BLAS. Choose one.")
endif()

# Feature requirements
include(CheckCXXCompilerFlag)

# Check for OpenMP support
if(USE_OPENMP)
  find_package(OpenMP)

  if(NOT OpenMP_CXX_FOUND)
    message(WARNING "OpenMP requested but not found. Disabling.")
    set(USE_OPENMP OFF)
  endif()
endif()

# Check for AVX support
if(USE_AVX)
  check_cxx_compiler_flag("-mavx" COMPILER_SUPPORTS_AVX)

  if(NOT COMPILER_SUPPORTS_AVX)
    message(WARNING "AVX instructions requested but not supported. Disabling.")
    set(USE_AVX OFF)
  endif()
endif()

# Sanitizer configuration
if(ENABLE_SANITIZER)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize=undefined")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address -fsanitize=undefined")
  else()
    message(WARNING "Sanitizer support not available for current compiler. Disabling.")
    set(ENABLE_SANITIZER OFF)
  endif()
endif()

# LTO configuration
if(ENABLE_LTO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT LTO_SUPPORTED)

  if(NOT LTO_SUPPORTED)
    message(WARNING "LTO requested but not supported. Disabling.")
    set(ENABLE_LTO OFF)
  endif()
endif()

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()