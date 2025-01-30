include(FetchContent)

# Function to handle dependencies
function(brezel_fetch_dependency NAME VERSION URL TAG)
  string(TOLOWER ${NAME} NAME_LOWER)

  if(NOT TARGET ${NAME}::${NAME})
    message(STATUS "Fetching ${NAME} ${VERSION}")

    FetchContent_Declare(
      ${NAME_LOWER}
      GIT_REPOSITORY ${URL}
      GIT_TAG ${TAG}
      GIT_SHALLOW TRUE
    )

    # Custom options for specific dependencies
    if(NAME_LOWER STREQUAL "googletest")
      set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
      set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
      set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    elseif(NAME_LOWER STREQUAL "benchmark")
      set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
      set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    elseif(NAME_LOWER STREQUAL "fmt")
      set(FMT_DOC OFF CACHE BOOL "" FORCE)
      set(FMT_TEST OFF CACHE BOOL "" FORCE)
      set(FMT_INSTALL ON CACHE BOOL "" FORCE)
    endif()

    FetchContent_MakeAvailable(${NAME_LOWER})
  endif()
endfunction()

# Required core dependencies
find_package(Threads REQUIRED)
find_package(fmt CONFIG QUIET)

if(NOT fmt_FOUND)
  brezel_fetch_dependency(fmt "9.1.0"
    "https://github.com/fmtlib/fmt.git"
    "9.1.0")
endif()

# Boost dependencies
set(BOOST_MIN_VERSION "1.80.0")
find_package(Boost ${BOOST_MIN_VERSION} COMPONENTS
  container
  system
  filesystem
  QUIET)

if(NOT Boost_FOUND)
  message(STATUS "Boost ${BOOST_MIN_VERSION} not found, fetching from source")
  brezel_fetch_dependency(
    Boost
    ${BOOST_MIN_VERSION}
    "https://github.com/boostorg/boost.git"
    "boost-${BOOST_MIN_VERSION}")
endif()

# CUDA related dependencies (optional)
if(USE_CUDA)
  find_package(CUDAToolkit REQUIRED)

  # CUB for CUDA primitives
  if(NOT TARGET cub::cub)
    brezel_fetch_dependency(
      cub
      "2.1.0"
      "https://github.com/NVIDIA/cub.git"
      "2.1.0")
  endif()

  # CUTLASS for CUDA templates
  if(NOT TARGET cutlass::cutlass)
    brezel_fetch_dependency(
      cutlass
      "3.1.0"
      "https://github.com/NVIDIA/cutlass.git"
      "v3.1.0")
  endif()
endif()

# Testing dependencies
if(BUILD_TESTING)
  find_package(GTest CONFIG QUIET)

  if(NOT GTest_FOUND)
    message(STATUS "GoogleTest not found, fetching from source")
    brezel_fetch_dependency(
      googletest
      "1.14.0"
      "https://github.com/google/googletest.git"
      "v1.14.0")
  endif()
endif()

# Benchmarking dependencies
if(BUILD_BENCHMARKS)
  find_package(benchmark CONFIG QUIET)

  if(NOT benchmark_FOUND)
    message(STATUS "Google Benchmark not found, fetching from source")
    brezel_fetch_dependency(
      benchmark
      "1.8.3"
      "https://github.com/google/benchmark.git"
      "v1.8.3")
  endif()
endif()

# Math libraries (optional)
if(USE_BLAS)
  find_package(BLAS REQUIRED)
  find_package(LAPACK REQUIRED)
endif()

if(USE_MKL)
  find_package(MKL REQUIRED)
endif()

# OpenMP support (optional)
if(USE_OPENMP)
  find_package(OpenMP REQUIRED)
endif()

# Documentation dependencies
if(BUILD_DOCS)
  find_package(Doxygen REQUIRED)
  find_package(Sphinx QUIET)

  if(NOT Sphinx_FOUND)
    find_program(SPHINX_EXECUTABLE
      NAMES sphinx-build
      DOC "Sphinx documentation generator")

    if(NOT SPHINX_EXECUTABLE)
      message(WARNING "Sphinx not found. Documentation will be limited to Doxygen only.")
    endif()
  endif()
endif()

# Optional development tools
find_program(CCACHE_PROGRAM ccache)

if(CCACHE_PROGRAM)
  set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")

  if(USE_CUDA)
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
  endif()
endif()

# Package validation
function(brezel_validate_dependencies)
  # Core dependencies check
  if(NOT TARGET fmt::fmt)
    message(FATAL_ERROR "fmt library not found or failed to build")
  endif()

  if(NOT Boost_FOUND)
    message(FATAL_ERROR "Boost libraries not found or failed to build")
  endif()

  # CUDA checks
  if(USE_CUDA)
    if(NOT TARGET CUDA::cudart)
      message(FATAL_ERROR "CUDA runtime not found")
    endif()

    if(NOT TARGET cub::cub)
      message(FATAL_ERROR "NVIDIA CUB not found or failed to build")
    endif()

    if(NOT TARGET cutlass::cutlass)
      message(FATAL_ERROR "NVIDIA CUTLASS not found or failed to build")
    endif()
  endif()

  # Testing checks
  if(BUILD_TESTING AND NOT TARGET GTest::gtest)
    message(FATAL_ERROR "GoogleTest not found or failed to build")
  endif()

  # Benchmark checks
  if(BUILD_BENCHMARKS AND NOT TARGET benchmark::benchmark)
    message(FATAL_ERROR "Google Benchmark not found or failed to build")
  endif()

  # Math library checks
  if(USE_BLAS AND NOT BLAS_FOUND)
    message(FATAL_ERROR "BLAS library requested but not found")
  endif()

  if(USE_MKL AND NOT MKL_FOUND)
    message(FATAL_ERROR "Intel MKL requested but not found")
  endif()

  message(STATUS "All dependencies validated successfully")
endfunction()

# Validate all dependencies at the end
brezel_validate_dependencies()