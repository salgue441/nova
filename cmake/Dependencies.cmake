include(FetchContent)

# Function to handle dependencies that might need to be fetched
function(brezel_fetch_dependency NAME VERSION URL TAG)
  string(TOLOWER ${NAME} NAME_LOWER)

  # First try to find the package using vcpkg
  find_package(${NAME} CONFIG QUIET)

  if(${NAME}_FOUND OR TARGET ${NAME}::${NAME})
    message(STATUS "Found ${NAME} via vcpkg or system package")
    return()
  endif()

  # If not found, fetch it
  message(STATUS "Fetching ${NAME} ${VERSION} from ${URL}")

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
  elseif(NAME_LOWER STREQUAL "tbb")
    set(TBB_TEST OFF CACHE BOOL "" FORCE)
    set(TBB_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
  endif()

  FetchContent_MakeAvailable(${NAME_LOWER})
endfunction()

# Verify core dependencies - these should come from vcpkg
function(verify_core_dependencies)
  # Core dependencies
  if(NOT TARGET fmt::fmt)
    message(FATAL_ERROR "fmt library not found. Please make sure vcpkg is properly configured.")
  endif()

  if(NOT TARGET TBB::tbb)
    message(FATAL_ERROR "TBB library not found. Please make sure vcpkg is properly configured.")
  endif()

  if(NOT Boost_FOUND)
    message(FATAL_ERROR "Boost libraries not found. Please make sure vcpkg is properly configured.")
  endif()

  if(NOT TARGET tl::expected)
    message(FATAL_ERROR "tl-expected library not found. Please make sure vcpkg is properly configured.")
  endif()

  # Testing libraries
  if(BUILD_TESTING)
    if(NOT TARGET GTest::gtest)
      message(FATAL_ERROR "GoogleTest not found. Please make sure vcpkg is properly configured with the 'tests' feature.")
    endif()
  endif()

  # Benchmarking libraries
  if(BUILD_BENCHMARKS)
    if(NOT TARGET benchmark::benchmark)
      message(FATAL_ERROR "Google Benchmark not found. Please make sure vcpkg is properly configured with the 'benchmarks' feature.")
    endif()
  endif()

  # CUDA related dependencies
  if(USE_CUDA)
    if(NOT TARGET CUDA::cudart)
      message(FATAL_ERROR "CUDA runtime not found. Please make sure CUDA toolkit is installed.")
    endif()

    # CUTLASS is required for CUDA support
    if(NOT TARGET cutlass::cutlass)
      # Try to fetch if not available through vcpkg
      brezel_fetch_dependency(
        cutlass
        "3.1.0"
        "https://github.com/NVIDIA/cutlass.git"
        "v3.1.0")
    endif()

    # CUB might be included in newer CUDA versions
    if(CUDAToolkit_VERSION VERSION_LESS 11.0)
      if(NOT TARGET cub::cub)
        # Try to fetch if not available through vcpkg
        brezel_fetch_dependency(
          cub
          "2.1.0"
          "https://github.com/NVIDIA/cub.git"
          "2.1.0")
      endif()
    endif()
  endif()

  message(STATUS "All dependencies verified successfully")
endfunction()

# Verify all dependencies
verify_core_dependencies()