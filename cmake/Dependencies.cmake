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
    elseif(NAME_LOWER STREQUAL "tbb")
      set(TBB_TEST OFF CACHE BOOL "" FORCE)
      set(TBB_EXAMPLES OFF CACHE BOOL "" FORCE)
      set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
    endif()

    FetchContent_MakeAvailable(${NAME_LOWER})
  endif()
endfunction()

# Required core dependencies
find_package(Threads REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(TBB CONFIG REQUIRED)
find_package(tl-expected CONFIG REQUIRED)

if(NOT TARGET TBB::tbb)
  message(FATAL_ERROR "TBB targets not available. Please install TBB through vcpkg: vcpkg install tbb:${VCPKG_TARGET_TRIPLET}")
endif()

# Boost dependencies
set(BOOST_MIN_VERSION "1.80.0")
find_package(Boost ${BOOST_MIN_VERSION} COMPONENTS
  container
  system
  filesystem
  REQUIRED)

# CUDA related dependencies (optional)
if(USE_CUDA)
  find_package(CUDAToolkit REQUIRED)

  # CUB for CUDA primitives
  if(NOT TARGET cub::cub)
    find_package(CUB CONFIG QUIET)

    if(NOT CUB_FOUND)
      brezel_fetch_dependency(
        cub
        "2.1.0"
        "https://github.com/NVIDIA/cub.git"
        "2.1.0")
    endif()
  endif()

  # CUTLASS for CUDA templates
  if(NOT TARGET cutlass::cutlass)
    find_package(CUTLASS CONFIG QUIET)

    if(NOT CUTLASS_FOUND)
      brezel_fetch_dependency(
        cutlass
        "3.1.0"
        "https://github.com/NVIDIA/cutlass.git"
        "v3.1.0")
    endif()
  endif()
endif()

# Testing dependencies
if(BUILD_TESTING)
  find_package(GTest CONFIG REQUIRED)
endif()

# Package validation
function(brezel_validate_dependencies)
  # Core dependencies check
  if(NOT TARGET fmt::fmt)
    message(FATAL_ERROR "fmt library not found. Install via: vcpkg install fmt:${VCPKG_TARGET_TRIPLET}")
  endif()

  if(NOT TARGET TBB::tbb)
    message(FATAL_ERROR "TBB library not found. Install via: vcpkg install tbb:${VCPKG_TARGET_TRIPLET}")
  endif()

  if(NOT Boost_FOUND)
    message(FATAL_ERROR "Boost libraries not found. Install via: vcpkg install boost-container boost-system boost-filesystem:${VCPKG_TARGET_TRIPLET}")
  endif()

  # CUDA checks
  if(USE_CUDA)
    if(NOT TARGET CUDA::cudart)
      message(FATAL_ERROR "CUDA runtime not found")
    endif()

    if(NOT TARGET cub::cub)
      message(FATAL_ERROR "NVIDIA CUB not found")
    endif()

    if(NOT TARGET cutlass::cutlass)
      message(FATAL_ERROR "NVIDIA CUTLASS not found")
    endif()
  endif()

  # Testing checks
  if(BUILD_TESTING AND NOT TARGET GTest::gtest)
    message(FATAL_ERROR "GoogleTest not found. Install via: vcpkg install gtest:${VCPKG_TARGET_TRIPLET}")
  endif()

  message(STATUS "All dependencies validated successfully")
endfunction()

# Validate all dependencies at the end
brezel_validate_dependencies()