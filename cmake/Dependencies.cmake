# cmake/Dependencies.cmake
include(FetchContent)

# Function to handle external dependencies
function(nova_handle_dependency NAME VERSION REPO TAG)
  FetchContent_Declare(
    ${NAME}
    GIT_REPOSITORY ${REPO}
    GIT_TAG ${TAG}
  )
  FetchContent_MakeAvailable(${NAME})
endfunction()

# Boost
set(BOOST_MIN_VERSION "1.80.0")
find_package(Boost ${BOOST_MIN_VERSION} COMPONENTS container)
if(NOT Boost_FOUND)
  message(STATUS "System Boost not found, building from source")
  set(BOOST_INCLUDE_LIBRARIES container)
  set(BOOST_ENABLE_CMAKE ON)
  
  nova_handle_dependency(
    boost
    ${BOOST_MIN_VERSION}
    "https://github.com/boostorg/boost.git"
    "boost-${BOOST_MIN_VERSION}"
  )
endif()

# Development dependencies
if(BUILD_TESTING)
  # Try to find system GTest first
  find_package(GTest QUIET)

  if(NOT GTest_FOUND)
    message(STATUS "System GTest not found, building from source")
    nova_handle_dependency(
      googletest
      "1.14.0"
      "https://github.com/google/googletest.git"
      "v1.14.0"
    )

    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

    # Create aliases if they don't exist
    if(NOT TARGET GTest::gtest)
      add_library(GTest::gtest ALIAS gtest)
      add_library(GTest::gtest_main ALIAS gtest_main)
      add_library(GTest::gmock ALIAS gmock)
      add_library(GTest::gmock_main ALIAS gmock_main)
    endif()
  endif()
endif()

# Required dependencies
find_package(Threads REQUIRED)
find_package(fmt REQUIRED)

# Optional dependencies based on build options
if(USE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

if(BUILD_BENCHMARKS)
  nova_handle_dependency(
    benchmark
    "1.8.0"
    "https://github.com/google/benchmark.git"
    "v1.8.0"
  )
endif()