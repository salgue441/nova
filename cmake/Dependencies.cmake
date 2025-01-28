include(FetchContent)

# Common function for handling dependencies
function(nova_handle_dependency NAME VERSION REPOSITORY TAG)
  string(TOLOWER ${NAME} NAME_LOWER)

  if(NOT TARGET ${NAME}::${NAME} AND NOT TARGET ${NAME})
    message(STATUS "Fetching ${NAME} ${VERSION} from ${REPOSITORY}")
    FetchContent_Declare(
      ${NAME_LOWER}
      GIT_REPOSITORY ${REPOSITORY}
      GIT_TAG ${TAG}
    )
    FetchContent_MakeAvailable(${NAME_LOWER})
  endif()
endfunction()

# Required dependencies
find_package(Threads REQUIRED)
find_package(fmt REQUIRED)

# Optional dependencies based on build options
if(USE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

# Development dependencies
if(BUILD_TESTING)
  # Google Test
  nova_handle_dependency(
    GTest
    "1.12.1"
    "https://github.com/google/googletest.git"
    "release-1.12.1"
  )
endif()

if(BUILD_BENCHMARKS)
  # Google Benchmark
  nova_handle_dependency(
    benchmark
    "1.8.0"
    "https://github.com/google/benchmark.git"
    "v1.8.0"
  )
endif()
