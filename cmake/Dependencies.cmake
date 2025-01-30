include(FetchContent)

# Function to handle dependencies
function(brezel_handle_dependency NAME VERSION REPO TAG)
  FetchContent_Declare(
    ${NAME}
    GIT_REPOSITORY ${REPO}
    GIT_TAG ${TAG}
  )

  FetchContent_MakeAvailable(${NAME})
endfunction()

# Boost (build from source if not found)
set(BOOST_MIN_VERSION "1.80.0")
find_package(Boost ${BOOST_MIN_VERSION} COMPONENTS container QUIET)

if(NOT Boost_FOUND)
  message(STATUS "Boost not found, fetching from source")
  brezel_handle_dependency(boost ${BOOST_MIN_VERSION} "https://github.com/boostorg/boost.git" "boost-${BOOST_MIN_VERSION}")
endif()

# Required dependencies
find_package(Threads REQUIRED)
find_package(fmt CONFIG REQUIRED)

# CUDA (condionally enabled)
if(USE_CUDA)
  enable_language(CUDA)
  find_package(CudaToolkit REQUIRED)
endif()

# Testing dependencies
if(BUILD_TESTING)
  find_package(GTest CONFIG QUIET)

  if(NOT GTest_FOUND)
    message(STATUS "Fetching GoogleTest from source...")
    brezel_handle_dependency(googletest "1.14.0" "https://github.com/google/googletest.git" "v1.14.0")
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  endif()
endif()