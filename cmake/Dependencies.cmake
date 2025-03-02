include(FetchContent)

# tl-expected
FetchContent_Declare(
  tl-expected
  GIT_REPOSITORY https://github.com/TartanLlama/expected.git
  GIT_TAG v1.1.0
)

FetchContent_MakeAvailable(tl-expected)

# Required core dependencies
find_package(Threads REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(TBB CONFIG REQUIRED)
find_package(Eigen3 3.3 REQUIRED)

if(NOT TARGET TBB::tbb)
  message(FATAL_ERROR "TBB targets not available. Please install TBB system-wide.")
endif()

# Boost dependencies
set(BOOST_MIN_VERSION "1.83.0")
find_package(Boost ${BOOST_MIN_VERSION} COMPONENTS
  container
  system
  filesystem
  stacktrace_backtrace
  REQUIRED)

if(Boost_FOUND)
  message(STATUS "Boost version: ${Boost_VERSION}")
  message(STATUS "Boost includes: ${Boost_INCLUDE_DIRS}")
  message(STATUS "Boost libraries: ${Boost_LIBRARIES}")
endif()

# Testing dependencies
if(BUILD_TESTING)
  find_package(GTest CONFIG REQUIRED)
endif()

# Package validation
function(brezel_validate_dependencies)
  # Core dependencies check
  if(NOT TARGET fmt::fmt)
    message(FATAL_ERROR "fmt library not found. Please install fmt system-wide.")
  endif()

  if(NOT TARGET TBB::tbb)
    message(FATAL_ERROR "TBB library not found. Please install TBB system-wide.")
  endif()

  if(NOT Boost_FOUND)
    message(FATAL_ERROR "Boost libraries not found. Please install Boost system-wide.")
  endif()

  # Testing checks
  if(BUILD_TESTING AND NOT TARGET GTest::gtest)
    message(FATAL_ERROR "GoogleTest not found. Please install GTest system-wide.")
  endif()

  if(NOT Eigen3_FOUND)
    message(FATAL_ERROR "Eigen3 library not found. Please install libeigen3-dev")
  endif()

  message(STATUS "All dependencies validated successfully")
endfunction()