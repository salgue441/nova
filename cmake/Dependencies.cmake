include(FetchContent)

# Required core dependencies
find_package(Threads REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(TBB CONFIG REQUIRED)
find_package(tl-expected CONFIG REQUIRED)

if(NOT TARGET TBB::tbb)
  message(FATAL_ERROR "TBB targets not available. Please install TBB through vcpkg: vcpkg install tbb:${VCPKG_TARGET_TRIPLET}")
endif()

# Boost dependencies
set(BOOST_MIN_VERSION "1.86.0")
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
    message(FATAL_ERROR "fmt library not found. Install via: vcpkg install fmt:${VCPKG_TARGET_TRIPLET}")
  endif()

  if(NOT TARGET TBB::tbb)
    message(FATAL_ERROR "TBB library not found. Install via: vcpkg install tbb:${VCPKG_TARGET_TRIPLET}")
  endif()

  if(NOT Boost_FOUND)
    message(FATAL_ERROR "Boost libraries not found. Install via: vcpkg install boost-container boost-system boost-filesystem:${VCPKG_TARGET_TRIPLET}")
  endif()

  # Testing checks
  if(BUILD_TESTING AND NOT TARGET GTest::gtest)
    message(FATAL_ERROR "GoogleTest not found. Install via: vcpkg install gtest:${VCPKG_TARGET_TRIPLET}")
  endif()

  message(STATUS "All dependencies validated successfully")
endfunction()