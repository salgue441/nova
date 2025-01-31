# Core library
add_library(brezel_core INTERFACE)
add_library(Brezel::Core ALIAS brezel_core)

target_include_directories(brezel_core
  INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_features(brezel_core
  INTERFACE
  cxx_std_20
)

# Add compile definitions based on configuration
target_compile_definitions(brezel_core
  INTERFACE
  $<$<BOOL:${USE_CUDA}>:BREZEL_WITH_CUDA>
  $<$<CONFIG:Debug>:BREZEL_DEBUG>
  BREZEL_VERSION="${PROJECT_VERSION}"
  BREZEL_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
  BREZEL_VERSION_MINOR=${PROJECT_VERSION_MINOR}
  BREZEL_VERSION_PATCH=${PROJECT_VERSION_PATCH}
)

# Core dependencies
target_link_libraries(brezel_core
  INTERFACE
  fmt::fmt
  Threads::Threads
  TBB::tbb
  Boost::container
  Boost::system
  Boost::filesystem
)

# CUDA dependencies
if(USE_CUDA)
  target_link_libraries(brezel_core
    INTERFACE
    CUDA::cudart
    cub::cub
    cutlass::cutlass
  )

  # Set CUDA architectures if not already set
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set_property(TARGET brezel_core PROPERTY CUDA_ARCHITECTURES native)
  endif()
endif()

# Math libraries
if(USE_BLAS)
  target_link_libraries(brezel_core
    INTERFACE
    BLAS::BLAS
    LAPACK::LAPACK
  )
endif()

if(USE_MKL)
  target_link_libraries(brezel_core
    INTERFACE
    MKL::MKL
  )

  target_compile_definitions(brezel_core
    INTERFACE
    BREZEL_USE_MKL
  )
endif()

# Parallel processing
if(USE_OPENMP)
  find_package(OpenMP REQUIRED)
  target_link_libraries(brezel_core
    INTERFACE
    OpenMP::OpenMP_CXX
  )

  target_compile_definitions(brezel_core
    INTERFACE
    BREZEL_USE_OPENMP
  )
endif()

# Optional AVX/SSE support
include(CheckCXXCompilerFlag)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|Intel")
  set(SIMD_FLAGS
    -mavx2
    -mfma
    -mavx
    -msse4.2
    -msse4.1
    -mssse3
    -msse3
    -msse2
    -msse
  )

  foreach(FLAG ${SIMD_FLAGS})
    string(REPLACE "-m" "" FLAG_NAME ${FLAG})
    string(REPLACE "." "_" FLAG_NAME ${FLAG_NAME})
    string(TOUPPER ${FLAG_NAME} FLAG_NAME)

    check_cxx_compiler_flag(${FLAG} COMPILER_SUPPORTS_${FLAG_NAME})

    if(COMPILER_SUPPORTS_${FLAG_NAME})
      target_compile_options(brezel_core INTERFACE $<$<COMPILE_LANGUAGE:CXX>:${FLAG}>)
      target_compile_definitions(brezel_core INTERFACE BREZEL_HAS_${FLAG_NAME})
    endif()
  endforeach()
endif()

# Export the target
export(TARGETS brezel_core
  NAMESPACE Brezel::
  FILE "${CMAKE_CURRENT_BINARY_DIR}/BrezelTargets.cmake"
)

# Generate and install version header
configure_file(
  "${PROJECT_SOURCE_DIR}/include/brezel/core/version.hpp.in"
  "${PROJECT_BINARY_DIR}/include/brezel/core/version.hpp"
  @ONLY
)

target_include_directories(brezel_core
  INTERFACE
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
)