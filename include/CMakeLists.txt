add_library(brezel INTERFACE)
add_library(Brezel::brezel ALIAS brezel)

target_include_directories(brezel
  INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_features(brezel
  INTERFACE
  cxx_std_20
)

# Add compile definitions based on configuration
target_compile_definitions(brezel
  INTERFACE
  $<$<BOOL:${USE_CUDA}>:BREZEL_WITH_CUDA>
  $<$<CONFIG:Debug>:BREZEL_DEBUG>
  BREZEL_VERSION="${PROJECT_VERSION}"
  BREZEL_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
  BREZEL_VERSION_MINOR=${PROJECT_VERSION_MINOR}
  BREZEL_VERSION_PATCH=${PROJECT_VERSION_PATCH}
)

# Core dependencies
target_link_libraries(brezel
  INTERFACE
  fmt::fmt
  Threads::Threads
  TBB::tbb
  TBB::tbbmalloc
  Boost::container
  Boost::system
  Boost::filesystem
  tl::expected
)

# CUDA dependencies
if(USE_CUDA)
  target_link_libraries(brezel
    INTERFACE
    CUDA::cudart
    cub::cub
    cutlass::cutlass
  )
endif()

# Math libraries
if(USE_BLAS)
  target_link_libraries(brezel
    INTERFACE
    BLAS::BLAS
    LAPACK::LAPACK
  )
endif()

if(USE_MKL)
  target_link_libraries(brezel
    INTERFACE
    MKL::MKL
  )
endif()

if(USE_OPENMP)
  find_package(OpenMP REQUIRED)
  target_link_libraries(brezel
    INTERFACE
    OpenMP::OpenMP_CXX
  )
endif()

# SIMD support
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|Intel")
  foreach(FLAG ${SIMD_FLAGS})
    string(REPLACE "-m" "" FLAG_NAME ${FLAG})
    string(REPLACE "." "_" FLAG_NAME ${FLAG_NAME})
    string(TOUPPER ${FLAG_NAME} FLAG_NAME)

    check_cxx_compiler_flag(${FLAG} COMPILER_SUPPORTS_${FLAG_NAME})

    if(COMPILER_SUPPORTS_${FLAG_NAME})
      target_compile_options(brezel INTERFACE $<$<COMPILE_LANGUAGE:CXX>:${FLAG}>)
    endif()
  endforeach()
endif()