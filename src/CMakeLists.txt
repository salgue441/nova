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
)

# Link dependencies
target_link_libraries(brezel_core
  INTERFACE
  fmt::fmt
  Threads::Threads
  $<$<BOOL:${USE_CUDA}>:CUDA::cudart>
)

# Optional dependencies
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
endif()

if(USE_OPENMP)
  target_link_libraries(brezel_core
    INTERFACE
    OpenMP::OpenMP_CXX
  )
endif()

# Export the target (without installing it here)
export(TARGETS brezel_core
  NAMESPACE Brezel::
  FILE "${CMAKE_CURRENT_BINARY_DIR}/BrezelTargets.cmake"
)