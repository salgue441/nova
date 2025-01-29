add_library(nova_core INTERFACE)

# Add include directories to the library
target_include_directories(nova_core
  INTERFACE
  ${CMAKE_SOURCE_DIR}/include
)

# Add only compile definitions if needed
target_compile_definitions(nova_core
  INTERFACE
  $<$<BOOL:${USE_CUDA}>:NOVA_WITH_CUDA>
)

# Link required libraries
target_link_libraries(nova_core
  INTERFACE
  fmt::fmt
  $<$<BOOL:${USE_CUDA}>:CUDA::cudart>
)