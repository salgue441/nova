add_library(brezel_core INTERFACE)

target_include_directories(brezel_core INTERFACE ${CMAKE_SOURCE_DIR}/include)

target_compile_definitions(brezel_core INTERFACE
  $<$<BOOL:${USE_CUDA}>:NOVA_WITH_CUDA>)

target_link_libraries(brezel_core INTERFACE
  fmt::fmt $<$<BOOL:${USE_CUDA}>:CUDA::cudart>)
