# Detect compiler
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  set(BREZEL_COMPILER_GNU TRUE)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(BREZEL_COMPILER_CLANG TRUE)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  set(BREZEL_COMPILER_MSVC TRUE)
endif()

# Helper function to add compiler flags
function(brezel_add_compiler_flags)
  foreach(flag ${ARGN})
    string(REPLACE "+" "x" flag_var ${flag})
    check_cxx_compiler_flag(${flag} COMPILER_SUPPORTS${flag_var})

    if(COMPILER_SUPPORTS${flag_var})
      add_compile_options(${flag})
    endif()
  endforeach()
endfunction()

# Common flags for all compilers
add_compile_options(
  $<$<CONFIG:Debug>:-DBREZEL_DEBUG>
  $<$<CONFIG:Release>:-DBREZEL_RELEASE>
)

# Compiler-specific flags
if(BREZEL_COMPILER_GNU OR BREZEL_COMPILER_CLANG)
  # Warning flags
  brezel_add_compiler_flags(
    -Wall
    -Wextra
    -Wpedantic
    -Wconversion
    -Wsign-conversion
    -Wcast-align
    -Wformat=2
    -Wunused
    -Wnull-dereference
    -Wno-unused-function
  )

  # CPU specific optimizations
  if(USE_AVX)
    brezel_add_compiler_flags(-mavx -mavx2)
  endif()

  # Optimization flags
  if(CMAKE_BUILD_TYPE STREQUAL "Release")
    brezel_add_compiler_flags(-O3 -DNDEBUG)

    if(ENABLE_LTO)
      set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()
  endif()

  # Debug flags
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    brezel_add_compiler_flags(-O0 -g3 -fno-omit-frame-pointer)
  endif()

elseif(BREZEL_COMPILER_MSVC)
  # Warning flags
  add_compile_options(
    /W4
    /WX
    /permissive-
    /Zc:__cplusplus
    /volatile:iso
  )

  # Optimization flags
  if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(
      /O2
      /Oi
      /Ot
      /GL
      /DNDEBUG
    )

    if(ENABLE_LTO)
      set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()
  endif()

  # Debug flags
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(/Od /Zi /JMC)
  endif()
endif()

# CUDA specific flags
if(USE_CUDA)
  if(BREZEL_COMPILER_GNU OR BREZEL_COMPILER_CLANG)
    list(APPEND CMAKE_CUDA_FLAGS "-std=c++17")

    if(CMAKE_BUILD_TYPE STREQUAL "Release")
      list(APPEND CMAKE_CUDA_FLAGS "-O3")
    endif()

    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
      list(APPEND CMAKE_CUDA_FLAGS "-G")
    endif()
  endif()
endif()

# Sanitizer flags
if(ENABLE_SANITIZER)
  if(BREZEL_COMPILER_GNU OR BREZEL_COMPILER_CLANG)
    add_compile_options(-fsanitize=address -fsanitize=undefined)
    add_link_options(-fsanitize=address -fsanitize=undefined)
  endif()
endif()

# Coverage flags
if(ENABLE_COVERAGE)
  if(BREZEL_COMPILER_GNU OR BREZEL_COMPILER_CLANG)
    add_compile_options(--coverage)
    add_link_options(--coverage)
  endif()
endif()

# Apply flags to all targets
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "C++ compiler flags" FORCE)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}" CACHE STRING "CUDA compiler flags" FORCE)