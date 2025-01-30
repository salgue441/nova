# Add a Brezel library target with modern CMake practices
function(brezel_add_library target_name)
  cmake_parse_arguments(PARSE_ARGV 1 ARG
    "CUDA;SHARED;STATIC;INTERFACE"
    "VERSION;SOVERSION"
    "SOURCES;PUBLIC_HEADERS;PRIVATE_HEADERS;DEPENDS;PRIVATE_DEPENDS"
  )

  # Validate arguments
  if(ARG_SHARED AND ARG_STATIC)
    message(FATAL_ERROR "Cannot specify both SHARED and STATIC for ${target_name}")
  endif()

  # Determine library type
  if(ARG_INTERFACE)
    add_library(${target_name} INTERFACE)
  elseif(ARG_SHARED)
    add_library(${target_name} SHARED ${ARG_SOURCES})
  elseif(ARG_STATIC)
    add_library(${target_name} STATIC ${ARG_SOURCES})
  else()
    add_library(${target_name} ${ARG_SOURCES})
  endif()

  add_library(BrezelTensor::${target_name} ALIAS ${target_name})

  # Set target properties
  if(NOT ARG_INTERFACE)
    if(ARG_VERSION)
      set_target_properties(${target_name} PROPERTIES VERSION ${ARG_VERSION})
    endif()

    if(ARG_SOVERSION)
      set_target_properties(${target_name} PROPERTIES SOVERSION ${ARG_SOVERSION})
    endif()

    target_sources(${target_name}
      PRIVATE
      ${ARG_SOURCES}
      ${ARG_PRIVATE_HEADERS}
      PUBLIC
      ${ARG_PUBLIC_HEADERS}
    )
  endif()

  # Include directories
  target_include_directories(${target_name}
    ${ARG_INTERFACE_KEYWORD}
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
  )

  # Dependencies
  if(ARG_DEPENDS)
    target_link_libraries(${target_name}
      ${ARG_INTERFACE_KEYWORD} ${ARG_DEPENDS})
  endif()

  if(ARG_PRIVATE_DEPENDS)
    target_link_libraries(${target_name} PRIVATE ${ARG_PRIVATE_DEPENDS})
  endif()

  # CUDA configuration
  if(ARG_CUDA AND USE_CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
      message(FATAL_ERROR "CUDA compiler required for target ${target_name} but not found")
    endif()

    set_target_properties(${target_name} PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON
    )
  endif()

  # Enable compiler warnings
  if(NOT ARG_INTERFACE)
    target_compile_options(${target_name} PRIVATE
      $<$<CXX_COMPILER_ID:MSVC>:/W4>
      $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>
    )
  endif()
endfunction()

# Add a Brezel test target
function(brezel_add_test test_name)
  cmake_parse_arguments(PARSE_ARGV 1 ARG
    "CUDA"
    ""
    "SOURCES;DEPENDS"
  )

  add_executable(${test_name} ${ARG_SOURCES})

  target_link_libraries(${test_name}
    PRIVATE
    ${ARG_DEPENDS}
    GTest::gtest
    GTest::gtest_main
    Threads::Threads
  )

  if(ARG_CUDA AND USE_CUDA)
    set_target_properties(${test_name} PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON
    )
  endif()

  gtest_discover_tests(${test_name}
    PROPERTIES
    LABELS "unit"
    DISCOVERY_TIMEOUT 60
  )
endfunction()

# Add a Brezel benchmark target
function(brezel_add_benchmark bench_name)
  cmake_parse_arguments(PARSE_ARGV 1 ARG
    "CUDA"
    ""
    "SOURCES;DEPENDS"
  )

  add_executable(${bench_name} ${ARG_SOURCES})

  target_link_libraries(${bench_name}
    PRIVATE
    ${ARG_DEPENDS}
    benchmark::benchmark
    benchmark::benchmark_main
  )

  if(ARG_CUDA AND USE_CUDA)
    set_target_properties(${bench_name} PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON
    )
  endif()
endfunction()