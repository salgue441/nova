function(nova_add_target target_name)
  cmake_parse_arguments(ARG "CUDA" "" "SOURCES;DEPENDS" ${ARGN})

  add_library(${target_name} ${ARG_SOURCES})
  add_library(Nova::${target_name} ALIAS ${target_name})

  target_include_directories(${target_name}
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  )

  if(ARG_DEPENDS)
    target_link_libraries(${target_name} PUBLIC ${ARG_DEPENDS})
  endif()

  if(ARG_CUDA AND USE_CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
      message(FATAL_ERROR "CUDA compiler required for target ${target_name} but not found")
    endif()

    set_target_properties(${target_name} PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_STANDARD 17
    )
  endif()
endfunction()