function(brezel_add_target target_name)
  cmake_parse_arguments(ARG "CUDA" "" "SOURCES;DEPENDS" ${ARGN})

  add_library(${target_name} ${ARG_SOURCES})
  add_library(Brezel::${target_name} ALIAS ${target_name})

  target_include_directories(${target_name} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  )

  if(ARG_DEPENDS)
    target_link_libraries(${target_name} PUBLIC ${ARG_DEPENDS})
  endif()

  if(ARG_CUDA and USE_CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
      message(FATAL_ERROR "CUDA compiler required for target ${target_name} but not found")
    endif()

    set_target_properties(${target_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_STANDARD 17)
  endif()
endfunction()

function(brezel_add_test test_name)
  cmake_parse_arguments(TEST "" "" "SOURCES;DEPENDS" ${ARGN})

  add_executable(${test_name} ${TEST_SOURCES})
  target_link_libraries(${test_name} PRIVATE ${TEST_DEPENDS} GTest::gtest GTest::gtest_main Threads::Threads)
  add_test(NAME ${test_name} COMMAND ${test_name})

  if(USE_CUDA)
    set_target_properties(${test_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  endif()

  target_include_directories(${test_name} PRIVATE ${CMAKE_SOURCE_DIR}/include)
endfunction()