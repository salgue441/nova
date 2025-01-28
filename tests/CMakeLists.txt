enable_testing()

function(nova_add_test test_name)
  cmake_parse_arguments(TEST "" "" "SOURCES;DEPENDS" ${ARGN})

  add_executable(${test_name} ${TEST_SOURCES})
  target_link_libraries(${test_name} PRIVATE
    GTest::gtest
    GTest::gtest_main
    ${TEST_DEPENDS}
  )

  add_test(
    NAME ${test_name}
    COMMAND ${test_name}
    WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
  )
endfunction()