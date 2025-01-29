add_executable(shape_tests
  shape_tests.cpp
  main.cpp
)

target_link_libraries(shape_tests
  PRIVATE
  nova_core
  GTest::gtest
  GTest::gtest_main
  fmt::fmt
)

if(UNIX AND NOT APPLE)
  target_link_libraries(shape_tests PRIVATE pthread)
endif()

target_include_directories(shape_tests
  PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)

add_test(
  NAME shape_tests
  COMMAND shape_tests
  WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)