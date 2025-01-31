add_executable(shape_test
  error_category_test.cpp
  error_code_test.cpp
  error_test.cpp
  main.cpp
)

target_link_libraries(shape_test
  PRIVATE
  brezel_core
  GTest::gtest
  GTest::gtest_main
  fmt::fmt
)

if(UNIX AND NOT APPLE)
  target_link_libraries(shape_test PRIVATE pthread)
endif()

target_include_directories(shape_test
  PRIVATE
  ${CMAKE_SOURCE_DIR}/include
)

add_test(
  NAME shape_test
  COMMAND shape_test
  WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)