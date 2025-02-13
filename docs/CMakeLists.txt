# docs/CMakeLists.txt
find_package(Doxygen REQUIRED)

if(DOXYGEN_FOUND)
  # Download doxygen-awesome-css
  include(FetchContent)
  FetchContent_Declare(
    doxygen-awesome-css
    GIT_REPOSITORY https://github.com/jothepro/doxygen-awesome-css.git
    GIT_TAG v2.3.1
  )
  FetchContent_MakeAvailable(doxygen-awesome-css)

  # Ensure theme directory exists
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/theme)

  # Copy theme files
  configure_file(
    ${doxygen-awesome-css_SOURCE_DIR}/doxygen-awesome.css
    ${CMAKE_CURRENT_SOURCE_DIR}/theme/doxygen-awesome.css
    COPYONLY
  )

  configure_file(
    ${doxygen-awesome-css_SOURCE_DIR}/doxygen-awesome-darkmode-toggle.js
    ${CMAKE_CURRENT_SOURCE_DIR}/theme/doxygen-awesome-darkmode-toggle.js
    COPYONLY
  )

  configure_file(
    ${doxygen-awesome-css_SOURCE_DIR}/doxygen-awesome-fragment-copy-button.js
    ${CMAKE_CURRENT_SOURCE_DIR}/theme/doxygen-awesome-fragment-copy-button.js
    COPYONLY
  )

  configure_file(
    ${doxygen-awesome-css_SOURCE_DIR}/doxygen-awesome-paragraph-link.js
    ${CMAKE_CURRENT_SOURCE_DIR}/theme/doxygen-awesome-paragraph-link.js
    COPYONLY
  )

  configure_file(
    ${doxygen-awesome-css_SOURCE_DIR}/doxygen-awesome-interactive-toc.js
    ${CMAKE_CURRENT_SOURCE_DIR}/theme/doxygen-awesome-interactive-toc.js
    COPYONLY
  )

  # Set output directories
  set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

  # Configure Doxyfile
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
    ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)

  # Add documentation target
  add_custom_target(docs
    COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating documentation with Doxygen"
    VERBATIM)

  # Install documentation
  install(DIRECTORY ${DOXYGEN_OUTPUT_DIR}/html
    DESTINATION share/doc/${PROJECT_NAME}
    OPTIONAL)
else()
  message(WARNING "Doxygen not found - documentation will not be built")
endif()