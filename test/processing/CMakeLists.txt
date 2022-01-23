set(GEOAR_PROCESSING_TEST_SRCS
  "${CMAKE_CURRENT_SOURCE_DIR}/projection_test.cpp"
)
set(GEOAR_PROCESSING_TEST_SRCS ${GEOAR_PROCESSING_TEST_SRCS} PARENT_SCOPE)

set(GEOAR_PROCESSING_TEST_LIBS geoar_processing)
set(GEOAR_PROCESSING_TEST_LIBS ${GEOAR_PROCESSING_TEST_LIBS} PARENT_SCOPE)


add_executable(
  geoar_processing_test
  ${GEOAR_PROCESSING_TEST_SRCS}
)
target_link_libraries(
  geoar_processing_test
  ${GEOAR_PROCESSING_TEST_LIBS}
  gtest_main
)

include(GoogleTest)
gtest_discover_tests(geoar_processing_test)