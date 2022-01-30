set(LAR_PROCESSING_TEST_SRCS
  "${CMAKE_CURRENT_SOURCE_DIR}/projection_test.cpp"
)
set(LAR_PROCESSING_TEST_SRCS ${LAR_PROCESSING_TEST_SRCS} PARENT_SCOPE)

set(LAR_PROCESSING_TEST_LIBS lar_processing)
set(LAR_PROCESSING_TEST_LIBS ${LAR_PROCESSING_TEST_LIBS} PARENT_SCOPE)


add_executable(
  lar_processing_test
  ${LAR_PROCESSING_TEST_SRCS}
)
target_link_libraries(
  lar_processing_test
  ${LAR_PROCESSING_TEST_LIBS}
  gtest_main
)

include(GoogleTest)
gtest_discover_tests(lar_processing_test)