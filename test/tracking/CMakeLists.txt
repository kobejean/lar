set(GEOAR_TRACKING_TEST_SRCS
  "${CMAKE_CURRENT_SOURCE_DIR}/tracker_test.cpp"
)
set(GEOAR_TRACKING_TEST_SRCS ${GEOAR_TRACKING_TEST_SRCS} PARENT_SCOPE)

set(GEOAR_TRACKING_TEST_LIBS geoar_tracking opencv_imgcodecs)
set(GEOAR_TRACKING_TEST_LIBS ${GEOAR_TRACKING_TEST_LIBS} PARENT_SCOPE)


add_executable(
  geoar_tracking_test
  ${GEOAR_TRACKING_TEST_SRCS}
)
target_link_libraries(
  geoar_tracking_test
  ${GEOAR_TRACKING_TEST_LIBS}
  gtest_main
)

include(GoogleTest)
gtest_discover_tests(geoar_tracking_test)