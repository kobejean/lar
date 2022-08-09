set(LAR_TRACKING_TEST_SRCS
  "${CMAKE_CURRENT_SOURCE_DIR}/tracker_test.cpp"
)
set(LAR_TRACKING_TEST_LIBS lar_tracking opencv_imgcodecs)

add_executable(lar_tracking_test ${LAR_TRACKING_TEST_SRCS})
target_link_libraries(lar_tracking_test ${LAR_TRACKING_TEST_LIBS} gtest_main)

include(GoogleTest)
gtest_discover_tests(lar_tracking_test)

# add tests to parent targets
target_sources(lar_test PRIVATE ${LAR_TRACKING_TEST_SRCS})
target_link_libraries(lar_test ${LAR_TRACKING_TEST_LIBS})