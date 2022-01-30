set(LAR_MAPPING_TEST_SRCS
  "${CMAKE_CURRENT_SOURCE_DIR}/frame_test.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/location_matcher_test.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/mapper_test.cpp"
)
set(LAR_MAPPING_TEST_SRCS ${LAR_MAPPING_TEST_SRCS} PARENT_SCOPE)

set(LAR_MAPPING_TEST_LIBS lar_mapping opencv_imgcodecs nlohmann_json::nlohmann_json Eigen3::Eigen)
set(LAR_MAPPING_TEST_LIBS ${LAR_MAPPING_TEST_LIBS} PARENT_SCOPE)


add_executable(
  lar_mapping_test
  ${LAR_MAPPING_TEST_SRCS}
)
target_link_libraries(
  lar_mapping_test
  ${LAR_MAPPING_TEST_LIBS}
  gtest_main
)

include(GoogleTest)
gtest_discover_tests(lar_mapping_test)