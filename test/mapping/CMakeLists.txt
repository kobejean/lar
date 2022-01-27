set(GEOAR_MAPPING_TEST_SRCS
  "${CMAKE_CURRENT_SOURCE_DIR}/location_matcher_test.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/mapper_test.cpp"
)
set(GEOAR_MAPPING_TEST_SRCS ${GEOAR_MAPPING_TEST_SRCS} PARENT_SCOPE)

set(GEOAR_MAPPING_TEST_LIBS geoar_mapping opencv_imgcodecs nlohmann_json::nlohmann_json Eigen3::Eigen)
set(GEOAR_MAPPING_TEST_LIBS ${GEOAR_MAPPING_TEST_LIBS} PARENT_SCOPE)


add_executable(
  geoar_mapping_test
  ${GEOAR_MAPPING_TEST_SRCS}
)
target_link_libraries(
  geoar_mapping_test
  ${GEOAR_MAPPING_TEST_LIBS}
  gtest_main
)

include(GoogleTest)
gtest_discover_tests(geoar_mapping_test)