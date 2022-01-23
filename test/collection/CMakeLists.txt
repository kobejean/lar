set(GEOAR_COLLECTION_TEST_SRCS
  "${CMAKE_CURRENT_SOURCE_DIR}/collection_test.cpp"
)
set(GEOAR_COLLECTION_TEST_SRCS ${GEOAR_COLLECTION_TEST_SRCS} PARENT_SCOPE)

set(GEOAR_COLLECTION_TEST_LIBS geoar_collection opencv_imgcodecs nlohmann_json::nlohmann_json Eigen3::Eigen)
set(GEOAR_COLLECTION_TEST_LIBS ${GEOAR_COLLECTION_TEST_LIBS} PARENT_SCOPE)


add_executable(
  geoar_collection_test
  ${GEOAR_COLLECTION_TEST_SRCS}
)
target_link_libraries(
  geoar_collection_test
  ${GEOAR_COLLECTION_TEST_LIBS}
  gtest_main
)

include(GoogleTest)
gtest_discover_tests(geoar_collection_test)