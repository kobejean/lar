set(LAR_CORE_UTILS_TEST_SRCS
  "${CMAKE_CURRENT_SOURCE_DIR}/json_test.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/wgs84_test.cpp"
)

# add tests to parent targets
target_sources(lar_core_test  PRIVATE ${LAR_CORE_UTILS_TEST_SRCS})
target_sources(lar_test  PRIVATE ${LAR_CORE_UTILS_TEST_SRCS})