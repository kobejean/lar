set(LAR_CORE_TEST_SRCS
  "${CMAKE_CURRENT_SOURCE_DIR}/utils/json_test.cpp"
)
set(LAR_CORE_TEST_SRCS ${LAR_CORE_TEST_SRCS} PARENT_SCOPE)

set(LAR_CORE_TEST_LIBS lar_core)
set(LAR_CORE_TEST_LIBS ${LAR_CORE_TEST_LIBS} PARENT_SCOPE)


add_executable(
  lar_core_test
  ${LAR_CORE_TEST_SRCS}
)
target_link_libraries(
  lar_core_test
  ${LAR_CORE_TEST_LIBS}
  gtest_main
)

include(GoogleTest)
gtest_discover_tests(lar_core_test)