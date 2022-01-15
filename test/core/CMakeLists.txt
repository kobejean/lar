set(GEOAR_CORE_TEST_SRCS
  "${CMAKE_CURRENT_SOURCE_DIR}/utils/json_test.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/frame_test.cpp"
)
set(GEOAR_CORE_TEST_SRCS ${GEOAR_CORE_TEST_SRCS} PARENT_SCOPE)

set(GEOAR_CORE_TEST_LIBS geoar_core)
set(GEOAR_CORE_TEST_LIBS ${GEOAR_CORE_TEST_LIBS} PARENT_SCOPE)


add_executable(
  geoar_core_test
  ${GEOAR_CORE_TEST_SRCS}
)
target_link_libraries(
  geoar_core_test
  ${GEOAR_CORE_TEST_LIBS}
  gtest_main
)

include(GoogleTest)
gtest_discover_tests(geoar_core_test)