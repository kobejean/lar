set(GEOAR_PROCESS_TEST_SRCS
  "${CMAKE_CURRENT_SOURCE_DIR}/projection_test.cpp"
)
set(GEOAR_PROCESS_TEST_SRCS ${GEOAR_PROCESS_TEST_SRCS} PARENT_SCOPE)

set(GEOAR_PROCESS_TEST_LIBS geoar_process)
set(GEOAR_PROCESS_TEST_LIBS ${GEOAR_PROCESS_TEST_LIBS} PARENT_SCOPE)


add_executable(
  geoar_process_test
  ${GEOAR_PROCESS_TEST_SRCS}
)
target_link_libraries(
  geoar_process_test
  ${GEOAR_PROCESS_TEST_LIBS}
  gtest_main
)

include(GoogleTest)
gtest_discover_tests(geoar_process_test)