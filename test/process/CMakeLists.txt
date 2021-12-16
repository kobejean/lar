add_executable(
  geoar_process_test
  projection_test.cpp
)
target_link_libraries(
  geoar_process_test
  geoar_process
)

include(GoogleTest)
gtest_discover_tests(geoar_process_test)