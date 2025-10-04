set(LAR_CORE_TEST_SRCS
  "${CMAKE_CURRENT_SOURCE_DIR}/map_test.cpp"
)

set(LAR_CORE_TEST_LIBS lar_core)

add_executable(lar_core_test ${LAR_CORE_TEST_SRCS})
target_link_libraries(lar_core_test ${LAR_CORE_TEST_LIBS} gtest_main)

add_subdirectory(spatial)
add_subdirectory(utils)

include(GoogleTest)
gtest_discover_tests(lar_core_test)

# add tests to parent targets
target_sources(lar_test PRIVATE ${LAR_CORE_TEST_SRCS})
target_link_libraries(lar_test ${LAR_CORE_TEST_LIBS})