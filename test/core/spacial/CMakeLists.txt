set(LAR_CORE_SPACIAL_TEST_SRCS
  "${CMAKE_CURRENT_SOURCE_DIR}/rectangle_test.cpp"
)

# add tests to parent targets
target_sources(lar_core_test PRIVATE ${LAR_CORE_SPACIAL_TEST_SRCS})
target_sources(lar_test PRIVATE ${LAR_CORE_SPACIAL_TEST_SRCS})