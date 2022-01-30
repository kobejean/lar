
# Setup GoogleTest
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/release-1.11.0.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_subdirectory(core)
add_subdirectory(mapping)
add_subdirectory(processing)
add_subdirectory(tracking)

set(LAR_CORE_TEST_LIBS 
  ${LAR_CORE_TEST_LIBS}
  ${LAR_MAPPING_TEST_LIBS}
  ${LAR_PROCESSING_TEST_LIBS}
  ${LAR_RTREE_TEST_LIBS}
  ${LAR_TRACKING_TEST_LIBS}
)
list(REMOVE_DUPLICATES LAR_CORE_TEST_LIBS)

add_executable(
  lar_test
  all_tests.cpp
  ${LAR_CORE_TEST_SRCS}
  ${LAR_MAPPING_TEST_SRCS}
  ${LAR_PROCESSING_TEST_SRCS}
  ${LAR_RTREE_TEST_SRCS}
  ${LAR_TRACKING_TEST_SRCS}
)
target_link_libraries(
  lar_test
  ${LAR_CORE_TEST_LIBS}
  gtest
)