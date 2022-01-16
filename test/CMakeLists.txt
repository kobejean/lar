
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
add_subdirectory(process)
add_subdirectory(tracking)

set(GEOAR_CORE_TEST_LIBS 
  ${GEOAR_CORE_TEST_LIBS}
  ${GEOAR_PROCESS_TEST_LIBS}
  ${GEOAR_RTREE_TEST_LIBS}
  ${GEOAR_TRACKING_TEST_LIBS}
)
list(REMOVE_DUPLICATES GEOAR_CORE_TEST_LIBS)

add_executable(
  geoar_test
  all_tests.cpp
  ${GEOAR_CORE_TEST_SRCS}
  ${GEOAR_PROCESS_TEST_SRCS}
  ${GEOAR_RTREE_TEST_SRCS}
  ${GEOAR_TRACKING_TEST_SRCS}
)
target_link_libraries(
  geoar_test
  ${GEOAR_CORE_TEST_LIBS}
  gtest
)