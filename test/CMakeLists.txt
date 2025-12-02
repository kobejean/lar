
# Setup GoogleTest
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(lar_test all_tests.cpp)
target_link_libraries(lar_test gtest)

# Add OpenGL libraries for g2o visualization support
if(OPENGL_LIBRARIES)
    target_link_libraries(lar_test ${OPENGL_LIBRARIES})
endif()

add_subdirectory(core)
add_subdirectory(mapping)
add_subdirectory(processing)
add_subdirectory(tracking)