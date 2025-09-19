set(LAR_PROCESSING_TEST_SRCS
  "${CMAKE_CURRENT_SOURCE_DIR}/global_alignment_test.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/projection_test.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/colmap_database_test.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/colmap_refiner_test.cpp"
)
set(LAR_PROCESSING_TEST_LIBS lar_processing)


add_executable(lar_processing_test ${LAR_PROCESSING_TEST_SRCS})
target_link_libraries(lar_processing_test ${LAR_PROCESSING_TEST_LIBS} gtest_main)

# Add OpenGL libraries for g2o visualization support
if(OPENGL_LIBRARIES)
    target_link_libraries(lar_processing_test ${OPENGL_LIBRARIES})
endif()

include(GoogleTest)
gtest_discover_tests(lar_processing_test)

# add tests to parent targets
target_sources(lar_test PRIVATE ${LAR_PROCESSING_TEST_SRCS})
target_link_libraries(lar_test ${LAR_PROCESSING_TEST_LIBS})