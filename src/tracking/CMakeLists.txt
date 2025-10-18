# Optional Metal-accelerated SIFT (macOS/iOS only)
option(LAR_USE_METAL_SIFT "Enable Metal-accelerated SIFT" ON)

set(HEADER_LIST
  "${PROJECT_SOURCE_DIR}/include/lar/tracking/tracker.h"
  "${PROJECT_SOURCE_DIR}/include/lar/tracking/filtered_tracker.h"
  "${PROJECT_SOURCE_DIR}/include/lar/tracking/filtered_tracker_config.h"
  "${PROJECT_SOURCE_DIR}/include/lar/tracking/measurement_context.h"
  "${PROJECT_SOURCE_DIR}/include/lar/tracking/confidence_estimation/confidence_estimator.h"
  "${PROJECT_SOURCE_DIR}/include/lar/tracking/outlier_detection/outlier_detector.h"
  "${PROJECT_SOURCE_DIR}/include/lar/tracking/pose_filtering/pose_filter_strategy.h"
  "${PROJECT_SOURCE_DIR}/include/lar/tracking/vision.h"
  "${PROJECT_SOURCE_DIR}/include/lar/tracking/matching/flann_matcher.h"
  "${PROJECT_SOURCE_DIR}/include/lar/tracking/matching/bf_matcher_metal.h"
  # Confidence estimation headers
  "${PROJECT_SOURCE_DIR}/include/lar/tracking/confidence_estimation/confidence_estimator_base.h"
  "${PROJECT_SOURCE_DIR}/include/lar/tracking/confidence_estimation/geometric_confidence_estimator.h"
  "${PROJECT_SOURCE_DIR}/include/lar/tracking/confidence_estimation/simple_confidence_estimator.h"
  "${PROJECT_SOURCE_DIR}/include/lar/tracking/confidence_estimation/distance_stratified_confidence_estimator.h"
  "${PROJECT_SOURCE_DIR}/include/lar/tracking/confidence_estimation/reprojection_based_confidence_estimator.h"
  # Outlier detection headers
  "${PROJECT_SOURCE_DIR}/include/lar/tracking/outlier_detection/outlier_detector_base.h"
  "${PROJECT_SOURCE_DIR}/include/lar/tracking/outlier_detection/chi_squared_outlier_detector.h"
  "${PROJECT_SOURCE_DIR}/include/lar/tracking/outlier_detection/distance_outlier_detector.h"
  "${PROJECT_SOURCE_DIR}/include/lar/tracking/outlier_detection/confidence_outlier_detector.h"
  # Pose filtering headers
  "${PROJECT_SOURCE_DIR}/include/lar/tracking/pose_filtering/pose_state.h"
  "${PROJECT_SOURCE_DIR}/include/lar/tracking/pose_filtering/pose_filter_strategy_base.h"
  "${PROJECT_SOURCE_DIR}/include/lar/tracking/pose_filtering/extended_kalman_filter.h"
  "${PROJECT_SOURCE_DIR}/include/lar/tracking/pose_filtering/averaging_filter.h"
  "${PROJECT_SOURCE_DIR}/include/lar/tracking/pose_filtering/pass_through_filter.h"
  "${PROJECT_SOURCE_DIR}/include/lar/tracking/pose_filtering/sliding_window_ba.h"
)

# Collect source files
set(TRACKING_SOURCES
  tracker.cpp
  filtered_tracker.cpp
  vision.cpp
  matching/flann_matcher.cpp
  matching/bf_matcher_metal.mm
  # Confidence estimation sources
  confidence_estimation/geometric_confidence_estimator.cpp
  confidence_estimation/simple_confidence_estimator.cpp
  confidence_estimation/distance_stratified_confidence_estimator.cpp
  confidence_estimation/reprojection_based_confidence_estimator.cpp
  # Outlier detection sources
  outlier_detection/outlier_detector_base.cpp
  outlier_detection/chi_squared_outlier_detector.cpp
  outlier_detection/distance_outlier_detector.cpp
  outlier_detection/confidence_outlier_detector.cpp
  # Pose filtering sources
  pose_filtering/pose_state.cpp
  pose_filtering/extended_kalman_filter.cpp
  pose_filtering/averaging_filter.cpp
  pose_filtering/pass_through_filter.cpp
  pose_filtering/sliding_window_ba.cpp
)

add_library(lar_tracking
  ${TRACKING_SOURCES}
  ${HEADER_LIST}
)

# Floating-point precision configuration
# These flags balance performance with IEEE 754 compliance for reproducible results
target_compile_options(lar_tracking PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:
    -O3                      # High optimization
    -march=native            # Use best available SIMD for current CPU
    # -mfpmath=sse             # Use SSE for FP math (consistent 32/64-bit precision)
    -fno-fast-math
    -ffp-contract=fast       # Allow FMA (fused multiply-add) - MORE accurate than separate ops
    # DO NOT USE -ffast-math # This breaks IEEE 754 compliance!
    -fno-unsafe-math-optimizations  # Uncomment for strictest compliance
  >
  $<$<CXX_COMPILER_ID:MSVC>:
    /O3                      # Optimize for speed
    /arch:AVX2               # AVX2 SIMD (or /arch:AVX512 if available)
    /fp:precise              # Precise floating-point model (default, good)
    /fp:strict             # Uncomment for strictest IEEE 754 compliance (slower)
  >
)

target_link_libraries(lar_tracking
  PUBLIC
    opencv_core
    opencv_calib3d
    opencv_imgproc
    opencv_features2d  # For KeyPointsFilter utilities
    opencv_flann       # For custom FlannMatcher
    lar_core
    lar_mapping
  PRIVATE
    g2o_core
    g2o_types_sba
    g2o_types_slam3d
    g2o_solver_eigen
)

# Set C++ standard if not already set
target_compile_features(lar_tracking PUBLIC cxx_std_11)

# Add include directories
target_include_directories(lar_tracking
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Add subdirectories for modular components
add_subdirectory(sift)

# ============================================================================
# Metal BFMatcher Shader Compilation (macOS/iOS only)
# ============================================================================

if(APPLE)
  find_library(METAL_LIBRARY Metal)

  if(METAL_LIBRARY)
    target_link_libraries(lar_tracking PRIVATE ${METAL_LIBRARY})

    # Compile Metal shader library for BFMatcher
    set(BF_MATCHER_SHADER_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/matching/metal/bf_matcher.metal")
    set(BF_MATCHER_AIR_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/bf_matcher.air")
    set(BF_MATCHER_LIB_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/bf_matcher.metallib")

    # Step 1: Compile .metal to .air
    add_custom_command(
      OUTPUT ${BF_MATCHER_AIR_OUTPUT}
      COMMAND xcrun -sdk macosx metal -c -O2 -fno-fast-math ${BF_MATCHER_SHADER_SOURCE} -o ${BF_MATCHER_AIR_OUTPUT}
      DEPENDS ${BF_MATCHER_SHADER_SOURCE}
      COMMENT "Compiling Metal shader: bf_matcher.metal → bf_matcher.air"
    )

    # Step 2: Link .air to .metallib
    add_custom_command(
      OUTPUT ${BF_MATCHER_LIB_OUTPUT}
      COMMAND xcrun -sdk macosx metallib ${BF_MATCHER_AIR_OUTPUT} -o ${BF_MATCHER_LIB_OUTPUT}
      DEPENDS ${BF_MATCHER_AIR_OUTPUT}
      COMMENT "Linking Metal library: bf_matcher.air → bf_matcher.metallib"
    )

    # Step 3: Copy .metallib to runtime bin directory
    set(RUNTIME_BIN_DIR "${CMAKE_SOURCE_DIR}/bin")
    add_custom_command(
      OUTPUT ${RUNTIME_BIN_DIR}/bf_matcher.metallib
      COMMAND ${CMAKE_COMMAND} -E make_directory ${RUNTIME_BIN_DIR}
      COMMAND ${CMAKE_COMMAND} -E copy ${BF_MATCHER_LIB_OUTPUT} ${RUNTIME_BIN_DIR}/bf_matcher.metallib
      DEPENDS ${BF_MATCHER_LIB_OUTPUT}
      COMMENT "Copying bf_matcher.metallib to ${RUNTIME_BIN_DIR}/"
    )

    # Also copy to build directory bin for consistency
    add_custom_command(
      OUTPUT ${PROJECT_BINARY_DIR}/bin/bf_matcher.metallib
      COMMAND ${CMAKE_COMMAND} -E copy ${BF_MATCHER_LIB_OUTPUT} ${PROJECT_BINARY_DIR}/bin/bf_matcher.metallib
      DEPENDS ${BF_MATCHER_LIB_OUTPUT}
      COMMENT "Copying bf_matcher.metallib to ${PROJECT_BINARY_DIR}/bin/"
    )

    # Create custom target to ensure Metal library is built and copied to both locations
    add_custom_target(metal_bf_matcher_shaders ALL
      DEPENDS ${RUNTIME_BIN_DIR}/bf_matcher.metallib ${PROJECT_BINARY_DIR}/bin/bf_matcher.metallib
    )

    # Make lar_tracking depend on the Metal shader compilation
    add_dependencies(lar_tracking metal_bf_matcher_shaders)

    message(STATUS "Metal BFMatcher shader compilation configured: bf_matcher.metal → bf_matcher.metallib")
  else()
    message(WARNING "Metal framework not found. Metal BFMatcher will not be available.")
  endif()
endif()