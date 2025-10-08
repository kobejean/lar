set(HEADER_LIST
  "${PROJECT_SOURCE_DIR}/include/lar/tracking/tracker.h"
  "${PROJECT_SOURCE_DIR}/include/lar/tracking/filtered_tracker.h"
  "${PROJECT_SOURCE_DIR}/include/lar/tracking/filtered_tracker_config.h"
  "${PROJECT_SOURCE_DIR}/include/lar/tracking/measurement_context.h"
  "${PROJECT_SOURCE_DIR}/include/lar/tracking/confidence_estimation/confidence_estimator.h"
  "${PROJECT_SOURCE_DIR}/include/lar/tracking/outlier_detection/outlier_detector.h"
  "${PROJECT_SOURCE_DIR}/include/lar/tracking/pose_filtering/pose_filter_strategy.h"
  "${PROJECT_SOURCE_DIR}/include/lar/tracking/vision.h"
  "${PROJECT_SOURCE_DIR}/include/lar/tracking/sift.h"
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

add_library(lar_tracking
  tracker.cpp
  filtered_tracker.cpp
  vision.cpp
  sift.cpp
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
  ${HEADER_LIST}
)

# Floating-point precision configuration
# These flags balance performance with IEEE 754 compliance for reproducible results
target_compile_options(lar_tracking PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:
    -O3                      # High optimization
    -march=native            # Use best available SIMD for current CPU
    # -mfpmath=sse             # Use SSE for FP math (consistent 32/64-bit precision)
    -ffp-contract=fast       # Allow FMA (fused multiply-add) - MORE accurate than separate ops
    # DO NOT USE -ffast-math # This breaks IEEE 754 compliance!
    # -fno-unsafe-math-optimizations  # Uncomment for strictest compliance
  >
  $<$<CXX_COMPILER_ID:MSVC>:
    /O2                      # Optimize for speed
    /arch:AVX2               # AVX2 SIMD (or /arch:AVX512 if available)
    /fp:precise              # Precise floating-point model (default, good)
    # /fp:strict             # Uncomment for strictest IEEE 754 compliance (slower)
  >
)

target_link_libraries(lar_tracking
  PUBLIC
    opencv_core 
    opencv_calib3d 
    opencv_imgproc
    opencv_features2d  # For KeyPointsFilter utilities
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