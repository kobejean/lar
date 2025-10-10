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

# Optional Metal-accelerated SIFT (macOS/iOS only)
option(LAR_USE_METAL_SIFT "Enable Metal-accelerated Gaussian pyramid for SIFT" ON)
option(LAR_USE_METAL_SIFT_FUSED "Enable experimental fused Metal SIFT kernel" ON)

# Collect source files
set(TRACKING_SOURCES
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
)

# Add Metal implementation on Apple platforms if enabled
if(LAR_USE_METAL_SIFT AND APPLE)
  list(APPEND TRACKING_SOURCES sift_metal.mm)
  # if(LAR_USE_METAL_SIFT_FUSED)
  #   list(APPEND TRACKING_SOURCES sift_metal_fused.mm)
  # endif()
  message(STATUS "Metal-accelerated SIFT enabled")
endif()

add_library(lar_tracking
  ${TRACKING_SOURCES}
  ${HEADER_LIST}
)

# Floating-point precision configuration
# These flags balance performance with IEEE 754 compliance for reproducible results
target_compile_options(lar_tracking PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:
    -O2                      # High optimization
    -march=native            # Use best available SIMD for current CPU
    # -mfpmath=sse             # Use SSE for FP math (consistent 32/64-bit precision)
    -fno-fast-math
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

# Link Metal frameworks if Metal SIFT is enabled
if(LAR_USE_METAL_SIFT AND APPLE)
  find_library(METAL_LIBRARY Metal)
  find_library(METAL_PERFORMANCE_SHADERS MetalPerformanceShaders)

  if(METAL_LIBRARY AND METAL_PERFORMANCE_SHADERS)
    target_link_libraries(lar_tracking PRIVATE
      ${METAL_LIBRARY}
      ${METAL_PERFORMANCE_SHADERS}
    )
    target_compile_definitions(lar_tracking PRIVATE LAR_USE_METAL_SIFT)
    # Optional: Enable profiling
    # target_compile_definitions(lar_tracking PRIVATE LAR_PROFILE_METAL_SIFT)    target_compile_definitions(lar_tracking PRIVATE LAR_USE_METAL_SIFT)
    # Optional: Enable profiling
    # target_compile_definitions(lar_tracking PRIVATE LAR_PROFILE_METAL_SIFT)

    # Compile Metal shader library
    set(METAL_SHADER_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/sift.metal")
    set(METAL_AIR_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/sift.air")
    set(METAL_LIB_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/sift.metallib")

    # Step 1: Compile .metal to .air (Apple Intermediate Representation)
    # Flags optimize for precision and reproducibility:
    #   -O2: High optimization without aggressive transforms
    #   -fno-fast-math: Disable unsafe optimizations (reassociation, reciprocals, etc.)
    #   -ffp-contract=off: Disable FMA to match CPU computation exactly
    add_custom_command(
      OUTPUT ${METAL_AIR_OUTPUT}
      COMMAND xcrun -sdk macosx metal -c -O2 -fno-fast-math -ffp-contract=off ${METAL_SHADER_SOURCE} -o ${METAL_AIR_OUTPUT}
      DEPENDS ${METAL_SHADER_SOURCE}
      COMMENT "Compiling Metal shader: sift.metal → sift.air (precision mode)"
    )

    # Step 2: Link .air to .metallib
    add_custom_command(
      OUTPUT ${METAL_LIB_OUTPUT}
      COMMAND xcrun -sdk macosx metallib ${METAL_AIR_OUTPUT} -o ${METAL_LIB_OUTPUT}
      DEPENDS ${METAL_AIR_OUTPUT}
      COMMENT "Linking Metal library: sift.air → sift.metallib"
    )

    # Step 3: Copy .metallib to runtime bin directory (where executables are installed)
    # This is CMAKE_SOURCE_DIR/bin for the lar project
    set(RUNTIME_BIN_DIR "${CMAKE_SOURCE_DIR}/bin")
    add_custom_command(
      OUTPUT ${RUNTIME_BIN_DIR}/sift.metallib
      COMMAND ${CMAKE_COMMAND} -E make_directory ${RUNTIME_BIN_DIR}
      COMMAND ${CMAKE_COMMAND} -E copy ${METAL_LIB_OUTPUT} ${RUNTIME_BIN_DIR}/sift.metallib
      DEPENDS ${METAL_LIB_OUTPUT}
      COMMENT "Copying sift.metallib to ${RUNTIME_BIN_DIR}/"
    )

    # Also copy to build directory bin for consistency
    add_custom_command(
      OUTPUT ${PROJECT_BINARY_DIR}/bin/sift.metallib
      COMMAND ${CMAKE_COMMAND} -E copy ${METAL_LIB_OUTPUT} ${PROJECT_BINARY_DIR}/bin/sift.metallib
      DEPENDS ${METAL_LIB_OUTPUT}
      COMMENT "Copying sift.metallib to ${PROJECT_BINARY_DIR}/bin/"
    )

    # Create custom target to ensure Metal library is built and copied to both locations
    add_custom_target(metal_sift_shaders ALL
      DEPENDS ${RUNTIME_BIN_DIR}/sift.metallib ${PROJECT_BINARY_DIR}/bin/sift.metallib
    )

    # Make lar_tracking depend on the Metal shader compilation
    add_dependencies(lar_tracking metal_sift_shaders)

    message(STATUS "Metal shader compilation configured: sift.metal → sift.metallib")
  else()
    message(WARNING "Metal frameworks not found. Metal SIFT will be disabled.")
  endif()
endif()

# Optional: Compile experimental fused Metal shader
if(LAR_USE_METAL_SIFT_FUSED AND LAR_USE_METAL_SIFT AND APPLE)
  if(METAL_LIBRARY AND METAL_PERFORMANCE_SHADERS)
    # Fused shader compilation
    set(METAL_FUSED_SHADER_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/sift_fused.metal")
    set(METAL_FUSED_AIR_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/sift_fused.air")
    set(METAL_FUSED_LIB_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/sift_fused.metallib")

    # Step 1: Compile .metal to .air (Apple Intermediate Representation)
    # Match precision flags with main sift.metal shader
    add_custom_command(
      OUTPUT ${METAL_FUSED_AIR_OUTPUT}
      COMMAND xcrun -sdk macosx metal -c -O2 -fno-fast-math -ffp-contract=off ${METAL_FUSED_SHADER_SOURCE} -o ${METAL_FUSED_AIR_OUTPUT}
      DEPENDS ${METAL_FUSED_SHADER_SOURCE}
      COMMENT "Compiling fused Metal shader: sift_fused.metal → sift_fused.air (precision mode)"
    )

    # Step 2: Link .air to .metallib
    add_custom_command(
      OUTPUT ${METAL_FUSED_LIB_OUTPUT}
      COMMAND xcrun -sdk macosx metallib ${METAL_FUSED_AIR_OUTPUT} -o ${METAL_FUSED_LIB_OUTPUT}
      DEPENDS ${METAL_FUSED_AIR_OUTPUT}
      COMMENT "Linking fused Metal library: sift_fused.air → sift_fused.metallib"
    )

    # Step 3: Copy .metallib to runtime bin directory
    add_custom_command(
      OUTPUT ${RUNTIME_BIN_DIR}/sift_fused.metallib
      COMMAND ${CMAKE_COMMAND} -E copy ${METAL_FUSED_LIB_OUTPUT} ${RUNTIME_BIN_DIR}/sift_fused.metallib
      DEPENDS ${METAL_FUSED_LIB_OUTPUT}
      COMMENT "Copying sift_fused.metallib to ${RUNTIME_BIN_DIR}/"
    )

    # Also copy to build directory bin for consistency
    add_custom_command(
      OUTPUT ${PROJECT_BINARY_DIR}/bin/sift_fused.metallib
      COMMAND ${CMAKE_COMMAND} -E copy ${METAL_FUSED_LIB_OUTPUT} ${PROJECT_BINARY_DIR}/bin/sift_fused.metallib
      DEPENDS ${METAL_FUSED_LIB_OUTPUT}
      COMMENT "Copying sift_fused.metallib to ${PROJECT_BINARY_DIR}/bin/"
    )

    # Create custom target to ensure fused Metal library is built and copied
    add_custom_target(metal_sift_fused_shaders ALL
      DEPENDS ${RUNTIME_BIN_DIR}/sift_fused.metallib ${PROJECT_BINARY_DIR}/bin/sift_fused.metallib
    )

    # Make lar_tracking depend on the fused Metal shader compilation
    add_dependencies(lar_tracking metal_sift_fused_shaders)

    # Define compile flag so C++ code can use the fused kernel
    target_compile_definitions(lar_tracking PRIVATE LAR_USE_METAL_SIFT_FUSED)

    message(STATUS "Experimental fused Metal SIFT shader enabled: sift_fused.metal → sift_fused.metallib")
  else()
    message(WARNING "Metal frameworks not found. Fused Metal SIFT will be disabled.")
  endif()
endif()

# Set C++ standard if not already set
target_compile_features(lar_tracking PUBLIC cxx_std_11)

# Add include directories
target_include_directories(lar_tracking
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)