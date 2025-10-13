# SIFT (Scale-Invariant Feature Transform) implementation
# Includes CPU and optional Metal GPU-accelerated variants

set(SIFT_HEADER_LIST
  "${PROJECT_SOURCE_DIR}/include/lar/tracking/sift/sift.h"
  "${PROJECT_SOURCE_DIR}/include/lar/tracking/sift/sift_config.h"
  "${PROJECT_SOURCE_DIR}/include/lar/tracking/sift/sift_common.h"
  "${PROJECT_SOURCE_DIR}/include/lar/tracking/sift/sift_constants.h"
)

set(SIFT_SOURCES
  sift.cpp
  sift_common.cpp
)

# Add Metal implementation on Apple platforms if enabled
if(LAR_USE_METAL_SIFT AND APPLE)
  # Add Metal SIFT header and consolidated implementation (RAII-based)
  list(APPEND SIFT_HEADER_LIST "${PROJECT_SOURCE_DIR}/include/lar/tracking/sift/metal_sift.h")
  list(APPEND SIFT_SOURCES metal_sift.mm)
  message(STATUS "Metal-accelerated SIFT enabled")
endif()

# Add SIFT sources to lar_tracking library
target_sources(lar_tracking PRIVATE
  ${SIFT_HEADER_LIST}
  ${SIFT_SOURCES}
)

# ============================================================================
# Metal Shader Compilation
# ============================================================================

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

    # Compile Metal shader library
    set(METAL_SHADER_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/metal/sift.metal")
    set(METAL_CONSTANTS_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/metal/sift_constants.metal")
    set(METAL_AIR_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/sift.air")
    set(METAL_LIB_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/sift.metallib")

    # Step 1: Compile .metal to .air (Apple Intermediate Representation)
    # Flags optimize for precision and reproducibility:
    #   -O2: High optimization without aggressive transforms
    #   -fno-fast-math: Disable unsafe optimizations (reassociation, reciprocals, etc.)
    #   -ffp-contract=off: Disable FMA to match CPU computation exactly
    #   -I: Include directory for sift_constants.metal header
    add_custom_command(
      OUTPUT ${METAL_AIR_OUTPUT}
      COMMAND xcrun -sdk macosx metal -c -fno-fast-math -fmetal-math-mode=safe -I${CMAKE_CURRENT_SOURCE_DIR}/metal ${METAL_SHADER_SOURCE} -o ${METAL_AIR_OUTPUT}
      DEPENDS ${METAL_SHADER_SOURCE} ${METAL_CONSTANTS_HEADER}
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
