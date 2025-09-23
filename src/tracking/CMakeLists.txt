set(HEADER_LIST
  "${PROJECT_SOURCE_DIR}/include/lar/tracking/tracker.h"
  "${PROJECT_SOURCE_DIR}/include/lar/tracking/filtered_tracker.h"
  "${PROJECT_SOURCE_DIR}/include/lar/tracking/filtered_tracker_config.h"
  "${PROJECT_SOURCE_DIR}/include/lar/tracking/camera_pose_filter.h"
  "${PROJECT_SOURCE_DIR}/include/lar/tracking/coordinate_transform_manager.h"
  "${PROJECT_SOURCE_DIR}/include/lar/tracking/vio_motion_estimator.h"
  "${PROJECT_SOURCE_DIR}/include/lar/tracking/confidence_estimator.h"
  "${PROJECT_SOURCE_DIR}/include/lar/tracking/outlier_detector.h"
  "${PROJECT_SOURCE_DIR}/include/lar/tracking/pose_filter_strategy.h"
  "${PROJECT_SOURCE_DIR}/include/lar/tracking/vision.h"
  "${PROJECT_SOURCE_DIR}/include/lar/tracking/sift.h"
)

add_library(lar_tracking
  tracker.cpp
  filtered_tracker.cpp
  camera_pose_filter.cpp
  coordinate_transform_manager.cpp
  vio_motion_estimator.cpp
  confidence_estimator.cpp
  outlier_detector.cpp
  pose_filter_strategy.cpp
  vision.cpp
  sift.cpp
  ${HEADER_LIST}
)

target_link_libraries(lar_tracking
  opencv_core opencv_calib3d opencv_imgproc
  lar_core lar_mapping
)