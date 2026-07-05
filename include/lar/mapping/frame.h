#ifndef LAR_MAPPING_FRAME_H
#define LAR_MAPPING_FRAME_H

#include <nlohmann/json.hpp>
#include <opencv2/features2d.hpp>

#include "lar/core/utils/json.h"
#include "lar/core/landmark.h"

namespace lar {

  // Platform-neutral confidence label for the odometry edge *into* a frame (the
  // interval from the previous keyframe up to this one). Recorded at capture time as
  // the worst tracking state observed over that interval; see docs/ODOMETRY_CONFIDENCE.md.
  // iOS maps ARCamera.trackingState -> these; a future ARCore app maps TrackingState +
  // TrackingFailureReason -> the same values. Severity increases with the value so a
  // plain max() is the correct worst-of reduction. Offline tiers: {Normal}=HIGH,
  // {1,2,3}=MEDIUM, {4,5}=LOW.
  enum class OdometryConfidence : int {
    Normal = 0,
    LimitedInitializing = 1,
    LimitedExcessiveMotion = 2,
    LimitedInsufficientVisual = 3,
    Relocalizing = 4,
    Unavailable = 5,
  };

  class Frame {
    public:
      // TODO: Make these attributes const
      size_t id;
      long long timestamp;
      Eigen::Matrix3d intrinsics;
      Eigen::Matrix4d extrinsics;
      // Worst platform tracking state over the interval into this frame (see
      // OdometryConfidence). Default 0 = Normal so captures predating this field keep
      // the previous uniform-trust behavior.
      int odom_state{0};
      // Auxilary data
      bool processed{false};

      Frame();
  };
  // _WITH_DEFAULT so older frames.json without odom_state still parse (missing key ->
  // default-constructed value, i.e. 0 = Normal). No migration needed.
  NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(Frame, id, timestamp, intrinsics, extrinsics, odom_state)
}

#endif /* LAR_MAPPING_FRAME_H */