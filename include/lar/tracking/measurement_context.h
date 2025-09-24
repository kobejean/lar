#ifndef LAR_TRACKING_MEASUREMENT_CONTEXT_H
#define LAR_TRACKING_MEASUREMENT_CONTEXT_H

#include <vector>
#include <memory>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>

namespace lar {

// Forward declarations
class Landmark;
struct Frame;

/**
 * Minimal context for measurement processing
 * Contains essential data needed by confidence estimators and pose filters
 */
struct MeasurementContext {
    // === Core Data ===
    std::shared_ptr<std::vector<std::pair<Landmark*, cv::KeyPoint>>> inliers;  // Shared inlier landmark-keypoint pairs (zero-copy)
    size_t total_matches = 0;                                                   // Total feature matches found
    const Frame* frame = nullptr;                                               // Camera frame with intrinsics/extrinsics

    // === Poses ===
    Eigen::Matrix4d measured_pose = Eigen::Matrix4d::Identity();     // LAR measurement result
    Eigen::Matrix4d predicted_pose = Eigen::Matrix4d::Identity();    // Filter prediction

    // === Quality ===
    double confidence = 0.0;                                         // Measurement confidence
};

} // namespace lar

#endif /* LAR_TRACKING_MEASUREMENT_CONTEXT_H */