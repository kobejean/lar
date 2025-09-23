#ifndef LAR_TRACKING_CONFIDENCE_ESTIMATION_CONFIDENCE_ESTIMATOR_BASE_H
#define LAR_TRACKING_CONFIDENCE_ESTIMATION_CONFIDENCE_ESTIMATOR_BASE_H

#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>

namespace lar {

// Forward declarations
class Landmark;
struct FilteredTrackerConfig;
struct Frame;

/**
 * Abstract interface for confidence estimation strategies.
 * Allows pluggable confidence calculation algorithms.
 */
class ConfidenceEstimator {
public:
    virtual ~ConfidenceEstimator() = default;

    /**
     * Calculate overall confidence for a measurement
     * @param inliers Vector of matched landmark-keypoint pairs
     * @param T_lar_from_camera Camera pose in LAR world coordinates
     * @param frame Frame containing camera intrinsics and metadata
     * @param config Configuration parameters
     * @return Confidence value in range [0,1]
     */
    virtual double calculateConfidence(
        const std::vector<std::pair<Landmark*, cv::KeyPoint>>& inliers,
        const Eigen::Matrix4d& T_lar_from_camera,
        const Frame& frame,
        const FilteredTrackerConfig& config) const = 0;

    /**
     * Calculate measurement noise covariance matrix
     * @param inliers Vector of matched landmark-keypoint pairs
     * @param T_lar_from_camera Camera pose in LAR world coordinates
     * @param frame Frame containing camera intrinsics and metadata
     * @param confidence Overall confidence value
     * @param config Configuration parameters
     * @return 6x6 measurement noise covariance matrix
     */
    virtual Eigen::MatrixXd calculateMeasurementNoise(
        const std::vector<std::pair<Landmark*, cv::KeyPoint>>& inliers,
        const Eigen::Matrix4d& T_lar_from_camera,
        const Frame& frame,
        double confidence,
        const FilteredTrackerConfig& config) const = 0;
};

} // namespace lar

#endif /* LAR_TRACKING_CONFIDENCE_ESTIMATION_CONFIDENCE_ESTIMATOR_BASE_H */