#ifndef LAR_TRACKING_CONFIDENCE_ESTIMATOR_H
#define LAR_TRACKING_CONFIDENCE_ESTIMATOR_H

#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>

namespace lar {

// Forward declarations
class Landmark;
struct FilteredTrackerConfig;

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
     * @param config Configuration parameters
     * @return Confidence value in range [0,1]
     */
    virtual double calculateConfidence(
        const std::vector<std::pair<Landmark*, cv::KeyPoint>>& inliers,
        const Eigen::Matrix4d& T_lar_from_camera,
        const FilteredTrackerConfig& config) const = 0;

    /**
     * Calculate measurement noise covariance matrix
     * @param inliers Vector of matched landmark-keypoint pairs
     * @param T_lar_from_camera Camera pose in LAR world coordinates
     * @param confidence Overall confidence value
     * @param config Configuration parameters
     * @return 6x6 measurement noise covariance matrix
     */
    virtual Eigen::MatrixXd calculateMeasurementNoise(
        const std::vector<std::pair<Landmark*, cv::KeyPoint>>& inliers,
        const Eigen::Matrix4d& T_lar_from_camera,
        double confidence,
        const FilteredTrackerConfig& config) const = 0;
};

/**
 * Geometric confidence estimator based on inlier count and spatial distribution.
 * This is the current implementation extracted from FilteredTracker.
 */
class GeometricConfidenceEstimator : public ConfidenceEstimator {
public:
    double calculateConfidence(
        const std::vector<std::pair<Landmark*, cv::KeyPoint>>& inliers,
        const Eigen::Matrix4d& T_lar_from_camera,
        const FilteredTrackerConfig& config) const override;

    Eigen::MatrixXd calculateMeasurementNoise(
        const std::vector<std::pair<Landmark*, cv::KeyPoint>>& inliers,
        const Eigen::Matrix4d& T_lar_from_camera,
        double confidence,
        const FilteredTrackerConfig& config) const override;

private:
    /**
     * Calculate spatial distribution factor based on landmark spread
     * @param inliers Vector of matched landmark-keypoint pairs
     * @param camera_position Camera position in LAR world
     * @param config Configuration parameters
     * @return Spatial distribution factor [0,1]
     */
    double calculateSpatialDistribution(
        const std::vector<std::pair<Landmark*, cv::KeyPoint>>& inliers,
        const Eigen::Vector3d& camera_position,
        const FilteredTrackerConfig& config) const;
};

/**
 * Simple confidence estimator that only considers inlier count.
 * Useful for comparison and debugging.
 */
class SimpleConfidenceEstimator : public ConfidenceEstimator {
public:
    double calculateConfidence(
        const std::vector<std::pair<Landmark*, cv::KeyPoint>>& inliers,
        const Eigen::Matrix4d& T_lar_from_camera,
        const FilteredTrackerConfig& config) const override;

    Eigen::MatrixXd calculateMeasurementNoise(
        const std::vector<std::pair<Landmark*, cv::KeyPoint>>& inliers,
        const Eigen::Matrix4d& T_lar_from_camera,
        double confidence,
        const FilteredTrackerConfig& config) const override;
};

} // namespace lar

#endif /* LAR_TRACKING_CONFIDENCE_ESTIMATOR_H */