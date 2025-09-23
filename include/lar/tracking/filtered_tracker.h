#ifndef LAR_TRACKING_FILTERED_TRACKER_H
#define LAR_TRACKING_FILTERED_TRACKER_H

#include <memory>
#include <chrono>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include "lar/tracking/tracker.h"
#include "lar/tracking/filtered_tracker_config.h"
#include "lar/core/utils/transform.h"
#include "lar/tracking/confidence_estimator.h"
#include "lar/tracking/outlier_detector.h"
#include "lar/tracking/pose_filter_strategy.h"

namespace lar {

/**
 * FilteredTracker provides camera-centric tracking with drift correction
 * between VIO coordinates and LAR map coordinates.
 *
 * COORDINATE CONVENTIONS:
 * - LAR World: Right-handed, Y-up, landmarks and camera poses
 * - VIO World: Right-handed, Y-up, VIO coordinate system (ARKit, ARCore, etc.)
 * - T_lar_from_camera: Camera pose IN LAR world coordinates (primary state)
 * - T_vio_from_camera: Camera pose IN VIO world coordinates
 * - T_lar_from_vio: Coordinate transform FROM VIO TO LAR world (output)
 */
class FilteredTracker {
public:

    struct MeasurementResult {
        bool success;
        Eigen::Matrix4d map_to_vio_transform;  // LAR world → VIO world coordinate transform
        double confidence;
        std::vector<Landmark*> matched_landmarks;
        std::vector<std::pair<Landmark*, cv::KeyPoint>> inliers;
    };

private:
    std::unique_ptr<Tracker> base_tracker_;

    // Configuration
    FilteredTrackerConfig config_;

    // Pluggable architectural components
    std::unique_ptr<PoseFilterStrategy> filter_strategy_;
    std::unique_ptr<ConfidenceEstimator> confidence_estimator_;
    std::unique_ptr<OutlierDetector> outlier_detector_;

    // VIO motion estimation state
    Eigen::Matrix4d last_vio_camera_pose_;
    Eigen::Matrix4d current_vio_camera_pose_;
    bool has_vio_poses_;

    // Coordinate transform state
    MeasurementResult last_transform_result_;

    // Timing
    std::chrono::steady_clock::time_point last_prediction_time_;

    // Helper methods
    Eigen::Matrix4d computeMotionDelta();
    MeasurementResult computeCoordinateTransform(const Eigen::Matrix4d& T_lar_from_camera, double confidence);

public:
    /**
     * Constructor with default strategies (backwards compatible)
     * @param tracker Existing LAR tracker for measurements
     * @param measurement_interval Update interval for LAR measurements (default: 2.0s)
     */
    FilteredTracker(std::unique_ptr<Tracker> tracker, double measurement_interval = 2.0);

    /**
     * Constructor with custom configuration
     * @param tracker Existing LAR tracker for measurements
     * @param config Configuration parameters
     */
    FilteredTracker(std::unique_ptr<Tracker> tracker, const FilteredTrackerConfig& config);

    /**
     * Constructor with full dependency injection (most flexible)
     * @param tracker Existing LAR tracker for measurements
     * @param config Configuration parameters
     * @param filter_strategy Pluggable filtering algorithm
     * @param confidence_estimator Pluggable confidence estimation
     * @param outlier_detector Pluggable outlier detection
     */
    FilteredTracker(std::unique_ptr<Tracker> tracker,
                   const FilteredTrackerConfig& config,
                   std::unique_ptr<PoseFilterStrategy> filter_strategy,
                   std::unique_ptr<ConfidenceEstimator> confidence_estimator,
                   std::unique_ptr<OutlierDetector> outlier_detector);

    /**
     * Update current VIO camera pose - call every VIO frame BEFORE predictStep
     * @param T_vio_from_camera Current VIO camera pose transform
     */
    void updateVIOCameraPose(const Eigen::Matrix4d& T_vio_from_camera);

    /**
     * Prediction step - call every VIO frame AFTER updateVIOCameraPose
     * Updates internal state based on VIO motion
     */
    void predictStep();

    /**
     * Measurement update - call every measurement_interval
     * @param image Input image for localization
     * @param frame Frame data with GPS and camera info
     * @param query_x GPS query center X
     * @param query_z GPS query center Z
     * @param query_diameter GPS query radius
     * @return Measurement result with success status and transform
     */
    MeasurementResult measurementUpdate(cv::InputArray image,
                                      const Frame& frame,
                                      double query_x,
                                      double query_z,
                                      double query_diameter);

    /**
     * Get current filtered VIO → LAR map transform
     * This transform can be applied to VIO poses to get map-aligned poses
     */
    Eigen::Matrix4d getFilteredTransform() const;

    /**
     * Get prediction uncertainty (trace of position covariance)
     */
    double getPositionUncertainty() const;

    /**
     * Check if tracker is initialized and ready for use
     */
    bool isInitialized() const;

    /**
     * Check if tracker is currently animating between measurements
     */
    bool isAnimating() const { return false; } // Animation removed in camera-centric approach

    /**
     * Reset tracker state (call when tracking is lost)
     */
    void reset();

    /**
     * Access underlying tracker for direct operations
     */
    const Tracker& getBaseTracker() const { return *base_tracker_; }


};

} // namespace lar

#endif /* LAR_TRACKING_FILTERED_TRACKER_H */
