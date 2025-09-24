#ifndef LAR_TRACKING_OUTLIER_DETECTION_OUTLIER_DETECTOR_BASE_H
#define LAR_TRACKING_OUTLIER_DETECTION_OUTLIER_DETECTOR_BASE_H

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace lar {

// Forward declarations
struct FilteredTrackerConfig;

/**
 * Result of outlier analysis including pattern detection
 */
struct OutlierAnalysisResult {
    bool is_outlier = false;           // Standard outlier detection result
    bool likely_bad_state = false;     // Pattern suggests bad filter state
    bool should_reset_motion = false;  // Recommend resetting accumulated motion
    int consecutive_rejections = 0;    // Current streak of rejections
};

/**
 * Abstract interface for outlier detection strategies.
 * Allows pluggable outlier detection algorithms.
 */
class OutlierDetector {
public:
    virtual ~OutlierDetector() = default;

    /**
     * Check if a measurement is an outlier
     * @param measurement Camera pose measurement
     * @param predicted_state Predicted camera pose
     * @param covariance Current pose covariance matrix
     * @param confidence Measurement confidence
     * @param config Configuration parameters
     * @return True if measurement should be rejected as outlier
     */
    virtual bool isOutlier(
        const Eigen::Matrix4d& measurement,
        const Eigen::Matrix4d& predicted_state,
        const Eigen::MatrixXd& covariance,
        double confidence,
        const FilteredTrackerConfig& config) const = 0;

    /**
     * Enhanced outlier analysis with pattern detection
     * Default implementation provides simple consecutive rejection tracking
     * @param measurement Camera pose measurement
     * @param predicted_state Predicted camera pose
     * @param covariance Current pose covariance matrix
     * @param confidence Measurement confidence
     * @param config Configuration parameters
     * @return Detailed analysis result including pattern recommendations
     */
    virtual OutlierAnalysisResult analyzeOutlier(
        const Eigen::Matrix4d& measurement,
        const Eigen::Matrix4d& predicted_state,
        const Eigen::MatrixXd& covariance,
        double confidence,
        const FilteredTrackerConfig& config);

protected:
    // Mutable to allow state tracking in const methods
    mutable int consecutive_rejections_ = 0;
};

} // namespace lar

#endif /* LAR_TRACKING_OUTLIER_DETECTION_OUTLIER_DETECTOR_BASE_H */