#include "lar/tracking/outlier_detection/outlier_detector_base.h"
#include "lar/tracking/filtered_tracker_config.h"

namespace lar {

OutlierAnalysisResult OutlierDetector::analyzeOutlier(
    const Eigen::Matrix4d& measurement,
    const Eigen::Matrix4d& predicted_state,
    const Eigen::MatrixXd& covariance,
    double confidence,
    const FilteredTrackerConfig& config) {

    // Use the subclass's outlier detection method
    bool is_geometric_outlier = isOutlier(measurement, predicted_state, covariance, confidence, config);

    OutlierAnalysisResult result;
    result.is_outlier = is_geometric_outlier;

    if (is_geometric_outlier) {
        consecutive_rejections_++;

        // Simple pattern analysis with tunable parameters
        // Default: 3 consecutive rejections + confidence > 0.4 suggests bad state
        bool likely_bad_state = (consecutive_rejections_ >= 3) && (confidence > 0.4);

        result.likely_bad_state = likely_bad_state;
        result.should_reset_motion = likely_bad_state;
        result.consecutive_rejections = consecutive_rejections_;

    } else {
        // Good measurement - reset consecutive counter
        consecutive_rejections_ = 0;
        result.likely_bad_state = false;
        result.should_reset_motion = false;
        result.consecutive_rejections = 0;
    }

    return result;
}

} // namespace lar