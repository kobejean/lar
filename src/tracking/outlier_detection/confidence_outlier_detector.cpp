//
// ConfidenceOutlierDetector.cpp - Confidence-based outlier detection
//

#include "lar/tracking/outlier_detection/confidence_outlier_detector.h"
#include "lar/tracking/filtered_tracker_config.h"
#include <iostream>

namespace lar {

bool ConfidenceOutlierDetector::isOutlier(
    const Eigen::Matrix4d& measurement,
    const Eigen::Matrix4d& predicted_state,
    const Eigen::MatrixXd& covariance,
    double confidence,
    const FilteredTrackerConfig& config) const {

    if (!config.enable_outlier_detection) {
        return false;
    }

    bool is_outlier = confidence < MIN_CONFIDENCE_THRESHOLD;

    if (config.enable_debug_output && is_outlier) {
        std::cout << "Confidence outlier detected: confidence = " << confidence
                  << " (threshold = " << MIN_CONFIDENCE_THRESHOLD << ")" << std::endl;
    }

    return is_outlier;
}

} // namespace lar