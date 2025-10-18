#include "lar/tracking/filtered_tracker_config.h"
#include "lar/tracking/pose_filtering/extended_kalman_filter.h"
#include "lar/tracking/pose_filtering/sliding_window_ba.h"
#include "lar/tracking/pose_filtering/averaging_filter.h"
#include "lar/tracking/pose_filtering/pass_through_filter.h"
#include "lar/tracking/pose_filtering/smooth_filter.h"
#include <iostream>

namespace lar {

std::unique_ptr<PoseFilterStrategy> FilteredTrackerConfig::createFilterStrategy() const {
    switch (filter_strategy) {
        case FilterStrategy::EXTENDED_KALMAN_FILTER:
            std::cout << "FilteredTracker: Using Extended Kalman Filter" << std::endl;
            return std::make_unique<ExtendedKalmanFilter>();

        case FilterStrategy::SLIDING_WINDOW_BA:
            std::cout << "FilteredTracker: Using Sliding Window Bundle Adjustment" << std::endl;
            return std::make_unique<SlidingWindowBA>();

        case FilterStrategy::AVERAGING:
            std::cout << "FilteredTracker: Using Averaging Filter" << std::endl;
            return std::make_unique<AveragingFilter>();

        case FilterStrategy::PASS_THROUGH:
            std::cout << "FilteredTracker: Using Pass Through Filter" << std::endl;
            return std::make_unique<PassThroughFilter>();

        case FilterStrategy::SMOOTH_FILTER:
            std::cout << "FilteredTracker: Using Smooth Filter" << std::endl;
            return std::make_unique<SmoothFilter>();

        default:
            std::cout << "FilteredTracker: Unknown strategy, defaulting to SWBA" << std::endl;
            return std::make_unique<SlidingWindowBA>();
    }
}

} // namespace lar