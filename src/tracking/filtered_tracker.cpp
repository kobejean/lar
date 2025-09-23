//
// FilteredTracker.cpp - Pluggable Architecture Implementation
//
// COORDINATE CONVENTIONS:
// - LAR World: Right-handed, Y-up, landmarks and camera poses
// - VIO World: Right-handed, Y-up, VIO coordinate system (ARKit, ARCore, etc.)
// - T_lar_from_camera: Camera pose IN LAR world coordinates (primary state)
// - T_vio_from_camera: Camera pose IN VIO world coordinates
// - T_lar_from_vio: Coordinate transform FROM VIO TO LAR world (output)

#include "lar/tracking/filtered_tracker.h"
#include <iostream>
#include <cmath>

namespace lar {

// ============================================================================
// Constructors
// ============================================================================

FilteredTracker::FilteredTracker(std::unique_ptr<Tracker> tracker, double measurement_interval)
    : FilteredTracker(std::move(tracker), FilteredTrackerConfig{measurement_interval}) {
}

FilteredTracker::FilteredTracker(std::unique_ptr<Tracker> tracker, const FilteredTrackerConfig& config)
    : FilteredTracker(std::move(tracker),
                     config,
                     std::make_unique<ExtendedKalmanFilter>(),
                     std::make_unique<GeometricConfidenceEstimator>(),
                     std::make_unique<ChiSquaredOutlierDetector>()) {
}

FilteredTracker::FilteredTracker(std::unique_ptr<Tracker> tracker,
                               const FilteredTrackerConfig& config,
                               std::unique_ptr<PoseFilterStrategy> filter_strategy,
                               std::unique_ptr<ConfidenceEstimator> confidence_estimator,
                               std::unique_ptr<OutlierDetector> outlier_detector)
    : base_tracker_(std::move(tracker))
    , config_(config)
    , filter_strategy_(std::move(filter_strategy))
    , confidence_estimator_(std::move(confidence_estimator))
    , outlier_detector_(std::move(outlier_detector)) {

    // Validate configuration
    if (!config_.isValid()) {
        throw std::invalid_argument("Invalid FilteredTracker configuration");
    }

    if (config_.enable_debug_output) {
        std::cout << "=== FilteredTracker Constructor (Pluggable Architecture) ===" << std::endl;
    }

    // Initialize non-pluggable components
    transform_manager_ = std::make_unique<CoordinateTransformManager>();
    motion_estimator_ = std::make_unique<VIOMotionEstimator>();

    // Initialize timing
    last_prediction_time_ = std::chrono::steady_clock::now();

    // Initialize last measurement result
    last_measurement_result_.success = false;
    last_measurement_result_.map_to_vio_transform = Eigen::Matrix4d::Identity();
    last_measurement_result_.confidence = 0.0;

    if (config_.enable_debug_output) {
        std::cout << "FilteredTracker initialized with pluggable architecture" << std::endl;
        std::cout << "Measurement interval: " << config_.measurement_interval_seconds << " seconds" << std::endl;
    }
}

// ============================================================================
// VIO Integration
// ============================================================================

void FilteredTracker::updateVIOCameraPose(const Eigen::Matrix4d& T_vio_from_camera) {
    // Delegate to motion estimator and transform manager
    motion_estimator_->updateVIOCameraPose(T_vio_from_camera);
    transform_manager_->updateVIOCameraPose(T_vio_from_camera);
}

// ============================================================================
// Prediction Step (VIO Integration)
// ============================================================================

void FilteredTracker::predictStep() {
    auto current_time = std::chrono::steady_clock::now();
    double dt = std::chrono::duration<double>(current_time - last_prediction_time_).count();
    last_prediction_time_ = current_time;

    if (!filter_strategy_->isInitialized() || dt <= 0) {
        return;
    }

    // Get motion from VIO estimator
    Eigen::Matrix4d motion = motion_estimator_->getMotionDelta();

    // Perform prediction using pluggable filter strategy
    filter_strategy_->predict(motion, dt, config_);

    if (config_.enable_debug_output) {
        std::cout << "Prediction step: dt=" << dt << "s, uncertainty="
                  << filter_strategy_->getPositionUncertainty() << std::endl;
    }
}

// ============================================================================
// LAR Measurement Update
// ============================================================================

FilteredTracker::MeasurementResult FilteredTracker::measurementUpdate(
    cv::InputArray image,
    const Frame& frame,
    double query_x,
    double query_z,
    double query_diameter) {

    if (config_.enable_debug_output) {
        std::cout << "=== FilteredTracker::measurementUpdate (Pluggable) ===" << std::endl;
    }

    MeasurementResult result;
    result.success = false;
    result.confidence = 0.0;

    // Update VIO pose from frame
    updateVIOCameraPose(frame.extrinsics);

    // Perform LAR localization (returns camera pose in LAR world)
    Eigen::Matrix4d T_lar_from_camera_measured;
    bool localization_success = base_tracker_->localize(
        image, frame, query_x, query_z, query_diameter, T_lar_from_camera_measured);

    if (!localization_success) {
        if (config_.enable_debug_output) {
            std::cout << "LAR localization failed" << std::endl;
        }
        return result;
    }

    // Store measurement results
    result.success = true;
    result.matched_landmarks = base_tracker_->local_landmarks;
    result.inliers = base_tracker_->inliers;

    if (config_.enable_debug_output) {
        std::cout << "LAR localization successful: " << result.inliers.size() << " inliers, "
                  << result.matched_landmarks.size() << " matches" << std::endl;
    }

    // Calculate confidence using pluggable estimator
    result.confidence = confidence_estimator_->calculateConfidence(result.inliers, T_lar_from_camera_measured, config_);

    if (!filter_strategy_->isInitialized()) {
        // Initialize filter from first measurement
        if (config_.enable_debug_output) {
            std::cout << "Initializing filter strategy from first LAR measurement" << std::endl;
        }
        filter_strategy_->initialize(T_lar_from_camera_measured, config_);
        // Keep the calculated confidence from the estimator
    } else {
        // Check for outliers using pluggable detector
        Eigen::Matrix4d predicted_pose = filter_strategy_->getState().toTransform();
        Eigen::MatrixXd covariance = filter_strategy_->getCovariance();

        if (outlier_detector_->isOutlier(T_lar_from_camera_measured, predicted_pose, covariance, result.confidence, config_)) {
            if (config_.enable_debug_output) {
                std::cout << "Rejecting camera pose measurement (outlier detection)" << std::endl;
            }
            // Return last known transform for rejected measurements
            result.map_to_vio_transform = last_measurement_result_.map_to_vio_transform;
            return result;
        }

        // Perform measurement update using pluggable filter strategy
        Eigen::MatrixXd measurement_noise = confidence_estimator_->calculateMeasurementNoise(
            result.inliers, T_lar_from_camera_measured, result.confidence, config_);
        filter_strategy_->update(T_lar_from_camera_measured, measurement_noise, config_);
    }

    // Compute coordinate transform using FILTERED camera pose estimate
    Eigen::Matrix4d T_lar_from_camera_filtered = filter_strategy_->getState().toTransform();
    auto transform_result = transform_manager_->computeTransform(T_lar_from_camera_filtered, result.confidence);
    result.map_to_vio_transform = transform_result.map_to_vio_transform;

    // Store result for future reference
    last_measurement_result_ = result;

    if (config_.enable_debug_output) {
        std::cout << "Measurement update completed successfully" << std::endl;
    }

    return result;
}

// ============================================================================
// State Access Methods
// ============================================================================

Eigen::Matrix4d FilteredTracker::getFilteredTransform() const {
    return transform_manager_->getCurrentTransform();
}

double FilteredTracker::getPositionUncertainty() const {
    if (!filter_strategy_->isInitialized()) {
        return std::numeric_limits<double>::infinity();
    }
    return filter_strategy_->getPositionUncertainty();
}

bool FilteredTracker::isInitialized() const {
    return filter_strategy_->isInitialized();
}

void FilteredTracker::reset() {
    if (config_.enable_debug_output) {
        std::cout << "Resetting FilteredTracker (all components)" << std::endl;
    }

    filter_strategy_->reset();
    transform_manager_->reset();
    motion_estimator_->reset();

    last_measurement_result_.success = false;
    last_measurement_result_.map_to_vio_transform = Eigen::Matrix4d::Identity();
    last_measurement_result_.confidence = 0.0;

    last_prediction_time_ = std::chrono::steady_clock::now();
}

} // namespace lar