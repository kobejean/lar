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
#include "lar/tracking/measurement_context.h"
#include <iostream>
#include <cmath>

namespace lar {

// ============================================================================
// Constructors
// ============================================================================

FilteredTracker::FilteredTracker(std::unique_ptr<Tracker> tracker, double measurement_interval)
    : FilteredTracker(std::move(tracker), [measurement_interval]() {
        FilteredTrackerConfig config;
        config.measurement_interval_seconds = measurement_interval;
        return config;
    }()) {
}

FilteredTracker::FilteredTracker(std::unique_ptr<Tracker> tracker, const FilteredTrackerConfig& config)
    : FilteredTracker(std::move(tracker),
                     config,
                     config.createFilterStrategy(),
                     std::make_unique<ReprojectionBasedConfidenceEstimator>(),
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

    // Initialize VIO motion estimation state
    last_vio_camera_pose_ = Eigen::Matrix4d::Identity();
    current_vio_camera_pose_ = Eigen::Matrix4d::Identity();
    has_vio_poses_ = false;

    // Initialize coordinate transform state
    last_transform_result_.success = false;
    last_transform_result_.map_to_vio_transform = Eigen::Matrix4d::Identity();
    last_transform_result_.confidence = 0.0;

    // Initialize timing
    last_prediction_time_ = std::chrono::steady_clock::now();

    if (config_.enable_debug_output) {
        std::cout << "FilteredTracker initialized with pluggable architecture" << std::endl;
        std::cout << "Measurement interval: " << config_.measurement_interval_seconds << " seconds" << std::endl;
    }
}

// ============================================================================
// VIO Integration
// ============================================================================

void FilteredTracker::updateVIOCameraPose(const Eigen::Matrix4d& T_vio_from_camera) {
    if (!utils::TransformUtils::validateTransformMatrix(T_vio_from_camera, "T_vio_from_camera")) {
        std::cout << "WARNING: Invalid VIO camera pose provided to FilteredTracker" << std::endl;
        return;
    }

    // Update VIO pose tracking
    if (!has_vio_poses_) {
        // First pose - initialize both current and last
        last_vio_camera_pose_ = T_vio_from_camera;
        current_vio_camera_pose_ = T_vio_from_camera;
        has_vio_poses_ = true;

        if (config_.enable_debug_output) {
            std::cout << "FilteredTracker: First VIO pose received" << std::endl;
        }
    } else {
        // Update poses: current becomes last, new pose becomes current
        last_vio_camera_pose_ = current_vio_camera_pose_;
        current_vio_camera_pose_ = T_vio_from_camera;
    }
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

    // Get motion from VIO poses
    Eigen::Matrix4d motion = computeMotionDelta();

    // Perform prediction using pluggable filter strategy
    filter_strategy_->predict(motion, dt, config_);
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

    // Perform LAR localization (returns camera pose in LAR world)
    Eigen::Matrix4d T_lar_from_camera_measured;
    bool localization_success;

    if (filter_strategy_->isInitialized()) {
        // Use pose prediction as initial guess for RANSAC
        Eigen::Matrix4d predicted_pose = filter_strategy_->getState().toTransform();

        if (config_.enable_debug_output) {
            std::cout << "Using prediction as initial guess for PnP" << std::endl;
        }

        localization_success = base_tracker_->localize(
            image, frame, query_x, query_z, query_diameter, T_lar_from_camera_measured, predicted_pose, true);
    } else {
        // No prediction available, use standard localization
        localization_success = base_tracker_->localize(
            image, frame, query_x, query_z, query_diameter, T_lar_from_camera_measured);
    }

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

    // Create measurement context for pluggable components
    MeasurementContext context;
    context.inliers = std::make_shared<std::vector<std::pair<Landmark*, cv::KeyPoint>>>(result.inliers);
    context.total_matches = base_tracker_->matches.size();
    context.frame = &frame;
    context.measured_pose = T_lar_from_camera_measured;
    if (filter_strategy_->isInitialized()) {
        context.predicted_pose = filter_strategy_->getState().toTransform();
    }

    // Calculate confidence using context
    result.confidence = confidence_estimator_->calculateConfidence(context, config_);
    context.confidence = result.confidence; // Update context with calculated confidence

    // Calculate measurement noise using context (compute once, use many times)
    context.measurement_noise = confidence_estimator_->calculateMeasurementNoise(context, config_);

    if (!filter_strategy_->isInitialized()) {
        // Initialize filter from first measurement
        if (config_.enable_debug_output) {
            std::cout << "Initializing filter strategy from first LAR measurement" << std::endl;
        }
        filter_strategy_->initialize(context, config_);
        // Keep the calculated confidence from the estimator
    } else {
        // Check for outliers using pluggable detector
        Eigen::Matrix4d predicted_pose = filter_strategy_->getState().toTransform();
        Eigen::MatrixXd covariance = filter_strategy_->getCovariance();

        // Use enhanced outlier analysis with pattern detection
        auto outlier_result = outlier_detector_->analyzeOutlier(T_lar_from_camera_measured, predicted_pose, covariance, result.confidence, config_);

        if (outlier_result.likely_bad_state) {
            if (config_.enable_debug_output) {
                std::cout << "Pattern analysis suggests bad filter state - resetting filter (consecutive rejections: "
                          << outlier_result.consecutive_rejections << ")" << std::endl;
            }

            // Reset filter and reinitialize with current measurement
            filter_strategy_->reset();
            filter_strategy_->initialize(context, config_);

            // Reset outlier detection counter
            outlier_result.is_outlier = false; // Accept this measurement after reset
        }

        if (outlier_result.is_outlier && !outlier_result.likely_bad_state) {
            if (config_.enable_debug_output) {
                std::cout << "Rejecting camera pose measurement (outlier detection)" << std::endl;
            }
            // Return last known transform for rejected measurements
            result.map_to_vio_transform = last_transform_result_.map_to_vio_transform;
            return result;
        }

        // Continue processing (either good measurement or recovered from bad state)

        // Perform measurement update using pluggable filter strategy with context
        // Note: measurement_noise is already calculated and stored in context
        filter_strategy_->update(context, config_);
    }

    // Compute coordinate transform using FILTERED camera pose estimate
    Eigen::Matrix4d T_lar_from_camera_filtered = filter_strategy_->getState().toTransform();
    MeasurementResult transform_result = computeCoordinateTransform(T_lar_from_camera_filtered, result.confidence);
    result.map_to_vio_transform = transform_result.map_to_vio_transform;

    // Store result for future reference (keeping legacy name for compatibility)
    last_transform_result_ = result;

    if (config_.enable_debug_output) {
        std::cout << "Measurement update completed successfully" << std::endl;
    }

    return result;
}

// ============================================================================
// State Access Methods
// ============================================================================

Eigen::Matrix4d FilteredTracker::getFilteredTransform() const {
	return current_vio_camera_pose_ * filter_strategy_->getState().toTransform().inverse();
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

    // Reset VIO motion estimation state
    has_vio_poses_ = false;
    last_vio_camera_pose_ = Eigen::Matrix4d::Identity();
    current_vio_camera_pose_ = Eigen::Matrix4d::Identity();

    // Reset coordinate transform state
    last_transform_result_.success = false;
    last_transform_result_.map_to_vio_transform = Eigen::Matrix4d::Identity();
    last_transform_result_.confidence = 0.0;

    last_prediction_time_ = std::chrono::steady_clock::now();
}

// ============================================================================
// Helper Methods (merged from VIOMotionEstimator and CoordinateTransformManager)
// ============================================================================

Eigen::Matrix4d FilteredTracker::computeMotionDelta() {
    if (!has_vio_poses_) {
        if (config_.enable_debug_output) {
            std::cout << "WARNING: No VIO poses available for motion computation" << std::endl;
        }
        return Eigen::Matrix4d::Identity();
    }

    static int motion_call_count = 0;
    motion_call_count++;

    // Compute relative camera motion in camera's own frame
    // We need: T_camera_current_from_camera_last
    // This is: T_camera_from_vio_last * T_vio_from_camera_current
    Eigen::Matrix4d motion_delta = last_vio_camera_pose_.inverse() * current_vio_camera_pose_;

    // Log every 20th call to reduce spam
    if (motion_call_count % 20 == 0) {
        std::cout << "VIO motion delta [call " << motion_call_count << "]: [" << motion_delta(0,3) << ", " << motion_delta(1,3) << ", " << motion_delta(2,3) << "]" << std::endl;
    }

    if (!utils::TransformUtils::validateTransformMatrix(motion_delta, "motion_delta")) {
        std::cout << "WARNING: Computed motion delta is invalid, returning identity" << std::endl;
        return Eigen::Matrix4d::Identity();
    }

    return motion_delta;
}

FilteredTracker::MeasurementResult FilteredTracker::computeCoordinateTransform(
    const Eigen::Matrix4d& T_lar_from_camera, double confidence) {

    MeasurementResult result;
    result.success = false;
    result.confidence = confidence;

    if (!has_vio_poses_) {
        std::cout << "ERROR: No VIO camera pose available for transform computation" << std::endl;
        return result;
    }

    if (!utils::TransformUtils::validateTransformMatrix(T_lar_from_camera, "T_lar_from_camera")) {
        std::cout << "ERROR: Invalid LAR camera pose" << std::endl;
        return result;
    }

    // CRITICAL: Use synchronized poses to compute coordinate transform
    // We have:
    // - T_lar_from_camera: Camera pose in LAR world (filtered estimate)
    // - current_vio_camera_pose_: Camera pose from VIO (synchronized with measurement)
    //
    // We want T_vio_from_lar such that:
    // T_lar_from_camera = T_lar_from_vio * T_vio_from_camera
    // Therefore: T_lar_from_vio = T_lar_from_camera * T_vio_from_camera^-1
    // But we need the reverse transform: T_vio_from_lar = T_lar_from_vio^-1
    // So: map_to_vio_transform = (T_lar_from_camera * T_vio_from_camera^-1)^-1
    //                          = T_vio_from_camera * T_lar_from_camera^-1

    result.map_to_vio_transform = current_vio_camera_pose_ * T_lar_from_camera.inverse();

    if (!utils::TransformUtils::validateTransformMatrix(result.map_to_vio_transform, "map_to_vio_transform")) {
        std::cout << "ERROR: Computed transform is invalid" << std::endl;
        return result;
    }

    result.success = true;

    if (config_.enable_debug_output) {
        std::cout << "Transform computed successfully (confidence: " << confidence << ")" << std::endl;
    }

    return result;
}

} // namespace lar
