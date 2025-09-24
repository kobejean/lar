//
// ExtendedKalmanFilter.cpp - Extended Kalman Filter implementation
//

#include "lar/tracking/pose_filtering/extended_kalman_filter.h"
#include "lar/tracking/filtered_tracker_config.h"
#include <iostream>
#include <cmath>

namespace lar {

void ExtendedKalmanFilter::initialize(const Eigen::Matrix4d& initial_pose, const FilteredTrackerConfig& config) {
    state_.fromTransform(initial_pose);

    // Initialize covariance
    covariance_ = Eigen::MatrixXd::Identity(6, 6);
    covariance_.block<3,3>(0,0) *= config.initial_position_uncertainty * config.initial_position_uncertainty;
    covariance_.block<3,3>(3,3) *= config.initial_orientation_uncertainty * config.initial_orientation_uncertainty;

    is_initialized_ = true;

    // Initialize anchor constraints timing
    start_time_ = std::chrono::steady_clock::now();

    if (config.enable_debug_output) {
        std::cout << "ExtendedKalmanFilter initialized with pose:" << std::endl;
        std::cout << "  Position: [" << state_.position.transpose() << "]" << std::endl;
        std::cout << "  Orientation: [" << state_.orientation.transpose() << "]" << std::endl;
    }
}

void ExtendedKalmanFilter::predict(const Eigen::Matrix4d& motion, double dt, const FilteredTrackerConfig& config) {
    if (!is_initialized_) {
        return;
    }

    // Apply motion to current state
    Eigen::Matrix4d T_current = state_.toTransform();
    Eigen::Matrix4d T_predicted = T_current * motion;
    state_.fromTransform(T_predicted);

    // Compute process noise based on motion magnitude and time
    Eigen::Vector3d motion_translation = motion.block<3,1>(0,3);
    double motion_magnitude = motion_translation.norm();

    // Process noise scales with motion and time
    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(6, 6);
    Q.block<3,3>(0,0) *= (config.motion_position_noise_scale * motion_magnitude + config.time_position_noise_scale * dt);
    Q.block<3,3>(3,3) *= (config.motion_orientation_noise_scale * motion_magnitude + config.time_orientation_noise_scale * dt);

    // Covariance prediction: P = P + Q
    covariance_ = covariance_ + Q;
}

void ExtendedKalmanFilter::update(const MeasurementContext& context,
                                 const Eigen::MatrixXd& measurement_noise,
                                 const FilteredTrackerConfig& config) {
    const Eigen::Matrix4d& measurement = context.measured_pose;
    if (!is_initialized_) {
        return;
    }

    // Convert measurement to state vector
    PoseState measured_state;
    measured_state.fromTransform(measurement);
    Eigen::VectorXd z = measured_state.toVector();

    // Current state prediction
    Eigen::VectorXd x = state_.toVector();

    // Innovation (measurement residual)
    Eigen::VectorXd y = z - x;

    // Innovation covariance: S = H*P*H' + R
    // Since H = I (direct measurement), S = P + R
    Eigen::MatrixXd S = covariance_ + measurement_noise;

    // Kalman gain: K = P*H'*S^-1 = P*S^-1
    Eigen::MatrixXd K = covariance_ * S.inverse();

    // State update: x = x + K*y
    Eigen::VectorXd x_updated = x + K * y;
    state_.fromVector(x_updated);

    // Covariance update: P = (I - K*H)*P = (I - K)*P
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(6, 6);
    covariance_ = (I - K) * covariance_;

    if (config.enable_debug_output) {
        std::cout << "ExtendedKalmanFilter updated with measurement" << std::endl;
        std::cout << "  Innovation norm: " << y.norm() << std::endl;
        std::cout << "  Position uncertainty: " << getPositionUncertainty() << std::endl;
    }
}

double ExtendedKalmanFilter::getPositionUncertainty() const {
    return covariance_.block<3,3>(0,0).trace();
}

void ExtendedKalmanFilter::reset() {
    is_initialized_ = false;
    state_.position = Eigen::Vector3d::Zero();
    state_.orientation = Eigen::Vector3d::Zero();
    covariance_ = Eigen::MatrixXd::Identity(6, 6);

    // Reset anchor constraints
    anchor_history_.clear();
    start_time_ = std::chrono::steady_clock::now();
}

// ============================================================================
// Historical Anchor Constraints Implementation
// ============================================================================

void ExtendedKalmanFilter::updateWithAnchors(const MeasurementContext& context,
                                            const Eigen::MatrixXd& measurement_noise,
                                            const FilteredTrackerConfig& config) {
    // First, do standard EKF update
    update(context, measurement_noise, config);

    // Check if this measurement qualifies as an anchor
    if (shouldAddAnchor(context.measured_pose, context.confidence, context.inliers ? context.inliers->size() : 0, config)) {
        addAnchor(context.measured_pose, context.confidence, context.inliers ? context.inliers->size() : 0, config);
    }

    // Apply rotational constraints from historical anchors
    applyAnchorConstraints(config);
}

bool ExtendedKalmanFilter::shouldAddAnchor(const Eigen::Matrix4d& measurement,
                                          double confidence,
                                          size_t observation_count,
                                          const FilteredTrackerConfig& config) const {
    // High quality localization required
    if (confidence < config.anchor_min_confidence) return false;

    // Sufficient features tracked
    if (observation_count < config.anchor_min_features) return false;

    // Sufficient spacing from other anchors
    Eigen::Vector3d pos = measurement.block<3,1>(0,3);
    for (const auto& anchor : anchor_history_) {
        if ((pos - anchor.position).norm() < config.anchor_min_spacing) {
            return false;
        }
    }

    return true;
}

void ExtendedKalmanFilter::addAnchor(const Eigen::Matrix4d& measurement,
                                    double confidence,
                                    size_t observation_count,
                                    const FilteredTrackerConfig& config) {
    double timestamp = getCurrentTimestamp();

    anchor_history_.emplace_back(measurement, confidence, timestamp, observation_count);

    // Maintain maximum size
    if (anchor_history_.size() > config.anchor_max_count) {
        anchor_history_.pop_front();
    }

    if (config.enable_debug_output) {
        std::cout << "Added anchor pose (confidence=" << confidence
                  << ", features=" << observation_count
                  << ", total_anchors=" << anchor_history_.size() << ")" << std::endl;
    }
}

void ExtendedKalmanFilter::applyAnchorConstraints(const FilteredTrackerConfig& config) {
    if (anchor_history_.empty() || !config.enable_anchor_constraints) return;

    Eigen::Vector3d current_position = state_.position;

    // Find relevant anchors (nearby and not too old)
    std::vector<std::pair<const AnchorPose*, double>> relevant_anchors;
    double current_time = getCurrentTimestamp();

    for (const auto& anchor : anchor_history_) {
        double distance = (current_position - anchor.position).norm();
        double age = current_time - anchor.timestamp;

        // Skip if too far or too old
        if (distance > config.anchor_max_distance || age > config.anchor_max_age) {
            continue;
        }

        // Calculate relevance weight
        double distance_weight = std::exp(-distance / config.anchor_distance_scale);
        double age_weight = std::exp(-age / config.anchor_time_scale);
        double confidence_weight = anchor.confidence;
        double feature_weight = std::sqrt(anchor.observation_count / 50.0); // normalize by 50 features

        double relevance = distance_weight * age_weight * confidence_weight * feature_weight;

        if (relevance > config.anchor_min_relevance) {
            relevant_anchors.push_back({&anchor, relevance});
        }
    }

    if (relevant_anchors.empty()) return;

    // Compute weighted rotational correction
    Eigen::Vector3d rotation_correction = Eigen::Vector3d::Zero();
    double total_weight = 0.0;

    for (const auto& [anchor, relevance] : relevant_anchors) {
        // Extract rotations
        Eigen::Matrix3d anchor_rotation = anchor->transform.block<3,3>(0,0);
        Eigen::Matrix3d current_rotation = state_.toTransform().block<3,3>(0,0);

        // Compute rotation difference (anchor relative to current)
        Eigen::Matrix3d rotation_error = anchor_rotation * current_rotation.transpose();
        Eigen::Vector3d axis_angle_error = utils::TransformUtils::rotationMatrixToAxisAngle(rotation_error);

        rotation_correction += relevance * axis_angle_error;
        total_weight += relevance;
    }

    if (total_weight > 0) {
        rotation_correction /= total_weight;

        // Limit correction magnitude to prevent jumps
        double correction_magnitude = rotation_correction.norm();
        if (correction_magnitude > config.anchor_max_correction) {
            rotation_correction *= config.anchor_max_correction / correction_magnitude;
        }

        // Apply soft correction to state
        state_.orientation += config.anchor_correction_strength * rotation_correction;

        // Reduce orientation covariance (we're more confident now)
        double confidence_boost = 1.0 - config.anchor_correction_strength * 0.3;
        covariance_.block<3,3>(3,3) *= confidence_boost;

        if (config.enable_debug_output) {
            std::cout << "Applied anchor constraint:" << std::endl;
            std::cout << "  Rotation correction: " << rotation_correction.transpose()
                     << " (magnitude: " << correction_magnitude << ")" << std::endl;
            std::cout << "  Used " << relevant_anchors.size() << " anchors" << std::endl;
        }
    }
}

double ExtendedKalmanFilter::getCurrentTimestamp() const {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(now - start_time_).count();
}

} // namespace lar