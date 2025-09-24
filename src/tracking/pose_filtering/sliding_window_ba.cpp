//
// SlidingWindowBA.cpp - Sliding Window Bundle Adjustment implementation
//

#include "lar/tracking/pose_filtering/sliding_window_ba.h"
#include "lar/tracking/filtered_tracker_config.h"
#include "lar/core/landmark.h"
#include "lar/core/utils/transform.h"
#include "lar/mapping/frame.h"

// g2o includes
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/slam3d/types_slam3d.h>

#include <iostream>
#include <cmath>

namespace lar {

// ============================================================================
// Constructor/Destructor
// ============================================================================

SlidingWindowBA::SlidingWindowBA() {
    // Initialize g2o optimizer
    auto linearSolver = g2o::make_unique<g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>>();
    auto blockSolver = g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver));
    auto algorithm = new g2o::OptimizationAlgorithmLevenberg(std::move(blockSolver));

    optimizer_ = std::make_unique<g2o::SparseOptimizer>();
    optimizer_->setAlgorithm(algorithm);
    optimizer_->setVerbose(true);  // Enable debug output

    current_covariance_ = Eigen::MatrixXd::Identity(6, 6) * 0.01;  // Small initial uncertainty
}

SlidingWindowBA::~SlidingWindowBA() = default;

// ============================================================================
// PoseFilterStrategy Interface
// ============================================================================

void SlidingWindowBA::initialize(const MeasurementContext& context, const FilteredTrackerConfig& config) {
    // Set configuration from config
    max_window_size_ = config.sliding_window_max_keyframes;
    keyframe_distance_ = config.sliding_window_keyframe_distance;  // meters
    keyframe_angle_ = config.sliding_window_keyframe_angle * M_PI / 180.0;  // Convert degrees to radians
    min_observations_ = config.sliding_window_min_observations;
    optimization_iterations_ = config.sliding_window_optimization_iterations;

    // Extract camera intrinsics from frame - must be present
    if (!context.frame) {
        throw std::invalid_argument("SlidingWindowBA::initialize requires frame in MeasurementContext");
    }

    if (config.enable_debug_output) {
        const Eigen::Matrix3d& K = context.frame->intrinsics;
        std::cout << "SlidingWindowBA camera intrinsics: fx=" << K(0,0)
                  << ", fy=" << K(1,1) << ", cx=" << K(0,2) << ", cy=" << K(1,2) << std::endl;
    }

    // Initialize state from measurement context
    current_state_.fromTransform(context.measured_pose);

    // Use measurement noise from context as initial covariance
    current_covariance_ = context.measurement_noise;

    // Create first keyframe directly from context
    auto kf = std::make_shared<Keyframe>(next_keyframe_id_++, context);
    keyframes_.push_back(kf);

    std::cout << "Created keyframe " << kf->id << " with confidence=" << kf->confidence
              << ", intrinsics [fx=" << kf->intrinsics(0,0) << ", fy=" << kf->intrinsics(1,1)
              << ", cx=" << kf->intrinsics(0,2) << ", cy=" << kf->intrinsics(1,2)
              << "] (total: " << keyframes_.size() << ")" << std::endl;

    // Reset motion accumulator
    accumulated_motion_ = Eigen::Matrix4d::Identity();

    initialized_ = true;

    std::cout << "SlidingWindowBA initialized with first keyframe" << std::endl;
}

void SlidingWindowBA::predict(const Eigen::Matrix4d& motion, double dt, const FilteredTrackerConfig& config) {
    if (!initialized_) return;

    static int predict_call_count = 0;
    predict_call_count++;

    // Log every 20th call to reduce spam
    bool should_log = (predict_call_count % 40 == 0);

    if (should_log) {
        std::cout << "SlidingWindowBA::predict() [call " << predict_call_count << "]" << std::endl;
        std::cout << "  Motion translation: [" << motion(0,3) << ", " << motion(1,3) << ", " << motion(2,3) << "]" << std::endl;
    }

    // Apply motion to current state
    Eigen::Matrix4d current_pose = current_state_.toTransform();
    current_pose = current_pose * motion;
    current_state_.fromTransform(current_pose);

    // Accumulate motion since last keyframe
    accumulated_motion_ = accumulated_motion_ * motion;

    if (should_log) {
        std::cout << "  Accumulated motion: [" << accumulated_motion_(0,3) << ", " << accumulated_motion_(1,3) << ", " << accumulated_motion_(2,3) << "]" << std::endl;
    }

    // Propagate uncertainty using motion model
    // Simple approach: add process noise based on motion magnitude
    Eigen::Vector3d translation = motion.block<3,1>(0,3);
    double motion_magnitude = translation.norm();

    // Process noise scaled by motion and time
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(6, 6);
    Q.block<3,3>(0,0) = Eigen::Matrix3d::Identity() *
        (config.motion_position_noise_scale * motion_magnitude + config.time_position_noise_scale * dt);
    Q.block<3,3>(3,3) = Eigen::Matrix3d::Identity() *
        (config.motion_orientation_noise_scale * motion_magnitude + config.time_orientation_noise_scale * dt);

    current_covariance_ += Q;  // Add process noise directly (Q is already a covariance matrix)
}

void SlidingWindowBA::update(const MeasurementContext& context,
                              const FilteredTrackerConfig& config) {
    const Eigen::Matrix4d& measurement = context.measured_pose;
    std::cout << "SlidingWindowBA::update() called" << std::endl;
    std::cout << "  Measurement position: [" << measurement(0,3) << ", " << measurement(1,3) << ", " << measurement(2,3) << "]" << std::endl;
    std::cout << "  Current prediction: [" << current_state_.position.x() << ", " << current_state_.position.y() << ", " << current_state_.position.z() << "]" << std::endl;

    if (!initialized_) {
        initialize(context, config);
        return;
    }

    // Check if we should create a new keyframe directly from context
    if (shouldCreateKeyframe(context)) {
        // Track landmark observations
        if (context.inliers) {
            for (const auto& [landmark, keypoint] : *context.inliers) {
                landmark_observations_[landmark]++;
                active_landmarks_.insert(landmark);
            }
        }

        // Create keyframe directly from context
        auto kf = std::make_shared<Keyframe>(next_keyframe_id_++, context);
        keyframes_.push_back(kf);

        std::cout << "Created keyframe " << kf->id << " with confidence=" << kf->confidence
                  << ", intrinsics [fx=" << kf->intrinsics(0,0) << ", fy=" << kf->intrinsics(1,1)
                  << ", cx=" << kf->intrinsics(0,2) << ", cy=" << kf->intrinsics(1,2)
                  << "] (total: " << keyframes_.size() << ")" << std::endl;

        // Marginalize old keyframes if window is full
        while (keyframes_.size() > max_window_size_) {
            marginalizeOldestKeyframe();
        }

        // Reset motion accumulator after creating keyframe
        accumulated_motion_ = Eigen::Matrix4d::Identity();

        // Run bundle adjustment optimization after every measurement if we have multiple keyframes
        if (keyframes_.size() >= 2) {
            buildOptimizationGraph(config);
            optimize();
            updateStateFromOptimization();
        }
    } else {
        // Just update current state without creating keyframe
        // Calculate measurement weight based on relative covariances
        // Higher measurement noise → lower measurement weight (trust prediction more)
        // Higher prediction uncertainty → higher measurement weight (trust measurement more)
        double measurement_noise_trace = context.measurement_noise.trace();
        double prediction_uncertainty_trace = current_covariance_.trace();

        // Weight based on inverse of uncertainties (more certain source gets higher weight)
        double prediction_weight = 1.0 / (prediction_uncertainty_trace + 1e-6);
        double measurement_weight = 1.0 / (measurement_noise_trace + 1e-6);

        // Normalize weights to sum to 1
        double total_weight = prediction_weight + measurement_weight;
        measurement_weight = measurement_weight / total_weight;

        // Use proper SE(3) pose interpolation
        Eigen::Matrix4d predicted_pose = current_state_.toTransform();
        Eigen::Matrix4d updated_pose = utils::TransformUtils::interpolatePose(
            predicted_pose, measurement, measurement_weight);
        current_state_.fromTransform(updated_pose);

        // Update covariance using proper information matrix fusion
        // Convert covariances to information matrices for proper combination
        Eigen::MatrixXd prediction_info = calculateInformationMatrix(current_covariance_);
        Eigen::MatrixXd measurement_info = calculateInformationMatrix(context.measurement_noise);

        // Weight information matrices (higher weight = more trusted source)
        double prediction_info_weight = 1.0 - measurement_weight;  // Complementary weight
        Eigen::MatrixXd fused_info = prediction_info_weight * prediction_info + measurement_weight * measurement_info;

        // Convert back to covariance
        current_covariance_ = calculateInformationMatrix(fused_info);
    }

    std::cout << "SlidingWindowBA state update:" << std::endl;
    std::cout << "  Accumulated motion:" << std::endl;
    std::cout << "    Translation: [" << accumulated_motion_(0,3) << ", " << accumulated_motion_(1,3) << ", " << accumulated_motion_(2,3) << "]" << std::endl;
    std::cout << "  Final state pose:" << std::endl;
    Eigen::Matrix4d final_pose = current_state_.toTransform();
    std::cout << "    Position: [" << final_pose(0,3) << ", " << final_pose(1,3) << ", " << final_pose(2,3) << "]" << std::endl;
    std::cout << "  Real BA Covariance:" << std::endl;
    std::cout << "    Position uncertainty: ["
                << sqrt(current_covariance_(0,0)) << ", "
                << sqrt(current_covariance_(1,1)) << ", "
                << sqrt(current_covariance_(2,2)) << "] m" << std::endl;
    std::cout << "    Orientation uncertainty: ["
                << sqrt(current_covariance_(3,3)) * 180.0 / M_PI << "°, "
                << sqrt(current_covariance_(4,4)) * 180.0 / M_PI << "°, "
                << sqrt(current_covariance_(5,5)) * 180.0 / M_PI << "°]" << std::endl;
}

PoseState SlidingWindowBA::getState() const {
    return current_state_;
}

Eigen::MatrixXd SlidingWindowBA::getCovariance() const {
    return current_covariance_;
}

double SlidingWindowBA::getPositionUncertainty() const {
    // Return trace of position covariance
    return std::sqrt((current_covariance_(0,0) + current_covariance_(1,1) + current_covariance_(2,2)) / 3.0);
}

bool SlidingWindowBA::isInitialized() const {
    return initialized_;
}

void SlidingWindowBA::reset() {
    keyframes_.clear();
    landmark_observations_.clear();
    active_landmarks_.clear();
    pose_vertices_.clear();
    landmark_vertices_.clear();
    optimizer_->clear();
    accumulated_motion_ = Eigen::Matrix4d::Identity();
    next_keyframe_id_ = 0;
    initialized_ = false;
}


// ============================================================================
// Private Helper Methods
// ============================================================================

bool SlidingWindowBA::shouldCreateKeyframe(const MeasurementContext& context) const {
    if (keyframes_.empty()) return true;

    const Eigen::Matrix4d& current_pose = context.measured_pose;
    const auto& last_kf = keyframes_.back();

    // Check distance criterion
    Eigen::Vector3d translation_diff = current_pose.block<3,1>(0,3) - last_kf->pose.block<3,1>(0,3);
    if (translation_diff.norm() > keyframe_distance_) {
        return true;
    }

    // Check rotation criterion
    Eigen::Matrix3d R1 = last_kf->pose.block<3,3>(0,0);
    Eigen::Matrix3d R2 = current_pose.block<3,3>(0,0);
    Eigen::Matrix3d R_diff = R1.transpose() * R2;

    // Clamp to prevent acos domain errors due to numerical precision issues
    double cos_angle = (R_diff.trace() - 1.0) / 2.0;
    cos_angle = std::max(-1.0, std::min(1.0, cos_angle));
    double angle = std::acos(cos_angle);

    if (std::abs(angle) > keyframe_angle_) {
        return true;
    }

    // Check observation criterion
    if (context.inliers && context.inliers->size() >= min_observations_) {
        return true;
    }

    return false;
}


void SlidingWindowBA::marginalizeOldestKeyframe() {
    if (keyframes_.empty()) return;

    auto oldest = keyframes_.front();

    // Proper marginalization: extract marginal information before removal
    Eigen::MatrixXd marginal_info = extractMarginalInformation(oldest->id);

    // Apply prior from marginalized pose to connected keyframes
    if (keyframes_.size() > 1) {
        auto next_keyframe = keyframes_[1];
        applyMarginalPrior(next_keyframe->id, marginal_info, oldest->pose);
    }

    // Remove the keyframe from the window
    keyframes_.pop_front();

    // Clean up landmark observations
    if (oldest->inliers) {
        for (const auto& [landmark, keypoint] : *oldest->inliers) {
            auto it = landmark_observations_.find(landmark);
            if (it != landmark_observations_.end()) {
                it->second--;
                // Keep landmarks with observations in remaining keyframes
                if (it->second == 0) {
                    landmark_observations_.erase(it);
                    active_landmarks_.erase(landmark);
                }
            }
        }
    }

    std::cout << "Properly marginalized keyframe " << oldest->id
              << " (applied prior to keyframe " << (keyframes_.empty() ? -1 : keyframes_.front()->id) << ")" << std::endl;
}

void SlidingWindowBA::buildOptimizationGraph(const FilteredTrackerConfig& config) {
    // Clear previous graph
    optimizer_->clear();
    optimizer_->clearParameters();
    pose_vertices_.clear();
    landmark_vertices_.clear();

    std::cout << "Building optimization graph with " << keyframes_.size() << " keyframes" << std::endl;

    int vertex_id = 0;

    // Add pose vertices and camera parameters for each keyframe
    for (const auto& kf : keyframes_) {
        // Add pose vertex
        auto v = new g2o::VertexSE3Expmap();
        v->setId(vertex_id++);

        // Convert pose to g2o format using centralized function
        g2o::SE3Quat pose = lar::utils::TransformUtils::arkitToG2oPose(kf->pose);
        v->setEstimate(pose);

        optimizer_->addVertex(v);
        pose_vertices_[kf->id] = v;

        // Add camera parameters for this keyframe using per-frame intrinsics
        const Eigen::Matrix3d& K = kf->intrinsics;
        double fx = K(0,0), fy = K(1,1), cx = K(0,2), cy = K(1,2);

        Eigen::Vector2d principal_point(cx, cy);
        auto* cam_params = new g2o::CameraParameters(fx, principal_point, 0.0);  // baseline=0 for monocular
        cam_params->setId(kf->id);
        if (!optimizer_->addParameter(cam_params)) {
            // Camera parameters already exist - this should never happen with unique keyframe IDs
            delete cam_params;
            throw std::runtime_error("Camera parameters already exist for keyframe " + std::to_string(kf->id) +
                                   " - duplicate keyframe IDs detected");
        }
    }

    // Fix the keyframe with highest confidence (best measurement quality) for gauge freedom
    size_t best_keyframe_id = 0;
    double highest_confidence = -1.0;

    for (const auto& kf : keyframes_) {
        if (kf->confidence > highest_confidence) {
            highest_confidence = kf->confidence;
            best_keyframe_id = kf->id;
        }
    }

    // Fix the most confident keyframe
    auto best_vertex_it = pose_vertices_.find(best_keyframe_id);
    if (best_vertex_it != pose_vertices_.end()) {
        best_vertex_it->second->setFixed(true);
        std::cout << "Fixed keyframe " << best_keyframe_id << " (confidence: "
                  << highest_confidence << ") for gauge freedom" << std::endl;
    }

    // Add landmark vertices - no fixing needed since we fix one pose for gauge freedom

    for (const auto& landmark : active_landmarks_) {
        auto v = new g2o::VertexPointXYZ();
        v->setId(vertex_id++);

        Eigen::Vector3d position(landmark->position[0],
                                 landmark->position[1],
                                 landmark->position[2]);
        v->setEstimate(position);

        // Marginalize all landmarks for efficiency (no landmark fixing needed with pose-based gauge freedom)
        v->setMarginalized(true);

        optimizer_->addVertex(v);
        landmark_vertices_[landmark] = v;
    }

    // Add reprojection edges
    int edge_id = 0;
    for (const auto& kf : keyframes_) {
        auto pose_vertex = pose_vertices_[kf->id];

        if (kf->inliers) {
            for (const auto& [landmark, keypoint] : *kf->inliers) {
                auto lm_it = landmark_vertices_.find(landmark);
                if (lm_it == landmark_vertices_.end()) continue;

                auto landmark_vertex = lm_it->second;

                // Create edge - use the same type as bundle_adjustment.cpp
                auto edge = new g2o::EdgeProjectXYZ2UVD();
                edge->setId(edge_id++);
                edge->setVertex(0, landmark_vertex);  // 3D point
                edge->setVertex(1, pose_vertex);       // SE3 pose

                // Set measurement (u, v, depth=0 for monocular) - extract from keypoint
                Eigen::Vector3d measurement_with_depth(keypoint.pt.x, keypoint.pt.y, 0.0);
                edge->setMeasurement(measurement_with_depth);

                // Set information matrix using well-tuned parameters from bundle_adjustment.cpp
                // Use BASE_INFORMATION = 1.0 directly (not pixel_noise squared)
                // This matches the proven scaling in bundle_adjustment.cpp lines 413-442
                static constexpr double BASE_INFORMATION = 1.0;
                static constexpr double MIN_INFORMATION = 0.1;
                static constexpr double MAX_INFORMATION = 100.0;

                double xy_information = BASE_INFORMATION;

                // Clamp information to reasonable bounds (matching bundle_adjustment.cpp)
                xy_information = std::max(MIN_INFORMATION, std::min(MAX_INFORMATION, xy_information));

                // Set information matrix (u, v, depth=0 for monocular)
                Eigen::Vector3d information_diag(xy_information, xy_information, 0.0);
                edge->setInformation(information_diag.asDiagonal());

                // Add robust kernel matching bundle_adjustment.cpp
                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                rk->setDelta(sqrt(5.991));  // Chi-squared threshold for 2 DOF at 95% confidence
                edge->setRobustKernel(rk);

                // Link to camera parameters for this keyframe
                edge->setParameterId(0, kf->id);

                optimizer_->addEdge(edge);
            }
        }
    }

    // Add odometry edges between consecutive keyframes
    for (size_t i = 1; i < keyframes_.size(); ++i) {
        auto kf1 = keyframes_[i-1];
        auto kf2 = keyframes_[i];

        auto v1 = pose_vertices_[kf1->id];
        auto v2 = pose_vertices_[kf2->id];

        // Calculate relative transformation in ARKit coordinates
        Eigen::Matrix4d T_rel = kf1->pose.inverse() * kf2->pose;

        // Convert relative transform to g2o coordinates
        g2o::SE3Quat rel_measurement = lar::utils::TransformUtils::arkitToG2oPose(T_rel);

        // Create odometry edge
        auto edge = new g2o::EdgeSE3Expmap();
        edge->setId(edge_id++);
        edge->setVertex(0, v1);
        edge->setVertex(1, v2);
        edge->setMeasurement(rel_measurement);

        // Set information based on accumulated motion uncertainty
        Eigen::MatrixXd info = Eigen::MatrixXd::Identity(6, 6);
        info.block<3,3>(0,0) *= 100.0;  // Position weight
        info.block<3,3>(3,3) *= 100.0;  // Orientation weight
        edge->setInformation(info);

        optimizer_->addEdge(edge);
    }
}

void SlidingWindowBA::optimize() {
    std::cout << "=== SlidingWindowBA Optimization ===" << std::endl;
    std::cout << "Keyframes: " << keyframes_.size() << std::endl;
    std::cout << "Vertices: " << optimizer_->vertices().size() << std::endl;
    std::cout << "Edges: " << optimizer_->edges().size() << std::endl;

    optimizer_->initializeOptimization();
    std::cout << "Initial chi2: " << optimizer_->chi2() << std::endl;

    optimizer_->optimize(optimization_iterations_);

    std::cout << "Final chi2: " << optimizer_->chi2() << std::endl;
    std::cout << "=== End Optimization ===" << std::endl;
}

void SlidingWindowBA::updateStateFromOptimization() {
    // Update keyframe poses from optimization
    for (auto& kf : keyframes_) {
        auto it = pose_vertices_.find(kf->id);
        if (it != pose_vertices_.end()) {
            // Get optimized pose and convert back to ARKit coordinates
            g2o::SE3Quat optimized = it->second->estimate();
            kf->pose = lar::utils::TransformUtils::g2oToArkitPose(optimized);
        }
    }

    // Update current state from last keyframe
    if (!keyframes_.empty()) {
        const auto& last_kf = keyframes_.back();
        current_state_.fromTransform(last_kf->pose * accumulated_motion_);

        // Extract true covariance from bundle adjustment
        current_covariance_ = extractCovarianceFromBA(last_kf->id);

        std::cout << "SlidingWindowBA state update:" << std::endl;
        std::cout << "  Last keyframe pose:" << std::endl;
        std::cout << "    Position: [" << last_kf->pose(0,3) << ", " << last_kf->pose(1,3) << ", " << last_kf->pose(2,3) << "]" << std::endl;
    }
}

Eigen::MatrixXd SlidingWindowBA::extractCovarianceFromBA(size_t keyframe_id) const {
    // Find the vertex for this keyframe
    auto vertex_it = pose_vertices_.find(keyframe_id);
    if (vertex_it == pose_vertices_.end()) {
        std::cout << "ERROR: Cannot find pose vertex for keyframe " << keyframe_id
                  << " in optimization graph!" << std::endl;
        std::cout << "Available vertex IDs: ";
        for (const auto& [id, vertex] : pose_vertices_) {
            std::cout << id << " ";
        }
        std::cout << std::endl;

        // This indicates a serious bug - don't hide it with fallbacks
        throw std::runtime_error("Pose vertex not found in optimization graph - this should never happen");
    }

    auto target_vertex = vertex_it->second;

    // Try to extract true marginal covariance using g2o
    g2o::SparseBlockMatrix<Eigen::MatrixXd> spinv;
    if (optimizer_->computeMarginals(spinv, target_vertex)) {
        int vertex_idx = target_vertex->hessianIndex();
        if (vertex_idx >= 0) {
            auto block = spinv.block(vertex_idx, vertex_idx);
            if (block && block->rows() == 6 && block->cols() == 6) {
                std::cout << "Extracted true marginal covariance from BA" << std::endl;
                return *block;
            }
        }
    }

    // Fallback: residual-based estimate
    std::cout << "Marginal computation failed, using residual-based estimate" << std::endl;

    double total_error = 0.0;
    int edge_count = 0;

    for (auto edge : target_vertex->edges()) {
        // Try to cast to specific edge types to access error computation
        if (auto reprojection_edge = dynamic_cast<g2o::EdgeProjectXYZ2UVD*>(edge)) {
            reprojection_edge->computeError();
            auto error_vec = reprojection_edge->error();
            for (int i = 0; i < error_vec.size(); ++i) {
                total_error += error_vec[i] * error_vec[i];
            }
            edge_count++;
        } else if (auto odometry_edge = dynamic_cast<g2o::EdgeSE3Expmap*>(edge)) {
            odometry_edge->computeError();
            auto error_vec = odometry_edge->error();
            for (int i = 0; i < error_vec.size(); ++i) {
                total_error += error_vec[i] * error_vec[i];
            }
            edge_count++;
        }
        // Skip unknown edge types silently
    }

    if (edge_count > 0) {
        double rms_error = std::sqrt(total_error / edge_count);
        double error_scale = std::max(1.0, rms_error / 2.0);

        Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(6, 6);
        double pos_var = (0.05 * error_scale) * (0.05 * error_scale);  // 5cm base
        double rot_var = (0.02 * error_scale) * (0.02 * error_scale);  // ~1° base

        cov.block<3,3>(0,0) *= pos_var;
        cov.block<3,3>(3,3) *= rot_var;

        std::cout << "Using residual-based covariance (RMS: " << rms_error << ")" << std::endl;
        return cov;
    }

    // Final fallback - no edges found
    std::cout << "WARNING: No edges found for vertex, using conservative defaults" << std::endl;

    Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(6, 6);
    cov.block<3,3>(0,0) *= 0.2 * 0.2;  // Conservative 20cm
    cov.block<3,3>(3,3) *= (5.0 * M_PI/180.0) * (5.0 * M_PI/180.0);  // Conservative 5°
    return cov;
}

Eigen::MatrixXd SlidingWindowBA::calculateInformationMatrix(const Eigen::MatrixXd& covariance) const {
    // Add small regularization to ensure positive definiteness
    Eigen::MatrixXd regularized = covariance + Eigen::MatrixXd::Identity(6, 6) * 1e-6;
    return regularized.inverse();
}

// ============================================================================
// Marginalization Implementation
// ============================================================================

Eigen::MatrixXd SlidingWindowBA::extractMarginalInformation(size_t keyframe_id) const {
    // Find the pose vertex for this keyframe
    auto vertex_it = pose_vertices_.find(keyframe_id);
    if (vertex_it == pose_vertices_.end()) {
        std::cout << "WARNING: Could not find vertex for keyframe " << keyframe_id
                  << " in marginalization, using default uncertainty" << std::endl;

        // Return default information matrix
        Eigen::MatrixXd info = Eigen::MatrixXd::Identity(6, 6);
        info.block<3,3>(0,0) *= 10.0;  // Position: 10cm^-2
        info.block<3,3>(3,3) *= 100.0; // Orientation: ~6°^-2
        return info;
    }

    auto target_vertex = vertex_it->second;

    // Try to extract marginal covariance using g2o
    g2o::SparseBlockMatrix<Eigen::MatrixXd> spinv;
    if (optimizer_->computeMarginals(spinv, target_vertex)) {
        int vertex_idx = target_vertex->hessianIndex();
        if (vertex_idx >= 0) {
            auto block = spinv.block(vertex_idx, vertex_idx);
            if (block && block->rows() == 6 && block->cols() == 6) {
                // Return information matrix (inverse of covariance)
                Eigen::MatrixXd cov = *block;
                return calculateInformationMatrix(cov);
            }
        }
    }

    // Fallback: use information from connected edges
    double total_info = 0.0;
    int edge_count = 0;

    for (auto edge : target_vertex->edges()) {
        if (auto reprojection_edge = dynamic_cast<g2o::EdgeProjectXYZ2UVD*>(edge)) {
            // Get information from edge
            auto edge_info = reprojection_edge->information();
            total_info += edge_info.trace();
            edge_count++;
        } else if (auto odometry_edge = dynamic_cast<g2o::EdgeSE3Expmap*>(edge)) {
            auto edge_info = odometry_edge->information();
            total_info += edge_info.trace();
            edge_count++;
        }
    }

    // Create information matrix based on connectivity
    Eigen::MatrixXd info = Eigen::MatrixXd::Identity(6, 6);
    if (edge_count > 0) {
        double avg_info = total_info / edge_count;
        double scale = std::max(1.0, avg_info / 100.0); // Normalize to reasonable scale
        info *= scale;
    } else {
        // Conservative default
        info.block<3,3>(0,0) *= 5.0;  // Position
        info.block<3,3>(3,3) *= 50.0; // Orientation
    }

    std::cout << "Extracted marginal information for keyframe " << keyframe_id
              << " using " << edge_count << " edges" << std::endl;

    return info;
}

void SlidingWindowBA::applyMarginalPrior(size_t target_keyframe_id,
                                        const Eigen::MatrixXd& marginal_info,
                                        const Eigen::Matrix4d& marginalized_pose) {
    // Find the target keyframe
    std::shared_ptr<Keyframe> target_kf = nullptr;
    for (const auto& kf : keyframes_) {
        if (kf->id == target_keyframe_id) {
            target_kf = kf;
            break;
        }
    }

    if (!target_kf) {
        std::cout << "WARNING: Could not find target keyframe " << target_keyframe_id
                  << " for marginal prior application" << std::endl;
        return;
    }

    // Apply soft constraint: increase information about relative pose
    // This prevents drift by maintaining connection to marginalized pose

    // Calculate relative transformation from marginalized to target
    Eigen::Matrix4d T_rel = marginalized_pose.inverse() * target_kf->pose;

    // Add weighted uncertainty based on relative motion
    Eigen::Vector3d rel_translation = T_rel.block<3,1>(0,3);
    double rel_distance = rel_translation.norm();

    // Scale marginal information by distance (closer = stronger constraint)
    double distance_factor = std::exp(-rel_distance / 2.0); // 2m characteristic distance
    Eigen::MatrixXd scaled_info = marginal_info * distance_factor;

    // Apply to target keyframe covariance (information addition)
    Eigen::MatrixXd target_info = calculateInformationMatrix(target_kf->covariance);
    target_info += scaled_info * 0.3; // 30% contribution from marginal prior

    // Convert back to covariance
    target_kf->covariance = calculateInformationMatrix(target_info);

    std::cout << "Applied marginal prior to keyframe " << target_keyframe_id
              << " (distance factor: " << distance_factor << ")" << std::endl;
}

} // namespace lar
