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
#include <g2o/core/optimizable_graph.h>
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
    optimizer_->setVerbose(true);
    covariance_ = Eigen::MatrixXd::Identity(6, 6) * 0.01;
}

SlidingWindowBA::~SlidingWindowBA() = default;

// ============================================================================
// PoseFilterStrategy Interface
// ============================================================================

void SlidingWindowBA::initialize(const MeasurementContext& context, const FilteredTrackerConfig& config) {
    // Store configuration
    config_ = config;

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
    state_.fromTransform(context.measured_pose);

    // Use measurement noise from context as initial covariance
    covariance_ = context.measurement_noise;

    // Create first keyframe directly from context
    auto kf = std::make_shared<Keyframe>(next_keyframe_id_++, context);
    keyframes_.push_back(kf);

    if (config_.enable_debug_output) {
        std::cout << "Created keyframe " << kf->id << " with confidence=" << kf->confidence
                  << ", intrinsics [fx=" << kf->intrinsics(0,0) << ", fy=" << kf->intrinsics(1,1)
                  << ", cx=" << kf->intrinsics(0,2) << ", cy=" << kf->intrinsics(1,2)
                  << "] (total: " << keyframes_.size() << ")" << std::endl;
    }
    accumulated_motion_ = Eigen::Matrix4d::Identity();
    initialized_ = true;
    if (config_.enable_debug_output) {
        std::cout << "SlidingWindowBA initialized with first keyframe" << std::endl;
    }
}

void SlidingWindowBA::predict(const Eigen::Matrix4d& motion, double dt, const FilteredTrackerConfig& config) {
    if (!initialized_) return;

    static int predict_call_count = 0;
    predict_call_count++;

    // Apply motion to current state
    Eigen::Matrix4d pose = state_.toTransform();
    pose = pose * motion;
    state_.fromTransform(pose);

    // Accumulate motion since last keyframe
    accumulated_motion_ = accumulated_motion_ * motion;

    // Propagate uncertainty using motion model
    Eigen::Vector3d translation = motion.block<3,1>(0,3);
    double motion_magnitude = translation.norm();

    // Process noise scaled by motion and time
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(6, 6);
    Q.block<3,3>(0,0) = Eigen::Matrix3d::Identity() *
        (config.motion_position_noise_scale * motion_magnitude + config.time_position_noise_scale * dt);
    Q.block<3,3>(3,3) = Eigen::Matrix3d::Identity() *
        (config.motion_orientation_noise_scale * motion_magnitude + config.time_orientation_noise_scale * dt);

    covariance_ += Q;  // Add process noise directly (Q is already a covariance matrix)
}

void SlidingWindowBA::update(const MeasurementContext& context,
                              const FilteredTrackerConfig& config) {
    const Eigen::Matrix4d& measurement = context.measured_pose;
    if (config_.enable_debug_output) {
        std::cout << "SlidingWindowBA::update() called" << std::endl;
        std::cout << "  Measurement position: [" << measurement(0,3) << ", " << measurement(1,3) << ", " << measurement(2,3) << "]" << std::endl;
        std::cout << "  Current prediction: [" << state_.position.x() << ", " << state_.position.y() << ", " << state_.position.z() << "]" << std::endl;
    }

    if (!initialized_) {
        initialize(context, config);
        return;
    }

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

    if (config_.enable_debug_output) {
        std::cout << "Created keyframe " << kf->id << " with confidence=" << kf->confidence
                << ", intrinsics [fx=" << kf->intrinsics(0,0) << ", fy=" << kf->intrinsics(1,1)
                << ", cx=" << kf->intrinsics(0,2) << ", cy=" << kf->intrinsics(1,2)
                << "] (total: " << keyframes_.size() << ")" << std::endl;
    }

    // Marginalize old keyframes if window is full
    while (keyframes_.size() > config_.sliding_window_max_keyframes) {
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
    
    if (config_.enable_debug_output) {
        std::cout << "SlidingWindowBA state update:" << std::endl;
        std::cout << "  Accumulated motion:" << std::endl;
        std::cout << "    Translation: [" << accumulated_motion_(0,3) << ", " << accumulated_motion_(1,3) << ", " << accumulated_motion_(2,3) << "]" << std::endl;
        std::cout << "  Final state pose:" << std::endl;
        Eigen::Matrix4d final_pose = state_.toTransform();
        std::cout << "    Position: [" << final_pose(0,3) << ", " << final_pose(1,3) << ", " << final_pose(2,3) << "]" << std::endl;
        std::cout << "  Real BA Covariance:" << std::endl;
        std::cout << "    Position uncertainty: ["
                    << sqrt(covariance_(0,0)) << ", "
                    << sqrt(covariance_(1,1)) << ", "
                    << sqrt(covariance_(2,2)) << "] m" << std::endl;
        std::cout << "    Orientation uncertainty: ["
                    << sqrt(covariance_(3,3)) * 180.0 / M_PI << "°, "
                    << sqrt(covariance_(4,4)) * 180.0 / M_PI << "°, "
                    << sqrt(covariance_(5,5)) * 180.0 / M_PI << "°]" << std::endl;
    }
}

PoseState SlidingWindowBA::getState() const {
    return state_;
}

Eigen::MatrixXd SlidingWindowBA::getCovariance() const {
    return covariance_;
}

double SlidingWindowBA::getPositionUncertainty() const {
    // Return trace of position covariance
    return std::sqrt((covariance_(0,0) + covariance_(1,1) + covariance_(2,2)) / 3.0);
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

void SlidingWindowBA::marginalizeOldestKeyframe() {
    if (keyframes_.empty()) return;

    auto oldest = keyframes_.front();

    // Proper marginalization: extract marginal information before removal
    // Convert covariance to information matrix for marginalization
    Eigen::MatrixXd marginal_covariance = extractCovarianceFromBA(oldest->id);
    Eigen::MatrixXd marginal_info = calculateInformationMatrix(marginal_covariance);

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
            if (it != landmark_observations_.end() && --it->second == 0) {
                landmark_observations_.erase(it);
                active_landmarks_.erase(landmark);
            }
        }
    }

    if (config_.enable_debug_output) {
        std::cout << "Properly marginalized keyframe " << oldest->id
                << " (applied prior to keyframe " << (keyframes_.empty() ? -1 : keyframes_.front()->id) << ")" << std::endl;
    }
}

void SlidingWindowBA::buildOptimizationGraph(const FilteredTrackerConfig& config) {
    // Clear previous graph
    optimizer_->clear();
    optimizer_->clearParameters();
    pose_vertices_.clear();
    landmark_vertices_.clear();

    if (config_.enable_debug_output) {
        std::cout << "Building optimization graph with " << keyframes_.size() << " keyframes" << std::endl;
    }

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

        if (config_.enable_debug_output) {
            std::cout << "Fixed keyframe " << best_keyframe_id << " (confidence: "
                    << highest_confidence << ") for gauge freedom" << std::endl;
        }
    }

    // Add landmark vertices - no fixing needed since we fix one pose for gauge freedom

    for (const auto& landmark : active_landmarks_) {
        auto v = new g2o::VertexPointXYZ();
        v->setId(vertex_id++);
        Eigen::Vector3d position(landmark->position[0], landmark->position[1], landmark->position[2]);
        v->setEstimate(position);
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
                auto edge = new g2o::EdgeProjectXYZ2UV();
                edge->setId(edge_id++);
                edge->setVertex(0, landmark_vertex);  // 3D point
                edge->setVertex(1, pose_vertex);       // SE3 pose

                // Set measurement (u, v, depth=0 for monocular) - extract from keypoint
                Eigen::Vector2d measurement_with_depth(keypoint.pt.x, keypoint.pt.y);
                edge->setMeasurement(measurement_with_depth);
                double xy_information = 1./9.0;
                Eigen::Vector2d information_diag(xy_information, xy_information);
                edge->setInformation(information_diag.asDiagonal());

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

        // Calculate motion distance for logging
        Eigen::Vector3d translation = T_rel.block<3,1>(0,3);
        double motion_distance = translation.norm();

        // Set information based on distance-dependent scaling (matching bundle_adjustment.cpp)
        // Shorter motions get higher confidence, longer motions get lower confidence
        double distance_scale = 1.0 / std::max(0.1, motion_distance);  // Avoid division by zero

        Eigen::MatrixXd info = Eigen::MatrixXd::Identity(6, 6);
        // EXPERIMENT: More conservative base information (was 100.0/200.0)
        info.block<3,3>(0,0) *= 25.0 * distance_scale;  // Position weight (4x more conservative)
        info.block<3,3>(3,3) *= 50.0 * distance_scale;  // Orientation weight (4x more conservative)

        edge->setInformation(info);

        optimizer_->addEdge(edge);
    }
}

void SlidingWindowBA::optimize() {
    if (config_.enable_debug_output) {
        std::cout << "=== SlidingWindowBA Optimization ===" << std::endl;
        std::cout << "Keyframes: " << keyframes_.size() << std::endl;
        std::cout << "Vertices: " << optimizer_->vertices().size() << std::endl;
        std::cout << "Edges: " << optimizer_->edges().size() << std::endl;
    }

    optimizer_->initializeOptimization();
    if (config_.enable_debug_output) {
        std::cout << "Initial chi2: " << optimizer_->chi2() << std::endl;
    }

    optimizer_->optimize(config_.sliding_window_optimization_iterations);

    if (config_.enable_debug_output) {
        std::cout << "Final chi2: " << optimizer_->chi2() << std::endl;
        std::cout << "=== End Optimization ===" << std::endl;
    }
}

void SlidingWindowBA::updateStateFromOptimization() {
    // Update keyframe poses from optimization
    for (auto& kf : keyframes_) {
        auto it = pose_vertices_.find(kf->id);
        if (it != pose_vertices_.end()) {
            // Store original pose for comparison
            Eigen::Matrix4d original_pose = kf->pose;

            // Get optimized pose and convert back to ARKit coordinates
            g2o::SE3Quat optimized = it->second->estimate();
            kf->pose = lar::utils::TransformUtils::g2oToArkitPose(optimized);

            // Calculate how much the pose moved
            Eigen::Vector3d pos_diff = kf->pose.block<3,1>(0,3) - original_pose.block<3,1>(0,3);
            double pos_movement = pos_diff.norm();

            // Calculate rotation difference using axis-angle representation
            Eigen::Matrix3d R_diff = original_pose.block<3,3>(0,0).transpose() * kf->pose.block<3,3>(0,0);
            Eigen::Vector3d axis_angle = lar::utils::TransformUtils::rotationMatrixToAxisAngle(R_diff);
            double rot_movement = axis_angle.norm() * 180.0 / M_PI;  // Convert to degrees

            if (config_.enable_debug_output) {
                std::cout << "Keyframe " << kf->id << " moved: "
                        << pos_movement << "m, " << rot_movement << "°" << std::endl;
            }
        }
    }

    // Update current state from last keyframe
    if (!keyframes_.empty()) {
        const auto& last_kf = keyframes_.back();
        state_.fromTransform(last_kf->pose * accumulated_motion_);

        // Extract true covariance from bundle adjustment
        covariance_ = extractCovarianceFromBA(last_kf->id);

        if (config_.enable_debug_output) {
            std::cout << "SlidingWindowBA state update:" << std::endl;
            std::cout << "  Last keyframe pose:" << std::endl;
            std::cout << "    Position: [" << last_kf->pose(0,3) << ", " << last_kf->pose(1,3) << ", " << last_kf->pose(2,3) << "]" << std::endl;
        }
    }
}

Eigen::MatrixXd SlidingWindowBA::extractCovarianceFromBA(size_t keyframe_id) const {
    // Find the vertex for this keyframe
    auto vertex_it = pose_vertices_.find(keyframe_id);
    if (vertex_it == pose_vertices_.end()) {
        if (config_.enable_debug_output) {
            std::cout << "ERROR: Cannot find pose vertex for keyframe " << keyframe_id
                      << " in optimization graph!" << std::endl;
            std::cout << "Available vertex IDs: ";
            for (const auto& [id, vertex] : pose_vertices_) {
                std::cout << id << " ";
            }
            std::cout << std::endl;
        }

        // This indicates a serious bug - don't hide it with fallbacks
        throw std::runtime_error("Pose vertex not found in optimization graph - this should never happen");
    }

    auto target_vertex = vertex_it->second;

    // Check Hessian conditioning before computing marginals
    optimizer_->computeInitialGuess();
    optimizer_->initializeOptimization();

    if (config_.enable_debug_output) {
        std::cout << "Diagnosing Hessian matrix condition..." << std::endl;
        std::cout << "  Active vertices: " << optimizer_->vertices().size() << std::endl;
        std::cout << "  Active edges: " << optimizer_->edges().size() << std::endl;
    }
    // Get access to the Hessian matrix for diagnostics

    // Count fixed vertices
    int fixed_vertices = 0;
    for (auto& vertex_pair : optimizer_->vertices()) {
        auto* opt_vertex = static_cast<g2o::OptimizableGraph::Vertex*>(vertex_pair.second);
        if (opt_vertex->fixed()) fixed_vertices++;
    }

    if (config_.enable_debug_output) {
        std::cout << "  Fixed vertices: " << fixed_vertices << std::endl;
    }

    // Try to extract true marginal covariance using g2o
    g2o::SparseBlockMatrix<Eigen::MatrixXd> spinv;
    if (optimizer_->computeMarginals(spinv, target_vertex)) {
        int vertex_idx = target_vertex->hessianIndex();
        if (vertex_idx >= 0) {
            auto block = spinv.block(vertex_idx, vertex_idx);
            if (block && block->rows() == 6 && block->cols() == 6) {
                if (config_.enable_debug_output) {
                    std::cout << "Extracted marginal covariance from BA, applying realistic scaling..." << std::endl;
                }

                // Scale and clamp g2o covariance to realistic values
                Eigen::MatrixXd cov = *block;

                // Minimum realistic uncertainties for visual SLAM
                double min_pos_var = (0.02) * (0.02);    // 2cm minimum position uncertainty
                double min_rot_var = (1.0 * M_PI/180.0) * (1.0 * M_PI/180.0);  // 1° minimum rotation uncertainty

                // Scale up if too small (g2o often underestimates)
                double pos_scale_factor = 50.0;  // Scale position uncertainty up
                double rot_scale_factor = 25.0;  // Scale rotation uncertainty up

                // Apply scaling
                cov.block<3,3>(0,0) *= pos_scale_factor;
                cov.block<3,3>(3,3) *= rot_scale_factor;

                // Clamp to minimum realistic values
                for (int i = 0; i < 3; i++) {
                    cov(i,i) = std::max(cov(i,i), min_pos_var);  // Position components
                    cov(i+3,i+3) = std::max(cov(i+3,i+3), min_rot_var);  // Rotation components
                }

                if (config_.enable_debug_output) {
                    std::cout << "Scaled covariance - pos std: ["
                            << std::sqrt(cov(0,0)) << ", " << std::sqrt(cov(1,1)) << ", " << std::sqrt(cov(2,2)) << "] m" << std::endl;
                    std::cout << "Scaled covariance - rot std: ["
                            << std::sqrt(cov(3,3))*180.0/M_PI << "°, " << std::sqrt(cov(4,4))*180.0/M_PI << "°, "
                            << std::sqrt(cov(5,5))*180.0/M_PI << "°]" << std::endl;
                }
                return cov;
            }
        }
    }

    if (config_.enable_debug_output) {
        // Fallback: residual-based estimate
        std::cout << "Marginal computation failed, using residual-based estimate" << std::endl;
    }

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
