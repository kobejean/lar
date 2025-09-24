//
// SlidingWindowBA.cpp - Sliding Window Bundle Adjustment implementation
//

#include "lar/tracking/pose_filtering/sliding_window_ba.h"
#include "lar/tracking/filtered_tracker_config.h"
#include "lar/core/landmark.h"
#include "lar/core/utils/transform.h"

// g2o includes
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
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

void SlidingWindowBA::initialize(const Eigen::Matrix4d& initial_pose, const FilteredTrackerConfig& config) {
    // Set configuration from config
    max_window_size_ = config.sliding_window_max_keyframes;
    keyframe_distance_ = config.sliding_window_keyframe_distance;  // meters
    keyframe_angle_ = config.sliding_window_keyframe_angle * M_PI / 180.0;  // Convert degrees to radians
    min_observations_ = config.sliding_window_min_observations;
    optimization_iterations_ = config.sliding_window_optimization_iterations;

    // Initialize state
    current_state_.fromTransform(initial_pose);

    // Set initial covariance based on config
    current_covariance_ = Eigen::MatrixXd::Identity(6, 6);
    current_covariance_.block<3,3>(0,0) *= config.initial_position_uncertainty * config.initial_position_uncertainty;
    current_covariance_.block<3,3>(3,3) *= config.initial_orientation_uncertainty * config.initial_orientation_uncertainty;

    // Create first keyframe
    createKeyframe(initial_pose, current_covariance_);

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
    bool should_log = (predict_call_count % 20 == 0);

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

    current_covariance_ += Q * Q;  // Add process noise
}

void SlidingWindowBA::update(const MeasurementContext& context,
                              const Eigen::MatrixXd& measurement_noise,
                              const FilteredTrackerConfig& config) {
    const Eigen::Matrix4d& measurement = context.measured_pose;
    std::cout << "SlidingWindowBA::update() called" << std::endl;
    std::cout << "  Measurement position: [" << measurement(0,3) << ", " << measurement(1,3) << ", " << measurement(2,3) << "]" << std::endl;
    std::cout << "  Current prediction: [" << current_state_.position.x() << ", " << current_state_.position.y() << ", " << current_state_.position.z() << "]" << std::endl;

    if (!initialized_) {
        initialize(measurement, config);
        return;
    }

    // Check if we should create a new keyframe
    if (shouldCreateKeyframe(measurement)) {
        createKeyframe(measurement, measurement_noise);

        // Marginalize old keyframes if window is full
        while (keyframes_.size() > max_window_size_) {
            marginalizeOldestKeyframe();
        }

        // Reset motion accumulator after creating keyframe
        accumulated_motion_ = Eigen::Matrix4d::Identity();
    } else {
        // Just update current state without creating keyframe
        // Simple weighted average between prediction and measurement
        double measurement_weight = 0.5;  // TODO: Calculate based on covariances

        Eigen::Matrix4d predicted_pose = current_state_.toTransform();
        Eigen::Matrix4d updated_pose = predicted_pose * measurement_weight +
                                       measurement * (1.0 - measurement_weight);
        current_state_.fromTransform(updated_pose);

        // Update covariance (simplified Kalman-like update)
        current_covariance_ = current_covariance_ * measurement_weight +
                             measurement_noise * (1.0 - measurement_weight);
    }

    // Run bundle adjustment optimization after every measurement if we have multiple keyframes
    if (keyframes_.size() >= 2) {
        buildOptimizationGraph(config);
        optimize();
        updateStateFromOptimization();
    }
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
    current_frame_.reset();
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
// Public Methods
// ============================================================================

void SlidingWindowBA::addObservations(const std::vector<std::pair<Landmark*, Eigen::Vector2d>>& observations) {
    if (!current_frame_) {
        current_frame_ = std::make_shared<Keyframe>();
        current_frame_->pose = current_state_.toTransform();
        current_frame_->covariance = current_covariance_;
    }

    current_frame_->observations = observations;

    // Track landmark observation counts
    for (const auto& [landmark, pixel] : observations) {
        landmark_observations_[landmark]++;
        active_landmarks_.insert(landmark);
    }
}

void SlidingWindowBA::setCameraIntrinsics(double fx, double fy, double cx, double cy) {
    fx_ = fx;
    fy_ = fy;
    cx_ = cx;
    cy_ = cy;
}

// ============================================================================
// Private Helper Methods
// ============================================================================

bool SlidingWindowBA::shouldCreateKeyframe(const Eigen::Matrix4d& current_pose) const {
    if (keyframes_.empty()) return true;

    // Get last keyframe
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
    double angle = std::acos((R_diff.trace() - 1.0) / 2.0);
    if (std::abs(angle) > keyframe_angle_) {
        return true;
    }

    // Check observation criterion
    if (current_frame_ && current_frame_->observations.size() >= min_observations_) {
        return true;
    }

    return false;
}

void SlidingWindowBA::createKeyframe(const Eigen::Matrix4d& pose, const Eigen::MatrixXd& covariance) {
    auto kf = std::make_shared<Keyframe>();
    kf->id = next_keyframe_id_++;
    kf->timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
    kf->pose = pose;
    kf->covariance = covariance;

    // Transfer observations from current frame
    if (current_frame_) {
        kf->observations = current_frame_->observations;
    }

    keyframes_.push_back(kf);
    current_frame_.reset();

    std::cout << "Created keyframe " << kf->id
              << " (total: " << keyframes_.size() << ")" << std::endl;
}

void SlidingWindowBA::marginalizeOldestKeyframe() {
    if (keyframes_.empty()) return;

    // TODO: Implement proper marginalization
    // For now, just remove the oldest keyframe
    auto oldest = keyframes_.front();
    keyframes_.pop_front();

    // Remove observations of marginalized landmarks
    for (const auto& [landmark, pixel] : oldest->observations) {
        auto it = landmark_observations_.find(landmark);
        if (it != landmark_observations_.end()) {
            it->second--;
            if (it->second == 0) {
                landmark_observations_.erase(it);
                active_landmarks_.erase(landmark);
            }
        }
    }

    std::cout << "Marginalized keyframe " << oldest->id << std::endl;
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

        // Fix first keyframe
        if (kf == keyframes_.front()) {
            v->setFixed(true);
        }

        optimizer_->addVertex(v);
        pose_vertices_[kf->id] = v;

        // Add camera parameters for this keyframe
        // Use keyframe id as parameter id
        // For now using default intrinsics, should be updated per frame
        Eigen::Vector2d principal_point(cx_, cy_);
        auto* cam_params = new g2o::CameraParameters(fx_, principal_point, 0.0);  // baseline=0 for monocular
        cam_params->setId(kf->id);
        if (!optimizer_->addParameter(cam_params)) {
            // Parameters already exist
            delete cam_params;
        }
    }

    // Add landmark vertices
    for (const auto& landmark : active_landmarks_) {
        auto v = new g2o::VertexPointXYZ();
        v->setId(vertex_id++);

        Eigen::Vector3d position(landmark->position[0],
                                 landmark->position[1],
                                 landmark->position[2]);
        v->setEstimate(position);
        v->setMarginalized(true);  // Marginalize landmarks for efficiency

        optimizer_->addVertex(v);
        landmark_vertices_[landmark] = v;
    }

    // Add reprojection edges
    int edge_id = 0;
    for (const auto& kf : keyframes_) {
        auto pose_vertex = pose_vertices_[kf->id];

        for (const auto& [landmark, observation] : kf->observations) {
            auto lm_it = landmark_vertices_.find(landmark);
            if (lm_it == landmark_vertices_.end()) continue;

            auto landmark_vertex = lm_it->second;

            // Create edge - use the same type as bundle_adjustment.cpp
            auto edge = new g2o::EdgeProjectXYZ2UVD();
            edge->setId(edge_id++);
            edge->setVertex(0, landmark_vertex);  // 3D point
            edge->setVertex(1, pose_vertex);       // SE3 pose

            // Set measurement (u, v, depth=0 for monocular)
            Eigen::Vector3d measurement_with_depth(observation[0], observation[1], 0.0);
            edge->setMeasurement(measurement_with_depth);

            // Set information matrix (only weight u,v; ignore depth)
            double pixel_noise = config.sliding_window_pixel_noise;
            Eigen::Vector3d information_diag(1.0/(pixel_noise*pixel_noise),
                                            1.0/(pixel_noise*pixel_noise),
                                            0.0);  // Zero weight for depth
            edge->setInformation(information_diag.asDiagonal());

            // Link to camera parameters for this keyframe
            edge->setParameterId(0, kf->id);

            optimizer_->addEdge(edge);
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

Eigen::Vector2d SlidingWindowBA::project(const Eigen::Vector3d& point_camera) const {
    double x = point_camera[0] / point_camera[2];
    double y = point_camera[1] / point_camera[2];

    double u = fx_ * x + cx_;
    double v = fy_ * y + cy_;

    return Eigen::Vector2d(u, v);
}

Eigen::Matrix<double, 2, 6> SlidingWindowBA::calculateReprojectionJacobian(
    const Eigen::Vector3d& point_world,
    const Eigen::Matrix4d& T_camera_from_world) const {

    // Transform point to camera coordinates
    Eigen::Vector4d pw_homo(point_world[0], point_world[1], point_world[2], 1.0);
    Eigen::Vector4d pc_homo = T_camera_from_world * pw_homo;
    Eigen::Vector3d pc = pc_homo.head<3>();

    double X = pc[0], Y = pc[1], Z = pc[2];

    // Jacobian of projection w.r.t camera coordinates
    Eigen::Matrix<double, 2, 3> J_proj;
    J_proj(0, 0) = fx_ / Z;
    J_proj(0, 1) = 0;
    J_proj(0, 2) = -fx_ * X / (Z * Z);
    J_proj(1, 0) = 0;
    J_proj(1, 1) = fy_ / Z;
    J_proj(1, 2) = -fy_ * Y / (Z * Z);

    // Jacobian of camera point w.r.t pose (SE3 parameterization)
    Eigen::Matrix<double, 3, 6> J_pose;

    // Translation part
    J_pose.block<3,3>(0,0) = Eigen::Matrix3d::Identity();

    // Rotation part (using cross product)
    J_pose(0, 3) = 0;
    J_pose(0, 4) = Z;
    J_pose(0, 5) = -Y;
    J_pose(1, 3) = -Z;
    J_pose(1, 4) = 0;
    J_pose(1, 5) = X;
    J_pose(2, 3) = Y;
    J_pose(2, 4) = -X;
    J_pose(2, 5) = 0;

    // Chain rule
    return J_proj * J_pose;
}

} // namespace lar