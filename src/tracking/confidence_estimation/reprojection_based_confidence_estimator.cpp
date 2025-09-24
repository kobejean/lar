//
// ReprojectionBasedConfidenceEstimator.cpp - Data-driven confidence estimation
//

#include "lar/tracking/confidence_estimation/reprojection_based_confidence_estimator.h"
#include "lar/tracking/filtered_tracker_config.h"
#include "lar/core/landmark.h"
#include "lar/mapping/frame.h"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace lar {

double ReprojectionBasedConfidenceEstimator::calculateConfidence(
    const MeasurementContext& context,
    const FilteredTrackerConfig& config) const {

    const auto& inliers = context.inliers;
    const Eigen::Matrix4d& T_lar_from_camera = context.measured_pose;
    const Frame& frame = *context.frame;

    // Early exit for insufficient inliers
    if (inliers.size() < config.min_inliers_for_tracking) {
        return 0.0;
    }

    // Calculate reprojection RMSE
    std::vector<double> errors = calculateReprojectionErrors(inliers, T_lar_from_camera, frame);
    if (errors.empty()) return 0.0;

    double rmse = calculateRMSE(errors);

    // === DATA-DRIVEN CONFIDENCE CALCULATION ===
    // Based on Fisher Information Matrix approach with corrected coordinate system

    // Calculate Fisher Information Matrix for uncertainty estimation
    Eigen::MatrixXd fisher_cov = calculateFisherInformationCovariance(inliers, T_lar_from_camera, frame, config);

    // Extract position and orientation uncertainties from diagonal
    double pos_uncertainty = std::sqrt((fisher_cov(0,0) + fisher_cov(1,1) + fisher_cov(2,2)) / 3.0);
    double ori_uncertainty = std::sqrt((fisher_cov(3,3) + fisher_cov(4,4) + fisher_cov(5,5)) / 3.0);

    // Reprojection quality component
    double target_rmse = 5.0;  // Target: 5 pixels or better
    double error_score = std::exp(-rmse / target_rmse);

    // Uncertainty-based confidence (lower uncertainty = higher confidence)
    double pos_score = std::exp(-pos_uncertainty / 0.1);  // Target: 10cm position uncertainty
    double ori_score = std::exp(-ori_uncertainty / 0.05); // Target: ~3° orientation uncertainty

    // Inlier quality
    double inlier_ratio = context.total_matches > 0 ?
        static_cast<double>(inliers.size()) / context.total_matches : 1.0;
    double count_score = std::min(1.0, inliers.size() / 50.0);  // Normalize by 50 inliers
    double ratio_score = std::min(1.0, inlier_ratio * 5.0);    // Cap at 20% ratio

    // Combined confidence with data-driven weighting
    double confidence = 0.3 * error_score +   // 30% reprojection quality
                       0.3 * pos_score +      // 30% position uncertainty
                       0.2 * ori_score +      // 20% orientation uncertainty
                       0.1 * count_score +    // 10% inlier count
                       0.1 * ratio_score;     // 10% inlier ratio

    // Clamp to reasonable range
    confidence = std::max(0.01, std::min(0.95, confidence));

    // Debug output
    if (config.enable_debug_output) {
        std::cout << "=== Reprojection-Based Measurement Confidence ===" << std::endl;
        std::cout << "  RMSE: " << rmse << " pixels" << std::endl;
        std::cout << "  Position uncertainty: " << pos_uncertainty << " m" << std::endl;
        std::cout << "  Orientation uncertainty: " << ori_uncertainty << " rad" << std::endl;
        std::cout << "  Inliers: " << inliers.size() << ", ratio: " << (inlier_ratio * 100) << "%" << std::endl;
        std::cout << "  Error score: " << error_score << std::endl;
        std::cout << "  Position score: " << pos_score << std::endl;
        std::cout << "  Orientation score: " << ori_score << std::endl;
        std::cout << "  Count/ratio scores: " << count_score << "/" << ratio_score << std::endl;
        std::cout << "  Final measurement confidence: " << confidence << std::endl;
    }

    return confidence;
}

Eigen::MatrixXd ReprojectionBasedConfidenceEstimator::calculateMeasurementNoise(
    const MeasurementContext& context,
    const FilteredTrackerConfig& config) const {

    const auto& inliers = context.inliers;
    const Eigen::Matrix4d& T_lar_from_camera = context.measured_pose;
    const Frame& frame = *context.frame;
    double confidence = context.confidence;

    if (inliers.size() < 4) {
        // Fallback to diagonal matrix for insufficient data
        Eigen::MatrixXd R = Eigen::MatrixXd::Identity(6, 6);
        R.block<3,3>(0,0) *= config.base_position_noise * config.base_position_noise;
        R.block<3,3>(3,3) *= config.base_orientation_noise * config.base_orientation_noise;
        return R;
    }

    // Calculate Fisher Information based covariance
    Eigen::MatrixXd fisher_cov = calculateFisherInformationCovariance(inliers, T_lar_from_camera, frame, config);

    // Scale by confidence (lower confidence = higher uncertainty)
    double confidence_scale = 1.0 / std::max(0.1, confidence);
    fisher_cov *= confidence_scale;

    // Add minimum uncertainty floor to prevent overconfidence
    Eigen::MatrixXd min_cov = Eigen::MatrixXd::Identity(6, 6);
    min_cov.block<3,3>(0,0) *= config.reprojection_min_position_noise * config.reprojection_min_position_noise;
    min_cov.block<3,3>(3,3) *= config.reprojection_min_orientation_noise * config.reprojection_min_orientation_noise;

    // Take element-wise maximum to ensure minimum uncertainty
    for (int i = 0; i < 6; ++i) {
        fisher_cov(i, i) = std::max(fisher_cov(i, i), min_cov(i, i));
    }

    return fisher_cov;
}

std::vector<double> ReprojectionBasedConfidenceEstimator::calculateReprojectionErrors(
    const std::vector<std::pair<Landmark*, cv::KeyPoint>>& inliers,
    const Eigen::Matrix4d& T_lar_from_camera,
    const Frame& frame) const {

    std::vector<double> errors;
    errors.reserve(inliers.size());

    int points_behind_camera = 0;
    int valid_points = 0;

    for (const auto& [landmark, keypoint] : inliers) {
        // Transform landmark to camera coordinates
        Eigen::Vector4d world_point(landmark->position[0], landmark->position[1],
                                   landmark->position[2], 1.0);
        Eigen::Vector4d camera_point = T_lar_from_camera.inverse() * world_point;

        // ARKit to OpenCV coordinate conversion
        // ARKit: Y up, Z backward -> OpenCV: Y down, Z forward
        // Following pattern from tracker.cpp and colmap_database.cpp
        double X = camera_point[0];
        double Y = -camera_point[1];  // Flip Y: ARKit Y-up -> OpenCV Y-down
        double Z = -camera_point[2];  // Flip Z: ARKit Z-backward -> OpenCV Z-forward

        // Check if point is behind camera in OpenCV coordinates
        if (Z <= 0.0) {
            points_behind_camera++;
            continue;
        }

        // Update camera_point with converted coordinates for projection
        camera_point[0] = X;
        camera_point[1] = Y;
        camera_point[2] = Z;

        // Project to pixel coordinates
        Eigen::Vector2d projected_pixel = projectToPixel(camera_point.head<3>(), frame);
        Eigen::Vector2d observed_pixel(keypoint.pt.x, keypoint.pt.y);

        // Calculate reprojection error
        double error = (projected_pixel - observed_pixel).norm();
        errors.push_back(error);
        valid_points++;
    }


    return errors;
}

double ReprojectionBasedConfidenceEstimator::calculateScaleQuality(
    const std::vector<std::pair<Landmark*, cv::KeyPoint>>& inliers) const {

    if (inliers.empty()) return 0.0;

    double mean_scale = 0.0;
    double scale_variance = 0.0;

    // Calculate mean scale (use keypoint.size instead of octave for SIFT)
    // SIFT octave can be very large, but keypoint.size is more stable
    for (const auto& [landmark, keypoint] : inliers) {
        mean_scale += keypoint.size;  // Use size instead of octave
    }
    mean_scale /= inliers.size();

    // Calculate scale variance for consistency measure
    for (const auto& [landmark, keypoint] : inliers) {
        double diff = keypoint.size - mean_scale;
        scale_variance += diff * diff;
    }
    scale_variance /= inliers.size();

    // Higher mean scale = better (more distinctive) features
    // Lower variance = more consistent feature quality
    // keypoint.size is typically in range 1-50+ pixels, adjust accordingly
    double scale_bonus = std::min(1.0, mean_scale / 20.0);  // Normalize by typical max size
    double consistency_bonus = std::exp(-scale_variance / 100.0);  // Adjust for size variance

    double result = scale_bonus * consistency_bonus;


    return result;
}

double ReprojectionBasedConfidenceEstimator::calculateRMSE(const std::vector<double>& errors) const {
    if (errors.empty()) return std::numeric_limits<double>::infinity();

    double sum_squared = 0.0;
    for (double error : errors) {
        sum_squared += error * error;
    }

    return std::sqrt(sum_squared / errors.size());
}

Eigen::MatrixXd ReprojectionBasedConfidenceEstimator::calculateFisherInformationCovariance(
    const std::vector<std::pair<Landmark*, cv::KeyPoint>>& inliers,
    const Eigen::Matrix4d& T_lar_from_camera,
    const Frame& frame,
    const FilteredTrackerConfig& config) const {

    // Initialize Fisher Information Matrix
    Eigen::MatrixXd fisher_info = Eigen::MatrixXd::Zero(6, 6);

    // Calculate reprojection errors for weighting
    std::vector<double> errors = calculateReprojectionErrors(inliers, T_lar_from_camera, frame);

    for (size_t i = 0; i < inliers.size() && i < errors.size(); ++i) {
        const auto& [landmark, keypoint] = inliers[i];

        // Calculate Jacobian of reprojection w.r.t. pose
        Eigen::MatrixXd J_pose = calculateReprojectionJacobian(landmark, T_lar_from_camera, frame);

        // Calculate Jacobian of reprojection w.r.t. landmark position
        Eigen::MatrixXd J_landmark = calculateLandmarkJacobian(landmark, T_lar_from_camera, frame);

        // Calculate feature weight (based on reprojection error and feature quality)
        double weight = calculateFeatureWeight(keypoint, landmark, errors[i]);

        // Create measurement covariance matrix accounting for:
        // 1. Pixel measurement noise (weighted by feature quality)
        // 2. Landmark position uncertainty propagated through observation

        // Pixel measurement noise (2x2)
        double pixel_noise = 1.0 / weight;  // Higher weight = lower noise
        Eigen::Matrix2d R_pixel = pixel_noise * Eigen::Matrix2d::Identity();

        // Landmark position uncertainty (3x3)
        double landmark_noise = config.landmark_position_noise;
        Eigen::Matrix3d Q_landmark = (landmark_noise * landmark_noise) * Eigen::Matrix3d::Identity();

        // Total measurement covariance: R = R_pixel + J_landmark * Q_landmark * J_landmark^T
        Eigen::Matrix2d R_total = R_pixel + J_landmark * Q_landmark * J_landmark.transpose();

        // Information matrix (inverse of covariance)
        Eigen::Matrix2d W = R_total.inverse();

        // Add to Fisher Information: J_pose^T * W * J_pose
        fisher_info += J_pose.transpose() * W * J_pose;
    }

    // Add regularization to prevent singularity
    fisher_info += 1e-6 * Eigen::MatrixXd::Identity(6, 6);

    // Covariance = Fisher Information^-1
    Eigen::MatrixXd covariance = fisher_info.inverse();

    return covariance;
}

Eigen::MatrixXd ReprojectionBasedConfidenceEstimator::calculateReprojectionJacobian(
    const Landmark* landmark,
    const Eigen::Matrix4d& T_lar_from_camera,
    const Frame& frame) const {

    // Transform landmark to camera coordinates
    Eigen::Vector4d world_point(landmark->position[0], landmark->position[1],
                               landmark->position[2], 1.0);
    Eigen::Vector4d camera_point = T_lar_from_camera.inverse() * world_point;

    // ARKit to OpenCV coordinate conversion (same as reprojection error calculation)
    // ARKit: Y up, Z backward -> OpenCV: Y down, Z forward
    double X = camera_point[0];
    double Y = -camera_point[1];  // Flip Y: ARKit Y-up -> OpenCV Y-down
    double Z = -camera_point[2];  // Flip Z: ARKit Z-backward -> OpenCV Z-forward

    // Get camera intrinsics from frame (3x3 matrix)
    double fx = frame.intrinsics(0, 0);
    double fy = frame.intrinsics(1, 1);
    double cx = frame.intrinsics(0, 2);
    double cy = frame.intrinsics(1, 2);

    // Jacobian matrix (2 x 6): [∂u/∂pose, ∂v/∂pose]
    // pose = [tx, ty, tz, rx, ry, rz]
    Eigen::MatrixXd J(2, 6);

    // Avoid division by zero
    if (std::abs(Z) < 1e-6) {
        J.setZero();
        return J;
    }

    // Standard OpenCV Jacobian derivatives
    // (coordinates already converted from ARKit to OpenCV)
    J(0, 0) = fx / Z;                    // ∂u/∂tx
    J(0, 1) = 0;                         // ∂u/∂ty
    J(0, 2) = -fx * X / (Z * Z);         // ∂u/∂tz

    J(1, 0) = 0;                         // ∂v/∂tx
    J(1, 1) = fy / Z;                    // ∂v/∂ty
    J(1, 2) = -fy * Y / (Z * Z);         // ∂v/∂tz

    // Rotational derivatives (rotation around x, y, z axes)
    J(0, 3) = -fx * X * Y / (Z * Z);           // ∂u/∂rx
    J(0, 4) = fx * (1 + X * X / (Z * Z));      // ∂u/∂ry
    J(0, 5) = -fx * Y / Z;                     // ∂u/∂rz

    J(1, 3) = -fy * (1 + Y * Y / (Z * Z));     // ∂v/∂rx
    J(1, 4) = fy * X * Y / (Z * Z);            // ∂v/∂ry
    J(1, 5) = fy * X / Z;                      // ∂v/∂rz

    return J;
}

Eigen::MatrixXd ReprojectionBasedConfidenceEstimator::calculateLandmarkJacobian(
    const Landmark* landmark,
    const Eigen::Matrix4d& T_lar_from_camera,
    const Frame& frame) const {

    // Transform landmark to camera coordinates
    Eigen::Vector4d world_point(landmark->position[0], landmark->position[1],
                               landmark->position[2], 1.0);
    Eigen::Vector4d camera_point = T_lar_from_camera.inverse() * world_point;

    // ARKit to OpenCV coordinate conversion (same as other functions)
    double X = camera_point[0];
    double Y = -camera_point[1];  // Flip Y: ARKit Y-up -> OpenCV Y-down
    double Z = -camera_point[2];  // Flip Z: ARKit Z-backward -> OpenCV Z-forward

    // Get camera intrinsics
    double fx = frame.intrinsics(0, 0);
    double fy = frame.intrinsics(1, 1);

    // Jacobian matrix (2 x 3): [∂u/∂landmark, ∂v/∂landmark]
    // landmark = [lx, ly, lz] in LAR world coordinates
    Eigen::MatrixXd J(2, 3);

    // Avoid division by zero
    if (std::abs(Z) < 1e-6) {
        J.setZero();
        return J;
    }

    // Get camera transformation matrix in ARKit coordinates
    Eigen::Matrix4d T_camera_from_lar = T_lar_from_camera.inverse();
    Eigen::Matrix3d R_arkit = T_camera_from_lar.block<3,3>(0,0);

    // Create coordinate conversion matrix: ARKit -> OpenCV
    Eigen::Matrix3d coord_flip = Eigen::Matrix3d::Identity();
    coord_flip(1, 1) = -1.0;  // Y flip: ARKit Y-up -> OpenCV Y-down
    coord_flip(2, 2) = -1.0;  // Z flip: ARKit Z-backward -> OpenCV Z-forward

    // Rotation matrix in OpenCV coordinates
    Eigen::Matrix3d R_opencv = coord_flip * R_arkit;

    // Chain rule: ∂projection/∂landmark = ∂projection/∂camera_coords * ∂camera_coords/∂landmark

    // ∂projection/∂camera_coords (standard OpenCV projection derivatives)
    Eigen::MatrixXd J_proj_camera(2, 3);
    J_proj_camera(0, 0) = fx / Z;                    // ∂u/∂X
    J_proj_camera(0, 1) = 0;                         // ∂u/∂Y
    J_proj_camera(0, 2) = -fx * X / (Z * Z);         // ∂u/∂Z

    J_proj_camera(1, 0) = 0;                         // ∂v/∂X
    J_proj_camera(1, 1) = fy / Z;                    // ∂v/∂Y
    J_proj_camera(1, 2) = -fy * Y / (Z * Z);         // ∂v/∂Z

    // ∂camera_coords/∂landmark (using OpenCV-converted rotation matrix)
    // camera_coords_opencv = R_opencv * landmark_lar
    Eigen::Matrix3d J_camera_landmark = R_opencv;

    // Final Jacobian: chain rule
    J = J_proj_camera * J_camera_landmark;

    return J;
}

double ReprojectionBasedConfidenceEstimator::calculateFeatureWeight(
    const cv::KeyPoint& keypoint,
    const Landmark* landmark,
    double reprojection_error) const {

    // Base weight
    double weight = 1.0;

    // Scale-based weight (larger scale features are more reliable)
    double scale_weight = std::min(2.0, (keypoint.octave + 2.0) / 3.0);

    // Error-based weight (lower error = higher weight)
    double error_weight = std::exp(-reprojection_error / 2.0);  // 2 pixel decay

    // Response-based weight (stronger features get more weight)
    double response_weight = std::min(2.0, keypoint.response / 0.01);  // Normalize by typical response

    return weight * scale_weight * error_weight * response_weight;
}

Eigen::Vector2d ReprojectionBasedConfidenceEstimator::projectToPixel(
    const Eigen::Vector3d& camera_point,
    const Frame& frame) const {

    // Use actual camera intrinsics from frame (3x3 matrix)
    double fx = frame.intrinsics(0, 0);
    double fy = frame.intrinsics(1, 1);
    double cx = frame.intrinsics(0, 2);
    double cy = frame.intrinsics(1, 2);


    double X = camera_point[0];
    double Y = camera_point[1];
    double Z = camera_point[2];

    // Standard OpenCV perspective projection
    // (coordinates already converted from ARKit to OpenCV in calling function)
    double u = fx * X / Z + cx;
    double v = fy * Y / Z + cy;  // Standard projection since Y is already flipped

    return Eigen::Vector2d(u, v);
}

} // namespace lar