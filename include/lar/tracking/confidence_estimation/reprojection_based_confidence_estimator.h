#ifndef LAR_TRACKING_CONFIDENCE_ESTIMATION_REPROJECTION_BASED_CONFIDENCE_ESTIMATOR_H
#define LAR_TRACKING_CONFIDENCE_ESTIMATION_REPROJECTION_BASED_CONFIDENCE_ESTIMATOR_H

#include "confidence_estimator_base.h"
#include <vector>

namespace lar {

/**
 * Reprojection-based confidence estimator that uses actual reprojection errors,
 * feature scale information, and Fisher Information Matrix to estimate
 * pose uncertainty in a data-driven way.
 *
 * Based on approaches used in modern SLAM systems like ORB-SLAM3, VINS-Mono, etc.
 */
class ReprojectionBasedConfidenceEstimator : public ConfidenceEstimator {
public:
    ReprojectionBasedConfidenceEstimator() = default;
    virtual ~ReprojectionBasedConfidenceEstimator() = default;

    /**
     * Calculate confidence based on reprojection errors and feature quality
     */
    double calculateConfidence(
        const MeasurementContext& context,
        const FilteredTrackerConfig& config) const override;

    /**
     * Calculate measurement noise covariance using Fisher Information Matrix
     * from reprojection error Jacobians
     */
    Eigen::MatrixXd calculateMeasurementNoise(
        const MeasurementContext& context,
        const FilteredTrackerConfig& config) const override;


private:

    /**
     * Calculate reprojection errors for all inlier features
     */
    std::vector<double> calculateReprojectionErrors(
        const std::vector<std::pair<Landmark*, cv::KeyPoint>>& inliers,
        const Eigen::Matrix4d& T_lar_from_camera,
        const Frame& frame) const;

    /**
     * Calculate quality metric based on feature scales (SIFT octaves)
     */
    double calculateScaleQuality(
        const std::vector<std::pair<Landmark*, cv::KeyPoint>>& inliers) const;

    /**
     * Calculate RMSE of reprojection errors
     */
    double calculateRMSE(const std::vector<double>& errors) const;

    /**
     * Calculate pose covariance using Fisher Information Matrix
     */
    Eigen::MatrixXd calculateFisherInformationCovariance(
        const std::vector<std::pair<Landmark*, cv::KeyPoint>>& inliers,
        const Eigen::Matrix4d& T_lar_from_camera,
        const Frame& frame,
        const FilteredTrackerConfig& config) const;

    /**
     * Calculate Jacobian of reprojection w.r.t. 6DOF pose
     */
    Eigen::MatrixXd calculateReprojectionJacobian(
        const Landmark* landmark,
        const Eigen::Matrix4d& T_lar_from_camera,
        const Frame& frame) const;

    /**
     * Calculate Jacobian of reprojection w.r.t. 3D landmark position
     */
    Eigen::MatrixXd calculateLandmarkJacobian(
        const Landmark* landmark,
        const Eigen::Matrix4d& T_lar_from_camera,
        const Frame& frame) const;

    /**
     * Calculate weight for individual feature based on quality metrics
     */
    double calculateFeatureWeight(
        const cv::KeyPoint& keypoint,
        const Landmark* landmark,
        double reprojection_error) const;

    /**
     * Project 3D point to pixel coordinates
     */
    Eigen::Vector2d projectToPixel(
        const Eigen::Vector3d& camera_point,
        const Frame& frame) const;

};

} // namespace lar

#endif /* LAR_TRACKING_CONFIDENCE_ESTIMATION_REPROJECTION_BASED_CONFIDENCE_ESTIMATOR_H */