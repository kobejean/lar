#ifndef LAR_TRACKING_POSE_FILTERING_SLIDING_WINDOW_BA_H
#define LAR_TRACKING_POSE_FILTERING_SLIDING_WINDOW_BA_H

#include "pose_filter_strategy_base.h"
#include "../measurement_context.h"
#include <deque>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <opencv2/opencv.hpp>

// Forward declarations for g2o
namespace g2o {
    class SparseOptimizer;
    class VertexSE3Expmap;
    class VertexPointXYZ;
}

namespace lar {

// Forward declarations
class Landmark;

/**
 * Keyframe for sliding window bundle adjustment
 * Uses inlier format to avoid unnecessary allocations
 */
struct Keyframe {
    size_t id;
    double timestamp;
    Eigen::Matrix4d pose;                                                            // T_world_from_camera
    std::shared_ptr<std::vector<std::pair<Landmark*, cv::KeyPoint>>> inliers;       // Shared inlier landmark-keypoint pairs (zero-copy from MeasurementContext)
    Eigen::MatrixXd covariance;                                                      // 6x6 pose covariance
    bool is_marginalized = false;
};

/**
 * Sliding Window Bundle Adjustment pose filter strategy.
 *
 * This filter maintains a sliding window of keyframes and jointly optimizes
 * their poses along with observed landmark positions using g2o.
 *
 * Key features:
 * - Maintains N recent keyframes in optimization window
 * - Jointly optimizes poses and landmark positions
 * - Marginalizes old keyframes to maintain computational efficiency
 * - Uses g2o for efficient sparse optimization
 */
class SlidingWindowBA : public PoseFilterStrategy {
public:
    SlidingWindowBA();
    virtual ~SlidingWindowBA();

    // PoseFilterStrategy interface
    void initialize(const Eigen::Matrix4d& initial_pose, const FilteredTrackerConfig& config) override;
    void predict(const Eigen::Matrix4d& motion, double dt, const FilteredTrackerConfig& config) override;
    void update(const MeasurementContext& context,
                const Eigen::MatrixXd& measurement_noise,
                const FilteredTrackerConfig& config) override;
    PoseState getState() const override;
    Eigen::MatrixXd getCovariance() const override;
    double getPositionUncertainty() const override;
    bool isInitialized() const override;
    void reset() override;

    /**
     * Set camera intrinsics for reprojection constraints
     */
    void setCameraIntrinsics(double fx, double fy, double cx, double cy);

private:
    // === Configuration ===
    size_t max_window_size_ = 10;          // Maximum keyframes in window
    size_t min_observations_ = 20;         // Minimum observations to create keyframe
    double keyframe_distance_ = 0.5;       // Minimum distance between keyframes (meters)
    double keyframe_angle_ = 15.0;         // Minimum angle between keyframes (degrees)
    int optimization_iterations_ = 10;     // g2o optimization iterations

    // === Camera parameters ===
    double fx_ = 600.0, fy_ = 600.0;      // Focal lengths
    double cx_ = 320.0, cy_ = 240.0;      // Principal point

    // === State ===
    bool initialized_ = false;
    PoseState current_state_;              // Current pose estimate
    Eigen::MatrixXd current_covariance_;   // Current 6x6 covariance

    // === Sliding window ===
    std::deque<std::shared_ptr<Keyframe>> keyframes_;  // Active keyframes
    std::shared_ptr<Keyframe> current_frame_;          // Current (non-keyframe) frame
    size_t next_keyframe_id_ = 0;

    // === Landmark tracking ===
    std::unordered_map<Landmark*, size_t> landmark_observations_;  // Track observation counts
    std::unordered_set<Landmark*> active_landmarks_;               // Landmarks in current window

    // === Motion model ===
    Eigen::Matrix4d accumulated_motion_ = Eigen::Matrix4d::Identity();  // Motion since last keyframe

    // === g2o optimizer ===
    std::unique_ptr<g2o::SparseOptimizer> optimizer_;
    std::unordered_map<size_t, g2o::VertexSE3Expmap*> pose_vertices_;
    std::unordered_map<Landmark*, g2o::VertexPointXYZ*> landmark_vertices_;

    // === Keyframe Management ===

    /**
     * Check if current frame should be a keyframe
     */
    bool shouldCreateKeyframe(const Eigen::Matrix4d& current_pose) const;

    /**
     * Create new keyframe from current state and MeasurementContext
     */
    void createKeyframe(const Eigen::Matrix4d& pose, const Eigen::MatrixXd& covariance);

    /**
     * Process inliers from MeasurementContext and store them efficiently
     */
    void processInliers(const MeasurementContext& context);

    /**
     * Marginalize oldest keyframe when window is full
     */
    void marginalizeOldestKeyframe();

    // === Bundle Adjustment ===

    /**
     * Build optimization graph with current keyframes and landmarks
     */
    void buildOptimizationGraph(const FilteredTrackerConfig& config);

    /**
     * Run g2o optimization
     */
    void optimize();

    /**
     * Update state from optimized graph
     */
    void updateStateFromOptimization();

    // === Covariance Extraction ===

    /**
     * Extract covariance matrix from bundle adjustment optimization
     */
    Eigen::MatrixXd extractCovarianceFromBA(size_t keyframe_id) const;

    /**
     * Calculate information matrix from covariance
     */
    Eigen::MatrixXd calculateInformationMatrix(const Eigen::MatrixXd& covariance) const;

    // === Projection Utilities ===

    /**
     * Project 3D point to image coordinates
     */
    Eigen::Vector2d project(const Eigen::Vector3d& point_camera) const;

    /**
     * Calculate reprojection Jacobian
     */
    Eigen::Matrix<double, 2, 6> calculateReprojectionJacobian(
        const Eigen::Vector3d& point_world,
        const Eigen::Matrix4d& T_camera_from_world) const;
};

} // namespace lar

#endif /* LAR_TRACKING_POSE_FILTERING_SLIDING_WINDOW_BA_H */