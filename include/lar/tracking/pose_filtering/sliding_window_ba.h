#ifndef LAR_TRACKING_POSE_FILTERING_SLIDING_WINDOW_BA_H
#define LAR_TRACKING_POSE_FILTERING_SLIDING_WINDOW_BA_H

#include "pose_filter_strategy_base.h"
#include "../measurement_context.h"
#include "../filtered_tracker_config.h"
#include "lar/mapping/frame.h"
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
 * Each keyframe stores its own camera intrinsics for accurate optimization
 */
struct Keyframe {
    // Constructor from MeasurementContext
    Keyframe(size_t keyframe_id, const MeasurementContext& context)
        : id(keyframe_id)
        , timestamp(std::chrono::steady_clock::now().time_since_epoch().count())
        , pose(context.measured_pose)
        , intrinsics(context.frame->intrinsics)
        , inliers(context.inliers)
        , covariance(context.measurement_noise)
        , confidence(context.confidence)
        , is_marginalized(false) {
    }

    size_t id;
    double timestamp;
    Eigen::Matrix4d pose;                                                            // T_world_from_camera
    Eigen::Matrix3d intrinsics;
    std::shared_ptr<std::vector<std::pair<Landmark*, cv::KeyPoint>>> inliers;
    Eigen::MatrixXd covariance;                                                      // 6x6 pose covariance
    double confidence = 0.0;                                                         // Measurement confidence from LAR
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
    void initialize(const MeasurementContext& context, const FilteredTrackerConfig& config) override;
    void predict(const Eigen::Matrix4d& motion, double dt, const FilteredTrackerConfig& config) override;
    void update(const MeasurementContext& context,
                const FilteredTrackerConfig& config) override;
    PoseState getState() const override;
    Eigen::MatrixXd getCovariance() const override;
    double getPositionUncertainty() const override;
    bool isInitialized() const override;
    void reset() override;

private:
    // === Configuration ===
    FilteredTrackerConfig config_;         // Store entire configuration

    // === State ===
    bool initialized_ = false;
    PoseState state_;              // Current pose estimate
    Eigen::MatrixXd covariance_;   // Current 6x6 covariance

    // === Sliding window ===
    std::deque<std::shared_ptr<Keyframe>> keyframes_;  // Active keyframes
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
     * Marginalize oldest keyframe when window is full
     */
    void marginalizeOldestKeyframe();


    /**
     * Apply marginal prior constraint to connected keyframe
     */
    void applyMarginalPrior(size_t target_keyframe_id, const Eigen::MatrixXd& marginal_info, const Eigen::Matrix4d& marginalized_pose);

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

};

} // namespace lar

#endif /* LAR_TRACKING_POSE_FILTERING_SLIDING_WINDOW_BA_H */