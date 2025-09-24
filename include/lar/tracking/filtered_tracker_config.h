#ifndef LAR_TRACKING_FILTERED_TRACKER_CONFIG_H
#define LAR_TRACKING_FILTERED_TRACKER_CONFIG_H

namespace lar {

/**
 * Configuration object for FilteredTracker with all tunable parameters.
 * Centralizes all configuration to make tuning and experimentation easier.
 */
struct FilteredTrackerConfig {
    // === Filter Strategy Selection ===
    enum class FilterStrategy {
        EXTENDED_KALMAN_FILTER,
        SLIDING_WINDOW_BA,
        AVERAGING,
        PASS_THROUGH
    };
    FilterStrategy filter_strategy = FilterStrategy::SLIDING_WINDOW_BA;  // Default to Sliding Window BA

    // === Core Timing ===
    double measurement_interval_seconds = 2.0;  // LAR measurement update interval

    // === Initial Uncertainties ===
    double initial_position_uncertainty = 1.0;   // Initial position uncertainty (meters)
    double initial_orientation_uncertainty = 0.1; // Initial orientation uncertainty (radians)

    // === Outlier Detection ===
    double outlier_threshold = 12.592;  // Chi-squared threshold for 6 DOF, 95% confidence
    bool enable_outlier_detection = true;

    // === Measurement Noise Estimation ===
    double base_position_noise = 0.2;      // Base position uncertainty (10cm)
    double base_orientation_noise = 0.1;  // Base orientation uncertainty (~6 degrees)
    double landmark_position_noise = 0.1; // Landmark 3D position uncertainty from mapping (10cm)
    double min_confidence_factor = 0.1;    // Minimum confidence to prevent division by zero
    double min_inliers_for_noise = 20.0;   // Minimum inliers for noise calculation

    // === Process Noise Scaling ===
    double motion_position_noise_scale = 0.01;   // Position noise per meter of motion
    double motion_orientation_noise_scale = 0.005; // Orientation noise per meter of motion
    double time_position_noise_scale = 0.001;    // Position noise per second
    double time_orientation_noise_scale = 0.0005; // Orientation noise per second

    // === Confidence Estimation ===
    double max_inliers_for_confidence = 50.0;  // Inlier count for 100% confidence
    double spatial_distribution_scale = 2.0;   // Scale factor for spatial distribution
    double min_inliers_for_tracking = 4;       // Minimum inliers to consider valid

    // === Distance Stratified Confidence Estimation ===
    double near_threshold = 5.0;               // Distance threshold for near features (meters)
    double far_threshold = 25.0;               // Distance threshold for far features (meters)

    // === Historical Anchor Constraints ===
    bool enable_anchor_constraints = false;    // Enable/disable anchor constraints
    double anchor_min_confidence = 0.6;        // Minimum confidence to qualify as anchor
    size_t anchor_min_features = 40;           // Minimum features to qualify as anchor
    double anchor_min_spacing = 2.0;           // Minimum distance between anchors (meters)
    size_t anchor_max_count = 20;              // Maximum number of anchors to keep
    double anchor_max_distance = 10.0;         // Maximum distance to consider anchors (meters)
    double anchor_max_age = 120.0;             // Maximum age of anchors to use (seconds)
    double anchor_distance_scale = 5.0;        // Distance scaling for anchor relevance
    double anchor_time_scale = 60.0;           // Time scaling for anchor relevance
    double anchor_min_relevance = 0.1;         // Minimum relevance to use anchor
    double anchor_max_correction = 0.1;        // Maximum rotation correction per update (radians)
    double anchor_correction_strength = 0.2;   // Strength of anchor corrections (0-1)

    // === Reprojection-Based Confidence Estimation ===
    double reprojection_expected_noise = 2.0;      // Expected reprojection error (pixels)
    double reprojection_min_position_noise = 0.01; // Minimum position uncertainty (meters)
    double reprojection_min_orientation_noise = 0.01; // Minimum orientation uncertainty (radians)
    // Note: Camera intrinsics are now obtained directly from Frame object

    // === Debugging ===
    bool enable_debug_output = true;
    bool enable_coordinate_debugging = false;
    bool enable_motion_debugging = false;

    // === Animation (Legacy) ===
    bool use_animation = false;  // Enable/disable visual smoothing

    // === Validation ===
    /**
     * Validate configuration parameters
     * @return True if configuration is valid
     */
    bool isValid() const {
        return measurement_interval_seconds > 0.0 &&
               initial_position_uncertainty > 0.0 &&
               initial_orientation_uncertainty > 0.0 &&
               outlier_threshold > 0.0 &&
               base_position_noise > 0.0 &&
               base_orientation_noise > 0.0 &&
               landmark_position_noise > 0.0 &&
               min_confidence_factor > 0.0 &&
               min_inliers_for_noise > 0.0 &&
               max_inliers_for_confidence > 0.0 &&
               min_inliers_for_tracking >= 3 &&  // Need at least 3 points for pose estimation
               near_threshold > 0.0 &&
               far_threshold > near_threshold &&  // Far threshold must be greater than near
               anchor_min_confidence > 0.0 && anchor_min_confidence <= 1.0 &&
               anchor_min_features > 0 &&
               anchor_min_spacing > 0.0 &&
               anchor_max_count > 0 &&
               anchor_max_distance > 0.0 &&
               anchor_max_age > 0.0 &&
               anchor_distance_scale > 0.0 &&
               anchor_time_scale > 0.0 &&
               anchor_min_relevance > 0.0 && anchor_min_relevance <= 1.0 &&
               anchor_max_correction > 0.0 &&
               anchor_correction_strength > 0.0 && anchor_correction_strength <= 1.0 &&
               reprojection_expected_noise > 0.0 &&
               reprojection_min_position_noise > 0.0 &&
               reprojection_min_orientation_noise > 0.0;
    }

    /**
     * Create a conservative configuration (higher uncertainties, stricter thresholds)
     */
    static FilteredTrackerConfig createConservative() {
        FilteredTrackerConfig config;
        config.initial_position_uncertainty = 2.0;
        config.initial_orientation_uncertainty = 0.2;
        config.outlier_threshold = 9.488;  // 90% confidence instead of 95%
        config.base_position_noise = 0.2;
        config.base_orientation_noise = 0.1;
        config.landmark_position_noise = 0.1;  // Higher uncertainty for fast tracking
        return config;
    }

    /**
     * Create an aggressive configuration (lower uncertainties, looser thresholds)
     */
    static FilteredTrackerConfig createAggressive() {
        FilteredTrackerConfig config;
        config.initial_position_uncertainty = 0.5;
        config.initial_orientation_uncertainty = 0.05;
        config.outlier_threshold = 16.812;  // 99% confidence
        config.base_position_noise = 0.05;
        config.base_orientation_noise = 0.025;
        config.landmark_position_noise = 0.02;  // Lower uncertainty for precision tracking
        config.min_inliers_for_tracking = 6;
        return config;
    }

    /**
     * Create a debug configuration (verbose output, conservative settings)
     */
    static FilteredTrackerConfig createDebug() {
        FilteredTrackerConfig config = createConservative();
        config.enable_debug_output = true;
        config.enable_coordinate_debugging = true;
        config.enable_motion_debugging = true;
        return config;
    }
};

} // namespace lar

#endif /* LAR_TRACKING_FILTERED_TRACKER_CONFIG_H */