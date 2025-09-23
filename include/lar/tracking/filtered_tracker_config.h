#ifndef LAR_TRACKING_FILTERED_TRACKER_CONFIG_H
#define LAR_TRACKING_FILTERED_TRACKER_CONFIG_H

namespace lar {

/**
 * Configuration object for FilteredTracker with all tunable parameters.
 * Centralizes all configuration to make tuning and experimentation easier.
 */
struct FilteredTrackerConfig {
    // === Core Timing ===
    double measurement_interval_seconds = 2.0;  // LAR measurement update interval

    // === Initial Uncertainties ===
    double initial_position_uncertainty = 1.0;   // Initial position uncertainty (meters)
    double initial_orientation_uncertainty = 0.1; // Initial orientation uncertainty (radians)

    // === Outlier Detection ===
    double outlier_threshold = 12.592;  // Chi-squared threshold for 6 DOF, 95% confidence
    bool enable_outlier_detection = true;

    // === Measurement Noise Estimation ===
    double base_position_noise = 0.1;      // Base position uncertainty (10cm)
    double base_orientation_noise = 0.05;  // Base orientation uncertainty (~3 degrees)
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
               min_confidence_factor > 0.0 &&
               min_inliers_for_noise > 0.0 &&
               max_inliers_for_confidence > 0.0 &&
               min_inliers_for_tracking >= 3;  // Need at least 3 points for pose estimation
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