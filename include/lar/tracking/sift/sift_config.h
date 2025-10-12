// SIFT algorithm configuration parameters
// Centralizes all tunable parameters for SIFT feature detection and description
#ifndef LAR_TRACKING_SIFT_CONFIG_H
#define LAR_TRACKING_SIFT_CONFIG_H

namespace lar {

/// Configuration parameters for SIFT algorithm
/// Controls scale space construction, keypoint detection, and descriptor computation
struct SiftConfig {
    // Scale space parameters
    int nOctaveLayers = 3;          ///< Number of layers per octave (typically 3)
    double sigma = 1.6;             ///< Base sigma for Gaussian blur

    // Upsampling configuration
    bool enableUpsampling = false;  ///< If true, use firstOctave = -1 (2x upsampling)

    /// Returns the first octave index based on upsampling configuration
    /// @return -1 if upsampling enabled (processes 2x upscaled image), 0 otherwise
    int firstOctave() const {
        return enableUpsampling ? -1 : 0;
    }

    // Detection thresholds
    double contrastThreshold = 0.04;  ///< Contrast threshold for keypoint filtering
    double edgeThreshold = 10.0;      ///< Edge response threshold (ratio of principal curvatures)

    // Descriptor parameters (compile-time constants)
    static constexpr int DESCR_WIDTH = 4;       ///< Descriptor grid width (4x4 = 16 histograms)
    static constexpr int DESCR_HIST_BINS = 8;   ///< Bins per histogram (8 orientations)
    static constexpr int DESCR_SIZE = DESCR_WIDTH * DESCR_WIDTH * DESCR_HIST_BINS;  ///< Total descriptor size (128)
};

} // namespace lar

#endif // LAR_TRACKING_SIFT_CONFIG_H