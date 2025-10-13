// SIFT algorithm configuration parameters
// Centralizes all tunable parameters for SIFT feature detection and description
#ifndef LAR_TRACKING_SIFT_CONFIG_H
#define LAR_TRACKING_SIFT_CONFIG_H

#include <cmath>
#include <algorithm>
#include <opencv2/core/types.hpp>

namespace lar {

/// Configuration parameters for SIFT algorithm
/// Controls scale space construction, keypoint detection, and descriptor computation
struct SIFTConfig {
    // Image dimensions (required for buffer pre-allocation)
    cv::Size imageSize = cv::Size(0, 0);  ///< Expected input image dimensions (width, height)

    // Scale space parameters
    int nOctaveLayers = 3;          ///< Number of layers per octave (typically 3)
    double sigma = 1.6;             ///< Base sigma for Gaussian blur

    // Upsampling configuration
    bool enableUpsampling = false;  ///< If true, use firstOctave = -1 (2x upsampling)

    // Detection thresholds
    double contrastThreshold = 0.04;  ///< Contrast threshold for keypoint filtering
    double edgeThreshold = 10.0;      ///< Edge response threshold (ratio of principal curvatures)

    // Descriptor parameters
    int descriptorType = 5;           ///< Descriptor type: CV_32F (5) or CV_8U (0)

    /// Default constructor
    SIFTConfig() = default;

    /// Convenience constructor with image size and sensible defaults
    /// @param size Expected input image dimensions
    explicit SIFTConfig(cv::Size size)
        : imageSize(size)
        , nOctaveLayers(3)
        , sigma(1.6)
        , enableUpsampling(false)
        , contrastThreshold(0.04)
        , edgeThreshold(10.0)
        , descriptorType(5)  // CV_32F
    {}

    /// Returns the first octave index based on upsampling configuration
    /// @return -1 if upsampling enabled (processes 2x upscaled image), 0 otherwise
    int firstOctave() const {
        return enableUpsampling ? -1 : 0;
    }

    // Descriptor parameters (compile-time constants)
    static constexpr int DESCR_WIDTH = 4;       ///< Descriptor grid width (4x4 = 16 histograms)
    static constexpr int DESCR_HIST_BINS = 8;   ///< Bins per histogram (8 orientations)
    static constexpr int DESCR_SIZE = DESCR_WIDTH * DESCR_WIDTH * DESCR_HIST_BINS;  ///< Total descriptor size (128)

    /// Compute number of octaves for given base image dimensions
    /// @param baseWidth Width of base image after initial processing
    /// @param baseHeight Height of base image after initial processing
    /// @return Number of octaves to process
    int computeNumOctaves(int baseWidth, int baseHeight) const {
        return static_cast<int>(std::round(std::log(std::min(baseWidth, baseHeight)) / std::log(2.0) - 2.0)) - firstOctave();
    }

    /// Get number of pyramid levels per octave (includes extra layers for DoG)
    int pyramidLevels() const {
        return nOctaveLayers + 3;
    }
};

} // namespace lar

#endif // LAR_TRACKING_SIFT_CONFIG_H