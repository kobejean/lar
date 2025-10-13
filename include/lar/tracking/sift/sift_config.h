// SIFT algorithm configuration parameters
// Centralizes all tunable parameters for SIFT feature detection and description
#ifndef LAR_TRACKING_SIFT_CONFIG_H
#define LAR_TRACKING_SIFT_CONFIG_H

#include <cmath>
#include <algorithm>
#include <opencv2/core/types.hpp>

namespace lar {

struct SIFTConfig {
    cv::Size imageSize = cv::Size(0, 0);
    int nOctaveLayers = 3;
    double sigma = 1.6;
    bool enableUpsampling = false;
    double contrastThreshold = 0.02;
    double edgeThreshold = 10.0;
    int descriptorType = CV_8U;

    SIFTConfig() = default;

    explicit SIFTConfig(cv::Size size)
        : imageSize(size)
        , nOctaveLayers(3)
        , sigma(1.6)
        , enableUpsampling(false)
        , contrastThreshold(0.04)
        , edgeThreshold(10.0)
        , descriptorType(CV_8U)
    {}

    int firstOctave() const { return enableUpsampling ? -1 : 0; }

    static constexpr int DESCR_WIDTH = 4;
    static constexpr int DESCR_HIST_BINS = 8;
    static constexpr int DESCR_SIZE = DESCR_WIDTH * DESCR_WIDTH * DESCR_HIST_BINS;

    int computeNumOctaves(int baseWidth, int baseHeight) const {
        return static_cast<int>(std::round(std::log(std::min(baseWidth, baseHeight)) / std::log(2.0) - 2.0)) - firstOctave();
    }

    int pyramidLevels() const { return nOctaveLayers + 3; }
};

} // namespace lar

#endif // LAR_TRACKING_SIFT_CONFIG_H