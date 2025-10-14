// Shared utilities for SIFT implementations
// Contains coordinate space conversions and helper functions used across CPU, SIMD, and Metal variants
#ifndef LAR_TRACKING_SIFT_COMMON_H
#define LAR_TRACKING_SIFT_COMMON_H

#include <opencv2/core.hpp>
#include <vector>

namespace lar {

// SIFT algorithm constants
// These values are based on the original SIFT paper (Lowe 2004)
// and are shared between CPU and Metal GPU implementations

// Descriptor parameters
constexpr int SIFT_DESCR_WIDTH = 4;
constexpr int SIFT_DESCR_HIST_BINS = 8;
constexpr float SIFT_INIT_SIGMA = 0.5f;

// Image border for extrema detection
constexpr int SIFT_IMG_BORDER = 5;

// Keypoint refinement
constexpr int SIFT_MAX_INTERP_STEPS = 5;

// Orientation histogram parameters
constexpr int SIFT_ORI_HIST_BINS = 36;
constexpr float SIFT_ORI_SIG_FCTR = 1.5f;
constexpr float SIFT_ORI_RADIUS = 4.5f;
constexpr float SIFT_ORI_PEAK_RATIO = 0.8f;

// Descriptor computation parameters
constexpr float SIFT_DESCR_SCL_FCTR = 3.f;
constexpr float SIFT_DESCR_MAG_THR = 0.2f;
constexpr float SIFT_INT_DESCR_FCTR = 512.f;

// Fixed-point scale factor (1 = floating-point, higher values for fixed-point arithmetic)
constexpr int SIFT_FIXPT_SCALE = 1;

struct SIFTConfig {
    cv::Size imageSize = cv::Size(0, 0);
    int nOctaveLayers = 3;
    double sigma = 1.6;
    bool enableUpsampling = false;
    double contrastThreshold = 0.02;
    double edgeThreshold = 10.0;
    int descriptorType = CV_8U;
    // derived members
    int firstOctave;
    int nOctaves;
    int nLevels;
    int threshold;

    SIFTConfig() = default;

    explicit SIFTConfig(cv::Size size)
        : imageSize(size)
        , nOctaveLayers(3)
        , sigma(1.6)
        , enableUpsampling(false)
        , contrastThreshold(0.04)
        , edgeThreshold(10.0)
        , descriptorType(CV_8U)
    {
        firstOctave = enableUpsampling ? -1 : 0;
        nOctaves = static_cast<int>(cvRound(std::log(std::min(imageSize.width, imageSize.height)) / std::log(2.0) - 2.0)) - firstOctave;
        nLevels = nOctaveLayers + 3;
        threshold = cvFloor(0.5 * contrastThreshold / nOctaveLayers * 255 * SIFT_FIXPT_SCALE);
    }

    static constexpr int DESCR_WIDTH = 4;
    static constexpr int DESCR_HIST_BINS = 8;
    static constexpr int DESCR_SIZE = DESCR_WIDTH * DESCR_WIDTH * DESCR_HIST_BINS;
};

inline cv::Point2f toFullResolution(cv::Point2f pt, int octave, int firstOctave) {
    float scale = 1.0f / (1 << (octave - firstOctave));
    return cv::Point2f(pt.x * scale, pt.y * scale);
}

inline cv::Point2f toOctaveSpace(cv::Point2f pt, int octave) {
    float scale = 1.0f / (1 << octave);
    return cv::Point2f(pt.x * scale, pt.y * scale);
}

inline float convertScale(float size, int octave) {
    return size * 0.5f / (1 << octave);
}

std::vector<float> createGaussianKernel(double sigma);

void calcSIFTDescriptor(
    const cv::Mat& img,
    cv::Point2f ptf,
    float ori,
    float scl,
    int d,
    int n,
    cv::Mat& dstMat,
    int row
);

bool adjustLocalExtrema(
    const std::vector<cv::Mat>& dog_pyr,
    cv::KeyPoint& kpt,
    int octave,
    int& layer,
    int& r,
    int& c,
    int nOctaveLayers,
    float contrastThreshold,
    float edgeThreshold,
    float sigma
);

float calcOrientationHist(
    const cv::Mat& img,
    cv::Point pt,
    int radius,
    float sigma,
    float* hist,
    int n
);

} // namespace lar

#endif // LAR_TRACKING_SIFT_COMMON_H