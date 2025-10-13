// Shared utilities for SIFT implementations
// Contains coordinate space conversions and helper functions used across CPU, SIMD, and Metal variants
#ifndef LAR_TRACKING_SIFT_COMMON_H
#define LAR_TRACKING_SIFT_COMMON_H

#include "lar/tracking/sift/sift_config.h"
#include <opencv2/core.hpp>
#include <vector>

namespace lar {
namespace sift_common {

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

} // namespace sift_common
} // namespace lar

#endif // LAR_TRACKING_SIFT_COMMON_H