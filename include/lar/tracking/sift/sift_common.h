// Shared utilities for SIFT implementations
// Contains coordinate space conversions and helper functions used across CPU, SIMD, and Metal variants
#ifndef LAR_TRACKING_SIFT_COMMON_H
#define LAR_TRACKING_SIFT_COMMON_H

#include "lar/tracking/sift/sift_config.h"
#include <opencv2/core.hpp>
#include <vector>

namespace lar {
namespace sift_common {

// ============================================================================
// Coordinate Space Conversions
// ============================================================================

/// Convert point from detection space to full-resolution space
/// Detection space: where extrema are found (octave resolution)
/// Full-resolution: where keypoints are stored (scaled by 1<<(octave-firstOctave))
/// @param pt Point in detection space
/// @param octave Current octave index
/// @param firstOctave First octave index (-1 for upsampling, 0 otherwise)
/// @return Point in full-resolution space
inline cv::Point2f toFullResolution(cv::Point2f pt, int octave, int firstOctave) {
    float scale = 1.0f / (1 << (octave - firstOctave));
    return cv::Point2f(pt.x * scale, pt.y * scale);
}

/// Convert point from full-resolution space to octave space
/// Octave space: where descriptors are computed (scaled by 1<<octave)
/// @param pt Point in full-resolution space
/// @param octave Current octave index
/// @return Point in octave space
inline cv::Point2f toOctaveSpace(cv::Point2f pt, int octave) {
    float scale = 1.0f / (1 << octave);
    return cv::Point2f(pt.x * scale, pt.y * scale);
}

/// Convert keypoint scale from full-resolution to octave space
/// @param size Keypoint size in full-resolution space
/// @param octave Current octave index
/// @return Scale value for descriptor computation in octave space
inline float convertScale(float size, int octave) {
    return size * 0.5f / (1 << octave);
}

// ============================================================================
// Gaussian Kernel Utilities
// ============================================================================

/// Create 1D Gaussian kernel using OpenCV's bit-exact implementation
/// @param sigma Standard deviation of Gaussian
/// @return 1D Gaussian kernel weights (symmetric, normalized)
std::vector<float> createGaussianKernel(double sigma);

// ============================================================================
// SIFT Descriptor Computation
// ============================================================================

/// Compute SIFT descriptor for a single keypoint
/// Implements the standard SIFT descriptor: 4×4 grid of 8-bin orientation histograms
/// @param img Gaussian image at appropriate scale (must be in octave space)
/// @param ptf Keypoint position (must be in octave space)
/// @param ori Primary orientation in degrees (0-360, inverted convention)
/// @param scl Scale factor (keypoint size * 0.5, in octave space)
/// @param d Descriptor width (typically 4)
/// @param n Number of histogram bins (typically 8)
/// @param dstMat Output descriptor matrix (row will be written)
/// @param row Row index in dstMat to write descriptor
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

// ============================================================================
// Keypoint Refinement Helpers
// ============================================================================

/// Adjust local extrema position using quadratic interpolation
/// Implements Brown & Lowe's sub-pixel refinement via 3D quadratic fitting
/// @param dog_pyr Difference-of-Gaussian pyramid
/// @param kpt Keypoint to refine (updated in-place)
/// @param octave Current octave index
/// @param layer Current layer index within octave
/// @param r Border size
/// @param nOctaveLayers Number of layers per octave
/// @param contrastThreshold Contrast threshold for rejection
/// @param edgeThreshold Edge threshold for rejection
/// @param sigma Base sigma value
/// @return true if refinement succeeded, false if keypoint should be rejected
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

/// Calculate orientation histogram for keypoint orientation assignment
/// Computes gradient orientation histogram in circular region around keypoint
/// @param img Gaussian image at appropriate scale
/// @param pt Keypoint position
/// @param radius Radius of circular region
/// @param sigma Sigma for Gaussian weighting
/// @param hist Output histogram (36 bins for 10-degree intervals)
/// @param n Size of hist array (typically 36)
/// @return Maximum histogram value
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