#ifndef LAR_TRACKING_SIFT_H
#define LAR_TRACKING_SIFT_H

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include "lar/tracking/sift/sift_common.h"

// Conditional compilation: Use Metal on Apple platforms, CPU otherwise
#define LAR_USE_METAL_SIFT 1

namespace lar {

/**
 * SIFT (Scale-Invariant Feature Transform) detector and descriptor
 *
 * Uses Pimpl idiom to hide implementation details.
 * Implementation automatically selected at compile time:
 * - Metal backend on Apple platforms (when LAR_USE_METAL_SIFT=1)
 * - CPU backend otherwise
 */
class SIFT {
public:
    explicit SIFT(const SIFTConfig& config);
    ~SIFT();

    // Delete copy operations (implementation is not copyable)
    SIFT(const SIFT&) = delete;
    SIFT& operator=(const SIFT&) = delete;

    // Allow move operations
    SIFT(SIFT&& other) noexcept;
    SIFT& operator=(SIFT&& other) noexcept;

    void detectAndCompute(cv::InputArray image, cv::InputArray mask,
                         std::vector<cv::KeyPoint>& keypoints,
                         cv::OutputArray descriptors,
                         bool useProvidedKeypoints = false);

    int descriptorSize() const;
    int descriptorType() const;
    int defaultNorm() const;

private:
    struct Impl;  // Forward declaration - implementation in separate files
    Impl* impl_;
};

} // namespace lar

#endif // LAR_TRACKING_SIFT_H