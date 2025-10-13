// Metal-accelerated SIFT processor with RAII resource management
// This class encapsulates all Metal GPU processing for SIFT feature detection
// Each instance owns its own Metal resources, enabling multi-threaded operation
#ifndef LAR_TRACKING_METAL_SIFT_H
#define LAR_TRACKING_METAL_SIFT_H

#ifdef __OBJC__
#import <Metal/Metal.h>
#else
// Forward declarations for C++ compatibility
typedef struct objc_object *id;
#endif

#include <opencv2/core.hpp>
#include <vector>
#include "lar/tracking/sift/sift_config.h"

namespace lar {

// Forward declarations
class MetalSIFT;

/// Metal-accelerated SIFT processor using GPU compute pipelines
/// Implements full SIFT pipeline: Gaussian pyramid, DoG pyramid, extrema detection, and descriptors
/// Resources are owned by this instance (RAII), enabling safe multi-threading
class MetalSIFT {
public:
    MetalSIFT(const SIFTConfig& config);
    ~MetalSIFT();

    // Delete copy operations (Metal resources shouldn't be copied)
    MetalSIFT(const MetalSIFT&) = delete;
    MetalSIFT& operator=(const MetalSIFT&) = delete;

    // Allow move operations for flexibility
    MetalSIFT(MetalSIFT&& other) noexcept;
    MetalSIFT& operator=(MetalSIFT&& other) noexcept;

    bool detectAndCompute(const cv::Mat& base,
                         std::vector<cv::KeyPoint>& keypoints,
                         cv::OutputArray descriptors,
                         int nOctaves);

    bool isAvailable() const;

    struct Impl;
    Impl* impl_;

private:
    SIFTConfig config_;
};

} // namespace lar

#endif // LAR_TRACKING_METAL_SIFT_H