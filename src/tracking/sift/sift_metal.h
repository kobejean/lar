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
#include "lar/tracking/sift/sift_common.h"

namespace lar {

// Forward declarations
class SIFTMetal;

/// Metal-accelerated SIFT processor using GPU compute pipelines
/// Implements full SIFT pipeline: Gaussian pyramid, DoG pyramid, extrema detection, and descriptors
/// Resources are owned by this instance (RAII), enabling safe multi-threading
class SIFTMetal {
public:
    SIFTMetal(const SIFTConfig& config);
    ~SIFTMetal();

    // Delete copy operations (Metal resources shouldn't be copied)
    SIFTMetal(const SIFTMetal&) = delete;
    SIFTMetal& operator=(const SIFTMetal&) = delete;

    // Allow move operations for flexibility
    SIFTMetal(SIFTMetal&& other) noexcept;
    SIFTMetal& operator=(SIFTMetal&& other) noexcept;

    bool detectAndCompute(const cv::Mat& base,
                         std::vector<cv::KeyPoint>& keypoints,
                         cv::OutputArray descriptors);

    bool isAvailable() const;

    struct Impl;
    Impl* impl_;

private:
    SIFTConfig config_;

    // Metal command encoding helper methods
    void encodeResizeCommand(
        id cmdBuf,
        id sourceTexture,
        id destTexture);

    void encodeInitialBlurCommand(
        id cmdBuf,
        id imageTexture,
        id tempTexture,
        id destTexture,
        int level);

    void encodeBlurAndDoGCommand(
        id cmdBuf,
        id prevGaussTexture,
        id gaussTexture,
        id dogTexture,
        int level);

    void encodeExtremaDetectionCommand(
        id cmdBuf,
        id dogTextureBelow,
        id dogTextureCenter,
        id dogTextureAbove,
        id extremaBitarray);

    // Octave construction strategies
    void encodeStandardOctaveConstruction(
        id cmdBuf,
        int octave);

    void encodeBatchedOctaveConstruction(
        id cmdBuf,
        int octave);
};

} // namespace lar

#endif // LAR_TRACKING_METAL_SIFT_H