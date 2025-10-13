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

namespace lar {

// Forward declarations
class MetalSIFT;

/// Metal-accelerated SIFT processor using GPU compute pipelines
/// Implements full SIFT pipeline: Gaussian pyramid, DoG pyramid, extrema detection, and descriptors
/// Resources are owned by this instance (RAII), enabling safe multi-threading
class MetalSIFT {
public:
    /// Constructor - initializes Metal device, command queue, and compute pipelines
    /// @param nOctaveLayers Number of layers per octave (typically 3)
    /// @param contrastThreshold Contrast threshold for keypoint filtering
    /// @param edgeThreshold Edge response threshold
    /// @param sigma Base sigma for Gaussian blur
    /// @param descriptorType CV_32F or CV_8U
    /// @throws std::runtime_error if Metal initialization fails
    MetalSIFT(int nOctaveLayers = 3,
              double contrastThreshold = 0.04,
              double edgeThreshold = 10.0,
              double sigma = 1.6,
              int descriptorType = CV_32F);

    /// Destructor - automatically releases all Metal resources (RAII)
    ~MetalSIFT();

    // Delete copy operations (Metal resources shouldn't be copied)
    MetalSIFT(const MetalSIFT&) = delete;
    MetalSIFT& operator=(const MetalSIFT&) = delete;

    // Allow move operations for flexibility
    MetalSIFT(MetalSIFT&& other) noexcept;
    MetalSIFT& operator=(MetalSIFT&& other) noexcept;

    /// Process image through full Metal SIFT pipeline
    /// Performs Gaussian pyramid, DoG pyramid, extrema detection, and descriptor computation
    /// @param base Input image (already preprocessed to correct scale)
    /// @param gauss_pyr Output Gaussian pyramid (will be resized and populated)
    /// @param keypoints Output detected keypoints with orientations
    /// @param descriptors Output SIFT descriptors
    /// @param nOctaves Number of octaves to process
    /// @return true if processing succeeded, false if Metal processing failed
    bool detectAndCompute(const cv::Mat& base,
                         std::vector<cv::Mat>& gauss_pyr,
                         std::vector<cv::KeyPoint>& keypoints,
                         cv::Mat& descriptors,
                         int nOctaves);

    /// Check if Metal processing is available and initialized
    bool isAvailable() const;

    // Implementation details (public for helper functions, but still hidden via pImpl idiom)
    struct Impl;
    Impl* impl_;

private:
    // SIFT configuration parameters
    int nOctaveLayers_;
    double contrastThreshold_;
    double edgeThreshold_;
    double sigma_;
    int descriptorType_;
};

} // namespace lar

#endif // LAR_TRACKING_METAL_SIFT_H