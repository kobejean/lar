// Shared utilities for Metal-accelerated SIFT implementations
// This header contains types and functions shared between sift_metal.mm and sift_metal_fused.mm
#ifndef LAR_TRACKING_SIFT_METAL_COMMON_H
#define LAR_TRACKING_SIFT_METAL_COMMON_H

#import <Metal/Metal.h>
#include <opencv2/core.hpp>
#include <vector>

// Helper macro for conditional release
#if !__has_feature(objc_arc)
    #define RELEASE_IF_MANUAL(obj) [obj release]
#else
    #define RELEASE_IF_MANUAL(obj) (void)0
#endif

#define METAL_BUFFER_ALIGNMENT 16  // Metal prefers 16-byte alignment

namespace lar {

// ============================================================================
// Metal Resource Management
// ============================================================================

// Metal resource cache for reusing GPU buffers/textures across frames
struct MetalSiftResources {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> commandQueue = nil;

    // Cache buffers/textures by [octave][level]
    std::vector<std::vector<id<MTLBuffer>>> octaveBuffers;
    std::vector<std::vector<id<MTLTexture>>> octaveTextures;

    // Temporary textures for separable convolution (one per octave)
    std::vector<id<MTLBuffer>> tempBuffers;
    std::vector<id<MTLTexture>> tempTextures;

    // DoG pyramid buffers/textures by [octave][level]
    std::vector<std::vector<id<MTLBuffer>>> dogBuffers;
    std::vector<std::vector<id<MTLTexture>>> dogTextures;

    // Cached dimensions
    int cachedBaseWidth = 0;
    int cachedBaseHeight = 0;
    int cachedNOctaves = 0;
    int cachedNLevels = 0;

    bool needsReallocation(int baseWidth, int baseHeight, int nOctaves, int nLevels) {
        return cachedBaseWidth != baseWidth ||
               cachedBaseHeight != baseHeight ||
               cachedNOctaves != nOctaves ||
               cachedNLevels != nLevels;
    }

    void releaseBuffersAndTextures() {
        for (auto& octave : octaveTextures) {
            for (auto& tex : octave) {
                RELEASE_IF_MANUAL(tex);
            }
        }
        for (auto& octave : octaveBuffers) {
            for (auto& buf : octave) {
                RELEASE_IF_MANUAL(buf);
            }
        }
        for (auto& octave : dogTextures) {
            for (auto& tex : octave) {
                RELEASE_IF_MANUAL(tex);
            }
        }
        for (auto& octave : dogBuffers) {
            for (auto& buf : octave) {
                RELEASE_IF_MANUAL(buf);
            }
        }
        for (auto& tex : tempTextures) {
            RELEASE_IF_MANUAL(tex);
        }
        for (auto& buf : tempBuffers) {
            RELEASE_IF_MANUAL(buf);
        }
        octaveTextures.clear();
        octaveBuffers.clear();
        dogTextures.clear();
        dogBuffers.clear();
        tempTextures.clear();
        tempBuffers.clear();
    }

    ~MetalSiftResources() {
        releaseBuffersAndTextures();
        RELEASE_IF_MANUAL(commandQueue);
        RELEASE_IF_MANUAL(device);
#if __has_feature(objc_arc)
        commandQueue = nil;
        device = nil;
#endif
    }
};

// Thread-safe singleton for Metal resources
MetalSiftResources& getMetalResources();

// ============================================================================
// Gaussian Kernel Utilities
// ============================================================================

// Create 1D Gaussian kernel using OpenCV's bit-exact implementation
std::vector<float> createGaussianKernel(double sigma);

// Gaussian blur kernel modes
enum class GaussianKernelMode {
    MPS = 0,           // Apple's Metal Performance Shaders (most accurate)
    CustomSeparable = 1,  // Separate horizontal + vertical passes (OpenCV-pattern)
    CustomFused = 2       // Fused pass using threadgroup memory (faster, experimental)
};

// ============================================================================
// Structures matching Metal shader layouts
// ============================================================================

// Gaussian blur parameters (for custom Metal kernels)
struct GaussianBlurParams {
    int width;
    int height;
    int rowStride;
    int kernelSize;
};

// Candidate keypoint structure
struct KeypointCandidate {
    int x;
    int y;
    int octave;
    int layer;
    float value;
};

// Extrema detection parameters
struct ExtremaParams {
    int width;
    int height;
    int rowStride;
    float threshold;
    int border;
    int octave;
    int layer;
};

// Fused extrema detection parameters
struct FusedExtremaParams {
    int width;
    int height;
    int rowStride;
    float threshold;
    int border;
    int octave;
    int layer;
    int kernelSize;
};

} // namespace lar

#endif // LAR_TRACKING_SIFT_METAL_COMMON_H
