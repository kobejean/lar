// Metal-accelerated Gaussian pyramid for SIFT
// Usage: Build with -DLAR_USE_METAL_SIFT=ON
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <chrono>
#include <vector>
#define LAR_PROFILE_METAL_SIFT 1
namespace lar {

// Metal resource cache for reusing across frames
struct MetalSiftResources {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> commandQueue = nil;

    // Cache buffers/textures by [octave][level]
    std::vector<std::vector<id<MTLBuffer>>> octaveBuffers;
    std::vector<std::vector<id<MTLTexture>>> octaveTextures;

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
        // Release textures
        for (auto& octave : octaveTextures) {
            for (auto& tex : octave) {
                if (tex) [tex release];
            }
        }
        octaveTextures.clear();

        // Release buffers
        for (auto& octave : octaveBuffers) {
            for (auto& buf : octave) {
                if (buf) [buf release];
            }
        }
        octaveBuffers.clear();
    }

    ~MetalSiftResources() {
        releaseBuffersAndTextures();
        if (commandQueue) [commandQueue release];
        if (device) [device release];
    }
};

// Thread-safe singleton for Metal resources
static MetalSiftResources& getMetalResources() {
    static MetalSiftResources resources;
    return resources;
}

// Single-threaded Metal implementation with resource reuse
void buildGaussianPyramidMetal(const cv::Mat& base, std::vector<cv::Mat>& pyr,
                                int nOctaves, const std::vector<double>& sigmas) {
    @autoreleasepool {
        MetalSiftResources& resources = getMetalResources();
        int nLevels = (int)sigmas.size();

        // Initialize device and command queue once
        if (!resources.device) {
            resources.device = MTLCreateSystemDefaultDevice();
            if (!resources.device) {
                std::cerr << "Metal not available, falling back to CPU" << std::endl;
                return;
            }
            resources.commandQueue = [resources.device newCommandQueue];
        }

        id<MTLDevice> device = resources.device;
        id<MTLCommandQueue> commandQueue = resources.commandQueue;

#ifdef LAR_PROFILE_METAL_SIFT
        auto startTotal = std::chrono::high_resolution_clock::now();
        double uploadTime = 0, gpuTime = 0, downloadTime = 0;
        double allocationTime = 0;
#endif

        // Reallocate buffers only if dimensions changed
        if (resources.needsReallocation(base.cols, base.rows, nOctaves, nLevels)) {
#ifdef LAR_PROFILE_METAL_SIFT
            auto allocStart = std::chrono::high_resolution_clock::now();
#endif
            resources.releaseBuffersAndTextures();
            resources.octaveBuffers.resize(nOctaves);
            resources.octaveTextures.resize(nOctaves);

            for (int o = 0; o < nOctaves; o++) {
                int octaveWidth = base.cols >> o;
                int octaveHeight = base.rows >> o;

                size_t rowBytes = octaveWidth * sizeof(float);
                size_t alignedRowBytes = ((rowBytes + 15) / 16) * 16;
                size_t bufferSize = alignedRowBytes * octaveHeight;

                resources.octaveBuffers[o].resize(nLevels);
                resources.octaveTextures[o].resize(nLevels);

                for (int i = 0; i < nLevels; i++) {
                    // Create shared buffer (CPU & GPU accessible)
                    resources.octaveBuffers[o][i] = [device newBufferWithLength:bufferSize
                                                          options:MTLResourceStorageModeShared];

                    // Create texture descriptor for shared storage
                    MTLTextureDescriptor* desc = [MTLTextureDescriptor
                        texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float
                        width:octaveWidth height:octaveHeight mipmapped:NO];
                    desc.storageMode = MTLStorageModeShared;
                    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;

                    // Create texture backed by the shared buffer
                    resources.octaveTextures[o][i] = [resources.octaveBuffers[o][i]
                                                       newTextureWithDescriptor:desc
                                                       offset:0
                                                       bytesPerRow:alignedRowBytes];
                }
            }

            resources.cachedBaseWidth = base.cols;
            resources.cachedBaseHeight = base.rows;
            resources.cachedNOctaves = nOctaves;
            resources.cachedNLevels = nLevels;

#ifdef LAR_PROFILE_METAL_SIFT
            allocationTime = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - allocStart).count();
#endif
        }

        // Process each octave using cached buffers
        for (int o = 0; o < nOctaves; o++) {
            int octaveWidth = base.cols >> o;
            int octaveHeight = base.rows >> o;

            // Determine source for this octave
            cv::Mat octaveBase;
            if (o == 0) {
                octaveBase = base;
            } else {
                // Downsample from previous octave
                const cv::Mat& prevOctaveLayer = pyr[(o-1)*nLevels + (nLevels-3)];
                cv::resize(prevOctaveLayer, octaveBase,
                          cv::Size(octaveWidth, octaveHeight), 0, 0, cv::INTER_NEAREST);
            }

            // Reuse cached buffers and textures
            size_t rowBytes = octaveWidth * sizeof(float);
            size_t alignedRowBytes = ((rowBytes + 15) / 16) * 16;

            std::vector<id<MTLBuffer>>& levelBuffers = resources.octaveBuffers[o];
            std::vector<id<MTLTexture>>& levelTextures = resources.octaveTextures[o];

            // Copy base data to first level's shared buffer (CPU-side)
#ifdef LAR_PROFILE_METAL_SIFT
            auto uploadStart = std::chrono::high_resolution_clock::now();
#endif
            float* dstPtr = (float*)levelBuffers[0].contents;
            size_t alignedRowFloats = alignedRowBytes / sizeof(float);

            // Row-by-row copy accounting for aligned stride
            for (int row = 0; row < octaveHeight; row++) {
                memcpy(dstPtr + row * alignedRowFloats,
                      octaveBase.ptr<float>(row),
                      rowBytes);
            }
#ifdef LAR_PROFILE_METAL_SIFT
            uploadTime += std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - uploadStart).count();
#endif

            // Batch all blurs for this octave in single command buffer
#ifdef LAR_PROFILE_METAL_SIFT
            auto gpuStart = std::chrono::high_resolution_clock::now();
#endif
            id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];

            for (int i = 1; i < nLevels; i++) {
                // Apply Gaussian blur using MPS
                float sigma = sigmas[i];
                MPSImageGaussianBlur* blur = [[MPSImageGaussianBlur alloc]
                    initWithDevice:device sigma:sigma];

                [blur encodeToCommandBuffer:commandBuffer
                              sourceTexture:levelTextures[i-1]
                         destinationTexture:levelTextures[i]];
                [blur release];
            }

            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
#ifdef LAR_PROFILE_METAL_SIFT
            gpuTime += std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - gpuStart).count();
#endif

            // Create OpenCV Mats wrapping the shared buffers (zero-copy!)
#ifdef LAR_PROFILE_METAL_SIFT
            auto downloadStart = std::chrono::high_resolution_clock::now();
#endif
            for (int i = 0; i < nLevels; i++) {
                // Wrap shared buffer with OpenCV Mat (no copy!)
                float* bufferPtr = (float*)levelBuffers[i].contents;
                pyr[o * nLevels + i] = cv::Mat(octaveHeight, octaveWidth, CV_32F,
                                                bufferPtr, alignedRowBytes);

                // Clone to independent memory (Metal buffers stay cached for reuse)
                // This is the actual "download" - copying from shared buffer to owned memory
                pyr[o * nLevels + i] = pyr[o * nLevels + i].clone();
            }
#ifdef LAR_PROFILE_METAL_SIFT
            downloadTime += std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - downloadStart).count();
#endif
        }

        // Buffers, textures, device, and command queue remain cached for next frame

#ifdef LAR_PROFILE_METAL_SIFT
        auto endTotal = std::chrono::high_resolution_clock::now();
        double totalTime = std::chrono::duration<double, std::milli>(
            endTotal - startTotal).count();
        std::cout << "Metal Gaussian Pyramid Profile:\n";
        if (allocationTime > 0) {
            std::cout << "  Alloc:    " << allocationTime << " ms (first frame only)\n";
        }
        std::cout << "  Upload:   " << uploadTime << " ms\n"
                  << "  GPU:      " << gpuTime << " ms\n"
                  << "  Download: " << downloadTime << " ms\n"
                  << "  Total:    " << totalTime << " ms\n";
#endif
    }
}

} // namespace lar