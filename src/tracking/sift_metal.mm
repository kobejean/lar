// Metal-accelerated Gaussian pyramid for SIFT
// Usage: Build with -DLAR_USE_METAL_SIFT=ON
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "lar/tracking/sift.h"
#include <iostream>
#include <chrono>
#include <vector>

#define LAR_PROFILE_METAL_SIFT 1
#define METAL_BUFFER_ALIGNMENT 16  // Metal prefers 16-byte alignment

// Helper macro for conditional release
#if !__has_feature(objc_arc)
    #define RELEASE_IF_MANUAL(obj) [obj release]
#else
    #define RELEASE_IF_MANUAL(obj) (void)0
#endif

namespace lar {

// Metal resource cache for reusing across frames
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
static MetalSiftResources& getMetalResources() {
    static MetalSiftResources resources;
    return resources;
}

// Create 1D Gaussian kernel using OpenCV's bit-exact implementation
static std::vector<float> createGaussianKernel(double sigma) {
    // OpenCV formula for CV_32F: cvRound(sigma*4*2+1)|1 ensures odd size
    int ksize = cvRound(sigma * 8 + 1) | 1;
    cv::Mat kernelMat = cv::getGaussianKernel(ksize, sigma, CV_32F);

    // Convert cv::Mat to std::vector<float>
    std::vector<float> kernel(ksize);
    for (int i = 0; i < ksize; i++) {
        kernel[i] = kernelMat.at<float>(i);
    }

    return kernel;
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
            resources.dogBuffers.resize(nOctaves);
            resources.dogTextures.resize(nOctaves);
            resources.tempBuffers.resize(nOctaves);
            resources.tempTextures.resize(nOctaves);

            for (int o = 0; o < nOctaves; o++) {
                int octaveWidth = base.cols >> o;
                int octaveHeight = base.rows >> o;

                size_t rowBytes = octaveWidth * sizeof(float);
                size_t alignedRowBytes = ((rowBytes + METAL_BUFFER_ALIGNMENT - 1) / METAL_BUFFER_ALIGNMENT) * METAL_BUFFER_ALIGNMENT;
                size_t bufferSize = alignedRowBytes * octaveHeight;

                resources.octaveBuffers[o].resize(nLevels);
                resources.octaveTextures[o].resize(nLevels);
                resources.dogBuffers[o].resize(nLevels - 1);  // DoG has one fewer level
                resources.dogTextures[o].resize(nLevels - 1);

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

                // Allocate DoG buffers/textures (nLevels - 1 differences)
                for (int i = 0; i < nLevels - 1; i++) {
                    resources.dogBuffers[o][i] = [device newBufferWithLength:bufferSize
                                                                      options:MTLResourceStorageModeShared];

                    MTLTextureDescriptor* dogDesc = [MTLTextureDescriptor
                        texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float
                        width:octaveWidth height:octaveHeight mipmapped:NO];
                    dogDesc.storageMode = MTLStorageModeShared;
                    dogDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;

                    resources.dogTextures[o][i] = [resources.dogBuffers[o][i]
                                                    newTextureWithDescriptor:dogDesc
                                                    offset:0
                                                    bytesPerRow:alignedRowBytes];
                }

                // Allocate temporary texture for separable convolution (sized for this octave)
                resources.tempBuffers[o] = [device newBufferWithLength:bufferSize
                                                            options:MTLResourceStorageModeShared];
                MTLTextureDescriptor* tempDesc = [MTLTextureDescriptor
                    texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float
                    width:octaveWidth height:octaveHeight mipmapped:NO];
                tempDesc.storageMode = MTLStorageModeShared;
                tempDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
                resources.tempTextures[o] = [resources.tempBuffers[o] newTextureWithDescriptor:tempDesc
                                                                                         offset:0
                                                                                    bytesPerRow:alignedRowBytes];
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

            size_t rowBytes = octaveWidth * sizeof(float);
            size_t alignedRowBytes = ((rowBytes + METAL_BUFFER_ALIGNMENT - 1) / METAL_BUFFER_ALIGNMENT) * METAL_BUFFER_ALIGNMENT;

            std::vector<id<MTLBuffer>>& levelBuffers = resources.octaveBuffers[o];
            std::vector<id<MTLTexture>>& levelTextures = resources.octaveTextures[o];
            id<MTLTexture> tempTexture = resources.tempTextures[o];

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
                // Create exact Gaussian kernel matching OpenCV
                std::vector<float> kernel = createGaussianKernel(sigmas[i]);
                int kernelSize = (int)kernel.size();

                // Horizontal convolution (1 x kernelSize kernel)
                MPSImageConvolution* horizConv = [[MPSImageConvolution alloc]
                    initWithDevice:device
                       kernelWidth:kernelSize
                      kernelHeight:1
                           weights:kernel.data()];
                horizConv.edgeMode = MPSImageEdgeModeClamp;  // Replicate border

                [horizConv encodeToCommandBuffer:commandBuffer
                                   sourceTexture:levelTextures[i-1]
                              destinationTexture:tempTexture];

                // Vertical convolution (kernelSize x 1 kernel)
                MPSImageConvolution* vertConv = [[MPSImageConvolution alloc]
                    initWithDevice:device
                       kernelWidth:1
                      kernelHeight:kernelSize
                           weights:kernel.data()];
                vertConv.edgeMode = MPSImageEdgeModeClamp;  // Replicate border

                [vertConv encodeToCommandBuffer:commandBuffer
                                  sourceTexture:tempTexture
                             destinationTexture:levelTextures[i]];

                RELEASE_IF_MANUAL(horizConv);
                RELEASE_IF_MANUAL(vertConv);
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
                // pyr[o * nLevels + i] = pyr[o * nLevels + i].clone();
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

// Metal-accelerated DoG pyramid computation
void buildDoGPyramidMetal(const std::vector<cv::Mat>& gauss_pyr, std::vector<cv::Mat>& dog_pyr,
                          int nOctaves, int nLevels) {
    @autoreleasepool {
        MetalSiftResources& resources = getMetalResources();

        if (!resources.device) {
            std::cerr << "Metal not initialized, call buildGaussianPyramidMetal first" << std::endl;
            return;
        }

        id<MTLDevice> device = resources.device;
        id<MTLCommandQueue> commandQueue = resources.commandQueue;

#ifdef LAR_PROFILE_METAL_SIFT
        auto startTotal = std::chrono::high_resolution_clock::now();
        double gpuTime = 0, downloadTime = 0;
#endif

        // Batch all DoG subtractions in single command buffer
#ifdef LAR_PROFILE_METAL_SIFT
        auto gpuStart = std::chrono::high_resolution_clock::now();
#endif
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];

        for (int o = 0; o < nOctaves; o++) {
            std::vector<id<MTLTexture>>& gaussTextures = resources.octaveTextures[o];
            std::vector<id<MTLTexture>>& dogTextures = resources.dogTextures[o];

            for (int i = 0; i < nLevels - 1; i++) {
                // DoG[i] = Gauss[i+1] - Gauss[i]
                MPSImageSubtract* subtract = [[MPSImageSubtract alloc] initWithDevice:device];

                [subtract encodeToCommandBuffer:commandBuffer
                                 primaryTexture:gaussTextures[i+1]  // Gauss[i+1]
                               secondaryTexture:gaussTextures[i]    // Gauss[i]
                             destinationTexture:dogTextures[i]];    // Result

                RELEASE_IF_MANUAL(subtract);
            }
        }

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

#ifdef LAR_PROFILE_METAL_SIFT
        gpuTime = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - gpuStart).count();
#endif

        // Wrap DoG buffers in cv::Mat (zero-copy!)
#ifdef LAR_PROFILE_METAL_SIFT
        auto downloadStart = std::chrono::high_resolution_clock::now();
#endif
        for (int o = 0; o < nOctaves; o++) {
            int octaveWidth = gauss_pyr[o * nLevels].cols;
            int octaveHeight = gauss_pyr[o * nLevels].rows;

            size_t rowBytes = octaveWidth * sizeof(float);
            size_t alignedRowBytes = ((rowBytes + METAL_BUFFER_ALIGNMENT - 1) / METAL_BUFFER_ALIGNMENT) * METAL_BUFFER_ALIGNMENT;

            std::vector<id<MTLBuffer>>& dogBuffers = resources.dogBuffers[o];

            for (int i = 0; i < nLevels - 1; i++) {
                // Wrap shared buffer with OpenCV Mat (no copy!)
                float* bufferPtr = (float*)dogBuffers[i].contents;
                dog_pyr[o * (nLevels - 1) + i] = cv::Mat(octaveHeight, octaveWidth, CV_32F,
                                                          bufferPtr, alignedRowBytes);

                // Clone to independent memory (Metal buffers stay cached for reuse)
                // Comment out for zero-copy if DoG pyramid consumed immediately
                // dog_pyr[o * (nLevels - 1) + i] = dog_pyr[o * (nLevels - 1) + i].clone();
            }
        }
#ifdef LAR_PROFILE_METAL_SIFT
        downloadTime = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - downloadStart).count();
#endif

#ifdef LAR_PROFILE_METAL_SIFT
        auto endTotal = std::chrono::high_resolution_clock::now();
        double totalTime = std::chrono::duration<double, std::milli>(
            endTotal - startTotal).count();
        std::cout << "Metal DoG Pyramid Profile:\n";
        std::cout << "  GPU:      " << gpuTime << " ms\n"
                  << "  Download: " << downloadTime << " ms\n"
                  << "  Total:    " << totalTime << " ms\n";
#endif
    }
}

// Structure matching Metal shader KeypointCandidate
struct KeypointCandidate {
    int x;
    int y;
    int octave;
    int layer;
    float value;
};

// Structure matching Metal shader ExtremaParams
struct ExtremaParams {
    int width;
    int height;
    int rowStride;
    float threshold;
    int border;
    int octave;
    int layer;
};

// Metal-accelerated scale-space extrema detection (hybrid GPU/CPU)
// GPU: Parallel 3D extrema detection → candidate list
// CPU: Per-candidate refinement + orientation (existing functions)
void findScaleSpaceExtremaMetal(
    const std::vector<cv::Mat>& gauss_pyr,
    const std::vector<cv::Mat>& dog_pyr,
    std::vector<cv::KeyPoint>& keypoints,
    int nOctaves,
    int nOctaveLayers,
    float threshold,
    double contrastThreshold,
    double edgeThreshold,
    double sigma)
{
    @autoreleasepool {
        MetalSiftResources& resources = getMetalResources();

        if (!resources.device) {
            std::cerr << "Metal not initialized" << std::endl;
            return;
        }

        id<MTLDevice> device = resources.device;
        id<MTLCommandQueue> commandQueue = resources.commandQueue;

#ifdef LAR_PROFILE_METAL_SIFT
        auto startTotal = std::chrono::high_resolution_clock::now();
        double gpuTime = 0, cpuTime = 0;
#endif

        // Load compiled Metal shader library
        NSError* error = nil;

        // Try to load from runtime bin directory first (where executables are)
        NSString* binPath = @"bin/sift.metallib";
        NSBundle* mainBundle = [NSBundle mainBundle];
        NSString* bundlePath = [[mainBundle resourcePath] stringByAppendingPathComponent:binPath];
        NSURL* libraryURL = [NSURL fileURLWithPath:bundlePath];

        // Fallback: try current working directory
        if (![[NSFileManager defaultManager] fileExistsAtPath:bundlePath]) {
            libraryURL = [NSURL fileURLWithPath:@"bin/sift.metallib"];
        }

        // Fallback: try relative to executable
        if (![[NSFileManager defaultManager] fileExistsAtPath:[libraryURL path]]) {
            NSString* execPath = [[NSBundle mainBundle] executablePath];
            NSString* execDir = [execPath stringByDeletingLastPathComponent];
            NSString* metalLibPath = [execDir stringByAppendingPathComponent:@"sift.metallib"];
            libraryURL = [NSURL fileURLWithPath:metalLibPath];
        }

        id<MTLLibrary> library = [device newLibraryWithURL:libraryURL error:&error];
        if (!library) {
            std::cerr << "Failed to load Metal library: " << [[error localizedDescription] UTF8String] << std::endl;
            std::cerr << "Searched paths: " << [bundlePath UTF8String] << std::endl;
            return;
        }

        id<MTLFunction> extremaFunction = [library newFunctionWithName:@"detectScaleSpaceExtrema"];
        if (!extremaFunction) {
            std::cerr << "Failed to find Metal function: detectScaleSpaceExtrema" << std::endl;
            RELEASE_IF_MANUAL(library);
            return;
        }

        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:extremaFunction
                                                                                      error:&error];
        if (!pipeline) {
            std::cerr << "Failed to create compute pipeline: " << [[error localizedDescription] UTF8String] << std::endl;
            RELEASE_IF_MANUAL(extremaFunction);
            RELEASE_IF_MANUAL(library);
            return;
        }

        // Maximum candidates buffer (generous upper bound: ~10% of pixels per layer)
        const uint32_t MAX_CANDIDATES = 50000;

        // Allocate candidate buffer (shared for CPU access)
        id<MTLBuffer> candidateBuffer = [device newBufferWithLength:MAX_CANDIDATES * sizeof(KeypointCandidate)
                                                             options:MTLResourceStorageModeShared];

        // Allocate atomic counter buffer
        id<MTLBuffer> counterBuffer = [device newBufferWithLength:sizeof(uint32_t)
                                                           options:MTLResourceStorageModeShared];

        keypoints.clear();

#ifdef LAR_PROFILE_METAL_SIFT
        auto gpuStart = std::chrono::high_resolution_clock::now();
#endif

        // Process each octave and layer
        for (int o = 0; o < nOctaves; o++) {
            int octaveWidth = dog_pyr[o * (nOctaveLayers + 2)].cols;
            int octaveHeight = dog_pyr[o * (nOctaveLayers + 2)].rows;

            size_t rowBytes = octaveWidth * sizeof(float);
            size_t alignedRowBytes = ((rowBytes + METAL_BUFFER_ALIGNMENT - 1) / METAL_BUFFER_ALIGNMENT) * METAL_BUFFER_ALIGNMENT;
            int rowStride = (int)(alignedRowBytes / sizeof(float));

            std::vector<id<MTLBuffer>>& dogBuffers = resources.dogBuffers[o];

            // Process layers 1 to nOctaveLayers (middle layers only)
            for (int i = 1; i <= nOctaveLayers; i++) {
                // Reset counter to zero
                uint32_t* counterPtr = (uint32_t*)counterBuffer.contents;
                *counterPtr = 0;

                // Setup parameters
                ExtremaParams params;
                params.width = octaveWidth;
                params.height = octaveHeight;
                params.rowStride = rowStride;
                params.threshold = threshold;
                params.border = 5; // SIFT_IMG_BORDER
                params.octave = o;
                params.layer = i;

                // Create parameter buffer
                id<MTLBuffer> paramsBuffer = [device newBufferWithBytes:&params
                                                                  length:sizeof(ExtremaParams)
                                                                 options:MTLResourceStorageModeShared];

                // Create max candidates buffer
                id<MTLBuffer> maxCandidatesBuffer = [device newBufferWithBytes:&MAX_CANDIDATES
                                                                         length:sizeof(uint32_t)
                                                                        options:MTLResourceStorageModeShared];

                // Create command buffer
                id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:dogBuffers[i-1] offset:0 atIndex:0]; // prevLayer
                [encoder setBuffer:dogBuffers[i]   offset:0 atIndex:1]; // currLayer
                [encoder setBuffer:dogBuffers[i+1] offset:0 atIndex:2]; // nextLayer
                [encoder setBuffer:counterBuffer offset:0 atIndex:3];
                [encoder setBuffer:candidateBuffer offset:0 atIndex:4];
                [encoder setBuffer:paramsBuffer offset:0 atIndex:5];
                [encoder setBuffer:maxCandidatesBuffer offset:0 atIndex:6];

                // Dispatch threads (one per pixel, excluding border)
                MTLSize gridSize = MTLSizeMake(octaveWidth, octaveHeight, 1);
                NSUInteger threadExecutionWidth = [pipeline threadExecutionWidth];
                MTLSize threadgroupSize = MTLSizeMake(threadExecutionWidth, 1, 1);

                [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
                [encoder endEncoding];

                [commandBuffer commit];
                [commandBuffer waitUntilCompleted];

                RELEASE_IF_MANUAL(paramsBuffer);
                RELEASE_IF_MANUAL(maxCandidatesBuffer);

                // Read back candidate count
                uint32_t candidateCount = *counterPtr;
                candidateCount = std::min(candidateCount, MAX_CANDIDATES);

                if (candidateCount > 0) {
                    // Process candidates on CPU (refinement + orientation)
#ifdef LAR_PROFILE_METAL_SIFT
                    auto cpuStart = std::chrono::high_resolution_clock::now();
#endif
                    KeypointCandidate* candidates = (KeypointCandidate*)candidateBuffer.contents;

                    for (uint32_t c = 0; c < candidateCount; c++) {
                        KeypointCandidate& cand = candidates[c];

                        // Skip if outside valid bounds (should not happen, but be safe)
                        if (cand.x < 5 || cand.x >= octaveWidth - 5 ||
                            cand.y < 5 || cand.y >= octaveHeight - 5) {
                            continue;
                        }

                        // Create keypoint for refinement
                        cv::KeyPoint kpt;
                        int layer = cand.layer;
                        int r = cand.y;
                        int c_pos = cand.x;

                        // Call adjustLocalExtrema (from sift.cpp) for subpixel refinement
                        if (!adjustLocalExtrema(dog_pyr, kpt, cand.octave, layer, r, c_pos,
                                               nOctaveLayers, (float)contrastThreshold,
                                               (float)edgeThreshold, (float)sigma)) {
                            continue;
                        }

                        // Calculate orientation histogram (from sift.cpp)
                        static const int n = 36; // SIFT_ORI_HIST_BINS
                        float hist[n];
                        float scl_octv = kpt.size * 0.5f / (1 << cand.octave);

                        int gaussIdx = cand.octave * (nOctaveLayers + 3) + layer;
                        float omax = calcOrientationHist(gauss_pyr[gaussIdx],
                                                        cv::Point(c_pos, r),
                                                        cvRound(4.5f * scl_octv), // SIFT_ORI_RADIUS
                                                        1.5f * scl_octv,           // SIFT_ORI_SIG_FCTR
                                                        hist, n);

                        float mag_thr = omax * 0.8f; // SIFT_ORI_PEAK_RATIO

                        // Find orientation peaks and create keypoints
                        for (int j = 0; j < n; j++) {
                            int l = j > 0 ? j - 1 : n - 1;
                            int r2 = j < n-1 ? j + 1 : 0;

                            if (hist[j] > hist[l] && hist[j] > hist[r2] && hist[j] >= mag_thr) {
                                float bin = j + 0.5f * (hist[l]-hist[r2]) / (hist[l] - 2*hist[j] + hist[r2]);
                                bin = bin < 0 ? n + bin : bin >= n ? bin - n : bin;
                                kpt.angle = 360.f - (360.f/n) * bin;
                                if (std::abs(kpt.angle - 360.f) < FLT_EPSILON)
                                    kpt.angle = 0.f;

                                keypoints.push_back(kpt);
                            }
                        }
                    }

#ifdef LAR_PROFILE_METAL_SIFT
                    cpuTime += std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - cpuStart).count();
#endif
                }
            }
        }

#ifdef LAR_PROFILE_METAL_SIFT
        gpuTime = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - gpuStart).count();
#endif

        RELEASE_IF_MANUAL(candidateBuffer);
        RELEASE_IF_MANUAL(counterBuffer);
        RELEASE_IF_MANUAL(pipeline);
        RELEASE_IF_MANUAL(extremaFunction);
        RELEASE_IF_MANUAL(library);

#ifdef LAR_PROFILE_METAL_SIFT
        auto endTotal = std::chrono::high_resolution_clock::now();
        double totalTime = std::chrono::duration<double, std::milli>(
            endTotal - startTotal).count();
        std::cout << "Metal Extrema Detection Profile:\n";
        std::cout << "  GPU:      " << gpuTime << " ms\n"
                  << "  CPU:      " << cpuTime << " ms\n"
                  << "  Total:    " << totalTime << " ms\n";
#endif
    }
}

} // namespace lar