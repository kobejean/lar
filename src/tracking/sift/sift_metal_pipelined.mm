// Metal-accelerated pipelined SIFT scale-space extrema detection
// Usage: Build with -DLAR_USE_METAL_SIFT_PIPELINED=ON
//
// This implementation pipelines octave processing for maximum GPU utilization:
// - Each octave runs in its own command buffer (parallel GPU execution)
// - CPU prepares octave bases while GPU processes previous octaves
// - Separate bitarray sections per octave to avoid write conflicts
// - Final keypoint extraction happens after all GPU work completes
//
#import <Metal/Metal.h>
#include <utility>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "lar/tracking/sift/sift.h"
#include "lar/tracking/sift/sift_constants.h"
#include "sift_metal_common.h"
#include <iostream>
#include <chrono>
#include <vector>

#define LAR_PROFILE_METAL_SIFT 1

namespace lar {

// ============================================================================
// Helper Functions (reused from fused version)
// ============================================================================

// Initialize Metal pipelines (cached across calls)
bool initializeMetalPipelines(
    id<MTLDevice> device,
    __strong id<MTLComputePipelineState>& blurAndDoGPipeline,
    __strong id<MTLComputePipelineState>& extremaPipeline,
    __strong id<MTLComputePipelineState>& fusedPipeline)
{
    static id<MTLLibrary> cachedLibrary = nil;
    static id<MTLComputePipelineState> cachedBlurAndDoGPipeline = nil;
    static id<MTLComputePipelineState> cachedExtremaPipeline = nil;

    if (!cachedLibrary) {
        // Load Metal library using shared function
        cachedLibrary = loadMetalLibrary(device, @"sift_fused");
        if (!cachedLibrary) {
            return false;
        }

        NSError* error = nil;

        // Create pipeline for fused blur
        id<MTLFunction> fusedBlurAndDoGFunc = [cachedLibrary newFunctionWithName:@"gaussianBlurAndDoGFused"];
        if (!fusedBlurAndDoGFunc) {
            std::cerr << "Failed to find Metal function: gaussianBlurFused" << std::endl;
            return false;
        }

        cachedBlurAndDoGPipeline = [device newComputePipelineStateWithFunction:fusedBlurAndDoGFunc error:&error];
        RELEASE_IF_MANUAL(fusedBlurAndDoGFunc);
        if (!cachedBlurAndDoGPipeline) {
            std::cerr << "Failed to create blur pipeline: " << [[error localizedDescription] UTF8String] << std::endl;
            return false;
        }

        id<MTLFunction> extremaFunction = [cachedLibrary newFunctionWithName:@"detectExtrema"];
        if (!extremaFunction) {
            std::cerr << "Failed to find Metal function: detectExtrema" << std::endl;
            return false;
        }

        cachedExtremaPipeline = [device newComputePipelineStateWithFunction:extremaFunction error:&error];
        RELEASE_IF_MANUAL(extremaFunction);
        if (!cachedExtremaPipeline) {
            std::cerr << "Failed to create compute pipeline: " << [[error localizedDescription] UTF8String] << std::endl;
            return false;
        }
    }

    blurAndDoGPipeline = cachedBlurAndDoGPipeline;
    extremaPipeline = cachedExtremaPipeline;
    return true;
}

// Compute Gaussian kernels for all pyramid levels
void computeGaussianKernels(
    int nLevels,
    int nOctaveLayers,
    double sigma,
    std::vector<std::vector<float>>& kernels,
    std::vector<double>& sigmas)
{
    kernels.resize(nLevels);
    sigmas.resize(nLevels);

    double k = std::pow(2.0, 1.0 / nOctaveLayers);
    sigmas[0] = sigma;

    for (int i = 1; i < nLevels; i++) {
        double sig_prev = std::pow(k, (double)(i-1)) * sigma;
        double sig_total = sig_prev * k;
        sigmas[i] = std::sqrt(sig_total*sig_total - sig_prev*sig_prev);
        kernels[i] = createGaussianKernel(sigmas[i]);
    }
}

// Allocate Metal buffers and textures for an octave
void allocateOctaveResources(
    id<MTLDevice> device,
    MetalSiftResources& resources,
    int octave,
    int octaveWidth,
    int octaveHeight,
    int nLevels)
{
    size_t rowBytes = octaveWidth * sizeof(float);
    size_t alignedRowBytes = ((rowBytes + METAL_BUFFER_ALIGNMENT - 1) / METAL_BUFFER_ALIGNMENT) * METAL_BUFFER_ALIGNMENT;
    size_t bufferSize = alignedRowBytes * octaveHeight;

    resources.octaveBuffers[octave].resize(nLevels);
    resources.octaveTextures[octave].resize(nLevels);
    resources.dogBuffers[octave].resize(nLevels - 1);
    resources.dogTextures[octave].resize(nLevels - 1);

    // Allocate Gaussian pyramid buffers/textures
    for (int i = 0; i < nLevels; i++) {
        resources.octaveBuffers[octave][i] = [device newBufferWithLength:bufferSize
                                                      options:MTLResourceStorageModeShared];

        MTLTextureDescriptor* desc = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float
            width:octaveWidth height:octaveHeight mipmapped:NO];
        desc.storageMode = MTLStorageModeShared;
        desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;

        resources.octaveTextures[octave][i] = [resources.octaveBuffers[octave][i]
                                               newTextureWithDescriptor:desc
                                               offset:0
                                               bytesPerRow:alignedRowBytes];
    }

    // Allocate DoG buffers/textures
    for (int i = 0; i < nLevels - 1; i++) {
        resources.dogBuffers[octave][i] = [device newBufferWithLength:bufferSize
                                                  options:MTLResourceStorageModeShared];

        MTLTextureDescriptor* dogDesc = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float
            width:octaveWidth height:octaveHeight mipmapped:NO];
        dogDesc.storageMode = MTLStorageModeShared;
        dogDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;

        resources.dogTextures[octave][i] = [resources.dogBuffers[octave][i]
                                            newTextureWithDescriptor:dogDesc
                                            offset:0
                                            bytesPerRow:alignedRowBytes];
    }

    // Allocate temporary texture for separable convolution
    resources.tempBuffers[octave] = [device newBufferWithLength:bufferSize
                                        options:MTLResourceStorageModeShared];
    MTLTextureDescriptor* tempDesc = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float
        width:octaveWidth height:octaveHeight mipmapped:NO];
    tempDesc.storageMode = MTLStorageModeShared;
    tempDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    resources.tempTextures[octave] = [resources.tempBuffers[octave] newTextureWithDescriptor:tempDesc
                                                                                     offset:0
                                                                                bytesPerRow:alignedRowBytes];
}

// Prepare octave base image on CPU (copy or downsample)
cv::Mat prepareOctaveBase(
    const cv::Mat& base,
    MetalSiftResources& resources,
    int octave,
    int nLevels,
    int octaveWidth,
    int octaveHeight)
{
    cv::Mat octaveBase;

    if (octave == 0) {
        // First octave: use base image directly
        octaveBase = base.clone();
    } else {
        // Downsample from previous octave's Gauss[nLevels-3]
        std::vector<id<MTLBuffer>>& prevOctaveBuffers = resources.octaveBuffers[octave-1];
        int prevOctaveWidth = base.cols >> (octave-1);
        int prevOctaveHeight = base.rows >> (octave-1);
        size_t prevRowBytes = prevOctaveWidth * sizeof(float);
        size_t prevAlignedRowBytes = ((prevRowBytes + METAL_BUFFER_ALIGNMENT - 1) / METAL_BUFFER_ALIGNMENT) * METAL_BUFFER_ALIGNMENT;

        float* prevBufPtr = (float*)prevOctaveBuffers[nLevels-3].contents;
        cv::Mat prevOctaveLayer(prevOctaveHeight, prevOctaveWidth, CV_32F, prevBufPtr, prevAlignedRowBytes);

        cv::resize(prevOctaveLayer, octaveBase,
                  cv::Size(octaveWidth, octaveHeight), 0, 0, cv::INTER_NEAREST);
    }

    return octaveBase;
}

// Upload octave base image to GPU
void uploadOctaveBase(
    const cv::Mat& octaveBase,
    id<MTLBuffer> gaussBuffer,
    int octaveWidth,
    int octaveHeight,
    size_t alignedRowBytes)
{
    float* gauss0Ptr = (float*)gaussBuffer.contents;
    size_t alignedRowFloats = alignedRowBytes / sizeof(float);

    for (int row = 0; row < octaveHeight; row++) {
        memcpy(gauss0Ptr + row * alignedRowFloats,
              octaveBase.ptr<float>(row),
              octaveWidth * sizeof(float));
    }
}

// Extract keypoints from bitarray for a specific octave/layer
void extractKeypoints(
    uint32_t* bitarray,
    int octaveBitarrayOffset,
    int octave,
    int nLevels,
    int octaveBitarraySize,
    int octaveWidth,
    int layer,
    int nOctaveLayers,
    float contrastThreshold,
    float edgeThreshold,
    float sigma,
    int gaussIdx,
    std::vector<cv::Mat>& gauss_pyr,
    std::vector<cv::Mat>& dog_pyr,
    std::vector<cv::KeyPoint>& keypoints)
{
    uint32_t* octaveBitarray = bitarray + octaveBitarrayOffset;
    int count = 0;

    for (uint32_t chunkIdx = 0; chunkIdx < octaveBitarraySize; chunkIdx++) {
        uint32_t chunk = octaveBitarray[chunkIdx];
        if (chunk == 0) continue;

        for (int bitOffset = 0; bitOffset < 32; bitOffset++) {
            if (chunk & (1u << bitOffset)) {
                uint32_t bitIndex = chunkIdx * 32 + bitOffset;
                int r = bitIndex / octaveWidth;
                int c_pos = bitIndex % octaveWidth;

                cv::KeyPoint kpt;
                int keypoint_layer = layer; // make copy so that adjustLocalExtrema mutates this one

                if (!adjustLocalExtrema(dog_pyr, kpt, octave, keypoint_layer, r, c_pos,
                                    nOctaveLayers, (float)contrastThreshold,
                                    (float)edgeThreshold, (float)sigma)) {
                    continue;
                }

                // Calculate orientation histogram
                static const int n = SIFT_ORI_HIST_BINS;
                float hist[n];
                float scl_octv = kpt.size * 0.5f / (1 << octave);

                int gaussIdx = octave * (nOctaveLayers + 3) + keypoint_layer;
                float omax = calcOrientationHist(gauss_pyr[gaussIdx],
                                                cv::Point(c_pos, r),
                                                cvRound(SIFT_ORI_RADIUS * scl_octv),
                                                SIFT_ORI_SIG_FCTR * scl_octv,
                                                hist, n);

                float mag_thr = omax * SIFT_ORI_PEAK_RATIO;

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
                        count++;
                    }
                }
            }
        }
    }

    std::cout << "extracted keypoints from octave " << octave << " layer " << layer << " adding " << count << " keypoints" << std::endl;
}

// ============================================================================
// Pipelined SIFT Implementation
// ============================================================================
// This version processes all octaves in parallel on the GPU by:
// 1. Preparing octave base images on CPU (can overlap with GPU work)
// 2. Submitting command buffers for each octave independently
// 3. Each octave gets its own bitarray section to avoid write conflicts
// 4. Extracting keypoints after all GPU work completes
//
void findScaleSpaceExtremaMetalPipelined(
    const cv::Mat& base,
    std::vector<cv::Mat>& gauss_pyr,
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
        int nLevels = nOctaveLayers + 3;

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

        double cpuPrepTime = 0;
        double cpuExtractTime = 0;
        double gpuBlurTime = 0;
        double gpuExtremaTime = 0;
#ifdef LAR_PROFILE_METAL_SIFT
        auto startTotal = std::chrono::high_resolution_clock::now();
#endif

        // Initialize Metal pipelines
        id<MTLComputePipelineState> blurAndDoGPipeline = nil;
        id<MTLComputePipelineState> extremaPipeline = nil;
        id<MTLComputePipelineState> fusedPipeline = nil;

        if (!initializeMetalPipelines(device, blurAndDoGPipeline, extremaPipeline, fusedPipeline)) {
            return;
        }

        // Reallocate buffers only if dimensions changed
        if (resources.needsReallocation(base.cols, base.rows, nOctaves, nLevels)) {
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
                allocateOctaveResources(device, resources, o, octaveWidth, octaveHeight, nLevels);
            }

            resources.cachedBaseWidth = base.cols;
            resources.cachedBaseHeight = base.rows;
            resources.cachedNOctaves = nOctaves;
            resources.cachedNLevels = nLevels;
        }

        // Pre-compute Gaussian kernels for all sigma values
        std::vector<std::vector<float>> gaussianKernels;
        std::vector<double> sigmas;
        computeGaussianKernels(nLevels, nOctaveLayers, sigma, gaussianKernels, sigmas);

        // Pre-allocate pyramid storage
        std::vector<cv::Mat> dog_pyr(nOctaves * (nOctaveLayers + 2));
        gauss_pyr.resize(nOctaves * nLevels);

        // Calculate total bitarray size (sum of all octaves)
        std::vector<uint32_t> octaveBitarrayOffsets(nOctaves);
        std::vector<uint32_t> octaveBitarraySizes(nOctaves);
        uint32_t totalBitarraySize = 0;

        for (int o = 0; o < nOctaves; o++) {
            int octaveWidth = base.cols >> o;
            int octaveHeight = base.rows >> o;
            uint32_t octavePixels = octaveWidth * octaveHeight;
            uint32_t octaveSize = (octavePixels + 31) / 32;  // Number of uint32 chunks

            octaveBitarrayOffsets[o] = totalBitarraySize;
            octaveBitarraySizes[o] = octaveSize;
            totalBitarraySize += octaveSize * nOctaveLayers;  // Each layer needs its own bitarray section
        }

        // Allocate unified bitarray buffer for all octaves
        id<MTLBuffer> bitarrayBuffer = [device newBufferWithLength:totalBitarraySize * sizeof(uint32_t)
                                                            options:MTLResourceStorageModeShared];
        memset(bitarrayBuffer.contents, 0, totalBitarraySize * sizeof(uint32_t));

        keypoints.clear();

        // ========================================================================
        // PIPELINE PHASE 1: Prepare octave bases and submit GPU work
        // ========================================================================
        std::vector<id<MTLCommandBuffer>> commandBuffers(nOctaves);
        std::vector<cv::Mat> octaveBases(nOctaves);

#ifdef LAR_PROFILE_METAL_SIFT
        auto cpuPrepStart = std::chrono::high_resolution_clock::now();
#endif

        for (int o = 0; o < nOctaves; o++) {
            int octaveWidth = base.cols >> o;
            int octaveHeight = base.rows >> o;

            size_t rowBytes = octaveWidth * sizeof(float);
            size_t alignedRowBytes = ((rowBytes + METAL_BUFFER_ALIGNMENT - 1) / METAL_BUFFER_ALIGNMENT) * METAL_BUFFER_ALIGNMENT;
            int rowStride = (int)(alignedRowBytes / sizeof(float));

            // Populate gauss_pyr and dog_pyr with cv::Mat wrappers
            int gaussIdx = o * nLevels;
            int dogIdx = o * (nLevels - 1);

            std::vector<id<MTLBuffer>>& gaussBuffers = resources.octaveBuffers[o];
            std::vector<id<MTLBuffer>>& dogBuffers = resources.dogBuffers[o];

            for (int i = 0; i < nLevels; i++) {
                gauss_pyr[gaussIdx + i] = cv::Mat(octaveHeight, octaveWidth, CV_32F, gaussBuffers[i].contents, alignedRowBytes);
            }

            for (int i = 0; i < nLevels - 1; i++) {
                dog_pyr[dogIdx + i] = cv::Mat(octaveHeight, octaveWidth, CV_32F, dogBuffers[i].contents, alignedRowBytes);
            }

            // Prepare octave base on CPU
            octaveBases[o] = prepareOctaveBase(base, resources, o, nLevels, octaveWidth, octaveHeight);

            // Upload to GPU
            uploadOctaveBase(octaveBases[o], gaussBuffers[0], octaveWidth, octaveHeight, alignedRowBytes);

            // Create command buffer for this octave
            commandBuffers[o] = [commandQueue commandBuffer];
            commandBuffers[o].label = [NSString stringWithFormat:@"Octave %d", o];

            // Encode blur + DoG operations for all levels
            for (int i = 1; i < nLevels; i++) {
                GaussianBlurParams params;
                params.width = octaveWidth;
                params.height = octaveHeight;
                params.rowStride = rowStride;
                params.kernelSize = (int)gaussianKernels[i].size();

                id<MTLBuffer> paramsBuffer = [device newBufferWithBytes:&params
                                                    length:sizeof(GaussianBlurParams)
                                                    options:MTLResourceStorageModeShared];

                id<MTLBuffer> kernelBuffer = [device newBufferWithBytes:gaussianKernels[i].data()
                                                    length:gaussianKernels[i].size() * sizeof(float)
                                                    options:MTLResourceStorageModeShared];

                id<MTLComputeCommandEncoder> encoder = [commandBuffers[o] computeCommandEncoder];

                [encoder setComputePipelineState:blurAndDoGPipeline];
                [encoder setBuffer:gaussBuffers[i-1] offset:0 atIndex:0];
                [encoder setBuffer:gaussBuffers[i] offset:0 atIndex:1];
                [encoder setBuffer:dogBuffers[i-1] offset:0 atIndex:2];
                [encoder setBuffer:paramsBuffer offset:0 atIndex:3];
                [encoder setBuffer:kernelBuffer offset:0 atIndex:4];

                MTLSize gridSize = MTLSizeMake(octaveWidth, octaveHeight, 1);
                MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);

                [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
                [encoder endEncoding];

                RELEASE_IF_MANUAL(paramsBuffer);
                RELEASE_IF_MANUAL(kernelBuffer);
            }

            // Encode extrema detection for all layers
            for (int layer = 1; layer <= nOctaveLayers; layer++) {
                ExtremaParams params;
                params.width = octaveWidth;
                params.height = octaveHeight;
                params.rowStride = rowStride;
                params.threshold = threshold;
                params.border = SIFT_IMG_BORDER;
                params.octave = o;
                params.layer = layer;

                id<MTLBuffer> paramsBuffer = [device newBufferWithBytes:&params
                                                                    length:sizeof(ExtremaParams)
                                                                    options:MTLResourceStorageModeShared];

                id<MTLComputeCommandEncoder> encoder = [commandBuffers[o] computeCommandEncoder];

                [encoder setComputePipelineState:extremaPipeline];
                [encoder setBuffer:dogBuffers[layer-1] offset:0 atIndex:0]; // prevLayer
                [encoder setBuffer:dogBuffers[layer] offset:0 atIndex:1];   // currLayer
                [encoder setBuffer:dogBuffers[layer+1] offset:0 atIndex:2]; // nextLayer

                // Calculate bitarray offset for this octave and layer
                uint32_t layerBitarrayOffset = octaveBitarrayOffsets[o] + (layer - 1) * octaveBitarraySizes[o];
                [encoder setBuffer:bitarrayBuffer offset:layerBitarrayOffset * sizeof(uint32_t) atIndex:3];
                [encoder setBuffer:paramsBuffer offset:0 atIndex:5];

                MTLSize gridSize = MTLSizeMake(octaveWidth, octaveHeight, 1);
                MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);

                [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
                [encoder endEncoding];

                RELEASE_IF_MANUAL(paramsBuffer);
            }

            // ✅ Submit this octave's command buffer immediately
            // GPU will start processing while we prepare the next octave
            [commandBuffers[o] commit];
        }

#ifdef LAR_PROFILE_METAL_SIFT
        cpuPrepTime = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - cpuPrepStart).count();
#endif

        // ========================================================================
        // PIPELINE PHASE 2: Wait for all GPU work to complete
        // ========================================================================
#ifdef LAR_PROFILE_METAL_SIFT
        auto gpuWaitStart = std::chrono::high_resolution_clock::now();
#endif

        for (int o = 0; o < nOctaves; o++) {
            [commandBuffers[o] waitUntilCompleted];
        }

#ifdef LAR_PROFILE_METAL_SIFT
        double gpuWaitTime = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - gpuWaitStart).count();
#endif

        // ========================================================================
        // PIPELINE PHASE 3: Extract keypoints from all octaves/layers
        // ========================================================================
#ifdef LAR_PROFILE_METAL_SIFT
        auto cpuExtractStart = std::chrono::high_resolution_clock::now();
#endif

        uint32_t* bitarray = (uint32_t*)bitarrayBuffer.contents;

        for (int o = 0; o < nOctaves; o++) {
            int octaveWidth = base.cols >> o;
            int gaussIdx = o * nLevels;

            for (int layer = 1; layer <= nOctaveLayers; layer++) {
                uint32_t layerBitarrayOffset = octaveBitarrayOffsets[o] + (layer - 1) * octaveBitarraySizes[o];

                extractKeypoints(
                    bitarray,
                    layerBitarrayOffset,
                    o,
                    nLevels,
                    octaveBitarraySizes[o],
                    octaveWidth,
                    layer,
                    nOctaveLayers,
                    contrastThreshold,
                    edgeThreshold,
                    sigma,
                    gaussIdx,
                    gauss_pyr,
                    dog_pyr,
                    keypoints
                );
            }
        }

#ifdef LAR_PROFILE_METAL_SIFT
        cpuExtractTime = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - cpuExtractStart).count();
#endif

        RELEASE_IF_MANUAL(bitarrayBuffer);

#ifdef LAR_PROFILE_METAL_SIFT
        auto endTotal = std::chrono::high_resolution_clock::now();
        double totalTime = std::chrono::duration<double, std::milli>(
            endTotal - startTotal).count();

        std::cout << "\n=== Metal Pipelined SIFT Profile ===\n";
        std::cout << "CPU Operations:\n";
        std::cout << "  Octave prep:         " << cpuPrepTime << " ms\n";
        std::cout << "  Keypoint extraction: " << cpuExtractTime << " ms\n";
        std::cout << "GPU Operations:\n";
        std::cout << "  GPU wait time:       " << gpuWaitTime << " ms (wall-clock)\n";
        std::cout << "Total:\n";
        std::cout << "  Wall-clock time:     " << totalTime << " ms\n";
        std::cout << "  Speedup potential:   " << (cpuPrepTime + gpuWaitTime) / gpuWaitTime << "x (overlap)\n";
        std::cout << "====================================\n\n";
#endif
    }
}

} // namespace lar