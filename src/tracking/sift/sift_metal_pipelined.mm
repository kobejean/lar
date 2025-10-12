// This implementation uses per-layer command buffers for fine-grained GPU scheduling:
// - Each layer within each octave gets its own command buffer
// - Octave 1+ layer 1 includes GPU resize from previous octave (eliminates CPU/GPU sync)
// - Dependencies handled by Metal's command queue ordering
// - Separate extrema bitarray buffers per layer for cache coherency
// - Final keypoint extraction happens after all GPU work completes

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
    __strong id<MTLComputePipelineState>& resizePipeline)
{
    static id<MTLLibrary> cachedLibrary = nil;
    static id<MTLComputePipelineState> cachedBlurAndDoGPipeline = nil;
    static id<MTLComputePipelineState> cachedExtremaPipeline = nil;
    static id<MTLComputePipelineState> cachedResizePipeline = nil;

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
            std::cerr << "Failed to find Metal function: gaussianBlurAndDoGFused" << std::endl;
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
            std::cerr << "Failed to create extrema pipeline: " << [[error localizedDescription] UTF8String] << std::endl;
            return false;
        }

        id<MTLFunction> resizeFunction = [cachedLibrary newFunctionWithName:@"resizeNearestNeighbor2x"];
        if (!resizeFunction) {
            std::cerr << "Failed to find Metal function: resizeNearestNeighbor2x" << std::endl;
            return false;
        }

        cachedResizePipeline = [device newComputePipelineStateWithFunction:resizeFunction error:&error];
        RELEASE_IF_MANUAL(resizeFunction);
        if (!cachedResizePipeline) {
            std::cerr << "Failed to create resize pipeline: " << [[error localizedDescription] UTF8String] << std::endl;
            return false;
        }
    }

    blurAndDoGPipeline = cachedBlurAndDoGPipeline;
    extremaPipeline = cachedExtremaPipeline;
    resizePipeline = cachedResizePipeline;
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
// Per-Layer Command Buffer Pipeline
// ============================================================================
// Strategy:
// - Layer 1: blur(0→1) + blur(1→2) + blur(2→3) + DoG(0,1,2) + extrema(layer1)
// - Layer 2: blur(3→4) + DoG(3) + extrema(layer2)
// - Layer 3: blur(4→5) + DoG(4) + extrema(layer3)
// Each layer's command buffer is submitted immediately after encoding
// Metal's command queue handles dependencies automatically via submission order
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
#ifdef LAR_PROFILE_METAL_SIFT
        auto startTotal = std::chrono::high_resolution_clock::now();
#endif

        // Initialize Metal pipelines
        id<MTLComputePipelineState> blurAndDoGPipeline = nil;
        id<MTLComputePipelineState> extremaPipeline = nil;
        id<MTLComputePipelineState> resizePipeline = nil;

        if (!initializeMetalPipelines(device, blurAndDoGPipeline, extremaPipeline, resizePipeline)) {
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

        // Allocate separate extrema bitarray buffers for each layer (better cache coherency)
        // Structure: layerExtremaBitarrays[octave * nOctaveLayers + (layer-1)]
        std::vector<id<MTLBuffer>> layerExtremaBitarrays;
        std::vector<uint32_t> layerExtremaBitarraySizes;
        layerExtremaBitarrays.reserve(nOctaves * nOctaveLayers);
        layerExtremaBitarraySizes.reserve(nOctaves * nOctaveLayers);

        for (int o = 0; o < nOctaves; o++) {
            int octaveWidth = base.cols >> o;
            int octaveHeight = base.rows >> o;
            uint32_t octavePixels = octaveWidth * octaveHeight;
            uint32_t extremaBitarraySize = ((octavePixels + 31) / 32) * sizeof(uint32_t);  // Size in bytes

            for (int layer = 1; layer <= nOctaveLayers; layer++) {
                id<MTLBuffer> buffer = [device newBufferWithLength:extremaBitarraySize
                                                           options:MTLResourceStorageModeShared];
                memset(buffer.contents, 0, extremaBitarraySize);
                layerExtremaBitarrays.push_back(buffer);
                layerExtremaBitarraySizes.push_back(extremaBitarraySize / sizeof(uint32_t));  // Store size in uint32 count
            }
        }

        keypoints.clear();

        // Track command buffers for all layers across all octaves
        std::vector<id<MTLCommandBuffer>> allCommandBuffers;
        allCommandBuffers.reserve(nOctaves * nOctaveLayers);

#ifdef LAR_PROFILE_METAL_SIFT
        auto cpuPrepStart = std::chrono::high_resolution_clock::now();
#endif

        // ========================================================================
        // PIPELINE: Submit per-layer command buffers for all octaves
        // ========================================================================

        // First, prepare octave 0's base on CPU (no dependency)
        {
            int octaveWidth = base.cols;
            int octaveHeight = base.rows;
            size_t rowBytes = octaveWidth * sizeof(float);
            size_t alignedRowBytes = ((rowBytes + METAL_BUFFER_ALIGNMENT - 1) / METAL_BUFFER_ALIGNMENT) * METAL_BUFFER_ALIGNMENT;

            std::vector<id<MTLBuffer>>& gaussBuffers = resources.octaveBuffers[0];
            cv::Mat octaveBase = base.clone();
            uploadOctaveBase(octaveBase, gaussBuffers[0], octaveWidth, octaveHeight, alignedRowBytes);
        }

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

            // Now submit per-layer command buffers
            // Layer 1: needs blur 0→1, 1→2, 2→3 + DoG 0,1,2 + extrema
            // Layer 2: needs blur 3→4 + DoG 3 + extrema
            // Layer 3: needs blur 4→5 + DoG 4 + extrema

            for (int layer = 1; layer <= nOctaveLayers; layer++) {
                id<MTLCommandBuffer> layerCmdBuf = [commandQueue commandBuffer];
                layerCmdBuf.label = [NSString stringWithFormat:@"Octave %d Layer %d", o, layer];

                // For octave 1+ layer 1: encode GPU resize at the start of the command buffer
                // This creates dependency: octave N-1 layer 3 → resize → octave N layer 1 blurs
                if (layer == 1 && o > 0) {
                    int prevOctaveWidth = base.cols >> (o - 1);
                    int prevOctaveHeight = base.rows >> (o - 1);
                    size_t prevRowBytes = prevOctaveWidth * sizeof(float);
                    size_t prevAlignedRowBytes = ((prevRowBytes + METAL_BUFFER_ALIGNMENT - 1) / METAL_BUFFER_ALIGNMENT) * METAL_BUFFER_ALIGNMENT;
                    int prevRowStride = (int)(prevAlignedRowBytes / sizeof(float));

                    std::vector<id<MTLBuffer>>& prevGaussBuffers = resources.octaveBuffers[o - 1];

                    ResizeParams resizeParams;
                    resizeParams.srcWidth = prevOctaveWidth;
                    resizeParams.srcHeight = prevOctaveHeight;
                    resizeParams.srcRowStride = prevRowStride;
                    resizeParams.dstWidth = octaveWidth;
                    resizeParams.dstHeight = octaveHeight;
                    resizeParams.dstRowStride = rowStride;

                    id<MTLBuffer> resizeParamsBuffer = [device newBufferWithBytes:&resizeParams
                                                        length:sizeof(ResizeParams)
                                                        options:MTLResourceStorageModeShared];

                    id<MTLComputeCommandEncoder> resizeEncoder = [layerCmdBuf computeCommandEncoder];
                    [resizeEncoder setComputePipelineState:resizePipeline];
                    [resizeEncoder setBuffer:prevGaussBuffers[nLevels-3] offset:0 atIndex:0];  // src
                    [resizeEncoder setBuffer:gaussBuffers[0] offset:0 atIndex:1];  // dst
                    [resizeEncoder setBuffer:resizeParamsBuffer offset:0 atIndex:2];

                    MTLSize resizeGridSize = MTLSizeMake(octaveWidth, octaveHeight, 1);
                    MTLSize resizeThreadgroupSize = MTLSizeMake(16, 16, 1);

                    [resizeEncoder dispatchThreads:resizeGridSize threadsPerThreadgroup:resizeThreadgroupSize];
                    [resizeEncoder endEncoding];

                    RELEASE_IF_MANUAL(resizeParamsBuffer);
                }

                // Determine which blurs this layer needs
                // Layer 1: blur indices 1,2,3 (to get Gauss[1], Gauss[2], Gauss[3])
                // Layer 2: blur index 4 (to get Gauss[4])
                // Layer 3: blur index 5 (to get Gauss[5])
                int blurStart = (layer == 1) ? 1 : (layer + 2);
                int blurEnd = (layer == 1) ? 4 : (layer + 3);

                // Encode blur + DoG operations
                for (int i = blurStart; i < blurEnd && i < nLevels; i++) {
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

                    id<MTLComputeCommandEncoder> encoder = [layerCmdBuf computeCommandEncoder];

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

                // Encode extrema detection for this layer
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

                id<MTLComputeCommandEncoder> encoder = [layerCmdBuf computeCommandEncoder];

                [encoder setComputePipelineState:extremaPipeline];
                [encoder setBuffer:dogBuffers[layer-1] offset:0 atIndex:0]; // prevLayer
                [encoder setBuffer:dogBuffers[layer] offset:0 atIndex:1];   // currLayer
                [encoder setBuffer:dogBuffers[layer+1] offset:0 atIndex:2]; // nextLayer

                // Use the individual buffer for this layer (no offset needed)
                int layerIndex = o * nOctaveLayers + (layer - 1);
                [encoder setBuffer:layerExtremaBitarrays[layerIndex] offset:0 atIndex:3];
                [encoder setBuffer:paramsBuffer offset:0 atIndex:5];

                MTLSize gridSize = MTLSizeMake(octaveWidth, octaveHeight, 1);
                MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);

                [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
                [encoder endEncoding];

                RELEASE_IF_MANUAL(paramsBuffer);

                // ✅ Submit this layer's command buffer immediately
                [layerCmdBuf commit];
                allCommandBuffers.push_back(layerCmdBuf);
            }
        }

#ifdef LAR_PROFILE_METAL_SIFT
        cpuPrepTime = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - cpuPrepStart).count();
#endif

        // ========================================================================
        // Wait for all GPU work to complete
        // ========================================================================
#ifdef LAR_PROFILE_METAL_SIFT
        auto gpuWaitStart = std::chrono::high_resolution_clock::now();
#endif

        for (auto& cmdBuf : allCommandBuffers) {
            [cmdBuf waitUntilCompleted];
        }

#ifdef LAR_PROFILE_METAL_SIFT
        double gpuWaitTime = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - gpuWaitStart).count();
#endif

        // ========================================================================
        // Extract keypoints from all octaves/layers
        // ========================================================================
#ifdef LAR_PROFILE_METAL_SIFT
        auto cpuExtractStart = std::chrono::high_resolution_clock::now();
#endif

        for (int o = 0; o < nOctaves; o++) {
            int octaveWidth = base.cols >> o;
            int gaussIdx = o * nLevels;

            for (int layer = 1; layer <= nOctaveLayers; layer++) {
                int layerIndex = o * nOctaveLayers + (layer - 1);
                uint32_t* layerExtremaBitarray = (uint32_t*)layerExtremaBitarrays[layerIndex].contents;

                extractKeypoints(
                    layerExtremaBitarray,
                    0,  // No offset needed - each layer has its own buffer
                    o,
                    nLevels,
                    layerExtremaBitarraySizes[layerIndex],
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

        // Cleanup individual layer extrema bitarray buffers
        for (auto& buffer : layerExtremaBitarrays) {
            RELEASE_IF_MANUAL(buffer);
        }

#ifdef LAR_PROFILE_METAL_SIFT
        auto endTotal = std::chrono::high_resolution_clock::now();
        double totalTime = std::chrono::duration<double, std::milli>(
            endTotal - startTotal).count();

        std::cout << "\n=== Metal Pipelined SIFT Profile (Per-Layer) ===\n";
        std::cout << "CPU Operations:\n";
        std::cout << "  Octave prep:         " << cpuPrepTime << " ms\n";
        std::cout << "  Keypoint extraction: " << cpuExtractTime << " ms\n";
        std::cout << "GPU Operations:\n";
        std::cout << "  GPU wait time:       " << gpuWaitTime << " ms (wall-clock)\n";
        std::cout << "  Command buffers:     " << allCommandBuffers.size() << " (per-layer)\n";
        std::cout << "Total:\n";
        std::cout << "  Wall-clock time:     " << totalTime << " ms\n";
        std::cout << "  Speedup potential:   " << (cpuPrepTime + gpuWaitTime) / gpuWaitTime << "x (overlap)\n";
        std::cout << "=================================================\n\n";
#endif
    }
}

} // namespace lar