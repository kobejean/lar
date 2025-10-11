// Metal-accelerated fused SIFT scale-space extrema detection
// Usage: Build with -DLAR_USE_METAL_SIFT_FUSED=ON
#import <Metal/Metal.h>
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
// Helper Functions for Metal SIFT Processing
// ============================================================================

// Initialize Metal pipelines (cached across calls)
static bool initializeMetalPipelines(
    id<MTLDevice> device,
    id<MTLComputePipelineState>& blurPipeline,
    id<MTLComputePipelineState>& extremaPipeline)
{
    static id<MTLLibrary> cachedLibrary = nil;
    static id<MTLComputePipelineState> cachedBlurPipeline = nil;
    static id<MTLComputePipelineState> cachedPipeline = nil;

    if (!cachedLibrary) {
        NSError* error = nil;

        // Try to load from runtime bin directory first
        NSString* binPath = @"bin/sift_fused.metallib";
        NSURL* libraryURL = [NSURL fileURLWithPath:binPath];

        // Fallback: try relative to executable
        if (![[NSFileManager defaultManager] fileExistsAtPath:binPath]) {
            NSString* execPath = [[NSBundle mainBundle] executablePath];
            NSString* execDir = [execPath stringByDeletingLastPathComponent];
            NSString* metalLibPath = [execDir stringByAppendingPathComponent:@"sift_fused.metallib"];
            libraryURL = [NSURL fileURLWithPath:metalLibPath];
        }

        cachedLibrary = [device newLibraryWithURL:libraryURL error:&error];
        if (!cachedLibrary) {
            std::cerr << "Failed to load Metal library: " << [[error localizedDescription] UTF8String] << std::endl;
            std::cerr << "Searched path: " << [binPath UTF8String] << std::endl;
            return false;
        }

        // Create pipeline for fused blur
        id<MTLFunction> fusedBlurFunc = [cachedLibrary newFunctionWithName:@"gaussianBlurFused"];
        if (!fusedBlurFunc) {
            std::cerr << "Failed to find Metal function: gaussianBlurFused" << std::endl;
            return false;
        }

        cachedBlurPipeline = [device newComputePipelineStateWithFunction:fusedBlurFunc error:&error];
        RELEASE_IF_MANUAL(fusedBlurFunc);
        if (!cachedBlurPipeline) {
            std::cerr << "Failed to create blur pipeline: " << [[error localizedDescription] UTF8String] << std::endl;
            return false;
        }

        // Create pipeline for fused extrema detection
        id<MTLFunction> fusedFunction = [cachedLibrary newFunctionWithName:@"detectScaleSpaceExtremaFused"];
        if (!fusedFunction) {
            std::cerr << "Failed to find Metal function: detectScaleSpaceExtremaFused" << std::endl;
            return false;
        }

        cachedPipeline = [device newComputePipelineStateWithFunction:fusedFunction error:&error];
        RELEASE_IF_MANUAL(fusedFunction);
        if (!cachedPipeline) {
            std::cerr << "Failed to create extrema pipeline: " << [[error localizedDescription] UTF8String] << std::endl;
            return false;
        }
    }

    blurPipeline = cachedBlurPipeline;
    extremaPipeline = cachedPipeline;
    return true;
}

// Compute Gaussian kernels for all pyramid levels
static void computeGaussianKernels(
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
static void allocateOctaveResources(
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

// Upload octave base image to GPU
static void uploadOctaveBase(
    const cv::Mat& base,
    MetalSiftResources& resources,
    int octave,
    int nLevels,
    int octaveWidth,
    int octaveHeight,
    size_t alignedRowBytes)
{
    cv::Mat octaveBase;

    if (octave == 0) {
        octaveBase = base;
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

    // Upload to Gauss[0]
    float* gauss0Ptr = (float*)resources.octaveBuffers[octave][0].contents;
    size_t alignedRowFloats = alignedRowBytes / sizeof(float);

    for (int row = 0; row < octaveHeight; row++) {
        memcpy(gauss0Ptr + row * alignedRowFloats,
              octaveBase.ptr<float>(row),
              octaveWidth * sizeof(float));
    }
}

// Apply Gaussian blur using Metal compute shader
static void applyGaussianBlur(
    id<MTLDevice> device,
    id<MTLCommandQueue> commandQueue,
    id<MTLComputePipelineState> blurPipeline,
    id<MTLBuffer> sourceBuffer,
    id<MTLBuffer> destBuffer,
    const std::vector<float>& kernel,
    int width,
    int height,
    int rowStride)
{
    GaussianBlurParams blurParams;
    blurParams.width = width;
    blurParams.height = height;
    blurParams.rowStride = rowStride;
    blurParams.kernelSize = (int)kernel.size();

    id<MTLBuffer> blurParamsBuffer = [device newBufferWithBytes:&blurParams
                                                          length:sizeof(GaussianBlurParams)
                                                         options:MTLResourceStorageModeShared];

    id<MTLBuffer> blurKernelBuffer = [device newBufferWithBytes:kernel.data()
                                                          length:kernel.size() * sizeof(float)
                                                         options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer> blurCommandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> blurEncoder = [blurCommandBuffer computeCommandEncoder];

    [blurEncoder setComputePipelineState:blurPipeline];
    [blurEncoder setBuffer:sourceBuffer offset:0 atIndex:0];
    [blurEncoder setBuffer:destBuffer offset:0 atIndex:1];
    [blurEncoder setBuffer:blurParamsBuffer offset:0 atIndex:2];
    [blurEncoder setBuffer:blurKernelBuffer offset:0 atIndex:3];

    MTLSize gridSize = MTLSizeMake(width, height, 1);
    MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);

    [blurEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [blurEncoder endEncoding];

    [blurCommandBuffer commit];
    [blurCommandBuffer waitUntilCompleted];

    RELEASE_IF_MANUAL(blurParamsBuffer);
    RELEASE_IF_MANUAL(blurKernelBuffer);
}

// Compute initial Gaussian layers and DoG for an octave
static void computeInitialGaussianAndDoG(
    id<MTLDevice> device,
    id<MTLCommandQueue> commandQueue,
    id<MTLComputePipelineState> blurPipeline,
    MetalSiftResources& resources,
    const std::vector<std::vector<float>>& gaussianKernels,
    int octave,
    int octaveWidth,
    int octaveHeight,
    int rowStride)
{
    std::vector<id<MTLBuffer>>& gaussBuffers = resources.octaveBuffers[octave];
    std::vector<id<MTLBuffer>>& dogBuffers = resources.dogBuffers[octave];

    // Compute Gauss[1] and Gauss[2]
    for (int i = 1; i <= 2; i++) {
        applyGaussianBlur(device, commandQueue, blurPipeline,
                         gaussBuffers[i-1], gaussBuffers[i],
                         gaussianKernels[i], octaveWidth, octaveHeight, rowStride);
    }

    // Compute DoG[0] = Gauss[1] - Gauss[0]
    float* gauss0 = (float*)gaussBuffers[0].contents;
    float* gauss1 = (float*)gaussBuffers[1].contents;
    float* dog0 = (float*)dogBuffers[0].contents;
    for (int pixel = 0; pixel < octaveHeight * rowStride; pixel++) {
        dog0[pixel] = gauss1[pixel] - gauss0[pixel];
    }

    // Compute DoG[1] = Gauss[2] - Gauss[1]
    float* gauss2 = (float*)gaussBuffers[2].contents;
    float* dog1 = (float*)dogBuffers[1].contents;
    for (int pixel = 0; pixel < octaveHeight * rowStride; pixel++) {
        dog1[pixel] = gauss2[pixel] - gauss1[pixel];
    }
}

// Detect extrema in a single layer and extract keypoints
static void detectExtremaInLayer(
    id<MTLDevice> device,
    id<MTLCommandQueue> commandQueue,
    id<MTLComputePipelineState> extremaPipeline,
    MetalSiftResources& resources,
    const std::vector<std::vector<float>>& gaussianKernels,
    id<MTLBuffer> bitarrayBuffer,
    std::vector<cv::Mat>& gauss_pyr,
    std::vector<cv::Mat>& dog_pyr,
    std::vector<cv::KeyPoint>& keypoints,
    int octave,
    int layer,
    int nOctaveLayers,
    int nLevels,
    int octaveWidth,
    int octaveHeight,
    int rowStride,
    float threshold,
    double contrastThreshold,
    double edgeThreshold,
    double sigma,
    double& cpuTime)
{
    std::vector<id<MTLBuffer>>& gaussBuffers = resources.octaveBuffers[octave];
    std::vector<id<MTLBuffer>>& dogBuffers = resources.dogBuffers[octave];

    // Clear bitarray
    uint32_t octavePixels = octaveWidth * octaveHeight;
    uint32_t octaveBitarraySize = (octavePixels + 31) / 32;
    memset(bitarrayBuffer.contents, 0, octaveBitarraySize * sizeof(uint32_t));

    // Setup parameters
    FusedExtremaParams params;
    params.width = octaveWidth;
    params.height = octaveHeight;
    params.rowStride = rowStride;
    params.threshold = threshold;
    params.border = SIFT_IMG_BORDER;
    params.octave = octave;
    params.layer = layer;
    params.kernelSize = (int)gaussianKernels[layer+2].size();

    id<MTLBuffer> paramsBuffer = [device newBufferWithBytes:&params
                                                      length:sizeof(FusedExtremaParams)
                                                     options:MTLResourceStorageModeShared];

    id<MTLBuffer> kernelBuffer = [device newBufferWithBytes:gaussianKernels[layer+2].data()
                                                      length:gaussianKernels[layer+2].size() * sizeof(float)
                                                     options:MTLResourceStorageModeShared];

    // Dispatch extrema detection kernel
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    [encoder setComputePipelineState:extremaPipeline];
    [encoder setBuffer:dogBuffers[layer-1] offset:0 atIndex:0];
    [encoder setBuffer:dogBuffers[layer] offset:0 atIndex:1];
    [encoder setBuffer:gaussBuffers[layer+1] offset:0 atIndex:2];
    [encoder setBuffer:gaussBuffers[layer+2] offset:0 atIndex:3];
    [encoder setBuffer:dogBuffers[layer+1] offset:0 atIndex:4];
    [encoder setBuffer:bitarrayBuffer offset:0 atIndex:5];
    [encoder setBuffer:paramsBuffer offset:0 atIndex:7];
    [encoder setBuffer:kernelBuffer offset:0 atIndex:9];

    MTLSize gridSize = MTLSizeMake(octaveWidth, octaveHeight, 1);
    MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    RELEASE_IF_MANUAL(paramsBuffer);
    RELEASE_IF_MANUAL(kernelBuffer);

    // Scan bitarray and extract keypoints
#ifdef LAR_PROFILE_METAL_SIFT
    auto cpuStart = std::chrono::high_resolution_clock::now();
#endif

    uint32_t* bitarray = (uint32_t*)bitarrayBuffer.contents;
    int gaussIdx = octave * nLevels;

    for (uint32_t chunkIdx = 0; chunkIdx < octaveBitarraySize; chunkIdx++) {
        uint32_t chunk = bitarray[chunkIdx];
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

                float omax = calcOrientationHist(gauss_pyr[gaussIdx + keypoint_layer],
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
                    }
                }
            }
        }
    }

#ifdef LAR_PROFILE_METAL_SIFT
    cpuTime += std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - cpuStart).count();
#endif
}

// Process a single octave: compute Gaussian pyramid, DoG, and detect extrema
static void processOctave(
    id<MTLDevice> device,
    id<MTLCommandQueue> commandQueue,
    id<MTLComputePipelineState> blurPipeline,
    id<MTLComputePipelineState> extremaPipeline,
    const cv::Mat& base,
    MetalSiftResources& resources,
    const std::vector<std::vector<float>>& gaussianKernels,
    id<MTLBuffer> bitarrayBuffer,
    std::vector<cv::Mat>& gauss_pyr,
    std::vector<cv::Mat>& dog_pyr,
    std::vector<cv::KeyPoint>& keypoints,
    int octave,
    int nOctaves,
    int nOctaveLayers,
    int nLevels,
    float threshold,
    double contrastThreshold,
    double edgeThreshold,
    double sigma,
    double& cpuTime)
{
    int octaveWidth = base.cols >> octave;
    int octaveHeight = base.rows >> octave;

    size_t rowBytes = octaveWidth * sizeof(float);
    size_t alignedRowBytes = ((rowBytes + METAL_BUFFER_ALIGNMENT - 1) / METAL_BUFFER_ALIGNMENT) * METAL_BUFFER_ALIGNMENT;
    int rowStride = (int)(alignedRowBytes / sizeof(float));

    // Upload octave base image
    uploadOctaveBase(base, resources, octave, nLevels, octaveWidth, octaveHeight, alignedRowBytes);

    // Compute initial Gaussian layers and DoG
    computeInitialGaussianAndDoG(device, commandQueue, blurPipeline, resources,
                                 gaussianKernels, octave, octaveWidth, octaveHeight, rowStride);

    // Populate gauss_pyr and dog_pyr with cv::Mat wrappers
    int gaussIdx = octave * nLevels;
    int dogIdx = octave * (nLevels - 1);

    std::vector<id<MTLBuffer>>& gaussBuffers = resources.octaveBuffers[octave];
    std::vector<id<MTLBuffer>>& dogBuffers = resources.dogBuffers[octave];

    for (int i = 0; i < nLevels; i++) {
        gauss_pyr[gaussIdx + i] = cv::Mat(octaveHeight, octaveWidth, CV_32F, gaussBuffers[i].contents, alignedRowBytes);
    }
    for (int i = 0; i < nLevels - 1; i++) {
        dog_pyr[dogIdx + i] = cv::Mat(octaveHeight, octaveWidth, CV_32F, dogBuffers[i].contents, alignedRowBytes);
    }

    // Process each layer for extrema detection
    for (int layer = 1; layer <= nOctaveLayers; layer++) {
        detectExtremaInLayer(device, commandQueue, extremaPipeline, resources,
                           gaussianKernels, bitarrayBuffer, gauss_pyr, dog_pyr, keypoints,
                           octave, layer, nOctaveLayers, nLevels,
                           octaveWidth, octaveHeight, rowStride,
                           threshold, contrastThreshold, edgeThreshold, sigma, cpuTime);
    }
}

// ============================================================================
// Metal-accelerated fused scale-space extrema detection
// ============================================================================
// Combines: Gaussian blur + DoG computation + Extrema detection
// This is more efficient than the separate approach as it keeps intermediate
// DoG results in threadgroup memory instead of global memory.
//
// Takes base image, computes Gaussian pyramid and DoG on-the-fly, and detects
// extrema in a single fused kernel invocation per layer.
//
void findScaleSpaceExtremaMetalFused(
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

#ifdef LAR_PROFILE_METAL_SIFT
        auto startTotal = std::chrono::high_resolution_clock::now();
        double gpuTime = 0, cpuTime = 0;
#endif

        // Initialize Metal pipelines
        id<MTLComputePipelineState> blurPipeline = nil;
        id<MTLComputePipelineState> pipeline = nil;

        if (!initializeMetalPipelines(device, blurPipeline, pipeline)) {
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

        // Calculate maximum bitarray size needed (for largest octave)
        int maxWidth = base.cols;
        int maxHeight = base.rows;
        uint32_t maxPixels = maxWidth * maxHeight;
        uint32_t bitarraySize = (maxPixels + 31) / 32;  // Number of uint32 chunks

        // Allocate bitarray buffer (1 bit per pixel, packed as uint32)
        id<MTLBuffer> bitarrayBuffer = [device newBufferWithLength:bitarraySize * sizeof(uint32_t)
                                                            options:MTLResourceStorageModeShared];

        keypoints.clear();

        // Pre-compute Gaussian kernels for all sigma values
        std::vector<std::vector<float>> gaussianKernels;
        std::vector<double> sigmas;
        computeGaussianKernels(nLevels, nOctaveLayers, sigma, gaussianKernels, sigmas);

#ifdef LAR_PROFILE_METAL_SIFT
        auto gpuStart = std::chrono::high_resolution_clock::now();
#endif

        // Pre-allocate pyramid storage
        std::vector<cv::Mat> dog_pyr(nOctaves * (nOctaveLayers + 2));
        gauss_pyr.resize(nOctaves * nLevels);

        // Process each octave
        for (int o = 0; o < nOctaves; o++) {
            processOctave(device, commandQueue, blurPipeline, pipeline,
                         base, resources, gaussianKernels, bitarrayBuffer,
                         gauss_pyr, dog_pyr, keypoints,
                         o, nOctaves, nOctaveLayers, nLevels,
                         threshold, contrastThreshold, edgeThreshold, sigma, cpuTime);
        }

#ifdef LAR_PROFILE_METAL_SIFT
        gpuTime = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - gpuStart).count();
#endif

        RELEASE_IF_MANUAL(bitarrayBuffer);
        // Note: pipeline, blurPipeline, and library are cached as static and should not be released

#ifdef LAR_PROFILE_METAL_SIFT
        auto endTotal = std::chrono::high_resolution_clock::now();
        double totalTime = std::chrono::duration<double, std::milli>(
            endTotal - startTotal).count();
        std::cout << "Metal Fused Extrema Detection Profile:\n";
        std::cout << "  GPU:      " << gpuTime << " ms\n"
                  << "  CPU:      " << cpuTime << " ms\n"
                  << "  Total:    " << totalTime << " ms\n";
#endif
    }
}

} // namespace lar
