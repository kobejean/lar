// Metal-accelerated Gaussian pyramid for SIFT
// Usage: Build with -DLAR_USE_METAL_SIFT=ON
#import <Metal/Metal.h>
#include <cstdint>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "lar/tracking/sift/sift.h"
#include "sift_metal_common.h"
#include <iostream>
#include <chrono>
#include <vector>

#define LAR_PROFILE_METAL_SIFT 1

namespace lar {

// Thread-safe singleton for Metal resources
MetalSiftResources& getMetalResources() {
    static MetalSiftResources resources;
    return resources;
}

// Create 1D Gaussian kernel using OpenCV's bit-exact implementation
std::vector<float> createGaussianKernel(double sigma) {
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
    GaussianKernelMode kernelMode = GaussianKernelMode::CustomFused;
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
                                                            options:MTLResourceStorageModePrivate];
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

            // Determine which blur implementation to use
            if (kernelMode == GaussianKernelMode::MPS) {
                // === MPS Path (default, most accurate) ===
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
            }
            else if (kernelMode == GaussianKernelMode::CustomSeparable) {
                // === Custom Separable Kernel Path (OpenCV-pattern) ===
                // Load Metal library for custom Gaussian blur kernels
                static id<MTLLibrary> library = nil;
                static id<MTLComputePipelineState> horizPipeline = nil;
                static id<MTLComputePipelineState> vertPipeline = nil;

                if (!library) {
                    NSError* error = nil;
                    NSString* binPath = @"bin/sift.metallib";
                    NSURL* libraryURL = [NSURL fileURLWithPath:binPath];

                    if (![[NSFileManager defaultManager] fileExistsAtPath:binPath]) {
                        NSString* execPath = [[NSBundle mainBundle] executablePath];
                        NSString* execDir = [execPath stringByDeletingLastPathComponent];
                        libraryURL = [NSURL fileURLWithPath:[execDir stringByAppendingPathComponent:@"sift.metallib"]];
                    }

                    library = [device newLibraryWithURL:libraryURL error:&error];
                    if (!library) {
                        std::cerr << "Failed to load custom blur Metal library: " << [[error localizedDescription] UTF8String] << std::endl;
                        return;  // Fatal error
                    }

                    // Create pipeline for horizontal blur
                    id<MTLFunction> horizFunc = [library newFunctionWithName:@"gaussianBlurHorizontal"];
                    if (horizFunc) {
                        horizPipeline = [device newComputePipelineStateWithFunction:horizFunc error:&error];
                        RELEASE_IF_MANUAL(horizFunc);
                    }

                    // Create pipeline for vertical blur
                    id<MTLFunction> vertFunc = [library newFunctionWithName:@"gaussianBlurVertical"];
                    if (vertFunc) {
                        vertPipeline = [device newComputePipelineStateWithFunction:vertFunc error:&error];
                        RELEASE_IF_MANUAL(vertFunc);
                    }

                    if (!horizPipeline || !vertPipeline) {
                        std::cerr << "Failed to create custom blur pipelines" << std::endl;
                        return;  // Fatal error
                    }
                }

                // Use custom separable kernels for Gaussian blur
                id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
                int rowStride = (int)(alignedRowBytes / sizeof(float));

                for (int i = 1; i < nLevels; i++) {
                    // Create exact Gaussian kernel matching OpenCV
                    std::vector<float> kernel = createGaussianKernel(sigmas[i]);
                    int kernelSize = (int)kernel.size();

                    // Create kernel buffer
                    id<MTLBuffer> kernelBuffer = [device newBufferWithBytes:kernel.data()
                                                                      length:kernel.size() * sizeof(float)
                                                                     options:MTLResourceStorageModePrivate];

                    // Setup parameters
                    GaussianBlurParams params;
                    params.width = octaveWidth;
                    params.height = octaveHeight;
                    params.rowStride = rowStride;
                    params.kernelSize = kernelSize;

                    id<MTLBuffer> paramsBuffer = [device newBufferWithBytes:&params
                                                                      length:sizeof(GaussianBlurParams)
                                                                     options:MTLResourceStorageModePrivate];

                    // Horizontal pass: source → temp
                    id<MTLComputeCommandEncoder> horizEncoder = [commandBuffer computeCommandEncoder];
                    [horizEncoder setComputePipelineState:horizPipeline];
                    [horizEncoder setBuffer:levelBuffers[i-1] offset:0 atIndex:0];  // source
                    [horizEncoder setBuffer:resources.tempBuffers[o] offset:0 atIndex:1];  // temp destination
                    [horizEncoder setBuffer:paramsBuffer offset:0 atIndex:2];
                    [horizEncoder setBuffer:kernelBuffer offset:0 atIndex:3];

                    MTLSize gridSize = MTLSizeMake(octaveWidth, octaveHeight, 1);
                    NSUInteger threadExecutionWidth = [horizPipeline threadExecutionWidth];
                    MTLSize threadgroupHorizSize = MTLSizeMake(threadExecutionWidth, 1, 1);
                    MTLSize threadgroupVertSize = MTLSizeMake(4, threadExecutionWidth/4, 1);

                    [horizEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupHorizSize];
                    [horizEncoder endEncoding];

                    // Vertical pass: temp → destination
                    id<MTLComputeCommandEncoder> vertEncoder = [commandBuffer computeCommandEncoder];
                    [vertEncoder setComputePipelineState:vertPipeline];
                    [vertEncoder setBuffer:resources.tempBuffers[o] offset:0 atIndex:0];  // temp source
                    [vertEncoder setBuffer:levelBuffers[i] offset:0 atIndex:1];  // final destination
                    [vertEncoder setBuffer:paramsBuffer offset:0 atIndex:2];
                    [vertEncoder setBuffer:kernelBuffer offset:0 atIndex:3];

                    [vertEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupVertSize];
                    [vertEncoder endEncoding];

                    RELEASE_IF_MANUAL(paramsBuffer);
                    RELEASE_IF_MANUAL(kernelBuffer);
                }

                [commandBuffer commit];
                [commandBuffer waitUntilCompleted];
            }
            else if (kernelMode == GaussianKernelMode::CustomFused) {
                // === Custom Fused Kernel Path (Experimental) ===
                // Load Metal library for fused Gaussian blur kernel
                static id<MTLLibrary> library = nil;
                static id<MTLComputePipelineState> fusedPipeline = nil;

                if (!library) {
                    NSError* error = nil;
                    NSString* binPath = @"bin/sift.metallib";
                    NSURL* libraryURL = [NSURL fileURLWithPath:binPath];

                    if (![[NSFileManager defaultManager] fileExistsAtPath:binPath]) {
                        NSString* execPath = [[NSBundle mainBundle] executablePath];
                        NSString* execDir = [execPath stringByDeletingLastPathComponent];
                        libraryURL = [NSURL fileURLWithPath:[execDir stringByAppendingPathComponent:@"sift.metallib"]];
                    }

                    library = [device newLibraryWithURL:libraryURL error:&error];
                    if (!library) {
                        std::cerr << "Failed to load fused blur Metal library: " << [[error localizedDescription] UTF8String] << std::endl;
                        return;  // Fatal error
                    }

                    // Create pipeline for fused blur
                    id<MTLFunction> fusedFunc = [library newFunctionWithName:@"gaussianBlur"];
                    if (fusedFunc) {
                        fusedPipeline = [device newComputePipelineStateWithFunction:fusedFunc error:&error];
                        RELEASE_IF_MANUAL(fusedFunc);
                    }

                    if (!fusedPipeline) {
                        std::cerr << "Failed to create fused blur pipeline" << std::endl;
                        return;  // Fatal error
                    }
                }

                // Use fused kernel for Gaussian blur
                id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
                int rowStride = (int)(alignedRowBytes / sizeof(float));

                for (int i = 1; i < nLevels; i++) {
                    // Create exact Gaussian kernel matching OpenCV
                    std::vector<float> kernel = createGaussianKernel(sigmas[i]);
                    int kernelSize = (int)kernel.size();

                    // Create kernel buffer
                    id<MTLBuffer> kernelBuffer = [device newBufferWithBytes:kernel.data()
                                                                      length:kernel.size() * sizeof(float)
                                                                     options:MTLResourceStorageModeShared];

                    // Setup parameters
                    GaussianBlurParams params;
                    params.width = octaveWidth;
                    params.height = octaveHeight;
                    params.rowStride = rowStride;
                    params.kernelSize = kernelSize;

                    id<MTLBuffer> paramsBuffer = [device newBufferWithBytes:&params
                                                                      length:sizeof(GaussianBlurParams)
                                                                     options:MTLResourceStorageModeShared];

                    // Fused pass: source → destination (single pass)
                    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
                    [encoder setComputePipelineState:fusedPipeline];
                    [encoder setBuffer:levelBuffers[i-1] offset:0 atIndex:0];  // source
                    [encoder setBuffer:levelBuffers[i] offset:0 atIndex:1];    // destination
                    [encoder setBuffer:paramsBuffer offset:0 atIndex:2];
                    [encoder setBuffer:kernelBuffer offset:0 atIndex:3];

                    // Dispatch with 16×16 threadgroups (tile size)
                    MTLSize gridSize = MTLSizeMake(octaveWidth, octaveHeight, 1);
                    MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);

                    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
                    [encoder endEncoding];

                    RELEASE_IF_MANUAL(paramsBuffer);
                    RELEASE_IF_MANUAL(kernelBuffer);
                }

                [commandBuffer commit];
                [commandBuffer waitUntilCompleted];
            }


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
        dog_pyr.resize(nOctaves * (nLevels - 1));
        for (int o = 0; o < nOctaves; o++) {
            int octaveWidth = gauss_pyr[o * nLevels].cols;
            int octaveHeight = gauss_pyr[o * nLevels].rows;

            size_t rowBytes = octaveWidth * sizeof(float);
            size_t alignedRowBytes = ((rowBytes + METAL_BUFFER_ALIGNMENT - 1) / METAL_BUFFER_ALIGNMENT) * METAL_BUFFER_ALIGNMENT;

            // Bounds check
            if (o >= resources.dogBuffers.size()) {
                std::cerr << "ERROR: dogBuffers octave " << o << " out of bounds (size=" << resources.dogBuffers.size() << ")" << std::endl;
                return;
            }

            std::vector<id<MTLBuffer>>& dogBuffers = resources.dogBuffers[o];

            if (dogBuffers.size() < nLevels - 1) {
                std::cerr << "ERROR: dogBuffers[" << o << "] has size " << dogBuffers.size() << " but need " << (nLevels - 1) << std::endl;
                return;
            }

            for (int i = 0; i < nLevels - 1; i++) {
                if (!dogBuffers[i]) {
                    std::cerr << "ERROR: dogBuffers[" << o << "][" << i << "] is nil!" << std::endl;
                    return;
                }
                // Wrap shared buffer with OpenCV Mat (no copy!)
                float* bufferPtr = (float*)dogBuffers[i].contents;
                dog_pyr[o * (nLevels - 1) + i] = cv::Mat(octaveHeight, octaveWidth, CV_32F,
                                                          bufferPtr, alignedRowBytes);
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
        NSURL* libraryURL = nil;
        id<MTLLibrary> library = nil;

        // Priority 1: Check SPM resource bundle (for Swift Package distribution)
        NSString* resourcePath = [[NSBundle mainBundle] pathForResource:@"sift" ofType:@"metallib"];
        if (resourcePath) {
            libraryURL = [NSURL fileURLWithPath:resourcePath];
            library = [device newLibraryWithURL:libraryURL error:&error];
        }

        // Priority 2: Try runtime bin directory (for standalone C++ builds)
        if (!library) {
            NSString* binPath = @"bin/sift.metallib";
            if ([[NSFileManager defaultManager] fileExistsAtPath:binPath]) {
                libraryURL = [NSURL fileURLWithPath:binPath];
                library = [device newLibraryWithURL:libraryURL error:&error];
            }
        }

        // Priority 3: Try relative to executable
        if (!library) {
            NSString* execPath = [[NSBundle mainBundle] executablePath];
            NSString* execDir = [execPath stringByDeletingLastPathComponent];
            NSString* metalLibPath = [execDir stringByAppendingPathComponent:@"sift.metallib"];
            libraryURL = [NSURL fileURLWithPath:metalLibPath];
            library = [device newLibraryWithURL:libraryURL error:&error];
        }

        if (!library) {
            std::cerr << "Failed to load Metal library from any location" << std::endl;
            std::cerr << "Last error: " << [[error localizedDescription] UTF8String] << std::endl;
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

        // Calculate maximum bitarray size needed (for largest octave)
        int maxWidth = dog_pyr[0].cols;
        int maxHeight = dog_pyr[0].rows;
        uint32_t maxPixels = maxWidth * maxHeight;
        uint32_t bitarraySize = (maxPixels + 31) / 32;  // Number of uint32 chunks

        // Allocate bitarray buffer (1 bit per pixel, packed as uint32)
        id<MTLBuffer> bitarrayBuffer = [device newBufferWithLength:bitarraySize * sizeof(uint32_t)
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
                // Clear bitarray to zero
                uint32_t octavePixels = octaveWidth * octaveHeight;
                uint32_t octaveBitarraySize = (octavePixels + 31) / 32;
                memset(bitarrayBuffer.contents, 0, octaveBitarraySize * sizeof(uint32_t));

                // Setup parameters
                ExtremaParams params;
                params.width = octaveWidth;
                params.height = octaveHeight;
                params.rowStride = rowStride;
                params.threshold = threshold;
                params.border = SIFT_IMG_BORDER;
                params.octave = o;
                params.layer = i;

                // Create parameter buffer
                id<MTLBuffer> paramsBuffer = [device newBufferWithBytes:&params
                                                                  length:sizeof(ExtremaParams)
                                                                 options:MTLResourceStorageModeShared];

                // Create command buffer
                id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:dogBuffers[i-1] offset:0 atIndex:0]; // prevLayer
                [encoder setBuffer:dogBuffers[i]   offset:0 atIndex:1]; // currLayer
                [encoder setBuffer:dogBuffers[i+1] offset:0 atIndex:2]; // nextLayer
                [encoder setBuffer:bitarrayBuffer offset:0 atIndex:3];  // bitarray output
                [encoder setBuffer:paramsBuffer offset:0 atIndex:5];

                // Dispatch threads (one per pixel, excluding border)
                // Use 16×16 threadgroups for better cache locality in 2D image processing
                MTLSize gridSize = MTLSizeMake(octaveWidth, octaveHeight, 1);
                MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);

                [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
                [encoder endEncoding];

                [commandBuffer commit];
                [commandBuffer waitUntilCompleted];

                RELEASE_IF_MANUAL(paramsBuffer);

                // Scan bitarray for set bits (extrema candidates)
#ifdef LAR_PROFILE_METAL_SIFT
                auto cpuStart = std::chrono::high_resolution_clock::now();
#endif
                uint32_t* bitarray = (uint32_t*)bitarrayBuffer.contents;
                int count = 0;
                // Scan the bitarray for set bits
                for (uint32_t chunkIdx = 0; chunkIdx < octaveBitarraySize; chunkIdx++) {
                    uint32_t chunk = bitarray[chunkIdx];
                    if (chunk == 0) continue;  // Skip empty chunks

                    // Scan individual bits in this chunk
                    for (int bitOffset = 0; bitOffset < 32; bitOffset++) {
                        if (chunk & (1u << bitOffset)) {
                            // Calculate pixel coordinates from bit index
                            uint32_t bitIndex = chunkIdx * 32 + bitOffset;
                            int y = bitIndex / octaveWidth;
                            int x = bitIndex % octaveWidth;

                            // Create keypoint for refinement
                            cv::KeyPoint kpt;
                            int layer = params.layer;

                            // Call adjustLocalExtrema (from sift.cpp) for subpixel refinement
                            if (!adjustLocalExtrema(dog_pyr, kpt, params.octave, layer, y, x,
                                                   nOctaveLayers, (float)contrastThreshold,
                                                   (float)edgeThreshold, (float)sigma)) {
                                continue;
                            }

                            // Calculate orientation histogram (from sift.cpp)
                            static const int n = SIFT_ORI_HIST_BINS;
                            float hist[n];
                            float scl_octv = kpt.size * 0.5f / (1 << params.octave);

                            int gaussIdx = params.octave * (nOctaveLayers + 3) + layer;
                            float omax = calcOrientationHist(gauss_pyr[gaussIdx],
                                                            cv::Point(x, y),
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

                std::cout << "extracted keypoints from octave " << params.octave << " layer " << params.layer << " adding " << count << " keypoints" << std::endl;
#ifdef LAR_PROFILE_METAL_SIFT
                cpuTime += std::chrono::duration<double, std::milli>(
                    std::chrono::high_resolution_clock::now() - cpuStart).count();
#endif
            }
        }

#ifdef LAR_PROFILE_METAL_SIFT
        gpuTime = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - gpuStart).count();
#endif

        RELEASE_IF_MANUAL(bitarrayBuffer);
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