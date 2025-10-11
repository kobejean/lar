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
// Metal-accelerated fused scale-space extrema detection
// ============================================================================
// Combines: Gaussian blur + DoG computation + Extrema detection
// This is more efficient than the separate approach as it keeps intermediate
// DoG results in threadgroup memory instead of global memory.
//
// Takes base image, computes Gaussian pyramid and DoG on-the-fly, and detects
// extrema in a single fused kernel invocation per layer.
//
// TODO: User will paste the function body here
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

        // Load compiled Metal shader library
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

        // Cache Metal objects as static but with proper lazy initialization
        static id<MTLLibrary> cachedLibrary = nil;
        static id<MTLComputePipelineState> cachedBlurPipeline = nil;
        static id<MTLComputePipelineState> cachedPipeline = nil;

        // Lazy initialization on first call only
        if (!cachedLibrary) {
            cachedLibrary = [device newLibraryWithURL:libraryURL error:&error];
            if (!cachedLibrary) {
                std::cerr << "Failed to load Metal library: " << [[error localizedDescription] UTF8String] << std::endl;
                std::cerr << "Searched path: " << [binPath UTF8String] << std::endl;
                return;
            }

            // Create pipeline for fused blur
            id<MTLFunction> fusedBlurFunc = [cachedLibrary newFunctionWithName:@"gaussianBlurFused"];
            if (!fusedBlurFunc) {
                std::cerr << "Failed to find Metal function: gaussianBlurFused" << std::endl;
                return;
            }

            cachedBlurPipeline = [device newComputePipelineStateWithFunction:fusedBlurFunc error:&error];
            RELEASE_IF_MANUAL(fusedBlurFunc);
            if (!cachedBlurPipeline) {
                std::cerr << "Failed to create blur pipeline: " << [[error localizedDescription] UTF8String] << std::endl;
                return;
            }

            // Create pipeline for fused extrema detection
            id<MTLFunction> fusedFunction = [cachedLibrary newFunctionWithName:@"detectScaleSpaceExtremaFused"];
            if (!fusedFunction) {
                std::cerr << "Failed to find Metal function: detectScaleSpaceExtremaFused" << std::endl;
                return;
            }

            cachedPipeline = [device newComputePipelineStateWithFunction:fusedFunction error:&error];
            RELEASE_IF_MANUAL(fusedFunction);
            if (!cachedPipeline) {
                std::cerr << "Failed to create extrema pipeline: " << [[error localizedDescription] UTF8String] << std::endl;
                return;
            }
        }

        // Use the cached pipelines
        id<MTLComputePipelineState> blurPipeline = cachedBlurPipeline;
        id<MTLComputePipelineState> pipeline = cachedPipeline;

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
        std::vector<std::vector<float>> gaussianKernels(nLevels);
        std::vector<double> sigmas(nLevels);

        // Calculate sigmas (matching SIFT algorithm)
        double k = std::pow(2.0, 1.0 / nOctaveLayers);
        sigmas[0] = sigma;
        for (int i = 1; i < nLevels; i++) {
            double sig_prev = std::pow(k, (double)(i-1)) * sigma;
            double sig_total = sig_prev * k;
            sigmas[i] = std::sqrt(sig_total*sig_total - sig_prev*sig_prev);
            gaussianKernels[i] = createGaussianKernel(sigmas[i]);
        }

#ifdef LAR_PROFILE_METAL_SIFT
        auto gpuStart = std::chrono::high_resolution_clock::now();
#endif

        // DoG pyramid will be computed on-the-fly and wrapped in cv::Mat for adjustLocalExtrema
        // Pre-allocate dog_pyr to hold all DoG layers across all octaves
        std::vector<cv::Mat> dog_pyr(nOctaves * (nOctaveLayers + 2));

        // Allocate gauss_pyr to hold all Gaussian pyramid layers
        // Layout: gauss_pyr[octave * nLevels + level] where level = 0 to nLevels-1
        gauss_pyr.resize(nOctaves * nLevels);



        // Process each octave
        for (int o = 0; o < nOctaves; o++) {
            int octaveWidth = base.cols >> o;
            int octaveHeight = base.rows >> o;

            size_t rowBytes = octaveWidth * sizeof(float);
            size_t alignedRowBytes = ((rowBytes + METAL_BUFFER_ALIGNMENT - 1) / METAL_BUFFER_ALIGNMENT) * METAL_BUFFER_ALIGNMENT;
            int rowStride = (int)(alignedRowBytes / sizeof(float));

            std::vector<id<MTLBuffer>>& gaussBuffers = resources.octaveBuffers[o];
            std::vector<id<MTLBuffer>>& dogBuffers = resources.dogBuffers[o];
            size_t bufferSize = alignedRowBytes * octaveHeight;

            // === Compute Octave Base (Gauss[0]) ===
            // Determine source for this octave
            cv::Mat octaveBase;
            if (o == 0) {
                octaveBase = base;
            } else {
                // Downsample from previous octave's SELF-COMPUTED Gauss[nLevels-3]
                // Read from our own gaussBuffers, not from precomputed gauss_pyr!
                std::vector<id<MTLBuffer>>& prevOctaveBuffers = resources.octaveBuffers[o-1];
                int prevOctaveWidth = base.cols >> (o-1);
                int prevOctaveHeight = base.rows >> (o-1);
                size_t prevRowBytes = prevOctaveWidth * sizeof(float);
                size_t prevAlignedRowBytes = ((prevRowBytes + METAL_BUFFER_ALIGNMENT - 1) / METAL_BUFFER_ALIGNMENT) * METAL_BUFFER_ALIGNMENT;

                // Wrap our computed buffer in cv::Mat
                float* prevBufPtr = (float*)prevOctaveBuffers[nLevels-3].contents;
                cv::Mat prevOctaveLayer(prevOctaveHeight, prevOctaveWidth, CV_32F, prevBufPtr, prevAlignedRowBytes);

                // Downsample to create this octave's base
                cv::resize(prevOctaveLayer, octaveBase,
                          cv::Size(octaveWidth, octaveHeight), 0, 0, cv::INTER_NEAREST);
            }

            // Upload octaveBase to Gauss[0]
            float* gauss0Ptr = (float*)gaussBuffers[0].contents;
            size_t alignedRowFloats = alignedRowBytes / sizeof(float);
            for (int row = 0; row < octaveHeight; row++) {
                memcpy(gauss0Ptr + row * alignedRowFloats,
                      octaveBase.ptr<float>(row),
                      octaveWidth * sizeof(float));
            }

            // === Compute Gauss[1] and Gauss[2] using gaussianBlurFused kernel ===

            for (int i = 1; i <= 2; i++) {
                // Setup Gaussian blur parameters
                GaussianBlurParams blurParams;
                blurParams.width = octaveWidth;
                blurParams.height = octaveHeight;
                blurParams.rowStride = rowStride;
                blurParams.kernelSize = (int)gaussianKernels[i].size();

                id<MTLBuffer> blurParamsBuffer = [device newBufferWithBytes:&blurParams
                                                                      length:sizeof(GaussianBlurParams)
                                                                     options:MTLResourceStorageModeShared];

                id<MTLBuffer> blurKernelBuffer = [device newBufferWithBytes:gaussianKernels[i].data()
                                                                      length:gaussianKernels[i].size() * sizeof(float)
                                                                     options:MTLResourceStorageModeShared];

                // Dispatch gaussianBlurFused: Gauss[i-1] → Gauss[i]
                id<MTLCommandBuffer> blurCommandBuffer = [commandQueue commandBuffer];
                id<MTLComputeCommandEncoder> blurEncoder = [blurCommandBuffer computeCommandEncoder];

                [blurEncoder setComputePipelineState:blurPipeline];
                [blurEncoder setBuffer:gaussBuffers[i-1] offset:0 atIndex:0];  // source
                [blurEncoder setBuffer:gaussBuffers[i] offset:0 atIndex:1];    // destination
                [blurEncoder setBuffer:blurParamsBuffer offset:0 atIndex:2];
                [blurEncoder setBuffer:blurKernelBuffer offset:0 atIndex:3];

                MTLSize gridSize = MTLSizeMake(octaveWidth, octaveHeight, 1);
                MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);

                [blurEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
                [blurEncoder endEncoding];

                [blurCommandBuffer commit];
                [blurCommandBuffer waitUntilCompleted];

                RELEASE_IF_MANUAL(blurParamsBuffer);
                RELEASE_IF_MANUAL(blurKernelBuffer);
            }

            // Pre-compute the first two DoG layers (DoG[0] and DoG[1])
            // These are needed as prevDoG and currDoG for the first fused kernel iteration
            // DoG[0] = Gauss[1] - Gauss[0]
            float* gauss0 = (float*)gaussBuffers[0].contents;
            float* gauss1 = (float*)gaussBuffers[1].contents;
            float* dog0 = (float*)dogBuffers[0].contents;
            for (int pixel = 0; pixel < octaveHeight * rowStride; pixel++) {
                dog0[pixel] = gauss1[pixel] - gauss0[pixel];
            }

            // DoG[1] = Gauss[2] - Gauss[1]
            float* gauss2 = (float*)gaussBuffers[2].contents;
            float* dog1 = (float*)dogBuffers[1].contents;
            for (int pixel = 0; pixel < octaveHeight * rowStride; pixel++) {
                dog1[pixel] = gauss2[pixel] - gauss1[pixel];
            }

            // Populate gauss_pyr with initial Gaussian layers (Gauss[0], Gauss[1], Gauss[2])
            int gaussIdx = o * nLevels;
            gauss_pyr[gaussIdx + 0] = cv::Mat(octaveHeight, octaveWidth, CV_32F, gauss0, alignedRowBytes).clone();
            gauss_pyr[gaussIdx + 1] = cv::Mat(octaveHeight, octaveWidth, CV_32F, gauss1, alignedRowBytes).clone();
            gauss_pyr[gaussIdx + 2] = cv::Mat(octaveHeight, octaveWidth, CV_32F, gauss2, alignedRowBytes).clone();

            // Wrap DoG buffers in cv::Mat for use by adjustLocalExtrema
            // Initialize ALL DoG layers for this octave to prevent accessing uninitialized cv::Mat
            int dogIdx = o * (nOctaveLayers + 2);

            // Initialize DoG[0] and DoG[1] which we just computed
            float* dog0Ptr = (float*)dogBuffers[0].contents;
            float* dog1Ptr = (float*)dogBuffers[1].contents;
            dog_pyr[dogIdx + 0] = cv::Mat(octaveHeight, octaveWidth, CV_32F, dog0Ptr, alignedRowBytes).clone();
            dog_pyr[dogIdx + 1] = cv::Mat(octaveHeight, octaveWidth, CV_32F, dog1Ptr, alignedRowBytes).clone();

            // Pre-initialize remaining DoG slots with empty Mat objects to ensure they're valid
            // They will be populated in the loop below
            for (int i = 2; i < nOctaveLayers + 2; i++) {
                dog_pyr[dogIdx + i] = cv::Mat(octaveHeight, octaveWidth, CV_32F, cv::Scalar(0));
            }

            // Allocate temporary buffers for nextGauss and nextDoG
            id<MTLBuffer> nextGaussBuffer = [device newBufferWithLength:bufferSize
                                                                options:MTLResourceStorageModeShared];
            id<MTLBuffer> nextDogBuffer = [device newBufferWithLength:bufferSize
                                                              options:MTLResourceStorageModeShared];

            // Process layers 1 to nOctaveLayers (where we detect extrema)
            for (int i = 1; i <= nOctaveLayers; i++) {
                // Clear bitarray to zero
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
                params.octave = o;
                params.layer = i;
                params.kernelSize = (int)gaussianKernels[i+2].size(); // Kernel to blur Gauss[i+1] → Gauss[i+2]

                // Create parameter buffer
                id<MTLBuffer> paramsBuffer = [device newBufferWithBytes:&params
                                                                  length:sizeof(FusedExtremaParams)
                                                                 options:MTLResourceStorageModeShared];

                // Create kernel buffer (use i+2 to match ground truth blur kernel)
                id<MTLBuffer> kernelBuffer = [device newBufferWithBytes:gaussianKernels[i+2].data()
                                                                  length:gaussianKernels[i+2].size() * sizeof(float)
                                                                 options:MTLResourceStorageModeShared];

                // Create command buffer
                id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:dogBuffers[i-1] offset:0 atIndex:0];      // prevDoG (read-only, computed in previous iteration)
                [encoder setBuffer:dogBuffers[i] offset:0 atIndex:1];        // currDoG (read-only, computed in previous iteration)
                [encoder setBuffer:gaussBuffers[i+1] offset:0 atIndex:2];    // currGauss (Gauss[i+1], blur to get Gauss[i+2])
                [encoder setBuffer:nextGaussBuffer offset:0 atIndex:3];      // nextGauss (kernel writes Gauss[i+2])
                [encoder setBuffer:dogBuffers[i+1] offset:0 atIndex:4];      // nextDoG (kernel writes DoG[i+1], used for halo in next iteration)
                [encoder setBuffer:bitarrayBuffer offset:0 atIndex:5];       // extremaBitarray (1 bit per pixel)
                [encoder setBuffer:paramsBuffer offset:0 atIndex:7];         // params (FusedExtremaParams)
                [encoder setBuffer:kernelBuffer offset:0 atIndex:9];         // gaussKernel (1D Gaussian weights)

                // Dispatch with 16×16 threadgroups (matching kernel's TILE_SIZE)
                MTLSize gridSize = MTLSizeMake(octaveWidth, octaveHeight, 1);
                MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);

                [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
                [encoder endEncoding];

                [commandBuffer commit];
                [commandBuffer waitUntilCompleted];

                RELEASE_IF_MANUAL(paramsBuffer);
                RELEASE_IF_MANUAL(kernelBuffer);

                // Update pyramid with fused kernel's computed results
                // This makes the fused kernel self-sustaining (no longer needs pre-computed buffers)
                memcpy(gaussBuffers[i+2].contents, nextGaussBuffer.contents, bufferSize);

                // Populate gauss_pyr with the newly computed Gaussian layer (Gauss[i+2])
                float* gaussNextPtr = (float*)gaussBuffers[i+2].contents;
                gauss_pyr[o * nLevels + (i + 2)] = cv::Mat(octaveHeight, octaveWidth, CV_32F, gaussNextPtr, alignedRowBytes).clone();

                // Wrap the updated DoG buffer in cv::Mat for use by adjustLocalExtrema
                // MUST clone because dogBuffers[i+1] will be overwritten in next iteration
                float* dogPtr = (float*)dogBuffers[i+1].contents;
                dog_pyr[o * (nOctaveLayers + 2) + i + 1] = cv::Mat(octaveHeight, octaveWidth, CV_32F, dogPtr, alignedRowBytes).clone();

                // Scan bitarray for set bits (extrema candidates)
#ifdef LAR_PROFILE_METAL_SIFT
                auto cpuStart = std::chrono::high_resolution_clock::now();
#endif
                uint32_t* bitarray = (uint32_t*)bitarrayBuffer.contents;

                // Scan the bitarray for set bits
                for (uint32_t chunkIdx = 0; chunkIdx < octaveBitarraySize; chunkIdx++) {
                    uint32_t chunk = bitarray[chunkIdx];
                    if (chunk == 0) continue;  // Skip empty chunks

                    // Scan individual bits in this chunk
                    for (int bitOffset = 0; bitOffset < 32; bitOffset++) {
                        if (chunk & (1u << bitOffset)) {
                            // Calculate pixel coordinates from bit index
                            uint32_t bitIndex = chunkIdx * 32 + bitOffset;
                            int r = bitIndex / octaveWidth;
                            int c_pos = bitIndex % octaveWidth;

                            // Skip if outside valid bounds
                            if (c_pos < 5 || c_pos >= octaveWidth - 5 ||
                                r < 5 || r >= octaveHeight - 5) {
                                continue;
                            }

                            // Create keypoint for refinement
                            cv::KeyPoint kpt;
                            int layer = params.layer;

                            // Call adjustLocalExtrema for subpixel refinement
                            if (!adjustLocalExtrema(dog_pyr, kpt, params.octave, layer, r, c_pos,
                                                   nOctaveLayers, (float)contrastThreshold,
                                                   (float)edgeThreshold, (float)sigma)) {
                                continue;
                            }

                            // Calculate orientation histogram
                            static const int n = SIFT_ORI_HIST_BINS;
                            float hist[n];
                            float scl_octv = kpt.size * 0.5f / (1 << params.octave);

                            // Read from our self-computed gaussBuffers instead of precomputed gauss_pyr
                            // Wrap the buffer in cv::Mat for calcOrientationHist
                            float* gaussPtr = (float*)gaussBuffers[layer].contents;
                            cv::Mat gaussLayer(octaveHeight, octaveWidth, CV_32F, gaussPtr, alignedRowBytes);

                            float omax = calcOrientationHist(gaussLayer,
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

            RELEASE_IF_MANUAL(nextGaussBuffer);
            RELEASE_IF_MANUAL(nextDogBuffer);
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
