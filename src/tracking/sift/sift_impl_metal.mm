// Metal implementation of SIFT::Impl
// This file should only be compiled when LAR_USE_METAL_SIFT is defined
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <chrono>
#include <vector>
#include <stdexcept>

#include "lar/tracking/sift/sift.h"
#include "lar/tracking/sift/sift_common.h"

#if !__has_feature(objc_arc)
    #define RELEASE_IF_MANUAL(obj) [obj release]
#else
    #define RELEASE_IF_MANUAL(obj) (void)0
#endif

#define METAL_BUFFER_ALIGNMENT 16  // Metal prefers 16-byte alignment

namespace lar {

struct ExtremaParams {
    int threshold;
    int border;
};

// Metal-accelerated implementation of SIFT::Impl using GPU compute pipelines
struct SIFT::Impl {
    // Configuration (owned by Impl)
    SIFTConfig config;

    // Metal resources
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> commandQueue = nil;
    id<MTLComputePipelineState> blurAndDoGPipeline = nil;
    id<MTLComputePipelineState> blurHorizPipeline = nil;
    id<MTLComputePipelineState> blurVertPipeline = nil;
    id<MTLComputePipelineState> blurVertAndDoGPipeline = nil;
    id<MTLComputePipelineState> extremaPipeline = nil;
    id<MTLComputePipelineState> resizePipeline = nil;

    // Pre-allocated GPU buffers/textures
    std::vector<std::vector<id<MTLBuffer>>> octaveBuffers;
    std::vector<std::vector<id<MTLTexture>>> octaveTextures;
    std::vector<id<MTLBuffer>> tempBuffers;
    std::vector<id<MTLTexture>> tempTextures;
    std::vector<std::vector<id<MTLBuffer>>> dogBuffers;
    std::vector<std::vector<id<MTLTexture>>> dogTextures;

    // Staging buffer/texture for initial image upload
    id<MTLBuffer> imageBuffer = nil;
    id<MTLTexture> imageTexture = nil;

    // Pre-computed Gaussian kernels (CPU side)
    std::vector<std::vector<float>> gaussianKernels;
    std::vector<double> sigmas;

    // Pre-allocated kernel buffers (GPU side)
    std::vector<id<MTLBuffer>> kernelSizeBuffers;
    std::vector<id<MTLBuffer>> kernelDataBuffers;

    // Pre-allocated extrema bitarray buffers (one per layer across all octaves)
    std::vector<id<MTLBuffer>> extremaBitarrays;
    std::vector<uint32_t> extremaBitarraySizes;

    // Pre-allocated cv::Mat wrappers for GPU buffers (reused across frames)
    std::vector<cv::Mat> gauss_pyr;
    std::vector<cv::Mat> dog_pyr;

    bool initialized = false;

    // Constructor/destructor
    explicit Impl(const SIFTConfig& cfg);
    ~Impl();

    // Delete copy operations
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

    // Allow move operations
    Impl(Impl&& other) noexcept;
    Impl& operator=(Impl&& other) noexcept;

    // Public interface
    void detectAndCompute(cv::InputArray image, cv::InputArray mask,
                         std::vector<cv::KeyPoint>& keypoints,
                         cv::OutputArray descriptors,
                         bool useProvidedKeypoints);

    int descriptorSize() const;
    int descriptorType() const;
    int defaultNorm() const;
    bool isAvailable() const;

    // Metal command encoding helper methods
    void encodeResizeCommand(id cmdBuf, id sourceTexture, id destTexture);
    void encodeInitialBlurCommand(id cmdBuf, id imageTexture, id tempTexture, id destTexture, int level);
    void encodeBlurAndDoGCommand(id cmdBuf, id prevGaussTexture, id gaussTexture, id dogTexture, int level);
    void encodeExtremaDetectionCommand(id cmdBuf, id dogTextureBelow, id dogTextureCenter, id dogTextureAbove, id extremaBitarray);

    // Octave construction strategies
    void encodeStandardOctaveConstruction(id cmdBuf, int octave);
    void encodeBatchedOctaveConstruction(id cmdBuf, int octave);

    // Resource allocation helper
    void allocateOctaveResources(int octave, int octaveWidth, int octaveHeight, int nLevels);
};

static id<MTLLibrary> loadMetalLibrary(id<MTLDevice> device, NSString* libraryName) {
    @autoreleasepool {
        NSError* error = nil;
        NSURL* libraryURL = nil;
        id<MTLLibrary> library = nil;

        // Priority 1: Check SPM resource bundle
        for (NSBundle* bundle in @[[NSBundle mainBundle], [NSBundle bundleForClass:[device class]]]) {
            NSString* bundlePath = [bundle pathForResource:@"LocalizeAR_MetalShaderResources" ofType:@"bundle"];
            if (bundlePath) {
                NSBundle* resourceBundle = [NSBundle bundleWithPath:bundlePath];
                if (resourceBundle) {
                    NSString* resourcePath = [resourceBundle pathForResource:libraryName ofType:@"metallib"];
                    if (resourcePath) {
                        libraryURL = [NSURL fileURLWithPath:resourcePath];
                        library = [device newLibraryWithURL:libraryURL error:&error];
                        if (library) return library;
                    }
                }
            }
        }

        // Priority 2: Try loading default library
        if (!library) {
            @try {
                library = [device newDefaultLibrary];
                if (library) return library;
            } @catch (NSException *exception) {
                // Ignore - will try other methods
            }
        }

        // Priority 3: Try runtime bin directory
        if (!library) {
            NSString* binPath = [NSString stringWithFormat:@"bin/%@.metallib", libraryName];
            if ([[NSFileManager defaultManager] fileExistsAtPath:binPath]) {
                libraryURL = [NSURL fileURLWithPath:binPath];
                library = [device newLibraryWithURL:libraryURL error:&error];
                if (library) return library;
            }
        }

        // Priority 4: Try relative to executable
        if (!library) {
            NSString* execPath = [[NSBundle mainBundle] executablePath];
            NSString* execDir = [execPath stringByDeletingLastPathComponent];
            NSString* metalLibPath = [execDir stringByAppendingPathComponent:[NSString stringWithFormat:@"%@.metallib", libraryName]];
            if ([[NSFileManager defaultManager] fileExistsAtPath:metalLibPath]) {
                libraryURL = [NSURL fileURLWithPath:metalLibPath];
                library = [device newLibraryWithURL:libraryURL error:&error];
                if (library) return library;
            }
        }

        // Priority 5: Try app bundle Resources directory
        if (!library) {
            NSString* resourcesPath = [[NSBundle mainBundle] resourcePath];
            if (resourcesPath) {
                NSString* bundlePath = [resourcesPath stringByAppendingPathComponent:@"LocalizeAR_MetalShaderResources.bundle"];
                if ([[NSFileManager defaultManager] fileExistsAtPath:bundlePath]) {
                    NSBundle* resourceBundle = [NSBundle bundleWithPath:bundlePath];
                    if (resourceBundle) {
                        NSString* resourcePath = [resourceBundle pathForResource:libraryName ofType:@"metallib"];
                        if (resourcePath) {
                            libraryURL = [NSURL fileURLWithPath:resourcePath];
                            library = [device newLibraryWithURL:libraryURL error:&error];
                            if (library) return library;
                        }
                    }
                }
            }
        }

        // Priority 6: Try SPM build directory
        if (!library) {
            NSString* execPath = [[NSBundle mainBundle] executablePath];
            NSString* execDir = [execPath stringByDeletingLastPathComponent];
            NSString* bundlePath = [execDir stringByAppendingPathComponent:@"LocalizeAR_MetalShaderResources.bundle"];
            if ([[NSFileManager defaultManager] fileExistsAtPath:bundlePath]) {
                NSBundle* resourceBundle = [NSBundle bundleWithPath:bundlePath];
                if (resourceBundle) {
                    NSString* resourcePath = [resourceBundle pathForResource:libraryName ofType:@"metallib"];
                    if (resourcePath) {
                        libraryURL = [NSURL fileURLWithPath:resourcePath];
                        library = [device newLibraryWithURL:libraryURL error:&error];
                        if (library) return library;
                    }
                }
            }
        }

        // Priority 7: Try "default" metallib name
        if (!library && ![libraryName isEqualToString:@"default"]) {
            library = loadMetalLibrary(device, @"default");
            if (library) return library;
        }

        // Failed to load from any location
        std::cerr << "Failed to load Metal library '" << [libraryName UTF8String] << "' from any location" << std::endl;
        if (error) {
            std::cerr << "Last error: " << [[error localizedDescription] UTF8String] << std::endl;
        }

        return nil;
    }
}

static void computeGaussianKernels(
    int nLevels,
    int nOctaveLayers,
    double sigma,
    std::vector<std::vector<float>>& kernels,
    std::vector<double>& sigmas,
    bool enableUpsampling = false
    )
{
    kernels.resize(nLevels);
    sigmas.resize(nLevels);

    double k = std::pow(2.0, 1.0 / nOctaveLayers);
    sigmas[0] = sigma;

    if (enableUpsampling) {
        float sig_diff = std::sqrt(std::max((float)(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA * 4), 0.01f));
        kernels[0] = createGaussianKernel(sig_diff);
    } else {
        float sig_diff = std::sqrt(std::max((float)(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA), 0.01f));
        kernels[0] = createGaussianKernel(sig_diff);
    }

    for (int i = 1; i < nLevels; i++) {
        double sig_prev = std::pow(k, (double)(i-1)) * sigma;
        double sig_total = sig_prev * k;
        sigmas[i] = std::sqrt(sig_total*sig_total - sig_prev*sig_prev);
        kernels[i] = createGaussianKernel(sigmas[i]);
    }
}

// Resource allocation helper - implemented as member function
void SIFT::Impl::allocateOctaveResources(int octave, int octaveWidth, int octaveHeight, int nLevels)
{
    size_t rowBytes = octaveWidth * sizeof(float);
    size_t alignedRowBytes = ((rowBytes + METAL_BUFFER_ALIGNMENT - 1) / METAL_BUFFER_ALIGNMENT) * METAL_BUFFER_ALIGNMENT;
    size_t bufferSize = alignedRowBytes * octaveHeight;

    octaveBuffers[octave].resize(nLevels);
    octaveTextures[octave].resize(nLevels);
    dogBuffers[octave].resize(nLevels - 1);
    dogTextures[octave].resize(nLevels - 1);

    // Allocate Gaussian pyramid buffers/textures
    for (int i = 0; i < nLevels; i++) {
        octaveBuffers[octave][i] = [device newBufferWithLength:bufferSize
                                                      options:MTLResourceStorageModeShared];

        MTLTextureDescriptor* desc = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float
            width:octaveWidth height:octaveHeight mipmapped:NO];
        desc.storageMode = MTLStorageModeShared;
        desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;

        octaveTextures[octave][i] = [octaveBuffers[octave][i]
                                               newTextureWithDescriptor:desc
                                               offset:0
                                               bytesPerRow:alignedRowBytes];
    }

    // Allocate DoG buffers/textures
    for (int i = 0; i < nLevels - 1; i++) {
        dogBuffers[octave][i] = [device newBufferWithLength:bufferSize
                                                  options:MTLResourceStorageModeShared];

        MTLTextureDescriptor* dogDesc = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float
            width:octaveWidth height:octaveHeight mipmapped:NO];
        dogDesc.storageMode = MTLStorageModeShared;
        dogDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;

        dogTextures[octave][i] = [dogBuffers[octave][i]
                                            newTextureWithDescriptor:dogDesc
                                            offset:0
                                            bytesPerRow:alignedRowBytes];
    }

    // Allocate temporary texture for separable convolution
    tempBuffers[octave] = [device newBufferWithLength:bufferSize
                                        options:MTLResourceStorageModeShared];
    MTLTextureDescriptor* tempDesc = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float
        width:octaveWidth height:octaveHeight mipmapped:NO];
    tempDesc.storageMode = MTLStorageModeShared;
    tempDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    tempTextures[octave] = [tempBuffers[octave] newTextureWithDescriptor:tempDesc
                                                                                     offset:0
                                                                                bytesPerRow:alignedRowBytes];
}

// Pass 1: Extract keypoints only (lightweight, determines exact count)
static void extractKeypoints(
    uint32_t* bitarray,
    int octave,
    int octaveBitarraySize,
    int octaveWidth,
    int layer,
    int nOctaveLayers,
    float contrastThreshold,
    float edgeThreshold,
    float sigma,
    const std::vector<cv::Mat>& gauss_pyr,
    const std::vector<cv::Mat>& dog_pyr,
    std::vector<cv::KeyPoint>& keypoints)
{
    int count = 0;
    for (uint32_t chunkIdx = 0; chunkIdx < octaveBitarraySize; chunkIdx++) {
        uint32_t chunk = bitarray[chunkIdx];
        if (chunk == 0) continue;

        for (int bitOffset = 0; bitOffset < 32; bitOffset++) {
            if (chunk & (1u << bitOffset)) {
                uint32_t bitIndex = chunkIdx * 32 + bitOffset;
                int r = bitIndex / octaveWidth;
                int c_pos = bitIndex % octaveWidth;

                cv::KeyPoint kpt;
                int keypoint_layer = layer;

                if (!adjustLocalExtrema(dog_pyr, kpt, octave, keypoint_layer, r, c_pos,
                                    nOctaveLayers, contrastThreshold, edgeThreshold, sigma)) {
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
    std::cout << "octave " << octave << " layer " << layer << " added " << count << " keypoints" << std::endl;
}

// Pass 2: Compute descriptors for all keypoints (exact pre-allocation)
static void computeDescriptors(
    const std::vector<cv::KeyPoint>& keypoints,
    int octave,
    int nOctaveLayers,
    const std::vector<cv::Mat>& gauss_pyr,
    cv::Mat& descriptors,
    int descriptorType)
{
    const int descriptorSize = SIFT_DESCR_WIDTH * SIFT_DESCR_WIDTH * SIFT_DESCR_HIST_BINS;
    int numKeypoints = static_cast<int>(keypoints.size());

    if (numKeypoints == 0) {
        return;
    }

    // Exact pre-allocation - no wasted memory
    descriptors.create(numKeypoints, descriptorSize, descriptorType);

    for (int i = 0; i < numKeypoints; i++) {
        const cv::KeyPoint& kpt = keypoints[i];

        // Get the layer from the keypoint's octave packing
        int keypoint_layer = (kpt.octave >> 8) & 255;
        int finalGaussIdx = octave * (nOctaveLayers + 3) + keypoint_layer;
        const cv::Mat& img = gauss_pyr[finalGaussIdx];

        // Scale to octave space
        float octaveScale = 1.f / (1 << octave);
        cv::Point2f ptf(kpt.pt.x * octaveScale, kpt.pt.y * octaveScale);
        float scl = kpt.size * 0.5f * octaveScale;

        float angle = 360.f - kpt.angle;
        if (std::abs(angle - 360.f) < FLT_EPSILON)
            angle = 0.f;

        calcSIFTDescriptor(img, ptf, angle,
                         scl, SIFT_DESCR_WIDTH, SIFT_DESCR_HIST_BINS,
                         descriptors, i);
    }
}

// Metal command encoding methods
void SIFT::Impl::encodeResizeCommand(
    id cmdBuf,
    id sourceTexture,
    id destTexture)
{
    id<MTLTexture> destTex = (id<MTLTexture>)destTexture;
    int octaveWidth = (int)destTex.width;
    int octaveHeight = (int)destTex.height;

    id<MTLComputeCommandEncoder> resizeEncoder = [(id<MTLCommandBuffer>)cmdBuf computeCommandEncoder];
    [resizeEncoder setComputePipelineState:resizePipeline];
    [resizeEncoder setTexture:(id<MTLTexture>)sourceTexture atIndex:0];
    [resizeEncoder setTexture:destTex atIndex:1];

    MTLSize resizeGridSize = MTLSizeMake(octaveWidth, octaveHeight, 1);
    MTLSize resizeThreadgroupSize = MTLSizeMake(16, 16, 1);

    [resizeEncoder dispatchThreads:resizeGridSize threadsPerThreadgroup:resizeThreadgroupSize];
    [resizeEncoder endEncoding];
}

void SIFT::Impl::encodeInitialBlurCommand(
    id cmdBuf,
    id imageTexture,
    id tempTexture,
    id destTexture,
    int level)
{
    id<MTLTexture> destTex = (id<MTLTexture>)destTexture;
    int octaveWidth = (int)destTex.width;
    int octaveHeight = (int)destTex.height;

    MTLSize gridSize = MTLSizeMake(octaveWidth, octaveHeight, 1);
    MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);

    // Horizontal gaussian blur (staging → temp)
    id<MTLComputeCommandEncoder> blurHorizEncoder = [(id<MTLCommandBuffer>)cmdBuf computeCommandEncoder];
    [blurHorizEncoder setComputePipelineState:blurHorizPipeline];
    [blurHorizEncoder setTexture:(id<MTLTexture>)imageTexture atIndex:0];
    [blurHorizEncoder setTexture:(id<MTLTexture>)tempTexture atIndex:1];
    [blurHorizEncoder setBuffer:kernelSizeBuffers[level] offset:0 atIndex:0];
    [blurHorizEncoder setBuffer:kernelDataBuffers[level] offset:0 atIndex:1];
    [blurHorizEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [blurHorizEncoder endEncoding];

    // Vertical gaussian blur (temp → destTexture)
    id<MTLComputeCommandEncoder> blurVertEncoder = [(id<MTLCommandBuffer>)cmdBuf computeCommandEncoder];
    [blurVertEncoder setComputePipelineState:blurVertPipeline];
    [blurVertEncoder setTexture:(id<MTLTexture>)tempTexture atIndex:0];
    [blurVertEncoder setTexture:destTex atIndex:1];
    [blurVertEncoder setBuffer:kernelSizeBuffers[level] offset:0 atIndex:0];
    [blurVertEncoder setBuffer:kernelDataBuffers[level] offset:0 atIndex:1];
    [blurVertEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [blurVertEncoder endEncoding];
}

void SIFT::Impl::encodeBlurAndDoGCommand(
    id cmdBuf,
    id prevGaussTexture,
    id gaussTexture,
    id dogTexture,
    int level)
{
    id<MTLTexture> gaussTex = (id<MTLTexture>)gaussTexture;
    int octaveWidth = (int)gaussTex.width;
    int octaveHeight = (int)gaussTex.height;

    id<MTLComputeCommandEncoder> encoder = [(id<MTLCommandBuffer>)cmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:blurAndDoGPipeline];
    [encoder setTexture:(id<MTLTexture>)prevGaussTexture atIndex:0];
    [encoder setTexture:gaussTex atIndex:1];
    [encoder setTexture:(id<MTLTexture>)dogTexture atIndex:2];
    [encoder setBuffer:kernelSizeBuffers[level] offset:0 atIndex:0];
    [encoder setBuffer:kernelDataBuffers[level] offset:0 atIndex:1];

    MTLSize gridSize = MTLSizeMake(octaveWidth, octaveHeight, 1);
    MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
}

void SIFT::Impl::encodeExtremaDetectionCommand(
    id cmdBuf,
    id dogTextureBelow,
    id dogTextureCenter,
    id dogTextureAbove,
    id extremaBitarray)
{
    id<MTLTexture> centerTex = (id<MTLTexture>)dogTextureCenter;
    int octaveWidth = (int)centerTex.width;
    int octaveHeight = (int)centerTex.height;

    ExtremaParams params;
    params.threshold = config.threshold;
    params.border = SIFT_IMG_BORDER;

    id<MTLBuffer> paramsBuffer = [device newBufferWithBytes:&params
                                                            length:sizeof(ExtremaParams)
                                                           options:MTLResourceStorageModeShared];

    id<MTLComputeCommandEncoder> encoder = [(id<MTLCommandBuffer>)cmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:extremaPipeline];
    [encoder setTexture:(id<MTLTexture>)dogTextureBelow atIndex:0];
    [encoder setTexture:centerTex atIndex:1];
    [encoder setTexture:(id<MTLTexture>)dogTextureAbove atIndex:2];
    [encoder setBuffer:(id<MTLBuffer>)extremaBitarray offset:0 atIndex:0];
    [encoder setBuffer:paramsBuffer offset:0 atIndex:1];

    MTLSize gridSize = MTLSizeMake(octaveWidth, octaveHeight, 1);
    MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    RELEASE_IF_MANUAL(paramsBuffer);
}

// Octave construction strategies

void SIFT::Impl::encodeStandardOctaveConstruction(
    id cmdBuf,
    int octave)
{
    // STRATEGY: Interleaved blur+extrema detection (original SIFT approach)
    std::vector<id<MTLTexture>>& gaussTextures = octaveTextures[octave];
    std::vector<id<MTLTexture>>& dogTexturesVec = dogTextures[octave];

    for (int layer = 1; layer <= config.nOctaveLayers; layer++) {
        if (layer == 1) {
            if (octave > 0) {
                std::vector<id<MTLTexture>>& prevGaussTextures = octaveTextures[octave - 1];
                encodeResizeCommand(cmdBuf,
                                  prevGaussTextures[config.nOctaveLayers], gaussTextures[0]);
            } else {
                encodeInitialBlurCommand(cmdBuf,
                                       imageTexture, tempTextures[octave], gaussTextures[0],
                                       0);
            }
        }

        int blurStart = (layer == 1) ? 1 : (layer + 2);
        int blurEnd = (layer == 1) ? 4 : (layer + 3);

        for (int i = blurStart; i < blurEnd && i < config.nLevels; i++) {
            encodeBlurAndDoGCommand(cmdBuf,
                                  gaussTextures[i-1], gaussTextures[i], dogTexturesVec[i-1],
                                  i);
        }

        int layerIndex = octave * config.nOctaveLayers + (layer - 1);
        encodeExtremaDetectionCommand(cmdBuf,
                                    dogTexturesVec[layer-1], dogTexturesVec[layer], dogTexturesVec[layer+1],
                                    extremaBitarrays[layerIndex]);
    }
}

void SIFT::Impl::encodeBatchedOctaveConstruction(
    id cmdBuf,
    int octave)
{
    // STRATEGY: Fully batched pyramid construction
    std::vector<id<MTLTexture>>& gaussTextures = octaveTextures[octave];
    std::vector<id<MTLTexture>>& dogTexturesVec = dogTextures[octave];

    // Phase 1: Build complete Gaussian pyramid + DoG
    if (octave > 0) {
        std::vector<id<MTLTexture>>& prevGaussTextures = octaveTextures[octave - 1];
        encodeResizeCommand(cmdBuf,
                          prevGaussTextures[config.nOctaveLayers], gaussTextures[0]);
    } else {
        encodeInitialBlurCommand(cmdBuf,
                               imageTexture, tempTextures[octave], gaussTextures[0],
                               0);
    }

    for (int i = 1; i < config.nLevels; i++) {
        encodeBlurAndDoGCommand(cmdBuf,
                              gaussTextures[i-1], gaussTextures[i], dogTexturesVec[i-1],
                              i);
    }

    // Phase 2: Detect extrema across all layers
    for (int layer = 1; layer <= config.nOctaveLayers; layer++) {
        int layerIndex = octave * config.nOctaveLayers + (layer - 1);
        encodeExtremaDetectionCommand(cmdBuf,
                                    dogTexturesVec[layer-1], dogTexturesVec[layer], dogTexturesVec[layer+1],
                                    extremaBitarrays[layerIndex]);
    }
}

// SIFT::Impl Constructor
SIFT::Impl::Impl(const SIFTConfig& cfg)
    : config(cfg)
{
    @autoreleasepool {
        // Initialize Metal device
        device = MTLCreateSystemDefaultDevice();
        if (!device) {
            throw std::runtime_error("Metal device not available on this system");
        }

        // Create command queue
        commandQueue = [device newCommandQueue];
        if (!commandQueue) {
            throw std::runtime_error("Failed to create Metal command queue");
        }

        // Load Metal library
        id<MTLLibrary> library = loadMetalLibrary(device, @"sift");
        if (!library) {
            throw std::runtime_error("Failed to load Metal shader library");
        }

        NSError* error = nil;

        // Create blur+DoG pipeline
        id<MTLFunction> blurFunc = [library newFunctionWithName:@"gaussianBlurAndDoGFused"];
        if (!blurFunc) {
            throw std::runtime_error("Failed to find Metal function: gaussianBlurAndDoGFused");
        }
        blurAndDoGPipeline = [device newComputePipelineStateWithFunction:blurFunc error:&error];
        RELEASE_IF_MANUAL(blurFunc);
        if (!blurAndDoGPipeline) {
            std::string errorMsg = "Failed to create blur pipeline: ";
            if (error) errorMsg += [[error localizedDescription] UTF8String];
            throw std::runtime_error(errorMsg);
        }

        // Create blur horizontal pipeline
        id<MTLFunction> blurHorizFunc = [library newFunctionWithName:@"gaussianBlurHorizontal"];
        if (!blurHorizFunc) {
            throw std::runtime_error("Failed to find Metal function: gaussianBlurHorizontal");
        }
        blurHorizPipeline = [device newComputePipelineStateWithFunction:blurHorizFunc error:&error];
        RELEASE_IF_MANUAL(blurHorizFunc);
        if (!blurHorizPipeline) {
            std::string errorMsg = "Failed to create blur horizontal pipeline: ";
            if (error) errorMsg += [[error localizedDescription] UTF8String];
            throw std::runtime_error(errorMsg);
        }

        // Create blur vertical pipeline
        id<MTLFunction> blurVertFunc = [library newFunctionWithName:@"gaussianBlurVertical"];
        if (!blurVertFunc) {
            throw std::runtime_error("Failed to find Metal function: gaussianBlurVertical");
        }
        blurVertPipeline = [device newComputePipelineStateWithFunction:blurVertFunc error:&error];
        RELEASE_IF_MANUAL(blurVertFunc);
        if (!blurVertPipeline) {
            std::string errorMsg = "Failed to create blur vertical pipeline: ";
            if (error) errorMsg += [[error localizedDescription] UTF8String];
            throw std::runtime_error(errorMsg);
        }

        // Create blur vertical and DoG pipeline
        id<MTLFunction> blurVertAndDoGFunc = [library newFunctionWithName:@"gaussianBlurVerticalAndDoG"];
        if (!blurVertAndDoGFunc) {
            throw std::runtime_error("Failed to find Metal function: gaussianBlurVerticalAndDoG");
        }
        blurVertAndDoGPipeline = [device newComputePipelineStateWithFunction:blurVertAndDoGFunc error:&error];
        RELEASE_IF_MANUAL(blurVertAndDoGFunc);
        if (!blurVertAndDoGPipeline) {
            std::string errorMsg = "Failed to create blur vertical and DoG pipeline: ";
            if (error) errorMsg += [[error localizedDescription] UTF8String];
            throw std::runtime_error(errorMsg);
        }

        // Create extrema detection pipeline
        id<MTLFunction> extremaFunc = [library newFunctionWithName:@"detectExtrema"];
        if (!extremaFunc) {
            throw std::runtime_error("Failed to find Metal function: detectExtrema");
        }
        extremaPipeline = [device newComputePipelineStateWithFunction:extremaFunc error:&error];
        RELEASE_IF_MANUAL(extremaFunc);
        if (!extremaPipeline) {
            std::string errorMsg = "Failed to create extrema pipeline: ";
            if (error) errorMsg += [[error localizedDescription] UTF8String];
            throw std::runtime_error(errorMsg);
        }

        // Create resize pipeline
        id<MTLFunction> resizeFunc = [library newFunctionWithName:@"resizeNearestNeighbor2x"];
        if (!resizeFunc) {
            throw std::runtime_error("Failed to find Metal function: resizeNearestNeighbor2x");
        }
        resizePipeline = [device newComputePipelineStateWithFunction:resizeFunc error:&error];
        RELEASE_IF_MANUAL(resizeFunc);
        if (!resizePipeline) {
            std::string errorMsg = "Failed to create resize pipeline: ";
            if (error) errorMsg += [[error localizedDescription] UTF8String];
            throw std::runtime_error(errorMsg);
        }

        // Pre-compute Gaussian kernels once
        computeGaussianKernels(config.nLevels, config.nOctaveLayers, config.sigma,
                                gaussianKernels, sigmas, config.enableUpsampling);

        // Pre-allocate kernel buffers for GPU (one set per pyramid level)
        kernelSizeBuffers.resize(config.nLevels);
        kernelDataBuffers.resize(config.nLevels);
        for (int i = 0; i < config.nLevels; i++) {
            int kernelSize = (int)gaussianKernels[i].size();

            kernelSizeBuffers[i] = [device newBufferWithBytes:&kernelSize
                                                        length:sizeof(int)
                                                       options:MTLResourceStorageModeShared];

            kernelDataBuffers[i] = [device newBufferWithBytes:gaussianKernels[i].data()
                                                        length:gaussianKernels[i].size() * sizeof(float)
                                                       options:MTLResourceStorageModeShared];
        }

        // Pre-allocate all octave buffers and textures
        octaveBuffers.resize(config.nOctaves);
        octaveTextures.resize(config.nOctaves);
        dogBuffers.resize(config.nOctaves);
        dogTextures.resize(config.nOctaves);
        tempBuffers.resize(config.nOctaves);
        tempTextures.resize(config.nOctaves);

        for (int o = 0; o < config.nOctaves; o++) {
            int octaveWidth = config.imageSize.width >> o;
            int octaveHeight = config.imageSize.height >> o;
            allocateOctaveResources(o, octaveWidth, octaveHeight, config.nLevels);
        }

        // Allocate staging buffer/texture for octave 0 initial image upload
        {
            int baseWidth = config.imageSize.width;
            int baseHeight = config.imageSize.height;
            size_t rowBytes = baseWidth * sizeof(float);
            size_t alignedRowBytes = ((rowBytes + METAL_BUFFER_ALIGNMENT - 1) / METAL_BUFFER_ALIGNMENT) * METAL_BUFFER_ALIGNMENT;
            size_t bufferSize = alignedRowBytes * baseHeight;

            imageBuffer = [device newBufferWithLength:bufferSize
                                        options:MTLResourceStorageModeShared];

            MTLTextureDescriptor* imageDesc = [MTLTextureDescriptor
                texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float
                width:baseWidth height:baseHeight mipmapped:NO];
            imageDesc.storageMode = MTLStorageModeShared;
            imageDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;

            imageTexture = [imageBuffer newTextureWithDescriptor:imageDesc
                                                                offset:0
                                                           bytesPerRow:alignedRowBytes];
        }

        // Pre-allocate extrema bitarray buffers (one per layer across all octaves)
        extremaBitarrays.reserve(config.nOctaves * config.nOctaveLayers);
        extremaBitarraySizes.reserve(config.nOctaves * config.nOctaveLayers);

        for (int o = 0; o < config.nOctaves; o++) {
            int octaveWidth = config.imageSize.width >> o;
            int octaveHeight = config.imageSize.height >> o;
            uint32_t octavePixels = octaveWidth * octaveHeight;
            uint32_t extremaBitarraySize = ((octavePixels + 31) / 32) * sizeof(uint32_t);

            for (int layer = 1; layer <= config.nOctaveLayers; layer++) {
                id<MTLBuffer> buffer = [device newBufferWithLength:extremaBitarraySize
                                                          options:MTLResourceStorageModeShared];
                extremaBitarrays.push_back(buffer);
                extremaBitarraySizes.push_back(extremaBitarraySize / sizeof(uint32_t));
            }
        }

        // Pre-allocate cv::Mat wrappers for GPU buffers (reused across frames)
        gauss_pyr.resize(config.nOctaves * config.nLevels);
        dog_pyr.resize(config.nOctaves * (config.nLevels - 1));

        for (int o = 0; o < config.nOctaves; o++) {
            int octaveWidth = config.imageSize.width >> o;
            int octaveHeight = config.imageSize.height >> o;
            size_t rowBytes = octaveWidth * sizeof(float);
            size_t alignedRowBytes = ((rowBytes + METAL_BUFFER_ALIGNMENT - 1) / METAL_BUFFER_ALIGNMENT) * METAL_BUFFER_ALIGNMENT;

            int gaussIdx = o * config.nLevels;
            int dogIdx = o * (config.nLevels - 1);

            std::vector<id<MTLBuffer>>& gaussBuffers = octaveBuffers[o];
            std::vector<id<MTLBuffer>>& dogBuffersVec = dogBuffers[o];

            // Create cv::Mat wrappers pointing to GPU buffers
            for (int i = 0; i < config.nLevels; i++) {
                gauss_pyr[gaussIdx + i] = cv::Mat(octaveHeight, octaveWidth, CV_32F, gaussBuffers[i].contents, alignedRowBytes);
            }

            for (int i = 0; i < config.nLevels - 1; i++) {
                dog_pyr[dogIdx + i] = cv::Mat(octaveHeight, octaveWidth, CV_32F, dogBuffersVec[i].contents, alignedRowBytes);
            }
        }

        initialized = true;
    }
}

SIFT::Impl::~Impl() {
    @autoreleasepool {
        for (auto& octave : octaveTextures) {
            for (auto& tex : octave) RELEASE_IF_MANUAL(tex);
        }
        for (auto& octave : octaveBuffers) {
            for (auto& buf : octave) RELEASE_IF_MANUAL(buf);
        }
        for (auto& octave : dogTextures) {
            for (auto& tex : octave) RELEASE_IF_MANUAL(tex);
        }
        for (auto& octave : dogBuffers) {
            for (auto& buf : octave) RELEASE_IF_MANUAL(buf);
        }
        for (auto& tex : tempTextures) RELEASE_IF_MANUAL(tex);
        for (auto& buf : tempBuffers) RELEASE_IF_MANUAL(buf);
        RELEASE_IF_MANUAL(imageTexture);
        RELEASE_IF_MANUAL(imageBuffer);
        for (auto& buf : kernelSizeBuffers) RELEASE_IF_MANUAL(buf);
        for (auto& buf : kernelDataBuffers) RELEASE_IF_MANUAL(buf);
        for (auto& buf : extremaBitarrays) RELEASE_IF_MANUAL(buf);
        octaveTextures.clear();
        octaveBuffers.clear();
        dogTextures.clear();
        dogBuffers.clear();
        tempTextures.clear();
        tempBuffers.clear();
        kernelSizeBuffers.clear();
        kernelDataBuffers.clear();
        extremaBitarrays.clear();
        extremaBitarraySizes.clear();
        RELEASE_IF_MANUAL(blurAndDoGPipeline);
        RELEASE_IF_MANUAL(blurHorizPipeline);
        RELEASE_IF_MANUAL(blurVertPipeline);
        RELEASE_IF_MANUAL(blurVertAndDoGPipeline);
        RELEASE_IF_MANUAL(extremaPipeline);
        RELEASE_IF_MANUAL(resizePipeline);
        RELEASE_IF_MANUAL(commandQueue);
        RELEASE_IF_MANUAL(device);
#if __has_feature(objc_arc)
        blurAndDoGPipeline = nil;
        blurHorizPipeline = nil;
        blurVertPipeline = nil;
        blurVertAndDoGPipeline = nil;
        extremaPipeline = nil;
        resizePipeline = nil;
        commandQueue = nil;
        device = nil;
#endif
    }
}

SIFT::Impl::Impl(Impl&& other) noexcept
    : config(other.config)
    , device(other.device)
    , commandQueue(other.commandQueue)
    , blurAndDoGPipeline(other.blurAndDoGPipeline)
    , blurHorizPipeline(other.blurHorizPipeline)
    , blurVertPipeline(other.blurVertPipeline)
    , blurVertAndDoGPipeline(other.blurVertAndDoGPipeline)
    , extremaPipeline(other.extremaPipeline)
    , resizePipeline(other.resizePipeline)
    , octaveBuffers(std::move(other.octaveBuffers))
    , octaveTextures(std::move(other.octaveTextures))
    , tempBuffers(std::move(other.tempBuffers))
    , tempTextures(std::move(other.tempTextures))
    , dogBuffers(std::move(other.dogBuffers))
    , dogTextures(std::move(other.dogTextures))
    , imageBuffer(other.imageBuffer)
    , imageTexture(other.imageTexture)
    , gaussianKernels(std::move(other.gaussianKernels))
    , sigmas(std::move(other.sigmas))
    , kernelSizeBuffers(std::move(other.kernelSizeBuffers))
    , kernelDataBuffers(std::move(other.kernelDataBuffers))
    , extremaBitarrays(std::move(other.extremaBitarrays))
    , extremaBitarraySizes(std::move(other.extremaBitarraySizes))
    , gauss_pyr(std::move(other.gauss_pyr))
    , dog_pyr(std::move(other.dog_pyr))
    , initialized(other.initialized)
{
    // Clear other's resources so destructor doesn't release them
    other.device = nil;
    other.commandQueue = nil;
    other.blurAndDoGPipeline = nil;
    other.blurHorizPipeline = nil;
    other.blurVertPipeline = nil;
    other.blurVertAndDoGPipeline = nil;
    other.extremaPipeline = nil;
    other.resizePipeline = nil;
    other.imageBuffer = nil;
    other.imageTexture = nil;
    other.initialized = false;
}

SIFT::Impl& SIFT::Impl::operator=(Impl&& other) noexcept {
    if (this != &other) {
        // Clean up existing resources
        this->~Impl();

        // Move from other
        config = other.config;
        device = other.device;
        commandQueue = other.commandQueue;
        blurAndDoGPipeline = other.blurAndDoGPipeline;
        blurHorizPipeline = other.blurHorizPipeline;
        blurVertPipeline = other.blurVertPipeline;
        blurVertAndDoGPipeline = other.blurVertAndDoGPipeline;
        extremaPipeline = other.extremaPipeline;
        resizePipeline = other.resizePipeline;
        octaveBuffers = std::move(other.octaveBuffers);
        octaveTextures = std::move(other.octaveTextures);
        tempBuffers = std::move(other.tempBuffers);
        tempTextures = std::move(other.tempTextures);
        dogBuffers = std::move(other.dogBuffers);
        dogTextures = std::move(other.dogTextures);
        imageBuffer = other.imageBuffer;
        imageTexture = other.imageTexture;
        gaussianKernels = std::move(other.gaussianKernels);
        sigmas = std::move(other.sigmas);
        kernelSizeBuffers = std::move(other.kernelSizeBuffers);
        kernelDataBuffers = std::move(other.kernelDataBuffers);
        extremaBitarrays = std::move(other.extremaBitarrays);
        extremaBitarraySizes = std::move(other.extremaBitarraySizes);
        gauss_pyr = std::move(other.gauss_pyr);
        dog_pyr = std::move(other.dog_pyr);
        initialized = other.initialized;

        // Clear other
        other.device = nil;
        other.commandQueue = nil;
        other.blurAndDoGPipeline = nil;
        other.blurHorizPipeline = nil;
        other.blurVertPipeline = nil;
        other.blurVertAndDoGPipeline = nil;
        other.extremaPipeline = nil;
        other.resizePipeline = nil;
        other.imageBuffer = nil;
        other.imageTexture = nil;
        other.initialized = false;
    }
    return *this;
}

bool SIFT::Impl::isAvailable() const {
    return initialized && device && commandQueue;
}

int SIFT::Impl::descriptorSize() const {
    return SIFT_DESCR_WIDTH * SIFT_DESCR_WIDTH * SIFT_DESCR_HIST_BINS;
}

int SIFT::Impl::descriptorType() const {
    return config.descriptorType;
}

int SIFT::Impl::defaultNorm() const {
    return cv::NORM_L2;
}

void SIFT::Impl::detectAndCompute(cv::InputArray _image, cv::InputArray _mask,
                                   std::vector<cv::KeyPoint>& keypoints,
                                   cv::OutputArray _descriptors,
                                   bool useProvidedKeypoints)
{
    cv::Mat image = _image.getMat();

    if (!isAvailable()) {
        std::cerr << "Metal SIFT implementation not available" << std::endl;
        return;
    }

    @autoreleasepool {
        // Verify image dimensions match config
        if (image.cols != config.imageSize.width || image.rows != config.imageSize.height) {
            std::cerr << "Error: Input image dimensions (" << image.cols << "x" << image.rows
                        << ") don't match SIFTConfig dimensions (" << config.imageSize.width
                        << "x" << config.imageSize.height << ")" << std::endl;
            return;
        }

        cv::Mat base;
        image.convertTo(base, CV_32F, SIFT_FIXPT_SCALE, 0);

        // Zero out pre-allocated extrema bitarray buffers
        for (size_t i = 0; i < extremaBitarrays.size(); i++) {
            size_t bufferSize = extremaBitarraySizes[i] * sizeof(uint32_t);
            memset(extremaBitarrays[i].contents, 0, bufferSize);
        }

        keypoints.clear();

        // Upload octave 0 layer 0 to GPU buffer
        {
            int octaveWidth = base.cols;
            int octaveHeight = base.rows;
            size_t rowBytes = octaveWidth * sizeof(float);
            size_t alignedRowBytes = ((rowBytes + METAL_BUFFER_ALIGNMENT - 1) / METAL_BUFFER_ALIGNMENT) * METAL_BUFFER_ALIGNMENT;

            float* imagePtr = (float*)imageBuffer.contents;
            size_t alignedRowFloats = alignedRowBytes / sizeof(float);

            for (int row = 0; row < octaveHeight; row++) {
                memcpy(imagePtr + row * alignedRowFloats,
                    base.ptr<float>(row),
                    octaveWidth * sizeof(float));
            }
        }

        id<MTLCommandBuffer> cmdBuf = [commandQueue commandBuffer];
        cmdBuf.label = @"SIFT Feature Detection";

        // Encode all GPU work into single command buffer
        for (int o = 0; o < config.nOctaves; o++) {
            encodeBatchedOctaveConstruction(cmdBuf, o);
        }

        // Commit and wait for GPU work to complete
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        // Pass 1: Extract all keypoints
        for (int o = 0; o < config.nOctaves; o++) {
            int octaveWidth = base.cols >> o;

            for (int layer = 1; layer <= config.nOctaveLayers; layer++) {
                int layerIndex = o * config.nOctaveLayers + (layer - 1);
                uint32_t* layerExtremaBitarray = (uint32_t*)extremaBitarrays[layerIndex].contents;

                extractKeypoints(
                    layerExtremaBitarray,
                    o,
                    extremaBitarraySizes[layerIndex],
                    octaveWidth,
                    layer,
                    config.nOctaveLayers,
                    (float)config.contrastThreshold,
                    (float)config.edgeThreshold,
                    (float)config.sigma,
                    gauss_pyr,
                    dog_pyr,
                    keypoints
                );
            }
        }

        // Adjust keypoint positions for firstOctave
        for (size_t i = 0; i < keypoints.size(); i++) {
            cv::KeyPoint& kpt = keypoints[i];
            float scale = 1.f/(float)(1 << -config.firstOctave);
            kpt.octave = (kpt.octave & ~255) | ((kpt.octave + config.firstOctave) & 255);
            kpt.pt *= scale;
            kpt.size *= scale;
        }

        // Pass 2: Compute descriptors if needed
        if (_descriptors.needed()) {
            const int descriptorSize = SIFT_DESCR_WIDTH * SIFT_DESCR_WIDTH * SIFT_DESCR_HIST_BINS;
            int totalKeypoints = static_cast<int>(keypoints.size());

            if (totalKeypoints > 0) {
                cv::Mat descriptorsMat;
                descriptorsMat.create(totalKeypoints, descriptorSize, config.descriptorType);

                // Compute descriptors in octave groups for better cache locality
                int descriptorRow = 0;
                for (int o = 0; o < config.nOctaves; o++) {
                    // Find keypoints belonging to this octave
                    std::vector<cv::KeyPoint> octaveKeypoints;
                    for (const auto& kpt : keypoints) {
                        int kptOctave = kpt.octave & 255;
                        if (kptOctave == o + config.firstOctave) {
                            octaveKeypoints.push_back(kpt);
                        }
                    }

                    if (!octaveKeypoints.empty()) {
                        cv::Mat octaveDescriptors = descriptorsMat.rowRange(
                            descriptorRow,
                            descriptorRow + octaveKeypoints.size()
                        );

                        computeDescriptors(
                            octaveKeypoints,
                            o,
                            config.nOctaveLayers,
                            gauss_pyr,
                            octaveDescriptors,
                            config.descriptorType
                        );

                        descriptorRow += octaveKeypoints.size();
                    }
                }

                descriptorsMat.copyTo(_descriptors);
            } else {
                // Create empty descriptor matrix
                _descriptors.create(0, descriptorSize, config.descriptorType);
            }
        }
    }
}

// SIFT wrapper implementation for Metal backend
// These must be in the .mm file since Impl is defined here

SIFT::SIFT(const SIFTConfig& config)
    : impl_(new Impl(config))
{
}

SIFT::~SIFT() {
    delete impl_;
}

SIFT::SIFT(SIFT&& other) noexcept
    : impl_(other.impl_)
{
    other.impl_ = nullptr;
}

SIFT& SIFT::operator=(SIFT&& other) noexcept {
    if (this != &other) {
        delete impl_;
        impl_ = other.impl_;
        other.impl_ = nullptr;
    }
    return *this;
}

int SIFT::descriptorSize() const {
    return impl_->descriptorSize();
}

int SIFT::descriptorType() const {
    return impl_->descriptorType();
}

int SIFT::defaultNorm() const {
    return impl_->defaultNorm();
}

void SIFT::detectAndCompute(cv::InputArray image, cv::InputArray mask,
                            std::vector<cv::KeyPoint>& keypoints,
                            cv::OutputArray descriptors,
                            bool useProvidedKeypoints) {
    impl_->detectAndCompute(image, mask, keypoints, descriptors, useProvidedKeypoints);
}

} // namespace lar