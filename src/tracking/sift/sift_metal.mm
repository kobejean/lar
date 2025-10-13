// Metal-accelerated SIFT processor - consolidated implementation
// All Metal SIFT code in one place for easier maintenance and thread-safety
// Each instance owns its own Metal resources (RAII pattern)

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <chrono>
#include <vector>
#include <mutex>
#include <atomic>
#include <thread>
#include <stdexcept>

#include "sift_metal.h"
#include "lar/tracking/sift/sift_common.h"

#if !__has_feature(objc_arc)
    #define RELEASE_IF_MANUAL(obj) [obj release]
#else
    #define RELEASE_IF_MANUAL(obj) (void)0
#endif

#define METAL_BUFFER_ALIGNMENT 16  // Metal prefers 16-byte alignment

namespace lar {

struct ExtremaParams {
    float threshold;
    int border;
};

struct SIFTMetal::Impl {
    // Metal resources
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> commandQueue = nil;
    id<MTLComputePipelineState> blurAndDoGPipeline = nil;
    id<MTLComputePipelineState> extremaPipeline = nil;
    id<MTLComputePipelineState> resizePipeline = nil;

    // Pre-allocated GPU buffers/textures
    std::vector<std::vector<id<MTLBuffer>>> octaveBuffers;
    std::vector<std::vector<id<MTLTexture>>> octaveTextures;
    std::vector<id<MTLBuffer>> tempBuffers;
    std::vector<id<MTLTexture>> tempTextures;
    std::vector<std::vector<id<MTLBuffer>>> dogBuffers;
    std::vector<std::vector<id<MTLTexture>>> dogTextures;

    // Pre-computed Gaussian kernels
    std::vector<std::vector<float>> gaussianKernels;
    std::vector<double> sigmas;

    // Image dimensions
    int nOctaves = 0;
    int nLevels = 0;

    bool initialized = false;

    ~Impl() {
        @autoreleasepool {
            releaseBuffersAndTextures();
            RELEASE_IF_MANUAL(blurAndDoGPipeline);
            RELEASE_IF_MANUAL(extremaPipeline);
            RELEASE_IF_MANUAL(resizePipeline);
            RELEASE_IF_MANUAL(commandQueue);
            RELEASE_IF_MANUAL(device);
#if __has_feature(objc_arc)
            blurAndDoGPipeline = nil;
            extremaPipeline = nil;
            resizePipeline = nil;
            commandQueue = nil;
            device = nil;
#endif
        }
    }

    void releaseBuffersAndTextures() {
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
        octaveTextures.clear();
        octaveBuffers.clear();
        dogTextures.clear();
        dogBuffers.clear();
        tempTextures.clear();
        tempBuffers.clear();
    }
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

static void allocateOctaveResources(
    id<MTLDevice> device,
    SIFTMetal::Impl& impl,
    int octave,
    int octaveWidth,
    int octaveHeight,
    int nLevels)
{
    size_t rowBytes = octaveWidth * sizeof(float);
    size_t alignedRowBytes = ((rowBytes + METAL_BUFFER_ALIGNMENT - 1) / METAL_BUFFER_ALIGNMENT) * METAL_BUFFER_ALIGNMENT;
    size_t bufferSize = alignedRowBytes * octaveHeight;

    impl.octaveBuffers[octave].resize(nLevels);
    impl.octaveTextures[octave].resize(nLevels);
    impl.dogBuffers[octave].resize(nLevels - 1);
    impl.dogTextures[octave].resize(nLevels - 1);

    // Allocate Gaussian pyramid buffers/textures
    for (int i = 0; i < nLevels; i++) {
        impl.octaveBuffers[octave][i] = [device newBufferWithLength:bufferSize
                                                      options:MTLResourceStorageModeShared];

        MTLTextureDescriptor* desc = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float
            width:octaveWidth height:octaveHeight mipmapped:NO];
        desc.storageMode = MTLStorageModeShared;
        desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;

        impl.octaveTextures[octave][i] = [impl.octaveBuffers[octave][i]
                                               newTextureWithDescriptor:desc
                                               offset:0
                                               bytesPerRow:alignedRowBytes];
    }

    // Allocate DoG buffers/textures
    for (int i = 0; i < nLevels - 1; i++) {
        impl.dogBuffers[octave][i] = [device newBufferWithLength:bufferSize
                                                  options:MTLResourceStorageModeShared];

        MTLTextureDescriptor* dogDesc = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float
            width:octaveWidth height:octaveHeight mipmapped:NO];
        dogDesc.storageMode = MTLStorageModeShared;
        dogDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;

        impl.dogTextures[octave][i] = [impl.dogBuffers[octave][i]
                                            newTextureWithDescriptor:dogDesc
                                            offset:0
                                            bytesPerRow:alignedRowBytes];
    }

    // Allocate temporary texture for separable convolution
    impl.tempBuffers[octave] = [device newBufferWithLength:bufferSize
                                        options:MTLResourceStorageModeShared];
    MTLTextureDescriptor* tempDesc = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float
        width:octaveWidth height:octaveHeight mipmapped:NO];
    tempDesc.storageMode = MTLStorageModeShared;
    tempDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    impl.tempTextures[octave] = [impl.tempBuffers[octave] newTextureWithDescriptor:tempDesc
                                                                                     offset:0
                                                                                bytesPerRow:alignedRowBytes];
}

static void uploadOctaveBase(
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

static void extractKeypointsAndDescriptors(
    uint32_t* bitarray,
    int octave,
    int nLevels,
    int octaveBitarraySize,
    int octaveWidth,
    int layer,
    int nOctaveLayers,
    float contrastThreshold,
    float edgeThreshold,
    float sigma,
    const std::vector<cv::Mat>& gauss_pyr,
    const std::vector<cv::Mat>& dog_pyr,
    std::vector<cv::KeyPoint>& keypoints,
    cv::Mat& descriptors,
    int descriptorType)
{
    int count = 0;
    const int descriptorSize = SIFT_DESCR_WIDTH * SIFT_DESCR_WIDTH * SIFT_DESCR_HIST_BINS;
    int maxKeypoints = octaveWidth * octaveWidth;  // Upper bound estimate
    cv::Mat localDescriptors(maxKeypoints, descriptorSize, descriptorType);

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

                // Find orientation peaks and create keypoints + descriptors
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

                        // Compute descriptor for this keypoint immediately
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
                                         localDescriptors, count);

                        count++;
                    }
                }
            }
        }
    }

    // Resize local descriptors to actual count and assign to output
    if (count > 0) {
        descriptors = localDescriptors.rowRange(0, count).clone();
    }
}

SIFTMetal::SIFTMetal(const SIFTConfig& config) : impl_(new Impl()), config_(config)
{
    @autoreleasepool {
        // Initialize Metal device
        impl_->device = MTLCreateSystemDefaultDevice();
        if (!impl_->device) {
            throw std::runtime_error("Metal device not available on this system");
        }

        // Create command queue
        impl_->commandQueue = [impl_->device newCommandQueue];
        if (!impl_->commandQueue) {
            throw std::runtime_error("Failed to create Metal command queue");
        }

        // Load Metal library
        id<MTLLibrary> library = loadMetalLibrary(impl_->device, @"sift");
        if (!library) {
            throw std::runtime_error("Failed to load Metal shader library");
        }

        NSError* error = nil;

        // Create blur+DoG pipeline
        id<MTLFunction> blurFunc = [library newFunctionWithName:@"gaussianBlurAndDoGFused"];
        if (!blurFunc) {
            throw std::runtime_error("Failed to find Metal function: gaussianBlurAndDoGFused");
        }
        impl_->blurAndDoGPipeline = [impl_->device newComputePipelineStateWithFunction:blurFunc error:&error];
        RELEASE_IF_MANUAL(blurFunc);
        if (!impl_->blurAndDoGPipeline) {
            std::string errorMsg = "Failed to create blur pipeline: ";
            if (error) errorMsg += [[error localizedDescription] UTF8String];
            throw std::runtime_error(errorMsg);
        }

        // Create extrema detection pipeline
        id<MTLFunction> extremaFunc = [library newFunctionWithName:@"detectExtrema"];
        if (!extremaFunc) {
            throw std::runtime_error("Failed to find Metal function: detectExtrema");
        }
        impl_->extremaPipeline = [impl_->device newComputePipelineStateWithFunction:extremaFunc error:&error];
        RELEASE_IF_MANUAL(extremaFunc);
        if (!impl_->extremaPipeline) {
            std::string errorMsg = "Failed to create extrema pipeline: ";
            if (error) errorMsg += [[error localizedDescription] UTF8String];
            throw std::runtime_error(errorMsg);
        }

        // Create resize pipeline
        id<MTLFunction> resizeFunc = [library newFunctionWithName:@"resizeNearestNeighbor2x"];
        if (!resizeFunc) {
            throw std::runtime_error("Failed to find Metal function: resizeNearestNeighbor2x");
        }
        impl_->resizePipeline = [impl_->device newComputePipelineStateWithFunction:resizeFunc error:&error];
        RELEASE_IF_MANUAL(resizeFunc);
        if (!impl_->resizePipeline) {
            std::string errorMsg = "Failed to create resize pipeline: ";
            if (error) errorMsg += [[error localizedDescription] UTF8String];
            throw std::runtime_error(errorMsg);
        }

        // Pre-compute dimensions and allocate resources if image size is specified
        if (config_.imageSize.width > 0 && config_.imageSize.height > 0) {
            impl_->nLevels = config_.pyramidLevels();
            impl_->nOctaves = config_.computeNumOctaves(config_.imageSize.width, config_.imageSize.height);

            // Pre-compute Gaussian kernels once
            computeGaussianKernels(impl_->nLevels, config_.nOctaveLayers, config_.sigma,
                                 impl_->gaussianKernels, impl_->sigmas);

            // Pre-allocate all octave buffers and textures
            impl_->octaveBuffers.resize(impl_->nOctaves);
            impl_->octaveTextures.resize(impl_->nOctaves);
            impl_->dogBuffers.resize(impl_->nOctaves);
            impl_->dogTextures.resize(impl_->nOctaves);
            impl_->tempBuffers.resize(impl_->nOctaves);
            impl_->tempTextures.resize(impl_->nOctaves);

            for (int o = 0; o < impl_->nOctaves; o++) {
                int octaveWidth = config_.imageSize.width >> o;
                int octaveHeight = config_.imageSize.height >> o;
                allocateOctaveResources(impl_->device, *impl_, o, octaveWidth, octaveHeight, impl_->nLevels);
            }
        }

        impl_->initialized = true;
    }
}

SIFTMetal::~SIFTMetal() {
    delete impl_;
}

SIFTMetal::SIFTMetal(SIFTMetal&& other) noexcept : impl_(other.impl_), config_(other.config_)
{
    other.impl_ = nullptr;
}

SIFTMetal& SIFTMetal::operator=(SIFTMetal&& other) noexcept {
    if (this != &other) {
        delete impl_;
        impl_ = other.impl_;
        config_ = other.config_;
        other.impl_ = nullptr;
    }
    return *this;
}

bool SIFTMetal::isAvailable() const {
    return impl_ && impl_->initialized && impl_->device && impl_->commandQueue;
}

bool SIFTMetal::detectAndCompute(const cv::Mat& base,
                                 std::vector<cv::KeyPoint>& keypoints,
                                 cv::OutputArray descriptors,
                                 int nOctaves)
{
    if (!isAvailable()) {
        return false;
    }

    @autoreleasepool {
        // Verify image dimensions match config (if pre-allocated)
        if (impl_->nOctaves > 0) {
            if (base.cols != config_.imageSize.width || base.rows != config_.imageSize.height) {
                std::cerr << "Error: Input image dimensions (" << base.cols << "x" << base.rows
                         << ") don't match SIFTConfig dimensions (" << config_.imageSize.width
                         << "x" << config_.imageSize.height << ")" << std::endl;
                return false;
            }
            // Use pre-computed values from constructor
            nOctaves = impl_->nOctaves;
        }

        // Extract mutable Mat from OutputArray for internal processing
        cv::Mat descriptorsMat;
        int nLevels = impl_->nLevels > 0 ? impl_->nLevels : (config_.nOctaveLayers + 3);
        float threshold = 0.5f * config_.contrastThreshold / config_.nOctaveLayers * 255 * SIFT_FIXPT_SCALE;

        // Use pre-computed Gaussian kernels if available, otherwise compute them
        const std::vector<std::vector<float>>& gaussianKernels =
            !impl_->gaussianKernels.empty() ? impl_->gaussianKernels :
            [this, nLevels]() -> const std::vector<std::vector<float>>& {
                static std::vector<std::vector<float>> tempKernels;
                static std::vector<double> tempSigmas;
                computeGaussianKernels(nLevels, config_.nOctaveLayers, config_.sigma, tempKernels, tempSigmas);
                return tempKernels;
            }();

        // Pre-allocate pyramid storage (internal to SIFTMetal)
        std::vector<cv::Mat> gauss_pyr(nOctaves * nLevels);
        std::vector<cv::Mat> dog_pyr(nOctaves * (config_.nOctaveLayers + 2));

        // Allocate extrema bitarray buffers for each layer
        std::vector<id<MTLBuffer>> layerExtremaBitarrays;
        std::vector<uint32_t> layerExtremaBitarraySizes;
        layerExtremaBitarrays.reserve(nOctaves * config_.nOctaveLayers);
        layerExtremaBitarraySizes.reserve(nOctaves * config_.nOctaveLayers);

        for (int o = 0; o < nOctaves; o++) {
            int octaveWidth = base.cols >> o;
            int octaveHeight = base.rows >> o;
            uint32_t octavePixels = octaveWidth * octaveHeight;
            uint32_t extremaBitarraySize = ((octavePixels + 31) / 32) * sizeof(uint32_t);

            for (int layer = 1; layer <= config_.nOctaveLayers; layer++) {
                id<MTLBuffer> buffer = [impl_->device newBufferWithLength:extremaBitarraySize
                                                           options:MTLResourceStorageModeShared];
                memset(buffer.contents, 0, extremaBitarraySize);
                layerExtremaBitarrays.push_back(buffer);
                layerExtremaBitarraySizes.push_back(extremaBitarraySize / sizeof(uint32_t));
            }
        }

        keypoints.clear();

        // Producer-consumer synchronization
        auto keypointsMutex = std::make_shared<std::mutex>();
        int totalLayers = nOctaves * config_.nOctaveLayers;
        auto allKeypoints = std::make_shared<std::vector<std::vector<cv::KeyPoint>>>(totalLayers);
        auto allDescriptors = std::make_shared<std::vector<cv::Mat>>(totalLayers);
        auto completedLayers = std::make_shared<std::atomic<int>>(0);

        std::vector<id<MTLCommandBuffer>> allCommandBuffers;
        allCommandBuffers.reserve(totalLayers);

        // Upload octave 0 base image
        {
            int octaveWidth = base.cols;
            int octaveHeight = base.rows;
            size_t rowBytes = octaveWidth * sizeof(float);
            size_t alignedRowBytes = ((rowBytes + METAL_BUFFER_ALIGNMENT - 1) / METAL_BUFFER_ALIGNMENT) * METAL_BUFFER_ALIGNMENT;

            std::vector<id<MTLBuffer>>& gaussBuffers = impl_->octaveBuffers[0];
            cv::Mat octaveBase = base.clone();
            uploadOctaveBase(octaveBase, gaussBuffers[0], octaveWidth, octaveHeight, alignedRowBytes);
        }

        // Submit per-layer command buffers for all octaves
        for (int o = 0; o < nOctaves; o++) {
            int octaveWidth = base.cols >> o;
            int octaveHeight = base.rows >> o;

            size_t rowBytes = octaveWidth * sizeof(float);
            size_t alignedRowBytes = ((rowBytes + METAL_BUFFER_ALIGNMENT - 1) / METAL_BUFFER_ALIGNMENT) * METAL_BUFFER_ALIGNMENT;
            int rowStride = (int)(alignedRowBytes / sizeof(float));

            int gaussIdx = o * nLevels;
            int dogIdx = o * (nLevels - 1);

            std::vector<id<MTLBuffer>>& gaussBuffers = impl_->octaveBuffers[o];
            std::vector<id<MTLBuffer>>& dogBuffers = impl_->dogBuffers[o];

            // Create cv::Mat wrappers for GPU buffers
            for (int i = 0; i < nLevels; i++) {
                gauss_pyr[gaussIdx + i] = cv::Mat(octaveHeight, octaveWidth, CV_32F, gaussBuffers[i].contents, alignedRowBytes);
            }

            for (int i = 0; i < nLevels - 1; i++) {
                dog_pyr[dogIdx + i] = cv::Mat(octaveHeight, octaveWidth, CV_32F, dogBuffers[i].contents, alignedRowBytes);
            }

            // Submit per-layer command buffers
            for (int layer = 1; layer <= config_.nOctaveLayers; layer++) {
                id<MTLCommandBuffer> layerCmdBuf = [impl_->commandQueue commandBuffer];
                layerCmdBuf.label = [NSString stringWithFormat:@"Octave %d Layer %d", o, layer];

                // For octave 1+ layer 1: encode GPU resize)
                if (layer == 1 && o > 0) {
                    std::vector<id<MTLTexture>>& prevGaussTextures = impl_->octaveTextures[o - 1];
                    std::vector<id<MTLTexture>>& gaussTextures = impl_->octaveTextures[o];

                    id<MTLComputeCommandEncoder> resizeEncoder = [layerCmdBuf computeCommandEncoder];
                    [resizeEncoder setComputePipelineState:impl_->resizePipeline];
                    [resizeEncoder setTexture:prevGaussTextures[nLevels-3] atIndex:0];  // Source texture
                    [resizeEncoder setTexture:gaussTextures[0] atIndex:1];              // Destination texture

                    MTLSize resizeGridSize = MTLSizeMake(octaveWidth, octaveHeight, 1);
                    MTLSize resizeThreadgroupSize = MTLSizeMake(16, 16, 1);

                    [resizeEncoder dispatchThreads:resizeGridSize threadsPerThreadgroup:resizeThreadgroupSize];
                    [resizeEncoder endEncoding];
                }

                // Encode blur + DoG operations
                int blurStart = (layer == 1) ? 1 : (layer + 2);
                int blurEnd = (layer == 1) ? 4 : (layer + 3);

                for (int i = blurStart; i < blurEnd && i < nLevels; i++) {
                    int kernelSize = (int)gaussianKernels[i].size();

                    id<MTLBuffer> kernelSizeBuffer = [impl_->device newBufferWithBytes:&kernelSize
                                                        length:sizeof(int)
                                                        options:MTLResourceStorageModeShared];

                    id<MTLBuffer> kernelBuffer = [impl_->device newBufferWithBytes:gaussianKernels[i].data()
                                                        length:gaussianKernels[i].size() * sizeof(float)
                                                        options:MTLResourceStorageModeShared];

                    // Get texture views for this octave
                    std::vector<id<MTLTexture>>& gaussTextures = impl_->octaveTextures[o];
                    std::vector<id<MTLTexture>>& dogTextures = impl_->dogTextures[o];

                    id<MTLComputeCommandEncoder> encoder = [layerCmdBuf computeCommandEncoder];

                    [encoder setComputePipelineState:impl_->blurAndDoGPipeline];
                    [encoder setTexture:gaussTextures[i-1] atIndex:0];  // Previous Gaussian texture
                    [encoder setTexture:gaussTextures[i] atIndex:1];    // Output Gaussian texture
                    [encoder setTexture:dogTextures[i-1] atIndex:2];    // Output DoG texture
                    [encoder setBuffer:kernelSizeBuffer offset:0 atIndex:0]; // Kernel size (int)
                    [encoder setBuffer:kernelBuffer offset:0 atIndex:1]; // Kernel weights

                    MTLSize gridSize = MTLSizeMake(octaveWidth, octaveHeight, 1);
                    MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);

                    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
                    [encoder endEncoding];

                    RELEASE_IF_MANUAL(kernelSizeBuffer);
                    RELEASE_IF_MANUAL(kernelBuffer);
                }

                // Encode extrema detection
                ExtremaParams params;
                params.threshold = threshold;
                params.border = SIFT_IMG_BORDER;

                id<MTLBuffer> paramsBuffer = [impl_->device newBufferWithBytes:&params
                                                                    length:sizeof(ExtremaParams)
                                                                    options:MTLResourceStorageModeShared];

                // Get DoG texture views for this octave
                std::vector<id<MTLTexture>>& dogTextures = impl_->dogTextures[o];

                id<MTLComputeCommandEncoder> encoder = [layerCmdBuf computeCommandEncoder];

                [encoder setComputePipelineState:impl_->extremaPipeline];
                [encoder setTexture:dogTextures[layer-1] atIndex:0];  // DoG layer below
                [encoder setTexture:dogTextures[layer] atIndex:1];    // DoG center layer
                [encoder setTexture:dogTextures[layer+1] atIndex:2];  // DoG layer above

                int layerIndex = o * config_.nOctaveLayers + (layer - 1);
                [encoder setBuffer:layerExtremaBitarrays[layerIndex] offset:0 atIndex:0];  // Output bitarray
                [encoder setBuffer:paramsBuffer offset:0 atIndex:1];  // Parameters

                MTLSize gridSize = MTLSizeMake(octaveWidth, octaveHeight, 1);
                MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);

                [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
                [encoder endEncoding];

                RELEASE_IF_MANUAL(paramsBuffer);

                // Add completion handler for async keypoint+descriptor extraction
                [layerCmdBuf addCompletedHandler:^(id<MTLCommandBuffer> commandBuffer) {
                    uint32_t* layerExtremaBitarray = (uint32_t*)layerExtremaBitarrays[layerIndex].contents;

                    std::vector<cv::KeyPoint> localKeypoints;
                    cv::Mat localDescriptors;

                    extractKeypointsAndDescriptors(
                        layerExtremaBitarray,
                        o,
                        nLevels,
                        layerExtremaBitarraySizes[layerIndex],
                        octaveWidth,
                        layer,
                        config_.nOctaveLayers,
                        (float)config_.contrastThreshold,
                        (float)config_.edgeThreshold,
                        (float)config_.sigma,
                        gauss_pyr,
                        dog_pyr,
                        localKeypoints,
                        localDescriptors,
                        config_.descriptorType
                    );

                    {
                        std::lock_guard<std::mutex> lock(*keypointsMutex);
                        (*allKeypoints)[layerIndex] = localKeypoints;
                        (*allDescriptors)[layerIndex] = localDescriptors;
                    }

                    completedLayers->fetch_add(1, std::memory_order_release);
                }];

                [layerCmdBuf commit];
                allCommandBuffers.push_back(layerCmdBuf);
            }
        }

        // Wait for all command buffers to complete
        for (auto& cmdBuf : allCommandBuffers) {
            [cmdBuf waitUntilCompleted];
        }

        // Merge keypoints in deterministic layer order
        for (size_t i = 0; i < allKeypoints->size(); i++) {
            const auto& layerKeypoints = (*allKeypoints)[i];
            if (!layerKeypoints.empty()) {
                keypoints.insert(keypoints.end(), layerKeypoints.begin(), layerKeypoints.end());
            }
        }

        // Merge descriptors in same layer order
        std::vector<cv::Mat> nonEmptyDescriptors;
        nonEmptyDescriptors.reserve(allDescriptors->size());
        for (const auto& mat : *allDescriptors) {
            if (!mat.empty()) {
                nonEmptyDescriptors.push_back(mat);
            }
        }

        if (!nonEmptyDescriptors.empty()) {
            cv::vconcat(nonEmptyDescriptors, descriptorsMat);
        }

        // Assign result to OutputArray
        if (!descriptorsMat.empty()) {
            descriptorsMat.copyTo(descriptors);
        } else if (descriptors.needed()) {
            // Create empty descriptor matrix using shared constants
            descriptors.create(0, SIFT_DESCR_WIDTH * SIFT_DESCR_WIDTH * SIFT_DESCR_HIST_BINS, config_.descriptorType);
        }

        // Cleanup extrema bitarray buffers
        for (auto& buffer : layerExtremaBitarrays) {
            RELEASE_IF_MANUAL(buffer);
        }

        return true;
    }
}

} // namespace lar