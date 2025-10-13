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
#include <stdexcept>

#include "lar/tracking/sift/metal_sift.h"
#include "lar/tracking/sift/sift_common.h"
#include "lar/tracking/sift/sift_constants.h"

// Helper macro for conditional release
#if !__has_feature(objc_arc)
    #define RELEASE_IF_MANUAL(obj) [obj release]
#else
    #define RELEASE_IF_MANUAL(obj) (void)0
#endif

#define METAL_BUFFER_ALIGNMENT 16  // Metal prefers 16-byte alignment

namespace lar {

// ============================================================================
// Metal Shader Parameter Structures
// ============================================================================

struct GaussianBlurParams {
    int width;
    int height;
    int rowStride;
    int kernelSize;
};

struct KeypointCandidate {
    int x;
    int y;
    int octave;
    int layer;
    float value;
};

struct ExtremaParams {
    int width;
    int height;
    int rowStride;
    float threshold;
    int border;
    int octave;
    int layer;
};

struct ResizeParams {
    int srcWidth;
    int srcHeight;
    int srcRowStride;
    int dstWidth;
    int dstHeight;
    int dstRowStride;
};

// ============================================================================
// MetalSIFT Implementation (pImpl idiom)
// ============================================================================

struct MetalSIFT::Impl {
    // Metal resources (owned by this instance, RAII)
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> commandQueue = nil;
    id<MTLComputePipelineState> blurAndDoGPipeline = nil;
    id<MTLComputePipelineState> extremaPipeline = nil;
    id<MTLComputePipelineState> resizePipeline = nil;

    // Cached GPU buffers/textures (reused across frames if dimensions match)
    std::vector<std::vector<id<MTLBuffer>>> octaveBuffers;
    std::vector<std::vector<id<MTLTexture>>> octaveTextures;
    std::vector<id<MTLBuffer>> tempBuffers;
    std::vector<id<MTLTexture>> tempTextures;
    std::vector<std::vector<id<MTLBuffer>>> dogBuffers;
    std::vector<std::vector<id<MTLTexture>>> dogTextures;

    // Cached dimensions for buffer reuse
    int cachedBaseWidth = 0;
    int cachedBaseHeight = 0;
    int cachedNOctaves = 0;
    int cachedNLevels = 0;

    bool initialized = false;

    // Destructor - releases all Metal resources
    ~Impl() {
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
};

// ============================================================================
// Helper Functions
// ============================================================================

// Load Metal shader library with comprehensive fallback locations
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

// Create 1D Gaussian kernel using OpenCV's bit-exact implementation
static std::vector<float> createGaussianKernel(double sigma) {
    int ksize = cvRound(sigma * 8 + 1) | 1;
    cv::Mat kernelMat = cv::getGaussianKernel(ksize, sigma, CV_32F);

    std::vector<float> kernel(ksize);
    for (int i = 0; i < ksize; i++) {
        kernel[i] = kernelMat.at<float>(i);
    }

    return kernel;
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
    MetalSIFT::Impl& impl,
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

// Upload octave base image to GPU
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

// Extract keypoints from bitarray and compute descriptors incrementally
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

    // Pre-allocate local descriptor matrix
    const int SIFT_DESCR_WIDTH = 4;
    const int SIFT_DESCR_HIST_BINS = 8;
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

                if (!sift_common::adjustLocalExtrema(dog_pyr, kpt, octave, keypoint_layer, r, c_pos,
                                    nOctaveLayers, contrastThreshold, edgeThreshold, sigma)) {
                    continue;
                }

                // Calculate orientation histogram
                static const int n = SIFT_ORI_HIST_BINS;
                float hist[n];
                float scl_octv = kpt.size * 0.5f / (1 << octave);

                int gaussIdx = octave * (nOctaveLayers + 3) + keypoint_layer;
                float omax = sift_common::calcOrientationHist(gauss_pyr[gaussIdx],
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

                        sift_common::calcSIFTDescriptor(img, ptf, angle,
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

// ============================================================================
// MetalSIFT Public Interface Implementation
// ============================================================================

MetalSIFT::MetalSIFT(int nOctaveLayers,
                     double contrastThreshold,
                     double edgeThreshold,
                     double sigma,
                     int descriptorType)
    : impl_(new Impl())
    , nOctaveLayers_(nOctaveLayers)
    , contrastThreshold_(contrastThreshold)
    , edgeThreshold_(edgeThreshold)
    , sigma_(sigma)
    , descriptorType_(descriptorType)
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

        impl_->initialized = true;
    }
}

MetalSIFT::~MetalSIFT() {
    delete impl_;
}

MetalSIFT::MetalSIFT(MetalSIFT&& other) noexcept
    : impl_(other.impl_)
    , nOctaveLayers_(other.nOctaveLayers_)
    , contrastThreshold_(other.contrastThreshold_)
    , edgeThreshold_(other.edgeThreshold_)
    , sigma_(other.sigma_)
    , descriptorType_(other.descriptorType_)
{
    other.impl_ = nullptr;
}

MetalSIFT& MetalSIFT::operator=(MetalSIFT&& other) noexcept {
    if (this != &other) {
        delete impl_;
        impl_ = other.impl_;
        nOctaveLayers_ = other.nOctaveLayers_;
        contrastThreshold_ = other.contrastThreshold_;
        edgeThreshold_ = other.edgeThreshold_;
        sigma_ = other.sigma_;
        descriptorType_ = other.descriptorType_;
        other.impl_ = nullptr;
    }
    return *this;
}

bool MetalSIFT::isAvailable() const {
    return impl_ && impl_->initialized && impl_->device && impl_->commandQueue;
}

bool MetalSIFT::detectAndCompute(const cv::Mat& base,
                                 std::vector<cv::Mat>& gauss_pyr,
                                 std::vector<cv::KeyPoint>& keypoints,
                                 cv::Mat& descriptors,
                                 int nOctaves)
{
    if (!isAvailable()) {
        return false;
    }

    @autoreleasepool {
        int nLevels = nOctaveLayers_ + 3;
        float threshold = 0.5f * contrastThreshold_ / nOctaveLayers_ * 255 * SIFT_FIXPT_SCALE;

        // Reallocate buffers only if dimensions changed
        if (impl_->needsReallocation(base.cols, base.rows, nOctaves, nLevels)) {
            impl_->releaseBuffersAndTextures();
            impl_->octaveBuffers.resize(nOctaves);
            impl_->octaveTextures.resize(nOctaves);
            impl_->dogBuffers.resize(nOctaves);
            impl_->dogTextures.resize(nOctaves);
            impl_->tempBuffers.resize(nOctaves);
            impl_->tempTextures.resize(nOctaves);

            for (int o = 0; o < nOctaves; o++) {
                int octaveWidth = base.cols >> o;
                int octaveHeight = base.rows >> o;
                allocateOctaveResources(impl_->device, *impl_, o, octaveWidth, octaveHeight, nLevels);
            }

            impl_->cachedBaseWidth = base.cols;
            impl_->cachedBaseHeight = base.rows;
            impl_->cachedNOctaves = nOctaves;
            impl_->cachedNLevels = nLevels;
        }

        // Pre-compute Gaussian kernels
        std::vector<std::vector<float>> gaussianKernels;
        std::vector<double> sigmas;
        computeGaussianKernels(nLevels, nOctaveLayers_, sigma_, gaussianKernels, sigmas);

        // Pre-allocate pyramid storage
        std::vector<cv::Mat> dog_pyr(nOctaves * (nOctaveLayers_ + 2));
        gauss_pyr.resize(nOctaves * nLevels);

        // Allocate extrema bitarray buffers for each layer
        std::vector<id<MTLBuffer>> layerExtremaBitarrays;
        std::vector<uint32_t> layerExtremaBitarraySizes;
        layerExtremaBitarrays.reserve(nOctaves * nOctaveLayers_);
        layerExtremaBitarraySizes.reserve(nOctaves * nOctaveLayers_);

        for (int o = 0; o < nOctaves; o++) {
            int octaveWidth = base.cols >> o;
            int octaveHeight = base.rows >> o;
            uint32_t octavePixels = octaveWidth * octaveHeight;
            uint32_t extremaBitarraySize = ((octavePixels + 31) / 32) * sizeof(uint32_t);

            for (int layer = 1; layer <= nOctaveLayers_; layer++) {
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
        auto descriptorsMutex = std::make_shared<std::mutex>();
        int totalLayers = nOctaves * nOctaveLayers_;
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
            for (int layer = 1; layer <= nOctaveLayers_; layer++) {
                id<MTLCommandBuffer> layerCmdBuf = [impl_->commandQueue commandBuffer];
                layerCmdBuf.label = [NSString stringWithFormat:@"Octave %d Layer %d", o, layer];

                // For octave 1+ layer 1: encode GPU resize
                if (layer == 1 && o > 0) {
                    int prevOctaveWidth = base.cols >> (o - 1);
                    int prevOctaveHeight = base.rows >> (o - 1);
                    size_t prevRowBytes = prevOctaveWidth * sizeof(float);
                    size_t prevAlignedRowBytes = ((prevRowBytes + METAL_BUFFER_ALIGNMENT - 1) / METAL_BUFFER_ALIGNMENT) * METAL_BUFFER_ALIGNMENT;
                    int prevRowStride = (int)(prevAlignedRowBytes / sizeof(float));

                    std::vector<id<MTLBuffer>>& prevGaussBuffers = impl_->octaveBuffers[o - 1];

                    ResizeParams resizeParams;
                    resizeParams.srcWidth = prevOctaveWidth;
                    resizeParams.srcHeight = prevOctaveHeight;
                    resizeParams.srcRowStride = prevRowStride;
                    resizeParams.dstWidth = octaveWidth;
                    resizeParams.dstHeight = octaveHeight;
                    resizeParams.dstRowStride = rowStride;

                    id<MTLBuffer> resizeParamsBuffer = [impl_->device newBufferWithBytes:&resizeParams
                                                        length:sizeof(ResizeParams)
                                                        options:MTLResourceStorageModeShared];

                    id<MTLComputeCommandEncoder> resizeEncoder = [layerCmdBuf computeCommandEncoder];
                    [resizeEncoder setComputePipelineState:impl_->resizePipeline];
                    [resizeEncoder setBuffer:prevGaussBuffers[nLevels-3] offset:0 atIndex:0];
                    [resizeEncoder setBuffer:gaussBuffers[0] offset:0 atIndex:1];
                    [resizeEncoder setBuffer:resizeParamsBuffer offset:0 atIndex:2];

                    MTLSize resizeGridSize = MTLSizeMake(octaveWidth, octaveHeight, 1);
                    MTLSize resizeThreadgroupSize = MTLSizeMake(16, 16, 1);

                    [resizeEncoder dispatchThreads:resizeGridSize threadsPerThreadgroup:resizeThreadgroupSize];
                    [resizeEncoder endEncoding];

                    RELEASE_IF_MANUAL(resizeParamsBuffer);
                }

                // Encode blur + DoG operations
                int blurStart = (layer == 1) ? 1 : (layer + 2);
                int blurEnd = (layer == 1) ? 4 : (layer + 3);

                for (int i = blurStart; i < blurEnd && i < nLevels; i++) {
                    GaussianBlurParams params;
                    params.width = octaveWidth;
                    params.height = octaveHeight;
                    params.rowStride = rowStride;
                    params.kernelSize = (int)gaussianKernels[i].size();

                    id<MTLBuffer> paramsBuffer = [impl_->device newBufferWithBytes:&params
                                                        length:sizeof(GaussianBlurParams)
                                                        options:MTLResourceStorageModeShared];

                    id<MTLBuffer> kernelBuffer = [impl_->device newBufferWithBytes:gaussianKernels[i].data()
                                                        length:gaussianKernels[i].size() * sizeof(float)
                                                        options:MTLResourceStorageModeShared];

                    id<MTLComputeCommandEncoder> encoder = [layerCmdBuf computeCommandEncoder];

                    [encoder setComputePipelineState:impl_->blurAndDoGPipeline];
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

                // Encode extrema detection
                ExtremaParams params;
                params.width = octaveWidth;
                params.height = octaveHeight;
                params.rowStride = rowStride;
                params.threshold = threshold;
                params.border = SIFT_IMG_BORDER;
                params.octave = o;
                params.layer = layer;

                id<MTLBuffer> paramsBuffer = [impl_->device newBufferWithBytes:&params
                                                                    length:sizeof(ExtremaParams)
                                                                    options:MTLResourceStorageModeShared];

                id<MTLComputeCommandEncoder> encoder = [layerCmdBuf computeCommandEncoder];

                [encoder setComputePipelineState:impl_->extremaPipeline];
                [encoder setBuffer:dogBuffers[layer-1] offset:0 atIndex:0];
                [encoder setBuffer:dogBuffers[layer] offset:0 atIndex:1];
                [encoder setBuffer:dogBuffers[layer+1] offset:0 atIndex:2];

                int layerIndex = o * nOctaveLayers_ + (layer - 1);
                [encoder setBuffer:layerExtremaBitarrays[layerIndex] offset:0 atIndex:3];
                [encoder setBuffer:paramsBuffer offset:0 atIndex:5];

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
                        nOctaveLayers_,
                        (float)contrastThreshold_,
                        (float)edgeThreshold_,
                        (float)sigma_,
                        gauss_pyr,
                        dog_pyr,
                        localKeypoints,
                        localDescriptors,
                        descriptorType_
                    );

                    {
                        std::lock_guard<std::mutex> lock(*keypointsMutex);
                        (*allKeypoints)[layerIndex] = localKeypoints;
                    }

                    {
                        std::lock_guard<std::mutex> lock(*descriptorsMutex);
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
            cv::vconcat(nonEmptyDescriptors, descriptors);
        }

        // Cleanup extrema bitarray buffers
        for (auto& buffer : layerExtremaBitarrays) {
            RELEASE_IF_MANUAL(buffer);
        }

        return true;
    }
}

} // namespace lar