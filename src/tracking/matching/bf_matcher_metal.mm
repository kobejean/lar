#include "lar/tracking/matching/bf_matcher_metal.h"
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <iostream>
#include <chrono>
#include <stdexcept>

namespace lar {

// ============================================================================
// Implementation class (pImpl idiom to hide Metal types from header)
// ============================================================================

class BFMatcherMetal::Impl {
public:
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    id<MTLLibrary> library;
    id<MTLComputePipelineState> pipelineUint8;

    // Performance tracking
    double lastGpuTimeMs;
    double lastTransferTimeMs;
    double lastTotalTimeMs;

    Impl() : lastGpuTimeMs(0), lastTransferTimeMs(0), lastTotalTimeMs(0) {
        @autoreleasepool {
            // Get default Metal device
            device = MTLCreateSystemDefaultDevice();
            if (!device) {
                throw std::runtime_error("Metal is not supported on this device");
            }

            // Create command queue
            commandQueue = [device newCommandQueue];
            if (!commandQueue) {
                throw std::runtime_error("Failed to create Metal command queue");
            }

            // Load Metal library
            NSError* error = nil;

            // Try loading from the application's bin directory first (runtime location)
            NSString* binPath = [[NSBundle mainBundle] resourcePath];
            if (!binPath) {
                binPath = [[NSFileManager defaultManager] currentDirectoryPath];
            }
            NSString* metalLibPath = [binPath stringByAppendingPathComponent:@"bf_matcher.metallib"];

            // If not found, try the build directory
            if (![[NSFileManager defaultManager] fileExistsAtPath:metalLibPath]) {
                metalLibPath = @"bf_matcher.metallib";
            }

            NSURL* metalLibURL = [NSURL fileURLWithPath:metalLibPath];
            library = [device newLibraryWithURL:metalLibURL error:&error];

            if (!library || error) {
                // Fallback: try loading from default library (works if compiled into binary)
                library = [device newDefaultLibrary];
                if (!library) {
                    throw std::runtime_error("Failed to load Metal library: bf_matcher.metallib not found");
                }
            }

            // Load compute functions
            id<MTLFunction> kernelUint8 = [library newFunctionWithName:@"bf_match_knn_uint8"];

            if (!kernelUint8) {
                throw std::runtime_error("Failed to load Metal kernel functions");
            }

            // Create compute pipeline states
            pipelineUint8 = [device newComputePipelineStateWithFunction:kernelUint8 error:&error];
            if (!pipelineUint8 || error) {
                throw std::runtime_error("Failed to create Metal pipeline for uint8 kernel");
            }

            std::cout << "Metal BFMatcher initialized successfully" << std::endl;
            std::cout << "  Device: " << [[device name] UTF8String] << std::endl;
        }
    }

    ~Impl() {
        @autoreleasepool {
            // ARC will handle cleanup
            device = nil;
            commandQueue = nil;
            library = nil;
            pipelineUint8 = nil;
        }
    }
};

// ============================================================================
// BFMatcherMetal Implementation
// ============================================================================

BFMatcherMetal::BFMatcherMetal() : impl_(new Impl()) {}

BFMatcherMetal::~BFMatcherMetal() {
    delete impl_;
}

BFMatcherMetal::BFMatcherMetal(BFMatcherMetal&& other) noexcept : impl_(other.impl_) {
    other.impl_ = nullptr;
}

BFMatcherMetal& BFMatcherMetal::operator=(BFMatcherMetal&& other) noexcept {
    if (this != &other) {
        delete impl_;
        impl_ = other.impl_;
        other.impl_ = nullptr;
    }
    return *this;
}

bool BFMatcherMetal::isReady() const {
    return impl_ && impl_->device && impl_->pipelineUint8;
}

std::string BFMatcherMetal::getPerformanceStats() const {
    char buffer[256];
    snprintf(buffer, sizeof(buffer),
        "GPU: %.2fms | Transfer: %.2fms | Total: %.2fms",
        impl_->lastGpuTimeMs,
        impl_->lastTransferTimeMs,
        impl_->lastTotalTimeMs
    );
    return std::string(buffer);
}

void BFMatcherMetal::knnMatch(
    const cv::Mat& queryDescriptors,
    const cv::Mat& trainDescriptors,
    std::vector<std::vector<cv::DMatch>>& matches,
    int k
) {
    auto start_total = std::chrono::high_resolution_clock::now();

    // Validate input
    if (queryDescriptors.empty() || trainDescriptors.empty()) {
        matches.clear();
        return;
    }

    if (queryDescriptors.cols != trainDescriptors.cols) {
        throw std::invalid_argument("Query and train descriptors must have same dimension");
    }

    if (!queryDescriptors.isContinuous() || !trainDescriptors.isContinuous()) {
        throw std::invalid_argument("Descriptors must be continuous");
    }

    if (queryDescriptors.type() != trainDescriptors.type()) {
        throw std::invalid_argument("Query and train descriptors must have same type");
    }

    // Fixed k=8 for optimization (parameter kept for API compatibility)
    constexpr int K_MATCHES = 8;
    if (k != K_MATCHES) {
        std::cerr << "Warning: BFMatcherMetal is optimized for k=8, but k=" << k << " requested. Using k=8." << std::endl;
    }

    const int numQuery = queryDescriptors.rows;
    const int numTrain = trainDescriptors.rows;
    const int descriptorDim = queryDescriptors.cols;

    // Determine pipeline based on descriptor type
    id<MTLComputePipelineState> pipeline;
    size_t elementSize;
    bool isUint8 = (queryDescriptors.type() == CV_8U || queryDescriptors.type() == CV_8UC1);

    if (isUint8) {
        pipeline = impl_->pipelineUint8;
        elementSize = sizeof(uint8_t);
    } else {
        throw std::invalid_argument("Unsupported descriptor type. Use CV_8U or CV_32F");
    }

    @autoreleasepool {
        auto start_transfer = std::chrono::high_resolution_clock::now();

        // Create Metal buffers
        const size_t querySize = numQuery * descriptorDim * elementSize;
        const size_t trainSize = numTrain * descriptorDim * elementSize;
        const size_t indicesSize = numQuery * K_MATCHES * sizeof(int);
        const size_t distancesSize = numQuery * K_MATCHES * sizeof(float);

        id<MTLBuffer> queryBuffer = [impl_->device newBufferWithBytes:queryDescriptors.data
                                                               length:querySize
                                                              options:MTLResourceStorageModeShared];

        id<MTLBuffer> trainBuffer = [impl_->device newBufferWithBytes:trainDescriptors.data
                                                               length:trainSize
                                                              options:MTLResourceStorageModeShared];

        id<MTLBuffer> indicesBuffer = [impl_->device newBufferWithLength:indicesSize
                                                                  options:MTLResourceStorageModeShared];

        id<MTLBuffer> distancesBuffer = [impl_->device newBufferWithLength:distancesSize
                                                                    options:MTLResourceStorageModeShared];

        auto end_transfer = std::chrono::high_resolution_clock::now();
        impl_->lastTransferTimeMs = std::chrono::duration<double, std::milli>(end_transfer - start_transfer).count();

        // Create command buffer and encoder
        id<MTLCommandBuffer> commandBuffer = [impl_->commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        [encoder setComputePipelineState:pipeline];

        // Set buffers
        [encoder setBuffer:queryBuffer offset:0 atIndex:0];
        [encoder setBuffer:trainBuffer offset:0 atIndex:1];
        [encoder setBuffer:indicesBuffer offset:0 atIndex:2];
        [encoder setBuffer:distancesBuffer offset:0 atIndex:3];
        [encoder setBytes:&numQuery length:sizeof(int) atIndex:4];
        [encoder setBytes:&numTrain length:sizeof(int) atIndex:5];
        [encoder setBytes:&descriptorDim length:sizeof(int) atIndex:6];

        // Configure thread execution
        NSUInteger threadGroupSize = std::min((NSUInteger)pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)256);
        MTLSize threadsPerGroup = MTLSizeMake(threadGroupSize, 1, 1);
        MTLSize numThreadgroups = MTLSizeMake((numQuery + threadGroupSize - 1) / threadGroupSize, 1, 1);

        [encoder dispatchThreadgroups:numThreadgroups threadsPerThreadgroup:threadsPerGroup];
        [encoder endEncoding];

        // Measure GPU execution time
        auto start_gpu = std::chrono::high_resolution_clock::now();
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        auto end_gpu = std::chrono::high_resolution_clock::now();

        impl_->lastGpuTimeMs = std::chrono::duration<double, std::milli>(end_gpu - start_gpu).count();

        // Read back results
        const int* indices = (const int*)indicesBuffer.contents;
        const float* distances = (const float*)distancesBuffer.contents;

        // Convert to OpenCV DMatch format
        matches.resize(numQuery);
        for (int i = 0; i < numQuery; i++) {
            matches[i].clear();
            matches[i].reserve(K_MATCHES);

            for (int j = 0; j < K_MATCHES; j++) {
                int idx = indices[i * K_MATCHES + j];
                if (idx >= 0) {  // Valid match
                    float dist = distances[i * K_MATCHES + j];
                    matches[i].push_back(cv::DMatch(i, idx, 0, dist));
                }
            }
        }
    }

    auto end_total = std::chrono::high_resolution_clock::now();
    impl_->lastTotalTimeMs = std::chrono::duration<double, std::milli>(end_total - start_total).count();
}

} // namespace lar