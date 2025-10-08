// Example: Metal-accelerated Gaussian pyramid for SIFT
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "lar/tracking/sift.h"
#include <opencv2/core.hpp>

namespace lar {

class MetalGaussianPyramid {
private:
    id<MTLDevice> device_;
    id<MTLCommandQueue> commandQueue_;

public:
    MetalGaussianPyramid() {
        device_ = MTLCreateSystemDefaultDevice();
        commandQueue_ = [device_ newCommandQueue];
    }

    // Build entire Gaussian pyramid on GPU with single transfer
    void buildPyramid(const cv::Mat& base, std::vector<cv::Mat>& pyr,
                     int nOctaves, const std::vector<double>& sigmas) {
        @autoreleasepool {
            int nLevels = (int)sigmas.size();
            pyr.resize(nOctaves * nLevels);

            // Create Metal texture from OpenCV Mat
            MTLTextureDescriptor* desc = [MTLTextureDescriptor
                texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float
                width:base.cols height:base.rows mipmapped:NO];
            desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;

            id<MTLTexture> baseTexture = [device_ newTextureWithDescriptor:desc];

            // Upload base image once
            [baseTexture replaceRegion:MTLRegionMake2D(0, 0, base.cols, base.rows)
                           mipmapLevel:0
                             withBytes:base.data
                           bytesPerRow:base.step];

            // Process each octave
            for (int o = 0; o < nOctaves; o++) {
                int octaveWidth = base.cols >> o;
                int octaveHeight = base.rows >> o;

                // Create textures for this octave
                std::vector<id<MTLTexture>> levelTextures(nLevels);
                for (int i = 0; i < nLevels; i++) {
                    MTLTextureDescriptor* levelDesc = [MTLTextureDescriptor
                        texture2DDescriptorWithPixelFormat:MTLPixelFormatR32Float
                        width:octaveWidth height:octaveHeight mipmapped:NO];
                    levelDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
                    levelTextures[i] = [device_ newTextureWithDescriptor:levelDesc];
                }

                // Batch all blurs for this octave in single command buffer
                id<MTLCommandBuffer> commandBuffer = [commandQueue_ commandBuffer];

                for (int i = 0; i < nLevels; i++) {
                    id<MTLTexture> srcTexture = (i == 0 && o == 0) ? baseTexture :
                                                (i == 0) ? levelTextures[nLevels-3] : // from previous octave
                                                levelTextures[i-1];

                    // Downsample if first layer of new octave
                    if (i == 0 && o > 0) {
                        MPSImageLanczosScale* lanczos = [[MPSImageLanczosScale alloc] initWithDevice:device_];
                        [lanczos encodeToCommandBuffer:commandBuffer
                                         sourceTexture:srcTexture
                                    destinationTexture:levelTextures[i]];
                    }

                    // Apply Gaussian blur
                    float sigma = sigmas[i];
                    int kernelSize = (int)ceil(sigma * 3) * 2 + 1;
                    MPSImageGaussianBlur* blur = [[MPSImageGaussianBlur alloc]
                        initWithDevice:device_ sigma:sigma];

                    [blur encodeToCommandBuffer:commandBuffer
                                  sourceTexture:(i == 0 && o > 0) ? levelTextures[i] : srcTexture
                             destinationTexture:levelTextures[i]];
                }

                // Execute all blurs for this octave
                [commandBuffer commit];
                [commandBuffer waitUntilCompleted];

                // Download results
                for (int i = 0; i < nLevels; i++) {
                    pyr[o * nLevels + i] = cv::Mat(octaveHeight, octaveWidth, CV_32F);
                    [levelTextures[i] getBytes:pyr[o * nLevels + i].data
                                   bytesPerRow:pyr[o * nLevels + i].step
                                    fromRegion:MTLRegionMake2D(0, 0, octaveWidth, octaveHeight)
                                   mipmapLevel:0];
                }
            }
        }
    }
};

} // namespace lar