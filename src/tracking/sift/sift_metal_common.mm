// Shared utilities for Metal-accelerated SIFT implementations
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "sift_metal_common.h"
#include <iostream>

namespace lar {

// Load a Metal shader library with comprehensive fallback locations
id<MTLLibrary> loadMetalLibrary(id<MTLDevice> device, NSString* libraryName) {
    @autoreleasepool {
        NSError* error = nil;
        NSURL* libraryURL = nil;
        id<MTLLibrary> library = nil;

        // Priority 1: Check SPM resource bundle (for Swift Package distribution)
        // Look for bundle in mainBundle (app) or bundleForClass (framework)
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

        // Priority 2: Try loading default library (SPM may embed shaders in module)
        if (!library) {
            @try {
                library = [device newDefaultLibrary];
                if (library) return library;
            } @catch (NSException *exception) {
                // Ignore - will try other methods
            }
        }

        // Priority 3: Try runtime bin directory (for standalone C++ builds)
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

        // Priority 5: Try app bundle Resources directory (macOS/iOS apps)
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

        // Priority 6: Try SPM build directory (for running tests)
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

        // Priority 7: Try "default" metallib name (SPM compiles all Metal files into default.metallib)
        if (!library && ![libraryName isEqualToString:@"default"]) {
            // Recursively try with "default" name
            library = loadMetalLibrary(device, @"default");
            if (library) return library;
        }

        // Failed to load from any location
        std::cerr << "Failed to load Metal library '" << [libraryName UTF8String] << "' from any location" << std::endl;
        if (error) {
            std::cerr << "Last error: " << [[error localizedDescription] UTF8String] << std::endl;
        }

        // Debug: print paths we tried
        NSString* execPath = [[NSBundle mainBundle] executablePath];
        NSString* execDir = [execPath stringByDeletingLastPathComponent];
        std::cerr << "Searched locations:" << std::endl;
        std::cerr << "  1. Main bundle: " << [[[NSBundle mainBundle] bundlePath] UTF8String] << std::endl;
        std::cerr << "  2. bin/ directory: bin/" << [libraryName UTF8String] << ".metallib" << std::endl;
        std::cerr << "  3. Executable dir: " << [execDir UTF8String] << std::endl;
        if ([[NSBundle mainBundle] resourcePath]) {
            std::cerr << "  4. App Resources: " << [[[NSBundle mainBundle] resourcePath] UTF8String] << "/LocalizeAR_MetalShaderResources.bundle" << std::endl;
        }

        return nil;
    }
    
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

} // namespace lar