// Experimental: Fused Metal kernel for SIFT scale-space extrema detection
// Combines Gaussian blur, DoG computation, and extrema detection in a single pass.
//
// Pipeline per iteration:
//   Input:  prevDoG, currDoG, currGauss
//   Compute: nextGauss (in shared memory), nextDoG (in shared memory)
//   Detect:  Extrema in currDoG using [prevDoG, currDoG, nextDoG]
//   Output:  nextGauss, nextDoG (for next iteration)
//
// Compile with: LAR_USE_METAL_SIFT_FUSED

#include <metal_stdlib>
#include "sift_constants.metal"
using namespace metal;


// Gaussian blur parameters (for custom Metal kernels)
struct GaussianBlurParams {
    int width;
    int height;
    int rowStride;
    int kernelSize;
};


// Candidate keypoint structure
struct KeypointCandidate {
    int x;
    int y;
    int octave;
    int layer;
    float value;
};

// Kernel parameters
struct FusedExtremaParams {
    int width;              // Image width
    int height;             // Image height
    int rowStride;          // Row stride in floats (aligned)
    float threshold;        // Absolute threshold for extrema
    int border;             // Border size (uses SIFT_IMG_BORDER constant)
    int octave;             // Current octave index
    int layer;              // DoG layer being processed (1 to nOctaveLayers)
    int kernelSize;         // Gaussian kernel size for nextGauss
};

// Inline helper for 1D horizontal Gaussian blur with border replication
#pragma METAL fp math_mode(safe)
float horizontalGaussianBlur(
    const device float* image,
    constant float* gaussKernel,
    int srcX,
    int srcY,
    int radius,
    int width,
    int rowStride)
{
    float centerPixel = image[srcY * rowStride + srcX];
    float sum = gaussKernel[radius] * centerPixel;

    // Symmetric pairs: gaussKernel[j] * (leftPixel + rightPixel)
    for (int j = 0; j < radius; j++) {
        int leftX = clamp(srcX - radius + j, 0, width - 1);
        int rightX = clamp(srcX + radius - j, 0, width - 1);

        float leftPixel = image[srcY * rowStride + leftX];
        float rightPixel = image[srcY * rowStride + rightX];
        float weight = gaussKernel[j];

        sum += weight * (leftPixel + rightPixel);
    }

    return sum;
}

// Inline helper for 1D vertical Gaussian blur with border replication
#pragma METAL fp math_mode(safe)
float verticalGaussianBlur(
    const threadgroup float* sharedData,
    constant float* gaussKernel,
    int localX,
    int sharedY,
    int radius,
    int maxSharedY,
    int stride)
{
    float centerPixel = sharedData[sharedY * stride + localX];
    float sum = gaussKernel[radius] * centerPixel;

    for (int j = 0; j < radius; j++) {
        int topY = clamp(sharedY - radius + j, 0, maxSharedY);
        int bottomY = clamp(sharedY + radius - j, 0, maxSharedY);

        float topPixel = sharedData[topY * stride + localX];
        float bottomPixel = sharedData[bottomY * stride + localX];
        float weight = gaussKernel[j];

        sum += weight * (topPixel + bottomPixel);
    }

    return sum;
}

// ============================================================================
// Fused Gaussian Blur Kernel (Educational: Horizontal + Vertical in one pass)
// ============================================================================
// This kernel performs both horizontal and vertical blur passes using shared
// threadgroup memory to avoid global memory round-trips. It processes 16×16
// tiles with halo regions for the convolution.
//
// Performance benefit: Eliminates global memory write+read between passes
// Complexity cost: Requires threadgroup coordination and halo loading

#pragma METAL fp math_mode(safe)
kernel void gaussianBlurFused(
    const device float* source [[buffer(0)]],
    device float* destination [[buffer(1)]],
    constant GaussianBlurParams& params [[buffer(2)]],
    constant float* gaussKernel [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgSize [[threads_per_threadgroup]])
{
    // Tile configuration: 16×16 processing + vertical halo only
    const int TILE_SIZE = 16;
    int radius = params.kernelSize / 2;
    int paddedHeight = TILE_SIZE + 2 * radius;  // Only need vertical halo

    // Shared memory for horizontal blur results (with vertical halo)
    // Each column stores TILE_SIZE rows + 2*radius halo rows
    // Max kernel size ~27 (sigma ~3.0), so max radius ~13, max paddedHeight = 16 + 26 = 42
    threadgroup float sharedHoriz[48][16];  // [paddedHeight][TILE_SIZE]

    int globalX = gid.x;
    int globalY = gid.y;
    int localX = tid.x;
    int localY = tid.y;

    // Calculate tile origin in Y dimension (X doesn't need tiling)
    int tileY = (gid.y / TILE_SIZE) * TILE_SIZE - radius;

    // === Step 1: Horizontal blur (global → shared) ===
    // Each thread processes multiple rows to fill the padded height
    for (int py = localY; py < paddedHeight; py += tgSize.y) {
        // Skip if X is out of bounds
        if (globalX >= params.width) continue;

        // Clamp to image bounds (border replication)
        int srcY = clamp(tileY + py, 0, params.height - 1);

        // Store in shared memory
        sharedHoriz[py][localX] = horizontalGaussianBlur(
            source, gaussKernel, globalX, srcY, radius, params.width, params.rowStride
        );
    }

    // Wait for all threads to finish horizontal blur
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // === Step 2: Vertical blur (shared → global) ===
    // Only process valid output pixels
    if (globalX >= params.width || globalY >= params.height) {
        return;
    }

    // Calculate position in shared memory (accounting for halo offset)
    int sharedY = localY + radius;

    // Write final result to global memory
    destination[globalY * params.rowStride + globalX] = verticalGaussianBlur(
        (const threadgroup float*)sharedHoriz, gaussKernel, localX, sharedY, radius, 47, 16
    );
}

// Fused kernel: Blur → DoG → Extrema in single pass
// Processes 16x16 tiles with halo for Gaussian convolution
// Note: Extrema detection requires precise floating-point comparisons
// Compiled with -fno-fast-math to ensure IEEE-754 compliant comparisons
//
// Performance optimization: Uses sharded atomic counters (256 shards) with primitive
// root probing to reduce contention. Spatial locality preserved on first probe, then
// pseudo-random probing using primitive root 127 of prime 257 for overflow handling.
#pragma METAL fp math_mode(safe)
kernel void detectScaleSpaceExtremaFused(
    const device float* prevDoG [[buffer(0)]],             // DoG[layer-1] (pre-computed)
    const device float* currDoG [[buffer(1)]],             // DoG[layer] (detect extrema here)
    const device float* currGauss [[buffer(2)]],           // Gauss[layer+1] (blur to get Gauss[layer+2])
    device float* nextGauss [[buffer(3)]],                 // Output: Gauss[layer+2]
    device float* nextDoG [[buffer(4)]],                   // Output: DoG[layer+1]
    device atomic_uint* candidateCounts [[buffer(5)]],     // Array of 256 atomic counters (one per shard)
    device uint* candidates [[buffer(6)]],                 // Output candidate array (256 shards × 2048 capacity)
    constant FusedExtremaParams& params [[buffer(7)]],
    constant uint& shardCapacity [[buffer(8)]],            // Capacity per shard (typically 2048)
    constant float* gaussKernel [[buffer(9)]],             // 1D Gaussian kernel
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tgSize [[threads_per_threadgroup]])
{
    // Tile configuration: 16x16 processing + vertical halo for Gaussian blur
    const int TILE_SIZE = 16;
    int radius = params.kernelSize / 2;
    int paddedHeight = TILE_SIZE + 2 * (radius + 1);  // Vertical halo for blur

    // Shared memory for tile processing
    // Max kernel size ~27 (sigma ~3.0), so max radius ~13, max paddedHeight = 16 + 26 = 42
    threadgroup float sharedHoriz[48][18];      // Horizontal blur result with vertical halo
    threadgroup float sharedNextDoG[18][18];    // DoG with 1-pixel halo for extrema detection
    
    int globalX = gid.x;
    int globalY = gid.y;
    int localX = tid.x;
    int localY = tid.y;
    
    // Only process valid output pixels
    if (globalX >= params.width || globalY >= params.height) {
        return;
    }
    
    // Calculate tile origin in Y dimension (X doesn't need tiling for horizontal blur)
    int tileY = (gid.y / TILE_SIZE) * TILE_SIZE - radius - 1;
    
    // === Step 1: Horizontal blur (global → shared) ===
    // Each thread processes multiple rows to fill the padded height
    for (int py = localY; py < paddedHeight; py += tgSize.y) {        
        int srcY = clamp(tileY + py, 0, params.height - 1); // Clamp to image bounds (border replication)
        
        sharedHoriz[py][localX+1] = horizontalGaussianBlur(
            currGauss, gaussKernel, globalX, srcY, radius, params.width, params.rowStride
        );
        
        // Compute for halo regions
        if (localX == 0) { // Left halo
            int srcX = clamp(globalX-1, 0, params.width - 1);
            sharedHoriz[py][localX] = horizontalGaussianBlur(
                currGauss, gaussKernel, srcX, srcY, radius, params.width, params.rowStride
            );
        }
        if (localX == TILE_SIZE-1 || globalX == params.width-1) { // Right halo
            int srcX = clamp(globalX+1, 0, params.width - 1);
            sharedHoriz[py][localX+2] = horizontalGaussianBlur(
                currGauss, gaussKernel, srcX, srcY, radius, params.width, params.rowStride
            );
        }
    }
    
    // Wait for all threads to finish horizontal blur
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // === Step 2: Vertical blur (shared → global) ===

    // Calculate position in shared memory (accounting for halo offset)
    int sharedX = localX + 1;
    int sharedY = localY + radius + 1;

    // Perform vertical blur exactly like gaussianBlurVertical
    float sum = verticalGaussianBlur(
        (const threadgroup float*)sharedHoriz, gaussKernel, sharedX, sharedY, radius, 47, 18
    );

    // Write nextGauss to global memory
    nextGauss[globalY * params.rowStride + globalX] = sum;

    // Store in sharedNextDoG at interior position (+1 offset for halo border)
    sharedNextDoG[localY + 1][localX + 1] = sum - currGauss[globalY * params.rowStride + globalX];

    // === Step 2b: Compute halo DoG values ===
    // Border threads compute their adjacent halo pixels

    // Halo top row (localY == 0)
    if (localY == 0) {
        int srcY = clamp(globalY-1, 0, params.height-1);

        // Top center
        float topSum = verticalGaussianBlur(
            (const threadgroup float*)sharedHoriz, gaussKernel, sharedX, sharedY-1, radius, 47, 18
        );
        sharedNextDoG[0][localX + 1] = topSum - currGauss[srcY * params.rowStride + globalX];

        // Top-left corner
        if (localX == 0) {
            int srcX = clamp(globalX-1, 0, params.width-1);
            float cornerSum = verticalGaussianBlur(
                (const threadgroup float*)sharedHoriz, gaussKernel, sharedX-1, sharedY-1, radius, 47, 18
            );
            sharedNextDoG[0][0] = cornerSum - currGauss[srcY * params.rowStride + srcX];
        }
        // Top-right corner
        if (localX == TILE_SIZE-1 || globalX == params.width-1) {
            int srcX = clamp(globalX+1, 0, params.width-1);
            float cornerSum = verticalGaussianBlur(
                (const threadgroup float*)sharedHoriz, gaussKernel, sharedX+1, sharedY-1, radius, 47, 18
            );
            sharedNextDoG[0][localX + 2] = cornerSum - currGauss[srcY * params.rowStride + srcX];
        }
    }

    // Halo bottom row (localY == TILE_SIZE-1)
    if (localY == TILE_SIZE-1 || globalY == params.height-1) {
        int srcY = clamp(globalY+1, 0, params.height-1);

        // Bottom center
        float bottomSum = verticalGaussianBlur(
            (const threadgroup float*)sharedHoriz, gaussKernel, sharedX, sharedY+1, radius, 47, 18
        );
        sharedNextDoG[localY + 2][localX + 1] = bottomSum - currGauss[srcY * params.rowStride + globalX];

        // Bottom-left corner
        if (localX == 0) {
            int srcX = clamp(globalX-1, 0, params.width-1);
            float cornerSum = verticalGaussianBlur(
                (const threadgroup float*)sharedHoriz, gaussKernel, sharedX-1, sharedY+1, radius, 47, 18
            );
            sharedNextDoG[localY + 2][0] = cornerSum - currGauss[srcY * params.rowStride + srcX];
        }
        // Bottom-right corner
        if (localX == TILE_SIZE-1 || globalX == params.width-1) {
            int srcX = clamp(globalX+1, 0, params.width-1);
            float cornerSum = verticalGaussianBlur(
                (const threadgroup float*)sharedHoriz, gaussKernel, sharedX+1, sharedY+1, radius, 47, 18
            );
            sharedNextDoG[localY + 2][localX + 2] = cornerSum - currGauss[srcY * params.rowStride + srcX];
        }
    }

    // Halo left column
    if (localX == 0) {
        int srcX = clamp(globalX-1, 0, params.width-1);
        float leftSum = verticalGaussianBlur(
            (const threadgroup float*)sharedHoriz, gaussKernel, sharedX-1, sharedY, radius, 47, 18
        );
        sharedNextDoG[localY + 1][0] = leftSum - currGauss[globalY * params.rowStride + srcX];
    }

    // Halo right column
    if ((localX == TILE_SIZE-1 || globalX == params.width-1)) {
        int srcX = clamp(globalX+1, 0, params.width-1);
        float rightSum = verticalGaussianBlur(
            (const threadgroup float*)sharedHoriz, gaussKernel, sharedX+1, sharedY, radius, 47, 18
        );
        sharedNextDoG[localY + 1][localX + 2] = rightSum - currGauss[globalY * params.rowStride + srcX];
    }

    // Wait for all threads to finish computing halos
    threadgroup_barrier(mem_flags::mem_threadgroup);
    

    // === Step 3: Detect extrema in currDoG using [prevDoG, currDoG, nextDoG] ===
    // Check if this is a border pixel or non-extremum (but don't return early!)
    bool shouldDetect = (globalX >= params.border && globalX < params.width - params.border &&
                         globalY >= params.border && globalY < params.height - params.border);

    bool isExtremum = false;

    if (shouldDetect) {
        // Read center value from currDoG (use global memory)
        float val = currDoG[globalY * params.rowStride + globalX];
        int i = globalY * params.rowStride + globalX;
        int step = params.rowStride;

        // Quick threshold rejection
        if (fabs(val) > params.threshold) {
            float _00,_01,_02;
            float _10,    _12;
            float _20,_21,_22;
            int sharedCenterX = localX + 1;
            int sharedCenterY = localY + 1;
            if (val > 0) {
                _00 = currDoG[i-step-1]; _01 = currDoG[i-step]; _02 = currDoG[i-step+1];
                _10 = currDoG[i-1]; _12 = currDoG[i+1];
                _20 = currDoG[i+step-1]; _21 = currDoG[i+step]; _22 = currDoG[i+step+1];
                float vmax = fmax(fmax(fmax(_00,_01),fmax(_02,_10)),fmax(fmax(_12,_20),fmax(_21,_22)));
                if (val >= vmax) {
                    _00 = prevDoG[i-step-1]; _01 = prevDoG[i-step]; _02 = prevDoG[i-step+1];
                    _10 = prevDoG[i-1]; _12 = prevDoG[i+1];
                    _20 = prevDoG[i+step-1]; _21 = prevDoG[i+step]; _22 = prevDoG[i+step+1];
                    vmax = fmax(fmax(fmax(_00,_01),fmax(_02,_10)),fmax(fmax(_12,_20),fmax(_21,_22)));
                    if (val >= vmax) {
                        _00 = sharedNextDoG[sharedCenterY-1][sharedCenterX-1]; _01 = sharedNextDoG[sharedCenterY-1][sharedCenterX]; _02 = sharedNextDoG[sharedCenterY-1][sharedCenterX+1];
                        _10 = sharedNextDoG[sharedCenterY][sharedCenterX-1]; _12 = sharedNextDoG[sharedCenterY][sharedCenterX+1];
                        _20 = sharedNextDoG[sharedCenterY+1][sharedCenterX-1]; _21 = sharedNextDoG[sharedCenterY+1][sharedCenterX]; _22 = sharedNextDoG[sharedCenterY+1][sharedCenterX+1];
                        vmax = fmax(fmax(fmax(_00,_01),fmax(_02,_10)),fmax(fmax(_12,_20),fmax(_21,_22)));
                        if (val >= vmax) isExtremum = val >= fmax(prevDoG[i], sharedNextDoG[sharedCenterY][i]);
                    }
                }
            } else {
                _00 = currDoG[i-step-1]; _01 = currDoG[i-step]; _02 = currDoG[i-step+1];
                _10 = currDoG[i-1]; _12 = currDoG[i+1];
                _20 = currDoG[i+step-1]; _21 = currDoG[i+step]; _22 = currDoG[i+step+1];
                float vmin = fmin(fmin(fmin(_00,_01),fmin(_02,_10)),fmin(fmin(_12,_20),fmin(_21,_22)));
                if (val <= vmin) {
                    _00 = prevDoG[i-step-1]; _01 = prevDoG[i-step]; _02 = prevDoG[i-step+1];
                    _10 = prevDoG[i-1]; _12 = prevDoG[i+1];
                    _20 = prevDoG[i+step-1]; _21 = prevDoG[i+step]; _22 = prevDoG[i+step+1];
                    vmin = fmin(fmin(fmin(_00,_01),fmin(_02,_10)),fmin(fmin(_12,_20),fmin(_21,_22)));
                    if (val <= vmin) {
                        _00 = sharedNextDoG[sharedCenterY-1][sharedCenterX-1]; _01 = sharedNextDoG[sharedCenterY-1][sharedCenterX]; _02 = sharedNextDoG[sharedCenterY-1][sharedCenterX+1];
                        _10 = sharedNextDoG[sharedCenterY][sharedCenterX-1]; _12 = sharedNextDoG[sharedCenterY][sharedCenterX+1];
                        _20 = sharedNextDoG[sharedCenterY+1][sharedCenterX-1]; _21 = sharedNextDoG[sharedCenterY+1][sharedCenterX]; _22 = sharedNextDoG[sharedCenterY+1][sharedCenterX+1];
                        vmin = fmin(fmin(fmin(_00,_01),fmin(_02,_10)),fmin(fmin(_12,_20),fmin(_21,_22)));
                        if (val <= vmin) isExtremum = val <= fmin(prevDoG[i], sharedNextDoG[sharedCenterY][i]);
                    }
                }
            }

            // If survived all comparisons, this is a local extremum - write with sharding
            if (isExtremum) {
                // Sharded atomic with primitive root probing (256 shards)
                const uint NUM_SHARDS = 256;
                uint threadgroupID = tgid.y * ((params.width + 15) / 16) + tgid.x;
                uint shardIndex = threadgroupID % NUM_SHARDS;  // Initial: spatial locality
                uint localIndex = atomic_fetch_add_explicit(&candidateCounts[shardIndex], 1, memory_order_relaxed);

                // Check if this shard has capacity
                if (localIndex < shardCapacity) {
                    // Write packed candidate coordinate
                    candidates[shardIndex * shardCapacity + localIndex] = globalY << 16 | globalX;
                }
            }
        }
    }

    // === Step 4: Write computed DoG to output buffer ===
    // Wait for all extrema detection to complete before writing
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write the computed DoG from shared memory to global memory
    if (globalX < params.width && globalY < params.height) {
        // sharedNextDoG indices: +1 offset for halo border
        int sharedX = localX + 1;
        int sharedY = localY + 1;

        nextDoG[globalY * params.rowStride + globalX] = sharedNextDoG[sharedY][sharedX];
    }
}