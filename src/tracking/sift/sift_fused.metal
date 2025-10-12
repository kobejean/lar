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

struct ExtremaParams {
    int width;              // Image width for this layer
    int height;             // Image height for this layer
    int rowStride;          // Row stride in floats (for aligned buffers)
    float threshold;        // Absolute threshold for extrema detection
    int border;             // Border size (uses SIFT_IMG_BORDER constant)
    int octave;             // Current octave index
    int layer;              // Current layer index (1 to nOctaveLayers)
};

// Helper for 1D horizontal Gaussian blur with border replication
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
        int leftX = metal::clamp(srcX - radius + j, 0, width - 1);
        int rightX = metal::clamp(srcX + radius - j, 0, width - 1);

        float leftPixel = image[srcY * rowStride + leftX];
        float rightPixel = image[srcY * rowStride + rightX];
        float weight = gaussKernel[j];

        sum += weight * (leftPixel + rightPixel);
    }

    return sum;
}

// Helper for 1D vertical Gaussian blur with border replication
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
        int topY = metal::clamp(sharedY - radius + j, 0, maxSharedY);
        int bottomY = metal::clamp(sharedY + radius - j, 0, maxSharedY);

        float topPixel = sharedData[topY * stride + localX];
        float bottomPixel = sharedData[bottomY * stride + localX];
        float weight = gaussKernel[j];

        sum += weight * (topPixel + bottomPixel);
    }

    return sum;
}



// ============================================================================
// Fused Gaussian Blur Kernel (Horizontal + Vertical in one pass)
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
    const int MAX_PADDED_HEIGHT = TILE_SIZE + 26;
    int radius = params.kernelSize / 2;
    int paddedHeight = TILE_SIZE + 2 * radius;  // Only need vertical halo

    // Shared memory for horizontal blur results (with vertical halo)
    // Each column stores TILE_SIZE rows + 2*radius halo rows
    // Max kernel size ~27 (sigma ~3.0), so max radius ~13, max paddedHeight = 16 + 26 = 42
    threadgroup float sharedHoriz[MAX_PADDED_HEIGHT][256/TILE_SIZE];  // [paddedHeight][256/TILE_SIZE]

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

#pragma METAL fp math_mode(safe)
kernel void gaussianBlurAndDoGFused(
    const device float* currGauss [[buffer(0)]],
    device float* nextGauss [[buffer(1)]],
    device float* nextDoG [[buffer(2)]],
    constant GaussianBlurParams& params [[buffer(3)]],
    constant float* gaussKernel [[buffer(4)]],
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
        int srcY = tileY + py;

        // Clamp to image bounds (border replication)
        srcY = clamp(srcY, 0, params.height - 1);

        // Skip if X is out of bounds
        if (globalX >= params.width) {
            continue;
        }

        // Perform horizontal blur exactly like gaussianBlurHorizontal
        float centerPixel = currGauss[srcY * params.rowStride + globalX];
        float gauss = gaussKernel[radius] * centerPixel;

        // Symmetric pairs: m[j]*left + m[j]*right
        for (int j = 0; j < radius; j++) {
            int leftX = globalX - radius + j;
            int rightX = globalX + radius - j;

            // Border replication via clamp
            leftX = clamp(leftX, 0, params.width - 1);
            rightX = clamp(rightX, 0, params.width - 1);

            float leftPixel = currGauss[srcY * params.rowStride + leftX];
            float rightPixel = currGauss[srcY * params.rowStride + rightX];
            float weight = gaussKernel[j];

            gauss = gauss + (weight * leftPixel + weight * rightPixel);
        }

        // Store in shared memory
        sharedHoriz[py][localX] = gauss;
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

    // Perform vertical blur exactly like gaussianBlurVertical
    float centerPixel = sharedHoriz[sharedY][localX];
    float gauss = gaussKernel[radius] * centerPixel;

    // Symmetric pairs: m[j]*top + m[j]*bottom
    for (int j = 0; j < radius; j++) {
        int topY = sharedY - radius + j;
        int bottomY = sharedY + radius - j;

        // Clamp to shared memory bounds (sharedHoriz is [48][16])
        topY = clamp(topY, 0, 47);
        bottomY = clamp(bottomY, 0, 47);

        float topPixel = sharedHoriz[topY][localX];
        float bottomPixel = sharedHoriz[bottomY][localX];
        float weight = gaussKernel[j];

        gauss = gauss + (weight * topPixel + weight * bottomPixel);
    }

    // Write final result to global memory
    int idx = globalY * params.rowStride + globalX;
    nextGauss[idx] = gauss;
    nextDoG[idx] = gauss - currGauss[idx];
}

// Fused kernel: Blur → DoG → Extrema in single pass
// Processes 16x16 tiles with halo for Gaussian convolution
// Note: Extrema detection requires precise floating-point comparisons
// Compiled with -fno-fast-math to ensure IEEE-754 compliant comparisons
//
// Output: Bitarray encoding extrema positions (1 bit per pixel)
// - Deterministic output (no race conditions)
// - Memory efficient (1 bit per pixel vs 4 bytes per candidate)
// - Host scans bitarray using SIMD for fast candidate extraction
#pragma METAL fp math_mode(safe)
kernel void detectScaleSpaceExtremaFused(
    const device float* prevDoG [[buffer(0)]],             // DoG[layer-1] (pre-computed)
    const device float* currDoG [[buffer(1)]],             // DoG[layer] (detect extrema here)
    const device float* currGauss [[buffer(2)]],           // Gauss[layer+1] (blur to get Gauss[layer+2])
    device float* nextGauss [[buffer(3)]],                 // Output: Gauss[layer+2]
    device float* nextDoG [[buffer(4)]],                   // Output: DoG[layer+1]
    device atomic_uint* extremaBitarray [[buffer(5)]],     // Bitarray output (1 bit per pixel, packed as uint32)
    constant FusedExtremaParams& params [[buffer(7)]],
    constant float* gaussKernel [[buffer(9)]],             // 1D Gaussian kernel
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
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
    int sharedHorizX = localX + 1;
    int sharedHorizY = localY + radius + 1;
    int sharedDoGX = localX + 1;
    int sharedDoGY = localY + 1;
        
    // Only process valid output pixels
    if (globalX < params.width) {
        // Calculate tile origin in Y dimension (X doesn't need tiling for horizontal blur)
        int tileY = (gid.y / TILE_SIZE) * TILE_SIZE - radius - 1;
        
        // === Step 1: Horizontal blur (global → shared) ===
        // Each thread processes multiple rows to fill the padded height
        for (int py = localY; py < paddedHeight; py += tgSize.y) {        
            int srcY = clamp(tileY + py, 0, params.height - 1); // Clamp to image bounds (border replication)
            
            sharedHoriz[py][sharedHorizX] = horizontalGaussianBlur(
                currGauss, gaussKernel, globalX, srcY, radius, params.width, params.rowStride
            );
            
            // Compute for halo regions
            if (localX == 0) { // Left halo
                int srcX = clamp(globalX-1, 0, params.width - 1);
                sharedHoriz[py][sharedHorizX-1] = horizontalGaussianBlur(
                    currGauss, gaussKernel, srcX, srcY, radius, params.width, params.rowStride
                );
            } else if (localX == TILE_SIZE-1) { // Right halo
                int srcX = clamp(globalX+1, 0, params.width - 1);
                sharedHoriz[py][sharedHorizX+1] = horizontalGaussianBlur(
                    currGauss, gaussKernel, srcX, srcY, radius, params.width, params.rowStride
                );
            }
        }
    }
    
    // Wait for all threads to finish horizontal blur
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // === Step 2: Vertical blur (shared → global) ===
    
    if (globalX < params.width && globalY < params.height) {
        // Calculate position in shared memory (accounting for halo offset)

        // Perform vertical blur exactly like gaussianBlurVertical
        float gauss = verticalGaussianBlur(
            (const threadgroup float*)sharedHoriz, gaussKernel, sharedHorizX, sharedHorizY, radius, 47, 18
        );

        // Write nextGauss to global memory
        nextGauss[globalY * params.rowStride + globalX] = gauss;

        // Store in sharedNextDoG at interior position (+1 offset for halo border)
        float dogVal = gauss - currGauss[globalY * params.rowStride + globalX];
        sharedNextDoG[sharedDoGY][sharedDoGX] = dogVal;
        nextDoG[globalY * params.rowStride + globalX] = dogVal;

        // === Step 2b: Compute halo DoG values ===
        // Border threads compute their adjacent halo pixels

        // Halo top row (localY == 0)
        if (localY == 0) {
            int srcY = clamp(globalY-1, 0, params.height-1);

            // Top center
            float topSum = verticalGaussianBlur(
                (const threadgroup float*)sharedHoriz, gaussKernel, sharedHorizX, sharedHorizY-1, radius, 47, 18
            );
            sharedNextDoG[0][sharedDoGX] = topSum - currGauss[srcY * params.rowStride + globalX];

            // Top-left corner
            if (localX == 0) {
                int srcX = clamp(globalX-1, 0, params.width-1);
                float cornerSum = verticalGaussianBlur(
                    (const threadgroup float*)sharedHoriz, gaussKernel, sharedHorizX-1, sharedHorizY-1, radius, 47, 18
                );
                sharedNextDoG[0][0] = cornerSum - currGauss[srcY * params.rowStride + srcX];
            }
            // Top-right corner
            else if (localX == TILE_SIZE-1) {
                int srcX = clamp(globalX+1, 0, params.width-1);
                float cornerSum = verticalGaussianBlur(
                    (const threadgroup float*)sharedHoriz, gaussKernel, sharedHorizX+1, sharedHorizY-1, radius, 47, 18
                );
                sharedNextDoG[0][sharedDoGX + 1] = cornerSum - currGauss[srcY * params.rowStride + srcX];
            }
        }

        // Halo bottom row (localY == TILE_SIZE-1)
        else if (localY == TILE_SIZE-1) {
            int srcY = clamp(globalY+1, 0, params.height-1);

            // Bottom center
            float bottomSum = verticalGaussianBlur(
                (const threadgroup float*)sharedHoriz, gaussKernel, sharedHorizX, sharedHorizY+1, radius, 47, 18
            );
            sharedNextDoG[sharedDoGY + 1][sharedDoGX] = bottomSum - currGauss[srcY * params.rowStride + globalX];

            // Bottom-left corner
            if (localX == 0) {
                int srcX = clamp(globalX-1, 0, params.width-1);
                float cornerSum = verticalGaussianBlur(
                    (const threadgroup float*)sharedHoriz, gaussKernel, sharedHorizX-1, sharedHorizY+1, radius, 47, 18
                );
                sharedNextDoG[sharedDoGY + 1][0] = cornerSum - currGauss[srcY * params.rowStride + srcX];
            }
            // Bottom-right corner
            else if (localX == TILE_SIZE-1) {
                int srcX = clamp(globalX+1, 0, params.width-1);
                float cornerSum = verticalGaussianBlur(
                    (const threadgroup float*)sharedHoriz, gaussKernel, sharedHorizX+1, sharedHorizY+1, radius, 47, 18
                );
                sharedNextDoG[sharedDoGY + 1][sharedDoGX + 1] = cornerSum - currGauss[srcY * params.rowStride + srcX];
            }
        }

        // Halo left column
        if (localX == 0) {
            int srcX = clamp(globalX-1, 0, params.width-1);
            float leftSum = verticalGaussianBlur(
                (const threadgroup float*)sharedHoriz, gaussKernel, sharedHorizX-1, sharedHorizY, radius, 47, 18
            );
            sharedNextDoG[sharedDoGY][0] = leftSum - currGauss[globalY * params.rowStride + srcX];
        }

        // Halo right column
        else if (localX == TILE_SIZE-1) {
            int srcX = clamp(globalX+1, 0, params.width-1);
            float rightSum = verticalGaussianBlur(
                (const threadgroup float*)sharedHoriz, gaussKernel, sharedHorizX+1, sharedHorizY, radius, 47, 18
            );
            sharedNextDoG[sharedDoGY][sharedDoGX + 1] = rightSum - currGauss[globalY * params.rowStride + srcX];
        }
    }

    // Wait for all threads to finish computing halos
    threadgroup_barrier(mem_flags::mem_threadgroup);
    

    // === Step 3: Detect extrema in currDoG using [prevDoG, currDoG, nextDoG] ===
    // Check if this is a border pixel or non-extremum
    if (globalX < params.border || globalX >= params.width - params.border ||
        globalY < params.border || globalY >= params.height - params.border) return;

    // Read center value from currDoG (use global memory)
    float val = currDoG[globalY * params.rowStride + globalX];
    int i = globalY * params.rowStride + globalX;
    int step = params.rowStride;

    // Quick threshold rejection
    if (fabs(val) <= params.threshold) return;
    float _00,_01,_02;
    float _10,    _12;
    float _20,_21,_22;
    if (val > 0) {
        _00 = currDoG[i-step-1]; _01 = currDoG[i-step]; _02 = currDoG[i-step+1];
        _10 = currDoG[i-1]; _12 = currDoG[i+1];
        _20 = currDoG[i+step-1]; _21 = currDoG[i+step]; _22 = currDoG[i+step+1];
        float vmax = fmax(fmax(fmax(_00,_01),fmax(_02,_10)),fmax(fmax(_12,_20),fmax(_21,_22)));
        if (val < vmax) return;
        _00 = prevDoG[i-step-1]; _01 = prevDoG[i-step]; _02 = prevDoG[i-step+1];
        _10 = prevDoG[i-1]; _12 = prevDoG[i+1];
        _20 = prevDoG[i+step-1]; _21 = prevDoG[i+step]; _22 = prevDoG[i+step+1];
        vmax = fmax(fmax(fmax(_00,_01),fmax(_02,_10)),fmax(fmax(_12,_20),fmax(_21,_22)));
        if (val < vmax) return;
        _00 = sharedNextDoG[sharedDoGY-1][sharedDoGX-1]; _01 = sharedNextDoG[sharedDoGY-1][sharedDoGX]; _02 = sharedNextDoG[sharedDoGY-1][sharedDoGX+1];
        _10 = sharedNextDoG[sharedDoGY][sharedDoGX-1]; _12 = sharedNextDoG[sharedDoGY][sharedDoGX+1];
        _20 = sharedNextDoG[sharedDoGY+1][sharedDoGX-1]; _21 = sharedNextDoG[sharedDoGY+1][sharedDoGX]; _22 = sharedNextDoG[sharedDoGY+1][sharedDoGX+1];
        vmax = fmax(fmax(fmax(_00,_01),fmax(_02,_10)),fmax(fmax(_12,_20),fmax(_21,_22)));
        if (val < vmax) return;
        vmax = fmax(prevDoG[i], sharedNextDoG[sharedDoGY][sharedDoGX]);
        if (val < vmax) return;

    } else {
        _00 = currDoG[i-step-1]; _01 = currDoG[i-step]; _02 = currDoG[i-step+1];
        _10 = currDoG[i-1]; _12 = currDoG[i+1];
        _20 = currDoG[i+step-1]; _21 = currDoG[i+step]; _22 = currDoG[i+step+1];
        float vmin = fmin(fmin(fmin(_00,_01),fmin(_02,_10)),fmin(fmin(_12,_20),fmin(_21,_22)));
        if (val > vmin) return;
        _00 = prevDoG[i-step-1]; _01 = prevDoG[i-step]; _02 = prevDoG[i-step+1];
        _10 = prevDoG[i-1]; _12 = prevDoG[i+1];
        _20 = prevDoG[i+step-1]; _21 = prevDoG[i+step]; _22 = prevDoG[i+step+1];
        vmin = fmin(fmin(fmin(_00,_01),fmin(_02,_10)),fmin(fmin(_12,_20),fmin(_21,_22)));
        if (val > vmin) return;
        _00 = sharedNextDoG[sharedDoGY-1][sharedDoGX-1]; _01 = sharedNextDoG[sharedDoGY-1][sharedDoGX]; _02 = sharedNextDoG[sharedDoGY-1][sharedDoGX+1];
        _10 = sharedNextDoG[sharedDoGY][sharedDoGX-1]; _12 = sharedNextDoG[sharedDoGY][sharedDoGX+1];
        _20 = sharedNextDoG[sharedDoGY+1][sharedDoGX-1]; _21 = sharedNextDoG[sharedDoGY+1][sharedDoGX]; _22 = sharedNextDoG[sharedDoGY+1][sharedDoGX+1];
        vmin = fmin(fmin(fmin(_00,_01),fmin(_02,_10)),fmin(fmin(_12,_20),fmin(_21,_22)));
        if (val > vmin) return;
        vmin = fmin(prevDoG[i], sharedNextDoG[sharedDoGY][sharedDoGX]);
        if (val > vmin) return;
    }

    // Calculate linear bit index: row-major order
    uint bitIndex = globalY * params.width + globalX;

    // Calculate chunk index and bit offset
    // Each uint32 stores 32 bits (pixels)
    uint chunkIndex = bitIndex >> 5;      // Divide by 32
    uint bitOffset = bitIndex & 31;       // Modulo 32

    // Set the bit using atomic OR
    // This is safe because each pixel maps to exactly one bit
    atomic_fetch_or_explicit(&extremaBitarray[chunkIndex], (1u << bitOffset), memory_order_relaxed);
}


// Kernel: Detect local extrema in 3D scale space (26-neighbor comparison)
// Each thread processes one pixel in the middle layer
// Note: Extrema detection requires precise floating-point comparisons
// Compiled with -fno-fast-math to ensure IEEE-754 compliant comparisons
//
// Output: Bitarray encoding extrema positions (1 bit per pixel)
// - Deterministic output (no race conditions)
// - Memory efficient (1 bit per pixel vs 4 bytes per candidate)
// - Host scans bitarray using SIMD for fast candidate extraction
#pragma METAL fp math_mode(safe)
kernel void detectExtrema(
    const device float* prevDoG [[buffer(0)]],   // DoG layer i-1
    const device float* currDoG [[buffer(1)]],   // DoG layer i (center)
    const device float* nextDoG [[buffer(2)]],   // DoG layer i+1
    device atomic_uint* extremaBitarray [[buffer(3)]], // Bitarray output (1 bit per pixel, packed as uint32)
    constant ExtremaParams& params [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    int x = gid.x;
    int y = gid.y;

    // Check if this thread should process (not border, not out of bounds)
    if (x < params.border || x >= params.width - params.border ||
        y < params.border || y >= params.height - params.border) return;

    // Read center pixel value
    int step = params.rowStride;
    int i = y * step + x;
    float val = currDoG[i];

    // Quick threshold rejection
    if (fabs(val) <= params.threshold) return;
    float _00,_01,_02;
    float _10,    _12;
    float _20,_21,_22;

    if (val > 0) {
        _00 = currDoG[i-step-1]; _01 = currDoG[i-step]; _02 = currDoG[i-step+1];
        _10 = currDoG[i-1]; _12 = currDoG[i+1];
        _20 = currDoG[i+step-1]; _21 = currDoG[i+step]; _22 = currDoG[i+step+1];
        float vmax = fmax(fmax(fmax(_00,_01),fmax(_02,_10)),fmax(fmax(_12,_20),fmax(_21,_22)));
        if (val < vmax) return;
        _00 = prevDoG[i-step-1]; _01 = prevDoG[i-step]; _02 = prevDoG[i-step+1];
        _10 = prevDoG[i-1]; _12 = prevDoG[i+1];
        _20 = prevDoG[i+step-1]; _21 = prevDoG[i+step]; _22 = prevDoG[i+step+1];
        vmax = fmax(fmax(fmax(_00,_01),fmax(_02,_10)),fmax(fmax(_12,_20),fmax(_21,_22)));
        if (val < vmax) return;
        _00 = nextDoG[i-step-1]; _01 = nextDoG[i-step]; _02 = nextDoG[i-step+1];
        _10 = nextDoG[i-1]; _12 = nextDoG[i+1];
        _20 = nextDoG[i+step-1]; _21 = nextDoG[i+step]; _22 = nextDoG[i+step+1];
        vmax = fmax(fmax(fmax(_00,_01),fmax(_02,_10)),fmax(fmax(_12,_20),fmax(_21,_22)));
        if (val < vmax) return;
        vmax = fmax(prevDoG[i], nextDoG[i]);
        if (val < vmax) return;
    } else {
        _00 = currDoG[i-step-1]; _01 = currDoG[i-step]; _02 = currDoG[i-step+1];
        _10 = currDoG[i-1]; _12 = currDoG[i+1];
        _20 = currDoG[i+step-1]; _21 = currDoG[i+step]; _22 = currDoG[i+step+1];
        float vmin = fmin(fmin(fmin(_00,_01),fmin(_02,_10)),fmin(fmin(_12,_20),fmin(_21,_22)));
        if (val > vmin) return;
        _00 = prevDoG[i-step-1]; _01 = prevDoG[i-step]; _02 = prevDoG[i-step+1];
        _10 = prevDoG[i-1]; _12 = prevDoG[i+1];
        _20 = prevDoG[i+step-1]; _21 = prevDoG[i+step]; _22 = prevDoG[i+step+1];
        vmin = fmin(fmin(fmin(_00,_01),fmin(_02,_10)),fmin(fmin(_12,_20),fmin(_21,_22)));
        if (val > vmin) return;
        _00 = nextDoG[i-step-1]; _01 = nextDoG[i-step]; _02 = nextDoG[i-step+1];
        _10 = nextDoG[i-1]; _12 = nextDoG[i+1];
        _20 = nextDoG[i+step-1]; _21 = nextDoG[i+step]; _22 = nextDoG[i+step+1];
        vmin = fmin(fmin(fmin(_00,_01),fmin(_02,_10)),fmin(fmin(_12,_20),fmin(_21,_22)));
        if (val > vmin) return;
        vmin = fmin(prevDoG[i], nextDoG[i]);
        if (val > vmin) return;
    }

    // Calculate linear bit index: row-major order
    uint bitIndex = y * params.width + x;

    // Calculate chunk index and bit offset
    // Each uint32 stores 32 bits (pixels)
    uint chunkIndex = bitIndex >> 5;      // Divide by 32
    uint bitOffset = bitIndex & 31;       // Modulo 32

    // Set the bit using atomic OR
    // This is safe because each pixel maps to exactly one bit
    atomic_fetch_or_explicit(&extremaBitarray[chunkIndex], (1u << bitOffset), memory_order_relaxed);
}