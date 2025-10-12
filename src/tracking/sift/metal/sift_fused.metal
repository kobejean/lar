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

struct ExtremaParams {
    int width;              // Image width for this layer
    int height;             // Image height for this layer
    int rowStride;          // Row stride in floats (for aligned buffers)
    float threshold;        // Absolute threshold for extrema detection
    int border;             // Border size (uses SIFT_IMG_BORDER constant)
    int octave;             // Current octave index
    int layer;              // Current layer index (1 to nOctaveLayers)
};

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