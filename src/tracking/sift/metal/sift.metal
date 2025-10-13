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

struct ResizeParams {
    int srcWidth;           // Source image width
    int srcHeight;          // Source image height
    int srcRowStride;       // Source row stride in floats
    int dstWidth;           // Destination image width
    int dstHeight;          // Destination image height
    int dstRowStride;       // Destination row stride in floats
};

// ============================================================================
// Custom Gaussian Blur Kernels (OpenCV-inspired pattern)
// ============================================================================
// These kernels implement separable Gaussian convolution with the same
// accumulation pattern as OpenCV's hlineSmoothONa_yzy_a function:
// 1. Center tap first
// 2. Symmetric pairs: m[j]*left + m[j]*right (two multiplications per pair)
//
// Note: These produce ~3-6e-05 max error vs MPS but match OpenCV's approach

// Horizontal Gaussian blur pass (texture-based for 20-35% speedup)
#pragma METAL fp math_mode(safe)
kernel void gaussianBlurHorizontal(
    texture2d<float, access::read> currGauss [[texture(0)]],
    texture2d<float, access::write> nextGauss [[texture(1)]],
    constant GaussianBlurParams& params [[buffer(0)]],
    constant float* gaussKernel [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint x = gid.x;
    uint y = gid.y;

    // Use texture dimensions for bounds checking
    if (x >= currGauss.get_width() || y >= currGauss.get_height()) {
        return;
    }

    int radius = params.kernelSize / 2;

    // Center tap first (matching OpenCV pattern)
    float centerPixel = currGauss.read(uint2(x, y)).r;
    float sum = gaussKernel[radius] * centerPixel;

    // Symmetric pairs: m[j]*left + m[j]*right
    for (int j = 0; j < radius; j++) {
        int leftX = int(x) - radius + j;
        int rightX = int(x) + radius - j;

        // Border replication via clamp
        leftX = clamp(leftX, 0, int(currGauss.get_width()) - 1);
        rightX = clamp(rightX, 0, int(currGauss.get_width()) - 1);

        float leftPixel = currGauss.read(uint2(leftX, y)).r;
        float rightPixel = currGauss.read(uint2(rightX, y)).r;
        float weight = gaussKernel[j];

        // Match OpenCV: m[j]*left + m[j]*right (two muls, one add)
        sum = sum + (weight * leftPixel + weight * rightPixel);
    }

    nextGauss.write(sum, uint2(x, y));
}

// Vertical Gaussian blur pass (texture-based - currently unused, kept for reference)
#pragma METAL fp math_mode(safe)
kernel void gaussianBlurVerticalAndDoG(
    texture2d<float, access::read> currGauss [[texture(0)]],
    texture2d<float, access::write> nextGauss [[texture(1)]],
    texture2d<float, access::write> nextDoG [[texture(2)]],
    constant GaussianBlurParams& params [[buffer(0)]],
    constant float* gaussKernel [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint x = gid.x;
    uint y = gid.y;

    // Use texture dimensions for bounds checking
    if (x >= currGauss.get_width() || y >= currGauss.get_height()) {
        return;
    }

    int radius = params.kernelSize / 2;

    // Center tap first (matching OpenCV pattern)
    float centerPixel = currGauss.read(uint2(x, y)).r;
    float sum = gaussKernel[radius] * centerPixel;

    // Symmetric pairs: m[j]*top + m[j]*bottom
    for (int j = 0; j < radius; j++) {
        int topY = int(y) - radius + j;
        int bottomY = int(y) + radius - j;

        // Border replication via clamp
        topY = clamp(topY, 0, int(currGauss.get_height()) - 1);
        bottomY = clamp(bottomY, 0, int(currGauss.get_height()) - 1);

        float topPixel = currGauss.read(uint2(x, topY)).r;
        float bottomPixel = currGauss.read(uint2(x, bottomY)).r;
        float weight = gaussKernel[j];

        // Match OpenCV: m[j]*top + m[j]*bottom (two muls, one add)
        sum = sum + (weight * topPixel + weight * bottomPixel);
    }

    // Write result (scalar write for R32Float format)
    nextGauss.write(sum, uint2(x, y));
}

// Fused Gaussian blur + DoG (texture-based for 20-35% speedup)
#pragma METAL fp math_mode(safe)
kernel void gaussianBlurAndDoGFused(
    texture2d<float, access::read> currGauss [[texture(0)]],
    texture2d<float, access::write> nextGauss [[texture(1)]],
    texture2d<float, access::write> nextDoG [[texture(2)]],
    constant GaussianBlurParams& params [[buffer(0)]],
    constant float* gaussKernel [[buffer(1)]],
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

    uint globalX = gid.x;
    uint globalY = gid.y;
    uint localX = tid.x;
    uint localY = tid.y;

    // Calculate tile origin in Y dimension (X doesn't need tiling)
    int tileY = (int(gid.y) / TILE_SIZE) * TILE_SIZE - radius;

    // === Step 1: Horizontal blur (texture → shared) ===
    // Each thread processes multiple rows to fill the padded height
    for (int py = localY; py < paddedHeight; py += tgSize.y) {
        int srcY = tileY + py;

        // Clamp to image bounds (border replication)
        srcY = clamp(srcY, 0, int(currGauss.get_height()) - 1);

        // Skip if X is out of bounds
        if (globalX >= currGauss.get_width()) {
            continue;
        }

        // Perform horizontal blur with texture reads
        float centerPixel = currGauss.read(uint2(globalX, srcY)).r;
        float gauss = gaussKernel[radius] * centerPixel;

        // Symmetric pairs: m[j]*left + m[j]*right
        for (int j = 0; j < radius; j++) {
            int leftX = int(globalX) - radius + j;
            int rightX = int(globalX) + radius - j;

            // Border replication via clamp
            leftX = clamp(leftX, 0, int(currGauss.get_width()) - 1);
            rightX = clamp(rightX, 0, int(currGauss.get_width()) - 1);

            float leftPixel = currGauss.read(uint2(leftX, srcY)).r;
            float rightPixel = currGauss.read(uint2(rightX, srcY)).r;
            float weight = gaussKernel[j];

            gauss = gauss + (weight * leftPixel + weight * rightPixel);
        }

        // Store in shared memory
        sharedHoriz[py][localX] = gauss;
    }

    // Wait for all threads to finish horizontal blur
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // === Step 2: Vertical blur (shared → texture) ===
    // Only process valid output pixels
    if (globalX >= currGauss.get_width() || globalY >= currGauss.get_height()) {
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

    // Write final result to textures (scalar writes for R32Float format)
    float currVal = currGauss.read(uint2(globalX, globalY)).r;
    nextGauss.write(gauss, uint2(globalX, globalY));
    nextDoG.write(gauss - currVal, uint2(globalX, globalY));
}

// Extrema detection (texture-based for better cache performance)
#pragma METAL fp math_mode(safe)
kernel void detectExtrema(
    texture2d<float, access::read> prevDoG [[texture(0)]],   // DoG layer i-1
    texture2d<float, access::read> currDoG [[texture(1)]],   // DoG layer i (center)
    texture2d<float, access::read> nextDoG [[texture(2)]],   // DoG layer i+1
    device atomic_uint* extremaBitarray [[buffer(0)]], // Bitarray output (1 bit per pixel, packed as uint32)
    constant ExtremaParams& params [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint x = gid.x;
    uint y = gid.y;

    // Check if this thread should process (not border, not out of bounds)
    if (x < params.border || x >= currDoG.get_width() - params.border ||
        y < params.border || y >= currDoG.get_height() - params.border) return;

    // Read center pixel value
    float val = currDoG.read(uint2(x, y)).r;

    // Quick threshold rejection
    if (fabs(val) <= params.threshold) return;

    float _00,_01,_02;
    float _10,    _12;
    float _20,_21,_22;

    if (val > 0) {
        // Check current layer 3×3 neighborhood
        _00 = currDoG.read(uint2(x-1, y-1)).r; _01 = currDoG.read(uint2(x, y-1)).r; _02 = currDoG.read(uint2(x+1, y-1)).r;
        _10 = currDoG.read(uint2(x-1, y  )).r;                                      _12 = currDoG.read(uint2(x+1, y  )).r;
        _20 = currDoG.read(uint2(x-1, y+1)).r; _21 = currDoG.read(uint2(x, y+1)).r; _22 = currDoG.read(uint2(x+1, y+1)).r;
        float vmax = fmax(fmax(fmax(_00,_01),fmax(_02,_10)),fmax(fmax(_12,_20),fmax(_21,_22)));
        if (val < vmax) return;

        // Check previous layer 3×3 neighborhood
        _00 = prevDoG.read(uint2(x-1, y-1)).r; _01 = prevDoG.read(uint2(x, y-1)).r; _02 = prevDoG.read(uint2(x+1, y-1)).r;
        _10 = prevDoG.read(uint2(x-1, y  )).r;                                      _12 = prevDoG.read(uint2(x+1, y  )).r;
        _20 = prevDoG.read(uint2(x-1, y+1)).r; _21 = prevDoG.read(uint2(x, y+1)).r; _22 = prevDoG.read(uint2(x+1, y+1)).r;
        vmax = fmax(fmax(fmax(_00,_01),fmax(_02,_10)),fmax(fmax(_12,_20),fmax(_21,_22)));
        if (val < vmax) return;

        // Check next layer 3×3 neighborhood
        _00 = nextDoG.read(uint2(x-1, y-1)).r; _01 = nextDoG.read(uint2(x, y-1)).r; _02 = nextDoG.read(uint2(x+1, y-1)).r;
        _10 = nextDoG.read(uint2(x-1, y  )).r;                                      _12 = nextDoG.read(uint2(x+1, y  )).r;
        _20 = nextDoG.read(uint2(x-1, y+1)).r; _21 = nextDoG.read(uint2(x, y+1)).r; _22 = nextDoG.read(uint2(x+1, y+1)).r;
        vmax = fmax(fmax(fmax(_00,_01),fmax(_02,_10)),fmax(fmax(_12,_20),fmax(_21,_22)));
        if (val < vmax) return;

        // Check center pixels of prev/next layers
        vmax = fmax(prevDoG.read(uint2(x, y)).r, nextDoG.read(uint2(x, y)).r);
        if (val < vmax) return;
    } else {
        // Check current layer 3×3 neighborhood
        _00 = currDoG.read(uint2(x-1, y-1)).r; _01 = currDoG.read(uint2(x, y-1)).r; _02 = currDoG.read(uint2(x+1, y-1)).r;
        _10 = currDoG.read(uint2(x-1, y  )).r;                                      _12 = currDoG.read(uint2(x+1, y  )).r;
        _20 = currDoG.read(uint2(x-1, y+1)).r; _21 = currDoG.read(uint2(x, y+1)).r; _22 = currDoG.read(uint2(x+1, y+1)).r;
        float vmin = fmin(fmin(fmin(_00,_01),fmin(_02,_10)),fmin(fmin(_12,_20),fmin(_21,_22)));
        if (val > vmin) return;

        // Check previous layer 3×3 neighborhood
        _00 = prevDoG.read(uint2(x-1, y-1)).r; _01 = prevDoG.read(uint2(x, y-1)).r; _02 = prevDoG.read(uint2(x+1, y-1)).r;
        _10 = prevDoG.read(uint2(x-1, y  )).r;                                      _12 = prevDoG.read(uint2(x+1, y  )).r;
        _20 = prevDoG.read(uint2(x-1, y+1)).r; _21 = prevDoG.read(uint2(x, y+1)).r; _22 = prevDoG.read(uint2(x+1, y+1)).r;
        vmin = fmin(fmin(fmin(_00,_01),fmin(_02,_10)),fmin(fmin(_12,_20),fmin(_21,_22)));
        if (val > vmin) return;

        // Check next layer 3×3 neighborhood
        _00 = nextDoG.read(uint2(x-1, y-1)).r; _01 = nextDoG.read(uint2(x, y-1)).r; _02 = nextDoG.read(uint2(x+1, y-1)).r;
        _10 = nextDoG.read(uint2(x-1, y  )).r;                                      _12 = nextDoG.read(uint2(x+1, y  )).r;
        _20 = nextDoG.read(uint2(x-1, y+1)).r; _21 = nextDoG.read(uint2(x, y+1)).r; _22 = nextDoG.read(uint2(x+1, y+1)).r;
        vmin = fmin(fmin(fmin(_00,_01),fmin(_02,_10)),fmin(fmin(_12,_20),fmin(_21,_22)));
        if (val > vmin) return;

        // Check center pixels of prev/next layers
        vmin = fmin(prevDoG.read(uint2(x, y)).r, nextDoG.read(uint2(x, y)).r);
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

// ============================================================================
// Nearest-Neighbor Downsample (2x) - texture-based
// ============================================================================
// Implements OpenCV's cv::resize(..., cv::INTER_NEAREST) algorithm:
//   sx = floor(dst_x * inv_scale_x) = floor(dst_x * 2.0)
//   sy = floor(dst_y * inv_scale_y) = floor(dst_y * 2.0)
//   dst[dst_y][dst_x] = src[sy][sx]
//
// This specialized kernel eliminates CPU/GPU sync bottleneck when preparing
// octave base images (octave N+1 base = downsample octave N's gauss[nLevels-3])
//
#pragma METAL fp math_mode(safe)
kernel void resizeNearestNeighbor2x(
    texture2d<float, access::read> src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    constant ResizeParams& params [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint dstX = gid.x;
    uint dstY = gid.y;

    // Bounds check using texture dimensions
    if (dstX >= dst.get_width() || dstY >= dst.get_height()) {
        return;
    }

    // OpenCV INTER_NEAREST mapping: floor(dst_coord * inv_scale)
    // For 2x downsample: inv_scale = 2.0
    uint srcX = dstX * 2;
    uint srcY = dstY * 2;

    // Clamp to source bounds (OpenCV uses min(sx, width-1))
    srcX = min(srcX, src.get_width() - 1);
    srcY = min(srcY, src.get_height() - 1);

    // Read from source texture and write to destination texture (scalar for R32Float)
    float value = src.read(uint2(srcX, srcY)).r;
    dst.write(value, uint2(dstX, dstY));
}