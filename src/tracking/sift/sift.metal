// Metal compute shader for SIFT scale-space extrema detection
// Performs 3D local extrema detection across DoG pyramid layers
#include <metal_stdlib>
#include "sift_constants.metal"
using namespace metal;

// Kernel parameters
struct ExtremaParams {
    int width;              // Image width for this layer
    int height;             // Image height for this layer
    int rowStride;          // Row stride in floats (for aligned buffers)
    float threshold;        // Absolute threshold for extrema detection
    int border;             // Border size (uses SIFT_IMG_BORDER constant)
    int octave;             // Current octave index
    int layer;              // Current layer index (1 to nOctaveLayers)
};

// Gaussian blur parameters (for custom Metal kernels)
struct GaussianBlurParams {
    int width;
    int height;
    int rowStride;
    int kernelSize;
};

// Kernel: Detect local extrema in 3D scale space (26-neighbor comparison)
// Each thread processes one pixel in the middle layer
// Note: Extrema detection requires precise floating-point comparisons
// Compiled with -fno-fast-math to ensure IEEE-754 compliant comparisons
//
// Performance optimization: Uses sharded atomic counters (256 shards) with primitive
// root probing to reduce contention. Spatial locality preserved on first probe, then
// pseudo-random probing using primitive root 127 of prime 257 for overflow handling.
#pragma METAL fp math_mode(safe)
kernel void detectScaleSpaceExtrema(
    const device float* prevLayer [[buffer(0)]],   // DoG layer i-1
    const device float* currLayer [[buffer(1)]],   // DoG layer i (center)
    const device float* nextLayer [[buffer(2)]],   // DoG layer i+1
    device atomic_uint* candidateCounts [[buffer(3)]], // Array of 256 atomic counters (one per shard)
    device uint* candidates [[buffer(4)]], // Output candidate array (256 shards × 1024 capacity)
    constant ExtremaParams& params [[buffer(5)]],
    constant uint& shardCapacity [[buffer(6)]],    // Capacity per shard (typically 1024)
    uint2 gid [[thread_position_in_grid]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    int x = gid.x;
    int y = gid.y;
    bool isExtremum = false;
    float val;

    // Check if this thread should process (not border, not out of bounds)
    bool shouldProcess = (x >= params.border && x < params.width - params.border &&
                          y >= params.border && y < params.height - params.border);
    if (shouldProcess) {
        // Read center pixel value
        int step = params.rowStride;
        int i = y * step + x;
        val = currLayer[i];

        // Quick threshold rejection
        if (fabs(val) > params.threshold) {
            float _00,_01,_02;
            float _10,    _12;
            float _20,_21,_22;

            if (val > 0) {
                _00 = currLayer[i-step-1]; _01 = currLayer[i-step]; _02 = currLayer[i-step+1];
                _10 = currLayer[i-1]; _12 = currLayer[i+1];
                _20 = currLayer[i+step-1]; _21 = currLayer[i+step]; _22 = currLayer[i+step+1];
                float vmax = fmax(fmax(fmax(_00,_01),fmax(_02,_10)),fmax(fmax(_12,_20),fmax(_21,_22)));
                if (val >= vmax) {
                    _00 = prevLayer[i-step-1]; _01 = prevLayer[i-step]; _02 = prevLayer[i-step+1];
                    _10 = prevLayer[i-1]; _12 = prevLayer[i+1];
                    _20 = prevLayer[i+step-1]; _21 = prevLayer[i+step]; _22 = prevLayer[i+step+1];
                    vmax = fmax(fmax(fmax(_00,_01),fmax(_02,_10)),fmax(fmax(_12,_20),fmax(_21,_22)));
                    if (val >= vmax) {
                        _00 = nextLayer[i-step-1]; _01 = nextLayer[i-step]; _02 = nextLayer[i-step+1];
                        _10 = nextLayer[i-1]; _12 = nextLayer[i+1];
                        _20 = nextLayer[i+step-1]; _21 = nextLayer[i+step]; _22 = nextLayer[i+step+1];
                        vmax = fmax(fmax(fmax(_00,_01),fmax(_02,_10)),fmax(fmax(_12,_20),fmax(_21,_22)));
                        if (val >= vmax) {
                            vmax = fmax(prevLayer[i], nextLayer[i]);
                            if (val >= vmax) {
                                isExtremum = true;
                            }
                        }
                    }
                }
            } else {
                _00 = currLayer[i-step-1]; _01 = currLayer[i-step]; _02 = currLayer[i-step+1];
                _10 = currLayer[i-1]; _12 = currLayer[i+1];
                _20 = currLayer[i+step-1]; _21 = currLayer[i+step]; _22 = currLayer[i+step+1];
                float vmin = fmin(fmin(fmin(_00,_01),fmin(_02,_10)),fmin(fmin(_12,_20),fmin(_21,_22)));
                if (val <= vmin) {
                    _00 = prevLayer[i-step-1]; _01 = prevLayer[i-step]; _02 = prevLayer[i-step+1];
                    _10 = prevLayer[i-1]; _12 = prevLayer[i+1];
                    _20 = prevLayer[i+step-1]; _21 = prevLayer[i+step]; _22 = prevLayer[i+step+1];
                    vmin = fmin(fmin(fmin(_00,_01),fmin(_02,_10)),fmin(fmin(_12,_20),fmin(_21,_22)));
                    if (val <= vmin) {
                        _00 = nextLayer[i-step-1]; _01 = nextLayer[i-step]; _02 = nextLayer[i-step+1];
                        _10 = nextLayer[i-1]; _12 = nextLayer[i+1];
                        _20 = nextLayer[i+step-1]; _21 = nextLayer[i+step]; _22 = nextLayer[i+step+1];
                        vmin = fmin(fmin(fmin(_00,_01),fmin(_02,_10)),fmin(fmin(_12,_20),fmin(_21,_22)));
                        if (val <= vmin) {
                            vmin = fmin(prevLayer[i], nextLayer[i]);
                            if (val <= vmin) {
                                isExtremum = true;
                            }
                        }
                    }
                }
            }
        }
    }

    // Only write if this is actually an extremum
    if (isExtremum) {
        // Sharded atomic with primitive root probing (256 shards)
        const uint NUM_SHARDS = 256;
        const uint PRIME = 257;
        const uint PRIMITIVE_ROOT = 127;
        const uint PROBES = 4;

        uint threadgroupID = tgid.y * ((params.width + 15) / 16) + tgid.x;
        uint shardIndex = threadgroupID % NUM_SHARDS;  // Initial: spatial locality

        for (uint probe = 0; probe < PROBES; probe++) {
            uint localIndex = atomic_fetch_add_explicit(&candidateCounts[shardIndex], 1, memory_order_relaxed);
            
            // Check if this shard has capacity
            if (localIndex < shardCapacity) {
                // Write candidate
                candidates[shardIndex * shardCapacity + localIndex] = y << 16 | x;
                return;
            } else {
                // Probe next shard using primitive root: (k + 1) * 127 % 257 - 1
                // This generates a pseudo-random permutation of 0..255
                shardIndex = ((shardIndex + 1) * PRIMITIVE_ROOT) % PRIME - 1;
            }
        }
    }
    
}

// ============================================================================
// Custom Gaussian Blur Kernels (OpenCV-inspired pattern)
// ============================================================================
// These kernels implement separable Gaussian convolution with the same
// accumulation pattern as OpenCV's hlineSmoothONa_yzy_a function:
// 1. Center tap first
// 2. Symmetric pairs: m[j]*left + m[j]*right (two multiplications per pair)
//
// Note: These produce ~3-6e-05 max error vs MPS but match OpenCV's approach

// Horizontal Gaussian blur pass
#pragma METAL fp math_mode(safe)
kernel void gaussianBlurHorizontal(
    const device float* source [[buffer(0)]],
    device float* destination [[buffer(1)]],
    constant GaussianBlurParams& params [[buffer(2)]],
    constant float* gaussKernel [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    int x = gid.x;
    int y = gid.y;

    if (x >= params.width || y >= params.height) {
        return;
    }

    int radius = params.kernelSize / 2;

    // Center tap first (matching OpenCV pattern)
    float centerPixel = source[y * params.rowStride + x];
    float sum = gaussKernel[radius] * centerPixel;

    // Symmetric pairs: m[j]*left + m[j]*right
    for (int j = 0; j < radius; j++) {
        int leftX = x - radius + j;
        int rightX = x + radius - j;

        // Border replication via clamp
        leftX = clamp(leftX, 0, params.width - 1);
        rightX = clamp(rightX, 0, params.width - 1);

        float leftPixel = source[y * params.rowStride + leftX];
        float rightPixel = source[y * params.rowStride + rightX];
        float weight = gaussKernel[j];

        // Match OpenCV: m[j]*left + m[j]*right (two muls, one add)
        sum = sum + (weight * leftPixel + weight * rightPixel);
    }

    destination[y * params.rowStride + x] = sum;
}

// Vertical Gaussian blur pass
#pragma METAL fp math_mode(safe)
kernel void gaussianBlurVertical(
    const device float* source [[buffer(0)]],
    device float* destination [[buffer(1)]],
    constant GaussianBlurParams& params [[buffer(2)]],
    constant float* gaussKernel [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    int x = gid.x;
    int y = gid.y;

    if (x >= params.width || y >= params.height) {
        return;
    }

    int radius = params.kernelSize / 2;

    // Center tap first (matching OpenCV pattern)
    float centerPixel = source[y * params.rowStride + x];
    float sum = gaussKernel[radius] * centerPixel;

    // Symmetric pairs: m[j]*top + m[j]*bottom
    for (int j = 0; j < radius; j++) {
        int topY = y - radius + j;
        int bottomY = y + radius - j;

        // Border replication via clamp
        topY = clamp(topY, 0, params.height - 1);
        bottomY = clamp(bottomY, 0, params.height - 1);

        float topPixel = source[topY * params.rowStride + x];
        float bottomPixel = source[bottomY * params.rowStride + x];
        float weight = gaussKernel[j];

        // Match OpenCV: m[j]*top + m[j]*bottom (two muls, one add)
        sum = sum + (weight * topPixel + weight * bottomPixel);
    }

    destination[y * params.rowStride + x] = sum;
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
        int srcY = tileY + py;

        // Clamp to image bounds (border replication)
        srcY = clamp(srcY, 0, params.height - 1);

        // Skip if X is out of bounds
        if (globalX >= params.width) {
            continue;
        }

        // Perform horizontal blur exactly like gaussianBlurHorizontal
        float centerPixel = source[srcY * params.rowStride + globalX];
        float sum = gaussKernel[radius] * centerPixel;

        // Symmetric pairs: m[j]*left + m[j]*right
        for (int j = 0; j < radius; j++) {
            int leftX = globalX - radius + j;
            int rightX = globalX + radius - j;

            // Border replication via clamp
            leftX = clamp(leftX, 0, params.width - 1);
            rightX = clamp(rightX, 0, params.width - 1);

            float leftPixel = source[srcY * params.rowStride + leftX];
            float rightPixel = source[srcY * params.rowStride + rightX];
            float weight = gaussKernel[j];

            sum = sum + (weight * leftPixel + weight * rightPixel);
        }

        // Store in shared memory
        sharedHoriz[py][localX] = sum;
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
    float sum = gaussKernel[radius] * centerPixel;

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

        sum = sum + (weight * topPixel + weight * bottomPixel);
    }

    // Write final result to global memory
    destination[globalY * params.rowStride + globalX] = sum;
}
