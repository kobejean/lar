// Metal compute shader for SIFT scale-space extrema detection
// Performs 3D local extrema detection across DoG pyramid layers
#include <metal_stdlib>
using namespace metal;

// Candidate keypoint structure
struct KeypointCandidate {
    int x;          // Column position
    int y;          // Row position
    int octave;     // Octave index
    int layer;      // Layer within octave (1 to nOctaveLayers)
    float value;    // DoG value at this location
};

// Kernel parameters
struct ExtremaParams {
    int width;              // Image width for this layer
    int height;             // Image height for this layer
    int rowStride;          // Row stride in floats (for aligned buffers)
    float threshold;        // Absolute threshold for extrema detection
    int border;             // Border size (SIFT_IMG_BORDER = 5)
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

// Helper: Sample DoG image with bounds checking
inline float sampleDoG(const device float* img, int x, int y, int width, int height, int rowStride) {
    if (x < 0 || x >= width || y < 0 || y >= height) {
        return 0.0f;
    }
    return img[y * rowStride + x];
}

// Kernel: Detect local extrema in 3D scale space (26-neighbor comparison)
// Each thread processes one pixel in the middle layer
// Note: Extrema detection requires precise floating-point comparisons
// Compiled with -fno-fast-math to ensure IEEE-754 compliant comparisons
kernel void detectScaleSpaceExtrema(
    const device float* prevLayer [[buffer(0)]],   // DoG layer i-1
    const device float* currLayer [[buffer(1)]],   // DoG layer i (center)
    const device float* nextLayer [[buffer(2)]],   // DoG layer i+1
    device atomic_uint* candidateCount [[buffer(3)]], // Atomic counter for candidates
    device KeypointCandidate* candidates [[buffer(4)]], // Output candidate array
    constant ExtremaParams& params [[buffer(5)]],
    constant uint& maxCandidates [[buffer(6)]],    // Maximum candidates to prevent overflow
    uint2 gid [[thread_position_in_grid]])
{
    int x = gid.x;
    int y = gid.y;

    // Skip border pixels
    if (x < params.border || x >= params.width - params.border ||
        y < params.border || y >= params.height - params.border) {
        return;
    }

    // Read center pixel value
    float val = currLayer[y * params.rowStride + x];

    // Quick threshold rejection
    if (fabs(val) <= params.threshold) {
        return;
    }

    // Determine if we're looking for maxima or minima
    bool isMaxima = val > 0.0f;

    // Compare with 8 neighbors in current layer
    bool isExtremum = true;

    // Current layer 3x3 neighborhood (8 neighbors, excluding center)
    for (int dy = -1; dy <= 1 && isExtremum; dy++) {
        for (int dx = -1; dx <= 1 && isExtremum; dx++) {
            if (dx == 0 && dy == 0) continue; // Skip center

            float neighbor = currLayer[(y + dy) * params.rowStride + (x + dx)];

            if (isMaxima) {
                if (val < neighbor) isExtremum = false;
            } else {
                if (val > neighbor) isExtremum = false;
            }
        }
    }

    if (!isExtremum) return;

    // Compare with 9 neighbors in previous layer
    for (int dy = -1; dy <= 1 && isExtremum; dy++) {
        for (int dx = -1; dx <= 1 && isExtremum; dx++) {
            float neighbor = prevLayer[(y + dy) * params.rowStride + (x + dx)];

            if (isMaxima) {
                if (val < neighbor) isExtremum = false;
            } else {
                if (val > neighbor) isExtremum = false;
            }
        }
    }

    if (!isExtremum) return;

    // Compare with 9 neighbors in next layer
    for (int dy = -1; dy <= 1 && isExtremum; dy++) {
        for (int dx = -1; dx <= 1 && isExtremum; dx++) {
            float neighbor = nextLayer[(y + dy) * params.rowStride + (x + dx)];

            if (isMaxima) {
                if (val < neighbor) isExtremum = false;
            } else {
                if (val > neighbor) isExtremum = false;
            }
        }
    }

    // If we survived all comparisons, this is a local extremum
    if (isExtremum) {
        // Atomically increment candidate count and get index
        uint index = atomic_fetch_add_explicit(candidateCount, 1, memory_order_relaxed);

        // Bounds check to prevent buffer overflow
        if (index < maxCandidates) {
            candidates[index].x = x;
            candidates[index].y = y;
            candidates[index].octave = params.octave;
            candidates[index].layer = params.layer;
            candidates[index].value = val;
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
