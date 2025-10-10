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
    int border;             // Border size (SIFT_IMG_BORDER = 5)
    int octave;             // Current octave index
    int layer;              // DoG layer being processed (1 to nOctaveLayers)
    int kernelSize;         // Gaussian kernel size for nextGauss
};

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


// Fused kernel: Blur → DoG → Extrema in single pass
// Processes 16x16 tiles with halo for Gaussian convolution
// Note: Extrema detection requires precise floating-point comparisons
// Compiled with -fno-fast-math to ensure IEEE-754 compliant comparisons
kernel void detectScaleSpaceExtremaFused(
    const device float* prevDoG [[buffer(0)]],             // DoG[layer-1] (pre-computed)
    const device float* currDoG [[buffer(1)]],             // DoG[layer] (detect extrema here)
    const device float* currGauss [[buffer(2)]],           // Gauss[layer+1] (blur to get Gauss[layer+2])
    device float* nextGauss [[buffer(3)]],                 // Output: Gauss[layer+2]
    device float* nextDoG [[buffer(4)]],                   // Output: DoG[layer+1]
    device atomic_uint* candidateCount [[buffer(5)]],      // Atomic counter
    device KeypointCandidate* candidates [[buffer(6)]],    // Output candidates
    constant FusedExtremaParams& params [[buffer(7)]],
    constant uint& maxCandidates [[buffer(8)]],
    constant float* gaussKernel [[buffer(9)]],             // 1D Gaussian kernel
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgSize [[threads_per_threadgroup]])
{
    // Tile configuration: 16x16 processing + vertical halo for Gaussian blur
    const int TILE_SIZE = 16;
    int radius = params.kernelSize / 2;
    int paddedHeight = TILE_SIZE + 2 * radius;  // Vertical halo for blur

    // Shared memory for tile processing
    // Max kernel size ~27 (sigma ~3.0), so max radius ~13, max paddedHeight = 16 + 26 = 42
    threadgroup float sharedHoriz[48][16];      // Horizontal blur result with vertical halo
    threadgroup float sharedNextDoG[18][18];    // DoG with 1-pixel halo for extrema detection

    int globalX = gid.x;
    int globalY = gid.y;
    int localX = tid.x;
    int localY = tid.y;

    // Calculate tile origin in Y dimension (X doesn't need tiling for horizontal blur)
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
        float sum = gaussKernel[radius] * centerPixel;

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

    // Compute DoG = nextGauss - currGauss
    float currG = currGauss[globalY * params.rowStride + globalX];
    float dogVal = sum - currG;

    // Store in sharedNextDoG at interior position (+1 offset for halo border)
    sharedNextDoG[localY + 1][localX + 1] = dogVal;

    // Write nextGauss to global memory
    nextGauss[globalY * params.rowStride + globalX] = sum;

    // Now fill the 1-pixel halo border by reading from pre-computed nextDoG
    // Each thread may fill multiple halo pixels
    const int HALO_SIZE = 18;
    int tileXOrigin = (globalX / TILE_SIZE) * TILE_SIZE;
    int tileYOrigin = (globalY / TILE_SIZE) * TILE_SIZE;

    // Fill top/bottom halo rows (y=0 and y=17)
    for (int hx = localX; hx < HALO_SIZE; hx += tgSize.x) {
        // Top row (hy = 0)
        int gx = tileXOrigin + hx - 1;
        int gy = tileYOrigin - 1;
        gx = clamp(gx, 0, params.width - 1);
        gy = clamp(gy, 0, params.height - 1);
        sharedNextDoG[0][hx] = nextDoG[gy * params.rowStride + gx];

        // Bottom row (hy = 17)
        gy = tileYOrigin + TILE_SIZE;
        gy = clamp(gy, 0, params.height - 1);
        sharedNextDoG[HALO_SIZE-1][hx] = nextDoG[gy * params.rowStride + gx];
    }

    // Fill left/right halo columns (x=0 and x=17, excluding corners already filled)
    for (int hy = localY + 1; hy < HALO_SIZE - 1; hy += tgSize.y) {
        // Left column (hx = 0)
        int gx = tileXOrigin - 1;
        int gy = tileYOrigin + hy - 1;
        gx = clamp(gx, 0, params.width - 1);
        gy = clamp(gy, 0, params.height - 1);
        sharedNextDoG[hy][0] = nextDoG[gy * params.rowStride + gx];

        // Right column (hx = 17)
        gx = tileXOrigin + TILE_SIZE;
        gx = clamp(gx, 0, params.width - 1);
        sharedNextDoG[hy][HALO_SIZE-1] = nextDoG[gy * params.rowStride + gx];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    

    // === Step 3: Detect extrema in currDoG using [prevDoG, currDoG, nextDoG] ===
    // Check if this is a border pixel or non-extremum (but don't return early!)
    bool shouldDetect = (globalX >= params.border && globalX < params.width - params.border &&
                         globalY >= params.border && globalY < params.height - params.border);

    bool isExtremum = false;

    if (shouldDetect) {
        // Read center value from currDoG (use global memory)
        float val = currDoG[globalY * params.rowStride + globalX];

        // Quick threshold rejection
        if (fabs(val) > params.threshold) {
            // Determine if looking for maxima or minima
            bool isMaxima = val > 0.0f;
            isExtremum = true;

            // Compare with 8 neighbors in currDoG (current layer)
            // Use global memory (prevDoG and currDoG are already computed in prior kernel launches)
            for (int dy = -1; dy <= 1 && isExtremum; dy++) {
                for (int dx = -1; dx <= 1 && isExtremum; dx++) {
                    if (dx == 0 && dy == 0) continue;

                    float neighbor = currDoG[(globalY + dy) * params.rowStride + (globalX + dx)];

                    if (isMaxima) {
                        if (val < neighbor) isExtremum = false;
                    } else {
                        if (val > neighbor) isExtremum = false;
                    }
                }
            }

            // Compare with 9 neighbors in prevDoG (use global memory)
            for (int dy = -1; dy <= 1 && isExtremum; dy++) {
                for (int dx = -1; dx <= 1 && isExtremum; dx++) {
                    float neighbor = prevDoG[(globalY + dy) * params.rowStride + (globalX + dx)];

                    if (isMaxima) {
                        if (val < neighbor) isExtremum = false;
                    } else {
                        if (val > neighbor) isExtremum = false;
                    }
                }
            }

            // Compare with 9 neighbors in nextDoG (use SHARED memory with halo)
            // Calculate shared memory indices (with +1 offset for halo border)
            int sharedCenterX = localX + 1;
            int sharedCenterY = localY + 1;

            for (int dy = -1; dy <= 1 && isExtremum; dy++) {
                for (int dx = -1; dx <= 1 && isExtremum; dx++) {
                    float neighbor = sharedNextDoG[sharedCenterY + dy][sharedCenterX + dx];

                    if (isMaxima) {
                        if (val < neighbor) isExtremum = false;
                    } else {
                        if (val > neighbor) isExtremum = false;
                    }
                }
            }

            // If survived all comparisons, this is a local extremum
            if (isExtremum) {
                // Atomically increment candidate count and get index
                uint index = atomic_fetch_add_explicit(candidateCount, 1, memory_order_relaxed);

                // Bounds check to prevent buffer overflow
                if (index < maxCandidates) {
                    candidates[index].x = globalX;
                    candidates[index].y = globalY;
                    candidates[index].octave = params.octave;
                    candidates[index].layer = params.layer;
                    candidates[index].value = val;
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