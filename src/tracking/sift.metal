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

// Helper: Sample DoG image with bounds checking
inline float sampleDoG(const device float* img, int x, int y, int width, int height, int rowStride) {
    if (x < 0 || x >= width || y < 0 || y >= height) {
        return 0.0f;
    }
    return img[y * rowStride + x];
}

// Kernel: Detect local extrema in 3D scale space (26-neighbor comparison)
// Each thread processes one pixel in the middle layer
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