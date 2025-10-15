#include <metal_stdlib>

using namespace metal;

// Descriptor parameters
constant int SIFT_DESCR_WIDTH = 4;
constant int SIFT_DESCR_HIST_BINS = 8;
constant float SIFT_INIT_SIGMA = 0.5f;

// Image border for extrema detection
constant int SIFT_IMG_BORDER = 5;

// Keypoint refinement
constant int SIFT_MAX_INTERP_STEPS = 5;

// Orientation histogram parameters
constant int SIFT_ORI_HIST_BINS = 36;
constant float SIFT_ORI_SIG_FCTR = 1.5f;
constant float SIFT_ORI_RADIUS = 4.5f;
constant float SIFT_ORI_PEAK_RATIO = 0.8f;

// Descriptor computation parameters
constant float SIFT_DESCR_SCL_FCTR = 3.f;
constant float SIFT_DESCR_MAG_THR = 0.2f;
constant float SIFT_INT_DESCR_FCTR = 512.f;

struct ExtremaParams {
    int threshold;
    int border;
};

#pragma METAL fp math_mode(safe)
kernel void gaussianBlurHorizontal(
    texture2d<float, access::read> currGauss [[texture(0)]],
    texture2d<float, access::write> nextGauss [[texture(1)]],
    constant int& kernelSize [[buffer(0)]],
    constant float* gaussKernel [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint x = gid.x;
    uint y = gid.y;

    // Use texture dimensions for bounds checking
    if (x >= currGauss.get_width() || y >= currGauss.get_height()) return;

    int radius = kernelSize / 2;

    float centerPixel = currGauss.read(uint2(x, y)).r;
    float gauss = gaussKernel[radius] * centerPixel;

    for (int j = 0; j < radius; j++) {
        int leftX = int(x) - radius + j;
        int rightX = int(x) + radius - j;

        // Border replication via clamp
        leftX = clamp(leftX, 0, int(currGauss.get_width()) - 1);
        rightX = clamp(rightX, 0, int(currGauss.get_width()) - 1);

        float leftPixel = currGauss.read(uint2(leftX, y)).r;
        float rightPixel = currGauss.read(uint2(rightX, y)).r;
        float weight = gaussKernel[j];

        gauss = gauss + (weight * leftPixel + weight * rightPixel);
    }

    nextGauss.write(gauss, uint2(x, y));
}


#pragma METAL fp math_mode(safe)
kernel void gaussianBlurVertical(
    texture2d<float, access::read> currGauss [[texture(0)]],
    texture2d<float, access::write> nextGauss [[texture(1)]],
    constant int& kernelSize [[buffer(0)]],
    constant float* gaussKernel [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint x = gid.x;
    uint y = gid.y;

    if (x >= currGauss.get_width() || y >= currGauss.get_height()) return;

    int radius = kernelSize / 2;
    float centerPixel = currGauss.read(uint2(x, y)).r;
    float gauss = gaussKernel[radius] * centerPixel;

    for (int j = 0; j < radius; j++) {
        int topY = int(y) - radius + j;
        int bottomY = int(y) + radius - j;

        topY = clamp(topY, 0, int(currGauss.get_height()) - 1);
        bottomY = clamp(bottomY, 0, int(currGauss.get_height()) - 1);

        float topPixel = currGauss.read(uint2(x, topY)).r;
        float bottomPixel = currGauss.read(uint2(x, bottomY)).r;
        float weight = gaussKernel[j];

        gauss = gauss + (weight * topPixel + weight * bottomPixel);
    }

    nextGauss.write(gauss, uint2(x, y));
}

#pragma METAL fp math_mode(safe)
kernel void gaussianBlurVerticalAndDoG(
    texture2d<float, access::read> currGauss [[texture(0)]],
    texture2d<float, access::read> horizGauss [[texture(1)]],
    texture2d<float, access::write> nextGauss [[texture(2)]],
    texture2d<float, access::write> nextDoG [[texture(3)]],
    constant int& kernelSize [[buffer(0)]],
    constant float* gaussKernel [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint x = gid.x;
    uint y = gid.y;

    if (x >= horizGauss.get_width() || y >= horizGauss.get_height()) return;

    int radius = kernelSize / 2;
    float centerPixel = horizGauss.read(uint2(x, y)).r;
    float gauss = gaussKernel[radius] * centerPixel;

    for (int j = 0; j < radius; j++) {
        int topY = int(y) - radius + j;
        int bottomY = int(y) + radius - j;

        topY = clamp(topY, 0, int(horizGauss.get_height()) - 1);
        bottomY = clamp(bottomY, 0, int(horizGauss.get_height()) - 1);

        float topPixel = horizGauss.read(uint2(x, topY)).r;
        float bottomPixel = horizGauss.read(uint2(x, bottomY)).r;
        float weight = gaussKernel[j];

        gauss = gauss + weight * topPixel + weight * bottomPixel;
    }

    float currVal = currGauss.read(uint2(x, y)).r;
    nextGauss.write(gauss, uint2(x, y));
    nextDoG.write(gauss - currVal, uint2(x, y));
}

#pragma METAL fp math_mode(safe)
kernel void gaussianBlurAndDoGFused(
    texture2d<float, access::read> currGauss [[texture(0)]],
    texture2d<float, access::write> nextGauss [[texture(1)]],
    texture2d<float, access::write> nextDoG [[texture(2)]],
    constant int& kernelSize [[buffer(0)]],
    constant float* gaussKernel [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgSize [[threads_per_threadgroup]])
{
    const int TILE_SIZE = 16;
    int radius = kernelSize / 2;
    int paddedHeight = TILE_SIZE + 2 * radius;
    threadgroup float sharedHoriz[48][16];
    uint globalX = gid.x;
    uint globalY = gid.y;
    uint localX = tid.x;
    uint localY = tid.y;
    int tileY = (int(gid.y) / TILE_SIZE) * TILE_SIZE - radius;

    for (int py = localY; py < paddedHeight; py += tgSize.y) {
        int srcY = tileY + py;
        srcY = clamp(srcY, 0, int(currGauss.get_height()) - 1);

        if (globalX >= currGauss.get_width()) continue;
        
        float centerPixel = currGauss.read(uint2(globalX, srcY)).r;
        float gauss = gaussKernel[radius] * centerPixel;

        for (int j = 0; j < radius; j++) {
            int leftX = int(globalX) - radius + j;
            int rightX = int(globalX) + radius - j;
            leftX = clamp(leftX, 0, int(currGauss.get_width()) - 1);
            rightX = clamp(rightX, 0, int(currGauss.get_width()) - 1);

            float leftPixel = currGauss.read(uint2(leftX, srcY)).r;
            float rightPixel = currGauss.read(uint2(rightX, srcY)).r;
            float weight = gaussKernel[j];

            gauss = gauss + (weight * leftPixel + weight * rightPixel);
        }

        sharedHoriz[py][localX] = gauss;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (globalX >= currGauss.get_width() || globalY >= currGauss.get_height()) return;

    int sharedY = localY + radius;
    float centerPixel = sharedHoriz[sharedY][localX];
    float gauss = gaussKernel[radius] * centerPixel;

    for (int j = 0; j < radius; j++) {
        int topY = sharedY - radius + j;
        int bottomY = sharedY + radius - j;
        topY = clamp(topY, 0, 47);
        bottomY = clamp(bottomY, 0, 47);

        float topPixel = sharedHoriz[topY][localX];
        float bottomPixel = sharedHoriz[bottomY][localX];
        float weight = gaussKernel[j];

        gauss = gauss + (weight * topPixel + weight * bottomPixel);
    }

    float currVal = currGauss.read(uint2(globalX, globalY)).r;
    nextGauss.write(gauss, uint2(globalX, globalY));
    nextDoG.write(gauss - currVal, uint2(globalX, globalY));
}

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

    if (x < params.border || x >= currDoG.get_width() - params.border ||
        y < params.border || y >= currDoG.get_height() - params.border) return;

    // Read center pixel value
    float val = currDoG.read(uint2(x, y)).r;
    if (fabs(val) <= params.threshold) return;

    float _00,_01,_02;
    float _10,    _12;
    float _20,_21,_22;

    if (val > 0) {
        _00 = currDoG.read(uint2(x-1, y-1)).r; _01 = currDoG.read(uint2(x, y-1)).r; _02 = currDoG.read(uint2(x+1, y-1)).r;
        _10 = currDoG.read(uint2(x-1, y  )).r;                                      _12 = currDoG.read(uint2(x+1, y  )).r;
        _20 = currDoG.read(uint2(x-1, y+1)).r; _21 = currDoG.read(uint2(x, y+1)).r; _22 = currDoG.read(uint2(x+1, y+1)).r;
        float vmax = fmax(fmax(fmax(_00,_01),fmax(_02,_10)),fmax(fmax(_12,_20),fmax(_21,_22)));
        if (val < vmax) return;
        _00 = prevDoG.read(uint2(x-1, y-1)).r; _01 = prevDoG.read(uint2(x, y-1)).r; _02 = prevDoG.read(uint2(x+1, y-1)).r;
        _10 = prevDoG.read(uint2(x-1, y  )).r;                                      _12 = prevDoG.read(uint2(x+1, y  )).r;
        _20 = prevDoG.read(uint2(x-1, y+1)).r; _21 = prevDoG.read(uint2(x, y+1)).r; _22 = prevDoG.read(uint2(x+1, y+1)).r;
        vmax = fmax(fmax(fmax(_00,_01),fmax(_02,_10)),fmax(fmax(_12,_20),fmax(_21,_22)));
        if (val < vmax) return;
        _00 = nextDoG.read(uint2(x-1, y-1)).r; _01 = nextDoG.read(uint2(x, y-1)).r; _02 = nextDoG.read(uint2(x+1, y-1)).r;
        _10 = nextDoG.read(uint2(x-1, y  )).r;                                      _12 = nextDoG.read(uint2(x+1, y  )).r;
        _20 = nextDoG.read(uint2(x-1, y+1)).r; _21 = nextDoG.read(uint2(x, y+1)).r; _22 = nextDoG.read(uint2(x+1, y+1)).r;
        vmax = fmax(fmax(fmax(_00,_01),fmax(_02,_10)),fmax(fmax(_12,_20),fmax(_21,_22)));
        if (val < vmax) return;
        vmax = fmax(prevDoG.read(uint2(x, y)).r, nextDoG.read(uint2(x, y)).r);
        if (val < vmax) return;
    } else {
        _00 = currDoG.read(uint2(x-1, y-1)).r; _01 = currDoG.read(uint2(x, y-1)).r; _02 = currDoG.read(uint2(x+1, y-1)).r;
        _10 = currDoG.read(uint2(x-1, y  )).r;                                      _12 = currDoG.read(uint2(x+1, y  )).r;
        _20 = currDoG.read(uint2(x-1, y+1)).r; _21 = currDoG.read(uint2(x, y+1)).r; _22 = currDoG.read(uint2(x+1, y+1)).r;
        float vmin = fmin(fmin(fmin(_00,_01),fmin(_02,_10)),fmin(fmin(_12,_20),fmin(_21,_22)));
        if (val > vmin) return;

        _00 = prevDoG.read(uint2(x-1, y-1)).r; _01 = prevDoG.read(uint2(x, y-1)).r; _02 = prevDoG.read(uint2(x+1, y-1)).r;
        _10 = prevDoG.read(uint2(x-1, y  )).r;                                      _12 = prevDoG.read(uint2(x+1, y  )).r;
        _20 = prevDoG.read(uint2(x-1, y+1)).r; _21 = prevDoG.read(uint2(x, y+1)).r; _22 = prevDoG.read(uint2(x+1, y+1)).r;
        vmin = fmin(fmin(fmin(_00,_01),fmin(_02,_10)),fmin(fmin(_12,_20),fmin(_21,_22)));
        if (val > vmin) return;

        _00 = nextDoG.read(uint2(x-1, y-1)).r; _01 = nextDoG.read(uint2(x, y-1)).r; _02 = nextDoG.read(uint2(x+1, y-1)).r;
        _10 = nextDoG.read(uint2(x-1, y  )).r;                                      _12 = nextDoG.read(uint2(x+1, y  )).r;
        _20 = nextDoG.read(uint2(x-1, y+1)).r; _21 = nextDoG.read(uint2(x, y+1)).r; _22 = nextDoG.read(uint2(x+1, y+1)).r;
        vmin = fmin(fmin(fmin(_00,_01),fmin(_02,_10)),fmin(fmin(_12,_20),fmin(_21,_22)));
        if (val > vmin) return;

        vmin = fmin(prevDoG.read(uint2(x, y)).r, nextDoG.read(uint2(x, y)).r);
        if (val > vmin) return;
    }

    uint bitIndex = y * currDoG.get_width() + x;
    uint chunkIndex = bitIndex >> 5;      // Divide by 32
    uint bitOffset = bitIndex & 31;       // Modulo 32

    atomic_fetch_or_explicit(&extremaBitarray[chunkIndex], (1u << bitOffset), memory_order_relaxed);
}

#pragma METAL fp math_mode(safe)
kernel void resizeNearestNeighbor2x(
    texture2d<float, access::read> src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint dstX = gid.x;
    uint dstY = gid.y;

    if (dstX >= dst.get_width() || dstY >= dst.get_height()) return;

    uint srcX = dstX * 2;
    uint srcY = dstY * 2;
    srcX = min(srcX, src.get_width() - 1);
    srcY = min(srcY, src.get_height() - 1);

    float value = src.read(uint2(srcX, srcY)).r;
    dst.write(value, uint2(dstX, dstY));
}

// ============================================================================
// BUFFER-BASED GAUSSIAN BLUR KERNELS
// These write to buffers instead of textures to test L1 cache hypothesis
// ============================================================================

struct ImageDimensions {
    uint width;
    uint height;
    uint bytesPerRow;  // In bytes
};

#pragma METAL fp math_mode(safe)
kernel void gaussianBlurHorizontalBuffer(
    texture2d<float, access::read> currGauss [[texture(0)]],
    device float* nextGaussBuffer [[buffer(0)]],
    constant int& kernelSize [[buffer(1)]],
    constant float* gaussKernel [[buffer(2)]],
    constant ImageDimensions& dims [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint x = gid.x;
    uint y = gid.y;

    // Use texture dimensions for bounds checking
    if (x >= currGauss.get_width() || y >= currGauss.get_height()) return;

    int radius = kernelSize / 2;

    float centerPixel = currGauss.read(uint2(x, y)).r;
    float gauss = gaussKernel[radius] * centerPixel;

    for (int j = 0; j < radius; j++) {
        int leftX = int(x) - radius + j;
        int rightX = int(x) + radius - j;

        // Border replication via clamp
        leftX = clamp(leftX, 0, int(currGauss.get_width()) - 1);
        rightX = clamp(rightX, 0, int(currGauss.get_width()) - 1);

        float leftPixel = currGauss.read(uint2(leftX, y)).r;
        float rightPixel = currGauss.read(uint2(rightX, y)).r;
        float weight = gaussKernel[j];

        gauss = gauss + (weight * leftPixel + weight * rightPixel);
    }

    // Write to buffer: calculate linear index accounting for row padding
    uint floatsPerRow = dims.bytesPerRow / sizeof(float);
    uint bufferIndex = y * floatsPerRow + x;
    nextGaussBuffer[bufferIndex] = gauss;
}

#pragma METAL fp math_mode(safe)
kernel void gaussianBlurVerticalBuffer(
    device const float* currGaussBuffer [[buffer(0)]],
    device float* nextGaussBuffer [[buffer(1)]],
    constant int& kernelSize [[buffer(2)]],
    constant float* gaussKernel [[buffer(3)]],
    constant ImageDimensions& dims [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint x = gid.x;
    uint y = gid.y;

    if (x >= dims.width || y >= dims.height) return;

    int radius = kernelSize / 2;
    uint floatsPerRow = dims.bytesPerRow / sizeof(float);

    // Read center pixel from buffer
    uint centerIndex = y * floatsPerRow + x;
    float centerPixel = currGaussBuffer[centerIndex];
    float gauss = gaussKernel[radius] * centerPixel;

    for (int j = 0; j < radius; j++) {
        int topY = int(y) - radius + j;
        int bottomY = int(y) + radius - j;

        topY = clamp(topY, 0, int(dims.height) - 1);
        bottomY = clamp(bottomY, 0, int(dims.height) - 1);

        uint topIndex = topY * floatsPerRow + x;
        uint bottomIndex = bottomY * floatsPerRow + x;

        float topPixel = currGaussBuffer[topIndex];
        float bottomPixel = currGaussBuffer[bottomIndex];
        float weight = gaussKernel[j];

        gauss = gauss + (weight * topPixel + weight * bottomPixel);
    }

    // Write to buffer
    uint bufferIndex = y * floatsPerRow + x;
    nextGaussBuffer[bufferIndex] = gauss;
}

#pragma METAL fp math_mode(safe)
kernel void gaussianBlurVerticalAndDoGBuffer(
    texture2d<float, access::read> currGauss [[texture(0)]],
    device const float* horizGaussBuffer [[buffer(0)]],
    device float* nextGaussBuffer [[buffer(1)]],
    texture2d<float, access::write> nextDoG [[texture(1)]],
    constant int& kernelSize [[buffer(2)]],
    constant float* gaussKernel [[buffer(3)]],
    constant ImageDimensions& dims [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint x = gid.x;
    uint y = gid.y;

    if (x >= dims.width || y >= dims.height) return;

    int radius = kernelSize / 2;
    uint floatsPerRow = dims.bytesPerRow / sizeof(float);

    // Read center pixel from horizontal blur buffer
    uint centerIndex = y * floatsPerRow + x;
    float centerPixel = horizGaussBuffer[centerIndex];
    float gauss = gaussKernel[radius] * centerPixel;

    for (int j = 0; j < radius; j++) {
        int topY = int(y) - radius + j;
        int bottomY = int(y) + radius - j;

        topY = clamp(topY, 0, int(dims.height) - 1);
        bottomY = clamp(bottomY, 0, int(dims.height) - 1);

        uint topIndex = topY * floatsPerRow + x;
        uint bottomIndex = bottomY * floatsPerRow + x;

        float topPixel = horizGaussBuffer[topIndex];
        float bottomPixel = horizGaussBuffer[bottomIndex];
        float weight = gaussKernel[j];

        gauss = gauss + weight * topPixel + weight * bottomPixel;
    }

    // Read current Gaussian level from texture for DoG computation
    float currVal = currGauss.read(uint2(x, y)).r;

    // Write Gaussian to buffer
    uint bufferIndex = y * floatsPerRow + x;
    nextGaussBuffer[bufferIndex] = gauss;

    // Write DoG to texture (keeps DoG in texture cache)
    nextDoG.write(gauss - currVal, uint2(x, y));
}

#pragma METAL fp math_mode(safe)
kernel void gaussianBlurAndDoGFusedBuffer(
    texture2d<float, access::read> currGauss [[texture(0)]],
    device float* nextGaussBuffer [[buffer(0)]],
    texture2d<float, access::write> nextDoG [[texture(1)]],
    constant int& kernelSize [[buffer(1)]],
    constant float* gaussKernel [[buffer(2)]],
    constant ImageDimensions& dims [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgSize [[threads_per_threadgroup]])
{
    const int TILE_SIZE = 16;
    int radius = kernelSize / 2;
    int paddedHeight = TILE_SIZE + 2 * radius;
    threadgroup float sharedHoriz[48][16];
    uint globalX = gid.x;
    uint globalY = gid.y;
    uint localX = tid.x;
    uint localY = tid.y;
    int tileY = (int(gid.y) / TILE_SIZE) * TILE_SIZE - radius;

    // Horizontal blur into threadgroup memory
    for (int py = localY; py < paddedHeight; py += tgSize.y) {
        int srcY = tileY + py;
        srcY = clamp(srcY, 0, int(currGauss.get_height()) - 1);

        if (globalX >= currGauss.get_width()) continue;

        float centerPixel = currGauss.read(uint2(globalX, srcY)).r;
        float gauss = gaussKernel[radius] * centerPixel;

        for (int j = 0; j < radius; j++) {
            int leftX = int(globalX) - radius + j;
            int rightX = int(globalX) + radius - j;
            leftX = clamp(leftX, 0, int(currGauss.get_width()) - 1);
            rightX = clamp(rightX, 0, int(currGauss.get_width()) - 1);

            float leftPixel = currGauss.read(uint2(leftX, srcY)).r;
            float rightPixel = currGauss.read(uint2(rightX, srcY)).r;
            float weight = gaussKernel[j];

            gauss = gauss + (weight * leftPixel + weight * rightPixel);
        }

        sharedHoriz[py][localX] = gauss;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (globalX >= dims.width || globalY >= dims.height) return;

    // Vertical blur from threadgroup memory
    int sharedY = localY + radius;
    float centerPixel = sharedHoriz[sharedY][localX];
    float gauss = gaussKernel[radius] * centerPixel;

    for (int j = 0; j < radius; j++) {
        int topY = sharedY - radius + j;
        int bottomY = sharedY + radius - j;
        topY = clamp(topY, 0, 47);
        bottomY = clamp(bottomY, 0, 47);

        float topPixel = sharedHoriz[topY][localX];
        float bottomPixel = sharedHoriz[bottomY][localX];
        float weight = gaussKernel[j];

        gauss = gauss + (weight * topPixel + weight * bottomPixel);
    }

    // Read current Gaussian level from texture for DoG computation
    float currVal = currGauss.read(uint2(globalX, globalY)).r;

    // Write Gaussian result to buffer
    uint floatsPerRow = dims.bytesPerRow / sizeof(float);
    uint bufferIndex = globalY * floatsPerRow + globalX;
    nextGaussBuffer[bufferIndex] = gauss;

    // Write DoG to texture
    nextDoG.write(gauss - currVal, uint2(globalX, globalY));
}

// ============================================================================
// SIFT DESCRIPTOR COMPUTATION (GPU-ACCELERATED)
// ============================================================================

// ============================================================================
// OpenCV HAL Fast Math Approximations
// These MUST match OpenCV's implementations exactly for descriptor reproduction
// ============================================================================

// Fast atan2 approximation (OpenCV cv::hal::fastAtan2 polynomial coefficients)
constant float ATAN2_P1 = 0.9997878412794807f * (180.0f / M_PI_F);
constant float ATAN2_P3 = -0.3258083974640975f * (180.0f / M_PI_F);
constant float ATAN2_P5 = 0.1555786518463281f * (180.0f / M_PI_F);
constant float ATAN2_P7 = -0.04432655554792128f * (180.0f / M_PI_F);

/// Fast atan2 in degrees, matching OpenCV's fastAtan32f exactly
/// Returns angle in [0, 360) degrees
inline float fast_atan2_deg(float y, float x) {
    float ax = abs(x);
    float ay = abs(y);
    float a, c, c2;

    if (ax >= ay) {
        c = ay / (ax + FLT_EPSILON);  // DBL_EPSILON as float
        c2 = c * c;
        a = (((ATAN2_P7 * c2 + ATAN2_P5) * c2 + ATAN2_P3) * c2 + ATAN2_P1) * c;
    } else {
        c = ax / (ay + FLT_EPSILON);
        c2 = c * c;
        a = 90.0f - (((ATAN2_P7 * c2 + ATAN2_P5) * c2 + ATAN2_P3) * c2 + ATAN2_P1) * c;
    }

    if (x < 0.0f)
        a = 180.0f - a;
    if (y < 0.0f)
        a = 360.0f - a;

    return a;
}

// Note: For exp32f, OpenCV on macOS uses a polynomial + lookup table approximation.
// However, since we're computing exp of negative values (Gaussian weights) in a limited
// range, Metal's native exp() should have acceptable precision. If exact matching is
// required, we would need to port the full exp32f implementation with lookup tables.
// For now, we use native exp() and can verify descriptor matching empirically.

// Keypoint information for descriptor computation
struct KeypointInfo {
    float2 pt;              // Keypoint position in octave space (already scaled)
    uint32_t r;
    uint32_t c;
    float angle;                  // Orientation angle (degrees)
    float scale;                  // Scale in octave space
    float size;                  // Scale in octave space
    uint32_t gaussPyramidIndex;   // Which Gaussian pyramid texture to sample from
    uint32_t octave;
    float response;
};


// Configuration for descriptor computation
struct DescriptorConfig {
    int descriptorWidth;    // SIFT_DESCR_WIDTH (4)
    int histBins;          // SIFT_DESCR_HIST_BINS (8)
    float scaleFactor;     // SIFT_DESCR_SCL_FCTR (3.0)
    float magThreshold;    // SIFT_DESCR_MAG_THR (0.2)
    float intFactor;       // SIFT_INT_DESCR_FCTR (512.0)
};

#pragma METAL fp math_mode(safe)
kernel void computeSIFTDescriptors(
    // Input: Keypoint parameters
    constant KeypointInfo* keypoints [[buffer(0)]],
    constant uint& keypointCount [[buffer(1)]],

    // Configuration
    constant DescriptorConfig& config [[buffer(2)]],

    // Output: Descriptor matrix (each thread writes one row)
    device float* descriptors [[buffer(3)]],

    // Input: Gaussian pyramid textures (set via setTextures)
    texture2d<float, access::sample> gaussPyramid0 [[texture(0)]],
    texture2d<float, access::sample> gaussPyramid1 [[texture(1)]],
    texture2d<float, access::sample> gaussPyramid2 [[texture(2)]],
    texture2d<float, access::sample> gaussPyramid3 [[texture(3)]],
    texture2d<float, access::sample> gaussPyramid4 [[texture(4)]],
    texture2d<float, access::sample> gaussPyramid5 [[texture(5)]],
    // texture2d<float, access::sample> gaussPyramid6 [[texture(6)]],
    // texture2d<float, access::sample> gaussPyramid7 [[texture(7)]],
    // texture2d<float, access::sample> gaussPyramid8 [[texture(8)]],
    // texture2d<float, access::sample> gaussPyramid9 [[texture(9)]],
    // texture2d<float, access::sample> gaussPyramid10 [[texture(10)]],
    // texture2d<float, access::sample> gaussPyramid11 [[texture(11)]],
    // texture2d<float, access::sample> gaussPyramid12 [[texture(12)]],
    // texture2d<float, access::sample> gaussPyramid13 [[texture(13)]],
    // texture2d<float, access::sample> gaussPyramid14 [[texture(14)]],
    // texture2d<float, access::sample> gaussPyramid15 [[texture(15)]],
    // texture2d<float, access::sample> gaussPyramid16 [[texture(16)]],
    // texture2d<float, access::sample> gaussPyramid17 [[texture(17)]],
    // texture2d<float, access::sample> gaussPyramid18 [[texture(18)]],
    // texture2d<float, access::sample> gaussPyramid19 [[texture(19)]],
    // texture2d<float, access::sample> gaussPyramid20 [[texture(20)]],
    // texture2d<float, access::sample> gaussPyramid21 [[texture(21)]],
    // texture2d<float, access::sample> gaussPyramid22 [[texture(22)]],
    // texture2d<float, access::sample> gaussPyramid23 [[texture(23)]],

    uint gid [[thread_position_in_grid]])
{
    if (gid >= keypointCount) return;

    // Get keypoint info
    KeypointInfo kpt = keypoints[gid];

    // Skip invalid keypoints (angle = -1 marks unused orientation slots)
    if (kpt.angle < 0.0f) return;

    // Select correct Gaussian pyramid texture (manual switch since we can't use arrays)
    texture2d<float, access::sample> img;
    switch (kpt.gaussPyramidIndex) {
        case 0: img = gaussPyramid0; break;
        case 1: img = gaussPyramid1; break;
        case 2: img = gaussPyramid2; break;
        case 3: img = gaussPyramid3; break;
        case 4: img = gaussPyramid4; break;
        case 5: img = gaussPyramid5; break;
        // case 6: img = gaussPyramid6; break;
        // case 7: img = gaussPyramid7; break;
        // case 8: img = gaussPyramid8; break;
        // case 9: img = gaussPyramid9; break;
        // case 10: img = gaussPyramid10; break;
        // case 11: img = gaussPyramid11; break;
        // case 12: img = gaussPyramid12; break;
        // case 13: img = gaussPyramid13; break;
        // case 14: img = gaussPyramid14; break;
        // case 15: img = gaussPyramid15; break;
        // case 16: img = gaussPyramid16; break;
        // case 17: img = gaussPyramid17; break;
        // case 18: img = gaussPyramid18; break;
        // case 19: img = gaussPyramid19; break;
        // case 20: img = gaussPyramid20; break;
        // case 21: img = gaussPyramid21; break;
        // case 22: img = gaussPyramid22; break;
        // case 23: img = gaussPyramid23; break;
    }

    // Setup descriptor computation parameters (use constants directly)
    int d = SIFT_DESCR_WIDTH;        // 4
    int n = SIFT_DESCR_HIST_BINS;    // 8
    float hist_width = config.scaleFactor * kpt.scale;  // 3.0 * scale

    // Convert angle to radians and setup rotation
    // Note: kpt.angle has been flipped once by extractKeypoints (line 392),
    // then flipped again before being sent here (line 597), so it matches CPU's calcSIFTDescriptor input
    float ori_rad = kpt.angle * (M_PI_F / 180.0f);
    float cos_t = cos(ori_rad) / hist_width;
    float sin_t = sin(ori_rad) / hist_width;

    // Sampling parameters
    int radius = int(ceil(hist_width * 1.4142135f * (d + 1) * 0.5f));
    float exp_scale = -1.0f / (d * d * 0.5f);
    float bins_per_rad = n / 360.0f;

    // Allocate histogram in thread-private memory (padded for wrapping)
    // Use compile-time constant: (4 + 2) * (4 + 2) * (8 + 2) = 6 * 6 * 10 = 360
    const int hist_len = 360;
    float hist[hist_len];
    for (int i = 0; i < hist_len; i++) {
        hist[i] = 0.0f;
    }

    // Sampler for bilinear interpolation
    constexpr sampler s(coord::pixel, address::clamp_to_edge, filter::linear);

    // Stage 1-2: Sample gradients in rotated grid around keypoint
    for (int i = -radius; i <= radius; i++) {
        for (int j = -radius; j <= radius; j++) {
            // Rotate sampling coordinates
            float c_rot = j * cos_t - i * sin_t;
            float r_rot = j * sin_t + i * cos_t;

            // Compute descriptor bin positions
            float rbin = r_rot + d/2 - 0.5f;
            float cbin = c_rot + d/2 - 0.5f;

            // Check if sample is within descriptor bounds
            if (rbin > -1 && rbin < d && cbin > -1 && cbin < d) {
                // Sample position in image
                float2 sample_pos = kpt.pt + float2(j, i);

                // Bounds check
                if (sample_pos.x > 0 && sample_pos.x < img.get_width() - 1 &&
                    sample_pos.y > 0 && sample_pos.y < img.get_height() - 1) {

                    // Compute gradients using central differences
                    // Sample at integer pixel positions for gradient computation
                    // Use round() to match CPU's cvRound() instead of truncation
                    int px = int(round(sample_pos.x));
                    int py = int(round(sample_pos.y));

                    float dx = img.read(uint2(px + 1, py)).r - img.read(uint2(px - 1, py)).r;
                    float dy = img.read(uint2(px, py - 1)).r - img.read(uint2(px, py + 1)).r;

                    // Compute gradient magnitude and orientation using OpenCV HAL functions
                    float mag = sqrt(dx * dx + dy * dy);  // magnitude32f equivalent
                    float ori = fast_atan2_deg(dy, dx);   // fastAtan2 with angleInDegrees=true

                    // Gaussian weight based on distance from keypoint center
                    float weight = exp((c_rot * c_rot + r_rot * r_rot) * exp_scale);
                    float weighted_mag = mag * weight;

                    // Orientation bin (relative to keypoint orientation)
                    float obin = (ori - kpt.angle) * bins_per_rad;
                    if (obin < 0) obin += n;
                    if (obin >= n) obin -= n;

                    // Stage 3: Trilinear interpolation into histogram
                    // Compute integer bin indices and fractional parts
                    int r0 = int(floor(rbin));
                    int c0 = int(floor(cbin));
                    int o0 = int(floor(obin));

                    float rbin_frac = rbin - r0;
                    float cbin_frac = cbin - c0;
                    float obin_frac = obin - o0;

                    // Handle orientation wrapping
                    if (o0 < 0) o0 += n;
                    if (o0 >= n) o0 -= n;

                    // Trilinear interpolation weights
                    float v_r1 = weighted_mag * rbin_frac;
                    float v_r0 = weighted_mag - v_r1;

                    float v_rc11 = v_r1 * cbin_frac;
                    float v_rc10 = v_r1 - v_rc11;
                    float v_rc01 = v_r0 * cbin_frac;
                    float v_rc00 = v_r0 - v_rc01;

                    float v_rco111 = v_rc11 * obin_frac;
                    float v_rco110 = v_rc11 - v_rco111;
                    float v_rco101 = v_rc10 * obin_frac;
                    float v_rco100 = v_rc10 - v_rco101;
                    float v_rco011 = v_rc01 * obin_frac;
                    float v_rco010 = v_rc01 - v_rco011;
                    float v_rco001 = v_rc00 * obin_frac;
                    float v_rco000 = v_rc00 - v_rco001;

                    // Distribute into 8 neighboring bins (padded histogram)
                    int idx = ((r0 + 1) * (d + 2) + c0 + 1) * (n + 2) + o0;
                    hist[idx] += v_rco000;
                    hist[idx + 1] += v_rco001;
                    hist[idx + (n + 2)] += v_rco010;
                    hist[idx + (n + 3)] += v_rco011;
                    hist[idx + (d + 2) * (n + 2)] += v_rco100;
                    hist[idx + (d + 2) * (n + 2) + 1] += v_rco101;
                    hist[idx + (d + 3) * (n + 2)] += v_rco110;
                    hist[idx + (d + 3) * (n + 2) + 1] += v_rco111;
                }
            }
        }
    }

    // Stage 4: Finalize histogram (handle orientation wrapping)
    // Use compile-time constant: 4 * 4 * 8 = 128 for standard SIFT
    const int descriptor_len = 128;
    float raw_descriptor[descriptor_len];

    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            int idx = ((i + 1) * (d + 2) + (j + 1)) * (n + 2);

            // Add wrapped orientation bins
            hist[idx] += hist[idx + n];
            hist[idx + 1] += hist[idx + n + 1];

            // Copy to raw descriptor
            for (int k = 0; k < n; k++) {
                raw_descriptor[(i * d + j) * n + k] = hist[idx + k];
            }
        }
    }

    // Stage 5: Normalization with clamping
    // First normalization
    float nrm2 = 0.0f;
    for (int i = 0; i < descriptor_len; i++) {
        nrm2 += raw_descriptor[i] * raw_descriptor[i];
    }

    float threshold = sqrt(nrm2) * config.magThreshold;  // 0.2 * L2 norm

    // Clamp and re-normalize
    nrm2 = 0.0f;
    for (int i = 0; i < descriptor_len; i++) {
        float val = min(raw_descriptor[i], threshold);
        raw_descriptor[i] = val;
        nrm2 += val * val;
    }

    // Final normalization factor (use FLT_EPSILON to match CPU exactly)
    float norm_factor = config.intFactor / max(sqrt(nrm2), FLT_EPSILON);

    // Write final descriptor to output buffer
    device float* dst = descriptors + gid * descriptor_len;
    for (int i = 0; i < descriptor_len; i++) {
        dst[i] = raw_descriptor[i] * norm_factor;
    }
}

// ============================================================================
// ORIENTATION HISTOGRAM AND PEAK DETECTION (GPU-ACCELERATED)
// ============================================================================

constant int MAX_ORI_PEAKS = SIFT_ORI_HIST_BINS / 2;  // 18 max peaks

/// Combined orientation histogram computation and peak detection kernel
/// Each thread processes one extrema, computes its orientation histogram,
/// finds peaks, and writes oriented keypoints directly to the kpt_info buffer
#pragma METAL fp math_mode(safe)
kernel void computeOrientationHistogramsAndPeaks(
    // Input/Output: kpt_info buffer (pre-allocated with 18 slots per extrema)
    // This is the SAME buffer that will be read by descriptor kernel!
    device KeypointInfo* kpt_info [[buffer(0)]],
    constant uint& extremaCount [[buffer(1)]],        // Number of extrema (NOT total slots!)
    constant uint& extremaStride [[buffer(2)]],       // Stride between extrema (18)

    // Input: Gaussian pyramid textures
    texture2d<float, access::sample> gaussPyramid0 [[texture(0)]],
    texture2d<float, access::sample> gaussPyramid1 [[texture(1)]],
    texture2d<float, access::sample> gaussPyramid2 [[texture(2)]],
    texture2d<float, access::sample> gaussPyramid3 [[texture(3)]],
    texture2d<float, access::sample> gaussPyramid4 [[texture(4)]],
    texture2d<float, access::sample> gaussPyramid5 [[texture(5)]],

    uint gid [[thread_position_in_grid]])
{
    if (gid >= extremaCount) return;

    // Calculate index to first slot for this extrema
    uint baseIdx = gid * extremaStride;
    device KeypointInfo& baseInfo = kpt_info[baseIdx];

    // Select correct Gaussian pyramid texture (same pattern as descriptors)
    texture2d<float, access::sample> img;
    switch (baseInfo.gaussPyramidIndex) {
        case 0: img = gaussPyramid0; break;
        case 1: img = gaussPyramid1; break;
        case 2: img = gaussPyramid2; break;
        case 3: img = gaussPyramid3; break;
        case 4: img = gaussPyramid4; break;
        case 5: img = gaussPyramid5; break;
        default: return;  // Invalid index, skip this extrema
    }

    // ========================================================================
    // PART 1: Compute Orientation Histogram
    // ========================================================================

    // Compute parameters
    uint octave = baseInfo.octave & 255;
    float scl_octv = baseInfo.size * 0.5f / (1 << octave);
    int radius = int(round(SIFT_ORI_RADIUS * scl_octv));
    float sigma_val = SIFT_ORI_SIG_FCTR * scl_octv;
    float expf_scale = -1.0f / (2.0f * sigma_val * sigma_val);

    // Initialize histogram
    float hist[36];
    float temphist[40];  // 36 + 4 for smoothing wrap-around
    for (int i = 0; i < 36; i++) hist[i] = 0.0f;
    for (int i = 0; i < 40; i++) temphist[i] = 0.0f;

    // Sample gradients in circular region (matches calcOrientationHist)
    for (int i = -radius; i <= radius; i++) {
        int y = int(round(baseInfo.pt.y)) + i;
        if (y <= 0 || y >= img.get_height() - 1) continue;

        for (int j = -radius; j <= radius; j++) {
            int x = int(round(baseInfo.pt.x)) + j;
            if (x <= 0 || x >= img.get_width() - 1) continue;

            // Compute gradients using central differences
            float dx = img.read(uint2(x + 1, y)).r - img.read(uint2(x - 1, y)).r;
            float dy = img.read(uint2(x, y - 1)).r - img.read(uint2(x, y + 1)).r;

            // Magnitude and orientation
            float mag = sqrt(dx * dx + dy * dy);
            float ori = fast_atan2_deg(dy, dx);  // Use existing function

            // Gaussian weight
            float weight = exp((i * i + j * j) * expf_scale);
            float weighted_mag = mag * weight;

            // Accumulate into temporary histogram
            int bin = int(round((36.0f / 360.0f) * ori));
            if (bin >= 36) bin -= 36;
            if (bin < 0) bin += 36;
            temphist[bin + 2] += weighted_mag;
        }
    }

    // Smooth histogram (5-tap filter: [1, 4, 6, 4, 1] / 16)
    temphist[1] = temphist[37];  // n+1
    temphist[0] = temphist[36];  // n
    temphist[38] = temphist[2];  // n+2
    temphist[39] = temphist[3];  // n+3

    float omax = 0.0f;
    for (int i = 0; i < 36; i++) {
        hist[i] = (temphist[i] + temphist[i + 4]) * (1.0f / 16.0f) +
                  (temphist[i + 1] + temphist[i + 3]) * (4.0f / 16.0f) +
                  temphist[i + 2] * (6.0f / 16.0f);
        omax = max(omax, hist[i]);
    }

    // ========================================================================
    // PART 2: Peak Detection and Keypoint Creation
    // ========================================================================

    float mag_thr = omax * SIFT_ORI_PEAK_RATIO;
    float octaveScale = 1.f / (1 << octave);

    // Find peaks and write oriented keypoints to kpt_info slots
    for (int j = 0; j < 36; j++) {
        int l = j > 0 ? j - 1 : 35;
        int r = j < 35 ? j + 1 : 0;

        // Check if this is a local maximum above threshold
        if (hist[j] > hist[l] && hist[j] > hist[r] && hist[j] >= mag_thr) {
            // Parabolic interpolation for sub-bin accuracy
            float bin = j + 0.5f * (hist[l] - hist[r]) / (hist[l] - 2.0f * hist[j] + hist[r]);
            bin = bin < 0 ? 36 + bin : bin >= 36 ? bin - 36 : bin;
            float angle = (360.0f / 36.0f) * bin;

            // Write to appropriate slot
            device KeypointInfo& slot = kpt_info[baseIdx];
            baseIdx++;

            slot.pt.x *= octaveScale;
            slot.pt.y *= octaveScale;
            slot.scale = slot.size * 0.5f * octaveScale;
            slot.angle = angle;
        }
    }
}