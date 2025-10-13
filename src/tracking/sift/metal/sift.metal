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
    float threshold;
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

        gauss += weight * topPixel + weight * bottomPixel;
    }

    nextGauss.write(gauss, uint2(x, y));
}

#pragma METAL fp math_mode(safe)
kernel void gaussianBlurVerticalAndDoG(
    texture2d<float, access::read> currGauss [[texture(0)]],
    texture2d<float, access::write> nextGauss [[texture(1)]],
    texture2d<float, access::write> nextDoG [[texture(2)]],
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

        gauss += weight * topPixel + weight * bottomPixel;
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