#ifndef LAR_TRACKING_SIFT_CONSTANTS_H
#define LAR_TRACKING_SIFT_CONSTANTS_H

namespace lar {

// SIFT algorithm constants
// These values are based on the original SIFT paper (Lowe 2004)
// and are shared between CPU and Metal GPU implementations

// Descriptor parameters
constexpr int SIFT_DESCR_WIDTH = 4;
constexpr int SIFT_DESCR_HIST_BINS = 8;
constexpr float SIFT_INIT_SIGMA = 0.5f;

// Image border for extrema detection
constexpr int SIFT_IMG_BORDER = 5;

// Keypoint refinement
constexpr int SIFT_MAX_INTERP_STEPS = 5;

// Orientation histogram parameters
constexpr int SIFT_ORI_HIST_BINS = 36;
constexpr float SIFT_ORI_SIG_FCTR = 1.5f;
constexpr float SIFT_ORI_RADIUS = 4.5f;
constexpr float SIFT_ORI_PEAK_RATIO = 0.8f;

// Descriptor computation parameters
constexpr float SIFT_DESCR_SCL_FCTR = 3.f;
constexpr float SIFT_DESCR_MAG_THR = 0.2f;
constexpr float SIFT_INT_DESCR_FCTR = 512.f;

// Fixed-point scale factor (1 = floating-point, higher values for fixed-point arithmetic)
constexpr int SIFT_FIXPT_SCALE = 1;

} // namespace lar

#endif // LAR_TRACKING_SIFT_CONSTANTS_H
