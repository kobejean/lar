// SIFT algorithm constants for Metal shaders
// These values match the C++ constants in sift_constants.h
// Based on the original SIFT paper (Lowe 2004)

#ifndef SIFT_CONSTANTS_METAL_H
#define SIFT_CONSTANTS_METAL_H


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

#endif // SIFT_CONSTANTS_METAL_H
