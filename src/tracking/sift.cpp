#include "lar/tracking/sift.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/core/hal/hal.hpp>
#include <opencv2/core/hal/intrin.hpp>
#include <cmath>
#include <algorithm>
#include <iostream>

// Define LAR_USE_METAL_SIFT to enable Metal-accelerated Gaussian pyramid
// #define LAR_USE_METAL_SIFT

#ifdef LAR_USE_METAL_SIFT
// Forward declarations of Metal implementations (defined in sift_metal.mm)
namespace lar {
    void buildGaussianPyramidMetal(const cv::Mat& base, std::vector<cv::Mat>& pyr,
                                   int nOctaves, const std::vector<double>& sigmas);
    void buildDoGPyramidMetal(const std::vector<cv::Mat>& gauss_pyr, std::vector<cv::Mat>& dog_pyr,
                              int nOctaves, int nLevels);
    void findScaleSpaceExtremaMetal(const std::vector<cv::Mat>& gauss_pyr,
                                    const std::vector<cv::Mat>& dog_pyr,
                                    std::vector<cv::KeyPoint>& keypoints,
                                    int nOctaves, int nOctaveLayers, float threshold,
                                    double contrastThreshold, double edgeThreshold, double sigma);
#ifdef LAR_USE_METAL_SIFT_FUSED
    // void findScaleSpaceExtremaMetalFused(const std::vector<cv::Mat>& gauss_pyr,
    //                                      std::vector<cv::Mat>& dog_pyr,  // Non-const: populated by fused kernel
    //                                      std::vector<cv::KeyPoint>& keypoints,
    //                                      int nOctaves, int nOctaveLayers, float threshold,
    //                                      double contrastThreshold, double edgeThreshold, double sigma);
#endif
}
#endif

//// SIMD Helper macros for cleaner code
//#if (CV_SIMD || CV_SIMD_SCALABLE)
//    #define SIMD_ENABLED 1
//    using cv::v_float32;
//    using cv::v_int32;
//    using cv::v_uint16;
//    using cv::v_uint8;
//    using cv::VTraits;
//    
//    // Bring SIMD functions into scope
//    using cv::vx_load;
//    using cv::vx_load_aligned;
//    using cv::vx_store;
//    using cv::v_store;
//    using cv::v_store_aligned;
//    using cv::vx_setall_f32;
//    using cv::vx_setall_s32;
//    using cv::vx_setzero_s32;
//    using cv::vx_setzero_f32;
//    using cv::v_mul;
//    using cv::v_add;
//    using cv::v_sub;
//    using cv::v_fma;
//    using cv::v_round;
//    using cv::v_floor;
//    using cv::v_cvt_f32;
//    using cv::v_select;
//    using cv::v_ge;
//    using cv::v_lt;
//    using cv::v_gt;
//    using cv::v_le;
//    using cv::v_and;
//    using cv::v_or;
//    using cv::v_abs;
//    using cv::v_max;
//    using cv::v_min;
//    using cv::v_check_any;
//    using cv::v_signmask;
//    using cv::v_reduce_sum;
//    using cv::v_muladd;
//    using cv::v_pack_u;
//    using cv::v_pack_store;
//#else
//    #define SIMD_ENABLED 0
//#endif
#define SIMD_ENABLED 0
#define CV_SIMD 0
#define CV_SIMD_SCALABLE 0
// #define LAR_USE_METAL_SIFT 0

namespace lar {

// Constants
static const int SIFT_DESCR_WIDTH = 4;
static const int SIFT_DESCR_HIST_BINS = 8;
static const float SIFT_INIT_SIGMA = 0.5f;
static const int SIFT_IMG_BORDER = 5;
static const int SIFT_MAX_INTERP_STEPS = 5;
static const int SIFT_ORI_HIST_BINS = 36;
static const float SIFT_ORI_SIG_FCTR = 1.5f;
static const float SIFT_ORI_RADIUS = 4.5f;
static const float SIFT_ORI_PEAK_RATIO = 0.8f;
static const float SIFT_DESCR_SCL_FCTR = 3.f;
static const float SIFT_DESCR_MAG_THR = 0.2f;
static const float SIFT_INT_DESCR_FCTR = 512.f;

// Use float for DoG pyramids
typedef float sift_wt;
static const int SIFT_FIXPT_SCALE = 1;

// Simple aligned buffer allocator
class AlignedBuffer {
public:
    AlignedBuffer() : data_(nullptr), size_(0) {}
    ~AlignedBuffer() { if (data_) cv::fastFree(data_); }
    
    float* allocate(size_t size) {
        if (size > size_) {
            if (data_) cv::fastFree(data_);
            data_ = (float*)cv::fastMalloc(size * sizeof(float));
            size_ = size;
        }
        return data_;
    }
private:
    float* data_;
    size_t size_;
};

// Thread-local storage for aligned buffers
static thread_local AlignedBuffer tls_buffer1;
static thread_local AlignedBuffer tls_buffer2;
static thread_local AlignedBuffer tls_buffer3;

// Helper function to unpack octave information
static inline void unpackOctave(const cv::KeyPoint& kpt, int& octave, int& layer, float& scale) {
    octave = kpt.octave & 255;
    layer = (kpt.octave >> 8) & 255;
    octave = octave < 128 ? octave : (-128 | octave);
    scale = octave >= 0 ? 1.f/(1 << octave) : (float)(1 << -octave);
}

// Create initial image with upscaling and blurring
static cv::Mat createInitialImage(const cv::Mat& img, bool doubleImageSize, float sigma) {
    cv::Mat gray, gray_fpt;
    
    if (img.channels() == 3 || img.channels() == 4) {
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        gray.convertTo(gray_fpt, CV_32F, SIFT_FIXPT_SCALE, 0);
    } else {
        img.convertTo(gray_fpt, CV_32F, SIFT_FIXPT_SCALE, 0);
    }

    float sig_diff;

    if (doubleImageSize) {
        sig_diff = std::sqrt(std::max(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA * 4, 0.01f));
        
        cv::Mat dbl;
        resize(gray_fpt, dbl, cv::Size(gray_fpt.cols*2, gray_fpt.rows*2), 0, 0, cv::INTER_LINEAR);
        cv::Mat result;
        cv::GaussianBlur(dbl, result, cv::Size(), sig_diff, sig_diff);
        return result;
    } else {
        sig_diff = std::sqrt(std::max(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA, 0.01f));
        cv::Mat result;
        cv::GaussianBlur(gray_fpt, result, cv::Size(), sig_diff, sig_diff);
        return result;
    }
}

// Calculate orientation histogram with SIMD optimization
float calcOrientationHist(const cv::Mat& img, cv::Point pt, int radius,
                          float sigma, float* hist, int n) {
    int len = (radius*2+1)*(radius*2+1);
    float expf_scale = -1.f/(2.f * sigma * sigma);

    // Use thread-local aligned buffers
    float* X = tls_buffer1.allocate(len * 4);
    float* Y = X + len;
    float* W = Y + len;
    float* Ori = W + len;
    float* Mag = X; // Reuse X buffer
    
    std::vector<float> temphist(n+4, 0.f);
    
    int k = 0;
    for (int i = -radius; i <= radius; i++) {
        int y = pt.y + i;
        if (y <= 0 || y >= img.rows - 1) continue;
        
        for (int j = -radius; j <= radius; j++) {
            int x = pt.x + j;
            if (x <= 0 || x >= img.cols - 1) continue;

            float dx = img.at<sift_wt>(y, x+1) - img.at<sift_wt>(y, x-1);
            float dy = img.at<sift_wt>(y-1, x) - img.at<sift_wt>(y+1, x);

            X[k] = dx;
            Y[k] = dy;
            W[k] = (i*i + j*j) * expf_scale;
            k++;
        }
    }

    len = k;

    // Compute gradient values, orientations and weights
    cv::hal::exp32f(W, W, len);
    cv::hal::fastAtan2(Y, X, Ori, len, true);
    cv::hal::magnitude32f(X, Y, Mag, len);

    // Histogram accumulation with SIMD
    k = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    {
        const int vecsize = VTraits<v_float32>::vlanes();
        v_float32 nd360 = vx_setall_f32(n/360.f);
        v_int32 __n = vx_setall_s32(n);
        
        CV_DECL_ALIGNED(CV_SIMD_WIDTH) int bin_buf[16];
        CV_DECL_ALIGNED(CV_SIMD_WIDTH) float w_mul_mag_buf[16];

        for (; k <= len - vecsize; k += vecsize) {
            v_float32 w = vx_load_aligned(W + k);
            v_float32 mag = vx_load_aligned(Mag + k);
            v_float32 ori = vx_load_aligned(Ori + k);
            v_int32 bin = v_round(v_mul(nd360, ori));

            bin = v_select(v_ge(bin, __n), v_sub(bin, __n), bin);
            bin = v_select(v_lt(bin, vx_setzero_s32()), v_add(bin, __n), bin);

            w = v_mul(w, mag);
            v_store_aligned(bin_buf, bin);
            v_store_aligned(w_mul_mag_buf, w);
            
            for (int vi = 0; vi < vecsize; vi++) {
                temphist[bin_buf[vi] + 2] += w_mul_mag_buf[vi];
            }
        }
    }
#endif
    
    for (; k < len; k++) {
        int bin = cvRound((n/360.f) * Ori[k]);
        if (bin >= n) bin -= n;
        if (bin < 0) bin += n;
        temphist[bin + 2] += W[k] * Mag[k];
    }

    // Smooth the histogram
    temphist[1] = temphist[n+1];
    temphist[0] = temphist[n];
    temphist[n+2] = temphist[2];
    temphist[n+3] = temphist[3];

    int i = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    {
        v_float32 d_1_16 = vx_setall_f32(1.f/16.f);
        v_float32 d_4_16 = vx_setall_f32(4.f/16.f);
        v_float32 d_6_16 = vx_setall_f32(6.f/16.f);
        
        for (; i <= n - VTraits<v_float32>::vlanes(); i += VTraits<v_float32>::vlanes()) {
            v_float32 tn2 = vx_load_aligned(&temphist[i]);
            v_float32 tn1 = vx_load(&temphist[i+1]);
            v_float32 t0 = vx_load(&temphist[i+2]);
            v_float32 t1 = vx_load(&temphist[i+3]);
            v_float32 t2 = vx_load(&temphist[i+4]);
            v_float32 _hist = v_fma(v_add(tn2, t2), d_1_16,
                v_fma(v_add(tn1, t1), d_4_16, v_mul(t0, d_6_16)));
            v_store(hist + i, _hist);
        }
    }
#endif
    
    for (; i < n; i++) {
        hist[i] = (temphist[i] + temphist[i+4])*(1.f/16.f) +
                  (temphist[i+1] + temphist[i+3])*(4.f/16.f) +
                  temphist[i+2]*(6.f/16.f);
    }

    float maxval = hist[0];
    for (i = 1; i < n; i++)
        maxval = std::max(maxval, hist[i]);

    return maxval;
}

// Interpolate extremum location
bool adjustLocalExtrema(const std::vector<cv::Mat>& dog_pyr, cv::KeyPoint& kpt, int octv,
                        int& layer, int& r, int& c, int nOctaveLayers,
                        float contrastThreshold, float edgeThreshold, float sigma) {
    const float img_scale = 1.f/(255*SIFT_FIXPT_SCALE);
    const float deriv_scale = img_scale*0.5f;
    const float second_deriv_scale = img_scale;
    const float cross_deriv_scale = img_scale*0.25f;

    float xi=0, xr=0, xc=0, contr=0;
    int i = 0;

    for (; i < SIFT_MAX_INTERP_STEPS; i++) {
        int idx = octv*(nOctaveLayers+2) + layer;
        const cv::Mat& img = dog_pyr[idx];
        const cv::Mat& prev = dog_pyr[idx-1];
        const cv::Mat& next = dog_pyr[idx+1];

        cv::Vec3f dD((img.at<sift_wt>(r, c+1) - img.at<sift_wt>(r, c-1))*deriv_scale,
                     (img.at<sift_wt>(r+1, c) - img.at<sift_wt>(r-1, c))*deriv_scale,
                     (next.at<sift_wt>(r, c) - prev.at<sift_wt>(r, c))*deriv_scale);

        float v2 = img.at<sift_wt>(r, c)*2;
        float dxx = (img.at<sift_wt>(r, c+1) + img.at<sift_wt>(r, c-1) - v2)*second_deriv_scale;
        float dyy = (img.at<sift_wt>(r+1, c) + img.at<sift_wt>(r-1, c) - v2)*second_deriv_scale;
        float dss = (next.at<sift_wt>(r, c) + prev.at<sift_wt>(r, c) - v2)*second_deriv_scale;
        float dxy = (img.at<sift_wt>(r+1, c+1) - img.at<sift_wt>(r+1, c-1) -
                     img.at<sift_wt>(r-1, c+1) + img.at<sift_wt>(r-1, c-1))*cross_deriv_scale;
        float dxs = (next.at<sift_wt>(r, c+1) - next.at<sift_wt>(r, c-1) -
                     prev.at<sift_wt>(r, c+1) + prev.at<sift_wt>(r, c-1))*cross_deriv_scale;
        float dys = (next.at<sift_wt>(r+1, c) - next.at<sift_wt>(r-1, c) -
                     prev.at<sift_wt>(r+1, c) + prev.at<sift_wt>(r-1, c))*cross_deriv_scale;

        cv::Matx33f H(dxx, dxy, dxs,
                      dxy, dyy, dys,
                      dxs, dys, dss);

        cv::Vec3f X = H.solve(dD, cv::DECOMP_LU);

        xi = -X[2];
        xr = -X[1];
        xc = -X[0];

        if (std::abs(xi) < 0.5f && std::abs(xr) < 0.5f && std::abs(xc) < 0.5f)
            break;

        if (std::abs(xi) > (float)(INT_MAX/3) ||
            std::abs(xr) > (float)(INT_MAX/3) ||
            std::abs(xc) > (float)(INT_MAX/3))
            return false;

        c += cvRound(xc);
        r += cvRound(xr);
        layer += cvRound(xi);

        if (layer < 1 || layer > nOctaveLayers ||
            c < SIFT_IMG_BORDER || c >= img.cols - SIFT_IMG_BORDER ||
            r < SIFT_IMG_BORDER || r >= img.rows - SIFT_IMG_BORDER)
            return false;
    }

    if (i >= SIFT_MAX_INTERP_STEPS)
        return false;

    {
        int idx = octv*(nOctaveLayers+2) + layer;
        const cv::Mat& img = dog_pyr[idx];
        const cv::Mat& prev = dog_pyr[idx-1];
        const cv::Mat& next = dog_pyr[idx+1];
        
        cv::Matx31f dD((img.at<sift_wt>(r, c+1) - img.at<sift_wt>(r, c-1))*deriv_scale,
                       (img.at<sift_wt>(r+1, c) - img.at<sift_wt>(r-1, c))*deriv_scale,
                       (next.at<sift_wt>(r, c) - prev.at<sift_wt>(r, c))*deriv_scale);
        float t = dD.dot(cv::Matx31f(xc, xr, xi));

        contr = img.at<sift_wt>(r, c)*img_scale + t * 0.5f;
        if (std::abs(contr) * nOctaveLayers < contrastThreshold)
            return false;

        float v2 = img.at<sift_wt>(r, c)*2.f;
        float dxx = (img.at<sift_wt>(r, c+1) + img.at<sift_wt>(r, c-1) - v2)*second_deriv_scale;
        float dyy = (img.at<sift_wt>(r+1, c) + img.at<sift_wt>(r-1, c) - v2)*second_deriv_scale;
        float dxy = (img.at<sift_wt>(r+1, c+1) - img.at<sift_wt>(r+1, c-1) -
                     img.at<sift_wt>(r-1, c+1) + img.at<sift_wt>(r-1, c-1)) * cross_deriv_scale;
        float tr = dxx + dyy;
        float det = dxx * dyy - dxy * dxy;

        if (det <= 0 || tr*tr*edgeThreshold >= (edgeThreshold + 1)*(edgeThreshold + 1)*det)
            return false;
    }

    kpt.pt.x = (c + xc) * (1 << octv);
    kpt.pt.y = (r + xr) * (1 << octv);
    kpt.octave = octv + (layer << 8) + (cvRound((xi + 0.5)*255) << 16);
    kpt.size = sigma*powf(2.f, (layer + xi) / nOctaveLayers)*(1 << octv)*2;
    kpt.response = std::abs(contr);

    return true;
}

// Find extrema in one layer with SIMD optimization
static void findScaleSpaceExtremaInLayer(
    int o, int i, int threshold, int idx, int step, int cols, int nOctaveLayers,
    double contrastThreshold, double edgeThreshold, double sigma,
    const std::vector<cv::Mat>& gauss_pyr, const std::vector<cv::Mat>& dog_pyr,
    std::vector<cv::KeyPoint>& kpts, const cv::Range& range) {
    
    static const int n = SIFT_ORI_HIST_BINS;
    float hist[n];

    const cv::Mat& img = dog_pyr[idx];
    const cv::Mat& prev = dog_pyr[idx-1];
    const cv::Mat& next = dog_pyr[idx+1];

    for (int r = range.start; r < range.end; r++) {
        const sift_wt* currptr = img.ptr<sift_wt>(r);
        const sift_wt* prevptr = prev.ptr<sift_wt>(r);
        const sift_wt* nextptr = next.ptr<sift_wt>(r);
        int c = SIFT_IMG_BORDER;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        {
            const int vecsize = VTraits<v_float32>::vlanes();
            
            for (; c <= cols-SIFT_IMG_BORDER - vecsize; c += vecsize) {
                v_float32 val = vx_load(&currptr[c]);
                v_float32 _00,_01,_02;
                v_float32 _10,    _12;
                v_float32 _20,_21,_22;
                v_float32 vmin,vmax;

                v_float32 cond = v_gt(v_abs(val), vx_setall_f32((float)threshold));
                if (!v_check_any(cond))
                    continue;

                // Load 3x3 neighborhood in current layer
                _00 = vx_load(&currptr[c-step-1]); _01 = vx_load(&currptr[c-step]); _02 = vx_load(&currptr[c-step+1]);
                _10 = vx_load(&currptr[c     -1]);                                  _12 = vx_load(&currptr[c     +1]);
                _20 = vx_load(&currptr[c+step-1]); _21 = vx_load(&currptr[c+step]); _22 = vx_load(&currptr[c+step+1]);

                vmax = v_max(v_max(v_max(_00,_01),v_max(_02,_10)),v_max(v_max(_12,_20),v_max(_21,_22)));
                vmin = v_min(v_min(v_min(_00,_01),v_min(_02,_10)),v_min(v_min(_12,_20),v_min(_21,_22)));

                v_float32 condp = v_and(v_and(cond, v_gt(val, vx_setall_f32(0))), v_ge(val, vmax));
                v_float32 condm = v_and(v_and(cond, v_lt(val, vx_setall_f32(0))), v_le(val, vmin));

                cond = v_or(condp, condm);
                if (!v_check_any(cond))
                    continue;

                // Load 3x3 neighborhood in previous layer
                _00 = vx_load(&prevptr[c-step-1]); _01 = vx_load(&prevptr[c-step]); _02 = vx_load(&prevptr[c-step+1]);
                _10 = vx_load(&prevptr[c     -1]);                                  _12 = vx_load(&prevptr[c     +1]);
                _20 = vx_load(&prevptr[c+step-1]); _21 = vx_load(&prevptr[c+step]); _22 = vx_load(&prevptr[c+step+1]);

                vmax = v_max(v_max(v_max(_00,_01),v_max(_02,_10)),v_max(v_max(_12,_20),v_max(_21,_22)));
                vmin = v_min(v_min(v_min(_00,_01),v_min(_02,_10)),v_min(v_min(_12,_20),v_min(_21,_22)));

                condp = v_and(condp, v_ge(val, vmax));
                condm = v_and(condm, v_le(val, vmin));

                cond = v_or(condp, condm);
                if (!v_check_any(cond))
                    continue;

                v_float32 _11p = vx_load(&prevptr[c]);
                v_float32 _11n = vx_load(&nextptr[c]);

                v_float32 max_middle = v_max(_11n,_11p);
                v_float32 min_middle = v_min(_11n,_11p);

                // Load 3x3 neighborhood in next layer
                _00 = vx_load(&nextptr[c-step-1]); _01 = vx_load(&nextptr[c-step]); _02 = vx_load(&nextptr[c-step+1]);
                _10 = vx_load(&nextptr[c     -1]);                                  _12 = vx_load(&nextptr[c     +1]);
                _20 = vx_load(&nextptr[c+step-1]); _21 = vx_load(&nextptr[c+step]); _22 = vx_load(&nextptr[c+step+1]);

                vmax = v_max(v_max(v_max(_00,_01),v_max(_02,_10)),v_max(v_max(_12,_20),v_max(_21,_22)));
                vmin = v_min(v_min(v_min(_00,_01),v_min(_02,_10)),v_min(v_min(_12,_20),v_min(_21,_22)));

                condp = v_and(condp, v_ge(val, v_max(vmax, max_middle)));
                condm = v_and(condm, v_le(val, v_min(vmin, min_middle)));

                cond = v_or(condp, condm);
                if (!v_check_any(cond))
                    continue;

                int mask = v_signmask(cond);
                for (int k = 0; k < vecsize; k++) {
                    if ((mask & (1<<k)) == 0)
                        continue;

                    cv::KeyPoint kpt;
                    int r1 = r, c1 = c+k, layer = i;
                    if (!adjustLocalExtrema(dog_pyr, kpt, o, layer, r1, c1,
                                           nOctaveLayers, (float)contrastThreshold,
                                           (float)edgeThreshold, (float)sigma))
                        continue;
                        
                    float scl_octv = kpt.size*0.5f/(1 << o);
                    float omax = calcOrientationHist(gauss_pyr[o*(nOctaveLayers+3) + layer],
                                                    cv::Point(c1, r1),
                                                    cvRound(SIFT_ORI_RADIUS * scl_octv),
                                                    SIFT_ORI_SIG_FCTR * scl_octv,
                                                    hist, n);
                    float mag_thr = (float)(omax * SIFT_ORI_PEAK_RATIO);
                    
                    for (int j = 0; j < n; j++) {
                        int l = j > 0 ? j - 1 : n - 1;
                        int r2 = j < n-1 ? j + 1 : 0;

                        if (hist[j] > hist[l] && hist[j] > hist[r2] && hist[j] >= mag_thr) {
                            float bin = j + 0.5f * (hist[l]-hist[r2]) / (hist[l] - 2*hist[j] + hist[r2]);
                            bin = bin < 0 ? n + bin : bin >= n ? bin - n : bin;
                            kpt.angle = 360.f - (360.f/n) * bin;
                            if (std::abs(kpt.angle - 360.f) < FLT_EPSILON)
                                kpt.angle = 0.f;

                            kpts.push_back(kpt);
                        }
                    }
                }
            }
        }
#endif // CV_SIMD

        // Scalar fallback for remaining elements
        for (; c < cols-SIFT_IMG_BORDER; c++) {
            sift_wt val = currptr[c];
            if (std::abs(val) <= threshold)
                continue;

            sift_wt _00,_01,_02;
            sift_wt _10,    _12;
            sift_wt _20,_21,_22;
            
            _00 = currptr[c-step-1]; _01 = currptr[c-step]; _02 = currptr[c-step+1];
            _10 = currptr[c     -1];                        _12 = currptr[c     +1];
            _20 = currptr[c+step-1]; _21 = currptr[c+step]; _22 = currptr[c+step+1];

            bool calculate = false;
            if (val > 0) {
                sift_wt vmax = std::max(std::max(std::max(_00,_01),std::max(_02,_10)),
                                        std::max(std::max(_12,_20),std::max(_21,_22)));
                if (val >= vmax) {
                    _00 = prevptr[c-step-1]; _01 = prevptr[c-step]; _02 = prevptr[c-step+1];
                    _10 = prevptr[c     -1];                        _12 = prevptr[c     +1];
                    _20 = prevptr[c+step-1]; _21 = prevptr[c+step]; _22 = prevptr[c+step+1];
                    vmax = std::max(std::max(std::max(_00,_01),std::max(_02,_10)),
                                   std::max(std::max(_12,_20),std::max(_21,_22)));
                    if (val >= vmax) {
                        _00 = nextptr[c-step-1]; _01 = nextptr[c-step]; _02 = nextptr[c-step+1];
                        _10 = nextptr[c     -1];                        _12 = nextptr[c     +1];
                        _20 = nextptr[c+step-1]; _21 = nextptr[c+step]; _22 = nextptr[c+step+1];
                        vmax = std::max(std::max(std::max(_00,_01),std::max(_02,_10)),
                                       std::max(std::max(_12,_20),std::max(_21,_22)));
                        if (val >= vmax) {
                            sift_wt _11p = prevptr[c], _11n = nextptr[c];
                            calculate = (val >= std::max(_11p,_11n));
                        }
                    }
                }
            } else {
                sift_wt vmin = std::min(std::min(std::min(_00,_01),std::min(_02,_10)),
                                        std::min(std::min(_12,_20),std::min(_21,_22)));
                if (val <= vmin) {
                    _00 = prevptr[c-step-1]; _01 = prevptr[c-step]; _02 = prevptr[c-step+1];
                    _10 = prevptr[c     -1];                        _12 = prevptr[c     +1];
                    _20 = prevptr[c+step-1]; _21 = prevptr[c+step]; _22 = prevptr[c+step+1];
                    vmin = std::min(std::min(std::min(_00,_01),std::min(_02,_10)),
                                   std::min(std::min(_12,_20),std::min(_21,_22)));
                    if (val <= vmin) {
                        _00 = nextptr[c-step-1]; _01 = nextptr[c-step]; _02 = nextptr[c-step+1];
                        _10 = nextptr[c     -1];                        _12 = nextptr[c     +1];
                        _20 = nextptr[c+step-1]; _21 = nextptr[c+step]; _22 = nextptr[c+step+1];
                        vmin = std::min(std::min(std::min(_00,_01),std::min(_02,_10)),
                                       std::min(std::min(_12,_20),std::min(_21,_22)));
                        if (val <= vmin) {
                            sift_wt _11p = prevptr[c], _11n = nextptr[c];
                            calculate = (val <= std::min(_11p,_11n));
                        }
                    }
                }
            }

            if (calculate) {
                cv::KeyPoint kpt;
                int r1 = r, c1 = c, layer = i;
                if (!adjustLocalExtrema(dog_pyr, kpt, o, layer, r1, c1,
                                       nOctaveLayers, (float)contrastThreshold,
                                       (float)edgeThreshold, (float)sigma))
                    continue;
                    
                float scl_octv = kpt.size*0.5f/(1 << o);
                float omax = calcOrientationHist(gauss_pyr[o*(nOctaveLayers+3) + layer],
                                                cv::Point(c1, r1),
                                                cvRound(SIFT_ORI_RADIUS * scl_octv),
                                                SIFT_ORI_SIG_FCTR * scl_octv,
                                                hist, n);
                float mag_thr = (float)(omax * SIFT_ORI_PEAK_RATIO);
                
                for (int j = 0; j < n; j++) {
                    int l = j > 0 ? j - 1 : n - 1;
                    int r2 = j < n-1 ? j + 1 : 0;

                    if (hist[j] > hist[l] && hist[j] > hist[r2] && hist[j] >= mag_thr) {
                        float bin = j + 0.5f * (hist[l]-hist[r2]) / (hist[l] - 2*hist[j] + hist[r2]);
                        bin = bin < 0 ? n + bin : bin >= n ? bin - n : bin;
                        kpt.angle = 360.f - (360.f/n) * bin;
                        if (std::abs(kpt.angle - 360.f) < FLT_EPSILON)
                            kpt.angle = 0.f;

                        kpts.push_back(kpt);
                    }
                }
            }
        }
    }
}

// Calculate SIFT descriptor with SIMD optimization
static void calcSIFTDescriptor(const cv::Mat& img, cv::Point2f ptf, float ori, float scl,
                              int d, int n, cv::Mat& dstMat, int row) {
    cv::Point pt(cvRound(ptf.x), cvRound(ptf.y));
    float cos_t = cosf(ori*(float)(CV_PI/180));
    float sin_t = sinf(ori*(float)(CV_PI/180));
    float bins_per_rad = n / 360.f;
    float exp_scale = -1.f/(d * d * 0.5f);
    float hist_width = SIFT_DESCR_SCL_FCTR * scl;
    int radius = cvRound(hist_width * 1.4142135623730951f * (d + 1) * 0.5f);
    radius = std::min(radius, (int)std::sqrt((double)img.cols*img.cols + (double)img.rows*img.rows));
    cos_t /= hist_width;
    sin_t /= hist_width;

    int rows = img.rows, cols = img.cols;
    const int len = (radius*2+1)*(radius*2+1);
    const int len_hist = (d+2)*(d+2)*(n+2);
    const int len_ddn = d*d*n;
    
    // Use thread-local aligned buffers
    float* X = tls_buffer2.allocate(len * 6 + len_hist + len_ddn);
    float* Y = X + len;
    float* Ori = Y + len;
    float* W = Ori + len;
    float* RBin = W + len;
    float* CBin = RBin + len;
    float* hist = CBin + len;
    float* rawDst = hist + len_hist;
    float* Mag = Y; // Reuse Y buffer

    std::fill(hist, hist + len_hist, 0.f);

    int k = 0;
    for (int i = -radius; i <= radius; i++) {
        for (int j = -radius; j <= radius; j++) {
            float c_rot = j * cos_t - i * sin_t;
            float r_rot = j * sin_t + i * cos_t;
            float rbin = r_rot + d/2 - 0.5f;
            float cbin = c_rot + d/2 - 0.5f;
            int r = pt.y + i, c = pt.x + j;

            if (rbin > -1 && rbin < d && cbin > -1 && cbin < d &&
                r > 0 && r < rows - 1 && c > 0 && c < cols - 1) {
                float dx = img.at<sift_wt>(r, c+1) - img.at<sift_wt>(r, c-1);
                float dy = img.at<sift_wt>(r-1, c) - img.at<sift_wt>(r+1, c);
                X[k] = dx; Y[k] = dy; RBin[k] = rbin; CBin[k] = cbin;
                W[k] = (c_rot * c_rot + r_rot * r_rot)*exp_scale;
                k++;
            }
        }
    }

    int len_left = k;
    cv::hal::fastAtan2(Y, X, Ori, len_left, true);
    cv::hal::magnitude32f(X, Y, Mag, len_left);
    cv::hal::exp32f(W, W, len_left);

    // Trilinear interpolation with SIMD
    k = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    {
        const int vecsize = VTraits<v_float32>::vlanes();
        CV_DECL_ALIGNED(CV_SIMD_WIDTH) int idx_buf[16];
        CV_DECL_ALIGNED(CV_SIMD_WIDTH) float rco_buf[8*16];
        
        const v_float32 __ori  = vx_setall_f32(ori);
        const v_float32 __bins_per_rad = vx_setall_f32(bins_per_rad);
        const v_int32 __n = vx_setall_s32(n);
        const v_int32 __1 = vx_setall_s32(1);
        const v_int32 __d_plus_2 = vx_setall_s32(d+2);
        const v_int32 __n_plus_2 = vx_setall_s32(n+2);
        
        for (; k <= len_left - vecsize; k += vecsize) {
            v_float32 rbin = vx_load_aligned(RBin + k);
            v_float32 cbin = vx_load_aligned(CBin + k);
            v_float32 obin = v_mul(v_sub(vx_load_aligned(Ori + k), __ori), __bins_per_rad);
            v_float32 mag = v_mul(vx_load_aligned(Mag + k), vx_load_aligned(W + k));

            v_int32 r0 = v_floor(rbin);
            v_int32 c0 = v_floor(cbin);
            v_int32 o0 = v_floor(obin);
            rbin = v_sub(rbin, v_cvt_f32(r0));
            cbin = v_sub(cbin, v_cvt_f32(c0));
            obin = v_sub(obin, v_cvt_f32(o0));

            o0 = v_select(v_lt(o0, vx_setzero_s32()), v_add(o0, __n), o0);
            o0 = v_select(v_ge(o0, __n), v_sub(o0, __n), o0);

            v_float32 v_r1 = v_mul(mag, rbin), v_r0 = v_sub(mag, v_r1);
            v_float32 v_rc11 = v_mul(v_r1, cbin), v_rc10 = v_sub(v_r1, v_rc11);
            v_float32 v_rc01 = v_mul(v_r0, cbin), v_rc00 = v_sub(v_r0, v_rc01);
            v_float32 v_rco111 = v_mul(v_rc11, obin), v_rco110 = v_sub(v_rc11, v_rco111);
            v_float32 v_rco101 = v_mul(v_rc10, obin), v_rco100 = v_sub(v_rc10, v_rco101);
            v_float32 v_rco011 = v_mul(v_rc01, obin), v_rco010 = v_sub(v_rc01, v_rco011);
            v_float32 v_rco001 = v_mul(v_rc00, obin), v_rco000 = v_sub(v_rc00, v_rco001);

            v_int32 idx = v_muladd(v_muladd(v_add(r0, __1), __d_plus_2, v_add(c0, __1)), __n_plus_2, o0);
            v_store_aligned(idx_buf, idx);

            v_store_aligned(rco_buf,           v_rco000);
            v_store_aligned(rco_buf+vecsize,   v_rco001);
            v_store_aligned(rco_buf+vecsize*2, v_rco010);
            v_store_aligned(rco_buf+vecsize*3, v_rco011);
            v_store_aligned(rco_buf+vecsize*4, v_rco100);
            v_store_aligned(rco_buf+vecsize*5, v_rco101);
            v_store_aligned(rco_buf+vecsize*6, v_rco110);
            v_store_aligned(rco_buf+vecsize*7, v_rco111);

            for (int id = 0; id < vecsize; id++) {
                hist[idx_buf[id]] += rco_buf[id];
                hist[idx_buf[id]+1] += rco_buf[vecsize + id];
                hist[idx_buf[id]+(n+2)] += rco_buf[2*vecsize + id];
                hist[idx_buf[id]+(n+3)] += rco_buf[3*vecsize + id];
                hist[idx_buf[id]+(d+2)*(n+2)] += rco_buf[4*vecsize + id];
                hist[idx_buf[id]+(d+2)*(n+2)+1] += rco_buf[5*vecsize + id];
                hist[idx_buf[id]+(d+3)*(n+2)] += rco_buf[6*vecsize + id];
                hist[idx_buf[id]+(d+3)*(n+2)+1] += rco_buf[7*vecsize + id];
            }
        }
    }
#endif

    // Scalar fallback
    for (; k < len_left; k++) {
        float rbin = RBin[k], cbin = CBin[k];
        float obin = (Ori[k] - ori)*bins_per_rad;
        float mag = Mag[k]*W[k];

        int r0 = cvFloor(rbin);
        int c0 = cvFloor(cbin);
        int o0 = cvFloor(obin);
        rbin -= r0;
        cbin -= c0;
        obin -= o0;

        if (o0 < 0) o0 += n;
        if (o0 >= n) o0 -= n;

        float v_r1 = mag*rbin, v_r0 = mag - v_r1;
        float v_rc11 = v_r1*cbin, v_rc10 = v_r1 - v_rc11;
        float v_rc01 = v_r0*cbin, v_rc00 = v_r0 - v_rc01;
        float v_rco111 = v_rc11*obin, v_rco110 = v_rc11 - v_rco111;
        float v_rco101 = v_rc10*obin, v_rco100 = v_rc10 - v_rco101;
        float v_rco011 = v_rc01*obin, v_rco010 = v_rc01 - v_rco011;
        float v_rco001 = v_rc00*obin, v_rco000 = v_rc00 - v_rco001;

        int idx = ((r0+1)*(d+2) + c0+1)*(n+2) + o0;
        hist[idx] += v_rco000;
        hist[idx+1] += v_rco001;
        hist[idx+(n+2)] += v_rco010;
        hist[idx+(n+3)] += v_rco011;
        hist[idx+(d+2)*(n+2)] += v_rco100;
        hist[idx+(d+2)*(n+2)+1] += v_rco101;
        hist[idx+(d+3)*(n+2)] += v_rco110;
        hist[idx+(d+3)*(n+2)+1] += v_rco111;
    }

    // Finalize histogram
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            int idx = ((i+1)*(d+2) + (j+1))*(n+2);
            hist[idx] += hist[idx+n];
            hist[idx+1] += hist[idx+n+1];
            for (k = 0; k < n; k++)
                rawDst[(i*d + j)*n + k] = hist[idx+k];
        }
    }

    // Normalize with SIMD
    float nrm2 = 0;
    k = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    {
        v_float32 __nrm2 = vx_setzero_f32();
        for (; k <= len_ddn - VTraits<v_float32>::vlanes(); k += VTraits<v_float32>::vlanes()) {
            v_float32 __rawDst = vx_load_aligned(rawDst + k);
            __nrm2 = v_fma(__rawDst, __rawDst, __nrm2);
        }
        nrm2 = v_reduce_sum(__nrm2);
    }
#endif
    for (; k < len_ddn; k++)
        nrm2 += rawDst[k]*rawDst[k];

    float thr = std::sqrt(nrm2)*SIFT_DESCR_MAG_THR;
    
    nrm2 = 0;
    for (int i = 0; i < len_ddn; i++) {
        float val = std::min(rawDst[i], thr);
        rawDst[i] = val;
        nrm2 += val*val;
    }
    nrm2 = SIFT_INT_DESCR_FCTR/std::max(std::sqrt(nrm2), FLT_EPSILON);

    // Write to output with SIMD
    k = 0;
    if (dstMat.type() == CV_32F) {
        float* dst = dstMat.ptr<float>(row);
#if (CV_SIMD || CV_SIMD_SCALABLE)
        {
            v_float32 __min = vx_setzero_f32();
            v_float32 __max = vx_setall_f32(255.0f);
            v_float32 __nrm2 = vx_setall_f32(nrm2);
            for (; k <= len_ddn - VTraits<v_float32>::vlanes(); k += VTraits<v_float32>::vlanes()) {
                v_float32 __dst = vx_load_aligned(rawDst + k);
                __dst = v_min(v_max(v_cvt_f32(v_round(v_mul(__dst, __nrm2))), __min), __max);
                v_store(dst + k, __dst);
            }
        }
#endif
        for (; k < len_ddn; k++)
            dst[k] = cv::saturate_cast<uchar>(rawDst[k]*nrm2);
    } else { // CV_8U
        uint8_t* dst = dstMat.ptr<uint8_t>(row);
#if (CV_SIMD || CV_SIMD_SCALABLE)
        {
            v_float32 __nrm2 = vx_setall_f32(nrm2);
            for (; k <= len_ddn - VTraits<v_float32>::vlanes() * 2; k += VTraits<v_float32>::vlanes() * 2) {
                v_float32 __dst0 = vx_load_aligned(rawDst + k);
                v_float32 __dst1 = vx_load_aligned(rawDst + k + VTraits<v_float32>::vlanes());
                v_uint16 __pack01 = v_pack_u(v_round(v_mul(__dst0, __nrm2)), v_round(v_mul(__dst1, __nrm2)));
                v_pack_store(dst + k, __pack01);
            }
        }
#endif
        for (; k < len_ddn; k++)
            dst[k] = cv::saturate_cast<uchar>(rawDst[k]*nrm2);
    }
}

// SIFT Implementation
SIFT::SIFT(int nfeatures, int nOctaveLayers, double contrastThreshold,
           double edgeThreshold, double sigma, int descriptorType)
    : nfeatures_(nfeatures)
    , nOctaveLayers_(nOctaveLayers)
    , contrastThreshold_(contrastThreshold)
    , edgeThreshold_(edgeThreshold)
    , sigma_(sigma)
    , descriptorType_(descriptorType) {
}

cv::Ptr<SIFT> SIFT::create(int nfeatures, int nOctaveLayers, double contrastThreshold,
                            double edgeThreshold, double sigma, int descriptorType) {
    return cv::makePtr<SIFT>(nfeatures, nOctaveLayers, contrastThreshold,
                              edgeThreshold, sigma, descriptorType);
}

int SIFT::descriptorSize() const {
    return SIFT_DESCR_WIDTH*SIFT_DESCR_WIDTH*SIFT_DESCR_HIST_BINS;
}

int SIFT::descriptorType() const {
    return descriptorType_;
}

int SIFT::defaultNorm() const {
    return cv::NORM_L2;
}

void SIFT::buildGaussianPyramid(const cv::Mat& base, std::vector<cv::Mat>& pyr, int nOctaves) const {
    std::vector<double> sig(nOctaveLayers_ + 3);
    pyr.resize(nOctaves*(nOctaveLayers_ + 3));

    sig[0] = sigma_;
    double k = std::pow(2., 1. / nOctaveLayers_);
    for (int i = 1; i < nOctaveLayers_ + 3; i++) {
        double sig_prev = std::pow(k, (double)(i-1))*sigma_;
        double sig_total = sig_prev*k;
        sig[i] = std::sqrt(sig_total*sig_total - sig_prev*sig_prev);
    }

#ifdef LAR_USE_METAL_SIFT
    // Use Metal Performance Shaders for GPU-accelerated Gaussian pyramid
    buildGaussianPyramidMetal(base, pyr, nOctaves, sig);
#else
    // CPU+SIMD path (OpenCV GaussianBlur with hardware acceleration)
    for (int o = 0; o < nOctaves; o++) {
        for (int i = 0; i < nOctaveLayers_ + 3; i++) {
            cv::Mat& dst = pyr[o*(nOctaveLayers_ + 3) + i];
            if (o == 0 && i == 0) {
                dst = base;
            } else if (i == 0) {
                const cv::Mat& src = pyr[(o-1)*(nOctaveLayers_ + 3) + nOctaveLayers_];
                cv::resize(src, dst, cv::Size(src.cols/2, src.rows/2), 0, 0, cv::INTER_NEAREST);
            } else {
                const cv::Mat& src = pyr[o*(nOctaveLayers_ + 3) + i-1];
                cv::GaussianBlur(src, dst, cv::Size(), sig[i], sig[i]);
            }
        }
    }
#endif
}

void SIFT::buildDoGPyramid(const std::vector<cv::Mat>& gpyr, std::vector<cv::Mat>& dogpyr) const {
    int nOctaves = (int)gpyr.size()/(nOctaveLayers_ + 3);
    dogpyr.resize(nOctaves*(nOctaveLayers_ + 2));

#ifdef LAR_USE_METAL_SIFT
    // Use Metal Performance Shaders for GPU-accelerated DoG pyramid
    buildDoGPyramidMetal(gpyr, dogpyr, nOctaves, nOctaveLayers_ + 3);
#else
    for (int o = 0; o < nOctaves; o++) {
        for (int i = 0; i < nOctaveLayers_ + 2; i++) {
            const cv::Mat& src1 = gpyr[o*(nOctaveLayers_ + 3) + i];
            const cv::Mat& src2 = gpyr[o*(nOctaveLayers_ + 3) + i + 1];
            cv::Mat& dst = dogpyr[o*(nOctaveLayers_ + 2) + i];
            cv::subtract(src2, src1, dst, cv::noArray(), CV_32F);
        }
    }
#endif
}

void SIFT::findScaleSpaceExtrema(const std::vector<cv::Mat>& gauss_pyr,
                                  const std::vector<cv::Mat>& dog_pyr,
                                  std::vector<cv::KeyPoint>& keypoints) const {
    const int nOctaves = (int)gauss_pyr.size()/(nOctaveLayers_ + 3);
    const int threshold = cvFloor(0.5 * contrastThreshold_ / nOctaveLayers_ * 255 * SIFT_FIXPT_SCALE);

    keypoints.clear();

#ifdef LAR_USE_METAL_SIFT
    // Use Metal compute shader for GPU-accelerated extrema detection
    findScaleSpaceExtremaMetal(gauss_pyr, dog_pyr, keypoints, nOctaves, nOctaveLayers_,
                               (float)threshold, contrastThreshold_, edgeThreshold_, sigma_);
#else
    // CPU+SIMD path (per-layer processing with OpenCV intrinsics)
    for (int o = 0; o < nOctaves; o++) {
        for (int i = 1; i <= nOctaveLayers_; i++) {
            const int idx = o*(nOctaveLayers_+2)+i;
            const cv::Mat& img = dog_pyr[idx];
            const int step = (int)img.step1();
            const int rows = img.rows, cols = img.cols;

            findScaleSpaceExtremaInLayer(o, i, threshold, idx, step, cols,
                                        nOctaveLayers_, contrastThreshold_,
                                        edgeThreshold_, sigma_,
                                        gauss_pyr, dog_pyr, keypoints,
                                        cv::Range(SIFT_IMG_BORDER, rows-SIFT_IMG_BORDER));
        }
    }
#endif
}

void SIFT::detectAndCompute(cv::InputArray _image, cv::InputArray _mask,
                            std::vector<cv::KeyPoint>& keypoints,
                            cv::OutputArray _descriptors,
                            bool useProvidedKeypoints) {
    cv::Mat image = _image.getMat(), mask = _mask.getMat();

    if (image.empty() || image.depth() != CV_8U)
        CV_Error(cv::Error::StsBadArg, "image is empty or has incorrect depth (!=CV_8U)");

    if (!mask.empty() && mask.type() != CV_8UC1)
        CV_Error(cv::Error::StsBadArg, "mask has incorrect type (!=CV_8UC1)");

    int firstOctave = 0;
    
    cv::Mat base = createInitialImage(image, firstOctave < 0, (float)sigma_);
    std::vector<cv::Mat> gpyr;
    int nOctaves = cvRound(std::log((double)std::min(base.cols, base.rows)) / std::log(2.) - 2) - firstOctave;

    buildGaussianPyramid(base, gpyr, nOctaves);
    std::vector<cv::Mat> dogpyr;
    buildDoGPyramid(gpyr, dogpyr);
    findScaleSpaceExtrema(gpyr, dogpyr, keypoints);
    cv::KeyPointsFilter::removeDuplicatedSorted(keypoints);

    if( nfeatures_ > 0 ) {
        cv::KeyPointsFilter::retainBest(keypoints, nfeatures_);
        keypoints.resize(nfeatures_);
    }
        

    // Adjust keypoint positions for firstOctave = -1
    for (size_t i = 0; i < keypoints.size(); i++) {
        cv::KeyPoint& kpt = keypoints[i];
        float scale = 1.f/(float)(1 << -firstOctave);
        kpt.octave = (kpt.octave & ~255) | ((kpt.octave + firstOctave) & 255);
        kpt.pt *= scale;
        kpt.size *= scale;
    }

    if (!mask.empty())
        cv::KeyPointsFilter::runByPixelsMask(keypoints, mask);

    if (_descriptors.needed()) {
        int dsize = descriptorSize();
        _descriptors.create((int)keypoints.size(), dsize, descriptorType_);
        cv::Mat descriptors = _descriptors.getMat();

        static const int d = SIFT_DESCR_WIDTH, n = SIFT_DESCR_HIST_BINS;

        for (size_t i = 0; i < keypoints.size(); i++) {
            cv::KeyPoint kpt = keypoints[i];
            int octave, layer;
            float scale;
            unpackOctave(kpt, octave, layer, scale);
            
            float size = kpt.size*scale;
            cv::Point2f ptf(kpt.pt.x*scale, kpt.pt.y*scale);
            const cv::Mat& img = gpyr[(octave - firstOctave)*(nOctaveLayers_ + 3) + layer];

            float angle = 360.f - kpt.angle;
            if (std::abs(angle - 360.f) < FLT_EPSILON)
                angle = 0.f;
            
            calcSIFTDescriptor(img, ptf, angle, size*0.5f, d, n, descriptors, i);
        }
    }
}

} // namespace lar
