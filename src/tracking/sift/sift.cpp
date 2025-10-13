#include "lar/tracking/sift/sift.h"
#include "lar/tracking/sift/sift_common.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/core/hal/hal.hpp>
#include <opencv2/core/hal/intrin.hpp>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <chrono>

// Include SIFTMetal header when Metal is enabled
#ifdef LAR_USE_METAL_SIFT
#include "sift_metal.h"
#endif

// SIMD Helper macros for cleaner code
//#if (CV_SIMD || CV_SIMD_SCALABLE)
//   #define SIMD_ENABLED 1
//   using cv::v_float32;
//   using cv::v_int32;
//   using cv::v_uint16;
//   using cv::v_uint8;
//   using cv::VTraits;
//   
//   // Bring SIMD functions into scope
//   using cv::vx_load;
//   using cv::vx_load_aligned;
//   using cv::vx_store;
//   using cv::v_store;
//   using cv::v_store_aligned;
//   using cv::vx_setall_f32;
//   using cv::vx_setall_s32;
//   using cv::vx_setzero_s32;
//   using cv::vx_setzero_f32;
//   using cv::v_mul;
//   using cv::v_add;
//   using cv::v_sub;
//   using cv::v_fma;
//   using cv::v_round;
//   using cv::v_floor;
//   using cv::v_cvt_f32;
//   using cv::v_select;
//   using cv::v_ge;
//   using cv::v_lt;
//   using cv::v_gt;
//   using cv::v_le;
//   using cv::v_and;
//   using cv::v_or;
//   using cv::v_abs;
//   using cv::v_max;
//   using cv::v_min;
//   using cv::v_check_any;
//   using cv::v_signmask;
//   using cv::v_reduce_sum;
//   using cv::v_muladd;
//   using cv::v_pack_u;
//   using cv::v_pack_store;
//#else
//   #define SIMD_ENABLED 0
//#endif
 #define SIMD_ENABLED 0
 #define CV_SIMD 0
 #define CV_SIMD_SCALABLE 0

namespace lar {

// Use float for DoG pyramids
typedef float sift_wt;

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

// SIFT Implementation
SIFT::SIFT(const SIFTConfig& config)
    : config_(config)
#ifdef LAR_USE_METAL_SIFT
    , metalSift_(nullptr)
#endif
{
#ifdef LAR_USE_METAL_SIFT
    metalSift_ = std::make_unique<SIFTMetal>(config);
#endif
}

SIFT::~SIFT() = default;

SIFT::SIFT(SIFT&& other) noexcept
    : config_(other.config_)
#ifdef LAR_USE_METAL_SIFT
    , metalSift_(std::move(other.metalSift_))
#endif
{
}

SIFT& SIFT::operator=(SIFT&& other) noexcept {
    if (this != &other) {
        config_ = other.config_;
#ifdef LAR_USE_METAL_SIFT
        metalSift_ = std::move(other.metalSift_);
#endif
    }
    return *this;
}

int SIFT::descriptorSize() const {
    return SIFT_DESCR_WIDTH*SIFT_DESCR_WIDTH*SIFT_DESCR_HIST_BINS;
}

int SIFT::descriptorType() const {
    return config_.descriptorType;
}

int SIFT::defaultNorm() const {
    return cv::NORM_L2;
}

void SIFT::buildGaussianPyramid(const cv::Mat& base, std::vector<cv::Mat>& pyr, int nOctaves) const {
    std::vector<double> sig(config_.nOctaveLayers + 3);
    pyr.resize(nOctaves*(config_.nOctaveLayers + 3));

    sig[0] = config_.sigma;
    double k = std::pow(2., 1. / config_.nOctaveLayers);
    for (int i = 1; i < config_.nOctaveLayers + 3; i++) {
        double sig_prev = std::pow(k, (double)(i-1))*config_.sigma;
        double sig_total = sig_prev*k;
        sig[i] = std::sqrt(sig_total*sig_total - sig_prev*sig_prev);
    }

    for (int o = 0; o < nOctaves; o++) {
        for (int i = 0; i < config_.nOctaveLayers + 3; i++) {
            cv::Mat& dst = pyr[o*(config_.nOctaveLayers + 3) + i];
            if (o == 0 && i == 0) {
                dst = base;
            } else if (i == 0) {
                const cv::Mat& src = pyr[(o-1)*(config_.nOctaveLayers + 3) + config_.nOctaveLayers];
                cv::resize(src, dst, cv::Size(src.cols/2, src.rows/2), 0, 0, cv::INTER_NEAREST);
            } else {
                const cv::Mat& src = pyr[o*(config_.nOctaveLayers + 3) + i-1];
                cv::GaussianBlur(src, dst, cv::Size(), sig[i], sig[i]);
            }
        }
    }
}

void SIFT::buildDoGPyramid(const std::vector<cv::Mat>& gpyr, std::vector<cv::Mat>& dogpyr) const {
    int nOctaves = (int)gpyr.size()/(config_.nOctaveLayers + 3);
    dogpyr.resize(nOctaves*(config_.nOctaveLayers + 2));

    for (int o = 0; o < nOctaves; o++) {
        for (int i = 0; i < config_.nOctaveLayers + 2; i++) {
            const cv::Mat& src1 = gpyr[o*(config_.nOctaveLayers + 3) + i];
            const cv::Mat& src2 = gpyr[o*(config_.nOctaveLayers + 3) + i + 1];
            cv::Mat& dst = dogpyr[o*(config_.nOctaveLayers + 2) + i];
            cv::subtract(src2, src1, dst, cv::noArray(), CV_32F);
        }
    }
}

void SIFT::findScaleSpaceExtrema(const std::vector<cv::Mat>& gauss_pyr,
                                  const std::vector<cv::Mat>& dog_pyr,
                                  std::vector<cv::KeyPoint>& keypoints) const {
    const int nOctaves = (int)gauss_pyr.size()/(config_.nOctaveLayers + 3);
    const int threshold = cvFloor(0.5 * config_.contrastThreshold / config_.nOctaveLayers * 255 * SIFT_FIXPT_SCALE);

    keypoints.clear();

    for (int o = 0; o < nOctaves; o++) {
        for (int i = 1; i <= config_.nOctaveLayers; i++) {
            const int idx = o*(config_.nOctaveLayers+2)+i;
            const cv::Mat& img = dog_pyr[idx];
            const int step = (int)img.step1();
            const int rows = img.rows, cols = img.cols;

            findScaleSpaceExtremaInLayer(o, i, threshold, idx, step, cols,
                                        config_.nOctaveLayers, config_.contrastThreshold,
                                        config_.edgeThreshold, config_.sigma,
                                        gauss_pyr, dog_pyr, keypoints,
                                        cv::Range(SIFT_IMG_BORDER, rows-SIFT_IMG_BORDER));
        }
    }
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

    int firstOctave = config_.firstOctave();
    int nOctaves = config_.computeNumOctaves(image.cols, image.rows);

    #ifdef LAR_USE_METAL_SIFT
    metalSift_->detectAndCompute(image, keypoints, _descriptors, nOctaves);
    #else
    cv::Mat base = createInitialImage(image, firstOctave < 0, (float)config_.sigma);
    std::vector<cv::Mat> gpyr;
    // CPU+SIMD path: separate Gaussian pyramid, DoG pyramid, and extrema detection
    buildGaussianPyramid(base, gpyr, nOctaves);
    std::vector<cv::Mat> dogpyr;
    buildDoGPyramid(gpyr, dogpyr);
    findScaleSpaceExtrema(gpyr, dogpyr, keypoints);

    // Adjust keypoint positions for firstOctave = -1
    for (size_t i = 0; i < keypoints.size(); i++) {
        cv::KeyPoint& kpt = keypoints[i];
        float scale = 1.f/(float)(1 << -firstOctave);
        kpt.octave = (kpt.octave & ~255) | ((kpt.octave + firstOctave) & 255);
        kpt.pt *= scale;
        kpt.size *= scale;
    }
    
    if (_descriptors.needed()) {
        // CPU path: compute descriptors for all keypoints
        int dsize = descriptorSize();
        _descriptors.create((int)keypoints.size(), dsize, config_.descriptorType);

        if (keypoints.empty()) {
            // No keypoints found, descriptor matrix already created with 0 rows
            return;
        }

        cv::Mat descriptors = _descriptors.getMat();
        static const int d = SIFT_DESCR_WIDTH, n = SIFT_DESCR_HIST_BINS;

        for (size_t i = 0; i < keypoints.size(); i++) {
            cv::KeyPoint kpt = keypoints[i];
            int octave, layer;
            float scale;
            unpackOctave(kpt, octave, layer, scale);

            float size = kpt.size*scale;
            cv::Point2f ptf(kpt.pt.x*scale, kpt.pt.y*scale);
            const cv::Mat& img = gpyr[(octave - firstOctave)*(config_.nOctaveLayers + 3) + layer];

            float angle = 360.f - kpt.angle;
            if (std::abs(angle - 360.f) < FLT_EPSILON)
                angle = 0.f;

            calcSIFTDescriptor(img, ptf, angle, size*0.5f, d, n, descriptors, i);
        }
    }
#endif


}

} // namespace lar
