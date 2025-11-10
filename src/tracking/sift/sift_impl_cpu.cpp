// CPU implementation of SIFT::Impl
// This file should only be compiled when LAR_USE_METAL_SIFT is NOT defined

#ifndef LAR_USE_METAL_SIFT

#include "lar/tracking/sift/sift.h"
#include "lar/tracking/sift/sift_common.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/core/hal/hal.hpp>
#include <opencv2/core/hal/intrin.hpp>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <chrono>

namespace lar {

struct SIFT::Impl {
    SIFTConfig config;

    explicit Impl(const SIFTConfig& cfg);
    ~Impl() = default;

    // Delete copy operations
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

    // Allow move operations
    Impl(Impl&& other) noexcept = default;
    Impl& operator=(Impl&& other) noexcept = default;

    void detectAndCompute(cv::InputArray image, cv::InputArray mask,
                         std::vector<cv::KeyPoint>& keypoints,
                         cv::OutputArray descriptors,
                         bool useProvidedKeypoints);

    int descriptorSize() const;
    int descriptorType() const;
    int defaultNorm() const;

private:
    void buildGaussianPyramid(const cv::Mat& base, std::vector<cv::Mat>& pyr, int nOctaves) const;
    void buildDoGPyramid(const std::vector<cv::Mat>& gpyr, std::vector<cv::Mat>& dogpyr) const;
    void findScaleSpaceExtrema(const std::vector<cv::Mat>& gauss_pyr,
                              const std::vector<cv::Mat>& dog_pyr,
                              std::vector<cv::KeyPoint>& keypoints) const;
};

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

static thread_local AlignedBuffer tls_buffer1;
static thread_local AlignedBuffer tls_buffer2;
static thread_local AlignedBuffer tls_buffer3;

static inline void unpackOctave(const cv::KeyPoint& kpt, int& octave, int& layer, float& scale) {
    octave = kpt.octave & 255;
    layer = (kpt.octave >> 8) & 255;
    octave = octave < 128 ? octave : (-128 | octave);
    scale = octave >= 0 ? 1.f/(1 << octave) : (float)(1 << -octave);
}

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
    int count = 0;

    for (int r = range.start; r < range.end; r++) {
        const float* currptr = img.ptr<float>(r);
        const float* prevptr = prev.ptr<float>(r);
        const float* nextptr = next.ptr<float>(r);
        int c = SIFT_IMG_BORDER;

        for (; c < cols-SIFT_IMG_BORDER; c++) {
            float val = currptr[c];
            if (std::abs(val) <= threshold)
                continue;

            float _00,_01,_02;
            float _10,    _12;
            float _20,_21,_22;

            _00 = currptr[c-step-1]; _01 = currptr[c-step]; _02 = currptr[c-step+1];
            _10 = currptr[c     -1];                        _12 = currptr[c     +1];
            _20 = currptr[c+step-1]; _21 = currptr[c+step]; _22 = currptr[c+step+1];

            bool calculate = false;
            if (val > 0) {
                float vmax = std::max(std::max(std::max(_00,_01),std::max(_02,_10)),
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
                            float _11p = prevptr[c], _11n = nextptr[c];
                            calculate = (val >= std::max(_11p,_11n));
                        }
                    }
                }
            } else {
                float vmin = std::min(std::min(std::min(_00,_01),std::min(_02,_10)),
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
                            float _11p = prevptr[c], _11n = nextptr[c];
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
                        count++;
                    }
                }
            }
        }
    }
    // std::cout << "octave " << o << " layer " << i << " added " << count << " keypoints" << std::endl;
}

SIFT::Impl::Impl(const SIFTConfig& cfg) : config(cfg)
{}

int SIFT::Impl::descriptorSize() const {
    return SIFT_DESCR_WIDTH*SIFT_DESCR_WIDTH*SIFT_DESCR_HIST_BINS;
}

int SIFT::Impl::descriptorType() const {
    return config.descriptorType;
}

int SIFT::Impl::defaultNorm() const {
    return cv::NORM_L2;
}

void SIFT::Impl::buildGaussianPyramid(const cv::Mat& base, std::vector<cv::Mat>& pyr, int nOctaves) const {
    std::vector<double> sig(config.nOctaveLayers + 3);
    pyr.resize(nOctaves*(config.nOctaveLayers + 3));

    sig[0] = config.sigma;
    double k = std::pow(2., 1. / config.nOctaveLayers);
    for (int i = 1; i < config.nOctaveLayers + 3; i++) {
        double sig_prev = std::pow(k, (double)(i-1))*config.sigma;
        double sig_total = sig_prev*k;
        sig[i] = std::sqrt(sig_total*sig_total - sig_prev*sig_prev);
    }

    for (int o = 0; o < nOctaves; o++) {
        for (int i = 0; i < config.nOctaveLayers + 3; i++) {
            cv::Mat& dst = pyr[o*(config.nOctaveLayers + 3) + i];
            if (o == 0 && i == 0) {
                dst = base;
            } else if (i == 0) {
                const cv::Mat& src = pyr[(o-1)*(config.nOctaveLayers + 3) + config.nOctaveLayers];
                cv::resize(src, dst, cv::Size(src.cols/2, src.rows/2), 0, 0, cv::INTER_NEAREST);
            } else {
                const cv::Mat& src = pyr[o*(config.nOctaveLayers + 3) + i-1];
                cv::GaussianBlur(src, dst, cv::Size(), sig[i], sig[i]);
            }
        }
    }
}

void SIFT::Impl::buildDoGPyramid(const std::vector<cv::Mat>& gpyr, std::vector<cv::Mat>& dogpyr) const {
    int nOctaves = (int)gpyr.size()/(config.nOctaveLayers + 3);
    dogpyr.resize(nOctaves*(config.nOctaveLayers + 2));

    for (int o = 0; o < nOctaves; o++) {
        for (int i = 0; i < config.nOctaveLayers + 2; i++) {
            const cv::Mat& src1 = gpyr[o*(config.nOctaveLayers + 3) + i];
            const cv::Mat& src2 = gpyr[o*(config.nOctaveLayers + 3) + i + 1];
            cv::Mat& dst = dogpyr[o*(config.nOctaveLayers + 2) + i];
            cv::subtract(src2, src1, dst, cv::noArray(), CV_32F);
        }
    }
}

void SIFT::Impl::findScaleSpaceExtrema(const std::vector<cv::Mat>& gauss_pyr,
                                  const std::vector<cv::Mat>& dog_pyr,
                                  std::vector<cv::KeyPoint>& keypoints) const {
    const int nOctaves = (int)gauss_pyr.size()/(config.nOctaveLayers + 3);

    keypoints.clear();

    for (int o = 0; o < nOctaves; o++) {
        for (int i = 1; i <= config.nOctaveLayers; i++) {
            const int idx = o*(config.nOctaveLayers+2)+i;
            const cv::Mat& img = dog_pyr[idx];
            const int step = (int)img.step1();
            const int rows = img.rows, cols = img.cols;

            findScaleSpaceExtremaInLayer(o, i, config.threshold, idx, step, cols,
                                        config.nOctaveLayers, config.contrastThreshold,
                                        config.edgeThreshold, config.sigma,
                                        gauss_pyr, dog_pyr, keypoints,
                                        cv::Range(SIFT_IMG_BORDER, rows-SIFT_IMG_BORDER));
        }
    }
}

void SIFT::Impl::detectAndCompute(cv::InputArray _image, cv::InputArray _mask,
                            std::vector<cv::KeyPoint>& keypoints,
                            cv::OutputArray _descriptors,
                            bool useProvidedKeypoints) {
    cv::Mat image = _image.getMat(), mask = _mask.getMat();

    if (image.empty() || image.depth() != CV_8U)
        CV_Error(cv::Error::StsBadArg, "image is empty or has incorrect depth (!=CV_8U)");

    if (!mask.empty() && mask.type() != CV_8UC1)
        CV_Error(cv::Error::StsBadArg, "mask has incorrect type (!=CV_8UC1)");

    cv::Mat base = createInitialImage(image, config.enableUpsampling, (float)config.sigma);
    std::vector<cv::Mat> gpyr;
    buildGaussianPyramid(base, gpyr, config.nOctaves);
    std::vector<cv::Mat> dogpyr;
    buildDoGPyramid(gpyr, dogpyr);
    findScaleSpaceExtrema(gpyr, dogpyr, keypoints);

    // Adjust keypoint positions for firstOctave = -1
    for (size_t i = 0; i < keypoints.size(); i++) {
        cv::KeyPoint& kpt = keypoints[i];
        float scale = 1.f/(float)(1 << -config.firstOctave);
        kpt.octave = (kpt.octave & ~255) | ((kpt.octave + config.firstOctave) & 255);
        kpt.pt *= scale;
        kpt.size *= scale;
    }

    if (_descriptors.needed()) {
        int dsize = descriptorSize();
        _descriptors.create((int)keypoints.size(), dsize, config.descriptorType);

        if (keypoints.empty()) return;

        cv::Mat descriptors = _descriptors.getMat();
        static const int d = SIFT_DESCR_WIDTH, n = SIFT_DESCR_HIST_BINS;

        for (size_t i = 0; i < keypoints.size(); i++) {
            cv::KeyPoint kpt = keypoints[i];
            int octave, layer;
            float scale;
            unpackOctave(kpt, octave, layer, scale);

            float size = kpt.size*scale;
            cv::Point2f ptf(kpt.pt.x*scale, kpt.pt.y*scale);
            const cv::Mat& img = gpyr[(octave - config.firstOctave)*(config.nOctaveLayers + 3) + layer];

            float angle = 360.f - kpt.angle;
            if (std::abs(angle - 360.f) < FLT_EPSILON)
                angle = 0.f;

            calcSIFTDescriptor(img, ptf, angle, size*0.5f, d, n, descriptors, i);
        }
    }
}

SIFT::SIFT(const SIFTConfig& config) : impl_(new Impl(config)) {}

SIFT::~SIFT() {
    delete impl_;
}

SIFT::SIFT(SIFT&& other) noexcept : impl_(other.impl_) {
    other.impl_ = nullptr;
}

SIFT& SIFT::operator=(SIFT&& other) noexcept {
    if (this != &other) {
        delete impl_;
        impl_ = other.impl_;
        other.impl_ = nullptr;
    }
    return *this;
}

int SIFT::descriptorSize() const {
    return impl_->descriptorSize();
}

int SIFT::descriptorType() const {
    return impl_->descriptorType();
}

int SIFT::defaultNorm() const {
    return impl_->defaultNorm();
}

void SIFT::detectAndCompute(cv::InputArray image, cv::InputArray mask,
                            std::vector<cv::KeyPoint>& keypoints,
                            cv::OutputArray descriptors,
                            bool useProvidedKeypoints) {
    impl_->detectAndCompute(image, mask, keypoints, descriptors, useProvidedKeypoints);
}

} // namespace lar

#endif // !LAR_USE_METAL_SIFT