// Shared utilities for SIFT implementations
// Contains coordinate space conversions and helper functions used across CPU, SIMD, and Metal variants
#include "lar/tracking/sift/sift_common.h"
#include "lar/tracking/sift/sift_constants.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/core/hal/hal.hpp>
#include <opencv2/core/hal/intrin.hpp>
#include <cmath>
#include <algorithm>

namespace lar {
typedef float sift_wt;

// Thread-local buffer for descriptor computation
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

static thread_local AlignedBuffer tls_buffer_common;

namespace sift_common {

// ============================================================================
// Gaussian Kernel Utilities
// ============================================================================

std::vector<float> createGaussianKernel(double sigma) {
    // Use OpenCV's getGaussianKernel for bit-exact compatibility
    int ksize = cvRound(sigma * 3) * 2 + 1;
    cv::Mat kernel = cv::getGaussianKernel(ksize, sigma, CV_32F);

    std::vector<float> result(ksize);
    for (int i = 0; i < ksize; i++) {
        result[i] = kernel.at<float>(i, 0);
    }
    return result;
}

// ============================================================================
// SIFT Descriptor Computation
// ============================================================================

void calcSIFTDescriptor(const cv::Mat& img, cv::Point2f ptf, float ori, float scl,
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
    float* X = tls_buffer_common.allocate(len * 6 + len_hist + len_ddn);
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

    // Trilinear interpolation (scalar fallback - SIMD disabled in current build)
    for (k = 0; k < len_left; k++) {
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

    // Normalize (scalar fallback)
    float nrm2 = 0;
    for (k = 0; k < len_ddn; k++)
        nrm2 += rawDst[k]*rawDst[k];

    float thr = std::sqrt(nrm2)*SIFT_DESCR_MAG_THR;

    nrm2 = 0;
    for (int i = 0; i < len_ddn; i++) {
        float val = std::min(rawDst[i], thr);
        rawDst[i] = val;
        nrm2 += val*val;
    }
    nrm2 = SIFT_INT_DESCR_FCTR/std::max(std::sqrt(nrm2), FLT_EPSILON);

    // Write to output
    if (dstMat.type() == CV_32F) {
        float* dst = dstMat.ptr<float>(row);
        for (k = 0; k < len_ddn; k++)
            dst[k] = cv::saturate_cast<float>(rawDst[k]*nrm2);
    } else { // CV_8U
        uint8_t* dst = dstMat.ptr<uint8_t>(row);
        for (k = 0; k < len_ddn; k++)
            dst[k] = cv::saturate_cast<uchar>(rawDst[k]*nrm2);
    }
}

// ============================================================================
// Keypoint Refinement Helpers
// ============================================================================

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

float calcOrientationHist(const cv::Mat& img, cv::Point pt, int radius,
                          float sigma, float* hist, int n) {
    int len = (radius*2+1)*(radius*2+1);
    float expf_scale = -1.f/(2.f * sigma * sigma);

    // Use thread-local aligned buffers
    float* X = tls_buffer_common.allocate(len * 4);
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

    // Histogram accumulation (scalar fallback)
    for (k = 0; k < len; k++) {
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

    for (int i = 0; i < n; i++) {
        hist[i] = (temphist[i] + temphist[i+4])*(1.f/16.f) +
                  (temphist[i+1] + temphist[i+3])*(4.f/16.f) +
                  temphist[i+2]*(6.f/16.f);
    }

    float maxval = hist[0];
    for (int i = 1; i < n; i++)
        maxval = std::max(maxval, hist[i]);

    return maxval;
}

} // namespace sift_common
} // namespace lar