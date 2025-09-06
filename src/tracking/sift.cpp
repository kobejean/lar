#include "lar/tracking/sift.h"
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace lar {

// Constructor
SIFT::SIFT(int nfeatures, int nOctaveLayers, double contrastThreshold,
           double edgeThreshold, double sigma, int descriptorType)
    : nfeatures_(nfeatures)
    , nOctaveLayers_(nOctaveLayers)
    , contrastThreshold_(contrastThreshold)
    , edgeThreshold_(edgeThreshold)
    , sigma_(sigma)
    , descriptorType_(descriptorType) {
}

// Factory method
cv::Ptr<SIFT> SIFT::create(int nfeatures, int nOctaveLayers, double contrastThreshold,
                            double edgeThreshold, double sigma, int descriptorType) {
    return cv::makePtr<SIFT>(nfeatures, nOctaveLayers, contrastThreshold,
                              edgeThreshold, sigma, descriptorType);
}

// Main detect and compute function
void SIFT::detectAndCompute(cv::InputArray image, cv::InputArray mask,
                             std::vector<cv::KeyPoint>& keypoints,
                             cv::OutputArray descriptors,
                             bool useProvidedKeypoints) {
    (void)mask;  // Suppress unused parameter warning
    cv::Mat img = image.getMat();
    if (img.empty()) {
        return;
    }

    // Convert to grayscale float if necessary
    cv::Mat gray;
    if (img.channels() > 1) {
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = img;
    }
    
    cv::Mat floatImg;
    gray.convertTo(floatImg, CV_32F, 1.0 / 255.0);

    if (!useProvidedKeypoints) {
        // Build scale-space pyramid
        ScaleSpacePyramid pyramid;
        buildScaleSpace(floatImg, pyramid);
        
        // Find keypoints
        std::vector<Keypoint> siftKeypoints;
        findScaleSpaceExtrema(pyramid, siftKeypoints);
        
        // Refine keypoints
        eliminateEdgeResponse(pyramid, siftKeypoints);
        
        // Assign orientations
        assignOrientations(pyramid, siftKeypoints);
        
        // Retain best features if limit specified
        if (nfeatures_ > 0 && static_cast<int>(siftKeypoints.size()) > nfeatures_) {
            retainBestFeatures(siftKeypoints);
        }
        
        // Convert to OpenCV keypoints
        keypoints.clear();
        keypoints.reserve(siftKeypoints.size());
        for (const auto& kpt : siftKeypoints) {
            keypoints.push_back(kpt.toOpenCV());
        }
        
        // Compute descriptors if requested
        if (descriptors.needed() && !keypoints.empty()) {
            computeDescriptors(pyramid, siftKeypoints, descriptors.getMatRef());
        }
    } else {
        // Use provided keypoints, just compute descriptors
        if (descriptors.needed() && !keypoints.empty()) {
            ScaleSpacePyramid pyramid;
            buildScaleSpace(floatImg, pyramid);
            
            // Convert OpenCV keypoints to internal format
            std::vector<Keypoint> siftKeypoints;
            for (const auto& kp : keypoints) {
                Keypoint skp;
                skp.x = kp.pt.x;
                skp.y = kp.pt.y;
                skp.scale = kp.size;
                skp.angle = kp.angle;
                skp.response = kp.response;
                skp.octave = kp.octave;
                skp.layer = (kp.octave >> 8) & 255;
                siftKeypoints.push_back(skp);
            }
            
            computeDescriptors(pyramid, siftKeypoints, descriptors.getMatRef());
        }
    }
}

// Detect only
void SIFT::detect(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints,
                  cv::InputArray mask) {
    cv::noArray();
    detectAndCompute(image, mask, keypoints, cv::noArray(), false);
}

// Compute descriptors only
void SIFT::compute(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints,
                   cv::OutputArray descriptors) {
    detectAndCompute(image, cv::noArray(), keypoints, descriptors, true);
}

// Build scale-space pyramid
void SIFT::buildScaleSpace(const cv::Mat& image, ScaleSpacePyramid& pyramid) {
    pyramid.build(image, nOctaveLayers_, sigma_);
}

// ScaleSpacePyramid implementation
void SIFT::ScaleSpacePyramid::build(const cv::Mat& image, int nOctaveLayers, double sigma) {
    // Calculate number of octaves
    int minDim = std::min(image.rows, image.cols);
    nOctaves = static_cast<int>(std::log2(minDim)) - 2;  // Leave at least 4x4 at top
    
    octaves.resize(nOctaves);
    DoG.resize(nOctaves);
    
    // First octave - potentially double the image
    cv::Mat base;
    if (SIFT_IMG_DBL) {
        cv::resize(image, base, cv::Size(), 2.0, 2.0, cv::INTER_LINEAR);
        sigma = std::sqrt(sigma * sigma - 4 * SIFT_INIT_SIGMA * SIFT_INIT_SIGMA);
    } else {
        base = image.clone();
        sigma = std::sqrt(sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA);
    }
    
    // Build each octave
    for (int o = 0; o < nOctaves; o++) {
        int nLayers = nOctaveLayers + 3;  // +3 for DoG computation
        octaves[o].resize(nLayers);
        DoG[o].resize(nLayers - 1);
        
        // First image in octave
        if (o == 0) {
            cv::GaussianBlur(base, octaves[o][0], cv::Size(), sigma);
        } else {
            // Downsample from previous octave (layer nOctaveLayers)
            cv::resize(octaves[o-1][nOctaveLayers], octaves[o][0], 
                      cv::Size(octaves[o-1][0].cols/2, octaves[o-1][0].rows/2),
                      0, 0, cv::INTER_NEAREST);
        }
        
        // Build Gaussian pyramid for this octave
        double k = std::pow(2.0, 1.0 / nOctaveLayers);
        for (int i = 1; i < nLayers; i++) {
            double sig = sigma * std::pow(k, i);
            cv::GaussianBlur(octaves[o][i-1], octaves[o][i], cv::Size(), sig);
            
            // Compute Difference of Gaussians
            if (i > 0) {
                cv::subtract(octaves[o][i], octaves[o][i-1], DoG[o][i-1]);
            }
        }
    }
}

// Find scale-space extrema
void SIFT::findScaleSpaceExtrema(const ScaleSpacePyramid& pyramid, 
                                  std::vector<Keypoint>& keypoints) {
    keypoints.clear();
    const double threshold = contrastThreshold_ / nOctaveLayers_;
    
    for (int o = 0; o < pyramid.nOctaves; o++) {
        for (int i = 1; i < static_cast<int>(pyramid.DoG[o].size()) - 1; i++) {  // Skip first and last
            const cv::Mat& prev = pyramid.DoG[o][i-1];
            const cv::Mat& curr = pyramid.DoG[o][i];
            const cv::Mat& next = pyramid.DoG[o][i+1];
            
            for (int r = 1; r < curr.rows - 1; r++) {
                for (int c = 1; c < curr.cols - 1; c++) {
                    float val = curr.at<float>(r, c);
                    
                    // Check if abs(val) > threshold
                    if (std::abs(val) <= threshold) {
                        continue;
                    }
                    
                    // Check if extremum (max or min among 26 neighbors)
                    bool isMax = true, isMin = true;
                    
                    // Check 26 neighbors
                    for (int di = -1; di <= 1 && (isMax || isMin); di++) {
                        const cv::Mat& layer = (di == -1) ? prev : (di == 0 ? curr : next);
                        for (int dr = -1; dr <= 1 && (isMax || isMin); dr++) {
                            for (int dc = -1; dc <= 1 && (isMax || isMin); dc++) {
                                if (di == 0 && dr == 0 && dc == 0) continue;
                                
                                float neighbor = layer.at<float>(r + dr, c + dc);
                                if (neighbor >= val) isMax = false;
                                if (neighbor <= val) isMin = false;
                            }
                        }
                    }
                    
                    if (isMax || isMin) {
                        Keypoint kpt;
                        kpt.octave = o;
                        kpt.layer = i;
                        kpt.x = c;
                        kpt.y = r;
                        kpt.response = val;
                        
                        // Refine keypoint position with sub-pixel accuracy
                        if (localizeKeypoint(pyramid, kpt)) {
                            keypoints.push_back(kpt);
                        }
                    }
                }
            }
        }
    }
}

// Localize keypoint with sub-pixel accuracy using Taylor expansion
bool SIFT::localizeKeypoint(const ScaleSpacePyramid& pyramid, Keypoint& kpt) {
    const int octave = kpt.octave;
    const int layer = kpt.layer;
    float xi = 0, xr = 0, xc = 0;
    int i = 0;
    
    for (; i < SIFT_MAX_INTERP_STEPS; i++) {
        const cv::Mat& img = pyramid.DoG[octave][layer];
        const cv::Mat& prev = pyramid.DoG[octave][layer - 1];
        const cv::Mat& next = pyramid.DoG[octave][layer + 1];
        
        int r = cvRound(kpt.y + xr);
        int c = cvRound(kpt.x + xc);
        
        // Check bounds
        if (r < 1 || r >= img.rows - 1 || c < 1 || c >= img.cols - 1) {
            return false;
        }
        
        // Compute gradient and Hessian
        float dx = (img.at<float>(r, c+1) - img.at<float>(r, c-1)) * 0.5f;
        float dy = (img.at<float>(r+1, c) - img.at<float>(r-1, c)) * 0.5f;
        float ds = (next.at<float>(r, c) - prev.at<float>(r, c)) * 0.5f;
        
        float dxx = img.at<float>(r, c+1) + img.at<float>(r, c-1) - 2 * img.at<float>(r, c);
        float dyy = img.at<float>(r+1, c) + img.at<float>(r-1, c) - 2 * img.at<float>(r, c);
        float dss = next.at<float>(r, c) + prev.at<float>(r, c) - 2 * img.at<float>(r, c);
        
        float dxy = (img.at<float>(r+1, c+1) - img.at<float>(r+1, c-1) -
                     img.at<float>(r-1, c+1) + img.at<float>(r-1, c-1)) * 0.25f;
        float dxs = (next.at<float>(r, c+1) - next.at<float>(r, c-1) -
                     prev.at<float>(r, c+1) + prev.at<float>(r, c-1)) * 0.25f;
        float dys = (next.at<float>(r+1, c) - next.at<float>(r-1, c) -
                     prev.at<float>(r+1, c) + prev.at<float>(r-1, c)) * 0.25f;
        
        // Solve for offset using Newton's method
        cv::Matx33f H(dxx, dxy, dxs,
                      dxy, dyy, dys,
                      dxs, dys, dss);
        cv::Vec3f g(dx, dy, ds);
        cv::Vec3f X = -H.inv() * g;
        
        xi = X[2];
        xr = X[1];
        xc = X[0];
        
        // Check if converged
        if (std::abs(xi) < 0.5f && std::abs(xr) < 0.5f && std::abs(xc) < 0.5f) {
            break;
        }
        
        // Update position
        kpt.x += xc;
        kpt.y += xr;
        kpt.layer += cvRound(xi);
        
        // Check layer bounds
        if (kpt.layer < 1 || kpt.layer > static_cast<int>(pyramid.DoG[octave].size()) - 2) {
            return false;
        }
    }
    
    // Check if we didn't converge
    if (i >= SIFT_MAX_INTERP_STEPS) {
        return false;
    }
    
    // Update keypoint
    kpt.x += xc;
    kpt.y += xr;
    
    // Calculate scale
    double scale = sigma_ * std::pow(2.0, octave + (kpt.layer + xi) / nOctaveLayers_);
    if (!SIFT_IMG_DBL) {
        scale *= 2;
    }
    kpt.scale = scale;
    
    // Update response (contrast)
    const cv::Mat& img = pyramid.DoG[octave][kpt.layer];
    int r = cvRound(kpt.y);
    int c = cvRound(kpt.x);
    
    float dx = (img.at<float>(r, c+1) - img.at<float>(r, c-1)) * 0.5f;
    float dy = (img.at<float>(r+1, c) - img.at<float>(r-1, c)) * 0.5f;
    float ds = (pyramid.DoG[octave][kpt.layer+1].at<float>(r, c) - 
                pyramid.DoG[octave][kpt.layer-1].at<float>(r, c)) * 0.5f;
    
    float contrast = img.at<float>(r, c) + 0.5f * (dx * xc + dy * xr + ds * xi);
    
    // Reject low contrast
    if (std::abs(contrast) < contrastThreshold_) {
        return false;
    }
    
    kpt.response = std::abs(contrast);
    return true;
}

// Eliminate edge responses using principal curvature
void SIFT::eliminateEdgeResponse(const ScaleSpacePyramid& pyramid, 
                                  std::vector<Keypoint>& keypoints) {
    std::vector<Keypoint> filtered;
    const double edgeThreshold = (edgeThreshold_ + 1) * (edgeThreshold_ + 1) / edgeThreshold_;
    
    for (const auto& kpt : keypoints) {
        const cv::Mat& img = pyramid.DoG[kpt.octave][kpt.layer];
        int r = cvRound(kpt.y);
        int c = cvRound(kpt.x);
        
        // Compute 2x2 Hessian matrix
        float dxx = img.at<float>(r, c+1) + img.at<float>(r, c-1) - 2 * img.at<float>(r, c);
        float dyy = img.at<float>(r+1, c) + img.at<float>(r-1, c) - 2 * img.at<float>(r, c);
        float dxy = (img.at<float>(r+1, c+1) - img.at<float>(r+1, c-1) -
                     img.at<float>(r-1, c+1) + img.at<float>(r-1, c-1)) * 0.25f;
        
        float trace = dxx + dyy;
        float det = dxx * dyy - dxy * dxy;
        
        // Check ratio of principal curvatures
        if (det > 0 && trace * trace / det < edgeThreshold) {
            filtered.push_back(kpt);
        }
    }
    
    keypoints = filtered;
}

// Assign orientations to keypoints
void SIFT::assignOrientations(const ScaleSpacePyramid& pyramid, 
                               std::vector<Keypoint>& keypoints) {
    std::vector<Keypoint> oriented;
    
    for (auto& kpt : keypoints) {
        const cv::Mat& img = pyramid.octaves[kpt.octave][kpt.layer];
        float scale = kpt.scale;
        int radius = cvRound(SIFT_ORI_SIG_FCTR * scale);
        float sigma = SIFT_ORI_SIG_FCTR * scale;
        
        // Build orientation histogram
        std::vector<float> hist(SIFT_ORI_HIST_BINS, 0);
        
        for (int dr = -radius; dr <= radius; dr++) {
            for (int dc = -radius; dc <= radius; dc++) {
                int r = cvRound(kpt.y) + dr;
                int c = cvRound(kpt.x) + dc;
                
                if (r <= 0 || r >= img.rows - 1 || c <= 0 || c >= img.cols - 1) {
                    continue;
                }
                
                float dx = img.at<float>(r, c+1) - img.at<float>(r, c-1);
                float dy = img.at<float>(r+1, c) - img.at<float>(r-1, c);
                
                float mag = std::sqrt(dx * dx + dy * dy);
                float ori = std::atan2(dy, dx);
                
                float weight = std::exp(-(dr * dr + dc * dc) / (2 * sigma * sigma));
                
                int bin = cvRound((ori + CV_PI) * SIFT_ORI_HIST_BINS / (2 * CV_PI));
                bin = std::min(std::max(bin, 0), static_cast<int>(SIFT_ORI_HIST_BINS - 1));
                
                hist[bin] += weight * mag;
            }
        }
        
        // Smooth histogram
        std::vector<float> smooth_hist(SIFT_ORI_HIST_BINS);
        for (int i = 0; i < SIFT_ORI_HIST_BINS; i++) {
            smooth_hist[i] = hist[i] * 0.5f + 
                            hist[(i - 1 + SIFT_ORI_HIST_BINS) % SIFT_ORI_HIST_BINS] * 0.25f +
                            hist[(i + 1) % SIFT_ORI_HIST_BINS] * 0.25f;
        }
        
        // Find peaks
        float max_val = *std::max_element(smooth_hist.begin(), smooth_hist.end());
        
        for (int i = 0; i < SIFT_ORI_HIST_BINS; i++) {
            float val = smooth_hist[i];
            float prev = smooth_hist[(i - 1 + SIFT_ORI_HIST_BINS) % SIFT_ORI_HIST_BINS];
            float next = smooth_hist[(i + 1) % SIFT_ORI_HIST_BINS];
            
            if (val > prev && val > next && val >= SIFT_ORI_PEAK_RATIO * max_val) {
                // Parabolic interpolation for accurate peak position
                float peak = i + 0.5f * (prev - next) / (prev - 2 * val + next);
                peak = (peak < 0) ? peak + SIFT_ORI_HIST_BINS : 
                       (peak >= SIFT_ORI_HIST_BINS) ? peak - SIFT_ORI_HIST_BINS : peak;
                
                Keypoint oriented_kpt = kpt;
                oriented_kpt.angle = 360.0f - peak * 360.0f / SIFT_ORI_HIST_BINS;
                if (oriented_kpt.angle >= 360.0f) {
                    oriented_kpt.angle -= 360.0f;
                }
                oriented.push_back(oriented_kpt);
            }
        }
    }
    
    keypoints = oriented;
}

// Compute SIFT descriptors
void SIFT::computeDescriptors(const ScaleSpacePyramid& pyramid,
                               const std::vector<Keypoint>& keypoints,
                               cv::Mat& descriptors) {
    int nKeypoints = keypoints.size();
    descriptors.create(nKeypoints, 128, descriptorType_);
    
    for (int k = 0; k < nKeypoints; k++) {
        const Keypoint& kpt = keypoints[k];
        const cv::Mat& img = pyramid.octaves[kpt.octave][kpt.layer];
        
        float angle = kpt.angle * CV_PI / 180.0f;
        float cos_angle = std::cos(angle);
        float sin_angle = std::sin(angle);
        
        float hist_width = SIFT_DESCR_SCL_FCTR * kpt.scale;
        float radius = hist_width * std::sqrt(2) * (SIFT_DESCR_WIDTH + 1) * 0.5f;
        
        // Descriptor array
        std::vector<float> desc(128, 0);
        
        // Compute descriptor
        for (int i = -radius; i <= radius; i++) {
            for (int j = -radius; j <= radius; j++) {
                // Rotate to keypoint orientation
                float rot_x = cos_angle * j - sin_angle * i;
                float rot_y = sin_angle * j + cos_angle * i;
                
                float bin_x = rot_x / hist_width + SIFT_DESCR_WIDTH / 2 - 0.5f;
                float bin_y = rot_y / hist_width + SIFT_DESCR_WIDTH / 2 - 0.5f;
                
                if (bin_x < -1 || bin_x >= SIFT_DESCR_WIDTH ||
                    bin_y < -1 || bin_y >= SIFT_DESCR_WIDTH) {
                    continue;
                }
                
                int r = cvRound(kpt.y + i);
                int c = cvRound(kpt.x + j);
                
                if (r <= 0 || r >= img.rows - 1 || c <= 0 || c >= img.cols - 1) {
                    continue;
                }
                
                float dx = img.at<float>(r, c+1) - img.at<float>(r, c-1);
                float dy = img.at<float>(r+1, c) - img.at<float>(r-1, c);
                
                float mag = std::sqrt(dx * dx + dy * dy);
                float ori = std::atan2(dy, dx) - angle;
                while (ori < 0) ori += 2 * CV_PI;
                while (ori >= 2 * CV_PI) ori -= 2 * CV_PI;
                
                float weight = std::exp(-(rot_x * rot_x + rot_y * rot_y) / 
                                       (2 * hist_width * hist_width));
                
                // Trilinear interpolation
                int x0 = std::floor(bin_x);
                int y0 = std::floor(bin_y);
                int o0 = std::floor(ori * SIFT_DESCR_HIST_BINS / (2 * CV_PI));
                
                for (int dx = 0; dx <= 1; dx++) {
                    int xi = x0 + dx;
                    if (xi < 0 || xi >= SIFT_DESCR_WIDTH) continue;
                    
                    float wx = 1 - std::abs(bin_x - xi);
                    
                    for (int dy = 0; dy <= 1; dy++) {
                        int yi = y0 + dy;
                        if (yi < 0 || yi >= SIFT_DESCR_WIDTH) continue;
                        
                        float wy = 1 - std::abs(bin_y - yi);
                        
                        for (int dori = 0; dori <= 1; dori++) {
                            int oi = (o0 + dori) % SIFT_DESCR_HIST_BINS;
                            
                            float wo = 1 - std::abs(ori * SIFT_DESCR_HIST_BINS / (2 * CV_PI) - oi);
                            
                            int idx = (yi * SIFT_DESCR_WIDTH + xi) * SIFT_DESCR_HIST_BINS + oi;
                            desc[idx] += weight * mag * wx * wy * wo;
                        }
                    }
                }
            }
        }
        
        // Normalize descriptor
        float norm = 0;
        for (float val : desc) {
            norm += val * val;
        }
        norm = std::sqrt(norm);
        
        if (norm > 0) {
            for (float& val : desc) {
                val /= norm;
                val = std::min(val, static_cast<float>(SIFT_DESCR_MAG_THR));
            }
            
            // Re-normalize
            norm = 0;
            for (float val : desc) {
                norm += val * val;
            }
            norm = std::sqrt(norm);
            
            for (float& val : desc) {
                val /= norm;
            }
        }
        
        // Convert to output format
        if (descriptorType_ == CV_8U) {
            for (int i = 0; i < 128; i++) {
                descriptors.at<uchar>(k, i) = cv::saturate_cast<uchar>(desc[i] * SIFT_INT_DESCR_FCTR);
            }
        } else {
            for (int i = 0; i < 128; i++) {
                descriptors.at<float>(k, i) = desc[i];
            }
        }
    }
}

// Retain best features based on response
void SIFT::retainBestFeatures(std::vector<Keypoint>& keypoints) {
    if (static_cast<int>(keypoints.size()) <= nfeatures_) {
        return;
    }
    
    // Sort by response
    std::sort(keypoints.begin(), keypoints.end(), 
              [](const Keypoint& a, const Keypoint& b) {
                  return a.response > b.response;
              });
    
    keypoints.resize(nfeatures_);
}

// Convert internal keypoint to OpenCV format
cv::KeyPoint SIFT::Keypoint::toOpenCV() const {
    cv::KeyPoint kp;
    kp.pt.x = x;
    kp.pt.y = y;
    kp.size = scale;
    kp.angle = angle;
    kp.response = response;
    kp.octave = octave + (layer << 8);
    kp.class_id = -1;
    return kp;
}

} // namespace lar