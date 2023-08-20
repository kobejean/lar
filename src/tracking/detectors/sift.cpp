
#include "lar/tracking/detectors/sift.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <iostream>
#include <array>
#include <math.h>
#include <tuple>


namespace lar {

  template <size_t kernel_size>
  void applyGaussianFilter(const cv::Mat& input, cv::Mat& output, const std::array<float, kernel_size>& kernel) {
    cv::Mat temp;
    assert(input.type() == CV_8UC1);
    temp.create(input.size(), CV_32FC1);
    output.create(input.size(), input.type());
    int half_size = kernel.size() / 2;

    // Convolve with the 1D kernel along the rows
    for (int i = 0; i < input.rows; i++) {
      for (int j = 0; j < input.cols; j++) {
        float sum = 0.0f;
        for (int k = -half_size; k <= half_size; k++) {
          int idx = j + k;
          if (idx >= 0 && idx < input.cols) {
            sum += static_cast<float>(input.at<uchar>(i, idx)) * kernel[k + half_size];
          }
        }
        temp.at<float>(i, j) = sum;
      }
    }

    // Convolve with the 1D kernel along the columns
    for (int i = 0; i < input.rows; i++) {
      for (int j = 0; j < input.cols; j++) {
        float sum = 0.0f;
        for (int k = -half_size; k <= half_size; k++) {
          int idx = i + k;
          if (idx >= 0 && idx < input.rows) {
              sum += temp.at<float>(idx, j) * kernel[k + half_size];
          }
        }
        output.at<uchar>(i, j) = static_cast<uchar>(sum);
      }
    }
  }

  SIFT::SIFT() {
    
  }

  template <size_t kernel_size>
  void SIFT::processOctave(const cv::Mat& img, int octave) {
    static constexpr std::array<std::array<float, kernel_size>, 6> gaussian_kernels = computeGaussianKernels<kernel_size>();

    applyGaussianFilter<kernel_size>(img, gaussians[octave][0], gaussian_kernels[0]);
    // {
    //   std::string output = "./output/sift/gaussian_" + std::to_string(octave) + "_" + std::to_string(0) + ".jpeg";
    //   cv::imwrite(output, gaussians[octave][0]);
    // }
    for (size_t i = 1; i < gaussian_kernels.size(); i++) {
      applyGaussianFilter<kernel_size>(gaussians[octave][i-1], gaussians[octave][i], gaussian_kernels[i]);
      DoG[octave][i-1] = cv::abs(gaussians[octave][i-1] - gaussians[octave][i]) ;
      // {
      //   std::string output = "./output/sift/DoG_" + std::to_string(octave) + "_" + std::to_string(i-1) + ".jpeg";
      //   cv::imwrite(output, DoG[octave][i-1] * 10.0);
      // }
      // {
      //   std::string output = "./output/sift/gaussian_" + std::to_string(octave) + "_" + std::to_string(i) + ".jpeg";
      //   cv::imwrite(output, gaussians[octave][i]);
      // }
    }
  }

  void SIFT::computeDoG(const cv::Mat& img) {
    cv::Mat imgMat = img.clone(); 
    for (size_t i = 0; i < num_octaves; i++) {
      processOctave<7>(imgMat, i);
      cv::resize(imgMat, imgMat, cv::Size(), 0.5, 0.5);
    }
  }

  void computeGradients(const cv::Mat& image, cv::Mat& gradient) {
    // Initialize gradient matrices with the same size and type as the input image
    gradient.create(image.size(), CV_32FC2);

    // Iterate through the image pixels (skip the first and last row and column)
    for (int y = 1; y < image.rows - 1; ++y) {
      for (int x = 1; x < image.cols - 1; ++x) {
        // Compute gradient in x and y directions (central difference)
        float grad_x = static_cast<float>(image.at<uchar>(y, x + 1) - image.at<uchar>(y, x - 1)) * 0.5f;
        float grad_y = static_cast<float>(image.at<uchar>(y + 1, x) - image.at<uchar>(y - 1, x)) * 0.5f;
        gradient.at<cv::Vec2f>(y, x) = cv::Vec2f(grad_x, grad_y);
      }
    }
  }

  template <size_t num_bins>
  std::array<float, num_bins> weightedHistogram(const cv::Mat& data, const cv::Mat& weights) {
    std::array<float, num_bins> histogram = {};
    static constexpr float min_val = -M_PI;
    static constexpr float max_val = M_PI;
    static constexpr float bin_width = (max_val - min_val) / num_bins;
    const float* data_ptr = data.ptr<float>();
    const float* weights_ptr = weights.ptr<float>();

    for (size_t i = 0; i < static_cast<size_t>(data.rows * data.cols); ++i) {
      int bin_idx = static_cast<int>((data_ptr[i] - min_val) / bin_width);
      assert(bin_idx >= 0 && bin_idx < num_bins);
      histogram[bin_idx] += weights_ptr[i];
    }
    return histogram;
  }

  void derotatePatch(const cv::Mat& gradient, cv::Point2f pt, int patch_size, float angle_rad, cv::Mat& derotated_patch) {
    // std::cout << "Point: (" << pt.x << ", " << pt.y << "), Patch Size: " << patch_size << ", Angle: " << angle_rad << "\n";

    // Initialize the derotated patch
    derotated_patch.create(cv::Size(patch_size,patch_size), CV_32FC2);

    // Compute the derotated patch
    for (int px = 0; px < patch_size; ++px) {
      for (int py = 0; py < patch_size; ++py) {
        float x_origin = px - patch_size / 2.0f;
        float y_origin = py - patch_size / 2.0f;

        // Rotate patch by angle angle_rad
        float x_rotated = std::cos(angle_rad) * x_origin - std::sin(angle_rad) * y_origin;
        float y_rotated = std::sin(angle_rad) * x_origin + std::cos(angle_rad) * y_origin;

        // Move coordinates to patch
        float x_patch_rotated = pt.x + x_rotated;
        float y_patch_rotated = pt.y - y_rotated;

        // Sample image (using nearest-neighbor sampling)
        int y_img = std::ceil(y_patch_rotated);
        int x_img = std::ceil(x_patch_rotated);

        // Print debug information

        // Ensure coordinates are within bounds (optional, for safety)
        if (y_img < 0 || y_img >= gradient.rows || x_img < 0 || x_img >= gradient.cols) {
          // std::cout << "Processing patch coordinate (" << px << ", " << py << ")\n";
          // std::cout << "Original position: (" << x_origin << ", " << y_origin << ")\n";
          // std::cout << "Rotated position: (" << x_rotated << ", " << y_rotated << ")\n";
          // std::cout << "Image position: (" << x_img << ", " << y_img << ")\n";
          // std::cerr << "Out-of-bounds access! Skipping this pixel.\n";
          derotated_patch.at<cv::Vec2f>(py, px) = cv::Vec2f(0,0);
        } else {
          derotated_patch.at<cv::Vec2f>(py, px) = gradient.at<cv::Vec2f>(y_img, x_img);
        }
      }
    }
  }


  template <size_t PATCH_SIZE>
  void weightGradient(const cv::Mat& patch, cv::Mat& grad_dir, cv::Mat& grad_mag) {
    static constexpr int HALF_SIZE = PATCH_SIZE / 2;
    static constexpr std::array<float, PATCH_SIZE> gaussian_window = computeGaussianKernel<PATCH_SIZE>(1.5 * PATCH_SIZE, HALF_SIZE + 0.5); 
    for (size_t i = 0; i < PATCH_SIZE; i++) {
      for (size_t j = 0; j < PATCH_SIZE; j++) {
        cv::Vec2f grad = patch.at<cv::Vec2f>(i,j);
        float weight = gaussian_window[i] * gaussian_window[j];
        grad_dir.at<float>(i,j) = std::atan2(grad[0], grad[1]);
        grad_mag.at<float>(i,j) = weight * std::sqrt(grad[0]*grad[0] + grad[1]*grad[1]);
      }
    }
  }


  void SIFT::extractDescriptors(std::vector<cv::KeyPoint>& kpts, cv::Mat& desc, int start_idx) {
    static constexpr int PATCH_SIZE = 16;
    static constexpr int HALF_SIZE = PATCH_SIZE / 2;
    static constexpr int QUART_SIZE = PATCH_SIZE / 4;

    if (kpts.begin() + start_idx == kpts.end()) return; // No keypoints in scale

    const cv::KeyPoint& fisrt_kpt = *(kpts.begin() + start_idx);
    int octave = fisrt_kpt.octave / num_scales;
    int scale = fisrt_kpt.octave % num_scales;
    float multiplier = static_cast<float>(1 << octave);
    cv::Mat gaussian = gaussians[octave][scale+1];
    cv::Mat gradient;
    computeGradients(gaussian, gradient);
    desc.reserve(kpts.size());
    for (auto it = kpts.begin() + start_idx; it != kpts.end(); ++it) {
      cv::KeyPoint& kpt = *it;
      const cv::Point2f pt = kpt.pt / multiplier;

      // extract gradient
      int left = static_cast<int>(pt.x) - HALF_SIZE;
      int top = static_cast<int>(pt.y) - HALF_SIZE;
      cv::Rect roi(left, top, PATCH_SIZE, PATCH_SIZE);
      cv::Mat patch = gradient(roi);
      auto patch_size = cv::Size(PATCH_SIZE, PATCH_SIZE);

      cv::Mat grad_dir(patch_size, CV_32FC1);
      cv::Mat grad_mag(patch_size, CV_32FC1);

      weightGradient<PATCH_SIZE>(patch, grad_dir, grad_mag);

      auto patch_hist = weightedHistogram<36>(grad_dir, grad_mag);
      size_t max_idx = 0;
      for (size_t i = 1; i < patch_hist.size(); i++) {
        if (patch_hist[i] > patch_hist[max_idx]) max_idx = i;
      }
      float dom_angle = -M_PI + (static_cast<float>(max_idx) + 0.5) * 2 * M_PI / 36.0;
      kpt.angle = 180 * dom_angle / M_PI;
      derotatePatch(gradient, pt, PATCH_SIZE, dom_angle, patch);

      weightGradient<PATCH_SIZE>(patch, grad_dir, grad_mag);

      // compute histograms
      cv::Mat desc_row(cv::Size(8,0), CV_32FC1);
      desc_row.reserve(QUART_SIZE*QUART_SIZE);
      for (int i = 0; i < PATCH_SIZE; i += QUART_SIZE) {
        for (int j = 0; j < PATCH_SIZE; j += QUART_SIZE) {
          cv::Rect roi(i, j, QUART_SIZE, QUART_SIZE);
          cv::Mat sub_grad_dir = grad_dir(roi);
          cv::Mat sub_grad_mag = grad_mag(roi);
          auto hist = weightedHistogram<8>(sub_grad_dir, sub_grad_mag);
          cv::Mat hist_row(cv::Size(8,1), CV_32FC1, hist.data());
          desc_row.push_back(hist_row);
        }
      }
      desc.push_back(desc_row.reshape(1, 1));
    }
  }

  void SIFT::extractFeatures(std::vector<cv::KeyPoint>& kpts, cv::Mat& desc) {
    static constexpr std::array<float, 3> scale_sizes = { 2.01587367983, 2.5398416831, 3.2 };
    static constexpr int invalidation_size = 8;

    for (size_t octave = 0; octave < num_octaves; octave++) {
      for (size_t scale = 0; scale < num_scales; scale++) {
        int start_idx = kpts.size();
        for (int i = invalidation_size; i < DoG[octave][scale].rows - invalidation_size; i++) {
          for (int j = invalidation_size; j < DoG[octave][scale].cols - invalidation_size; j++) {
            cv::Mat s0 = DoG[octave][scale-1];
            cv::Mat s1 = DoG[octave][scale];
            cv::Mat s2 = DoG[octave][scale+1];
            uchar max = std::max({
              s0.at<uchar>(i-1, j-1), s0.at<uchar>(i-1, j  ), s0.at<uchar>(i-1, j+1),
              s0.at<uchar>(i  , j-1), s0.at<uchar>(i  , j  ), s0.at<uchar>(i  , j+1),
              s0.at<uchar>(i+1, j-1), s0.at<uchar>(i+1, j  ), s0.at<uchar>(i+1, j+1),

              s1.at<uchar>(i-1, j-1), s1.at<uchar>(i-1, j  ), s1.at<uchar>(i-1, j+1),
              s1.at<uchar>(i  , j-1), /*s1.at<uchar>(i, j),*/ s1.at<uchar>(i  , j+1),
              s1.at<uchar>(i+1, j-1), s1.at<uchar>(i+1, j  ), s1.at<uchar>(i+1, j+1),

              s2.at<uchar>(i-1, j-1), s2.at<uchar>(i-1, j  ), s2.at<uchar>(i-1, j+1),
              s2.at<uchar>(i  , j-1), s2.at<uchar>(i  , j  ), s2.at<uchar>(i  , j+1),
              s2.at<uchar>(i+1, j-1), s2.at<uchar>(i+1, j  ), s2.at<uchar>(i+1, j+1),
            });
            if (s1.at<uchar>(i, j) > max && s1.at<uchar>(i, j) >= contrast_threshold) {
              float multiplier = static_cast<float>(1 << octave);
              float y = i * multiplier;
              float x = j * multiplier;
              
              auto kpt = cv::KeyPoint({x, y}, 2*multiplier * scale_sizes[scale], -1, s1.at<uchar>(i, j), octave * num_scales + scale);
              kpts.push_back(kpt);
            }
          }
        }
        extractDescriptors(kpts, desc, start_idx);
      }
    }
  }

  void SIFT::detect(cv::InputArray image, std::vector<cv::KeyPoint>& kpts, cv::Mat& desc) {
    cv::Mat img = image.getMat();
    desc.create(cv::Size(128,0), CV_32FC1);
    // cv::Mat desc(cv::Size(128,0), CV_32FC1);

    computeDoG(img);

    extractFeatures(kpts, desc);
  }

}