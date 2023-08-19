
#include "lar/tracking/detectors/sift.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <iostream>
#include <array>
#include <math.h>
#include <tuple>


namespace lar {

  void applyGaussianFilter(const cv::Mat& input, cv::Mat& output, const std::array<float, 9>& kernel) {
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

  void SIFT::processOctave(const cv::Mat& img, int octave) {
    applyGaussianFilter(img, gaussians[octave][0], gaussian_kernels[0]);
    // {
    //   std::string output = "./output/sift/gaussian_" + std::to_string(octave) + "_" + std::to_string(0) + ".jpeg";
    //   cv::imwrite(output, gaussians[octave][0]);
    // }
    for (size_t i = 1; i < gaussian_kernels.size(); i++) {
      applyGaussianFilter(gaussians[octave][i-1], gaussians[octave][i], gaussian_kernels[i]);
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
      processOctave(imgMat, i);
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
        float grad_x = (image.at<float>(y, x + 1) - image.at<float>(y, x - 1)) * 0.5f;
        float grad_y = (image.at<float>(y + 1, x) - image.at<float>(y - 1, x)) * 0.5f;
        gradient.at<cv::Vec2f>(y, x) = cv::Vec2f(grad_x, grad_y);
      }
    }
  }


  void SIFT::extractDescriptors(std::vector<cv::KeyPoint>& kpts, cv::Mat& desc, int start_idx) {
    if (kpts.begin() + start_idx == kpts.end()) return; // No keypoints in scale

    const cv::KeyPoint& fisrt_kpt = *(kpts.begin() + start_idx);
    int octave = fisrt_kpt.octave / num_scales;
    int scale = fisrt_kpt.octave % num_scales;
    cv::Mat gaussian = gaussians[octave][scale+1];
    cv::Mat gradient;
    computeGradients(gaussian, gradient);
    desc.reserve(kpts.size());
    for (auto it = kpts.begin() + start_idx; it != kpts.end(); ++it) {
      const cv::KeyPoint& kpt = *it;
    }
  }

  void SIFT::extractFeatures(std::vector<cv::KeyPoint>& kpts, cv::Mat& desc) {
    for (size_t octave = 0; octave < num_octaves; octave++) {
      for (size_t scale = 0; scale < num_scales; scale++) {
        int invalidation_size = invalidation_sizes[scale];
        int start_idx = kpts.size();
        for (int i = invalidation_size; i < DoG[octave][scale].rows - invalidation_size; i++) {
          for (int j = invalidation_size; j < DoG[octave][scale].cols- invalidation_size; j++) {
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
              // std::cout << "octave: " << octave << ", scale: " << scale << ", (" << x << ", " << y << ")" << std::endl;
              auto kpt = cv::KeyPoint({x, y}, multiplier * scale_sizes[scale], -1, s1.at<uchar>(i, j), octave * num_scales + scale);
              kpts.push_back(kpt);
            }
          }
        }
        extractDescriptors(kpts, desc, start_idx);
      }
    }
  }

  void SIFT::detect(cv::InputArray image, std::vector<cv::KeyPoint>& kpts) {
    cv::Mat img = image.getMat();
    cv::Mat desc(cv::Size(128,0), CV_8UC1);

    computeDoG(img);

    extractFeatures(kpts, desc);
  }

}