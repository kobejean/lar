#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "lar/processing/depth.h"

namespace lar {

  Depth::Depth(cv::Size img_size) {
    _img_size = img_size;
  }
  
  std::vector<float> Depth::depthAt(std::vector<cv::KeyPoint> const &kpts) {
    return interpolate(_depth, kpts, cv::INTER_LINEAR);
  }
  
  std::vector<float> Depth::confidenceAt(std::vector<cv::KeyPoint> const &kpts) {
    return interpolate(_confidence, kpts, cv::INTER_NEAREST);
  }


  // Depth - Private Methods
  
  std::vector<float> Depth::interpolate(cv::Mat image, std::vector<cv::KeyPoint> const &kpts, cv::InterpolationFlags interpolation) {
    assert(image.channels() == 1);
    size_t keypoint_count = kpts.size();

    // Calculate pixel scale factor
    float x_scale = (float) image.size().width / (float) _img_size.width;
    float y_scale = (float) image.size().height / (float) _img_size.height;

    cv::Mat map(cv::Size(keypoint_count,1), CV_32FC2);
    for (size_t i = 0; i < keypoint_count; i++) {
      cv::Vec2f & pt = map.at<cv::Vec2f>(i);
      pt[0] = x_scale * kpts[i].pt.x;
      pt[1] = y_scale * kpts[i].pt.y;
    }

    cv::Mat output;
    cv::remap(image, output, map, cv::noArray(), interpolation);
    
    // Convert `output` to `std::vector`
    float *data = (float*)(output.isContinuous() ? output.data : output.clone().data);
    std::vector<float> values;
    values.assign(data, data + output.total());
    return values;
  }


  // SavedDepth

  SavedDepth::SavedDepth(cv::Size img_size, std::string path_prefix): Depth(img_size) {
    std::string depth_filepath = path_prefix + "depth.pfm";
    std::string confidence_filepath = path_prefix + "confidence.pfm";

    // Load depth map
    std::cout << "loading: " << depth_filepath << std::endl;
    _depth = cv::imread(depth_filepath, cv::IMREAD_UNCHANGED);

    // Load confidence map
    std::cout << "loading: " << confidence_filepath << std::endl;
    cv::Mat confidence = cv::imread(confidence_filepath, cv::IMREAD_UNCHANGED);
    _confidence = cv::Mat(confidence.size(), CV_32FC1);
    // Estimated inverse variance
    _confidence.setTo(0.001f, confidence == 0);
    _confidence.setTo(1.0f, confidence == 1);
    _confidence.setTo(50.0f, confidence == 2);
  }

}