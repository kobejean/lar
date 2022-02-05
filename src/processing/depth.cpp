#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "lar/processing/depth.h"

namespace lar {

  Depth::Depth(cv::Size img_size, Eigen::Matrix3d intrinsics, Eigen::Matrix4d extrinsics):
    _img_size(img_size), _intrinsics(intrinsics), _extrinsics(extrinsics) {
  }
  
  std::vector<float> Depth::depthAt(const std::vector<cv::KeyPoint>& kpts) {
    return interpolate(_depth, kpts, cv::INTER_LINEAR);
  }
  
  std::vector<float> Depth::confidenceAt(const std::vector<cv::KeyPoint>& kpts) {
    return interpolate(_confidence, kpts, cv::INTER_NEAREST);
  }

  std::vector<Eigen::Vector3f> Depth::surfaceNormaslAt(const std::vector<cv::KeyPoint>& kpts) {
    float image_scale_u = (float) _img_size.height / (float) _depth.size().height;
    float image_scale_v = (float) _img_size.width / (float) _depth.size().width;
    float y_per_uz = -image_scale_u / _intrinsics(1,1);
    float x_per_vz = image_scale_v / _intrinsics(0,0);
    cv::Mat rough_normals = cv::Mat::zeros(_depth.size().height-1, _depth.size().width-1, CV_32FC3);

    Eigen::Matrix3f rotation = _extrinsics.block<3,3>(0,0).cast<float>();

    for (int u = 0; u < _depth.rows-1; u++) {
      for (int v = 0; v < _depth.cols-1; v++) {
        float z = 0.25 * (_depth.at<float>(u, v) + _depth.at<float>(u+1, v) + _depth.at<float>(u, v+1) + _depth.at<float>(u+1, v+1));
        float dy = z * y_per_uz;
        float dx = z * x_per_vz;
        float y = (_depth.at<float>(u+1, v) - _depth.at<float>(u, v)) / dy;
        float x = (_depth.at<float>(u, v+1) - _depth.at<float>(u, v)) / dx;
        cv::Vec3f direction(x, y, 1);
        rough_normals.at<cv::Vec3f>(u, v) = cv::normalize(direction);
      }
    }

    // Visualize surface normals
    // cv::Mat vis = (1 + rough_normals)*255/2;
    // cv::Mat vis2 = (_depth - *std::min_element(_depth.begin<float>(), _depth.end<float>()))*255/ *std::max_element(_depth.begin<float>(), _depth.end<float>());
    // cv::imwrite("./output/normals.jpeg", vis);
    // cv::imwrite("./output/depth.jpeg", vis2);

    // TODO: Take into account, the key points size
    std::vector<Eigen::Vector3f> surface_normals;
    for (const cv::KeyPoint& kpt : kpts) {
      int u = std::max((int) round(kpt.pt.y / image_scale_u - 0.5), 0);
      int v = std::max((int) round(kpt.pt.x / image_scale_v - 0.5), 0);
      cv::Vec3f n = rough_normals.at<cv::Vec3f>(u, v);
      Eigen::Vector3f normal{ n[0], n[1], n[2] };
      normal = rotation * normal;
      surface_normals.push_back(normal);
    }
    return surface_normals;
  }


  // Depth - Private Methods
  
  std::vector<float> Depth::interpolate(const cv::Mat& image, const std::vector<cv::KeyPoint>& kpts, cv::InterpolationFlags interpolation) {
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

  SavedDepth::SavedDepth(cv::Size img_size, Eigen::Matrix3d intrinsics, Eigen::Matrix4d extrinsics, std::string path_prefix):
    Depth(img_size, intrinsics, extrinsics) {
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