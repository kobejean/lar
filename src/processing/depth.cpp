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
    cv::Mat rough_normals = cv::Mat::zeros(_depth.size(), CV_32FC3);
    float image_scale_u = (float) _img_size.height / (float) _depth.size().height;
    float image_scale_v = (float) _img_size.width / (float) _depth.size().width;
    float y_per_uz = -image_scale_u / _intrinsics(1,1);
    float x_per_vz = image_scale_v / _intrinsics(0,0);


    Eigen::Matrix3f rotation = _extrinsics.block<3,3>(0,0).cast<float>().transpose();

    // TODO: add proper padding
    for (int u = 1; u < _depth.rows-1; u++) {
      for (int v = 1; v < _depth.cols-1; v++) {
        float z = _depth.at<float>(u, v);
        float dy = z * y_per_uz * 2;
        float dx = z * x_per_vz * 2;
        float dzdy = (_depth.at<float>(u+1, v) - _depth.at<float>(u-1, v)) / dy;
        float dzdx = (_depth.at<float>(u, v+1) - _depth.at<float>(u, v-1)) / dx;

        cv::Vec3f d(-dzdx, -dzdy, 1);
        cv::Vec3f n = normalize(d);
        // Eigen::Vector3f normal{ n[0], n[1], n[2] };
        // normal = rotation * normal;
        // n[0] = normal.x();
        // n[1] = normal.y();
        // n[2] = normal.z();
        rough_normals.at<cv::Vec3f>(u, v) = n;
      }
    }

    // Visualize surface normals
    // cv::Mat vis = (rough_normals + 1)*255/2;
    // cv::Mat vis2 = (_depth - *std::min_element(_depth.begin<float>(), _depth.end<float>()))*255/ *std::max_element(_depth.begin<float>(), _depth.end<float>());
    // cv::imwrite("./output/normals.jpeg", vis);
    // cv::imwrite("./output/depth.jpeg", vis2);

    std::vector<Eigen::Vector3f> surface_normals;
    for (const cv::KeyPoint& kpt : kpts) {
      int u = (int) round(kpt.pt.y / image_scale_u);
      int v = (int) round(kpt.pt.x / image_scale_v);
      cv::Vec3f n = rough_normals.at<cv::Vec3f>(u, v);
      Eigen::Vector3f normal{ n[0], n[1], n[2] };
      surface_normals.push_back(rotation * normal);
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