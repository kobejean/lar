#ifndef LAR_PROCESSING_DEPTH_H
#define LAR_PROCESSING_DEPTH_H

#include <Eigen/Core>
#include <opencv2/core/types.hpp>

namespace lar {

  class Depth {
    public:
      Depth(cv::Size img_size, Eigen::Matrix3d intrinsics, Eigen::Matrix4d extrinsics);

      std::vector<float> depthAt(const std::vector<cv::KeyPoint> &kpts);
      std::vector<float> confidenceAt(const std::vector<cv::KeyPoint> &kpts);
      std::vector<Eigen::Vector3f> surfaceNormaslAt(const std::vector<cv::KeyPoint> &kpts);\

    protected:
      Eigen::Matrix3d _intrinsics;
      Eigen::Matrix4d _extrinsics;
      cv::Size _img_size;
      cv::Mat _depth;
      cv::Mat _confidence;

    private:
      std::vector<float> interpolate(const cv::Mat& image, const std::vector<cv::KeyPoint> &kpts, cv::InterpolationFlags interpolation);
  };


  class SavedDepth : public Depth {
    public:
      SavedDepth(cv::Size img_size, Eigen::Matrix3d intrinsics, Eigen::Matrix4d extrinsics, std::string path_prefix);
  };

}

#endif /* LAR_PROCESSING_DEPTH_H */