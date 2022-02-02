#ifndef LAR_PROCESSING_DEPTH_H
#define LAR_PROCESSING_DEPTH_H

#include <Eigen/Core>
#include <opencv2/core/types.hpp>

namespace lar {

  class Depth {
    public:
      Depth(cv::Size img_size);

      std::vector<float> depthAt(const std::vector<cv::KeyPoint> &kpts);
      std::vector<float> confidenceAt(const std::vector<cv::KeyPoint> &kpts);
      std::vector<Eigen::Vector3d> surfaceNormalAt(const std::vector<cv::KeyPoint> &kpts);

    protected:
      cv::Size _img_size;
      cv::Mat _depth;
      cv::Mat _confidence;

    private:
      std::vector<float> interpolate(const cv::Mat& image, const std::vector<cv::KeyPoint> &kpts, cv::InterpolationFlags interpolation);
  };


  class SavedDepth : public Depth {
    public:
      SavedDepth(cv::Size img_size, std::string path_prefix);
  };

}

#endif /* LAR_PROCESSING_DEPTH_H */