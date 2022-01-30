#ifndef LAR_PROCESSING_DEPTH_H
#define LAR_PROCESSING_DEPTH_H

#include <opencv2/core/types.hpp>

namespace lar {

  class Depth {
    public:
      Depth(cv::Size img_size);

      std::vector<float> depthAt(std::vector<cv::KeyPoint> const &kpts);
      std::vector<float> confidenceAt(std::vector<cv::KeyPoint> const &kpts);

    protected:
      cv::Size _img_size;
      cv::Mat _depth;
      cv::Mat _confidence;

    private:
      std::vector<float> interpolate(cv::Mat image, std::vector<cv::KeyPoint> const &kpts, cv::InterpolationFlags interpolation);
  };


  class SavedDepth : public Depth {
    public:
      SavedDepth(cv::Size img_size, std::string path_prefix);
  };

}

#endif /* LAR_PROCESSING_DEPTH_H */