#ifndef GEOAR_DEPTH_H
#define GEOAR_DEPTH_H

#include <opencv2/core.hpp>

namespace geoar {

  class Depth {
    public:
      Depth(cv::Size img_size);

      std::vector<double> at(std::vector<cv::KeyPoint> const &kpts);
      void loadDepthMap();
      void unloadDepthMap();
    protected:
      cv::Size _img_size;
      cv::Mat _map;
  };


  class SavedDepth : public Depth {
    public:
      SavedDepth(cv::Size img_size, std::string filepath);
      void loadDepthMap();
    private:
      std::string _filepath;
  };

}

#endif /* GEOAR_DEPTH_H */