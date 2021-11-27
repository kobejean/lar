#include <opencv2/features2d.hpp>

#include "geoar/core/landmark.h"

using namespace Eigen;
using namespace std;

namespace geoar {

  class LandmarkExtractor {
    public:
      cv::Ptr<cv::AKAZE> detector;

      LandmarkExtractor();
      vector<Landmark> extractLandmarks();
  };
}