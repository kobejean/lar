#include <opencv2/features2d.hpp>
#include <nlohmann/json.hpp>

#include "geoar/core/landmark.h"

using namespace Eigen;
using namespace std;
using json = nlohmann::json;

namespace geoar {

  class LandmarkExtractor {
    public:
      cv::Ptr<cv::AKAZE> detector;
      cv::BFMatcher matcher;
      vector<Landmark> discoveredLandmarks;

      LandmarkExtractor();
      vector<Landmark> extractLandmarks(json& frame_data, string directory);

    private:
      void extractFeatures(cv::InputArray image, cv::InputArray mask, vector<cv::KeyPoint> &kpts, cv::Mat &desc);
      vector<cv::DMatch> match(cv::Mat desc1, cv::Mat desc2);
  };
}