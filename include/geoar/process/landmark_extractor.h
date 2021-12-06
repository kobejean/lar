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
      cv::Mat all_desc;
      vector<Landmark> all_landmarks;

      LandmarkExtractor();
      vector<Landmark> extractLandmarks(json& frame_data, std::string directory);

    private:
      void extractFeatures(cv::InputArray image, cv::InputArray mask, vector<cv::KeyPoint> &kpts, cv::Mat &desc);
      void matchAndFilter(vector<cv::KeyPoint> &kpts, cv::Mat &desc);
      vector<Vector3f> projectKeyPoints(vector<cv::KeyPoint> &kpts, cv::Mat &depth, json& transform);
      vector<Landmark> createLandmarks(vector<Vector3f> &pts3d, vector<cv::KeyPoint> &kpts, cv::Mat &desc);
      void recordFeatures(vector<Landmark> &landmarks, cv::Mat &desc);

      vector<cv::DMatch> match(cv::Mat desc1, cv::Mat desc2);
  };

}