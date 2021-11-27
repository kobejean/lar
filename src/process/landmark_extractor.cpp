
#include <stdint.h>

#include <iostream>
#include <unordered_set>

#include <nlohmann/json.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "geoar/process/landmark_extractor.h"

using namespace Eigen;
using namespace std;
using json = nlohmann::json;

namespace geoar {

  LandmarkExtractor::LandmarkExtractor() {
    detector = cv::AKAZE::create();
  }

  vector<Landmark> LandmarkExtractor::extractLandmarks() {
    vector<Landmark> empty;
    return empty;
  }
}