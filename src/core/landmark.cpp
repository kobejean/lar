#include <stdint.h>

#include <iostream>

#include "geoar/core/landmark.h"

using namespace std;

namespace geoar {


  Landmark::Landmark(cv::KeyPoint kpt, cv::Mat desc) {
    this->kpt = kpt;
    this->desc = desc;
  }

  void Landmark::concatDescriptions(vector<Landmark> landmarks, cv::Mat &desc) {
    for(size_t i = 0; i < landmarks.size(); i++) {
      desc.push_back(landmarks[i].desc);
    }
    cout << "desc.row(10): " << desc.row(10) << endl;
  }

}