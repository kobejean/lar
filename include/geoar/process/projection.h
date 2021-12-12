#ifndef GEOAR_PROJECTION_H
#define GEOAR_PROJECTION_H

#include <nlohmann/json.hpp>
#include <Eigen/Core>
#include <opencv2/features2d.hpp>

using namespace Eigen;
using namespace std;
using json = nlohmann::json;

namespace geoar {

  class Projection {
    public:
      Projection(json const& frame_data);
      Vector3d projectToWorld(cv::Point2f pt, double depth);
      cv::Point2f projectToImage(Vector3d pt);

    private:
      Matrix3d R, RT;
      Vector3d t;
      double f, cx, cy;
  };

}

#endif /* GEOAR_PROJECTION_H */