#ifndef GEOAR_PROCESSING_PROJECTION_H
#define GEOAR_PROCESSING_PROJECTION_H

#include <nlohmann/json.hpp>
#include <Eigen/Core>
#include <opencv2/core/types.hpp>

namespace geoar {

  class Projection {
    public:
      Projection(Eigen::Matrix3d intrinsics, Eigen::Matrix4d extrinsics);
      Eigen::Vector3d projectToWorld(cv::Point2f pt, double depth);
      cv::Point2f projectToImage(Eigen::Vector3d pt);

    private:
      Eigen::Matrix3d R, RT;
      Eigen::Vector3d t;
      double f, cx, cy;
  };

}

#endif /* GEOAR_PROCESSING_PROJECTION_H */