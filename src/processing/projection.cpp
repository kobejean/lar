#include "geoar/processing/projection.h"

namespace geoar {

  Projection::Projection(Eigen::Matrix3d intrinsics, Eigen::Matrix4d extrinsics) : 
    R(extrinsics.block<3,3>(0,0)), RT(extrinsics.block<3,3>(0,0).transpose()), t(extrinsics.block<3,1>(0,3)) {
    f = intrinsics(0,0);
    cx = intrinsics(0,2);
    cy = intrinsics(1,2);
  }

  Eigen::Vector3d Projection::projectToWorld(cv::Point2f pt, double depth) {
    // Use intrinsics and depth to project image coordinates to 3d camera space point
    double scale = depth / f;
    double x = (pt.x - cx) * scale;
    double y = (pt.y - cy) * -scale;
    double z = -depth;
    Eigen::Vector3d c(x, y, z);

    // Convert camera space point to world space point
    return R * c + t;
  }

  cv::Point2f Projection::projectToImage(Eigen::Vector3d pt) {
    // Convert to camera space point
    Eigen::Vector3d c = RT * (pt - t);
    
    // Convert camera space point to image space point
    double scale = f / -c[2];
    float x = scale * c[0] + cx;
    float y = -scale * c[1] + cy;
    return cv::Point2f(x, y);
  }

}