#include "geoar/process/projection.h"

namespace geoar {

  Projection::Projection(nlohmann::json const& frame_data) {
    // Variables used for projection
    nlohmann::json T = frame_data["transform"];
    R << T[0][0], T[1][0], T[2][0],
        T[0][1], T[1][1], T[2][1],
        T[0][2], T[1][2], T[2][2]; 
    RT = R.transpose();
    t = Eigen::Vector3d(T[3][0], T[3][1], T[3][2]);

    // Intrinsics properties
    nlohmann::json in = frame_data["intrinsics"];
    nlohmann::json pp = in["principlePoint"];
    f = in["focalLength"];
    cx = pp["x"];
    cy = pp["y"];
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