#ifndef GEOAR_LANDMARK_H
#define GEOAR_LANDMARK_H

#include <Eigen/Core>
#include <opencv2/features2d.hpp>
#include <nlohmann/json.hpp>

namespace geoar {

  class Landmark {
    public:
      size_t id;
      Eigen::Vector3d position;
      cv::Mat desc;
      int sightings{0};

      // For r-tree indexing
      Eigen::Vector2d index_center;
      double index_radius;

      Landmark(Eigen::Vector3d &position, cv::Mat desc, size_t id);
      void recordSighting(nlohmann::json &cam_transform);

      static void concatDescriptions(std::vector<Landmark> landmarks, cv::Mat &desc);
  };
}

#endif /* GEOAR_LANDMARK_H */