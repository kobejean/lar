#include <iostream>

#include <nlohmann/json.hpp>
#include <Eigen/Core>

using namespace Eigen;
using json = nlohmann::json;

namespace geoar {

  class MapProcessor {
    public: 
      g2o::SparseOptimizer optimizer;

      MapProcessor();
      void parseMap(std::ifstream& map_ifs);

    private:

      struct Pose {
          int id;
          std::string uuid;
          g2o::SE3Quat transform;
      };

      struct Point {
          int id;
          std::string uuid;
          Vector3d position;
          int obs_count = 0;
      };

      std::map<std::string, Point> points;
      std::map<std::string, Pose> poses;
    
      void parseVertices(json& feature_points, json& camera_points);
      void addFeaturePoints(json& feature_points);
      void addCameraPoints(json& camera_points);
  };
}