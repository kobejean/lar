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
      /// Map from uuid string to vertex id integer
      std::map<std::string, int> vertex_id_map;
      std::map<std::string, int> observation_count;
      std::map<std::string, Vector3d> points;
    
      void addFeaturePoints(json& feature_points);
      void addCameraPoints(json& camera_points);
      void countObservations(json& camera_points);
  };

}