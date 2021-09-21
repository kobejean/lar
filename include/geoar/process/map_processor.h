#include <iostream>

#include <nlohmann/json.hpp>

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
    
      void addFeaturePoints(json& feature_points);
      void addCameraPoints(json& camera_points);
  };

}