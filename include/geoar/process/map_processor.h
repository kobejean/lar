#include <iostream>
#include <array>

#include <nlohmann/json.hpp>
#include <Eigen/Core>

#include "g2o/core/optimization_algorithm_factory.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/solvers/structure_only/structure_only_solver.h"
#include "g2o/stuff/sampler.h"
#include "g2o/types/sba/types_six_dof_expmap.h"
#include "g2o/types/slam3d/types_slam3d.h"

#include "geoar/types/frame.h"

using namespace Eigen;
using json = nlohmann::json;

namespace geoar {

  class MapProcessor {
    public: 
      g2o::SparseOptimizer optimizer;
      std::vector<Frame> frames;

      MapProcessor();
      void createMap(std::string directory);
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

      void createFrames(std::string directory);
  };
}