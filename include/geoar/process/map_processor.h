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

#include "geoar/process/landmark_extractor.h"
#include "geoar/core/frame.h"

using namespace Eigen;
using json = nlohmann::json;

namespace geoar {

  class MapProcessor {
    public: 
      g2o::SparseOptimizer optimizer;
      LandmarkExtractor extractor;
      std::vector<Frame> frames;

      MapProcessor();
      void createMap(std::string directory);

    private:

      void createFrames(std::string directory);
  };
}