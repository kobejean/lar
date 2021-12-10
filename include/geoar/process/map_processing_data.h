#ifndef GEOAR_MAP_PROCESSING_DATA_H
#define GEOAR_MAP_PROCESSING_DATA_H

#include "g2o/core/optimization_algorithm_factory.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/solvers/structure_only/structure_only_solver.h"
#include "g2o/stuff/sampler.h"
#include "g2o/types/sba/types_six_dof_expmap.h"
#include "g2o/types/slam3d/types_slam3d.h"

#include "geoar/core/frame.h"
#include "geoar/core/map.h"

using namespace std;

namespace geoar {

  class MapProcessingData {
    public:
      g2o::SparseOptimizer optimizer;
      Map map;
      vector<Frame> frames;
      cv::Mat desc;

      MapProcessingData();
  };

}

#endif /* GEOAR_MAP_PROCESSING_DATA_H */