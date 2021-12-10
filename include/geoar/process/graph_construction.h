#ifndef GEOAR_GRAPH_CONSTRUCTION_H
#define GEOAR_GRAPH_CONSTRUCTION_H

#include <opencv2/features2d.hpp>
#include <nlohmann/json.hpp>
#include <Eigen/Core>

#include "g2o/core/optimization_algorithm_factory.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/solvers/structure_only/structure_only_solver.h"
#include "g2o/stuff/sampler.h"
#include "g2o/types/sba/types_six_dof_expmap.h"
#include "g2o/types/slam3d/types_slam3d.h"

#include "geoar/core/landmark.h"
#include "geoar/core/map.h"
#include "geoar/process/map_processing_data.h"
#include "geoar/process/vision.h"

using namespace Eigen;
using namespace std;
using json = nlohmann::json;

namespace geoar {

  class GraphConstruction {
    public:
      Vision vision;
      MapProcessingData* data;

      GraphConstruction(MapProcessingData &data);
      void processRawData(std::string directory);
      void processFrameData(json& frame_data, std::string directory);

    private:
      void matchAndFilter(vector<cv::KeyPoint> &kpts, cv::Mat &desc);
      vector<Vector3f> projectKeyPoints(vector<cv::KeyPoint> &kpts, cv::Mat &depth, json& transform);
      vector<Landmark> createLandmarks(vector<Vector3f> &pts3d, vector<cv::KeyPoint> &kpts, cv::Mat &desc);
      void recordFeatures(vector<Landmark> &landmarks, cv::Mat &desc);
  };

}

#endif /* GEOAR_GRAPH_CONSTRUCTION_H */