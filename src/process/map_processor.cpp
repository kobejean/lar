
#include <stdint.h>

#include <iostream>
#include <unordered_set>

#include <nlohmann/json.hpp>

#include "g2o/core/optimization_algorithm_factory.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/solvers/structure_only/structure_only_solver.h"
#include "g2o/stuff/sampler.h"
#include "g2o/types/sba/types_six_dof_expmap.h"
#include "g2o/types/slam3d/types_slam3d.h"

#include "geoar/process/map_processor.h"

#if defined G2O_HAVE_CHOLMOD
G2O_USE_OPTIMIZATION_LIBRARY(cholmod);
#else
G2O_USE_OPTIMIZATION_LIBRARY(eigen);
#endif

G2O_USE_OPTIMIZATION_LIBRARY(dense);

using namespace Eigen;
using namespace std;
using json = nlohmann::json;

namespace geoar {

  MapProcessor::MapProcessor() {
    optimizer.setVerbose(true);
    string solverName = "lm_fix6_3";
  #ifdef G2O_HAVE_CHOLMOD
    solverName = "lm_fix6_3_cholmod";
  #else
    solverName = "lm_fix6_3";
  #endif
    g2o::OptimizationAlgorithmProperty solverProperty;
    auto algorithm = g2o::OptimizationAlgorithmFactory::instance()->construct(solverName, solverProperty);
    optimizer.setAlgorithm(algorithm);
  }

  void MapProcessor::parseMap(std::ifstream& map_ifs) {
    json map = json::parse(map_ifs);
    json point_cloud = map["pointCloud"];
    json camera_points = point_cloud["cameraPoints"];
    json feature_points = point_cloud["featurePoints"];
    // json location_points = point_cloud["locationPoints"];

    countObservations(camera_points);

    addFeaturePoints(feature_points);
    addCameraPoints(camera_points);

    // map.json layout:
    //
    // cameraPoints.featurePoints = []
    // cameraPoints.identifier = "402D8F8C-786B-4758-A914-89D2F05C1D26"
    // cameraPoints.intrinsics = {"focalLength":1596.0804443359375,"principlePoint":{"x":952.559326171875,"y":714.2745971679688}}
    // cameraPoints.locationPoints = ["B215810D-196F-45D6-B2EC-2DF6E9DA4448"]
    // cameraPoints.transform = [[0.7096548080444336,-0.05337836220860481,0.7025245428085327,0],[0.3269939422607422,0.9081810116767883,-0.2613084614276886,0],[-0.6240712404251099,0.4151601195335388,0.6619495153427124,0],[0.6320425271987915,0.36971643567085266,-1.1188805103302002,0.9999999403953552]]
    // cameraPoints[*].featurePoints[*].identifier
    // cameraPoints[*].featurePoints[*].keyPoint
    // cameraPoints[*].intrinsics.focalLength
    // cameraPoints[*].intrinsics.principlePoint
    // cameraPoints[*].intrinsics.principlePoint.x
    // cameraPoints[*].intrinsics.principlePoint.y
    // featurePoints[*].descriptor
    // featurePoints[*].identifier
    // featurePoints[*].response
    // featurePoints[*].transform
    // locationPoints[*].identifier
    // locationPoints[*].latitude
    // locationPoints[*].longitude
    // locationPoints[*].transform
    
    // for (auto& el : camera_points[0].items()) {
    //   std::cout << "cameraPoints." << el.key() << " = " << el.value() << '\n';
    // }
  }

  void MapProcessor::addFeaturePoints(json& feature_points) {
    for (auto& el : feature_points.items()) {
      json feature_point = el.value();
      auto identifier = feature_point["identifier"];

      if (observation_count[identifier] > 2) {
        // Register uuid -> vertex id mapping
        int id = optimizer.vertices().size()+1;
        vertex_id_map[identifier] = id;

        // Create position measurement
        json t = feature_point["transform"];
        Vector3d position(t[3][0], t[3][1], t[3][2]);
        points[identifier] = position;

        // Create feature point vertex
        g2o::VertexPointXYZ * vertex = new g2o::VertexPointXYZ();
        vertex->setId(id);
        vertex->setMarginalized(true);
        vertex->setEstimate(position);
        optimizer.addVertex(vertex);
      }
    }
  }

  void MapProcessor::addCameraPoints(json& camera_points) {
    int camera_point_count = 0;
    for (auto& el : camera_points.items()) {
      json camera_point = el.value();
      auto identifier = camera_point["identifier"];

      // Handle camera points that have associated feature points
      if (camera_point.contains("featurePoints") && camera_point["featurePoints"].size() > 0) {
      // if (observation_count[identifier] > 2) {
        // Register uuid -> vertex id mapping
        int camera_point_id = optimizer.vertices().size()+1;
        vertex_id_map[identifier] = camera_point_id;

        // Create pose measurement
        json t = camera_point["transform"];
        Matrix3d rot;
        rot << t[0][0], t[1][0], t[2][0],
               t[0][1], t[1][1], t[2][1],
               t[0][2], t[1][2], t[2][2]; 
        Vector3d position(t[3][0], t[3][1], t[3][2]);
        Quaterniond orientation(rot);
        g2o::SE3Quat pose;
        pose = g2o::SE3Quat(orientation, position).inverse();

        // Create camera point vertex
        g2o::VertexSE3Expmap * vertex = new g2o::VertexSE3Expmap();
        vertex->setId(camera_point_id);
        vertex->setEstimate(pose);
        if (camera_point_count < 2){
          vertex->setFixed(true); // Fix the first camera point
        }
        optimizer.addVertex(vertex);

        // Add edge from camera point to camera point after the second camera point has been added
        if (camera_point_count > 0) {
          g2o::EdgeSE3Expmap * edge = new g2o::EdgeSE3Expmap();
          edge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(camera_point_id-1)));
          edge->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(camera_point_id)));
          edge->information() = Eigen::MatrixXd::Identity(6,6);
          optimizer.addEdge(edge);
        }

        camera_point_count++;

        // Get camera intrinsics
        json intrinsics = camera_point["intrinsics"];
        double focal_length = intrinsics["focalLength"];
        Vector2d principle_point(intrinsics["principlePoint"]["x"], intrinsics["principlePoint"]["y"]);
        auto * camera_params = new g2o::CameraParameters(focal_length, principle_point, 0.);
        camera_params->setId(camera_point_count);
        if (!optimizer.addParameter(camera_params)) {
          assert(false);
        }

        // Add edges connecting from camera pose to feature point position
        for (auto& el : camera_point["featurePoints"].items()) {
          json feature_point = el.value();
          std::string identifier = feature_point["identifier"];
          int feature_point_id = vertex_id_map[identifier];
          json key_point = feature_point["keyPoint"];
          double key_point_x = key_point["x"];
          Vector2d kp = Vector2d(principle_point[0]*2-key_point_x, key_point["y"]);
          auto p = points[identifier];
          auto m = pose.map(p);
          Vector2d z = camera_params->cam_map(m);
          Vector2d diff = z - kp;

          // if (kp[0] >= 0 && kp[1] >= 0 && kp[0] < principle_point_x*2 && kp[1] < principle_point_y*2) {
          // if (z[0] >= 0 && z[1] >= 0 && z[0] < principle_point[0]*2 && z[1] < principle_point[1]*2 && observation_count[identifier] > 2 && feature_point_id > 0) {
          if (observation_count[identifier] > 2 && feature_point_id > 0 && abs(diff[0]) < 100 && abs(diff[1]) < 100) {
            g2o::EdgeProjectXYZ2UV * edge = new g2o::EdgeProjectXYZ2UV();
            edge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(feature_point_id)));
            edge->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(camera_point_id)));
            edge->setMeasurement(kp);
            edge->information() = Matrix2d::Identity();
            edge->setParameterId(0, camera_point_count);
            // g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            // rk->setDelta(100.0);
            // edge->setRobustKernel(rk);
            optimizer.addEdge(edge);
          }
        }
      }
    }
  }

  void MapProcessor::countObservations(json& camera_points) {
    for (auto& el : camera_points.items()) {
      json camera_point = el.value();
      int feature_point_count = 0;
      if (camera_point.contains("featurePoints") && camera_point["featurePoints"].size() > 0) {
        for (auto& el : camera_point["featurePoints"].items()) {
          json feature_point = el.value();
          std::string identifier = feature_point["identifier"];
          observation_count[identifier]++;
          feature_point_count++;
        }
      }

      std::string identifier = camera_point["identifier"];
      observation_count[identifier] = feature_point_count;
    }
  }
}