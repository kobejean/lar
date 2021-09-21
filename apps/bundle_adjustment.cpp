// g2o - General Graph Optimization
// Copyright (C) 2011 H. Strasdat
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
// TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
#include "g2o/core/factory.h"

#if defined G2O_HAVE_CHOLMOD
G2O_USE_OPTIMIZATION_LIBRARY(cholmod);
#else
G2O_USE_OPTIMIZATION_LIBRARY(eigen);
#endif

G2O_USE_OPTIMIZATION_LIBRARY(dense);

using namespace Eigen;
using namespace std;
using json = nlohmann::json;

int main(int argc, const char* argv[]){
  std::ifstream ifs("../input/map.json");
  if (ifs.fail()) {
    std::cout << "Could not read file at '../input/map.json'" << '\n';
  }

  json map = json::parse(ifs);
  json pointCloud = map["pointCloud"];
  json cameraPoints = pointCloud["cameraPoints"];
  json featurePoints = pointCloud["featurePoints"];
  json locationPoints = pointCloud["locationPoints"];
  
  for (auto& el : cameraPoints[0].items()) {
    std::cout << "cameraPoints." << el.key() << " = " << el.value() << '\n';
  }
  for (auto& el : cameraPoints[2]["featurePoints"][0].items()) {
    std::cout << "cameraPoints[*].featurePoints[*]." << el.key() << '\n';
  }
  for (auto& el : cameraPoints[2]["intrinsics"].items()) {
    std::cout << "cameraPoints[*].intrinsics." << el.key() << '\n';
  }
  for (auto& el : cameraPoints[2]["intrinsics"]["principlePoint"].items()) {
    std::cout << "cameraPoints[*].intrinsics.principlePoint." << el.key() << '\n';
  }
  for (auto& el : featurePoints[0].items()) {
    std::cout << "featurePoints[*]." << el.key() << '\n';
  }
  for (auto& el : locationPoints[0].items()) {
    std::cout << "locationPoints[*]." << el.key() << '\n';
  }

  g2o::SparseOptimizer optimizer;
  optimizer.setVerbose(false);
  string solverName = "lm_fix6_3";
#ifdef G2O_HAVE_CHOLMOD
  solverName = "lm_fix6_3_cholmod";
#else
  solverName = "lm_fix6_3";
#endif

  g2o::OptimizationAlgorithmProperty solverProperty;
  auto algorithm = g2o::OptimizationAlgorithmFactory::instance()->construct(solverName, solverProperty);
  optimizer.setAlgorithm(algorithm);

  int vertex_id = 0;
  std::map<std::string, int> vertex_ids;
  std::map<std::string, Vector3d> original_points;
  std::map<std::string, g2o::SE3Quat> original_poses;

  // Feature Points
  for (auto& el : featurePoints.items()) {
    json featurePoint = el.value();
    json transform = featurePoint["transform"];

    Vector3d trans(transform[3][0], transform[3][1], transform[3][2]);

    g2o::VertexPointXYZ * v_p = new g2o::VertexPointXYZ();
    v_p->setId(vertex_id);
    // v_p->setMarginalized(true);
    v_p->setEstimate(trans);

    optimizer.addVertex(v_p);
    
    auto identifier = featurePoint["identifier"];
    original_points[identifier] = trans;
    vertex_ids[identifier] = vertex_id;
    vertex_id++;
  }

  // Camera Points
  int camera_points = 0;
  for (auto& el : cameraPoints.items()) {
    std::string cameraIndex = el.key();
    json cameraPoint = el.value();

    if (cameraPoint.contains("featurePoints") && cameraPoint["featurePoints"].size() > 0) {
      json transform = cameraPoint["transform"];

      Matrix3d rot;
      rot << transform[0][0], transform[1][0], transform[2][0],
           transform[0][1], transform[1][1], transform[2][1],
           transform[0][2], transform[1][2], transform[2][2]; 
      Vector3d trans(transform[3][0], transform[3][1], transform[3][2]);

      Eigen::Quaterniond q(rot);
      g2o::SE3Quat pose(q, trans);

      g2o::VertexSE3Expmap * v_se3 = new g2o::VertexSE3Expmap();
      v_se3->setId(vertex_id);
      v_se3->setEstimate(pose);
      if (camera_points<1){
        v_se3->setFixed(true);
      }
      optimizer.addVertex(v_se3);

      if (cameraPoint > 0) {
        g2o::EdgeSE3Expmap * e = new g2o::EdgeSE3Expmap();
        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(vertex_id-1)));
        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(vertex_id)));
        e->information() = Eigen::MatrixXd::Identity(6,6) * 2;
        optimizer.addEdge(e);
      }


      json intrinsics = cameraPoint["intrinsics"];
      double focalLength = intrinsics["focalLength"];
      Vector2d principlePoint(intrinsics["principlePoint"]["x"], intrinsics["principlePoint"]["y"]);

      auto * cameraParams = new g2o::CameraParameters(focalLength, principlePoint, 0.);
      cameraParams->setId(camera_points);
      if (!optimizer.addParameter(cameraParams)) {
        assert(false);
      }

      for (auto& el : cameraPoint["featurePoints"].items()) {
        json featurePoint = el.value();
        std::string identifier = featurePoint["identifier"];
        int id = vertex_ids[identifier];
        json keyPoint = featurePoint["keyPoint"];
        Vector2d z = Vector2d(keyPoint["x"], keyPoint["y"]);

        if (z[0]>=0 && z[1]>=0 && z[0]<principlePoint[0]*2 && z[1]<principlePoint[1]*2) {
          g2o::EdgeProjectXYZ2UV * e = new g2o::EdgeProjectXYZ2UV();
          e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(vertex_id)));
          e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id)));
          e->setMeasurement(z);
          e->information() = Matrix2d::Identity() * 100;
          e->setParameterId(0, camera_points);
          optimizer.addEdge(e);
        }
      }

      auto identifier = cameraPoint["identifier"];
      original_poses[identifier] = pose;
      vertex_ids[identifier] = vertex_id;
      vertex_id++;
      camera_points++;
    }
  }

  optimizer.initializeOptimization();
  optimizer.setVerbose(true);

  optimizer.save("../output/map.g2o");
  cout << endl;
  cout << "Performing full BA:" << endl;
  optimizer.optimize(10);
  cout << endl;
  // cout << "Point error before optimisation (inliers only): " << sqrt(sum_diff2/inliers.size()) << endl;
  // point_num = 0;
  // sum_diff2 = 0;
}
