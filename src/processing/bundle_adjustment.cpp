#include <stdint.h>
#include <iostream>
#include <cmath>
#include <unordered_map>

#include "g2o/core/factory.h"
#include "g2o/core/optimization_algorithm_factory.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/types/sba/types_six_dof_expmap.h"
#include "g2o/types/sba/edge_se3_expmap_gravity.h"
#include "lar/processing/bundle_adjustment.h"
#include "lar/core/utils/transform.h"

// Information matrix strategy constants - uncomment to try different approaches
// #define INFORMATION_STRATEGY_UNIFORM        // Uniform information matrix
// #define INFORMATION_STRATEGY_SIZE_LINEAR    // Linear scaling by keypoint size
// #define INFORMATION_STRATEGY_SIZE_QUADRATIC // Quadratic scaling by keypoint size
// #define INFORMATION_STRATEGY_SIZE_INVERSE   // Inverse scaling by keypoint size
#define INFORMATION_STRATEGY_PYRAMID        // Pyramid-level based (ORB-SLAM style)
// #define INFORMATION_STRATEGY_RESPONSE       // Response-weighted
// #define INFORMATION_STRATEGY_ANISOTROPIC    // Anisotropic based on keypoint angle

// Strategy parameters
static constexpr double BASE_PIXEL_STDDEV = 0.5;
static constexpr double BASE_INFORMATION = 1.0 / (BASE_PIXEL_STDDEV * BASE_PIXEL_STDDEV);
static constexpr double SIZE_SCALING_FACTOR = 1.0;        // Scaling factor for size-based methods
static constexpr double BASE_KEYPOINT_SIZE = 3.0;         // Base keypoint size for pyramid calculation
static constexpr double PYRAMID_SCALE_FACTOR = 1.2599210498948732;  // 2^(1/3) - SIFT scale between pyramid levels
static constexpr double RESPONSE_WEIGHT = 0.3;            // Weight for response-based scaling
static constexpr double ANISOTROPY_RATIO = 2.0;           // Major/minor axis ratio for anisotropic
static constexpr double MIN_INFORMATION = 0.1;            // Minimum information value
static constexpr double MAX_INFORMATION = 100.0;          // Maximum information value

G2O_USE_OPTIMIZATION_LIBRARY(eigen);

namespace g2o {
  G2O_REGISTER_TYPE_GROUP(expmap);
  G2O_REGISTER_TYPE(PARAMS_CAMERAPARAMETERS, CameraParameters);
  G2O_REGISTER_TYPE(VERTEX_SE3:EXPMAP, VertexSE3Expmap);
  G2O_REGISTER_TYPE(EDGE_SE3:EXPMAP, EdgeSE3Expmap);
  G2O_REGISTER_TYPE(EDGE_SE3:EXPMAP:GRAVITY, EdgeSE3ExpmapGravity);
  G2O_REGISTER_TYPE(EDGE_PROJECT_XYZ2UVD:EXPMAP, EdgeProjectXYZ2UVD);

  G2O_REGISTER_TYPE_GROUP(slam3d);
  G2O_REGISTER_TYPE(VERTEX_TRACKXYZ, VertexPointXYZ);
}

namespace lar {

  BundleAdjustment::BundleAdjustment(std::shared_ptr<Mapper::Data> data) : data(data) {
    optimizer.setVerbose(true);
    std::string solver_name = "lm_fix6_3";
    g2o::OptimizationAlgorithmProperty solver_property;
    auto algorithm = g2o::OptimizationAlgorithmFactory::instance()->construct(solver_name, solver_property);
    optimizer.setAlgorithm(algorithm);
  }

  void BundleAdjustment::construct() {
    _stats = Stats();
    _stats.landmarks = std::vector<size_t>(data->frames.size(),0);
    _stats.usable_landmarks = std::vector<size_t>(data->frames.size(),0);

    // Find pose with most landmark observations (most constrained)
    size_t pose_to_fix = findMostConnectedPose();

    // Use frame data to add poses and measurements
    for (size_t frame_id = 0; frame_id < data->frames.size(); frame_id++) {
      // Add camera pose vertex
      Frame const& frame = data->frames[frame_id];
      addPose(frame.extrinsics, frame_id, frame_id == pose_to_fix);

      // Add odometry measurement edge if not first frame
      if (frame_id > 0) {
        addOdometry(frame_id);
      }

      if (frame_id < data->frames.size() - 1) {
        addGravityConstraint(frame_id);
      }

      // Add camera intrinsics parameters
      size_t params_id = frame_id+1;
      addIntrinsics(frame.intrinsics, params_id);
    }

    // Add landmarks to graph
    for (Landmark* landmark : data->map.landmarks.all()) {
      size_t id = landmark->id + data->frames.size();
      _stats.total_usable_landmarks += addLandmark(*landmark, id);
      addLandmarkMeasurements(*landmark, id);
    }
    
    // Print statistics for debuging purposes
    _stats.print();
  }

  void BundleAdjustment::reset() {
    optimizer.clear();
    optimizer.clearParameters();
    _landmark_edges.clear();
    _odometry_edges.clear();
    _landmark_vertices.clear();
  }

  void BundleAdjustment::optimize() {
    printReprojectionError();
    for (auto edge : _landmark_edges) {
      g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
      rk->setDelta(sqrt(5.991));
      edge->setRobustKernel(rk);
    }
    // rescaleToMatchOdometry();
    // Stage 1: Optimize poses only with landmarks fixed (pose alignment)
    std::cout << "Stage 1: Pose-only optimization (landmarks fixed)..." << std::endl;
    fixAllLandmarks(true);
    optimizer.initializeOptimization(0);
    optimizer.optimize(50);
    fixAllLandmarks(false);
    printReprojectionError();
    // markOutliers(1e20, 3*7.378);
    
    // Stage 2: Main optimization rounds
    constexpr size_t rounds = 4;
    double chi_threshold[4] = { 7.378, 5.991, 4.605, 3.841 };
    double odometry_chi_threshold[4] = { 7.378, 5.991, 4.605, 3.841 };
    size_t iteration[4] = { 300, 100, 60, 60 };

    for (size_t i = 0; i < rounds; i++) {
      std::cout << "Stage 2." << (i+1) << ": Full optimization..." << std::endl;
      optimizer.initializeOptimization(0);
      optimizer.optimize(iteration[i]);
      printReprojectionError();
      // rescaleToMatchOdometry();
      markOutliers(chi_threshold[i], 2.3*odometry_chi_threshold[i]);
    }

    // // Final round: remove robust kernels for fine-tuning
    // for (auto edge : _landmark_edges) {
    //   if (edge->robustKernel() != nullptr) {
    //     edge->setRobustKernel(nullptr);
    //   }
    // }
    // for (auto edge : _odometry_edges) {
    //   if (edge->robustKernel() != nullptr) {
    //     edge->setRobustKernel(nullptr);
    //   }
    // }

    markOutliers(3.841);
    optimizer.initializeOptimization(0);
    optimizer.optimize(100);
    printReprojectionError();
    markOutliers(3.841);
    printReprojectionError();
    // rescaleToMatchOdometry();
  }

  void BundleAdjustment::update(double marginRatio) {
    updateLandmarks(marginRatio);
    updateAnchors();
  }

  // Private methods

  size_t BundleAdjustment::findMostConnectedPose() {
    std::vector<size_t> observation_counts(data->frames.size(), 0);
    
    // Count landmark observations per frame
    for (Landmark* landmark : data->map.landmarks.all()) {
      if (landmark->isUseable()) {
        for (auto const &obs : landmark->obs) {
          if (obs.frame_id < observation_counts.size()) {
            observation_counts[obs.frame_id]++;
          }
        }
      }
    }
    
    // Find pose with maximum observations
    size_t best_pose = 0;
    size_t max_observations = 0;
    for (size_t i = 0; i < observation_counts.size(); i++) {
      if (observation_counts[i] > max_observations) {
        max_observations = observation_counts[i];
        best_pose = i;
      }
    }
    
    std::cout << "Selected pose " << best_pose << " to fix (" 
              << max_observations << " landmark observations)" << std::endl;
    return best_pose;
  }

  void BundleAdjustment::fixAllLandmarks(bool fixed) {
    for (auto vertex : _landmark_vertices) {
      vertex->setFixed(fixed);
    }
    if (fixed) {
      std::cout << "Fixed all landmark vertices" << std::endl;
    } else {
      std::cout << "Unfixed all landmark vertices" << std::endl;
    }
  }

  void BundleAdjustment::fixAllPoses(bool fixed) {
    size_t anchor_pose = findMostConnectedPose(); // Keep the anchor pose always fixed
    
    for (size_t frame_id = 0; frame_id < data->frames.size(); frame_id++) {
      g2o::VertexSE3Expmap* vertex = dynamic_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(frame_id));
      if (vertex) {
        // Always keep the anchor pose fixed for gauge freedom
        bool should_fix = fixed || (frame_id == anchor_pose);
        vertex->setFixed(should_fix);
      }
    }
    if (fixed) {
      std::cout << "Fixed all pose vertices (except anchor)" << std::endl;
    } else {
      std::cout << "Unfixed all pose vertices (keeping anchor fixed)" << std::endl;
    }
  }

  void BundleAdjustment::rescaleToMatchOdometry() {
    if (_odometry_edges.empty()) {
      std::cout << "No odometry edges for rescaling" << std::endl;
      return;
    }
    
    double total_optimized_distance = 0.0;
    double total_odometry_distance = 0.0;
    std::vector<double> scale_ratios;
    
    // Use actual odometry measurements from the edges
    for (size_t i = 0; i < _odometry_edges.size(); i++) {
      auto edge = _odometry_edges[i];
      g2o::VertexSE3Expmap* v1 = dynamic_cast<g2o::VertexSE3Expmap*>(edge->vertex(0));
      g2o::VertexSE3Expmap* v2 = dynamic_cast<g2o::VertexSE3Expmap*>(edge->vertex(1));
      
      if (v1 && v2) {
        // Current optimized camera positions in world coordinates
        Eigen::Vector3d pos1 = v1->estimate().inverse().translation();
        Eigen::Vector3d pos2 = v2->estimate().inverse().translation();
        double optimized_distance = (pos2 - pos1).norm();
        
        // ARKit camera positions in world coordinates
        Eigen::Vector3d arkit_pos1 = data->frames[v1->id()].extrinsics.block<3,1>(0,3);
        Eigen::Vector3d arkit_pos2 = data->frames[v2->id()].extrinsics.block<3,1>(0,3);
        double arkit_distance = (arkit_pos2 - arkit_pos1).norm();
        
        // Filter out very small movements to avoid noise
        double min_movement_threshold = 0.1; // 10cm minimum movement
        if (optimized_distance > min_movement_threshold && arkit_distance > min_movement_threshold) {
          total_optimized_distance += optimized_distance;
          total_odometry_distance += arkit_distance;
          scale_ratios.push_back(arkit_distance / optimized_distance);
        }
      }
    }
    
    if (scale_ratios.empty()) {
      std::cout << "No valid scale ratios for rescaling" << std::endl;
      return;
    }
    
    // Use median ratio for more robust scaling
    std::sort(scale_ratios.begin(), scale_ratios.end());
    double scale_factor = scale_ratios[scale_ratios.size() / 2];
    std::cout << "Rescaling by factor: " << scale_factor << std::endl;
    
    // Delegate to common rescaling implementation
    performRescaling(scale_factor);
  }

  void BundleAdjustment::performRescaling(double scale_factor) {
    // Find anchor pose to keep fixed during rescaling
    size_t anchor_pose = findMostConnectedPose();
    g2o::VertexSE3Expmap* anchor_vertex = dynamic_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(anchor_pose));
    
    if (!anchor_vertex) {
      std::cout << "Could not find anchor vertex for rescaling" << std::endl;
      return;
    }
    
    Eigen::Vector3d anchor_position = anchor_vertex->estimate().inverse().translation();
    
    // Rescale all camera positions relative to anchor
    for (size_t frame_id = 0; frame_id < data->frames.size(); frame_id++) {
      g2o::VertexSE3Expmap* vertex = dynamic_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(frame_id));
      if (vertex) {
        g2o::SE3Quat world_to_camera = vertex->estimate();
        
        // Convert to camera-to-world to get camera position
        g2o::SE3Quat camera_to_world = world_to_camera.inverse();
        Eigen::Vector3d camera_position = camera_to_world.translation();
        
        // Scale camera position relative to anchor
        Eigen::Vector3d scaled_position = anchor_position + (camera_position - anchor_position) * scale_factor;
        
        // Create new camera-to-world pose with scaled position
        g2o::SE3Quat scaled_camera_to_world(camera_to_world.rotation(), scaled_position);
        
        // Convert back to world-to-camera and store
        vertex->setEstimate(scaled_camera_to_world.inverse());
      }
    }
    
    // Rescale all landmark positions relative to anchor
    for (auto vertex : _landmark_vertices) {
      Eigen::Vector3d position = vertex->estimate();
      Eigen::Vector3d scaled_position = anchor_position + (position - anchor_position) * scale_factor;
      vertex->setEstimate(scaled_position);
    }

  }


  bool BundleAdjustment::addLandmark(Landmark const &landmark, size_t id) {
    if (landmark.isUseable()) {
      g2o::VertexPointXYZ * vertex = new g2o::VertexPointXYZ();
      vertex->setId(id);
      vertex->setMarginalized(true);
      vertex->setEstimate(landmark.position);
      optimizer.addVertex(vertex);
      _landmark_vertices.push_back(vertex);
      return true;
    }
    return false;
  }

  void BundleAdjustment::addPose(Eigen::Matrix4d const &extrinsics, size_t id, bool fixed) {
    g2o::SE3Quat pose = lar::utils::TransformUtils::arkitToG2oPose(extrinsics);
    g2o::VertexSE3Expmap * vertex = new g2o::VertexSE3Expmap();
    vertex->setId(id);
    vertex->setEstimate(pose);
    vertex->setFixed(fixed);
    optimizer.addVertex(vertex);
  }

  void BundleAdjustment::addOdometry(size_t frame_id) {
    g2o::VertexSE3Expmap* v1 = dynamic_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(frame_id-1));
    g2o::VertexSE3Expmap* v2 = dynamic_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(frame_id));
    
    // Use ARKit poses from frame data for odometry measurement
    Frame const& frame1 = data->frames[frame_id-1];
    Frame const& frame2 = data->frames[frame_id];
    
    // Convert to g2o poses first, then compute relative transform
    g2o::SE3Quat pose1 = lar::utils::TransformUtils::arkitToG2oPose(frame1.extrinsics); // converted to camera to world convention
    g2o::SE3Quat pose2 = lar::utils::TransformUtils::arkitToG2oPose(frame2.extrinsics); // converted to camera to world convention
    g2o::SE3Quat pose_change = pose2 * pose1.inverse();
    
    g2o::EdgeSE3Expmap * e = new g2o::EdgeSE3Expmap();
    e->setVertex(0, v1);
    e->setVertex(1, v2);
    e->setMeasurement(pose_change);
    double distance = pose_change.translation().norm();
    double distance_scale = 1.0 / std::max(0.1, distance);

    // Estimated uncertainties per meter of movement
    double translation_uncertainty = 0.2; // 20cm per meter
    double rotation_uncertainty_deg = 10.0; // 8 degrees per meter
    double rotation_uncertainty = rotation_uncertainty_deg * M_PI / 180.0; // Convert to radians

    Eigen::MatrixXd info = Eigen::MatrixXd::Zero(6,6);
    // Translation information: scales as 1/distance²
    info.block<3,3>(0,0) = Eigen::Matrix3d::Identity() * distance_scale * distance_scale / (translation_uncertainty*translation_uncertainty);
    // Rotation information: scales as 1/distance² (drift accumulates with time/distance)
    info.block<3,3>(3,3) = Eigen::Matrix3d::Identity() * distance_scale * distance_scale / (rotation_uncertainty*rotation_uncertainty);
    e->setInformation(info);
    // g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
    // rk->setDelta(sqrt(12.592));
    // e->setRobustKernel(rk);
    optimizer.addEdge(e);
    _odometry_edges.push_back(e);
  }
  
  void BundleAdjustment::addGravityConstraint(size_t frame_id) {
    // Get the initial pose to compute gravity direction in camera coordinates
    Frame const& frame = data->frames[frame_id];
    g2o::SE3Quat initial_pose = lar::utils::TransformUtils::arkitToG2oPose(frame.extrinsics);
    Eigen::Matrix3d R_initial = initial_pose.rotation().toRotationMatrix();
    
    // Compute gravity direction in this camera's coordinate frame
    Eigen::Vector3d world_gravity(0, -1, 0);  // g2o world gravity direction
    Eigen::Vector3d g_camera = R_initial.inverse() * world_gravity;
    
    g2o::EdgeSE3ExpmapGravity* gravity_edge = new g2o::EdgeSE3ExpmapGravity();
    
    // Connect to the pose vertex
    gravity_edge->setVertex(0, optimizer.vertex(frame_id));
    
    // Set the computed gravity direction in camera coordinates
    gravity_edge->setMeasurement(g_camera);
    
    // Set information matrix (weight of constraint)
    Eigen::Matrix3d gravity_info = Eigen::Matrix3d::Identity()*1;
    gravity_edge->setInformation(gravity_info);
    // g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
    // rk->setDelta(sqrt(7.815));
    // gravity_edge->setRobustKernel(rk);
    
    // Add to optimizer and keep track
    optimizer.addEdge(gravity_edge);
  }

  void BundleAdjustment::addIntrinsics(Eigen::Matrix3d const &intrinsics, size_t id) {
    Eigen::Vector2d principle_point(intrinsics.block<2,1>(0,2));
    auto * cam_params = new g2o::CameraParameters(intrinsics(0,0), principle_point, 0.);
    cam_params->setId(id);
    if (!optimizer.addParameter(cam_params)) {
      assert(false);
    }
  }

  void BundleAdjustment::addLandmarkMeasurements(const Landmark& landmark, size_t id) {
    for (auto const &obs : landmark.obs) {
      size_t frame_id = obs.frame_id;
      Eigen::Vector3d kp(obs.kpt.pt.x, obs.kpt.pt.y, obs.depth);
      
      if (landmark.isUseable()) {
        g2o::EdgeProjectXYZ2UVD * edge = new g2o::EdgeProjectXYZ2UVD();
        edge->setVertex(0, optimizer.vertex(id));
        edge->setVertex(1, optimizer.vertex(frame_id));
        edge->setMeasurement(kp);
        
        // Calculate XY information based on selected strategy
        double xy_information = BASE_INFORMATION;
        
        #ifdef INFORMATION_STRATEGY_UNIFORM
          xy_information = BASE_INFORMATION;
        #elif defined(INFORMATION_STRATEGY_SIZE_LINEAR)
          xy_information = BASE_INFORMATION * SIZE_SCALING_FACTOR * obs.kpt.size;
        #elif defined(INFORMATION_STRATEGY_SIZE_QUADRATIC)
          xy_information = BASE_INFORMATION * SIZE_SCALING_FACTOR * obs.kpt.size * obs.kpt.size;
        #elif defined(INFORMATION_STRATEGY_SIZE_INVERSE)
          xy_information = BASE_INFORMATION * SIZE_SCALING_FACTOR / std::max(1.0f, obs.kpt.size);
        #elif defined(INFORMATION_STRATEGY_PYRAMID)
          // Pyramid-level based (ORB-SLAM style)
          // Extract layer from packed octave field (bits 8-15)
          // kpt.octave format: octave (bits 0-7) | layer (bits 8-15) | subpixel_offset (bits 16-23)
          int layer = (obs.kpt.octave >> 8) & 255;

          // Pixel uncertainty scales as PYRAMID_SCALE_FACTOR^layer
          // Variance (sigma²) scales as PYRAMID_SCALE_FACTOR^(2*layer)
          double level_sigma2 = std::pow(PYRAMID_SCALE_FACTOR, 2.0 * layer);
          xy_information = BASE_INFORMATION / level_sigma2;
        #elif defined(INFORMATION_STRATEGY_RESPONSE)
          // Response-weighted information
          double normalized_response = std::max(0.0, std::min(1.0, obs.kpt.response / 100.0));
          double response_factor = 1.0 + RESPONSE_WEIGHT * normalized_response;
          xy_information = BASE_INFORMATION * response_factor;
        #elif defined(INFORMATION_STRATEGY_ANISOTROPIC)
          // For now, use uniform - anisotropic would need 2x2 matrix
          xy_information = BASE_INFORMATION;
        #endif
        
        // Clamp information to reasonable bounds
        xy_information = std::max(MIN_INFORMATION, std::min(MAX_INFORMATION, xy_information));
        
        // Set information matrix (u, v, depth)
        Eigen::Vector3d information_diag(xy_information, xy_information, obs.depth_confidence * 0.0);
        edge->setInformation(information_diag.asDiagonal());
        
        edge->setParameterId(0, frame_id+1);
        // g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
        // rk->setDelta(sqrt(5.991));
        // edge->setRobustKernel(rk);
        optimizer.addEdge(edge);
        _landmark_edges.push_back(edge);

        _stats.usable_landmarks[frame_id]++;
      }
      _stats.landmarks[frame_id]++;
    }
  }

  void BundleAdjustment::markOutliers(double chi_threshold) {
    markOutliers(chi_threshold, chi_threshold * 2.3); // Use higher threshold for odometry (6 DOF vs 2 DOF)
  }

  void BundleAdjustment::markOutliers(double landmark_chi_threshold, double odometry_chi_threshold) {
    int landmark_outliers = 0, odometry_outliers = 0;

    // Check landmark edges (reprojection errors - 2 DOF)
    for (auto edge : _landmark_edges) {
      edge->computeError();
      if (edge->chi2() > landmark_chi_threshold) {
        edge->setLevel(1);
        landmark_outliers++;
      } else {
        edge->setLevel(0);
      }
    }

    // Check odometry edges (pose-to-pose errors - 6 DOF)
    for (auto edge : _odometry_edges) {
      edge->computeError();
      if (edge->chi2() > odometry_chi_threshold) {
        // edge->setLevel(1);
        odometry_outliers++;
      } else {
        // edge->setLevel(0);
      }
    }

    std::cout << "Marked " << landmark_outliers << " landmark outliers, "
              << odometry_outliers << " odometry outliers" << std::endl;
  }

  void BundleAdjustment::printReprojectionError() {
    double total_error = 0.0;
    int inlier_count = 0;

    // Only compute error for level 0 (inlier) edges
    for (auto edge : _landmark_edges) {
      if (edge->level() == 0) {
        edge->computeError();
        // chi2() returns Mahalanobis distance (weighted squared error)
        // For pixel errors, we want the RMSE in pixels
        // Edge has 3 measurements (u, v, depth) but we only care about u,v
        Eigen::Vector3d error = edge->error();
        double pixel_error = std::sqrt(error[0]*error[0] + error[1]*error[1]);
        total_error += pixel_error;
        inlier_count++;
      }
    }

    if (inlier_count > 0) {
      double avg_error = total_error / inlier_count;
      std::cout << "Average reprojection error: " << avg_error
                << " pixels (" << inlier_count << " inliers)" << std::endl;
    } else {
      std::cout << "No inlier edges for reprojection error calculation" << std::endl;
    }
  }

  void BundleAdjustment::updateLandmarks(double marginRatio) {
    // Build lookup map: landmark vertex_id -> inlier camera positions (O(edges))
    std::unordered_map<size_t, std::vector<Eigen::Vector3d>> landmark_inlier_cameras;
    std::unordered_map<size_t, size_t> landmark_inlier_counts;
    
    for (auto edge : _landmark_edges) {
      if (edge->level() == 0) { // Only inliers
        g2o::VertexPointXYZ* landmark_vertex = dynamic_cast<g2o::VertexPointXYZ*>(edge->vertex(0));
        g2o::VertexSE3Expmap* pose_vertex = dynamic_cast<g2o::VertexSE3Expmap*>(edge->vertex(1));
        
        if (landmark_vertex && pose_vertex) {
          size_t vertex_id = landmark_vertex->id();
          
          // Get camera position from pose (convert g2o pose to camera position)
          g2o::SE3Quat pose = pose_vertex->estimate();
          Eigen::Matrix4d extrinsics = lar::utils::TransformUtils::g2oToArkitPose(pose);
          Eigen::Vector3d camera_pos = extrinsics.block<3,1>(0,3);
          
          landmark_inlier_cameras[vertex_id].push_back(camera_pos);
          landmark_inlier_counts[vertex_id]++;
        }
      }
    }
    
    // Update landmark positions and bounds (O(landmarks))
    for (Landmark* landmark : data->map.landmarks.all()) {
      if (landmark->isUseable()) {
        size_t vertex_id = landmark->id + data->frames.size();
        g2o::VertexPointXYZ* v = dynamic_cast<g2o::VertexPointXYZ*>(optimizer.vertex(vertex_id));
        landmark->position = v->estimate();
        
        // Get inlier camera positions from lookup map
        auto it = landmark_inlier_cameras.find(vertex_id);
        if (it != landmark_inlier_cameras.end()) {
          landmark->updateBounds(it->second, marginRatio);
          landmark->sightings = landmark_inlier_counts[vertex_id];
        } else {
          // No inliers - preserve existing bounds and set zero sightings
          landmark->sightings = 0;
        }
      }
    }
  }

  void BundleAdjustment::updateAnchors() {
    std::vector<std::reference_wrapper<Anchor>> updated_anchors;
    for (auto& it: data->map.anchors) {
      Anchor &anchor = it.second;
      size_t vertex_id = anchor.frame_id;
      g2o::VertexSE3Expmap* v = dynamic_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(vertex_id));
      Eigen::Matrix4d extrinsics = lar::utils::TransformUtils::g2oToArkitPose(v->estimate());
      Anchor::Transform transform(extrinsics * anchor.relative_transform.matrix());
      anchor.transform = transform;
      updated_anchors.push_back(anchor);
    }
    data->map.notifyDidUpdateAnchors(updated_anchors);
  }

  void BundleAdjustment::updateAfterRescaling(double scale_factor, double marginRatio) {
    // First do normal updates (positions and bounds with camera observations from optimizer)
    updateLandmarks(marginRatio);
    updateAnchors();

    // Find anchor pose used during rescaling
    size_t anchor_pose = findMostConnectedPose();
    g2o::VertexSE3Expmap* anchor_vertex = dynamic_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(anchor_pose));

    if (!anchor_vertex) {
      std::cout << "Could not find anchor vertex for rescaling" << std::endl;
      return;
    }

    Eigen::Vector3d anchor_position = anchor_vertex->estimate().inverse().translation();

    // Rescale landmarks that weren't updated by optimizer (no observations/edges)
    for (Landmark* landmark : data->map.landmarks.all()) {
      if (landmark->isUseable()) {
        // Check if this landmark was updated by the optimizer
        size_t vertex_id = landmark->id + data->frames.size();
        bool has_inlier_cameras = false;

        for (auto edge : _landmark_edges) {
          if (edge->level() == 0) { // inlier
            g2o::VertexPointXYZ* lm_vertex = dynamic_cast<g2o::VertexPointXYZ*>(edge->vertex(0));
            if (lm_vertex && static_cast<size_t>(lm_vertex->id()) == vertex_id) {
              has_inlier_cameras = true;
              break;
            }
          }
        }

        if (!has_inlier_cameras) {
          // Landmark wasn't in optimizer (no observations) - rescale position and bounds manually

          // Scale position relative to anchor
          landmark->position = anchor_position + (landmark->position - anchor_position) * scale_factor;

          // Scale bounds relative to anchor
          double min_x = landmark->bounds.lower.x;
          double min_z = landmark->bounds.lower.y;
          double max_x = landmark->bounds.upper.x;
          double max_z = landmark->bounds.upper.y;

          double scaled_min_x = anchor_position.x() + (min_x - anchor_position.x()) * scale_factor;
          double scaled_min_z = anchor_position.z() + (min_z - anchor_position.z()) * scale_factor;
          double scaled_max_x = anchor_position.x() + (max_x - anchor_position.x()) * scale_factor;
          double scaled_max_z = anchor_position.z() + (max_z - anchor_position.z()) * scale_factor;

          landmark->bounds = Rect(
            Point(scaled_min_x, scaled_min_z),
            Point(scaled_max_x, scaled_max_z)
          );
        }
      }
    }
  }


  void BundleAdjustment::Stats::print() {
    // for (size_t i = 0; i < landmarks.size(); i++) {
    //   std::cout << "frame landmarks: " << landmarks[i] << std::endl;
    //   std::cout << "frame usable landmarks: " << usable_landmarks[i] << std::endl;
    // }

    std::cout << "total usable landmarks: " << total_usable_landmarks << std::endl;
  }
}
