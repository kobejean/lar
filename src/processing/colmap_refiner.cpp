// Only compile COLMAP refiner on desktop platforms (no SQLite on iOS/iPadOS)
#if !defined(__APPLE__) || !TARGET_OS_IPHONE

#include <filesystem>
#include <fstream>
#include <sstream>
#include <map>
#include <sqlite3.h>

#include <opencv2/opencv.hpp>

#include "lar/processing/colmap_refiner.h"
#include "lar/core/utils/json.h"
#include "lar/mapping/location_matcher.h"

namespace lar {
  // map contains landmarks positioned by colmap, we will register images against the 3D point cloud
  // then use those localizations as initial pose for bundle adjustment
  // data->frames contains original ARKit data, we want to use the relative transforms between consecutive poses as odometry measurements
  ColmapRefiner::ColmapRefiner(std::shared_ptr<Mapper::Data> data) :
    data(data), tracker(data->map), bundle_adjustment(data), global_alignment(data) {
  }

  void ColmapRefiner::process() {
    // Update GPS alignment
    global_alignment.updateAlignment();
    std::cout << data->map.origin.matrix() << std::endl;
    localizations.clear();
    localizations.reserve(data->frames.size());

    // Localize frames
    for (auto& frame : data->frames) {
      std::cout << std::endl << "LOCALIZING FRAME " << frame.id << std::endl;
      std::string path_prefix = data->getPathPrefix(frame.id).string();
      std::string img_filepath = path_prefix + "image.jpeg";
      cv::Mat image = cv::imread(img_filepath, cv::IMREAD_GRAYSCALE);
      Eigen::Matrix4d extrinsics;
      // Use frame position for spatial query
      double query_x = frame.extrinsics(0, 3);
      double query_z = frame.extrinsics(2, 3);
      double query_diameter = 50.0; // 50 meter search radius
      
      if (tracker.localize(image, frame, query_x, query_z, query_diameter, extrinsics)) {
        std::cout << "extrinsics:" << extrinsics << std::endl;
        localizations.push_back(extrinsics);
      } else {
        localizations.push_back(frame.extrinsics);
      }
    }
    
  }

  void ColmapRefiner::processWithColmapData(const std::string& colmap_dir) {
    // Update GPS alignment
    global_alignment.updateAlignment();
    std::cout << data->map.origin.matrix() << std::endl;
    
    // Read COLMAP database and sparse reconstruction
    std::string database_path = colmap_dir + "/database.db";
    if (!colmap_db.readDatabase(database_path)) {
      std::cout << "Failed to read COLMAP database" << std::endl;
      return;
    }

    // Read from sparse reconstruction directory (aligned in place)
    std::string sparse_reconstruction_dir = colmap_dir + "/poses_txt";

    if (!colmap_db.readSparseReconstruction(sparse_reconstruction_dir)) {
      std::cout << "Failed to read COLMAP sparse reconstruction from " << sparse_reconstruction_dir << std::endl;
      return;
    }
    
    std::cout << "Loaded sparse reconstruction from " << sparse_reconstruction_dir << std::endl;

    // Construct complete landmarks from COLMAP data
    std::vector<Landmark> landmarks;
    colmap_db.constructLandmarksFromColmap(data->frames, landmarks, database_path);
    data->map.landmarks.insert(landmarks);

    // Extract camera poses for frames
    localizations.clear();
    localizations.reserve(data->frames.size());

    for (const auto& frame : data->frames) {
      std::string path_prefix = data->getPathPrefix(frame.id).string();
      std::string img_filename = std::filesystem::path(path_prefix + "image.jpeg").filename().string();
      
      const ColmapImage& colmap_img = colmap_db.images[frame.id+1];
      if (colmap_img.name == img_filename && colmap_img.positioned) {
        localizations.push_back(colmap_img.pose);
      } else {
        localizations.push_back(frame.extrinsics);
        std::cout << "No COLMAP pose found for frame " << frame.id << ", using ARKit pose" << std::endl;
      }
    }
  }

  void ColmapRefiner::optimize() {
    // Optimize
    bundle_adjustment.reset();
    // Construct first so that original ARKit transforms/extrinsics are used to compute odometry edges
    bundle_adjustment.construct(); 
    // Then we update the pose with our localizations before optimizing
    for (size_t i = 0; i < localizations.size(); i ++) {
      const Frame &frame = data->frames[i];
      g2o::VertexSE3Expmap* v = dynamic_cast<g2o::VertexSE3Expmap*>(bundle_adjustment.optimizer.vertex(frame.id));
      g2o::SE3Quat pose = BundleAdjustment::poseFromExtrinsics(localizations[i]);
      v->setEstimate(pose);
    }
    bundle_adjustment.optimize();

    // Save optimized poses from first bundle adjustment
    std::vector<Eigen::Matrix4d> optimized_poses(data->frames.size());
    for (size_t i = 0; i < data->frames.size(); i++) {
      g2o::VertexSE3Expmap* v = dynamic_cast<g2o::VertexSE3Expmap*>(bundle_adjustment.optimizer.vertex(i));
      if (v) {
        optimized_poses[i] = BundleAdjustment::extrinsicsFromPose(v->estimate());
      } else {
        optimized_poses[i] = data->frames[i].extrinsics; // fallback
      }
    }

    bundle_adjustment.update(0.2);

    // // Secondary localization step: use current camera poses for spatial queries
    // std::cout << "Starting secondary localization with refined poses..." << std::endl;

    // bool new_observations_added = false;
    // double query_diameter = 50.0; // 50 meter search radius as specified
    // double pose_tolerance = 2.0; // 2 meter position tolerance for pose validation

    // for (size_t i = 0; i < data->frames.size(); i++) {
    //   const Frame &frame = data->frames[i];

    //   // Get current optimized camera pose from bundle adjustment
    //   g2o::VertexSE3Expmap* v = dynamic_cast<g2o::VertexSE3Expmap*>(bundle_adjustment.optimizer.vertex(frame.id));
    //   if (!v) continue;

    //   Eigen::Matrix4d current_pose = BundleAdjustment::extrinsicsFromPose(v->estimate());
    //   double query_x = current_pose(0, 3);
    //   double query_z = current_pose(2, 3);

    //   // Load frame image
    //   std::string path_prefix = data->getPathPrefix(frame.id).string();
    //   std::string img_filepath = path_prefix + "image.jpeg";
    //   cv::Mat image = cv::imread(img_filepath, cv::IMREAD_GRAYSCALE);
    //   if (image.empty()) continue;

    //   // Perform localization with spatial query
    //   Eigen::Matrix4d localized_pose;
    //   if (tracker.localize(image, frame, query_x, query_z, query_diameter, localized_pose)) {

    //     // Validate pose by comparing with current camera pose
    //     Eigen::Vector3d current_pos = current_pose.block<3,1>(0,3);
    //     Eigen::Vector3d localized_pos = localized_pose.block<3,1>(0,3);
    //     double pose_distance = (current_pos - localized_pos).norm();

    //     if (pose_distance <= pose_tolerance) {
    //       int new_obs_count = 0;

    //       // Add new landmark observations from inliers
    //       for (const auto& inlier_pair : tracker.inliers) {
    //         Landmark* landmark = inlier_pair.first;
    //         const cv::KeyPoint& kpt = inlier_pair.second;

    //         // Check if this observation already exists
    //         bool observation_exists = false;
    //         for (const auto& obs : landmark->obs) {
    //           if (obs.frame_id == frame.id) {
    //             observation_exists = true;
    //             break;
    //           }
    //         }

    //         // Add new observation if it doesn't exist
    //         if (!observation_exists) {
    //           Landmark::Observation new_obs;
    //           new_obs.frame_id = frame.id;
    //           new_obs.timestamp = frame.timestamp;
    //           new_obs.cam_pose = current_pose;
    //           new_obs.kpt = kpt;
    //           new_obs.depth = 0.0; // No depth info from localization
    //           new_obs.depth_confidence = 0.0;
    //           new_obs.surface_normal = Eigen::Vector3f(0.0f, 0.0f, 1.0f); // Default

    //           landmark->obs.push_back(new_obs);
    //           landmark->sightings++;
    //           new_observations_added = true;
    //           new_obs_count++;
    //         }
    //       }

    //       std::cout << "Frame " << frame.id << ": pose validated (" << pose_distance << "m), added "
    //                 << new_obs_count << "/" << tracker.inliers.size() << " new observations" << std::endl;
    //     } else {
    //       std::cout << "Frame " << frame.id << ": pose validation failed (" << pose_distance << "m > "
    //                 << pose_tolerance << "m)" << std::endl;
    //     }
    //   }
    // }

    // // Perform secondary bundle adjustment if new observations were added
    // if (new_observations_added) {
    //   std::cout << "Performing secondary bundle adjustment with new observations..." << std::endl;

    //   // Reconstruct and optimize again
    //   bundle_adjustment.reset();
    //   bundle_adjustment.construct();

    //   // Initialize poses from first optimization results instead of original ARKit poses
    //   for (size_t i = 0; i < data->frames.size(); i++) {
    //     const Frame &frame = data->frames[i];
    //     g2o::VertexSE3Expmap* v = dynamic_cast<g2o::VertexSE3Expmap*>(bundle_adjustment.optimizer.vertex(frame.id));
    //     if (v) {
    //       g2o::SE3Quat pose = BundleAdjustment::poseFromExtrinsics(optimized_poses[i]);
    //       v->setEstimate(pose);
    //     }
    //   }

    //   bundle_adjustment.optimize();
    //   bundle_adjustment.update(0.2); // Use final margin ratio as specified
    // } else {
    //   std::cout << "No new observations added, using initial optimization results" << std::endl;
    //   bundle_adjustment.update(0.2); // Still need to update with final margin ratio
    // }
  }

  void ColmapRefiner::rescale(double scale_factor) {
    if (scale_factor <= 0.0) {
      std::cout << "Invalid scale factor: " << scale_factor << std::endl;
      return;
    }
    
    std::cout << "Manual rescaling by factor: " << scale_factor << std::endl;
    
    // Ensure bundle adjustment is constructed before rescaling
    bundle_adjustment.reset();
    bundle_adjustment.construct();
    
    // Perform rescaling using the core implementation
    bundle_adjustment.performRescaling(scale_factor);
    
    // Update the data structures with scaled values
    bundle_adjustment.update();
    
    std::cout << "Manual rescaling complete" << std::endl;
  }

  void ColmapRefiner::saveMap(std::string dir) {
    std::filesystem::create_directory(dir);
    std::string output = dir + "/map.g2o";

    std::cout << std::endl;
    bundle_adjustment.optimizer.save(output.c_str());
    std::cout << "Saved g2o file to: " << output << std::endl;

    data->map.landmarks.cull();

    for (size_t i = 0; i < localizations.size(); i ++) {
      Frame &frame = data->frames[i];
      g2o::VertexSE3Expmap* v = dynamic_cast<g2o::VertexSE3Expmap*>(bundle_adjustment.optimizer.vertex(frame.id));
      frame.extrinsics = BundleAdjustment::extrinsicsFromPose(v->estimate());
    }
    
    // Re-interpolate GPS observations using updated frame positions
    LocationMatcher temp_matcher;
    temp_matcher.matches = data->gps_obs;
    temp_matcher.reinterpolateMatches(data->frames);
    data->gps_obs = temp_matcher.matches;
    
    global_alignment.updateAlignment();

    // Serialize
    nlohmann::json frames_json = data->frames;
    std::ofstream(dir + "/frames.json") << frames_json << std::endl;
    
    nlohmann::json gps_json = data->gps_obs;
    std::ofstream(dir + "/gps.json") << gps_json << std::endl;

    nlohmann::json map_json = data->map;
    std::ofstream(dir + "/map.json") << map_json << std::endl;
  }

}

#endif // !defined(__APPLE__) || !TARGET_OS_IPHONE
