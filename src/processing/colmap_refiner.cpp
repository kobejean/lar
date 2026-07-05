// Only compile COLMAP refiner on desktop platforms (no SQLite on iOS/iPadOS)
#if !defined(__APPLE__) || !TARGET_OS_IPHONE

#include <filesystem>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstdio>
#include <array>
#include <map>
#include <unordered_map>
#include <sqlite3.h>

#include <opencv2/opencv.hpp>
#include <Eigen/Geometry>

#include "lar/io/image_io.h"
#include "lar/processing/colmap_refiner.h"
#include "lar/core/utils/json.h"
#include "lar/core/utils/transform.h"
#include "lar/mapping/location_matcher.h"

namespace lar {
  // map contains landmarks positioned by colmap, we will register images against the 3D point cloud
  // then use those localizations as initial pose for bundle adjustment
  // data->frames contains original ARKit data, we want to use the relative transforms between consecutive poses as odometry measurements
  ColmapRefiner::ColmapRefiner(std::shared_ptr<Mapper::Data> data) :
    data(data), tracker(data->map, cv::Size(1920, 1440)), bundle_adjustment(data), global_alignment(data) {
    // Note: Using typical ARKit image size (1920x1440). Tracker will verify dimensions on first localize() call.
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
      cv::Mat image = lar::io::imreadGray(img_filepath);
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
      g2o::SE3Quat pose = utils::TransformUtils::arkitToG2oPose(localizations[i]);
      v->setEstimate(pose);
    }
    bundle_adjustment.optimize();

    // Save optimized poses from first bundle adjustment
    std::vector<Eigen::Matrix4d> optimized_poses(data->frames.size());
    for (size_t i = 0; i < data->frames.size(); i++) {
      g2o::VertexSE3Expmap* v = dynamic_cast<g2o::VertexSE3Expmap*>(bundle_adjustment.optimizer.vertex(i));
      if (v) {
        optimized_poses[i] = utils::TransformUtils::g2oToArkitPose(v->estimate());
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

    // Update the data structures with scaled values (including bounds rescaling)
    bundle_adjustment.updateAfterRescaling(scale_factor);
    
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
      frame.extrinsics = utils::TransformUtils::g2oToArkitPose(v->estimate());
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

    // Also emit a COLMAP sparse model of the refined result for visual inspection.
    saveColmapModel(dir);
  }

  // Write the refined poses + landmarks as a COLMAP sparse text model at
  // <dir>/colmap/sparse/0. Poses come from the refined frame extrinsics (ARKit
  // camera-to-world) converted to COLMAP world-to-camera with the camera-axis
  // flip F = diag(1,-1,-1) (R_w2c = F * R_c2w^T, t = -R_w2c * C); landmark tracks
  // and 2D points are rebuilt from each landmark's observations; point colors are
  // sampled from the source images. Open with:
  //   colmap gui --database_path <session>/colmap/database.db
  //              --import_path <dir>/colmap/sparse/0 --image_path <session>/colmap
  void ColmapRefiner::saveColmapModel(std::string dir) {
    namespace fs = std::filesystem;
    fs::path model_dir = fs::path(dir) / "colmap" / "sparse" / "0";
    fs::create_directories(model_dir);

    const auto& frames = data->frames;
    const size_t n = frames.size();

    // frame.id -> contiguous index (ids are 0..n-1 in practice; map to be safe).
    std::unordered_map<size_t, size_t> id_to_index;
    for (size_t i = 0; i < n; i++) id_to_index[frames[i].id] = i;

    // Per-image POINTS2D lists and the landmark tracks that reference them by index.
    struct Pt2D { float x, y; size_t point3d_id; };
    std::vector<std::vector<Pt2D>> image_points2d(n);
    std::unordered_map<size_t, std::vector<std::pair<size_t, size_t>>> tracks; // lm id -> [(image_id, point2d_idx)]

    std::vector<Landmark*> lms = data->map.landmarks.all();
    for (Landmark* lm : lms) {
      for (const auto& obs : lm->obs) {
        auto it = id_to_index.find(obs.frame_id);
        if (it == id_to_index.end()) continue;
        std::vector<Pt2D>& vec = image_points2d[it->second];
        size_t image_id = obs.frame_id + 1;  // 1-indexed COLMAP image id
        tracks[lm->id].push_back({image_id, vec.size()});
        vec.push_back({obs.kpt.pt.x, obs.kpt.pt.y, lm->id});
      }
    }

    std::unordered_map<size_t, std::array<int, 3>> colors;  // lm id -> RGB

    std::ofstream fc(model_dir / "cameras.txt");
    std::ofstream fi(model_dir / "images.txt");
    fc << std::setprecision(9);
    fi << std::setprecision(9);
    fc << "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n";
    fi << "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
       << "#   POINTS2D[] as (X, Y, POINT3D_ID)\n";

    for (size_t i = 0; i < n; i++) {
      const Frame& frame = frames[i];
      size_t image_id = frame.id + 1;

      // Load the source image once: yields width/height + per-point color.
      std::string img_path = data->getPathPrefix(frame.id).string() + "image.jpeg";
      cv::Mat img = lar::io::imreadColor(img_path);  // BGR, empty on failure
      const Eigen::Matrix3d& K = frame.intrinsics;
      int w = img.empty() ? static_cast<int>(std::lround(K(0, 2) * 2.0)) : img.cols;
      int h = img.empty() ? static_cast<int>(std::lround(K(1, 2) * 2.0)) : img.rows;

      fc << image_id << " PINHOLE " << w << " " << h << " "
         << K(0, 0) << " " << K(1, 1) << " " << K(0, 2) << " " << K(1, 2) << "\n";

      // Refined ARKit camera-to-world -> COLMAP world-to-camera (shared SSOT helper).
      auto [R_w2c, t] = ColmapDatabase::arkitToColmapWorldToCamera(frame.extrinsics);
      Eigen::Quaterniond q(R_w2c);
      q.normalize();

      char name[32];
      std::snprintf(name, sizeof(name), "%08zu_image.jpeg", frame.id);
      fi << image_id << " " << q.w() << " " << q.x() << " " << q.y() << " " << q.z()
         << " " << t.x() << " " << t.y() << " " << t.z() << " " << image_id << " " << name << "\n";

      const std::vector<Pt2D>& pts = image_points2d[i];
      for (size_t k = 0; k < pts.size(); k++) {
        const Pt2D& p = pts[k];
        fi << p.x << " " << p.y << " " << p.point3d_id << (k + 1 < pts.size() ? " " : "");
        if (!img.empty() && colors.find(p.point3d_id) == colors.end()) {
          int px = static_cast<int>(std::lround(p.x)), py = static_cast<int>(std::lround(p.y));
          if (px >= 0 && px < w && py >= 0 && py < h) {
            const cv::Vec3b& bgr = img.at<cv::Vec3b>(py, px);
            colors[p.point3d_id] = {bgr[2], bgr[1], bgr[0]};
          }
        }
      }
      fi << "\n";  // POINTS2D line (may be empty)
    }

    std::ofstream fp(model_dir / "points3D.txt");
    fp << std::setprecision(9);
    fp << "# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n";
    size_t written = 0;
    for (Landmark* lm : lms) {
      auto ct = tracks.find(lm->id);
      if (ct == tracks.end()) continue;  // no surviving observations
      std::array<int, 3> col = colors.count(lm->id) ? colors[lm->id] : std::array<int, 3>{180, 180, 180};
      fp << lm->id << " " << lm->position.x() << " " << lm->position.y() << " " << lm->position.z()
         << " " << col[0] << " " << col[1] << " " << col[2] << " 1.0";
      for (const auto& [image_id, p2d_idx] : ct->second) fp << " " << image_id << " " << p2d_idx;
      fp << "\n";
      written++;
    }

    std::cout << "Wrote refined COLMAP model to " << model_dir.string()
              << " (" << n << " images, " << written << " points)" << std::endl;
  }

}

#endif // !defined(__APPLE__) || !TARGET_OS_IPHONE
