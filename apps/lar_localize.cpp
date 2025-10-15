#include <stdint.h>

#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include <unordered_set>
#include <cmath>
#include <algorithm>
#include <thread>
#include <mutex>
#include <chrono>
#include <atomic>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include "lar/core/utils/json.h"
#include "lar/tracking/tracker.h"
#include "lar/mapping/frame.h"

using namespace std;

std::string getPathPrefix(std::string directory, int id) {
  std::string id_string = std::to_string(id);
  int zero_count = 8 - static_cast<int>(id_string.length());
  std::string prefix = std::string(zero_count, '0') + id_string + '_';
  return directory + prefix;
};

struct FrameData {
  lar::Frame frame;
  cv::Mat image;
  bool localized = false;
  Eigen::Matrix4d result_transform;
};

int main(int argc, const char* argv[]){
  // Parse command line arguments
  int num_threads = std::thread::hardware_concurrency();
  if (argc > 1) {
    num_threads = std::stoi(argv[1]);
  }

  string localize = "./input/aizu-park-4-ext/";
  // string localize = "./input/aizu-park-sunny/";

  std::cout << "=== Multithreaded Localization Test ===" << std::endl;
  std::cout << "Using " << num_threads << " threads" << std::endl;
  std::cout << "Hardware concurrency: " << std::thread::hardware_concurrency() << std::endl;
  std::cout << std::endl;

  // Load map
  std::cout << "Loading map..." << std::endl;
  auto map_load_start = std::chrono::high_resolution_clock::now();
  std::ifstream map_data_ifs("./output/aizu-park-map/map.json");
  nlohmann::json map_data = nlohmann::json::parse(map_data_ifs);
  lar::Map map = map_data;
  auto map_load_end = std::chrono::high_resolution_clock::now();
  auto map_load_duration = std::chrono::duration_cast<std::chrono::milliseconds>(map_load_end - map_load_start);
  std::cout << "Map loaded in " << map_load_duration.count() << " ms" << std::endl;
  std::cout << std::endl;

  std::cout << "Pre-loading all images into memory..." << std::endl;
  auto load_start = std::chrono::high_resolution_clock::now();

  std::vector<lar::Frame> frames = nlohmann::json::parse(std::ifstream(localize+"frames.json"));
  size_t frame_count = std::min(400, (int)frames.size());
  std::vector<FrameData> frame_data(frame_count);


  for (size_t i = 0; i < frame_count; i++) {
    frame_data[i].frame = frames[i];
    std::string image_path = getPathPrefix(localize, frames[i].id) + "image.jpeg";
    frame_data[i].image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);

    if (frame_data[i].image.empty()) {
      std::cerr << "Warning: Failed to load image for frame " << frames[i].id << std::endl;
    }
  }

  auto load_end = std::chrono::high_resolution_clock::now();
  auto load_duration = std::chrono::duration_cast<std::chrono::milliseconds>(load_end - load_start);
  std::cout << "Loaded " << frame_data.size() << " images in " << load_duration.count() << " ms" << std::endl;
  std::cout << std::endl;

  std::cout << "FLANN matching enabled" << std::endl;
  std::cout << std::endl;

  // Thread synchronization
  std::atomic<int> next_frame_index(0);
  std::atomic<int> successful(0);
  std::atomic<int> processed(0);
  std::mutex cout_mutex;

  auto worker = [&]() {
    cv::Size imageSize(1920, 1440); // Default ARKit size

    // Create thread-local tracker 
    lar::Tracker tracker(map, imageSize);

    while (true) {
      int index = next_frame_index.fetch_add(1);
      if (index >= static_cast<int>(frame_data.size())) {
        break;
      }

      auto& data = frame_data[index];
      auto& frame = data.frame;
      auto& image = data.image;

      if (image.empty()) {
        processed.fetch_add(1);
        continue;
      }

      // Use frame position for spatial query
      double query_x = frame.extrinsics(0, 3);
      double query_z = frame.extrinsics(2, 3);
      double query_diameter = 20.0; // 20 meter search radius

      Eigen::Matrix4d result_transform;
      bool success = tracker.localize(image, frame, query_x, query_z, query_diameter, result_transform);

      if (success) {
        data.localized = true;
        data.result_transform = result_transform;
        successful.fetch_add(1);
      }

      int current_processed = processed.fetch_add(1) + 1;

      {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cout << "Thread " << std::this_thread::get_id()
                  << " - Frame " << frame.id << " [" << current_processed << "/" << frame_data.size() << "] "
                  << (success ? "✓ SUCCESS" : "✗ FAILED") << std::endl;
      }
    }

  };

  std::cout << "Starting multithreaded localization..." << std::endl;
  auto start_time = std::chrono::high_resolution_clock::now();

  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; i++) {
    threads.emplace_back(worker);
  }

  for (auto& thread : threads) {
    thread.join();
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

  // Calculate statistics
  double seconds = duration.count() / 1000.0;
  double fps = frame_data.size() / seconds;
  double avg_ms_per_frame = duration.count() / static_cast<double>(frame_data.size());

  std::cout << std::endl;
  std::cout << "=== Results ===" << std::endl;
  std::cout << "Successfully localized: " << successful.load() << "/" << frame_data.size() << " frames" << std::endl;
  std::cout << "Success rate: " << (100.0 * successful.load() / frame_data.size()) << "%" << std::endl;
  std::cout << "Total processing time: " << duration.count() << " ms (" << seconds << " s)" << std::endl;
  std::cout << "Average time per frame: " << avg_ms_per_frame << " ms" << std::endl;
  std::cout << "Throughput: " << fps << " FPS" << std::endl;
  std::cout << std::endl;
  std::cout << "Configuration: " << num_threads << " threads" << std::endl;

  return 0;
}
