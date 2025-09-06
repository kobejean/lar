#include "gtest/gtest.h"
#include <filesystem>
#include <fstream>
#include <memory>

#include "lar/processing/colmap_refiner.h"
#include "lar/mapping/mapper.h"
#include "lar/mapping/location_matcher.h"

namespace lar {

class ColmapRefinerTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_dir = std::filesystem::temp_directory_path() / "colmap_refiner_test";
        std::filesystem::create_directory(test_dir);
        
        colmap_dir = test_dir / "colmap";
        std::filesystem::create_directory(colmap_dir);
        
        // Create test data
        createTestColmapData();
        createTestMapperData();
    }
    
    void TearDown() override {
        std::filesystem::remove_all(test_dir);
    }
    
    void createTestColmapData() {
        // Create database.db (simplified SQLite database)
        std::string database_path = (colmap_dir / "database.db").string();
        sqlite3* db;
        int rc = sqlite3_open(database_path.c_str(), &db);
        ASSERT_EQ(rc, SQLITE_OK);
        
        // Create minimal tables and data
        const char* sql = R"(
            CREATE TABLE images (
                image_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                qw REAL, qx REAL, qy REAL, qz REAL,
                tx REAL, ty REAL, tz REAL
            );
            CREATE TABLE keypoints (image_id INTEGER, data BLOB);
            CREATE TABLE descriptors (image_id INTEGER, rows INTEGER, cols INTEGER, data BLOB);
            
            INSERT INTO images VALUES (1, '1_image.jpeg', 1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0);
        )";
        
        char* err_msg = nullptr;
        rc = sqlite3_exec(db, sql, nullptr, nullptr, &err_msg);
        if (rc != SQLITE_OK) {
            sqlite3_free(err_msg);
        }
        
        // Add keypoints
        float keypoint_data[] = {100.0f, 200.0f, 1.0f, 0.0f, 0.0f, 0.0f};
        const char* insert_keypoints = "INSERT INTO keypoints VALUES (1, ?)";
        sqlite3_stmt* stmt;
        sqlite3_prepare_v2(db, insert_keypoints, -1, &stmt, nullptr);
        sqlite3_bind_blob(stmt, 1, keypoint_data, sizeof(keypoint_data), SQLITE_STATIC);
        sqlite3_step(stmt);
        sqlite3_finalize(stmt);
        
        // Add descriptors
        uint8_t descriptor_data[128];
        for (int i = 0; i < 128; i++) {
            descriptor_data[i] = i % 256;
        }
        const char* insert_descriptors = "INSERT INTO descriptors VALUES (1, 1, 128, ?)";
        sqlite3_prepare_v2(db, insert_descriptors, -1, &stmt, nullptr);
        sqlite3_bind_blob(stmt, 1, descriptor_data, sizeof(descriptor_data), SQLITE_STATIC);
        sqlite3_step(stmt);
        sqlite3_finalize(stmt);
        
        sqlite3_close(db);
        
        // Create sparse reconstruction directory
        auto sparse_dir = colmap_dir / "aligned";
        std::filesystem::create_directory(sparse_dir);
        
        // Create images.txt
        std::ofstream images_file(sparse_dir / "images.txt");
        images_file << "# Image list\n";
        images_file << "1 1.0 0.0 0.0 0.0 1.0 2.0 3.0 1 1_image.jpeg\n";
        images_file.close();
        
        // Create points3D.txt
        std::ofstream points3d_file(sparse_dir / "points3D.txt");
        points3d_file << "# 3D point list\n";
        points3d_file << "1 5.0 6.0 7.0 255 128 64 0.1 1 0\n";
        points3d_file.close();
    }
    
    void createTestMapperData() {
        mapper_data = std::make_shared<Mapper::Data>();
        
        // Create test frame
        Frame frame;
        frame.id = 1;
        frame.timestamp = 1234567890;
        frame.extrinsics = Eigen::Matrix4d::Identity();
        frame.extrinsics(0, 3) = 1.0;
        frame.extrinsics(1, 3) = 2.0;
        frame.extrinsics(2, 3) = 3.0;
        
        mapper_data->frames.push_back(frame);
        
        // Initialize GPS data (required by GlobalAlignment)
        GPSObservation gps_obs;
        gps_obs.timestamp = frame.timestamp;
        gps_obs.global = Eigen::Vector3d(37.7749, -122.4194, 10.0);
        gps_obs.relative = Eigen::Vector3d(0.0, 0.0, 0.0);
        gps_obs.accuracy = Eigen::Vector3d(1.0, 1.0, 1.0);
        mapper_data->gps_obs.push_back(gps_obs);
        
        // Set up path prefix method
        mapper_data->directory = test_dir;
    }
    
    std::filesystem::path test_dir;
    std::filesystem::path colmap_dir;
    std::shared_ptr<Mapper::Data> mapper_data;
};

TEST_F(ColmapRefinerTest, Construction) {
    EXPECT_NO_THROW({
        ColmapRefiner refiner(mapper_data);
    });
}

TEST_F(ColmapRefinerTest, DISABLED_ProcessWithColmapData) {  // Disabled: needs proper COLMAP sparse reconstruction
    ColmapRefiner refiner(mapper_data);
    
    // This should not crash
    EXPECT_NO_THROW({
        refiner.processWithColmapData(colmap_dir.string());
    });
    
    // Verify localizations were computed
    EXPECT_EQ(refiner.localizations.size(), mapper_data->frames.size());
    
    // The localization should either be from COLMAP or fallback to ARKit
    EXPECT_FALSE(refiner.localizations.empty());
}

TEST_F(ColmapRefinerTest, ProcessWithColmapDataInvalidPath) {
    ColmapRefiner refiner(mapper_data);
    
    // Should handle invalid path gracefully
    EXPECT_NO_THROW({
        refiner.processWithColmapData("/nonexistent/colmap/dir");
    });
    
    // Should fallback gracefully
    EXPECT_EQ(refiner.localizations.size(), 0); // Failed to process, so no localizations
}

TEST_F(ColmapRefinerTest, OptimizeAfterProcessing) {
    ColmapRefiner refiner(mapper_data);
    
    // Process first
    refiner.processWithColmapData(colmap_dir.string());
    
    // Then optimize - should not crash
    EXPECT_NO_THROW({
        refiner.optimize();
    });
}

TEST_F(ColmapRefinerTest, DISABLED_SaveMapAfterOptimization) {  // Disabled: depends on ProcessWithColmapData
    ColmapRefiner refiner(mapper_data);
    
    // Process and optimize
    refiner.processWithColmapData(colmap_dir.string());
    refiner.optimize();
    
    // Save map
    auto output_dir = test_dir / "output";
    EXPECT_NO_THROW({
        refiner.saveMap(output_dir.string());
    });
    
    // Verify output files were created
    EXPECT_TRUE(std::filesystem::exists(output_dir / "map.g2o"));
    EXPECT_TRUE(std::filesystem::exists(output_dir / "map_refined.json"));
}

TEST_F(ColmapRefinerTest, DISABLED_CompareProcessMethods) {  // Disabled: needs SIFT features and proper image data
    ColmapRefiner refiner1(mapper_data);
    ColmapRefiner refiner2(mapper_data);
    
    // Process with different methods
    refiner1.process(); // Original tracker-based method
    refiner2.processWithColmapData(colmap_dir.string()); // COLMAP-based method
    
    // Both should produce localizations
    EXPECT_EQ(refiner1.localizations.size(), mapper_data->frames.size());
    EXPECT_EQ(refiner2.localizations.size(), mapper_data->frames.size());
    
    // The localizations might be different (COLMAP vs tracker)
    // but both should be valid 4x4 matrices
    for (const auto& loc : refiner1.localizations) {
        EXPECT_EQ(loc.rows(), 4);
        EXPECT_EQ(loc.cols(), 4);
    }
    
    for (const auto& loc : refiner2.localizations) {
        EXPECT_EQ(loc.rows(), 4);
        EXPECT_EQ(loc.cols(), 4);
    }
}

TEST_F(ColmapRefinerTest, DISABLED_MultipleSparseDirectories) {  // Disabled: needs proper COLMAP sparse reconstruction
    ColmapRefiner refiner(mapper_data);
    
    // Create additional sparse directories that the refiner might check
    auto sparse_dir_alt = colmap_dir / "sparse" / "0";
    std::filesystem::create_directories(sparse_dir_alt);
    
    // Create images.txt in alternative location
    std::ofstream images_file(sparse_dir_alt / "images.txt");
    images_file << "# Image list\n";
    images_file << "1 0.5 0.5 0.5 0.5 10.0 20.0 30.0 1 1_image.jpeg\n";
    images_file.close();
    
    // Create points3D.txt in alternative location
    std::ofstream points3d_file(sparse_dir_alt / "points3D.txt");
    points3d_file << "# 3D point list\n";
    points3d_file << "1 15.0 16.0 17.0 255 128 64 0.1 1 0\n";
    points3d_file.close();
    
    // Remove the primary aligned directory to force using alternative
    std::filesystem::remove_all(colmap_dir / "aligned");
    
    // Should still work with alternative sparse directory
    EXPECT_NO_THROW({
        refiner.processWithColmapData(colmap_dir.string());
    });
    
    EXPECT_EQ(refiner.localizations.size(), mapper_data->frames.size());
}

TEST_F(ColmapRefinerTest, EmptyFramesList) {
    // Test with empty frames
    auto empty_data = std::make_shared<Mapper::Data>();
    ColmapRefiner refiner(empty_data);
    
    EXPECT_NO_THROW({
        refiner.processWithColmapData(colmap_dir.string());
    });
    
    EXPECT_TRUE(refiner.localizations.empty());
}

} // namespace lar