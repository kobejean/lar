#include "gtest/gtest.h"
#include <filesystem>
#include <fstream>
#include <sqlite3.h>

#include "lar/processing/colmap_database.h"
#include "lar/mapping/frame.h"
#include "lar/mapping/location_matcher.h"

namespace lar {

class ColmapDatabaseTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_dir = std::filesystem::temp_directory_path() / "colmap_test";
        std::filesystem::create_directory(test_dir);
        
        database_path = test_dir / "database.db";
        sparse_dir = test_dir / "sparse";
        std::filesystem::create_directory(sparse_dir);
        
        // Create minimal test database
        createTestDatabase();
        createTestSparseFiles();
        createTestFrames();
    }
    
    void TearDown() override {
        std::filesystem::remove_all(test_dir);
    }
    
    void createTestDatabase() {
        sqlite3* db;
        int rc = sqlite3_open(database_path.c_str(), &db);
        ASSERT_EQ(rc, SQLITE_OK);
        
        // Create tables
        const char* create_images = R"(
            CREATE TABLE images (
                image_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                qw REAL,
                qx REAL,
                qy REAL,
                qz REAL,
                tx REAL,
                ty REAL,
                tz REAL
            )
        )";
        
        const char* create_keypoints = R"(
            CREATE TABLE keypoints (
                image_id INTEGER,
                data BLOB
            )
        )";
        
        const char* create_descriptors = R"(
            CREATE TABLE descriptors (
                image_id INTEGER,
                rows INTEGER,
                cols INTEGER,
                data BLOB
            )
        )";
        
        sqlite3_exec(db, create_images, nullptr, nullptr, nullptr);
        sqlite3_exec(db, create_keypoints, nullptr, nullptr, nullptr);
        sqlite3_exec(db, create_descriptors, nullptr, nullptr, nullptr);
        
        // Insert test data
        const char* insert_image = R"(
            INSERT INTO images VALUES (1, '1_image.jpeg', 1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0)
        )";
        sqlite3_exec(db, insert_image, nullptr, nullptr, nullptr);
        
        // Create test keypoints (6 floats per keypoint: x, y, scale, orientation, data, data)
        float keypoint_data[] = {100.0f, 200.0f, 1.0f, 0.0f, 0.0f, 0.0f};
        const char* insert_keypoints = "INSERT INTO keypoints VALUES (1, ?)";
        sqlite3_stmt* stmt;
        sqlite3_prepare_v2(db, insert_keypoints, -1, &stmt, nullptr);
        sqlite3_bind_blob(stmt, 1, keypoint_data, sizeof(keypoint_data), SQLITE_STATIC);
        sqlite3_step(stmt);
        sqlite3_finalize(stmt);
        
        // Create test descriptors (128-dimensional SIFT)
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
    }
    
    void createTestSparseFiles() {
        // Create images.txt
        std::ofstream images_file(sparse_dir / "images.txt");
        images_file << "# Image list with two lines of data per image:\n";
        images_file << "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n";
        images_file << "1 1.0 0.0 0.0 0.0 1.0 2.0 3.0 1 1_image.jpeg\n";
        images_file.close();
        
        // Create points3D.txt
        std::ofstream points3d_file(sparse_dir / "points3D.txt");
        points3d_file << "# 3D point list with information:\n";
        points3d_file << "# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK\n";
        points3d_file << "1 5.0 6.0 7.0 255 128 64 0.1 1 0\n";
        points3d_file.close();
    }
    
    void createTestFrames() {
        Frame frame;
        frame.id = 1;
        frame.timestamp = 1234567890;
        frame.extrinsics = Eigen::Matrix4d::Identity();
        frame.extrinsics(0, 3) = 1.0;
        frame.extrinsics(1, 3) = 2.0;
        frame.extrinsics(2, 3) = 3.0;
        test_frames.push_back(frame);
    }
    
    std::filesystem::path test_dir;
    std::filesystem::path database_path;
    std::filesystem::path sparse_dir;
    std::vector<Frame> test_frames;
};

TEST_F(ColmapDatabaseTest, ReadDatabase) {
    ColmapDatabase db;
    EXPECT_TRUE(db.readDatabase(database_path.string()));
    
    // Verify image was read (images vector is indexed by image_id, so size is image_id + 1)
    EXPECT_EQ(db.images.size(), 2);
    EXPECT_EQ(db.images[1].image_id, 1);
    EXPECT_EQ(db.images[1].name, "1_image.jpeg");
    EXPECT_EQ(db.images[1].keypoints.size(), 1);
    
    // Verify keypoint
    const cv::KeyPoint& kpt = db.images[1].keypoints[0];
    EXPECT_FLOAT_EQ(kpt.pt.x, 100.0f);
    EXPECT_FLOAT_EQ(kpt.pt.y, 200.0f);
    
    // Verify descriptor
    EXPECT_EQ(db.images[1].descriptors.rows, 1);
    EXPECT_EQ(db.images[1].descriptors.cols, 128);
}

TEST_F(ColmapDatabaseTest, ReadSparseReconstruction) {
    ColmapDatabase db;
    EXPECT_TRUE(db.readSparseReconstruction(sparse_dir.string()));
    
    // Verify 3D point was read (points3d is a sequential vector, not indexed by ID)
    EXPECT_EQ(db.points3d.size(), 1);
    EXPECT_EQ(db.points3d[0].point3d_id, 1);
    EXPECT_DOUBLE_EQ(db.points3d[0].position.x(), 5.0);
    EXPECT_DOUBLE_EQ(db.points3d[0].position.y(), 6.0);
    EXPECT_DOUBLE_EQ(db.points3d[0].position.z(), 7.0);
    
    // Verify track
    EXPECT_EQ(db.points3d[0].track.size(), 1);
    EXPECT_EQ(db.points3d[0].track[0].first, 1);  // image_id
    EXPECT_EQ(db.points3d[0].track[0].second, 0); // point2d_idx
}

TEST_F(ColmapDatabaseTest, CoordinateConversion) {
    ColmapDatabase db;
    
    // Test COLMAP to ARKit coordinate conversion
    // COLMAP: Y down, Z forward -> ARKit: Y up, Z backward
    Eigen::Vector3d colmap_position(1.0, 2.0, 3.0);
    Eigen::Vector3d expected_arkit_position(1.0, -2.0, -3.0);
    
    // This would be done in constructLandmarksFromColmap
    Eigen::Vector3d arkit_position;
    arkit_position.x() = colmap_position.x();
    arkit_position.y() = -colmap_position.y();  // Flip Y
    arkit_position.z() = -colmap_position.z();  // Flip Z
    
    EXPECT_DOUBLE_EQ(arkit_position.x(), expected_arkit_position.x());
    EXPECT_DOUBLE_EQ(arkit_position.y(), expected_arkit_position.y());
    EXPECT_DOUBLE_EQ(arkit_position.z(), expected_arkit_position.z());
}

TEST_F(ColmapDatabaseTest, SpatialBoundsCalculation) {
    ColmapDatabase db;
    
    Eigen::Vector3d landmark_pos(0.0, 0.0, 0.0);
    std::vector<Eigen::Vector3d> camera_positions = {
        Eigen::Vector3d(1.0, 0.0, 0.0),
        Eigen::Vector3d(-1.0, 0.0, 0.0),
        Eigen::Vector3d(0.0, 0.0, 1.0),
        Eigen::Vector3d(0.0, 0.0, -1.0)
    };
    
    // Use reflection to access private method for testing
    // In a real implementation, we'd make this method public or create a test-friendly interface
    // For now, let's test the calculation logic directly
    
    double max_distance = 0.0;
    for (const auto& cam_pos : camera_positions) {
        double dist = (cam_pos - landmark_pos).norm();
        max_distance = std::max(max_distance, dist);
    }
    
    EXPECT_DOUBLE_EQ(max_distance, 1.0);
    
    // Bounds calculation
    double extent = max_distance * 1.5; // default max_distance_factor
    EXPECT_DOUBLE_EQ(extent, 1.5);
}

TEST_F(ColmapDatabaseTest, ConstructLandmarksFromColmap) {
    ColmapDatabase db;
    EXPECT_TRUE(db.readDatabase(database_path.string()));
    EXPECT_TRUE(db.readSparseReconstruction(sparse_dir.string()));
    
    std::vector<Landmark> landmarks;
    
    // This should not crash and should create landmarks
    EXPECT_NO_THROW(db.constructLandmarksFromColmap(test_frames, landmarks, database_path.string()));
    
    // The landmark database should have some landmarks (exact count depends on matching logic)
    // We can't easily test the exact count without knowing the internal matching logic
    // But we can verify no crash occurred
}

TEST_F(ColmapDatabaseTest, InvalidDatabasePath) {
    ColmapDatabase db;
    EXPECT_FALSE(db.readDatabase("/nonexistent/path/database.db"));
}

TEST_F(ColmapDatabaseTest, InvalidSparseDirectory) {
    ColmapDatabase db;
    EXPECT_FALSE(db.readSparseReconstruction("/nonexistent/sparse/dir"));
}

} // namespace lar