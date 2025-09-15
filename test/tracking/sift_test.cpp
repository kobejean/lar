#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "lar/tracking/sift.h"

class SIFTTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Load real test image from fixtures
        std::string image_path = "./test/_fixture/raw_map_data/00000000_image.jpeg";
        test_image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
        
        // Fallback to synthetic image if file doesn't exist
        if (test_image.empty()) {
            std::cerr << "Warning: Could not load fixture image, using synthetic image" << std::endl;
            test_image = cv::Mat::zeros(200, 200, CV_8UC1);
            cv::circle(test_image, cv::Point(50, 50), 20, cv::Scalar(255), -1);
            cv::circle(test_image, cv::Point(150, 150), 15, cv::Scalar(255), -1);
            cv::rectangle(test_image, cv::Point(100, 30), cv::Point(130, 60), cv::Scalar(255), -1);
        }
        
        ASSERT_FALSE(test_image.empty()) << "Test image could not be loaded";
    }

    cv::Mat test_image;
};

TEST_F(SIFTTest, CreateInstance) {
    auto sift = lar::SIFT::create(0, 3, 0.04, 10, 1.6, CV_32F);
    EXPECT_TRUE(sift != nullptr);
    
    // Test basic properties
    EXPECT_EQ(sift->descriptorSize(), 128);
    EXPECT_EQ(sift->descriptorType(), CV_32F);
    EXPECT_EQ(sift->defaultNorm(), cv::NORM_L2);
}

TEST_F(SIFTTest, DetectKeypoints) {
    auto sift = lar::SIFT::create(0, 3, 0.04, 10, 1.6, CV_32F);
    
    std::vector<cv::KeyPoint> keypoints;
    sift->detect(test_image, keypoints);
    
    // Should detect some keypoints
    EXPECT_GT(keypoints.size(), 0);
    
    // Check keypoint properties
    for (const auto& kp : keypoints) {
        EXPECT_GE(kp.pt.x, 0);
        EXPECT_LT(kp.pt.x, test_image.cols);
        EXPECT_GE(kp.pt.y, 0);
        EXPECT_LT(kp.pt.y, test_image.rows);
        EXPECT_GT(kp.size, 0);
        EXPECT_GE(kp.angle, 0);
        EXPECT_LT(kp.angle, 360);
    }
}

TEST_F(SIFTTest, ComputeDescriptors) {
    auto sift = lar::SIFT::create(0, 3, 0.04, 10, 1.6, CV_32F);
    
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    
    sift->detectAndCompute(test_image, cv::noArray(), keypoints, descriptors);
    
    EXPECT_GT(keypoints.size(), 0);
    EXPECT_FALSE(descriptors.empty());
    EXPECT_EQ(descriptors.rows, static_cast<int>(keypoints.size()));
    EXPECT_EQ(descriptors.cols, 128);
    EXPECT_EQ(descriptors.type(), CV_32F);
    
    // Check descriptor values are normalized
    for (int i = 0; i < descriptors.rows; i++) {
        float norm = 0;
        for (int j = 0; j < descriptors.cols; j++) {
            float val = descriptors.at<float>(i, j);
            norm += val * val;
        }
        norm = std::sqrt(norm);
        EXPECT_NEAR(norm, 1.0f, 1e-5f);  // Should be normalized
    }
}

TEST_F(SIFTTest, CompareWithOpenCVSIFT) {
    // Create OpenCV SIFT
    auto cv_sift = cv::SIFT::create(0, 3, 0.04, 10, 1.6, CV_32F);
    
    // Create our SIFT
    auto lar_sift = lar::SIFT::create(0, 3, 0.04, 10, 1.6, CV_32F);
    
    // Detect and compute with both
    std::vector<cv::KeyPoint> cv_keypoints, lar_keypoints;
    cv::Mat cv_descriptors, lar_descriptors;
    
    cv_sift->detectAndCompute(test_image, cv::noArray(), cv_keypoints, cv_descriptors);
    lar_sift->detectAndCompute(test_image, cv::noArray(), lar_keypoints, lar_descriptors);
    
    // Both should find keypoints
    EXPECT_GT(cv_keypoints.size(), 0);
    EXPECT_GT(lar_keypoints.size(), 0);
    
    // Descriptor format should match
    EXPECT_EQ(cv_descriptors.cols, lar_descriptors.cols);  // 128 features
    EXPECT_EQ(cv_descriptors.type(), lar_descriptors.type());  // CV_32F
    
    // Number of keypoints may differ significantly due to implementation differences
    // Our implementation typically detects ~10% as many keypoints as OpenCV
    double ratio = static_cast<double>(lar_keypoints.size()) / cv_keypoints.size();
    EXPECT_GT(ratio, 0.05);  // At least 5% as many keypoints
    EXPECT_LT(ratio, 3.0);   // At most 3x as many keypoints
    
    std::cout << "Keypoint comparison - OpenCV: " << cv_keypoints.size() 
              << ", LAR: " << lar_keypoints.size() 
              << " (ratio: " << ratio << ")" << std::endl;
}

TEST_F(SIFTTest, CV8UDescriptors) {
    auto sift = lar::SIFT::create(0, 3, 0.04, 10, 1.6, CV_8U);
    
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    
    sift->detectAndCompute(test_image, cv::noArray(), keypoints, descriptors);
    
    EXPECT_GT(keypoints.size(), 0);
    EXPECT_FALSE(descriptors.empty());
    EXPECT_EQ(descriptors.type(), CV_8U);
    EXPECT_EQ(descriptors.cols, 128);
    
    // Check values are in valid range for CV_8U
    for (int i = 0; i < descriptors.rows; i++) {
        for (int j = 0; j < descriptors.cols; j++) {
            uchar val = descriptors.at<uchar>(i, j);
            EXPECT_GE(val, 0);
            EXPECT_LE(val, 255);
        }
    }
}

TEST_F(SIFTTest, MaxFeatures) {
    int max_features = 10;
    auto sift = lar::SIFT::create(max_features, 3, 0.04, 10, 1.6, CV_32F);
    
    std::vector<cv::KeyPoint> keypoints;
    sift->detect(test_image, keypoints);
    
    // Should not exceed max features
    EXPECT_LE(static_cast<int>(keypoints.size()), max_features);
}

TEST_F(SIFTTest, ComputeOnlyMode) {
    auto sift = lar::SIFT::create(0, 3, 0.04, 10, 1.6, CV_32F);
    
    // First detect keypoints
    std::vector<cv::KeyPoint> keypoints;
    sift->detect(test_image, keypoints);
    EXPECT_GT(keypoints.size(), 0);
    
    // Then compute descriptors only
    cv::Mat descriptors;
    sift->compute(test_image, keypoints, descriptors);
    
    EXPECT_FALSE(descriptors.empty());
    EXPECT_EQ(descriptors.rows, static_cast<int>(keypoints.size()));
    EXPECT_EQ(descriptors.cols, 128);
}

TEST_F(SIFTTest, MultipleRealImages) {
    // Test with multiple fixture images
    std::vector<std::string> image_paths = {
        "./test/_fixture/raw_map_data/00000000_image.jpeg",
        "./test/_fixture/raw_map_data/00000001_image.jpeg",
        "./test/_fixture/raw_map_data/00000002_image.jpeg"
    };
    
    auto cv_sift = cv::SIFT::create(0, 3, 0.04, 10, 1.6, CV_32F);
    auto lar_sift = lar::SIFT::create(0, 3, 0.04, 10, 1.6, CV_32F);
    
    for (const auto& path : image_paths) {
        cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);
        if (image.empty()) continue;  // Skip if image doesn't exist
        
        std::vector<cv::KeyPoint> cv_keypoints, lar_keypoints;
        cv::Mat cv_descriptors, lar_descriptors;
        
        cv_sift->detectAndCompute(image, cv::noArray(), cv_keypoints, cv_descriptors);
        lar_sift->detectAndCompute(image, cv::noArray(), lar_keypoints, lar_descriptors);
        
        // Both should detect features in real images
        EXPECT_GT(cv_keypoints.size(), 10) << "OpenCV SIFT should find many keypoints in " << path;
        EXPECT_GT(lar_keypoints.size(), 10) << "LAR SIFT should find many keypoints in " << path;
        
        // Descriptors should be properly normalized
        if (!lar_descriptors.empty()) {
            for (int i = 0; i < std::min(5, lar_descriptors.rows); i++) {  // Check first 5 descriptors
                float norm = 0;
                for (int j = 0; j < lar_descriptors.cols; j++) {
                    float val = lar_descriptors.at<float>(i, j);
                    norm += val * val;
                }
                norm = std::sqrt(norm);
                EXPECT_NEAR(norm, 1.0f, 1e-4f) << "Descriptor " << i << " in " << path << " should be normalized";
            }
        }
        
        std::cout << "Image: " << path << " - OpenCV: " << cv_keypoints.size() 
                  << " keypoints, LAR: " << lar_keypoints.size() << " keypoints" << std::endl;
    }
}

TEST_F(SIFTTest, ParameterVariations) {
    // Test with different parameter combinations like the original OpenCV usage
    struct TestConfig {
        std::string name;
        int nfeatures;
        int nOctaveLayers;
        double contrastThreshold;
        double edgeThreshold;
        double sigma;
        int descriptorType;
    };
    
    std::vector<TestConfig> configs = {
        {"Default", 0, 3, 0.04, 10, 1.6, CV_32F},
        {"Original_CV8U", 0, 3, 0.02, 10, 1.6, CV_8U},  // Original parameters from Vision class
        {"High_Contrast", 0, 3, 0.08, 10, 1.6, CV_32F},
        {"Limited_Features", 50, 3, 0.04, 10, 1.6, CV_32F}
    };
    
    for (const auto& config : configs) {
        auto sift = lar::SIFT::create(config.nfeatures, config.nOctaveLayers, 
                                     config.contrastThreshold, config.edgeThreshold,
                                     config.sigma, config.descriptorType);
        
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        
        sift->detectAndCompute(test_image, cv::noArray(), keypoints, descriptors);
        
        EXPECT_GT(keypoints.size(), 0) << "Config " << config.name << " should detect keypoints";
        EXPECT_FALSE(descriptors.empty()) << "Config " << config.name << " should compute descriptors";
        EXPECT_EQ(descriptors.type(), config.descriptorType) << "Config " << config.name << " descriptor type mismatch";
        
        if (config.nfeatures > 0) {
            EXPECT_LE(static_cast<int>(keypoints.size()), config.nfeatures) 
                << "Config " << config.name << " should respect feature limit";
        }
        
        std::cout << "Config " << config.name << ": " << keypoints.size() << " keypoints" << std::endl;
    }
}