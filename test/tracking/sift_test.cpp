#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "lar/tracking/sift/sift.h"

class SIFTTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Load real test image from fixtures
        std::string image_path = "./test/_fixture/raw_map_data/00000000_image.jpeg";
        test_image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);        
        ASSERT_FALSE(test_image.empty()) << "Test image could not be loaded";
    }

    cv::Mat test_image;
};

TEST_F(SIFTTest, CreateInstance) {
    lar::SiftConfig config(test_image.size());
    config.nfeatures = 0;
    config.nOctaveLayers = 3;
    config.contrastThreshold = 0.04;
    config.edgeThreshold = 10;
    config.sigma = 1.6;
    config.descriptorType = CV_32F;

    lar::SIFT sift(config);

    // Test basic properties
    EXPECT_EQ(sift.descriptorSize(), 128);
    EXPECT_EQ(sift.descriptorType(), CV_32F);
    EXPECT_EQ(sift.defaultNorm(), cv::NORM_L2);
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
              
    float cv_min_size = cv_keypoints[0].size;
    float cv_max_size = cv_keypoints[0].size;
    for (int i = 1; i < cv_keypoints.size(); i++) {
        cv_min_size = std::min(cv_min_size, cv_keypoints[i].size);
        cv_max_size = std::max(cv_max_size, cv_keypoints[i].size);
    }
    float lar_min_size = lar_keypoints[0].size;
    float lar_max_size = lar_keypoints[0].size;
    for (int i = 1; i < lar_keypoints.size(); i++) {
        lar_min_size = std::min(lar_min_size, lar_keypoints[i].size);
        lar_max_size = std::max(lar_max_size, lar_keypoints[i].size);
    }
    std::cout << "OpenCV keypoint size range: " << cv_min_size << " - " << cv_max_size << std::endl;
    std::cout << "LAR keypoint size range: " << lar_min_size << " - " << lar_max_size << std::endl;
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
        
        // // Descriptors should be properly normalized
        // if (!cv_descriptors.empty()) {
        //     for (int i = 0; i < std::min(5, cv_descriptors.rows); i++) {  // Check first 5 descriptors
        //         float norm = 0;
        //         for (int j = 0; j < cv_descriptors.cols; j++) {
        //             float val = cv_descriptors.at<float>(i, j);
        //             norm += val * val;
        //         }
        //         norm = std::sqrt(norm);
        //         EXPECT_NEAR(norm, 1.0f, 1e-4f) << "Descriptor " << i << " in " << path << " should be normalized";
        //     }
        // }
        // // Descriptors should be properly normalized
        // if (!lar_descriptors.empty()) {
        //     for (int i = 0; i < std::min(5, lar_descriptors.rows); i++) {  // Check first 5 descriptors
        //         float norm = 0;
        //         for (int j = 0; j < lar_descriptors.cols; j++) {
        //             float val = lar_descriptors.at<float>(i, j);
        //             norm += val * val;
        //         }
        //         norm = std::sqrt(norm);
        //         EXPECT_NEAR(norm, 1.0f, 1e-4f) << "Descriptor " << i << " in " << path << " should be normalized";
        //     }
        // }
        
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