#ifndef LAR_TRACKING_SIFT_H
#define LAR_TRACKING_SIFT_H

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <vector>

namespace lar {

class SIFT {
public:
    SIFT(int nfeatures = 0, int nOctaveLayers = 3,
         double contrastThreshold = 0.04, double edgeThreshold = 10,
         double sigma = 1.6, int descriptorType = CV_32F);

    static cv::Ptr<SIFT> create(int nfeatures = 0, int nOctaveLayers = 3,
                                 double contrastThreshold = 0.04, double edgeThreshold = 10,
                                 double sigma = 1.6, int descriptorType = CV_32F);

    void detectAndCompute(cv::InputArray image, cv::InputArray mask,
                         std::vector<cv::KeyPoint>& keypoints,
                         cv::OutputArray descriptors,
                         bool useProvidedKeypoints = false);

    int descriptorSize() const;
    int descriptorType() const;
    int defaultNorm() const;

private:
    void buildGaussianPyramid(const cv::Mat& base, std::vector<cv::Mat>& pyr, int nOctaves) const;
    void buildDoGPyramid(const std::vector<cv::Mat>& pyr, std::vector<cv::Mat>& dogpyr) const;
    void findScaleSpaceExtrema(const std::vector<cv::Mat>& gauss_pyr, 
                               const std::vector<cv::Mat>& dog_pyr,
                               std::vector<cv::KeyPoint>& keypoints) const;

    int nfeatures_;
    int nOctaveLayers_;
    double contrastThreshold_;
    double edgeThreshold_;
    double sigma_;
    int descriptorType_;
};

// Helper functions for SIFT keypoint refinement and orientation
// These are exposed for use by Metal-accelerated implementation
float calcOrientationHist(const cv::Mat& img, cv::Point pt, int radius,
                          float sigma, float* hist, int n);

bool adjustLocalExtrema(const std::vector<cv::Mat>& dog_pyr, cv::KeyPoint& kpt, int octv,
                        int& layer, int& r, int& c, int nOctaveLayers,
                        float contrastThreshold, float edgeThreshold, float sigma);

} // namespace lar

#endif // LAR_TRACKING_SIFT_H