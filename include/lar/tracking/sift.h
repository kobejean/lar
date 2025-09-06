#ifndef LAR_TRACKING_SIFT_H
#define LAR_TRACKING_SIFT_H

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <memory>

namespace lar {

/**
 * LAR's custom SIFT (Scale-Invariant Feature Transform) implementation
 * Drop-in replacement for cv::SIFT with identical interface
 */
class SIFT : public cv::Feature2D {
public:
    /**
     * Create SIFT detector/descriptor extractor
     * @param nfeatures Maximum number of features to retain (0 = no limit)
     * @param nOctaveLayers Number of layers in each octave (3 in Lowe's paper)
     * @param contrastThreshold Threshold for filtering weak features in low contrast regions
     * @param edgeThreshold Threshold for filtering edge-like features
     * @param sigma Sigma of the Gaussian applied to the input image at octave 0
     * @param descriptorType CV_32F (float) or CV_8U (unsigned char)
     */
    static cv::Ptr<SIFT> create(
        int nfeatures = 0,
        int nOctaveLayers = 3,
        double contrastThreshold = 0.04,
        double edgeThreshold = 10,
        double sigma = 1.6,
        int descriptorType = CV_32F
    );

    // Override cv::Feature2D interface
    virtual void detectAndCompute(
        cv::InputArray image,
        cv::InputArray mask,
        std::vector<cv::KeyPoint>& keypoints,
        cv::OutputArray descriptors,
        bool useProvidedKeypoints = false
    ) override;

    virtual void detect(
        cv::InputArray image,
        std::vector<cv::KeyPoint>& keypoints,
        cv::InputArray mask = cv::noArray()
    ) override;

    virtual void compute(
        cv::InputArray image,
        std::vector<cv::KeyPoint>& keypoints,
        cv::OutputArray descriptors
    ) override;

    virtual int descriptorSize() const override { return 128; }
    virtual int descriptorType() const override { return descriptorType_; }
    virtual int defaultNorm() const override { return cv::NORM_L2; }

    SIFT(int nfeatures, int nOctaveLayers, double contrastThreshold,
            double edgeThreshold, double sigma, int descriptorType);

private:
    // Parameters
    int nfeatures_;
    int nOctaveLayers_;
    double contrastThreshold_;
    double edgeThreshold_;
    double sigma_;
    int descriptorType_;

    // Internal structures
    struct ScaleSpacePyramid {
        std::vector<std::vector<cv::Mat>> octaves;  // octaves[octave][layer]
        std::vector<std::vector<cv::Mat>> DoG;      // Difference of Gaussians
        int nOctaves;
        
        void build(const cv::Mat& image, int nOctaveLayers, double sigma);
    };

    struct Keypoint {
        int octave;
        int layer;
        float x, y;
        float scale;
        float response;
        float angle;
        
        cv::KeyPoint toOpenCV() const;
    };

    // Core SIFT algorithms
    void buildScaleSpace(const cv::Mat& image, ScaleSpacePyramid& pyramid);
    void findScaleSpaceExtrema(const ScaleSpacePyramid& pyramid, std::vector<Keypoint>& keypoints);
    bool localizeKeypoint(const ScaleSpacePyramid& pyramid, Keypoint& kpt);
    void eliminateEdgeResponse(const ScaleSpacePyramid& pyramid, std::vector<Keypoint>& keypoints);
    void assignOrientations(const ScaleSpacePyramid& pyramid, std::vector<Keypoint>& keypoints);
    void computeDescriptors(const ScaleSpacePyramid& pyramid, const std::vector<Keypoint>& keypoints, cv::Mat& descriptors);
    
    // Helper functions
    cv::Mat createGaussianKernel(double sigma);
    void gaussianBlur(const cv::Mat& src, cv::Mat& dst, double sigma);
    double getPixelValue(const cv::Mat& img, float x, float y);
    void computeGradient(const cv::Mat& img, float x, float y, float& dx, float& dy);
    
    // Feature selection
    void retainBestFeatures(std::vector<Keypoint>& keypoints);
    
    // Constants (following Lowe's paper)
    static constexpr double SIFT_INIT_SIGMA = 0.5;        // Assumed blur in input image
    static constexpr bool SIFT_IMG_DBL = true;            // Double image size before processing
    static constexpr double SIFT_CONTR_THR = 0.04;        // Default contrast threshold
    static constexpr double SIFT_CURV_THR = 10.0;         // Default edge threshold
    static constexpr int SIFT_MAX_INTERP_STEPS = 5;       // Maximum interpolation steps
    static constexpr int SIFT_ORI_HIST_BINS = 36;         // Orientation histogram bins
    static constexpr double SIFT_ORI_SIG_FCTR = 1.5;      // Sigma factor for orientation
    static constexpr double SIFT_ORI_PEAK_RATIO = 0.8;    // Orientation peak ratio
    static constexpr int SIFT_DESCR_WIDTH = 4;            // Descriptor window width
    static constexpr int SIFT_DESCR_HIST_BINS = 8;        // Descriptor histogram bins
    static constexpr double SIFT_DESCR_SCL_FCTR = 3.0;    // Descriptor scale factor
    static constexpr double SIFT_DESCR_MAG_THR = 0.2;     // Descriptor magnitude threshold
    static constexpr double SIFT_INT_DESCR_FCTR = 512.0;  // Factor for CV_8U conversion
};

} // namespace lar

#endif // LAR_TRACKING_SIFT_H