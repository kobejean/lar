
#include <Eigen/Core>
#include <opencv2/features2d.hpp>

using namespace Eigen;
using namespace std;

namespace geoar {

  class Landmark {
    public: 

      cv::Mat desc;
      cv::KeyPoint kpt;

      Landmark(cv::KeyPoint kpt, cv::Mat desc);

      static void concatDescriptions(vector<Landmark> landmarks, cv::Mat &desc);
  };
}