#ifndef LAR_PROCESSING_GLOBAL_ALIGNMENT_H
#define LAR_PROCESSING_GLOBAL_ALIGNMENT_H

#include "lar/mapping/mapper.h"

namespace lar {

  class GlobalAlignment {
    public:
      Mapper::Data& data;

      GlobalAlignment(Mapper::Data &data);
      void updateAlignment();

      Eigen::Matrix3d crossCovariance(const Eigen::Vector3d rc, const Eigen::Vector3d gc, const Eigen::DiagonalMatrix<double,3> D);
      void centroids(Eigen::Vector3d& rc, Eigen::Vector3d& gc);
      static inline double weight(Eigen::Vector3d accuracy);
  };

}

#endif /* LAR_PROCESSING_GLOBAL_ALIGNMENT_H */