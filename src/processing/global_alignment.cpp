#include <Eigen/Core>
#include <iostream>
#include "lar/core/utils/wgs84.h"
#include "lar/processing/global_alignment.h"

namespace lar {

  GlobalAlignment::GlobalAlignment(Mapper::Data &data) : data(data) {
  }

  void GlobalAlignment::updateAlignment() {
    if (data.gps_obs.size() < 2) return;
    Eigen::Vector3d rc, gc;
    centroids(rc, gc);
    Eigen::DiagonalMatrix<double,3> D = wgs84::wgs84_scaling(gc);
    Eigen::Matrix3d CC = crossCovariance(rc, gc, D);
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(CC, Eigen::ComputeThinU | Eigen::ComputeThinV);
    
    Eigen::Matrix3d R = svd.matrixU() * svd.matrixV().transpose();
    R(0,1) = 0;
    R(1,1) = 0;
    R(2,0) = 0;
    R(2,2) = 0;
    Eigen::Matrix4d Tmat = Eigen::Matrix4d::Identity();
    Tmat.block<3,3>(0,0) = D.inverse() * R;
    Tmat.block<3,1>(0,3) = gc;
    Eigen::Transform<double, 3, Eigen::Affine> T = Eigen::Transform<double, 3, Eigen::Affine>(Tmat) * Eigen::Translation<double, 3>(-rc);
    data.map.origin = T;
  }

  Eigen::Matrix3d GlobalAlignment::crossCovariance(const Eigen::Vector3d rc, const Eigen::Vector3d gc, const Eigen::DiagonalMatrix<double,3> D) {
    assert(data.gps_obs.size() >= 2);
    Eigen::Matrix3d CC = Eigen::Matrix3d::Zero();
    
    for (size_t i=0; i < data.gps_obs.size(); i++) {
      GPSObservation& obs = data.gps_obs[i];
      double w = weight(obs.accuracy);
      auto dr = obs.relative - rc;
      auto dg = D * (obs.global - gc);
      CC += dg * w * dr.transpose();
    }

    CC(0,1) = 0;
    CC(1,1) = 0;
    CC(2,0) = 0;
    CC(2,2) = 0;
    return CC;
  }

  void GlobalAlignment::centroids(Eigen::Vector3d& rc, Eigen::Vector3d& gc) {
    assert(data.gps_obs.size() >= 2);
    rc = Eigen::Vector3d::Zero();
    gc = Eigen::Vector3d::Zero();
    if (data.gps_obs.size() == 0) return;
    double w_sum = 0;
    
    for (GPSObservation& obs : data.gps_obs) {
      double w = weight(obs.accuracy);
      rc += w * obs.relative;
      gc += w * obs.global;
      w_sum += w;
    }

    rc /= w_sum;
    gc /= w_sum;
  }

  inline double GlobalAlignment::weight(Eigen::Vector3d accuracy) {
    double ha = accuracy.x();
    if (ha <= 0) return 1e-4;
    // inverse variance weighting
    return 1 / (ha*ha);
  }
}