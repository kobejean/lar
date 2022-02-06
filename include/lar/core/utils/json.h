#ifndef LAR_CORE_UTILS_JSON_H
#define LAR_CORE_UTILS_JSON_H

#include <Eigen/Core>
#include <nlohmann/json.hpp>

namespace Eigen {

  template< typename Scalar_, int Rows_, int Cols_ >
  static void to_json( nlohmann::json& j, const Matrix< Scalar_, Rows_, Cols_ >& mat )
  {
    j = std::vector<Scalar_>(mat.data(), mat.data() + mat.size());
  }

  template< typename Scalar_, int Rows_, int Cols_ >
  static void from_json( const nlohmann::json& j, Matrix< Scalar_, Rows_, Cols_ >& mat )
  {
    mat = Eigen::Matrix<Scalar_, Rows_, Cols_>(j.get<std::vector<Scalar_>>().data());
  }

  template< typename Scalar_, int _Dim, int _Mode >
  static void to_json( nlohmann::json& j, const Eigen::Transform< Scalar_, _Dim, _Mode >& trans )
  {
    j = trans.matrix();
  }

  template< typename Scalar_, int _Dim, int _Mode >
  static void from_json( const nlohmann::json& j, Eigen::Transform< Scalar_, _Dim, _Mode >& trans )
  {
    constexpr int D = _Dim+1;
    Eigen::Matrix<Scalar_, D, D> mat = j;
    trans = mat;
  }
  
}

#endif /* LAR_CORE_UTILS_JSON_H */