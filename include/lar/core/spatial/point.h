#ifndef LAR_CORE_SPATIAL_POINT_H
#define LAR_CORE_SPATIAL_POINT_H

#include <nlohmann/json.hpp>
#include <iostream>

namespace lar {

  struct Point {
    double x, y;

    Point(double x, double y);
    Point();

    double dist2(Point &other) const;
    double l1() const;
  };

  NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Point, x, y)

}

#endif /* LAR_CORE_SPATIAL_POINT_H */