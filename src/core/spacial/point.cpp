#include <iostream>
#include "lar/core/spacial/point.h"

namespace lar {

  Point::Point(double x, double y) : x(x), y(y) {
  }

  Point::Point() : Point(0, 0) {
  }

  double Point::dist2(Point &other) const {
    return (x - other.x) * (x - other.x) + (y - other.y) * (y - other.y);
  }

  double Point::l1() const {
    return x + y;
  }
}
