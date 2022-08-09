#include <iostream>
#include "lar/core/spacial/point.h"

namespace lar {

  Point::Point(double x, double y) : x(x), y(y) {
  }

  Point::Point() : Point(0, 0) {
  }

}
