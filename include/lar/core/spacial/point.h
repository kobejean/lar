#ifndef LAR_CORE_SPACIAL_POINT_H
#define LAR_CORE_SPACIAL_POINT_H

#include <iostream>

namespace lar {

  struct Point {
    double x, y;

    Point(double x, double y);
    Point();
  };

}

#endif /* LAR_CORE_SPACIAL_POINT_H */