#ifndef LAR_CORE_SPACIAL_POINT_H
#define LAR_CORE_SPACIAL_POINT_H

#include <iostream>

namespace lar {

  struct Point {
    double x, y;

    Point(double x, double y);
    Point();

    double dist2(Point &other) const;
    double l1() const;
  };

}

#endif /* LAR_CORE_SPACIAL_POINT_H */