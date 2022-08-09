#ifndef LAR_CORE_SPACIAL_RECTANGLE_H
#define LAR_CORE_SPACIAL_RECTANGLE_H

#include <iostream>
#include "lar/core/spacial/point.h"

namespace lar {

  struct Rectangle {
    Point lower, upper;

    Rectangle(Point lower, Point upper);
    Rectangle(double x1, double y1, double x2, double y2);
    Rectangle(Point center, double width, double height);
    Rectangle();

    void print(std::ostream &os) const;
    bool intersectsWith(const Rectangle &other) const;
    bool isInsideOf(const Rectangle &other) const;
    Rectangle min_bounding(const Rectangle &other) const;
  };

}

#endif /* LAR_CORE_SPACIAL_RECTANGLE_H */