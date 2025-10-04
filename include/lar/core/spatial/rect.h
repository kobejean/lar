#ifndef LAR_CORE_SPATIAL_RECT_H
#define LAR_CORE_SPATIAL_RECT_H

#include <nlohmann/json.hpp>
#include <iostream>
#include "lar/core/spatial/point.h"

namespace lar {

  struct Rect {
    Point lower, upper;

    Rect(Point lower, Point upper);
    Rect(double x1, double y1, double x2, double y2);
    Rect(Point center, double width, double height);
    Rect();

    void print(std::ostream &os) const;
    bool intersectsWith(const Rect &other) const;
    bool isInsideOf(const Rect &other) const;
    Rect minBoundingBox(const Rect &other) const;
    double area() const;
    Rect intersection(const Rect &other) const;
    double overlap(const Rect &other) const;
  };

  NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Rect, lower, upper)

}

#endif /* LAR_CORE_SPATIAL_RECT_H */