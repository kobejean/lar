#include <iostream>
#include "lar/core/spacial/rectangle.h"

namespace lar {

  Rectangle::Rectangle(Point lower, Point upper) : lower(lower), upper(upper) {
  }

  Rectangle::Rectangle(double x1, double y1, double x2, double y2) : lower(x1, y1), upper(x2, y2) {
  }

  Rectangle::Rectangle(Point center, double width, double height) : lower(center.x - width / 2, center.y - height / 2), upper(center.x + width / 2, center.y + height / 2) {
  }

  Rectangle::Rectangle() : Rectangle(Point(), Point()) {
  }

  void Rectangle::print(std::ostream &os) const {
    os << "( " << lower.x << ", " << lower.y << " ) - ( " << upper.x << ", " << upper.y << " )";
  }

  bool Rectangle::intersectsWith(const Rectangle &other) const {
    return this->lower.y <= other.upper.y && this->upper.x >= other.lower.x
        && this->upper.y >= other.lower.y && this->lower.x <= other.upper.x;
  }

  bool Rectangle::isInsideOf(const Rectangle &other) const {
    return this->lower.x > other.lower.x && this->upper.x < other.upper.x
        && this->lower.y > other.lower.y && this->upper.y < other.upper.y;
  }

  Rectangle Rectangle::min_bounding(const Rectangle &other) const {
    Point lower(std::min(this->lower.x, other.lower.x), std::min(this->lower.y, other.lower.y));
    Point upper(std::max(this->upper.x, other.upper.x), std::max(this->upper.y, other.upper.y));
    return Rectangle(lower, upper);
  }
}
