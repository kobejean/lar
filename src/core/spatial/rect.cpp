#include <iostream>
#include "lar/core/spatial/rect.h"

namespace lar {

  Rect::Rect(Point lower, Point upper) : lower(lower), upper(upper) {
  }

  Rect::Rect(double x1, double y1, double x2, double y2) : lower(x1, y1), upper(x2, y2) {
  }

  Rect::Rect(Point center, double width, double height) : lower(center.x - width / 2, center.y - height / 2), upper(center.x + width / 2, center.y + height / 2) {
  }

  Rect::Rect() : Rect(Point(), Point()) {
  }

  void Rect::print(std::ostream &os) const {
    os << "( " << lower.x << ", " << lower.y << " ) - ( " << upper.x << ", " << upper.y << " )";
  }

  bool Rect::intersectsWith(const Rect &other) const {
    return this->lower.y <= other.upper.y && this->upper.x >= other.lower.x
        && this->upper.y >= other.lower.y && this->lower.x <= other.upper.x;
  }

  bool Rect::isInsideOf(const Rect &other) const {
    return this->lower.x > other.lower.x && this->upper.x < other.upper.x
        && this->lower.y > other.lower.y && this->upper.y < other.upper.y;
  }

  Rect Rect::minBoundingBox(const Rect &other) const {
    Point lower(std::min(this->lower.x, other.lower.x), std::min(this->lower.y, other.lower.y));
    Point upper(std::max(this->upper.x, other.upper.x), std::max(this->upper.y, other.upper.y));
    return Rect(lower, upper);
  }

  double Rect::area() const {
    return (upper.x - lower.x) * (upper.y - lower.y);
  }

  Rect Rect::intersection(const Rect &other) const {
      Rect rect = *this;
      if (this->lower.x < other.lower.x) rect.lower.x = other.lower.x;
      if (this->lower.y < other.lower.y) rect.lower.y = other.lower.y;
      if (this->upper.x > other.upper.x) rect.upper.x = other.upper.x;
      if (this->upper.y > other.upper.y) rect.upper.y = other.upper.y;
      return rect;
  }

  double Rect::overlap(const Rect &other) const {
    return this->intersection(other).area();
  }
}
