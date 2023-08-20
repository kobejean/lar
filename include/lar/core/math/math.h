#ifndef LAR_CORE_MATH_MATH_H
#define LAR_CORE_MATH_MATH_H

namespace lar {
  constexpr double epsilon = 1e-21;

  constexpr double abs(double x) {
    return x < 0 ? -x : x;
  }

  constexpr int ceil(double num) {
    int integer = static_cast<int>(num);
    return (static_cast<float>(integer) == num)
      ? integer : integer + ((num > 0) ? 1 : 0);
  }

  // Compute the square root using the Babylonian method
  constexpr double sqrt(double value) {
    double result = value;
    double delta = 0.0;
    do {
      double temp = (result + value / result) / 2.0;
      delta = temp - result;
      result = temp;
    } while (abs(delta) > epsilon);
    return result;
  }

  constexpr double log(double x) {
    double sum = 0.0, term = (x - 1) / (x + 1);
    double square_term = term * term;
    for (int i = 1; abs(term) > epsilon; i += 2) {
      sum += term / i;
      term *= square_term;
    }
    return 2 * sum;
  }

  // Compute e raised to the power of x using Taylor series expansion
  constexpr double exp(double x) {
    double sum = 1.0;
    double term = 1.0;
    for (int i = 1; i < 100; ++i) {
      term *= x / i;
      sum += term;
    }
    return sum;
  }

  constexpr double pow(double base, double exponent) {
    if (exponent == 0.0) return 1.0;
    if (base < 0.0) return 0.0 / 0.0; // NaN for negative base with fractional exponent
    return exp(exponent * log(base));
  }

}

#endif /* LAR_CORE_MATH_MATH_H */