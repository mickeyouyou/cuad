#include <chrono>
#include <iostream>
using namespace std;
#define ADOLC_TAPELESS
#include "adolc/adtl.h"
typedef adtl::adouble adouble;

int main(int argc, char *argv[]) {
  auto t1 = std::chrono::steady_clock::now();
  int PointSize = 1024;
  adtl::setNumDir(3);
  adouble x[3 * PointSize], y[3 * PointSize];

  for (int k = 0; k < PointSize; ++k) {
    // Initialize x_i
    for (int i = 0; i < 3; ++i) {
      x[k * 3 + i] = i + 1.0 / (k * 3 + i);
      for (int j = 0; j < 3; ++j)
        if (i == j) x[k * 3 + i].setADValue(j, 1);
    }
  }

  cout << endl
       << "Transform from Cartesian to spherical polar coordinates" << endl
       << endl;

  for (int k = 0; k < PointSize; ++k) {
    for (int i = 0; i < 3; i++) {
      // TODO Respect to every equaltion for every result
      y[k * 3 + i] = sqrt(x[k * 3 + i] * x[k * 3 + i] +
                          x[k * 3 + i] * x[k * 3 + i] + x[k * 3 + i] * x[k]);
    }
    // y[k] = sqrt(x[k] * x[k] + x[k] * x[k] + x[k] * x[k]);
    // y[1] = atan(sqrt(x[0] * x[0] + x[1] * x[1]) / x[2]);
    // y[2] = atan(x[1] / x[0]);
  }

  // cout << "cartesian coordinates: " << endl;
  // cout << "x[0] = " << x[0].getValue() << "  x[1] = " << x[1].getValue()
  //      << "  x[2] = " << x[2].getValue() << endl
  //      << endl;
  // cout << "cpherical polar coordinates: " << endl;
  // cout << "y[0] = " << y[0].getValue() << "  y[1] = " << y[1].getValue()
  //      << "  y[2] = " << y[2].getValue() << endl
  //      << endl;

  // "use" the derivative cout << "derivative:" << endl;
  // for (int k = 0; k < PointSize; ++k) {
  //   for (int i = 0; i < 3; ++i) {
  //     for (int j = 0; j < 3; ++j) cout << y[k * 3 + i].getADValue(j) << "  ";
  //     cout << endl;
  //   }
  // }
  cout << endl;

  auto t2 = std::chrono::steady_clock::now();

  std::cout << "Total time on CPU(ms):"
            << std::chrono::duration<double>(t2 - t1).count() * 1000
            << std::endl;

  return 0;
}
