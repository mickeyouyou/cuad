
#include <iostream>
using namespace std;

// #define ADOLC_TAPELESS
// #include "adolc/adouble.h"
#include "adolc/adtl.h"
using adouble = adtl::adouble;
// typedef adtl::adouble adouble;

/**
 * V1 for logic verify
 **/

/**
 * V2 for 1024 iteration
 **/

int main() {
  // adtl::setNumDir(3);
  adouble x[4];
  adouble v[3];
  adouble y[2];
  /* initialization of x */
  x[0] = 3.7;
  x[1] = 0.7;
  x[2] = 0.5;
  x[3] = 0.5;
  /* initialization of x˙ */
  x[0].setADValue(1);
  // x[1].setADValue(0);
  // x[2].setADValue(0);
  // x[3].setADValue(0);

  v[0] = x[3] * x[3];
  v[1] = tan(v[0]);
  v[0] = x[1] - v[1]; /* same as before */
  v[2] = x[0] * v[1];
  v[1] = v[2] / v[0];
  v[2] = v[1] * x[1];
  y[0] = v[1];
  y[1] = v[2];
  cout << "y0: " << y[0].getADValue() << " y1:" << y[1].getADValue() << endl;
}