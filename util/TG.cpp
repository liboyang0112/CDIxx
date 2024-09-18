#include "cuPlotter.hpp"
#include "cudaConfig.hpp"
#include "format.hpp"
#include "imgio.hpp"
#include <ctime>
#include <math.h>
#include <complex.h>

Real gaussian(Real x, Real y, Real sigma) {
  Real r2 = sq(x) + sq(y);
  return exp(-r2 / 2 / sq(sigma));
}

Real gaussian_norm(Real x, Real y, Real sigma) {
  return 1. / (2 * M_PI * sigma * sigma) * gaussian(x, y, sigma);
}

int main() {
  init_cuda_image();
  int row = 512;
  int column = 512;
  resize_cuda_image(row,column);
  // These are in mm;
  Real lambda = 800e-6;
  Real dhole = 2; // distance between two holes of pump light
  Real focus = 50;
  Real pixelsize = 3e-3;
  Real spotSize = 60e-3;
  Real dn = 1e-3; // n2
  Real dx = 0.1;  // thickness
  Real phi0 = 2 * M_PI * dn * dx / lambda;

  int spotpix = spotSize / pixelsize;
  Real k = sin(dhole / 2 / focus) * 2 * M_PI / lambda * pixelsize;
  myDMalloc(complexFormat, inputField, row * column);
  myCuDMalloc(complexFormat, d_inputField, row * column);
  myCuDMalloc(complexFormat, imageTarget, row * column);

  int index = 0;
  for (int x = 0; x < row; x++) {
    for (int y = 0; y < column; y++) {
      Real Emod = gaussian(x - 0.5 * row, y - 0.5 * column, spotpix);
      Real phase = Emod * Emod * 2 * (1 - cos(2 * k * x)) * phi0;
      inputField[index] = Emod * sin(phase) + Emod * cos(phase) * _Complex_I;
      index++;
    }
  }
  myMemcpyH2D(d_inputField, inputField, row*column*sizeof(complexFormat));
  plt.init(row,column);
  plt.plotComplexColor(d_inputField, 0, 1, "input", 0, 0);
  cudaConvertFO(d_inputField);
  init_fft(row,column);
  myFFT(d_inputField, imageTarget);
  plt.plotComplexColor(imageTarget, 1, 1./row, "output", 0, 0);
  return 0;
}
