#include <fmt/base.h>
#include <complex.h>
#include "cudaConfig.hpp"
#include "cuPlotter.hpp"
#include "cub_wrap.hpp"
#include "fmt/core.h"
#include "imageFile.hpp"
#include "beamDecomposition.hpp"

int main () {
//int main (int argc, char *argv[]) {
  int row = 512,col = 512;

  imageFile fdata;
  const char* filename = "pupilwave.bin";
  FILE* file = fopen(filename, "r");
  if(!fread(&fdata, sizeof(fdata), 1, file)){
    fmt::println("WARNING: file {} is empty!", filename);
  }
  row = fdata.rows;
  col = fdata.cols;
  size_t sz = row*col*sizeof(complexFormat);
  complexFormat *wf = (complexFormat*) ccmemMngr.borrowCache(sz);
  if(!fread(wf, sz, 1, file)){
    fmt::println("WARNING: file {} is empty!", filename);
  }

  myCuDMallocClean(complexFormat, img, row*col);
  myCuDMallocClean(complexFormat, img2, row*col);
  init_cuda_image();
  resize_cuda_image(row, col);
  /*
  complexFormat shift = 1;
  linearConst(img, img, 0, shift);
  myMemcpyD2D(img2, img, row*col*sizeof(complexFormat));
  multiplyZernikeConj(img, img, 48, 1, 1);
  //multiplyLaguerreConj(img, img, 40, 0, -2);
  //multiplyLaguerreConj(img2, img2, 20, 0, -1);
  //multiplyHermit(img, img, 20, 2, 2);
  //applyGaussMult(img, img, 40, 0);
  //add(img, img2);
  */
  myMemcpyH2D(img, wf, sz);
  ccmemMngr.returnCache(wf);
  plt.init(row,col);
  plt.plotComplexColor(img, 0, 1, "output");

  /*
     int maxn = 10;
     int maxl = 10;
     complexFormat** out = laguerreDecomposition(img, maxn, maxl, 40, NULL);
     int ibase = 0;
     for (int i = 0; i <= maxn; i++) {
     for (int j = -maxl; j <= maxl; j++) {
     fmt::print("{}, {} coefficient = ({}, {})\n", i, j, crealf(out[0][ibase]), cimagf(out[0][ibase]));
     ibase++;
     }
     }
     */

  //complexFormat** out = zernikeDecomposition(img, 3, 28, NULL);
  int maxn = 5;
  complexFormat** out = zernikeDecomposition(img, maxn, 48, NULL);
  int ibase = 0;
  for (int i = 0; i <= maxn; i++) {
    for (int j = -i; j <= i; j+=2) {
      fmt::print("{}, {} coefficient = ({:.2f}, {:.2f})\n", i, j, crealf(out[0][ibase]), cimagf(out[0][ibase]));
      ibase++;
    }
  }
  plt.plotComplexColor(out[1], 0, 1, "projected");
  add(img, out[1], -1);
  getMod2((Real*)img2, img);
  Real residual = findSum((Real*)img2, row*col) / (row*col);
  fmt::println("residual = {}\n", residual);
  return 0;
}
