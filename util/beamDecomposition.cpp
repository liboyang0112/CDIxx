#include <fmt/base.h>
#include <complex.h>
#include "cudaConfig.hpp"
#include "cuPlotter.hpp"
#include "cub_wrap.hpp"
#include "fmt/core.h"
#include "beamDecomposition.hpp"

int main () {
//int main (int argc, char *argv[]) {
  int row = 1024,col = 1024;
/*
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
*/

  myCuDMallocClean(complexFormat, img, row*col);
/*
  myMemcpyH2D(img, wf, sz);
  ccmemMngr.returnCache(wf);
*/

  myCuDMallocClean(complexFormat, img2, row*col);
  init_cuda_image();
  resize_cuda_image(row, col);
  complexFormat shift = 1;
  linearConst(img, img, 0, shift);
  myMemcpyD2D(img2, img, row*col*sizeof(complexFormat));
  int maxn = 40;
  Real radius = 500;
  multiplyZernikeConj(img, img, radius, 20, 0);
  //multiplyLaguerreConj(img, img, 20, 10, 10);
  //multiplyLaguerreConj(img2, img2, 20, 0, -1);
  //multiplyHermit(img, img, 200, 9, 9);
  //applyGaussMult(img, img, 40, 0);
  //add(img, img2);
  plt.init(row,col);
  plt.plotComplexColor(img, 0, 1, "output");

  //int maxl = 200;
  int ibase = 0;
  /*
  void* handle = laguerre_init(row, col, maxn, maxl, 0);
  laguerre_compute(handle , img, row>>1 , col>>1, 500);
  complexFormat* coeff = laguerre_coeff(handle);
  for (int j = 0; j <= maxl; j++) {
    for (int i = 0; i <= maxn; i++) {
      fmt::print("{}, {} coefficient = ({}, {})\n", i, j, crealf(coeff[ibase]), cimagf(coeff[ibase]));
      ibase++;
      if(j!=0){
        fmt::print("{}, {} coefficient = ({}, {})\n", i, -j, crealf(coeff[ibase]), cimagf(coeff[ibase]));
        ibase++;
      }
    }
  }
  laguerre_reconstruct(handle, img, 500);
  plt.plotComplexColor(img, 0, 1, "projected");
  */
  
  /*
  complexFormat** out = laguerreDecomposition(img, maxn, maxl, 40, NULL);
  int ibase = 0;
  for (int i = 0; i <= maxn; i++) {
    for (int j = -maxl; j <= maxl; j++) {
      fmt::print("{}, {} coefficient = ({}, {})\n", i, j, crealf(out[0][ibase]), cimagf(out[0][ibase]));
      ibase++;
    }
  }
  plt.plotComplexColor(out[1], 0, 1, "projected");
  */

  //  //complexFormat** out = zernikeDecomposition(img, 3, 28, NULL);
  //
  //  /*
  //  complexFormat** out = zernikeDecomposition(img, maxn, 48, NULL);
  //  for (int i = 0; i <= maxn; i++) {
  //    for (int j = -i; j <= i; j+=2) {
  //      fmt::print("{}, {} coefficient = ({:.2f}, {:.2f})\n", i, j, crealf(out[0][ibase]), cimagf(out[0][ibase]));
  //      ibase++;
  //    }
  //  }
  //  */
  //
    void* handle = zernike_init(row, col, maxn, 0);
    zernike_compute(handle , img, 511.5,511.5,radius);
    complexFormat* coeff = zernike_coeff(handle);
    Real threshold = 1e-4;
    for (int j = 0; j <= maxn; j++) {
      for (int i = j; i <= maxn; i+=2) {
        if(fabs(crealf(coeff[ibase]))>threshold||fabs(cimagf(coeff[ibase]))>threshold)
        fmt::print("{}, {} coefficient = ({}, {})\n", i, j, crealf(coeff[ibase]), cimagf(coeff[ibase]));
        ibase++;
        if(j!=0){
          if(fabs(crealf(coeff[ibase]))>threshold||fabs(cimagf(coeff[ibase]))>threshold)
          fmt::print("{}, {} coefficient = ({}, {})\n", i, -j, crealf(coeff[ibase]), cimagf(coeff[ibase]));
          ibase++;
        }
      }
    }
    zernike_reconstruct(handle, img2, radius);
    plt.plotComplexColor(img2, 0, 1, "projected");
    add(img, img2, -1);
    getMod2((Real*)img2, img);
    plt.plotFloat(img2, MOD, 0, 1e2, "residual", 0, 0, 1);
    Real residual = sqrt(findSum((Real*)img2, row*col) / (row*col));
    fmt::println("residual = {}\n", residual);
  return 0;
}
