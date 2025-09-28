#include "cudaConfig.hpp"
#include "cub_wrap.hpp"
#include <cmath>

complexFormat** zernikeDecomposition(complexFormat* img, int maxn, int radius, complexFormat* coefficient = NULL, complexFormat* projected = NULL){
  // Using cuda_row and cuda_column, please set resize_cuda_image(row, col) properly before calling this function
  myDMalloc(complexFormat*, output, 2);
  int n_base = (maxn+1)*(maxn+2)/2;
  size_t sz = getCudaCols()*getCudaRows();
  if(projected == NULL) myCuMallocClean(complexFormat, projected, sz);
  if(coefficient == NULL) myMalloc(complexFormat, coefficient, n_base);
  output[0] = coefficient;
  output[1] = projected;
  int ibase = 0;
  for (int n = 0; n <= maxn; n++) {
    for (int m = -n; m <= n; m+=2) {
      multiplyZernikeConj(projected, img, radius, n, m);
      coefficient[ibase] = findSum(projected)/(M_PI*radius*radius);
      ibase++;
    }
  }
  ibase = 0;
  clearCuMem(projected, sz*sizeof(complexFormat));
  for (int n = 0; n <= maxn; n++) {
    for (int m = -n; m <= n; m+=2) {
      addZernike(projected, coefficient[ibase], radius, n, m);
      ibase++;
    }
  }
  return output;
}

complexFormat** laguerreDecomposition(complexFormat* img, int maxn, int maxl, int radius, complexFormat* coefficient = NULL, complexFormat* projected = NULL){
  // Using cuda_row and cuda_column, please set resize_cuda_image(row, col) properly before calling this function
  int n_base = (maxn+1)*(2*maxl+1);
  size_t sz = getCudaCols()*getCudaRows();
  complexFormat** output = NULL;
  if(projected == NULL && coefficient == NULL) myMalloc(complexFormat*, output, 2);
  if(projected == NULL) myCuMallocClean(complexFormat, projected, sz);
  if(coefficient == NULL) myMalloc(complexFormat, coefficient, n_base);
  if(output){
    output[0] = coefficient;
    output[1] = projected;
  }
  int ibase = 0;
  for (int n = 0; n <= maxn; n++) {
    for (int m = -maxl; m <= maxl; m+=1) {
      multiplyLaguerreConj(projected, img, radius, n, m);
      coefficient[ibase] = findSum(projected)/(radius*radius);
      ibase++;
    }
  }
  ibase = 0;
  clearCuMem(projected, sz*sizeof(complexFormat));
  for (int n = 0; n <= maxn; n++) {
    for (int m = -maxl; m <= maxl; m+=1) {
      addLaguerre(projected, coefficient[ibase], radius, n, m);
      ibase++;
    }
  }
  return output;
}
