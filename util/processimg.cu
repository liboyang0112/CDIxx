#include <complex>
#include <cassert>
#include <stdio.h>
#include <time.h>
#include <stdio.h>
#include "common.h"
#include "cudaConfig.h"
#include "experimentConfig.h"
#include "cuPlotter.h"
#include "cub_wrap.h"

int main(int argc, char** argv )
{
  cudaFree(0); // to speed up the cuda malloc; https://forums.developer.nvidia.com/t/cudamalloc-slow/40238
  int nmerge = atoi(argv[3]);
  int row, col, row1, col1;
  Real* bkg = readImage(argv[1], row, col);
  size_t sz = row*col*sizeof(Real);
  Real* d_bkg = (Real*)memMngr.borrowCache(sz);
  cudaMemcpy(d_bkg, bkg, sz, cudaMemcpyHostToDevice);
  ccmemMngr.returnCache(bkg);
  Real* sig = readImage(argv[2], row1, col1);
  sz = row1*col1*sizeof(Real);
  Real* d_sig = (Real*)memMngr.borrowCache(sz);
  cudaMemcpy(d_sig, sig, sz, cudaMemcpyHostToDevice);
  ccmemMngr.returnCache(sig);
  init_cuda_image(rcolor, 1);
  if(row > row1){
    Real* tmp = (Real*)memMngr.borrowCache(row1*col1*sizeof(Real));
    resize_cuda_image(row1, col1);
    crop(d_bkg, tmp, row, col);
    memMngr.returnCache(d_bkg);
    d_bkg = tmp;
    row = row1;
    col = col1;
  }else{
    Real* tmp = (Real*)memMngr.borrowCache(row*col*sizeof(Real));
    resize_cuda_image(row, col);
    crop(d_sig, tmp, row1, col1);
    memMngr.returnCache(d_sig);
    d_sig = tmp;
  }

  add(d_sig, d_bkg, -1);
  plt.init(row, col);
  plt.plotFloat(d_sig, MOD, 0, 1, "logimage", 1);
  Real* d_bit = (Real*)memMngr.borrowCache(sz);
  applyThreshold(d_bit, d_sig, 0.01);
  auto mid = findMiddle(d_bit, row*col);
  memMngr.returnCache(d_bit);
  int step = nmerge*4;
  int outrow = (row-int(abs(mid.x)*row)*2)/step*step;
  int outcol = (col-int(abs(mid.y)*col)*2)/step*step;
  outrow = outcol = min(outrow,outcol);
  resize_cuda_image(outrow, outcol);
  Real* tmp = (Real*)memMngr.borrowCache(outrow*outcol*sizeof(Real));
  printf("mid= %f,%f\n",mid.x,mid.y);
  Real shiftx = int(mid.x*row)-mid.x*row;
  Real shifty = int(mid.y*col)-mid.y*col;
  printf("shift = %f,%f\n",shiftx, shifty);
  crop(d_sig, tmp, row, col,mid.x,mid.y);
  complexFormat* tmp1 = (complexFormat*)memMngr.borrowCache(outrow*outcol*sizeof(Real)*2);
  extendToComplex(tmp, tmp1);
  init_fft(outrow,outcol);
  if(argv[5][0]=='1') shiftMiddle(tmp1);
  getReal(tmp, tmp1);
  int finsize = outrow/nmerge;
  resize_cuda_image(finsize,finsize);
  mergePixel(d_sig, tmp, outrow, outcol, nmerge);
  plt.init(finsize, finsize);
  plt.plotFloat(d_sig, REAL, 0, 1, "logimagemerged", 1);
  plt.saveFloat(d_sig, argv[4]);
  return 0;
}

