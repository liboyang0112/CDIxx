#include <complex>
#include <stdio.h>
#include <stdio.h>
#include "imgio.h"
#include "cudaConfig.h"
#include "cuPlotter.h"
#include "cub_wrap.h"

int main(int argc, char** argv )
{
  int nmerge = atoi(argv[3]);
  int row, col, row1, col1;
  Real* bkg = readImage(argv[1], row, col);
  size_t sz = row*col*sizeof(Real);
  myCuDMalloc(Real,d_bkg,sz);
  myMemcpyH2D(d_bkg, bkg, sz);
  ccmemMngr.returnCache(bkg);
  Real* sig = readImage(argv[2], row1, col1);
  sz = row1*col1*sizeof(Real);
  myCuDMalloc(Real, d_sig, sz);
  myCuDMalloc(Real, d_stretched, sz);
  myMemcpyH2D(d_sig, sig, sz);
  ccmemMngr.returnCache(sig);
  init_cuda_image(rcolor, 1);
  if(row!=row1){
    if(row > row1){
      myCuDMalloc(Real, tmp, row1*col1);
      resize_cuda_image(row1, col1);
      crop(d_bkg, tmp, row, col);
      memMngr.returnCache(d_bkg);
      d_bkg = tmp;
      row = row1;
      col = col1;
    }else{
      myCuDMalloc(Real, tmp, row*col);
      resize_cuda_image(row, col);
      crop(d_sig, tmp, row1, col1);
      memMngr.returnCache(d_sig);
      d_sig = tmp;
    }
  }else{
      resize_cuda_image(row, col);
  }

  add(d_sig, d_bkg, -1);
  //plt.init(row, col);
  //plt.plotFloat(d_sig, MOD, 0, 1, "logimage", 1);
  myCuDMalloc(Real, d_bit, sz);
  myCuDMalloc(Real, d_mask, sz);
  rect spt;
  spt.startx = row/2-100;
  spt.starty = col/2-100;
  spt.endx = row/2+100;
  spt.endy = col/2+100;
  myCuDMalloc(rect, cuda_spt, 1);
  myMemcpyH2D(cuda_spt, &spt, sizeof(rect));
  myMemcpyD2D(d_bit, d_sig, sz);
  createMask(d_mask, cuda_spt);
  applyMask(d_bit, d_mask);
  applyThreshold(d_bit, d_sig, 0.2);
  //plt.plotFloat(d_bit, MOD, 0, 1, "bit", 1);
  std::complex<Real> mid(findMiddle(d_bit, row*col));
  memMngr.returnCache(d_bit);
  memMngr.returnCache(d_mask);
  memMngr.returnCache(cuda_spt);
  if(argc >= 7){
    mid += std::complex<Real>(std::stof(argv[7])/col - std::stof(argv[6])/row*1.0i);
  }
  int step = nmerge*4;
  int outrow = (row-int(std::abs(mid.real())*row)*2)/step*step;
  int outcol = (col-int(std::abs(mid.imag())*col)*2)/step*step;
  //int outrow = row/step*step;
  //int outcol = col/step*step;
  outrow = outcol = std::min(outrow,outcol);
  resize_cuda_image(outrow, outcol);
  myCuDMalloc(Real, tmp, outrow*outcol);
  printf("mid= %f,%f\n",mid.real(),mid.imag());
  Real shiftx = int(mid.real()*row)-mid.real()*row;
  Real shifty = int(mid.imag()*col)-mid.imag()*col;
  printf("shift = %f,%f\n",shiftx, shifty);
  crop(d_sig, tmp, row, col,mid.real(),mid.imag());
  //crop(d_sig, tmp, row, col);
  myCuDMalloc(complexFormat, tmp1, outrow*outcol);
  extendToComplex(tmp, tmp1);
  init_fft(outrow,outcol);
  if(argv[5][0]=='1') shiftMiddle(tmp1);
  else shiftWave(tmp1, shiftx, shifty);
  getReal(tmp, tmp1);
  int finsize = outrow/nmerge;
  resize_cuda_image(finsize,finsize);
  mergePixel(d_sig, tmp, outrow, outcol, nmerge);
  plt.init(finsize, finsize);
  myCuDMalloc(complexFormat, xc, finsize*finsize);
  extendToComplex(d_sig,xc);
  init_fft(finsize,finsize);
  myFFT(xc, xc);
  plt.plotFloat(d_sig, MOD, 0, 1, "logimagemerged", 1, 0, 1);
  plt.plotFloat(d_stretched, MOD, 0, 4, "logimagemerged_str", 1, 0, 1);
  plt.plotComplex(xc, MOD2, 1, 1./finsize, "autocorrelation", 1, 0, 1);
  plt.saveFloat(d_sig, argv[4]);
  return 0;
}

