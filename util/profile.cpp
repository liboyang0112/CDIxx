#include <complex>
#include <fstream>
#include <stdio.h>
#include <stdio.h>
#include "fmt/core.h"
#include "imgio.hpp"
#include <string.h>
#include <iostream>
#include "cudaConfig.hpp"
#include "cuPlotter.hpp"
#include "cub_wrap.hpp"
#include "misc.hpp"

Real *readFromCDIRecon(const char* fname, int &row, int &col){
  imageFile fdata;
  FILE* frestart = fopen(fname, "r");
  fmt::println("reading file: {}",fname);
  if(frestart)
    if(!fread(&fdata, sizeof(fdata), 1, frestart)){
      fmt::println("WARNING: file {} is empty!", fname);
    }
  row = fdata.rows;
  col = fdata.cols;
  size_t sz = row*col*sizeof(complexFormat);
  myDMalloc(complexFormat, wf, row*col);
  if(!fread(wf, sz, 1, frestart)){
    fmt::println("WARNING: file {} is empty!", fname);
  }
  myCuDMalloc(complexFormat, d_wfr, row*col);
  myMemcpyH2D(d_wfr, wf, sz);
  ccmemMngr.returnCache(wf);
  init_fft(row, col);
  myIFFT(d_wfr,d_wfr);
  myCuDMalloc(Real, ret, row*col);
  resize_cuda_image(row,col);
  getMod2(ret, d_wfr);
  return ret;
}

Real* cropToMiddle(Real* img, int row, int col, Real shiftx, Real shifty, int &outrow, int &outcol){
  int sz = row*col;

  myCuDMalloc(Real, d_bit, sz);
  resize_cuda_image(row,col);
  bitMap(d_bit, img, 0.2);
  std::complex<Real> mid(findMiddle(d_bit, row*col));
  mid += std::complex<Real>(shiftx, -shifty);
  fmt::println("mid= {:f},{:f}",mid.real(),mid.imag());
  memMngr.returnCache(d_bit);
  if(!outrow || !outcol){
    outrow = (row-int(std::abs(mid.real())*row)*2)/4*4;
    outcol = (col-int(std::abs(mid.imag())*col)*2)/4*4;
    outrow = outcol = std::min(outrow,outcol);
  }
  plt.init(row, col);
  plt.plotFloat(d_bit, MOD, 0, 1, "bitmap");
  myCuDMalloc(Real, imgout, outrow*outcol);
  resize_cuda_image(outrow, outcol);
  crop(img, imgout, row, col,mid.real(),mid.imag());
  shiftx = int(mid.real()*row)-mid.real()*row;
  shifty = int(mid.imag()*col)-mid.imag()*col;
  return imgout;
}

int main(int argc, char* argv[] )
{
  int row, col, row1, col1;
  init_cuda_image(rcolor, 1);
  Real mg = 1;
  Real crpn = 0.2;
  int linenumber = 39;
  if(argc > 6) mg = std::stof(argv[6]);
  if(argc > 7) crpn = std::stof(argv[7]);
  Real *d_sig = readFromCDIRecon(argv[2], row1, col1);
  if(argc > 5){
  if(strstr(argv[5], "tp"))
    transpose(d_sig);
  if(strstr(argv[5], "f"))
    flipx(d_sig);
  }
  int outrow = row1*crpn;
  int outcol = col1*crpn;
  int outrowsig = row1*crpn;
  int outcolsig = col1*crpn;
  Real shiftx = std::stof(argv[4])/col1;
  Real shifty = std::stof(argv[3])/row1;
  if(mg < 1){
    outrow*=mg;
    outcol*=mg;
  }else{
    outrowsig/=mg;
    outcolsig/=mg;
  }
  Real* sigout = cropToMiddle(d_sig, row1, col1, shiftx, shifty, outrowsig, outcolsig);
  applyNorm(sigout, 1./findMax(sigout));
  Real* d_bkg = readFromCDIRecon(argv[1], row, col);
  Real* bkgout = cropToMiddle(d_bkg, row, col, 0, 0, outrow, outcol);
  initCub();
  applyNorm(bkgout, 1./findMax(bkgout));
  myDMalloc(Real, hsig, outrowsig*outcolsig);
  myDMalloc(Real, hbkg, outrow*outcol);
  myMemcpyD2H(hsig, sigout, outrowsig*outcolsig*sizeof(Real));
  myMemcpyD2H(hbkg, bkgout, outrow*outcol*sizeof(Real));
  std::ofstream linedata("linedata.txt", std::ios::out);
  Real unit = 3.37;
  for(int i = 0; i < outrow; i++){
    linedata << i*unit/mg << " " << hbkg[linenumber+outcol*i]<<std::endl;
    linedata << (i+1)*unit/mg << " " << hbkg[linenumber+outcol*i]<<std::endl;
    hbkg[linenumber+outcol*i] = 1;
  }
  linedata.close();
  linenumber *= 1./mg;
  std::ofstream linedatasig("linedatasig.txt", std::ios::out);
  for(int i = 0; i < outrowsig; i++){
    linedatasig << i*unit << " " << hsig[linenumber + outcolsig*i] <<std::endl;
    linedatasig << (i+1)*unit << " " << hsig[linenumber + outcolsig*i] <<std::endl;
    hsig[linenumber+outcolsig*i] = 1;
  }
  linedatasig.close();
  myMemcpyH2D(sigout, hsig, outrowsig*outcolsig*sizeof(Real));
  myMemcpyH2D(bkgout, hbkg, outrow*outcol*sizeof(Real));
  resize_cuda_image(outrowsig, outcolsig);
  plt.init(outrowsig, outcolsig);
  plt.plotFloat(sigout, MOD, 0, 1, "sigout", 0, 0, 0);
  resize_cuda_image(outrow, outcol);
  plt.init(outrow, outcol);
  plt.plotFloat(bkgout, MOD, 0, 1, "bkgout", 0, 0, 0);
  
  return 0;
}

