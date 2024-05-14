#include <complex>
#include <stdio.h>
#include <stdio.h>
#include "imgio.hpp"
#include <string.h>
#include "cudaConfig.hpp"
#include "cuPlotter.hpp"
#include "cub_wrap.hpp"

Real* cropToMiddle(Real* img, int row, int col, Real shiftx, Real shifty, int &outrow, int &outcol){
  int sz = row*col;

  myCuDMalloc(Real, d_bit, sz);
  resize_cuda_image(row,col);
  bitMap(d_bit, img, 0.2);
  std::complex<Real> mid(findMiddle(d_bit, row*col)+shiftx-shifty*1.0i);
  printf("mid= %f,%f\n",mid.real(),mid.imag());
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
Real *readFromCDIRecon(const char* fname, int &row, int &col, Real mg, Real shiftx, Real shifty){
  imageFile fdata;
  FILE* frestart = fopen(fname, "r");
  printf("reading file: %s\n",fname);
  if(frestart)
    if(!fread(&fdata, sizeof(fdata), 1, frestart)){
      printf("WARNING: file %s is empty!\n", fname);
    }
  row = fdata.rows;
  col = fdata.cols;
  size_t sz = row*col*sizeof(complexFormat);
  myDMalloc(complexFormat, wf, row*col);
  if(!fread(wf, sz, 1, frestart)){
    printf("WARNING: file %s is empty!\n", fname);
  }
  complexFormat *d_wfr, *d_wf;
  myCuMalloc(complexFormat, d_wfr, row*col);
  myMemcpyH2D(d_wfr, wf, sz);
  ccmemMngr.returnCache(wf);
  row*=mg; col*=mg;
  resize_cuda_image(row,col);
  if(row != fdata.rows && col != fdata.cols){
    myCuMalloc(complexFormat, d_wf, row*col);
    cropinner(d_wfr, d_wf, fdata.rows, fdata.cols, 1./sqrt(row*col));
  }else{
    d_wf = d_wfr;
    applyNorm(d_wf, 1./sqrt(row*col));
  }
  cudaConvertFO(d_wf);
  multiplyShift(d_wf, shiftx, shifty);
  cudaConvertFO(d_wf);
  init_fft(row, col);
  myIFFT(d_wf, d_wf);
  plt.init(row,col);
  plt.plotComplex(d_wf, MOD2, 0, 1, "restart_pattern");
  myCuDMalloc(Real, d_int, row*col);
  getMod2(d_int, d_wf);
  memMngr.returnCache(d_wf);
  return d_int;
}

Real* readFromPng(const char* fname, int &row, int &col){
  Real* img = readImage(fname, row, col);
  size_t sz = row*col;
  myCuDMalloc(Real,d_img,sz);
  myMemcpyH2D(d_img, img, sz*sizeof(Real));
  ccmemMngr.returnCache(img);
  return d_img;
}

Real* readimg(const char* fname, int &row, int& col, Real mg = 1, Real shiftx = 0, Real shifty = 0){
  if(strstr(fname,".bin")){
    return readFromCDIRecon(fname, row, col, mg, shiftx, shifty);
  }
  return readFromPng(fname,row,col);
}

Real decimal(Real val){
    return val - floor(val);
}

int main(int argc, char* argv[] )
{
  int row, col, row1, col1;
  init_cuda_image(rcolor, 1);
  Real mg = 1;
  Real crpn = 0.4;
  if(argc > 6) mg = std::stof(argv[6]);
  Real shiftx = decimal(std::stof(argv[4]));
  Real shifty = decimal(std::stof(argv[3]));
  Real* d_sig = readimg(argv[2], row1, col1, mg, shiftx, shifty);
  if(argc > 5 && strstr(argv[5], "tp")){
    transpose(d_sig);
  }
  int outrow = 0;
  int outcol = 0;
  shiftx = std::stof(argv[4])/col1;
  shifty = std::stof(argv[3])/row1;
  Real* sigout = cropToMiddle(d_sig, row1, col1, shiftx, shifty, outrow, outcol);
  Real* d_bkg = readimg(argv[1], row, col);
  Real* bkgout = cropToMiddle(d_bkg, row, col, 0, 0, outrow, outcol);
  plt.init(outrow, outcol);
  plt.plotFloat(sigout, MOD, 0, 1, "sig", 0, 0, 1);
  plt.plotFloat(bkgout, MOD, 0, 1, "bkg", 0, 0, 1);
  Real rat = findSum(sigout) / findSum(bkgout);
  add(sigout, bkgout, -rat);
  Real maxr = findMax(bkgout);

  myCuDMalloc(Real, crpd, outrow*outcol*crpn*crpn);
  resize_cuda_image(outrow*crpn, outcol*crpn);
  crop(sigout, crpd, outrow, outcol);
  plt.init(outrow*crpn, outcol*crpn);
  plt.plotFloat(crpd, REAL, 0, 1./maxr, "diff", 0, 0, 1);
  getMod2(crpd, crpd);
  Real mse = findSum(crpd);
  Real psnr = 10*log10(outrow*outcol*maxr*maxr*crpn*crpn/mse);
  printf("psnr=%f\n",psnr);
  return 0;
}

