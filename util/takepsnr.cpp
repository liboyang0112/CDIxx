#include <complex>
#include <stdio.h>
#include <stdio.h>
#include "imgio.hpp"
#include <string.h>
#include "cudaConfig.hpp"
#include "cuPlotter.hpp"
#include "cub_wrap.hpp"
#include "misc.hpp"

Real* ssim_map(Real* img1, Real* img2, int row, int col, Real sigma){
  Real C1=1e-4, C2=1e-3;
  myCuDMalloc(Real, mu1, row*col);
  myCuDMalloc(Real, mu2, row*col);
  myCuDMalloc(Real, sigma1sq, row*col);
  myCuDMalloc(Real, sigma2sq, row*col);
  myCuDMalloc(Real, sigma12, row*col);
  myCuDMalloc(Real, kernel, 11*11);
  getMod2(mu1, img1);
  getMod2(mu2, img2);
  applyGaussConv(mu1, sigma1sq, kernel, sigma, 5);
  applyGaussConv(mu2, sigma2sq, kernel, sigma, 5);
  multiply(mu1, img1, img2);
  applyGaussConv(mu1, sigma12, kernel, sigma, 5);
  applyGaussConv(img1, mu1, kernel, sigma, 5);
  applyGaussConv(img2, mu2, kernel, sigma, 5);
  ssimMap(mu1, mu2, sigma1sq, sigma2sq, sigma12, C1, C2);
  return mu1;
}

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
Real* mergewf(complexFormat * d_wfr, int &row, int &col, Real mg, Real shiftx, Real shifty){
  complexFormat *d_wf;
  int initrow = row, initcol = col;
  row*=mg; col*=mg;
  row = row/2*2;
  col = col/2*2;
  resize_cuda_image(row,col);
  if(row != initrow && col != initcol){
    myCuMalloc(complexFormat, d_wf, row*col);
    cropinner(d_wfr, d_wf, initrow, initcol, 1./sqrt(row*col));
  }else{
    d_wf = d_wfr;
    applyNorm(d_wf, 1./sqrt(row*col));
  }
  cudaConvertFO(d_wf);
  multiplyShift(d_wf, shiftx/row, shifty/col);
  cudaConvertFO(d_wf);
  init_fft(row, col);
  myIFFT(d_wf, d_wf);
  //plt.init(row,col);
  //plt.plotComplex(d_wf, MOD2, 0, 1, "restart_pattern");
  myCuDMalloc(Real, d_int, row*col);
  getMod2(d_int, d_wf);
  memMngr.returnCache(d_wf);
  return d_int;
}
complexFormat *readFromCDIRecon(const char* fname, int &row, int &col){
  imageFile fdata;
  FILE* imgfile = fopen(fname, "r");
  printf("reading file: %s\n",fname);
  if(!imgfile || !fread(&fdata, sizeof(fdata), 1, imgfile)){
    printf("WARNING: file %s is empty!\n", fname); exit(0);
  }
  row = fdata.rows;
  col = fdata.cols;
  size_t sz = row*col*sizeof(complexFormat);
  myDMalloc(complexFormat, wf, row*col);
  if(!fread(wf, sz, 1, imgfile)){
    printf("WARNING: data %s is empty!\n", fname); exit(0);
  }
  myCuDMalloc(complexFormat, d_wfr, row*col);
  myMemcpyH2D(d_wfr, wf, sz);
  ccmemMngr.returnCache(wf);
  return d_wfr;
}

complexFormat* readFromPng(const char* fname, int &row, int &col){
  Real* img = readImage(fname, row, col);
  resize_cuda_image(row, col);
  size_t sz = row*col;
  myCuDMalloc(Real,d_img,sz);
  myMemcpyH2D(d_img, img, sz*sizeof(Real));
  ccmemMngr.returnCache(img);
  myCuDMalloc(complexFormat, d_wfr, sz);
  extendToComplex(d_img, d_wfr);
  memMngr.returnCache(d_img);
  init_fft(row, col);
  myFFT(d_wfr, d_wfr);
  applyNorm(d_wfr, 1./sqrt(sz));
  return d_wfr;
}

Real* readimg(const char* fname, int &row, int& col, Real mg = 1, Real shiftx = 0, Real shifty = 0){
  complexFormat* d_wfr;
  if(strstr(fname,".bin")){
    d_wfr = readFromCDIRecon(fname, row, col);
  }else d_wfr = readFromPng(fname,row,col);
  return mergewf(d_wfr, row, col, mg, shiftx, shifty);
}

Real decimal(Real val){
  return val - floor(val);
}

int main(int argc, char* argv[] )
{
  int row, col, row1, col1;
  init_cuda_image(rcolor, 1);
  Real mg = 1;
  Real crpn = 0.2;
  //int linenumber = 30;
  if(argc > 6) mg = std::stof(argv[6]);
  if(argc > 7) crpn = std::stof(argv[7]);
  Real shiftx = decimal(std::stof(argv[4]));
  Real shifty = decimal(std::stof(argv[3]));
  Real* d_sig = readimg(argv[2], row1, col1, mg<1?mg:1, shiftx, shifty);
  if(argc > 5){
  if(strstr(argv[5], "tp"))
    transpose(d_sig);
  if(strstr(argv[5], "f"))
    flipx(d_sig);
  }
  int outrow = row1*crpn;
  int outcol = col1*crpn;
  shiftx = std::stof(argv[4])/col1;
  shifty = std::stof(argv[3])/row1;
  Real* sigout = cropToMiddle(d_sig, row1, col1, shiftx, shifty, outrow, outcol);
  Real* d_bkg = readimg(argv[1], row, col, mg>1?1./mg:1);
  Real* bkgout = cropToMiddle(d_bkg, row, col, 0, 0, outrow, outcol);
  plt.init(outrow, outcol);
  //myDMalloc(Real, hsig, outrow*outcol);
  //myDMalloc(Real, hbkg, outrow*outcol);
  //myMemcpyD2H(hsig, sigout, outrow*outcol*sizeof(Real));
  //myMemcpyD2H(hbkg, bkgout, outrow*outcol*sizeof(Real));
  //std::ofstream linedata("linedata.txt", std::ios::out);
  //for(int i = 0; i < outrow; i++){
  //  linedata << i << " " << hsig[linenumber + outcol*i] << " " << hbkg[linenumber+outcol*i]<<std::endl;
  //  linedata << i+1 << " " << hsig[linenumber + outcol*i] << " " << hbkg[linenumber+outcol*i]<<std::endl;
  //  hsig[linenumber+outcol*i] = 1;
  //}
  //linedata.close();
  applyNorm(sigout, findSum(bkgout)/findSum(sigout));
  applyThreshold(sigout, sigout, 0.25);
  applyThreshold(bkgout, bkgout, 0.25);
  //plt.plotFloat(sigout, MOD, 0, 1, "sig", 0, 0, 1);
  //plt.plotFloat(bkgout, MOD, 0, 1, "bkg", 0, 0, 1);
  Real* ssimmap = ssim_map(sigout, bkgout, outrow, outcol, 1.5);
  printf("ssim=%f\n",findSum(ssimmap)/(outrow*outcol));
  //plt.plotFloat(ssimmap, REAL, 0, 1, "ssim", 0, 0, 1);
  Real maxs = findMax(sigout);
  Real maxb = findMax(bkgout);
  Real maxr = fmin(maxs, 2*maxb);
  add(sigout, bkgout, -1);
  applyNorm(sigout, 1./maxr);

  printf("maxr = %f, s = %f, b = %f\n", maxr, maxs, maxb);
  resize_cuda_image(outrow, outcol);
  plt.plotFloat(sigout, REAL, 0, 1., "diff", 0, 0, 1);
  getMod2(sigout, sigout);
  Real mse = findSum(sigout);
  Real psnr = 10*log10(outrow*outcol/mse);
  printf("psnr=%f\n",psnr);
  //myMemcpyH2D(sigout, hsig, outrow*outcol*sizeof(Real));
  //plt.plotFloat(sigout, MOD, 0, 1., "lineimg", 0, 0, 0);


  //myCuDMalloc(Real, crpd, outrow*outcol*crpn*crpn);
  //resize_cuda_image(outrow*crpn, outcol*crpn);
  //crop(sigout, crpd, outrow, outcol);
  //plt.init(outrow*crpn, outcol*crpn);
  ////plt.plotFloat(crpd, REAL, 0, 1./maxr, "diff", 0, 0, 1);
  //getMod2(crpd, crpd);
  //Real mse = findSum(crpd);
  //Real psnr = 10*log10(outrow*outcol*maxr*maxr*crpn*crpn/mse);
  //printf("psnr=%f\n",psnr);
  return 0;
}

