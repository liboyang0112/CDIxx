#include "cudaConfig.hpp"
#include "imgio.hpp"
#include "cub_wrap.hpp"
#include <math.h>
complexFormat findMiddle(complexFormat* d_in, int num){
  if(num==0) num = memMngr.getSize(d_in)/sizeof(complexFormat);
  myCuDMalloc(Real, tmp, num);
  getMod(tmp,d_in);
  Real norm = findSum(tmp, num);
  multiplyx(d_in,tmp);
  complexFormat mid;
  mid = findSum(tmp, num)/norm;
  multiplyy(d_in,tmp);
  mid += findSum(tmp, num)/norm*1.0i;
  memMngr.returnCache(tmp);
  return mid;
};
complexFormat findMiddle(Real* d_in, int num){
  if(num==0) num = memMngr.getSize(d_in);
  myCuDMalloc(Real, tmp, num);
  Real norm = findSum(d_in, num);
  multiplyx(d_in,tmp);
  complexFormat mid;
  mid = findSum(tmp, num)/norm;
  multiplyy(d_in,tmp);
  mid += findSum(tmp, num)/norm*1.0iF;
  memMngr.returnCache(tmp);
  return mid;
};
void readComplexWaveFront(const char* intensityFile, const char* phaseFile, Real* &d_intensity, Real* &d_phase, int &objrow, int &objcol){
  size_t sz = 0;
  if(intensityFile) {
    Real* intensity = readImage(intensityFile, objrow, objcol);
    sz = objrow*objcol*sizeof(Real);
    if(!d_intensity) d_intensity = (Real*)memMngr.borrowCache(sz); //use the memory allocated;
    myMemcpyH2D(d_intensity, intensity, sz);
    ccmemMngr.returnCache(intensity);
  }
  if(phaseFile) {
    int tmprow,tmpcol;
    Real* phase = readImage(phaseFile, tmprow,tmpcol);
    if(!intensityFile) {
      sz = tmprow*tmpcol*sizeof(Real);
      objrow = tmprow;
      objcol = tmpcol;
    }
    if(!d_phase) d_phase = (Real*)memMngr.borrowCache(sz);
    size_t tmpsz = tmprow*tmpcol*sizeof(Real);
    if(tmpsz!=sz){
      Real* d_phasetmp = (Real*)memMngr.borrowCache(tmpsz);
      myMemcpyH2D(d_phasetmp,phase,tmpsz);
      resize_cuda_image(objrow, objcol);
      if(tmpsz > sz){
        crop(d_phasetmp, d_phase, tmprow, tmpcol);
      }else{
        pad(d_phasetmp, d_phase, tmprow, tmpcol);
      }
      memMngr.returnCache(d_phasetmp);
    }
    else {
      myMemcpyH2D(d_phase,phase,sz);
    }

    ccmemMngr.returnCache(phase);
  }
}
void applyGaussConv(Real* input, Real* output, Real* gaussMem, Real sigma, int size){
  if(size == 0) size = int(floor(sigma*6))>>1; // r=3 sigma to ensure the contribution outside kernel is negligible (0.01 of the maximum)
  int width = (size<<1)+1;
  createGauss(gaussMem, width, sigma);
  int tilesize = (sq(width+15)+sq(width));
  applyConvolution(tilesize*sizeof(Real), input, output, gaussMem, size, size);
}

void shiftMiddle(complexFormat* wave){
  cudaConvertFO(wave);
  myFFT(wave, wave);
  rotateToReal(wave);
  applyNorm(wave, 1./(getCudaRows()*getCudaCols()));
  myIFFT(wave, wave);
  cudaConvertFO(wave);
}

void shiftWave(complexFormat* wave, Real shiftx, Real shifty){
  myFFT(wave, wave);
  cudaConvertFO(wave);
  multiplyShift(wave, shiftx, shifty);
  cudaConvertFO(wave);
  applyNorm(wave, 1./(getCudaRows()*getCudaCols()));
  myIFFT(wave, wave);
}

uint32_t* createMaskMap(Real* refMask, int &pixCount, int row, int col, int mrow, int mcol, int shiftx, int shifty){
  //==create reference mask and it's map: d_maskMap, allocate refs==
  pixCount = 0;
  for(int idx = 0; idx < mrow*mcol ; idx++){
    if(refMask[idx] > 0.5) pixCount++;
  }
  uint32_t* maskMap = (uint32_t*)ccmemMngr.borrowCache(pixCount*sizeof(uint32_t));
  int idx = 0, ic = 0;
  for(int x = 0; x < mrow ; x++){
    for(int y = 0; y < mcol ; y++){
      if(refMask[idx] > 0.5 && x+shiftx < row && y+shifty < col) {
        maskMap[ic] = (x+shiftx)*col + y+ shifty; //put mask in the middle
        ic++;
      }
      idx++;
    }
  }
  return maskMap;
}
