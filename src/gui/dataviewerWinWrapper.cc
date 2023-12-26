extern "C"{
#include "gui/dataviewerWinWrapper.hpp"
}
#include <stdio.h>
#include "cuPlotter.hpp"
#include "cudaConfig.hpp"
#include "imageFile.hpp"
void* to_gpu(void* ptr, struct imageFile *f){
  size_t sz = f->rows*f->cols*typeSizes[f->type];
  init_cuda_image();
  resize_cuda_image(f->rows, f->cols);
  plt.init(f->rows, f->cols);
  myCuDMalloc(char, dptr, sz);
  myMemcpyH2D(dptr, ptr, sz);
  return dptr;
};
void processFloat(void* cache, void* cudaData, char m, char isFrequency, Real decay, char islog, char isFlip, char isColor){
  plt.processFloatData(cudaData, (enum mode)m, isFrequency, decay, islog, isFlip);
  if(isColor) plt.cvtTurbo(cache);
  else plt.cvt8bit(cache);
};
void processComplex(void* cache, void* cudaData, char m, char isFrequency, Real decay, char islog, char isFlip, char isColor){
  plt.processComplexData(cudaData, (enum mode)m, isFrequency, decay, islog, isFlip);
  if(isColor) plt.cvtTurbo(cache);
  else plt.cvt8bit(cache);
};
