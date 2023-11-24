extern "C"{
#include "gui/dataviewerWinWrapper.h"
}
#include <stdio.h>
#include "cuPlotter.h"
#include "cudaConfig.h"
#include "imageFile.h"
void* to_gpu(void* ptr, struct imageFile *f){
  size_t sz = f->rows*f->cols*typeSizes[f->type];
  init_cuda_image();
  resize_cuda_image(f->rows, f->cols);
  plt.init(f->rows, f->cols);
  myCuDMalloc(char, dptr, sz);
  myMemcpyH2D(dptr, ptr, sz);
  return dptr;
};
void processFloat(void* cache, void* cudaData, char m, char isFrequency, Real decay, char islog, char isFlip){
  plt.processFloatData(cudaData, (enum mode)m, isFrequency, decay, islog, isFlip);
  plt.cvtTurbo(cache);
};
void processComplex(void* cache, void* cudaData, char m, char isFrequency, Real decay, char islog, char isFlip){
  plt.processComplexData(cudaData, (enum mode)m, isFrequency, decay, islog, isFlip);
  plt.cvtTurbo(cache);
};
