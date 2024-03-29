#include "format.h"
#include "cudaConfig.h"
#include <cufftw.h>
#include <iostream>
#include "fftw.h"
using namespace cv;
using namespace std;

void fftw_init(){
  
}
static size_t sz;

Mat* fftw ( Mat* in, Mat *out, bool isforward, Real ratio)
{
  int row = in->rows;
  int column = in->cols;
  resize_cuda_image(row, column);
  if(ratio==0) ratio = 1./sqrt(row*column);
  if(out == 0) out = new Mat(row,column,float_cv_format(2));

  if(cudaData==0) {
    sz = row*column*sizeof(complexFormat);
    cudaData = (complexFormat*)memMngr.borrowCache(sz);
    init_fft(row,column);
  }else{
    if(sz!=row*column*sizeof(complexFormat)){
      printf("reconfiguring CUFFT\n");
      sz = row*column*sizeof(complexFormat);
      memMngr.returnCache(cudaData);
      cudaData = (complexFormat*)memMngr.borrowCache(sz);
      init_fft(row,column);
    }
  }
  gpuErrchk(cudaMemcpy(cudaData, in->data, sz, cudaMemcpyHostToDevice));
    
  myCufftExec( *plan, cudaData,cudaData, isforward? CUFFT_FORWARD: CUFFT_INVERSE);

  applyNorm(cudaData, ratio);

  gpuErrchk(cudaMemcpy(out->data, cudaData, sz, cudaMemcpyDeviceToHost));
  
  return out;
}

