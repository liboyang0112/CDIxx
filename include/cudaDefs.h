#ifndef __CUDADEFS_H__
#define __CUDADEFS_H__
#include "format.h"
#include <cufft.h>
#include "memManager.h"
#define cudaF(a) a<<<numBlocks,threadsPerBlock>>>
#define cudaIdx() \
int x = blockIdx.x * blockDim.x + threadIdx.x;\
int y = blockIdx.y * blockDim.y + threadIdx.y;\
if(x >= cuda_row || y >= cuda_column) return;\
int index = x*cuda_column + y;
using complexFormat=cufftComplex;
extern const dim3 threadsPerBlock;
extern dim3 numBlocks;
extern __device__ __constant__ Real cuda_beta_HIO;
extern __device__ __constant__ int cuda_row;
extern __device__ __constant__ int cuda_column;
extern __device__ __constant__ int cuda_rcolor;
extern __device__ __constant__ Real cuda_scale;
extern __device__ __constant__ int cuda_totalIntensity;
extern __device__ __constant__ Real cuda_threshold;
extern complexFormat *cudaData;
extern cufftHandle *plan, *planR2C;

class cuMemManager : public memManager{
  void c_malloc(void*& ptr, size_t sz);
  public:
    cuMemManager():memManager(){};
};
extern cuMemManager memMngr;
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      abort();
   }
}

#endif