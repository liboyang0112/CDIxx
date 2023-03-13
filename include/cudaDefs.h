#ifndef __CUDADEFS_H__
#define __CUDADEFS_H__
#include <cufft.h>
#include "format.h"
#include "memManager.h"
#define cudaF(funcname, ...) funcname<<<numBlocks,threadsPerBlock>>>(cudaVar, __VA_ARGS__)
#define cudaIdx() \
int x = blockIdx.x * blockDim.x + threadIdx.x;\
int y = blockIdx.y * blockDim.y + threadIdx.y;\
int cuda_row = vars->rows;\
int cuda_column = vars->cols;\
if(x >= cuda_row || y >= cuda_column) return;\
int index = x*cuda_column + y;
extern const dim3 threadsPerBlock;
extern dim3 numBlocks;
struct cudaVars{
  int rows;
  int cols;
  int rcolor;
  Real beta_HIO;
  Real scale;
  Real threshold;
};
extern cudaVars* cudaVar;
extern cudaVars* cudaVarLocal;
extern complexFormat *cudaData;
extern cufftHandle *plan, *planR2C;
extern int cuda_row_, cuda_column_;
void init_cuda_image(int rows, int cols, int rcolor=0, Real scale=NAN);

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
