#ifndef __CUDADEFS_H__
#define __CUDADEFS_H__
#include <cufft.h>
#include "format.h"
#include "memManager.h"
#define addVar(args...) cudaVar, cuda_imgsz.x, cuda_imgsz.y, args
#define addVarArg(x...) cudaVars* vars, int cuda_row, int cuda_column, x
#define cuFunc(name,args,param,content...)\
  __global__ void name##Wrap(addVarArg args) content \
  void name args{\
    name##Wrap<<<numBlocks, threadsPerBlock>>>(addVar param);\
  }
#define cuFuncTemplate(name,args,param,content...)\
  template<typename T>\
  __global__ void name##Wrap(addVarArg args) content \
  template<typename T>\
  void name args{\
    name##Wrap<<<numBlocks, threadsPerBlock>>>(addVar param);\
  }
#define addSize(args...) size_t size, args
#define cuFuncSharedDec(funcname, ...) void funcname (size_t size, __VA_ARGS__)
#define cuFuncShared(name,args,param,content...)\
  __global__ void name##Wrap(addVarArg args) content \
  void name( addSize args){\
    dim3 nthd = {16,16};\
    dim3 nblk = {\
      (cuda_imgsz.x-1)/nthd.x+1,\
      (cuda_imgsz.y-1)/nthd.y+1\
    };\
    name##Wrap<<<nblk, nthd, size>>>(addVar param);\
  }
#define cudaoIdx() \
  int x = blockIdx.x * blockDim.x + threadIdx.x;\
  int y = blockIdx.y * blockDim.y + threadIdx.y;\
  if(x >= cuda_row || y >= cuda_column) return;\
  int index = x*cuda_column + y;
#define cuda1Idx() \
  int index = blockIdx.x * blockDim.x + threadIdx.x;\
  if(index >= cuda_row*cuda_column) return;
#define cudaIdx() \
  int index = blockIdx.x * blockDim.x + threadIdx.x;\
  if(index >= cuda_row*cuda_column) return;\
  int x = index/cuda_column;\
  int y = index%cuda_column;
extern const dim3 threadsPerBlock;
extern dim3 numBlocks;
struct cudaVars{
  int rcolor;
  Real beta_HIO;
  Real scale;
  Real threshold;
};
extern cudaVars* cudaVar;
extern cudaVars* cudaVarLocal;
extern complexFormat *cudaData;
extern cufftHandle *plan, *planR2C;
extern int2 cuda_imgsz;
void resize_cuda_image(int row, int col);
void init_cuda_image(int rcolor=0, Real scale=NAN);

class cuMemManager : public memManager{
  void c_malloc(void*& ptr, size_t sz);
  void c_memset(void*& ptr, size_t sz);
  public:
  cuMemManager():memManager(){
    //cudaFree(0); // to speed up the cuda malloc; https://forums.developer.nvidia.com/t/cudamalloc-slow/40238
  };
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
