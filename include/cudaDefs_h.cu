#ifndef __CUDADEFS_H__
#define __CUDADEFS_H__
#include "format.hpp"
#include <cuComplex.h>
#include <stdio.h>
#define cuComplex float2
#define addVar(args...) cudaVar, cuda_imgsz.x, cuda_imgsz.y, cuda_imgsz.z, args
#define addVarArg(x...) cudaVars* vars, int cuda_row, int cuda_column, int cuda_height, x
#define cuFunc(name,args,param,content...)\
  __global__ void name##Wrap(addVarArg args) content \
  void name args{\
    name##Wrap<<<numBlocks, threadsPerBlock>>>(addVar param);\
  }
#define cuFuncc(name,argsf,argsw,param,content...)\
  __global__ void name##Wrap(addVarArg argsw) content \
  void name argsf{\
    name##Wrap<<<numBlocks, threadsPerBlock>>>(addVar param);\
  }
#define cuFuncTemplate(name,args,param,content...)\
  template<typename T>\
  __global__ void name##Wrap(addVarArg args) content \
  template<typename T>\
  void name args{\
    name##Wrap<T><<<numBlocks, threadsPerBlock>>>(addVar param);\
  }
#define cuFunccTemplate(name,argsf,argsw,param,content...)\
  template<typename T>\
  __global__ void name##Wrap(addVarArg argsw) content \
  template<typename T>\
  void name argsf{\
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
#define cuda1Idx() \
  int index = blockIdx.x * blockDim.x + threadIdx.x;\
  if(index >= cuda_row*cuda_column) return;
#define cudaIdx() \
  int index = blockIdx.x * blockDim.x + threadIdx.x;\
  if(index >= cuda_row*cuda_column) return;\
  int x = index/cuda_column;\
  int y = index%cuda_column;
#define cuda3Idx() \
  int index = blockIdx.x * blockDim.x + threadIdx.x;\
  if(index >= cuda_row*cuda_column*cuda_height) return;\
  int z = index/(cuda_row*cuda_column);\
  int idxxy = index-z*(cuda_row*cuda_column);\
  int x = idxxy%cuda_row;\
  int y = idxxy/cuda_row;
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
extern int3 cuda_imgsz;
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
