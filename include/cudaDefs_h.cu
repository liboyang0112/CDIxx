#ifndef __CUDADEFS_H__
#define __CUDADEFS_H__
#include "format.hpp"
#define cuComplex float2
#define addVar(args...) cuda_imgsz.x, cuda_imgsz.y, args
#define addVarArg(x...) int cuda_row, int cuda_column, x
#define addVar3D(args...) cuda_imgsz.x, cuda_imgsz.y, cuda_imgsz.z, args
#define addVarArg3D(x...) int cuda_row, int cuda_column, int cuda_height, x
#define cuFunc(name,args,param,content...)\
  __global__ void name##Wrap(addVarArg args) content \
  void name args{\
    name##Wrap<<<numBlocks, threadsPerBlock>>>(addVar param);\
  }
#define cuFunc3D(name,args,param,content...)\
  __global__ void name##Wrap(addVarArg3D args) content \
  void name args{\
    name##Wrap<<<numBlocks, threadsPerBlock>>>(addVar3D param);\
  }
#define cuFuncTemplate(name,args,param,content...)\
  template<typename T>\
  __global__ void name##Wrap(addVarArg args) content \
  template<typename T>\
  void name args{\
    name##Wrap<T><<<numBlocks, threadsPerBlock>>>(addVar param);\
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
#define cuda1Idx3D() \
  int index = blockIdx.x * blockDim.x + threadIdx.x;\
  if(index >= cuda_row*cuda_column*cuda_height) return;
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
// ---- Generic elementwise macros ----
#define _g2d() dim3((cuda_imgsz.x*cuda_imgsz.y+255)/256)
#define EL1(o, x, ...) k_elem1<<<_g2d(),dim3(256)>>>(cuda_imgsz, o, x, __VA_ARGS__)
#define EL2(o, x, y, ...) k_elem2<<<_g2d(),dim3(256)>>>(cuda_imgsz, o, x, y, __VA_ARGS__)
#define EL3(o, x, y, z, ...) k_elem3<<<_g2d(),dim3(256)>>>(cuda_imgsz, o, x, y, z, __VA_ARGS__)
// Expression-only lambdas — input type determines macro, return type is deduced
#define E1r(o, inp, ...)       EL1(o, inp, [=]__device__(Real x){ return (__VA_ARGS__); })
#define E1c(o, inp, ...)       EL1(o, inp, [=]__device__(cuComplex x){ return (__VA_ARGS__); })
#define E2r(o, inp, inp2, ...) EL2(o, inp, inp2, [=]__device__(Real x, Real y){ return (__VA_ARGS__); })
#define E2c(o, inp, inp2, ...) EL2(o, inp, inp2, [=]__device__(cuComplex x, cuComplex y){ return (__VA_ARGS__); })

// Real* in-place: data[i] = expr(in, a, b, c)
#define OP1(name, expr) \
  void name(Real* data, Real a, Real b, Real c) { \
    EL1(data, data, \
       []__device__(Real x)->Real{ return (expr); }); \
  }

// Real* out-of-place: out[i] = expr(in, a, b, c)
#define OP1o(name, expr) \
  void name(Real* out, Real* in, Real a, Real b, Real c) { \
    EL1(out, in, \
       []__device__(Real x)->Real{ return (expr); }); \
  }

// Real* 2-input: out[i] = expr(in, y, a, b, c)
#define OP2(name, expr) \
  void name(Real* out, Real* xa, Real* yb, Real a, Real b, Real c) { \
    EL2(out, xa, yb, \
       []__device__(Real x, Real y)->Real{ return (expr); }); \
  }

// cuComplex in-place: data[i] = expr(in, a, b, c)
#define COP1(name, expr) \
  void name(complexFormat* data, Real a, Real b, Real c) { \
    EL1((cuComplex*)data, (cuComplex*)data, \
       []__device__(cuComplex x)->cuComplex{ return (expr); }); \
  }

// cuComplex in-place with complex scalar: data[i] = expr(in, z, b, c)
#define COP1z(name, expr) \
  void name(complexFormat* data, complexFormat z, Real b, Real c) { \
    EL1((cuComplex*)data, (cuComplex*)data, \
       []__device__(cuComplex x)->cuComplex{ return (expr); }); \
  }

// complexFormat* in -> Real* out: out[i] = expr(in, a, b, c)
#define C2R1(name, expr) \
  void name(Real* out, complexFormat* in, Real a, Real b, Real c) { \
    EL1(out, (const cuComplex*)in, \
       []__device__(cuComplex x)->Real{ return (expr); }); \
  }

// Real* in -> complexFormat* out: out[i] = expr(in, a, b, c)
#define R2C1(name, expr) \
  void name(Real* in, complexFormat* out, Real a, Real b, Real c) { \
    EL1((cuComplex*)out, in, \
       []__device__(Real x)->cuComplex{ return (expr); }); \
  }

// cuComplex 2-input: out[i] = expr(in, y, a, b, c)
#define COP2(name, expr) \
  void name(complexFormat* out, complexFormat* xa, complexFormat* yb, Real a, Real b, Real c) { \
    EL2((cuComplex*)out, (const cuComplex*)xa, (const cuComplex*)yb, \
       []__device__(cuComplex x, cuComplex y)->cuComplex{ return (expr); }); \
  }

// cuComplex 2-input, in-place on first: xa[i] = expr(in, y, a, b, c)
#define COP2i(name, expr) \
  void name(complexFormat* xa, complexFormat* yb, Real a, Real b, Real c) { \
    EL2((cuComplex*)xa, (const cuComplex*)xa, (const cuComplex*)yb, \
       []__device__(cuComplex x, cuComplex y)->cuComplex{ return (expr); }); \
  }

// Real 2-input, in-place on first: xa[i] = expr(in, y, a, b, c)
#define OP2i(name, expr) \
  void name(Real* xa, Real* yb, Real a, Real b, Real c) { \
    EL2(xa, xa, yb, \
       []__device__(Real x, Real y)->Real{ return (expr); }); \
  }

extern const dim3 threadsPerBlock;
extern dim3 numBlocks;
extern dim3 cuda_imgsz;
#endif
