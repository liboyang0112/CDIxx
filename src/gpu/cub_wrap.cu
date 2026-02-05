//#include <iostream>
#include "cudaDefs_h.cu"
#include "cudaConfig.hpp"
#include "cuComplex.h"
#include <cub/cub.cuh>

#define operatorStructT(name, expression...)\
struct Struct##name\
{\
  template <typename T>\
  __device__ __forceinline__\
    T operator()(const T &a, const T &b) const {\
      expression\
    }\
};\
Struct##name name;

#define operatorStruct(name, type, expression...)\
struct Struct##name\
{\
  __device__ __forceinline__\
    type operator()(const type &a, const type &b) const {\
      expression\
    }\
};\
Struct##name name;
operatorStructT(sum_op, return a+b;);
operatorStructT(rootsumsq_op, return sqrtf(a*a+b*b););
operatorStructT(max_op, return (b > a) ? b : a;);
operatorStructT(min_op, return (b < a) ? b : a;);
operatorStruct(sumcomplex_op, cuComplex, return {a.x+b.x, a.y+b.y};);
operatorStruct(mod2max_op, cuComplex, 
      Real mod2a = a.x*a.x+a.y*a.y;
      Real mod2b = b.x*b.x+b.y*b.y;
      return (mod2a > mod2b) ? a : b;
      );

#define store(name) \
static void   *store_##name = NULL;\
static size_t store_##name##_n = 0;
store(findMax);
store(findMax_int);
store(findMin_int);
store(findSum);
store(findRootSumSq);
store(findSumComplex);
store(findMod2Max);
store(findSumReal);
store(findMaxIdx);

#define initStore(name)\
  if(store_##name##_n) {\
    store_##name##_n = 0;\
    memMngr.returnCache(store_##name);\
    store_##name = 0;\
  }
void initCub(){
  initStore(findMax);
  initStore(findMax_int);
  initStore(findMin_int);
  initStore(findSum);
  initStore(findRootSumSq);
  initStore(findMod2Max);
  initStore(findSumReal);
  initStore(findSumComplex);
  initStore(findMaxIdx);
}
#define FUNC(T,OP,INIT,STORE)\
bool hascache = 1;\
if(!d_out) {\
  d_out = memMngr.borrowCache(sizeof(T));\
  hascache = 0;\
}\
size_t num_items = num;\
if(num_items == 0) num_items = memMngr.getSize(d_in)/sizeof(T);\
if(!STORE##_n){\
  gpuErrchk(cub::DeviceReduce::Reduce(STORE, STORE##_n, (T*)d_in, (T*)d_out, num_items, OP, INIT));\
  STORE = memMngr.borrowCache(STORE##_n);\
}\
gpuErrchk(cub::DeviceReduce::Reduce(STORE, STORE##_n, (T*)d_in, (T*)d_out, num_items, OP, INIT));\
T output = T();\
if(!hascache){\
myMemcpyD2H(&output, d_out, sizeof(T));\
if (d_out) memMngr.returnCache(d_out);\
}


Real findMax(Real* d_in, int num, void* d_out)
{
  FUNC(Real, max_op, Real(0), store_findMax);
  return output;
}

int findMaxIdx(Real* d_in, int num, void* d_out)
{
  /*
  cub::KeyValuePair<int, Real> output;
  bool hascache = 1;
  if(!d_out) {
    d_out = memMngr.borrowCache(sizeof(cub::KeyValuePair<int, Real>));
    hascache = 0;
  }
  int num_items = num;
  if(num_items == 0) num_items = memMngr.getSize(d_in)/sizeof(Real);
  if(!store_findMaxIdx_n){
    //gpuErrchk(cub::DeviceReduce::ArgMax(store_findMaxIdx, store_findMaxIdx_n, (Real*)d_in, (cub::KeyValuePair<int, Real>*)d_out, num_items)); //new cub api is not compatible with hip, so please ignore the warning
    gpuErrchk(cub::DeviceReduce::ArgMax(store_findMaxIdx, store_findMaxIdx_n, (Real*)d_in, &((cub::KeyValuePair<int, Real>*)d_out)->value, &((cub::KeyValuePair<int, Real>*)d_out)->key, num_items)); //new cub api, not compatible with hip
    store_findMaxIdx = memMngr.borrowCache(store_findMaxIdx_n);
  }
  //gpuErrchk(cub::DeviceReduce::ArgMax(store_findMaxIdx, store_findMaxIdx_n, (Real*)d_in, (cub::KeyValuePair<int, Real>*)d_out, num_items)); //new cub api is not compatible with hip, so please ignore the warning
  gpuErrchk(cub::DeviceReduce::ArgMax(store_findMaxIdx, store_findMaxIdx_n, (Real*)d_in, &((cub::KeyValuePair<int, Real>*)d_out)->value, &((cub::KeyValuePair<int, Real>*)d_out)->key, num_items)); //new cub api, not compatible with hip
  if(!hascache){
    myMemcpyD2H(&output, d_out, sizeof(cub::KeyValuePair<int, Real>));
    memMngr.returnCache(d_out);
    return output.key;
  }
  */
  return 0;
}

int findMax(int* d_in, int num, void* d_out)
{
  FUNC(int, max_op, 0, store_findMax_int);
  return output;
}
int findMin(int* d_in, int num, void* d_out)
{
  FUNC(int, min_op, 0, store_findMin_int);
  return output;
}


Real findMod2Max(complexFormat* d_in, int num, void* d_out)
{
  cuComplex tmp;
  tmp.x = tmp.y = 0;
  FUNC(cuComplex, mod2max_op, tmp, store_findMod2Max);
  return output.x*output.x + output.y*output.y;
}

Real findSumReal(complexFormat* d_in, int num, void* d_out)
{
  FUNC(cuComplex, sumcomplex_op, cuComplex(), store_findSumReal);
  return output.x;
}

complexFormat findSum(complexFormat* d_in, int num, void* d_out)
{
  cuComplex tmp;
  tmp.x = tmp.y = 0;
  FUNC(cuComplex, sumcomplex_op, tmp, store_findSumComplex);
  //return (*(complexFormat*)&output);
  return {output.x, output.y};
}

Real findSum(Real* d_in, int num, void* d_out)
{
  FUNC(Real, sum_op, Real(0), store_findSum);
  return output;
}

Real findRootSumSq(Real* d_in, int num, void* d_out)
{
  FUNC(Real, rootsumsq_op, Real(0), store_findRootSumSq);
  return output;
}


