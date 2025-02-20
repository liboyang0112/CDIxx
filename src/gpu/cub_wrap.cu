//#include <iostream>
#include "cudaDefs_h.cu"
#include "cudaConfig.hpp"
#include <cub/device/device_reduce.cuh>

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
}

#define FUNC(T,OP,INIT,STORE)\
T *d_out = (T*)memMngr.borrowCache(sizeof(T));\
size_t num_items = num;\
if(num_items == 0) num_items = memMngr.getSize(d_in)/sizeof(T);\
if(!STORE##_n){\
  gpuErrchk(cub::DeviceReduce::Reduce(STORE, STORE##_n, (T*)d_in, d_out, num_items, OP, INIT));\
  STORE = memMngr.borrowCache(STORE##_n);\
}\
gpuErrchk(cub::DeviceReduce::Reduce(STORE, STORE##_n, (T*)d_in, d_out, num_items, OP, INIT));\
T output;\
cudaMemcpy(&output, d_out, sizeof(T), cudaMemcpyDeviceToHost);\
if (d_out) memMngr.returnCache(d_out);

Real findMax(Real* d_in, int num)
{
  FUNC(Real, max_op, Real(0), store_findMax);
  return output;
}

int findMax(int* d_in, int num)
{
  FUNC(int, max_op, 0, store_findMax_int);
  return output;
}
int findMin(int* d_in, int num)
{
  FUNC(int, min_op, 0, store_findMin_int);
  return output;
}


Real findMod2Max(complexFormat* d_in, int num)
{
  cuComplex tmp;
  tmp.x = tmp.y = 0;
  FUNC(cuComplex, mod2max_op, tmp, store_findMod2Max);
  return output.x*output.x + output.y*output.y;
}

Real findSumReal(complexFormat* d_in, int num)
{
  FUNC(cuComplex, sumcomplex_op, cuComplex(), store_findSumReal);
  return output.x;
}

complexFormat findSum(complexFormat* d_in, int num)
{
  cuComplex tmp;
  tmp.x = tmp.y = 0;
  FUNC(cuComplex, sumcomplex_op, tmp, store_findSumComplex);
  //return (*(complexFormat*)&output);
  return {output.x, output.y};
}

Real findSum(Real* d_in, int num)
{
  FUNC(Real, sum_op, Real(0), store_findSum);
  return output;
}

Real findRootSumSq(Real* d_in, int num)
{
  FUNC(Real, rootsumsq_op, Real(0), store_findRootSumSq);
  return output;
}


