#include "cudaDefs.h"
#include "cudaConfig.h"
#include <cub/device/device_reduce.cuh>
#include <iostream>
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

using namespace std;

struct Mod2Max
{
  __device__ __forceinline__
    cuComplex operator()(const cuComplex &a, const cuComplex &b) const {
      Real mod2a = a.x*a.x+a.y*a.y;
      Real mod2b = b.x*b.x+b.y*b.y;
      return (mod2a > mod2b) ? a : b;
    }
};

static Mod2Max mod2max_op;

struct CustomSum
{
    template <typename T>
    __device__ __forceinline__
    T operator()(const T &a, const T &b) const {
        return a+b;
    }
};
static CustomSum sum_op;

struct CustomSumReal
{
  __device__ __forceinline__
    cuComplex operator()(const cuComplex &a, const cuComplex &b) const {
      return {a.x+b.x,0};
    }
};
CustomSumReal sumreal_op;

#define operatorStruct(name, type, expression...)\
struct Struct##name\
{\
  __device__ __forceinline__\
    type operator()(const type &a, const type &b) const {\
      expression\
    }\
};\
Struct##name name;
operatorStruct(max_op, Real, return (b > a) ? b : a;);
operatorStruct(max_op_int, int, return (b > a) ? b : a;);
operatorStruct(min_op_int, int, return (b < a) ? b : a;);

#define store(name) \
static void   *store_##name = NULL;\
static size_t store_##name##_n = 0;
store(findMax);
store(findMax_int);
store(findMin_int);
store(findSum);
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
  initStore(findMod2Max);
  initStore(findSumReal);
}
Real findMax(Real* d_in, int num)
{
  FUNC(Real, max_op, 0, store_findMax);
  return output;
}

int findMax(int* d_in, int num)
{
  FUNC(int, max_op_int, 0, store_findMax_int);
  return output;
}
int findMin(int* d_in, int num)
{
  FUNC(int, min_op_int, 0, store_findMin_int);
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
  cuComplex tmp;
  tmp.x = 0;
  FUNC(cuComplex, sumreal_op, tmp, store_findSumReal);
  return output.x;
}

Real findSum(Real* d_in, int num, bool debug=false)
{
  FUNC(Real, sum_op, Real(0), store_findSum);
  return output;
}

cuFuncc(multiplyx,(complexFormat* object, Real* out),(cuComplex* object, Real* out),((cuComplex*)object,out),{
  cuda1Idx();
  int x = index/cuda_column;
  out[index] = cuCabsf(object[index]) * ((x+0.5)/cuda_row-0.5);
})

cuFuncc(multiplyy,(complexFormat* object, Real* out),(cuComplex* object, Real* out),((cuComplex*)object,out),{
  cuda1Idx();
  int y = index%cuda_column;
  out[index] = cuCabsf(object[index]) * ((y+0.5)/cuda_column-0.5);
})
cuFunc(multiplyx,(Real* object, Real* out),(object,out),{
  cuda1Idx();
  int x = index/cuda_column;
  out[index] = object[index] * ((x+0.5)/cuda_row-0.5);
})

cuFunc(multiplyy,(Real* object, Real* out),(object,out),{
  cuda1Idx();
  int y = index%cuda_column;
  out[index] = object[index] * ((y+0.5)/cuda_column-0.5);
})
complexFormat findMiddle(complexFormat* d_in, int num){
  if(num==0) num = memMngr.getSize(d_in)/sizeof(complexFormat);
  myCuDMalloc(Real, tmp, num);
  getMod(tmp,d_in);
  Real norm = findSum(tmp, num);
  multiplyx(d_in,tmp);
  complexFormat mid;
  mid = findSum(tmp, num)/norm;
  multiplyy(d_in,tmp);
  mid += findSum(tmp, num)/norm*1.0iF;
  memMngr.returnCache(tmp);
  return mid;
};
complexFormat findMiddle(Real* d_in, int num){
  int num_items = memMngr.getSize(d_in);
  Real* tmp = (Real*) memMngr.borrowCache(num_items);
  num_items/=sizeof(Real);
  Real norm = findSum(d_in, num_items);
  multiplyx(d_in,tmp);
  complexFormat mid;
  mid = findSum(tmp, num_items)/norm;
  multiplyy(d_in,tmp);
  mid += findSum(tmp, num_items)/norm*1.0iF;
  memMngr.returnCache(tmp);
  return mid;
};
