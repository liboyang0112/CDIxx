#include "cudaConfig.h"
#include <cub/device/device_reduce.cuh>
#include <iostream>
#define FUNC(T,OP,INIT,STORE)\
T *d_out = (T*)memMngr.borrowCache(sizeof(T));\
size_t num_items = num;\
if(num_items == 0) num_items = memMngr.getSize(d_in)/sizeof(T);\
if(!STORE##_n){\
  gpuErrchk(cub::DeviceReduce::Reduce(STORE, STORE##_n, d_in, d_out, num_items, OP, INIT));\
  STORE = memMngr.borrowCache(STORE##_n);\
}\
gpuErrchk(cub::DeviceReduce::Reduce(STORE, STORE##_n, d_in, d_out, num_items, OP, INIT));\
T output;\
cudaMemcpy(&output, d_out, sizeof(T), cudaMemcpyDeviceToHost);\
if (d_out) memMngr.returnCache(d_out);

using namespace std;

struct Mod2Max
{
  __device__ __forceinline__
    complexFormat operator()(const complexFormat &a, const complexFormat &b) const {
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
    complexFormat operator()(const complexFormat &a, const complexFormat &b) const {
      return {a.x+b.x,0};
    }
};
CustomSumReal sumreal_op;

#define operatorStruct(name, type, expression...)\
struct Struct##name\
{\
  __device__ __forceinline__\
    Real operator()(const type &a, const type &b) const {\
      expression\
    }\
};\
Struct##name name;
operatorStruct(max_op, Real, return (b > a) ? b : a;);

#define store(name) \
static void   *store_##name = NULL;\
static size_t store_##name##_n = 0;
store(findMax);
store(findSum);
store(findMod2Max);
store(findSumReal);

#define initStore(name)\
  if(store_##name##_n) {\
    store_##name##_n = 0;\
    memMngr.returnCache(store_##name);\
  }
void initCub(){
  initStore(findMax);
  initStore(findSum);
  initStore(findMod2Max);
  initStore(findSumReal);
}
Real findMax(Real* d_in, int num)
{
  FUNC(Real, max_op, 0, store_findMax);
  return output;
}

Real findMod2Max(complexFormat* d_in, int num)
{
  complexFormat tmp;
  tmp.x = tmp.y = 0;
  FUNC(complexFormat, mod2max_op, tmp, store_findMod2Max);
  return output.x*output.x + output.y*output.y;
}

Real findSumReal(complexFormat* d_in, int num)
{
  complexFormat tmp;
  tmp.x = 0;
  FUNC(complexFormat, sumreal_op, tmp, store_findSumReal);
  return output.x;
}

Real findSum(Real* d_in, int num, bool debug=false)
{
  FUNC(Real, sum_op, Real(0), store_findSum);
  return output;
}

cuFunc(multiplyx,(complexFormat* object, Real* out),(object,out),{
  cuda1Idx();
  int x = index/cuda_column;
  out[index] = cuCabsf(object[index]) * (Real(x)/cuda_row-0.5);
})

cuFunc(multiplyy,(complexFormat* object, Real* out),(object,out),{
  cuda1Idx();
  int y = index%cuda_column;
  out[index] = cuCabsf(object[index]) * (Real(y)/cuda_column-0.5);
})
cuFunc(multiplyx,(Real* object, Real* out),(object,out),{
  cuda1Idx();
  int x = index/cuda_column;
  out[index] = object[index] * (Real(x)/cuda_row-0.5);
})

cuFunc(multiplyy,(Real* object, Real* out),(object,out),{
  cuda1Idx();
  int y = index%cuda_column;
  out[index] = object[index] * (Real(y)/cuda_column-0.5);
})
complexFormat findMiddle(complexFormat* d_in, int num){
  int num_items = memMngr.getSize(d_in)/sizeof(complexFormat);
  Real* tmp = (Real*) memMngr.borrowCache(num_items*sizeof(Real));
  getMod(tmp,d_in);
  Real norm = findSum(tmp, num_items);
  multiplyx(d_in,tmp);
  complexFormat mid;
  mid.x = findSum(tmp, num_items)/norm;
  multiplyy(d_in,tmp);
  mid.y = findSum(tmp, num_items)/norm;
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
  mid.x = findSum(tmp, num_items)/norm;
  multiplyy(d_in,tmp);
  mid.y = findSum(tmp, num_items)/norm;
  memMngr.returnCache(tmp);
  return mid;
};
