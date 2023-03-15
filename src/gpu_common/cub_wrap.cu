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

struct CustomMax
{
  __device__ __forceinline__
    Real operator()(const Real &a, const Real &b) const {
      return (b > a) ? b : a;
    }
};
CustomMax max_op;

static void   *store_findMax = NULL;
static size_t store_findMax_n = 0;
static void   *store_findSum = NULL;
static size_t store_findSum_n = 0;
static void   *store_findMod2Max = NULL;
static size_t store_findMod2Max_n = 0;
static void   *store_findSumReal = NULL;
static size_t store_findSumReal_n = 0;

void initCub(){
  if(store_findMax_n) {
    store_findMax_n = 0;
    memMngr.returnCache(store_findMax);
  }
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
  FUNC(Real, sum_op, 0, store_findSum);
  return output;
}

cuFunc(multiplyx,(cudaVars* vars, complexFormat* object, Real* out),(vars,object,out),{
  cudaIdx();
  out[index] = cuCabsf(object[index]) * (Real(x)/cuda_row-0.5);
})

cuFunc(multiplyy,(cudaVars* vars, complexFormat* object, Real* out),(vars,object,out),{
  cudaIdx();
  out[index] = cuCabsf(object[index]) * (Real(y)/cuda_column-0.5);
})
cuFunc(multiplyx,(cudaVars* vars, Real* object, Real* out),(vars,object,out),{
  cudaIdx();
  out[index] = object[index] * (Real(x)/cuda_row-0.5);
})

cuFunc(multiplyy,(cudaVars* vars, Real* object, Real* out),(vars,object,out),{
  cudaIdx();
  out[index] = object[index] * (Real(y)/cuda_column-0.5);
})
complexFormat findMiddle(complexFormat* d_in, int num){
  int num_items = memMngr.getSize(d_in)/sizeof(complexFormat);
  Real* tmp = (Real*) memMngr.borrowCache(num_items*sizeof(Real));
  cudaF(getMod,tmp,d_in);
  Real norm = findSum(tmp, num_items);
  cudaF(multiplyx,d_in,tmp);
  complexFormat mid;
  mid.x = findSum(tmp, num_items)/norm;
  cudaF(multiplyy,d_in,tmp);
  mid.y = findSum(tmp, num_items)/norm;
  memMngr.returnCache(tmp);
  return mid;
};
complexFormat findMiddle(Real* d_in, int num){
  int num_items = memMngr.getSize(d_in);
  Real* tmp = (Real*) memMngr.borrowCache(num_items);
  num_items/=sizeof(Real);
  Real norm = findSum(tmp, num_items);
  cudaF(multiplyx,d_in,tmp);
  complexFormat mid;
  mid.x = findSum(tmp, num_items)/norm;
  cudaF(multiplyy,d_in,tmp);
  mid.y = findSum(tmp, num_items)/norm;
  memMngr.returnCache(tmp);
  return mid;
};
