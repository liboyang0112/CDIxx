//
// CUDA implementation of Total Variation Filter
// Implementation of Nonlinear total variation based noise removal algorithms : 10.1016/0167-2789(92)90242-F
//
#include <iostream>
#include <string>
#include <stdio.h>
#include <cuda.h>
#include "cudaConfig.h"
#include "cudaDefs.h"
#include "imgio.h"

#define BLOCK_SIZE      16
#define FILTER_WIDTH    1      //total width=2n+1 
#define FILTER_HEIGHT   1       

#include <cub/device/device_reduce.cuh>
using namespace std;

// Run Total Variation Filter on GPU

Real *d_output, *d_bracket, *d_lambdacore, *lambda;
const short tilewidth=BLOCK_SIZE+2*FILTER_HEIGHT;
size_t  temp_storage_bytes = 0;
void  *d_temp_storage = NULL;
static int rows, cols;
size_t sz;

struct CustomSum
{
    template <typename T>
    __device__ __forceinline__
    T operator()(const T &a, const T &b) const {
        return a+b;
    }
};
static CustomSum sum_op;

template <typename T>
__device__ T minmod(T data1, T data2){
  if(data1*data2<=0) return T(0);  
  if(data1<0) return max(data1,data2);
  return min(data1,data2);
}


template <typename T>
__global__ void calcBracketLambda(cudaVars* vars,int cuda_row, int cuda_column, T *srcImage, T *bracket, T* u0, T* lambdacore, T noiseLevel)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  extern __shared__ float tile[];
  if(threadIdx.x<blockDim.x/2+FILTER_WIDTH && threadIdx.y<blockDim.y/2+FILTER_HEIGHT)
    tile[threadIdx.x*(tilewidth)+threadIdx.y]=(x>=FILTER_WIDTH && y>=FILTER_HEIGHT)?srcImage[(x-FILTER_WIDTH)*cuda_column+y-FILTER_HEIGHT]:0;
  if(threadIdx.x>=blockDim.x/2-FILTER_WIDTH && threadIdx.y<blockDim.y/2+FILTER_HEIGHT)
    tile[(threadIdx.x+2*FILTER_WIDTH)*(tilewidth)+threadIdx.y]=(x<cuda_row-FILTER_WIDTH && y>=FILTER_HEIGHT)?srcImage[(x+FILTER_WIDTH)*cuda_column+y-FILTER_HEIGHT]:0;
  if(threadIdx.x<blockDim.x/2+FILTER_WIDTH && threadIdx.y>=blockDim.y/2-FILTER_HEIGHT)
    tile[threadIdx.x*(tilewidth)+threadIdx.y+2*FILTER_HEIGHT]=(x>=FILTER_WIDTH && y<cuda_column-FILTER_HEIGHT)?srcImage[(x-FILTER_WIDTH)*cuda_column+y+FILTER_HEIGHT]:0;
  if(threadIdx.x>=blockDim.x/2-FILTER_WIDTH && threadIdx.y>=blockDim.y/2-FILTER_HEIGHT)
    tile[(threadIdx.x+2*FILTER_WIDTH)*(tilewidth)+threadIdx.y+2*FILTER_HEIGHT]=(x<cuda_row-FILTER_WIDTH && y<cuda_column-FILTER_HEIGHT)?srcImage[(x+FILTER_WIDTH)*cuda_column+y+FILTER_HEIGHT]:0;
  __syncthreads();
  if(x >= cuda_row || y >= cuda_column) return;
  int index = x*cuda_column+y;
  float sigma = sqrt(noiseLevel);
  float dt = 5e-8*sigma;
  float sigmafactor = vars->rcolor*5e-8*vars->rcolor/(cuda_row*cuda_column*2)/sigma;
  int centerIdx = (threadIdx.x+FILTER_WIDTH)*(tilewidth) + threadIdx.y+FILTER_HEIGHT;
  float dpxU = tile[centerIdx+tilewidth]-tile[centerIdx];
  float dpyU = tile[centerIdx+1]-tile[centerIdx];
  float dmxU = tile[centerIdx]-tile[centerIdx-tilewidth];
  float dmyU = tile[centerIdx]-tile[centerIdx-1];
  float sbracket = 0;
  float denom = sqrt(sq(dpxU)+sq(dpyU));
  if(denom && x<cuda_row-1 && y<cuda_column-1) lambdacore[index] = ((u0[index+cuda_column]*dpxU+u0[index+1]*dpyU-u0[index]*(dpxU+dpyU))/denom-denom)*sigmafactor;
  else lambdacore[index] = 0;
  denom = sqrt(sq(dpxU)+sq(minmod(dpyU,dmyU)));
  if(denom!=0) sbracket += dpxU/denom;
  denom = sqrt(sq(dpyU)+sq(minmod(dpxU,dmxU)));
  if(denom!=0) sbracket += dpyU/denom;
  centerIdx-=1;
  dpxU = tile[centerIdx+tilewidth]-tile[centerIdx];
  dpyU = tile[centerIdx+1]-tile[centerIdx];
  dmxU = tile[centerIdx]-tile[centerIdx-tilewidth];
  denom = sqrt(sq(dpyU)+sq(minmod(dpxU,dmxU)));
  if(denom != 0) sbracket -= dpyU/denom;
  centerIdx-=tilewidth-1;
  dpxU = tile[centerIdx+tilewidth]-tile[centerIdx];
  dpyU = tile[centerIdx+1]-tile[centerIdx];
  dmyU = tile[centerIdx]-tile[centerIdx-1];
  denom = sqrt(sq(dpxU)+sq(minmod(dpyU,dmyU)));
  if(denom != 0) sbracket -= dpxU/denom;
  bracket[index] = dt*sbracket;
}

template <typename T>
__global__ void tvFilter(int cuda_row, int cuda_column, T *srcImage, T *bracket, T* u0, T* slambda)
{
  cuda1Idx()
  srcImage[index]+=bracket[index]-(*slambda)*(srcImage[index]-u0[index]);
}

void inittvFilter(int row, int col){
  rows = row;
  cols = col;
  sz = rows * cols * sizeof(Real);
  // Allocate device memory
  d_output = (Real*)memMngr.borrowCache(sz);
  d_bracket = (Real*)memMngr.borrowCache(sz);
  d_lambdacore = (Real*)memMngr.borrowCache(sz);
  lambda = (Real*)memMngr.borrowCache(sizeof(Real));
  gpuErrchk(cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_lambdacore, lambda, rows*cols, sum_op, Real(0)));
  d_temp_storage = memMngr.borrowCache(temp_storage_bytes);
}
Real* tvFilterWrap(Real* d_input, Real noiseLevel, int nIters){
  gpuErrchk(cudaMemcpy(d_output,d_input,sz,cudaMemcpyDeviceToDevice));
	for(int i = 0; i<nIters; i++){
    calcBracketLambda<<<numBlocks,threadsPerBlock,sizeof(float)*(tilewidth)*(tilewidth)>>>(cudaVar, cuda_imgsz.x, cuda_imgsz.y, d_output, d_bracket, d_input, d_lambdacore, noiseLevel);
    gpuErrchk(cub::DeviceReduce::Reduce(d_temp_storage, sz, d_lambdacore, lambda, rows*cols, sum_op, Real(0)));
    tvFilter<<<numBlocks,threadsPerBlock>>>(cuda_imgsz.x, cuda_imgsz.y, d_output, d_bracket, d_input, lambda);
	}
  cudaMemcpy(d_input,d_output,sz,cudaMemcpyDeviceToDevice);
	return d_input;
}
void tvFilter(Real* input, Real noiseLevel, int nIters)
{
	Real* d_input;
	d_input = (Real*)memMngr.borrowCache(sz);
  cudaMemcpy(d_input,input,sz,cudaMemcpyHostToDevice);
	tvFilterWrap(d_input, noiseLevel, nIters);
  cudaMemcpy(input,d_input,sz,cudaMemcpyDeviceToHost);
  memMngr.returnCache(d_input);
}
