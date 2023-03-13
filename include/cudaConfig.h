#ifndef __CUDACONFIG_H__
#define __CUDACONFIG_H__
#include <iostream>
#include "format.h"
#include "cudaDefs.h"
#include <curand_kernel.h>
#define FFTformat CUFFT_C2C
#define FFTformatR2C CUFFT_R2C
#define myCufftExec cufftExecC2C
#define myCufftExecR2C cufftExecR2C
__global__ void forcePositive(cudaVars* vars, complexFormat* a);
__global__ void add(cudaVars* vars, Real* a, Real* b, Real c = 1);
__global__ void extendToComplex(cudaVars* vars, Real* a, complexFormat* b);
__global__ void applyNorm(cudaVars* vars, complexFormat* data, Real factor);
__global__ void applyNorm(cudaVars* vars, Real* data, Real factor);
__global__ void createWaveFront(cudaVars* vars, Real* d_intensity, Real* d_phase, complexFormat* objectWave, Real oversampling, Real shiftx = 0, Real shifty = 0);
__global__ void createWaveFront(cudaVars* vars, Real* d_intensity, Real* d_phase, complexFormat* objectWave, int row, int col, int shiftx = 0, int shifty = 0);
__global__ void applyConvolution(cudaVars* vars, Real *input, Real *output, Real* kernel, int kernelwidth, int kernelheight);
__global__ void getMod(cudaVars* vars, Real* mod, complexFormat* amp);
__global__ void getReal(cudaVars* vars, Real* mod, complexFormat* amp);
__global__ void getMod2(cudaVars* vars, Real* mod, complexFormat* amp);
__global__ void applyPoissonNoise(cudaVars* vars, Real* wave, Real noiseLevel, curandStateMRG32k3a *state, Real scale = 0);
__global__ void applyPoissonNoise_WO(cudaVars* vars, Real* wave, Real noiseLevel, curandStateMRG32k3a *state, Real scale = 0);
__global__ void initRand(cudaVars* vars, curandStateMRG32k3a *state,unsigned long long seed);
__global__ void fillRedundantR2C(cudaVars* vars, complexFormat* data, complexFormat* dataout, Real factor);
__global__ void applyMod(cudaVars* vars, complexFormat* source, Real* target, Real *bs = 0, bool loose=0, int iter = 0, int noiseLevel = 0);
__global__ void add(cudaVars* vars, complexFormat* a, complexFormat* b, Real c = 1);
__global__ void add(cudaVars* vars, complexFormat* store, complexFormat* a, complexFormat* b, Real c = 1);
__global__ void applyRandomPhase(cudaVars* vars, complexFormat* wave, Real* beamstop, curandStateMRG32k3a *state);
__global__ void multiply(cudaVars* vars, complexFormat* source, complexFormat* target);
__global__ void multiplyReal(cudaVars* vars, Real* store, complexFormat* source, complexFormat* target);
__global__ void multiply(cudaVars* vars, complexFormat* store, complexFormat* source, complexFormat* target);
void init_fft(int rows, int cols);
template <typename T>
__global__ void cudaConvertFO(cudaVars* vars, T* data){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int cuda_row = vars->rows;
  int cuda_column = vars->cols;
  if(x >= (cuda_row>>1) || y >= cuda_column) return;
  int index = x*cuda_column + y;
  int indexp = (x+(cuda_row>>1))*cuda_column + (y >= (cuda_column>>1)? y-(cuda_column>>1): (y+(cuda_column>>1)));
  T tmp = data[index];
  data[index]=data[indexp];
  data[indexp]=tmp;
}

template <typename T>
__global__ void cudaConvertFO(cudaVars* vars, T* data, T* out){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int cuda_row = vars->rows;
  int cuda_column = vars->cols;
  if(x >= (cuda_row>>1) || y >= cuda_column) return;
  int index = x*cuda_column + y;
  int indexp = (x+(cuda_row>>1))*cuda_column + (y >= (cuda_column>>1)? y-(cuda_column>>1): (y+(cuda_column>>1)));
  T tmp = data[index];
  out[index]=data[indexp];
  out[indexp]=tmp;
}

template <typename T>
__global__ void zeroEdge(cudaVars* vars, T* a, int n){
  cudaIdx()
  if(x<n || x>=cuda_row-n || y < n || y >= cuda_column-n)
    a[index] = T();
}

template <typename T1, typename T2>
__global__ void assignVal(cudaVars* vars, T1* out, T2* input){
  cudaIdx()
	out[index] = input[index];
}

template <typename T>
__global__ void crop(cudaVars* vars, T* src, T* dest, int row, int col){
  cudaIdx()
	int targetindex = (x+(row-cuda_row)/2)*col + y+(col-cuda_column)/2;
	dest[index] = src[targetindex];
}
template <typename T>
__global__ void crop(cudaVars* vars, T* src, T* dest, int row, int col, Real midx, Real midy){
  cudaIdx()
	int targetindex = (x+(row-cuda_row)/2+int(row*midx))*col + y+(col-cuda_column)/2+int(col*midy);
	dest[index] = src[targetindex];
}

template <typename T>
__global__ void pad(cudaVars* vars, T* src, T* dest, int row, int col, int shiftx = 0, int shifty = 0){
  cudaIdx()
	int marginx = (cuda_row-row)/2+shiftx;
	int marginy = (cuda_column-col)/2+shifty;
	if(x < marginx || x >= row+marginx || y < marginy || y >= col+marginy){
		dest[index] = T();
		return;
	}
	int targetindex = (x-marginx)*col + y-marginy;
	dest[index] = src[targetindex];
}

template <typename T>
__global__ void refine(cudaVars* vars, T* src, T* dest, int refinement){
  cudaIdx()
	int indexlu = (x/refinement)*(cuda_row/refinement) + y/refinement;
	int indexld = (x/refinement)*(cuda_row/refinement) + y/refinement+1;
	int indexru = (x/refinement+1)*(cuda_row/refinement) + y/refinement;
	int indexrd = (x/refinement+1)*(cuda_row/refinement) + y/refinement+1;
	Real dx = Real(x%refinement)/refinement;
	Real dy = Real(y%refinement)/refinement;
	dest[index] = 
		src[indexlu]*(1-dx)*(1-dy)
		+((y<cuda_column-refinement)?src[indexld]*(1-dx)*dy:0)
		+((x<cuda_row-refinement)?src[indexru]*dx*(1-dy):0)
		+((y<cuda_column-refinement&&x<cuda_row-refinement)?src[indexrd]*dx*dy:0);
}


class rect{
  public:
    int startx;
    int starty;
    int endx;
    int endy;
    __device__ __host__ bool isInside(int x, int y){
      if(x > startx && x <= endx && y > starty && y <= endy) return true;
      return false;
    }
};
class C_circle{
  public:
    int x0;
    int y0;
    Real r;
    __device__ __host__ bool isInside(int x, int y){
      Real dr = sqrt(pow(x-x0,2)+pow(y-y0,2));
      if(dr < r) return true;
      return false;
    }
};
template <typename sptType>
__global__ void createMask(cudaVars* vars, Real* data, sptType* spt, bool isFrequency=0){
  cudaIdx()
  if(isFrequency){
    if(x>=cuda_row/2) x-=cuda_row/2;
    else x+=cuda_row/2;
    if(y>=cuda_column/2) y-=cuda_column/2;
    else y+=cuda_column/2;
  }
  data[index]=spt->isInside(x,y);
}
#endif
