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
cuFuncDec(forcePositive, complexFormat* a);
cuFuncDec(forcePositive, Real* a);
cuFuncDec(add, Real* a, Real* b, Real c = 1);
cuFuncDec(add, Real* store, Real* a, Real* b, Real c = 1);
cuFuncDec(bitMap, Real* store, Real* data, Real threshold = 0.1);
cuFuncDec(bitMap, Real* store, complexFormat* data, Real threshold = 0.1);
cuFuncDec(extendToComplex, Real* a, complexFormat* b);
cuFuncDec(applyNorm, complexFormat* data, Real factor);
cuFuncDec(ceiling, complexFormat* data, Real ceilval);
cuFuncDec(applyNorm, Real* data, Real factor);
cuFuncDec(createWaveFront, Real* d_intensity, Real* d_phase, complexFormat* objectWave, Real oversampling, Real shiftx = 0, Real shifty = 0);
cuFuncDec(createWaveFront, Real* d_intensity, Real* d_phase, complexFormat* objectWave, int row, int col, int shiftx = 0, int shifty = 0);
cuFuncSharedDec(applyConvolution, Real *input, Real *output, Real* kernel, int kernelwidth, int kernelheight);
cuFuncDec(getMod, Real* mod, complexFormat* amp);
cuFuncDec(getReal, Real* mod, complexFormat* amp);
cuFuncDec(getImag, Real* mod, complexFormat* amp);
cuFuncDec(assignReal, Real* mod, complexFormat* amp);
cuFuncDec(assignImag, Real* mod, complexFormat* amp);
cuFuncDec(getMod2, Real* mod, complexFormat* amp);
cuFuncDec(applyPoissonNoise, Real* wave, Real noiseLevel, curandStateMRG32k3a *state, Real scale = 0);
cuFuncDec(applyPoissonNoise_WO, Real* wave, Real noiseLevel, curandStateMRG32k3a *state, Real scale = 0);
cuFuncDec(initRand, curandStateMRG32k3a *state,unsigned long long seed);
cuFuncDec(fillRedundantR2C, complexFormat* data, complexFormat* dataout, Real factor);
cuFuncDec(applyMod, complexFormat* source, Real* target, Real *bs = 0, bool loose=0, int iter = 0, int noiseLevel = 0);
cuFuncDec(applyModAbs, complexFormat* source, Real* target);
cuFuncDec(linearConst, Real* store, Real* data, Real factor, Real b);
cuFuncDec(add, complexFormat* a, complexFormat* b, Real c = 1);
cuFuncDec(add, complexFormat* store, complexFormat* a, complexFormat* b, Real c = 1);
cuFuncDec(applyRandomPhase, complexFormat* wave, Real* beamstop, curandStateMRG32k3a *state);
cuFuncDec(multiply, complexFormat* source, complexFormat* target);
cuFuncDec(multiplyReal, Real* store, complexFormat* source, complexFormat* target);
cuFuncDec(multiply, complexFormat* store, complexFormat* source, complexFormat* target);
cuFuncDec(convertFOPhase, complexFormat* data);
void createGauss(Real* data, int sz, Real sigma);
void applyGaussConv(Real* input, Real* output, Real* gaussMem, Real sigma);
void init_fft(int rows, int cols);
void readComplexWaveFront(const char* intensityFile, const char* phaseFile, Real* &d_intensity, Real* &d_phase, int &objrow, int &objcol);
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
cuFuncDec(cudaConvertFO, T* data){
  cudaConvertFO<<<numBlocks,threadsPerBlock>>>(vars,data);
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
cuFuncDec(cudaConvertFO, T* data, T* out){
  cudaConvertFO<<<numBlocks,threadsPerBlock>>>(vars,data,out);
}

template <typename T>
__global__ void zeroEdge(cudaVars* vars, T* a, int n){
  cudaIdx()
  if(x<n || x>=cuda_row-n || y < n || y >= cuda_column-n)
    a[index] = T();
}
template <typename T>
cuFuncDec(zeroEdge, T* a, int n){
  zeroEdge<<<numBlocks,threadsPerBlock>>>(vars,a,n);
}

template <typename T1, typename T2>
__global__ void assignVal(cudaVars* vars, T1* out, T2* input){
  cudaIdx()
	out[index] = input[index];
}
template <typename T1, typename T2>
cuFuncDec(assignVal, T1* out, T2* input){
  assignVal<<<numBlocks,threadsPerBlock>>>(vars,out,input);
}

template <typename T>
__global__ void crop(cudaVars* vars, T* src, T* dest, int row, int col, Real midx, Real midy){
  cudaIdx()
	int targetindex = (x+(row-cuda_row)/2+int(row*midx))*col + y+(col-cuda_column)/2+int(col*midy);
	dest[index] = src[targetindex];
}

template <typename T>
cuFuncDec(crop, T* src, T* dest, int row, int col, Real midx = 0, Real midy = 0){
  crop<<<numBlocks,threadsPerBlock>>>(vars,src,dest,row,col,midx,midy);
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
cuFuncDec(pad, T* src, T* dest, int row, int col, int shiftx = 0, int shifty = 0){
  pad<<<numBlocks,threadsPerBlock>>>(vars, src, dest, row, col, shiftx, shifty);
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
template <typename T>
cuFuncDec(refine, T* src, T* dest, int refinement){
  refine<<<numBlocks,threadsPerBlock>>>(vars,src,dest,refinement);
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

template <typename sptType>
cuFuncDec(createMask, Real* data, sptType* spt, bool isFrequency=0){
  createMask<<<numBlocks,threadsPerBlock>>>(vars,data,spt,isFrequency);
}
#endif
