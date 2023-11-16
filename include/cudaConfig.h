#ifndef __CUDACONFIG_H__
#define __CUDACONFIG_H__
#include "format.h"
#include "cudaDefs.h"
#include <curand_kernel.h>
#include<stdint.h>
#define FFTformat CUFFT_C2C
#define FFTformatR2C CUFFT_R2C
#define myCufftExec cufftExecC2C
#define myFFT(args...) myCufftExec(*plan, args, CUFFT_FORWARD)
#define myIFFT(args...) myCufftExec(*plan, args, CUFFT_INVERSE)
#define myCufftExecR2C cufftExecR2C
#define myFFTR2C(args...) myCufftExecR2C(*planR2C, args)
void forcePositive(complexFormat* a);
void forcePositive(Real* a);
void add(Real* a, Real* b, Real c = 1);
void add(Real* store, Real* a, Real* b, Real c = 1);
void addRemoveOE(Real* src, Real* sub, Real mult);
void bitMap(Real* store, Real* data, Real threshold = 0.1);
void bitMap(Real* store, complexFormat* data, Real threshold = 0.1);
void applyThreshold(Real* store, Real* input, Real threshold = 0.5);
void extendToComplex(Real* a, complexFormat* b);
void applyNorm(complexFormat* data, Real factor);
void ceiling(complexFormat* data, Real ceilval);
void applyNorm(Real* data, Real factor);
void interpolate(Real* out, Real* data0, Real* data1, Real dx);
void interpolate(complexFormat* out, complexFormat* data0, complexFormat* data1, Real dx);
void adamUpdateV(Real* v, Real* grad, Real beta2);
void adamUpdateV(Real* v, complexFormat* grad, Real beta2);
void adamUpdate(complexFormat* x, complexFormat* m, Real* v, Real lr, Real eps);
void createWaveFront(Real* d_intensity, Real* d_phase, complexFormat* objectWave, Real oversampling, Real shiftx = 0, Real shifty = 0, Real phaseFactor = 0);
void createWaveFront(Real* d_intensity, Real* d_phase, complexFormat* objectWave, int row, int col, int shiftx = 0, int shifty = 0, Real phaseFactor = 0);
void multiplyPropagatePhase(complexFormat* amp, Real a, Real b); // a=z/lambda, b = (s/lambda)^2, s is the image size
void applyConvolution(size_t sz, Real *input, Real *output, Real* kernel, int kernelwidth, int kernelheight);
void shiftWave(complexFormat* wave, Real shiftx, Real shifty);
void shiftMiddle(complexFormat* wave);
void getMod(Real* mod, complexFormat* amp);
void getReal(Real* mod, complexFormat* amp);
void getImag(Real* mod, complexFormat* amp);
void assignReal(Real* mod, complexFormat* amp);
void assignImag(Real* mod, complexFormat* amp);
void getMod2(Real* mod, complexFormat* amp);
void addMod2(Real* mod, complexFormat* amp, Real norm);
void getMod2(Real* mod2, Real* mod);
void applyPoissonNoise(Real* wave, Real noiseLevel, curandStateMRG32k3a *state, Real scale = 0);
void applyPoissonNoise_WO(Real* wave, Real noiseLevel, curandStateMRG32k3a *state, Real scale = 0);
void ccdRecord(uint16_t* data, Real* wave, int noiseLevel, curandStateMRG32k3a *state, Real exposure = 1);
void ccdRecord(uint16_t* data, complexFormat* wave, int noiseLevel, curandStateMRG32k3a *state, Real exposure = 1);
void ccdRecord(Real* data, Real* wave, int noiseLevel, curandStateMRG32k3a *state, Real exposure = 1);
void ccdRecord(Real* data, complexFormat* wave, int noiseLevel, curandStateMRG32k3a *state, Real exposure = 1);
void ccdRecord(complexFormat* data, complexFormat* wave, int noiseLevel, curandStateMRG32k3a *state, Real exposure = 1);
void initRand(curandStateMRG32k3a *state,unsigned long long seed);
void fillRedundantR2C(complexFormat* data, complexFormat* dataout, Real factor);
void applyMod(complexFormat* source, Real* target, Real *bs = 0, bool loose=0, int iter = 0, int noiseLevel = 0);
void applyModAbs(complexFormat* source, Real* target, curandStateMRG32k3a *state = 0);
void applyModAbsinner(complexFormat* source, Real* target,  int row, int col, Real norm, curandStateMRG32k3a *state);
void linearConst(Real* store, Real* data, Real factor, Real b);
void add(complexFormat* a, complexFormat* b, Real c = 1);
void add(complexFormat* store, complexFormat* a, complexFormat* b, Real c = 1);
void applyRandomPhase(complexFormat* wave, Real* beamstop, curandStateMRG32k3a *state);
void multiply(complexFormat* source, complexFormat* target);
void multiplyReal(Real* store, complexFormat* source, complexFormat* target);
void multiply(complexFormat* store, complexFormat* source, complexFormat* target);
void multiply(Real* store, Real* source, Real* target);
void stretch(Real* src, Real* dest, Real rat, int prec);
void convertFOPhase(complexFormat* data);
void mergePixel(Real* input, Real* output, int row, int col, int nmerge);
void cropinner(Real* src, Real* dest, int row, int col, Real norm);
void cropinner(complexFormat* src, complexFormat* dest, int row, int col, Real norm = 1);
void padinner(Real* src, Real* dest, int row, int col, Real norm);
void padinner(complexFormat* src, complexFormat* dest, int row, int col, Real norm);
void createGauss(Real* data, int sz, Real sigma);
void applyGaussConv(Real* input, Real* output, Real* gaussMem, Real sigma);
void init_fft(int rows, int cols, int batch = 1);
void readComplexWaveFront(const char* intensityFile, const char* phaseFile, Real* &d_intensity, Real* &d_phase, int &objrow, int &objcol);
cuFuncTemplate(cudaConvertFO, (T* data, T* out = 0),(data,out==0?data:out),{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index >= (cuda_row*cuda_column)/2) return;
  int x = index%(cuda_row>>1);
  int y = index/(cuda_row>>1);
  index = x*cuda_column+y;
  int indexp = (x+(cuda_row>>1))*cuda_column + (y >= (cuda_column>>1)? y-(cuda_column>>1): (y+(cuda_column>>1)));
  T tmp = data[index];
  out[index]=data[indexp];
  out[indexp]=tmp;
})

cuFuncTemplate(applyMask, (T* data, Real* mask, Real threshold = 0.5),(data,mask,threshold),{
  cuda1Idx();
  if(mask[index]<=threshold) data[index] = T();
})
cuFuncTemplate(applyMaskBar, (T* data, complexFormat* mask, Real threshold = 0.5),(data,mask,threshold),{
  cuda1Idx();
  if(mask[index].x>threshold) data[index] = T();
})


cuFuncTemplate(zeroEdge, (T* a, int n), (a,n),{
  cudaIdx()
  if(x<n || x>=cuda_row-n || y < n || y >= cuda_column-n)
    a[index] = T();
})

template <typename T1, typename T2>
__global__ void assignValWrap(int cuda_row, int cuda_column, T1* out, T2* input){
  cuda1Idx()
	out[index] = input[index];
}
template <typename T1, typename T2>
void assignVal(T1* out, T2* input){
  assignValWrap<<<numBlocks,threadsPerBlock>>>(cuda_imgsz.x, cuda_imgsz.y,out,input);
}
template<typename T>
void crop(T* src, T* dest, int row, int col, Real midx = 0, Real midy = 0);
cuFuncTemplate(pad,(T* src, T* dest, int row, int col, int shiftx = 0, int shifty = 0),(src, dest, row, col, shiftx, shifty),{
  cudaIdx()
	int marginx = (cuda_row-row)/2+shiftx;
	int marginy = (cuda_column-col)/2+shifty;
	if(x < marginx || x >= row+marginx || y < marginy || y >= col+marginy){
		dest[index] = T();
		return;
	}
	int targetindex = (x-marginx)*col + y-marginy;
	dest[index] = src[targetindex];
})
cuFuncTemplate(refine,(T* src, T* dest, int refinement),(src,dest,refinement),{
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
})

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
      Real dr = hypot(Real(x-x0),Real(y-y0));
      if(dr < r) return true;
      return false;
    }
};
cuFuncTemplate(createMask,(Real* data, T* spt, bool isFrequency=0),(data,spt,isFrequency),{
  cudaIdx()
  if(isFrequency){
    if(x>=cuda_row/2) x-=cuda_row/2;
    else x+=cuda_row/2;
    if(y>=cuda_column/2) y-=cuda_column/2;
    else y+=cuda_column/2;
  }
  data[index]=spt->isInside(x,y);
})
cuFuncTemplate(createMaskBar,(Real* data, T* spt, bool isFrequency=0),(data,spt,isFrequency),{
  cudaIdx()
  if(isFrequency){
    if(x>=cuda_row/2) x-=cuda_row/2;
    else x+=cuda_row/2;
    if(y>=cuda_column/2) y-=cuda_column/2;
    else y+=cuda_column/2;
  }
  data[index]=!spt->isInside(x,y);
})
#endif
