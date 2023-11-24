#ifndef __CUDACONFIG_H__
#define __CUDACONFIG_H__
#include "format.h"
#include<stddef.h>
#include<stdint.h>
#define FFTformat CUFFT_C2C
#define FFTformatR2C CUFFT_R2C
#define myCufftExec cufftExecC2C
#define myCufftExecR2C cufftExecR2C
#define myCuDMalloc(fmt, var, size) fmt* var = (fmt*)memMngr.borrowCache(size*sizeof(fmt));
#define myCuMalloc(fmt, var, size) var = (fmt*)memMngr.borrowCache(size*sizeof(fmt));
#define myCuFree(ptr) memMngr.returnCache(ptr); ptr = 0
#include "memManager.h"
void myMemcpyH2D(void*, void*, size_t sz);
void myMemcpyD2D(void*, void*, size_t sz);
void myMemcpyD2H(void*, void*, size_t sz);
void resize_cuda_image(int row, int col);
void init_cuda_image(int rcolor=0, Real scale=0);
void* newRand(size_t sz);

class cuMemManager : public memManager{
  void c_malloc(void*& ptr, size_t sz);
  void c_memset(void*& ptr, size_t sz);
  public:
  cuMemManager():memManager(){};
};
extern cuMemManager memMngr;
void myFFT(void* in, void* out);
void myIFFT(void* in, void* out);
void myFFTM(int handle, void* in, void* out);
void myIFFTM(int handle, void* in, void* out);
void myFFTR2C(void* in, void* out);
void createPlan(int* handle, int row, int col);
void createPlan1d(int* handle, int n);
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
void applyPoissonNoise(Real* wave, Real noiseLevel, void *state, Real scale = 0);
void applyPoissonNoise_WO(Real* wave, Real noiseLevel, void *state, Real scale = 0);
void ccdRecord(uint16_t* data, Real* wave, int noiseLevel, void *state, Real exposure = 1);
void ccdRecord(uint16_t* data, complexFormat* wave, int noiseLevel, void *state, Real exposure = 1);
void ccdRecord(Real* data, Real* wave, int noiseLevel, void *state, Real exposure = 1);
void ccdRecord(Real* data, complexFormat* wave, int noiseLevel, void *state, Real exposure = 1);
void ccdRecord(complexFormat* data, complexFormat* wave, int noiseLevel, void *state, Real exposure = 1);
void initRand(void *state,unsigned long long seed);
void fillRedundantR2C(complexFormat* data, complexFormat* dataout, Real factor);
void applyMod(complexFormat* source, Real* target, Real *bs = 0, bool loose=0, int iter = 0, int noiseLevel = 0);
void applyModAbs(complexFormat* source, Real* target, void *state = 0);
void applyModAbsinner(complexFormat* source, Real* target,  int row, int col, Real norm, void *state);
void linearConst(Real* store, Real* data, Real factor, Real b);
void add(complexFormat* a, complexFormat* b, Real c = 1);
void add(complexFormat* store, complexFormat* a, complexFormat* b, Real c = 1);
void applyRandomPhase(complexFormat* wave, Real* beamstop, void *state);
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
template<typename T>
void crop(T* src, T* dest, int row, int col, Real midx = 0, Real midy = 0);
void applyMaskBar(complexFormat* data, Real* mask, Real threshold = 0.5);
void applyMaskBar(Real* data, Real* mask, Real threshold = 0.5);
void applyMaskBar(Real* data, complexFormat* mask, Real threshold = 0.5);
void applyMask(complexFormat* data, Real* mask, Real threshold = 0.5);
void applyMask(Real* data, Real* mask, Real threshold = 0.5);

template<typename T>
void refine(T* src, T* dest, int refinement);
template<typename T>
void pad(T* src, T* dest, int row, int col, int shiftx = 0, int shifty = 0);
template <typename T1, typename T2>
void assignVal(T1* out, T2* input);
template<typename T>
void zeroEdge(T* a, int n);
template<typename T>
void cudaConvertFO(T* data, T* out = 0);
template<typename T>
void createMask(Real* data, T* spt, bool isFrequency=0);
template<typename T>
void createMaskBar(Real* data, T* spt, bool isFrequency);

class rect{
  public:
    int startx;
    int starty;
    int endx;
    int endy;
    bool isInside(int x, int y);
};
class C_circle{
  public:
    int x0;
    int y0;
    Real r;
    bool isInside(int x, int y);
};
#endif
