#ifndef __CUDACONFIG_H__
#define __CUDACONFIG_H__
#include "format.hpp"
#include<stddef.h>
#include<stdint.h>
#define FFTformat CUFFT_C2C
#define FFTformatR2C CUFFT_R2C
#define myCufftExec cufftExecC2C
#define myCufftExecR2C cufftExecR2C
#define myCufftExecC2R cufftExecC2R
#define myCuDMalloc(fmt, var, size) fmt* var = (fmt*)memMngr.borrowCache(size*sizeof(fmt));
#define myCuDMallocClean(fmt, var, size) fmt* var = (fmt*)memMngr.borrowCleanCache(size*sizeof(fmt));
#define myCuMalloc(fmt, var, size) var = (fmt*)memMngr.borrowCache(size*sizeof(fmt));
#define myCuMallocClean(fmt, var, size) var = (fmt*)memMngr.borrowCleanCache(size*sizeof(fmt));
#define myCuFree(ptr) memMngr.returnCache(ptr); ptr = 0
#include "memManager.hpp"
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
void gpuAssert(int code, const char *file, int line);
int getCudaRows();
int getCudaCols();
void myMemcpyH2D(void*, void*, size_t sz);
void myMemcpyD2D(void*, void*, size_t sz);
void myMemcpyD2H(void*, void*, size_t sz);
void myMemcpyH2DAsync(void*, void*, size_t sz);
void myMemcpyD2DAsync(void*, void*, size_t sz);
void myMemcpyD2HAsync(void*, void*, size_t sz);
void resize_cuda_image(int row, int col, int hei = 1);
void init_cuda_image(int rcolor=0, Real scale=0);
void initRand(void *state,unsigned long long seed);
void randMask(char* mask, void* state, Real ratio = .5);
void* newRand(size_t sz);
void clearCuMem(void*, size_t);
void setThreshold(Real);
void gpuerr();
size_t getGPUFreeMem();

class cuMemManager : public memManager{
  void c_malloc(void*& ptr, size_t sz);
  void c_memset(void*& ptr, size_t sz);
  public:
  cuMemManager();
};
extern cuMemManager memMngr;

int svd_init(int M, int N);
int svd_execute(int handle, const float *d_A_input);
int svd_destroy(int handle);
const char* svd_get_last_error();

void init_fft(int rows, int cols, int batch = 1);
void myFFT(void* in, void* out);
void myIFFT(void* in, void* out);
void myFFTM(int handle, void* in, void* out);
void myIFFTM(int handle, void* in, void* out);
void myFFTR2C(void* in, void* out);
void myFFTC2R(void* in, void* out);
void createPlan(int* handle, int row, int col);
void createPlan1d(int* handle, int n);
void destroyPlan(int handle);
void forcePositive(complexFormat* a);
void forcePositive(Real* a);
void add(Real* a, Real* b, Real c = 1);
void add(Real* store, Real* a, Real* b, Real c = 1);
void add(complexFormat* a, complexFormat* b, Real c = 1);
void add(complexFormat* store, complexFormat* a, complexFormat* b, Real c = 1);
void normAdd(complexFormat* store, complexFormat* a, complexFormat* b, Real c = 1, Real d = 1);
void addRemoveOE(Real* src, Real* sub, Real mult);
void bitMap(Real* store, Real* data, Real threshold = 0);
void bitMap(Real* store, complexFormat* data, Real threshold = 0);
void applyThreshold(Real* store, Real* input, Real threshold = 0.5);
void extendToComplex(Real* a, complexFormat* b);
void applyNorm(complexFormat* data, Real factor);
void ceiling(complexFormat* data, Real ceilval);
void applyNorm(Real* data, Real factor);
void invert(Real* data);
void rotateToReal(complexFormat* data);
void interpolate(Real* out, Real* data0, Real* data1, Real dx);
void interpolate(complexFormat* out, complexFormat* data0, complexFormat* data1, Real dx);
void adamUpdateV(Real* v, Real* grad, Real beta2);
void adamUpdateV(Real* v, complexFormat* grad, Real beta2);
void adamUpdate(complexFormat* x, complexFormat* m, Real* v, Real lr, Real eps);
void createWaveFront(Real* d_intensity, Real* d_phase, complexFormat* objectWave, Real oversampling, Real shiftx = 0, Real shifty = 0, Real phaseFactor = 0);
void createWaveFront(Real* d_intensity, Real* d_phase, complexFormat* objectWave, int row, int col, int shiftx = 0, int shifty = 0, Real phaseFactor = 0);
void multiplyPropagatePhase(complexFormat* amp, Real a, Real b); // a=z/lambda, b = (s/lambda)^2, s is the image size
void applyConvolution(size_t sz, Real *input, Real *output, Real* kernel, int kernelwidth, int kernelheight);
void applyGaussMult(complexFormat* input, complexFormat* output, Real sigma, bool isFreq);
void applyGaussMult(Real* input, Real* output, Real sigma, bool isFreq);
void multiplyShift(complexFormat* wave, Real shiftx, Real shifty);
void getMod(Real* mod, complexFormat* amp);
void getImag(Real* mod, complexFormat* amp);
void assignReal(Real* mod, complexFormat* amp);
void assignImag(Real* mod, complexFormat* amp);
void getMod2(Real* mod2, complexFormat* amp, Real norm = 1);
void getMod2(complexFormat* mod2, complexFormat* amp, Real norm = 1);
void addMod2(Real* mod2, complexFormat* amp, Real norm = 1);
void addReal(Real* mod, complexFormat* amp, Real norm = 1);
void getReal(Real* mod, complexFormat* amp, Real norm = 1);
void getMod2(Real* mod2, Real* mod, Real norm = 1);
void applyPoissonNoise(Real* wave, Real noiseLevel, void *state, Real scale = 0);
void applyPoissonNoise_WO(Real* wave, Real noiseLevel, void *state, Real scale = 0);
void ccdRecord(uint16_t* data, Real* wave, int noiseLevel, void *state, Real exposure = 1);
void ccdRecord(uint16_t* data, complexFormat* wave, int noiseLevel, void *state, Real exposure = 1);
void ccdRecord(Real* data, Real* wave, int noiseLevel, void *state, Real exposure = 1, int rcolor = 0);
void ccdRecord(Real* data, complexFormat* wave, int noiseLevel, void *state, Real exposure = 1);
void ccdRecord(complexFormat* data, complexFormat* wave, int noiseLevel, void *state, Real exposure = 1);
void fillRedundantR2C(complexFormat* data, complexFormat* dataout, Real factor);
void applyMod(complexFormat* source, Real* target, Real *bs = 0, int noiseLevel = 0, Real norm = 1);
void applyModAbs(complexFormat* source, Real* target, void *state = 0);
void applyModAbsinner(complexFormat* source, Real* target,  int row, int col, Real norm, void *state);
void linearConst(Real* store, Real* data, Real factor, Real b);
void linearConst(complexFormat* store, complexFormat* data, complexFormat factor, complexFormat b);
void applyRandomPhase(complexFormat* wave, Real* beamstop, void *state);
template <typename T1, typename T2>
void multiply(T1* store, T1* source, T2* target);
void multiplyReal(Real* store, complexFormat* source, complexFormat* target);
void multiplyConj(complexFormat* store, complexFormat* src, complexFormat* target);
void multiplyRegular(complexFormat* store, complexFormat* src, complexFormat* target, Real alpha);
void convertFOPhase(complexFormat* data, Real norm = 1);
void mergePixel(Real* input, Real* output, int row, int col, int nmerge);
void cropinner(Real* src, Real* dest, int row, int col, Real norm);
void cropinner(complexFormat* src, complexFormat* dest, int row, int col, Real norm = 1);
void padinner(Real* src, Real* dest, int row, int col, Real norm = 1);
void padinner(complexFormat* src, complexFormat* dest, int row, int col, Real norm = 1);
void createGauss(Real* data, int sz, Real sigma);
void ssimMap(Real* mu1, Real* mu2, Real* sigma1sq, Real* sigma2sq, Real* sigma12, Real C1, Real C2);
void readComplexWaveFront(const char* intensityFile, const char* phaseFile, Real* &d_intensity, Real* &d_phase, int &objrow, int &objcol);
void zeroEdgey(complexFormat* a, int n);
void zeroEdge(Real* a, int n);
void zeroEdge(complexFormat* a, int n);
void applyMaskBar(complexFormat* data, Real* mask, Real threshold = 0.5);
void applyMaskBar(Real* data, Real* mask, Real threshold = 0.5);
void applyMaskBar(Real* data, complexFormat* mask, Real threshold = 0.5);
void applyMask(complexFormat* data, Real* mask, Real threshold = 0.5);
void applyMask(Real* data, Real* mask, Real threshold = 0.5);
void applyMask(Real* data, char* mask);
void paste(Real* out, Real* in, int colout, int posx, int posy, bool replace = 0, Real norm = 1);
void paste(complexFormat* out, complexFormat* in, int colout, int posx, int posy, bool replace = 0);
void takeMod2Diff(complexFormat* a, Real* b, Real *output, Real *bs);
void takeMod2Sum(complexFormat* a, Real* b);
void applySupportOblique(complexFormat *gkp1, complexFormat *gkprime, int algo, Real *spt, int iter = 0, Real fresnelFactor = 0, Real costheta_r = 1);
void applySupport(void *gkp1, void *gkprime, int algo, Real *spt, int iter = 0, Real fresnelFactor = 0);
void getXYSlice(Real* slice, Real* data, int nx, int ny, int iz);
void getXZSlice(Real* slice, Real* data, int nx, int ny, int nz, int iy);
void getYZSlice(Real* slice, Real *data, int nx, int ny, int nz, int ix);
void createColorbar(complexFormat* output);
void multiplyx(complexFormat* object, Real* out);
void multiplyy(complexFormat* object, Real* out);
void multiplyx(Real* object, Real* out);
void multiplyy(Real* object, Real* out);
void getArg(Real* angle, complexFormat* amp);

void phaseUnwrapping(Real* d_wrapped_phase, Real* d_unwrapped_phase, int width, int height);
void solve_poisson_frequency_domain(complexFormat* d_fft_data, int width, int height);

template<typename T>
void crop(T* src, T* dest, int row, int col, Real midx = 0, Real midy = 0);
template<typename T>
void setValue(T* data, T value);
template<typename T>
void refine(T* src, T* dest, int refinement);
template<typename T>
void pad(T* src, T* dest, int row, int col, int shiftx = 0, int shifty = 0);
template<typename T>
void randZero(T* src, T* dest, void* state, Real ratio = 0.5, char step = 1);
template<typename T>
void resize(const T* input, T* output, int in_width, int in_height);
template <typename T1, typename T2>
void assignVal(T1* out, T2* input);
template<typename T>
void cudaConvertFO(T* data, T* out = 0, Real norm = 1);
template<typename T>
void getWindow(T* object, int shiftx, int shifty, int objrow, int objcol, T *window, bool replace = 0, Real norm = 1);
template<typename T>
void rotate90(T* data, T* out = 0, bool clockwise=1);
template<typename T>
void rotate(T* data, T* out, Real angle);
template<typename T>
void transpose(T* data, T* out = 0);
template<typename T>
void flipx(T* data, T* out = 0);
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
#ifdef __CUDADEFS_H__
    __device__ __host__ bool isInside(int x, int y);
#else
    bool isInside(int x, int y);
#endif // CUDACC
};
class C_circle{
  public:
    int x0;
    int y0;
    Real r;
#ifdef __CUDADEFS_H__
    __device__ __host__ bool isInside(int x, int y);
#else
    bool isInside(int x, int y);
#endif // CUDACC
};
class diamond{
  public:
    int startx;
    int starty;
    int width;
    int height;
#ifdef __CUDADEFS_H__
    __device__ __host__ bool isInside(int x, int y);
#else
    bool isInside(int x, int y);
#endif // CUDACC
};
#endif
