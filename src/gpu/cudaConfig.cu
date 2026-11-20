#include "cudaDefs_h.cu"
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#include "cudaConfig.hpp"
#include "fmt/core.h"
#include <complex.h>
#include <curand_kernel.h>
#include <cub_wrap.hpp>
#include <cufft.h>
#include <cublas_v2.h>
#define FFTformat CUFFT_C2C
#define FFTformatR2C CUFFT_R2C
#define myCufftExec cufftExecC2C
#define myCufftExecR2C cufftExecR2C
#define myCufftExecC2R cufftExecC2R
dim3 numBlocks;
const dim3 threadsPerBlock = 256;
complexFormat *cudaData = 0;
static cufftHandle *plan = 0, *planR2C = 0;
dim3 cuda_imgsz = {0,0,1};
void cuMemManager::c_malloc(void*& ptr, size_t sz) { gpuErrchk(cudaMalloc((void**)&ptr, sz)); }
void cuMemManager::c_memset(void*& ptr, size_t sz) { gpuErrchk(cudaMemset(ptr, 0, sz)); }
cuMemManager::cuMemManager():memManager(){}
cuMemManager memMngr;

__host__ __device__ Real gaussian(float x, float y, float sigma){
  return exp(-(x*x+y*y)/2/(sigma*sigma));
}
void gpuAssert(int code, const char *file, int line)
{
  if (code != cudaSuccess)
  {
    fmt::println(stderr,"GPUassert: {} {} {}", cudaGetErrorString((cudaError_t)code), file, line);
    abort();
  }
}

__forceinline__ __device__ cuComplex make_polar(Real r, Real theta) {
  Real c, s;
  sincosf(theta, &s, &c);
  return make_float2(r*c, r*s);
}
int getCudaRows(){
  return cuda_imgsz.x;
}
int getCudaCols(){
  return cuda_imgsz.y;
}
void myMemcpyH2D(void* d, void* s, size_t sz){
  gpuErrchk(cudaMemcpy(d, s, sz, cudaMemcpyHostToDevice));
}
void myMemcpyD2D(void* d, void* s, size_t sz){
  gpuErrchk(cudaMemcpy(d, s, sz, cudaMemcpyDeviceToDevice));
}
void myMemcpyD2H(void* d, void* s, size_t sz){
  gpuErrchk(cudaMemcpy(d, s, sz, cudaMemcpyDeviceToHost));
}
void myMemcpyH2DAsync(void* d, void* s, size_t sz){
  gpuErrchk(cudaMemcpy(d, s, sz, cudaMemcpyHostToDevice));
}
void myMemcpyD2DAsync(void* d, void* s, size_t sz){
  gpuErrchk(cudaMemcpy(d, s, sz, cudaMemcpyDeviceToDevice));
}
void myMemcpyD2HAsync(void* d, void* s, size_t sz){
  gpuErrchk(cudaMemcpyAsync(d, s, sz, cudaMemcpyDeviceToHost));
}
void clearCuMem(void* ptr, size_t sz){
  gpuErrchk(cudaMemset(ptr, 0, sz));
}

void resize_cuda_image(int rows, int cols, int layers){
  cuda_imgsz.x = rows;
  cuda_imgsz.y = cols;
  cuda_imgsz.z = layers;
  numBlocks.x=(rows*cols*layers-1)/threadsPerBlock.x+1;
}
size_t getGPUFreeMem(){
  size_t freeBytes, totalBytes;
  gpuErrchk(cudaMemGetInfo(&freeBytes, &totalBytes));
  return freeBytes >> 20;
}
void* newRand(size_t sz){
  return memMngr.borrowCache(sz * sizeof(curandStateMRG32k3a));
}

void gpuerr(){
  gpuErrchk(cudaGetLastError());
}

static int rows_fft, cols_fft, batch_fft;
__forceinline__ __device__ __host__ bool rect::isInside(int x, int y){
  if(x >= startx && x < endx && y >= starty && y < endy) return true;
  return false;
}
__forceinline__ __device__ __host__ bool C_circle::isInside(int x, int y){
  Real dx = x - x0, dy = y - y0;
  Real dr = sqrtf(dx*dx+dy*dy);
  if(dr < r) return true;
  return false;
}
__forceinline__ __device__ __host__ bool diamond::isInside(int x, int y){
  Real k = Real(height)/width;
  Real kx = (x-startx)*k;
  y -= starty;
  Real b1 = Real(height)/2;
  Real b2 = b1 + height;
  if(y < b1 - kx) return false;
  if(y > b2 - kx) return false;
  if(y > b1 + kx) return false;
  if(y < -b1 + kx) return false;
  return true;
}

__device__ __forceinline__ cuComplex multiply_dev(cuComplex val, cuComplex val2){
  return val * val2;
}

template<typename T2>
__device__ __forceinline__ cuComplex multiply_dev(cuComplex val, T2 norm){
  val.x *= norm;
  val.y *= norm;
  return val;
}

template<typename T1, typename T2>
__device__ __forceinline__ T1 multiply_dev(T1 val, T2 norm){
  return val*norm;
}

template<typename T1, typename T2>
__device__ __forceinline__ T1 add(T1 val1, T2 val2){
  return val1+val2;
}
__device__ __forceinline__ cuComplex add(cuComplex a, cuComplex b){
  return make_float2(a.x + b.x, a.y + b.y);
}

// ---- Generic elementwise kernel templates ----
template<typename O, typename X, typename F>
__global__ void k_elem1(dim3 cuda_imgsz, O* o, const X* x, F f){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i >= cuda_imgsz.x*cuda_imgsz.y) return;
  o[i] = f(x[i]);
}
template<typename O, typename X, typename Y, typename F>
__global__ void k_elem2(dim3 cuda_imgsz, O* o, const X* x, const Y* y, F f){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i >= cuda_imgsz.x*cuda_imgsz.y) return;
  o[i] = f(x[i], y[i]);
}
template<typename O, typename X, typename Y, typename Z, typename F>
__global__ void k_elem3(dim3 cuda_imgsz, O* o, const X* x, const Y* y, const Z* z, F f){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i >= cuda_imgsz.x*cuda_imgsz.y) return;
  o[i] = f(x[i], y[i], z[i]);
}

void init_fft(int rows, int cols, int batch){
  fmt::println("init fft: {} {}, old dim={}, {}", rows, cols, rows_fft, cols_fft);
  if(rows!=rows_fft||cols!=cols_fft||batch_fft!=batch){
    if(!plan){
      plan = new cufftHandle();
      planR2C = new cufftHandle();
    }else{
      cufftDestroy(*plan);
      cufftDestroy(*planR2C);
    }
    if(cols == 1){
      int dim[2] = {rows, batch};
      cufftPlanMany ( plan, 1, &rows, dim, 1, rows, dim, 1, rows, FFTformat, batch);
      //cufftPlan1d ( planR2C, rows, FFTformatR2C, batch);
    }else{
      cufftPlan2d ( plan, rows, cols, FFTformat);
      cufftPlan2d ( planR2C, rows, cols, FFTformatR2C);
    }
    cols_fft = cols;
    rows_fft = rows;
    batch_fft = batch;
  }
}
void createPlan(void** handle, int row, int col){ //pass in a pointer, pointing to cufftHandle, in cuda handle is int, in hip, handle is pointer, so we use larger storage among them, which is a pointer. so here we pass in pointer to a pointer, and use allocated void* to store int or pointer. Call this: void* handle = NULL; cufftPlan2d(&handle, row, col); after call, handle is a pointer to the handle, either a int or a pointer. 
  cufftPlan2d ((cufftHandle*)handle, row, col, FFTformat);
}
void createPlan1d(void** handle, int n){
  cufftPlan1d((cufftHandle*)handle, n, FFTformat, 1);
}
void destroyPlan(void* handle){
  cufftDestroy(*(cufftHandle*)&handle);
}
void myFFTM(void* handle, cuComplex* in, cuComplex* out){
  myCufftExec((cufftHandle)((long)handle), in, out, CUFFT_FORWARD);
}
void myIFFTM(void* handle, cuComplex* in, cuComplex* out){
  myCufftExec((cufftHandle)((long)handle), in, out, CUFFT_INVERSE);
}
void myFFT(cuComplex* in, cuComplex* out){
  myCufftExec(*plan, in, out, CUFFT_FORWARD);
}
void myIFFT(cuComplex* in, cuComplex* out){
  myCufftExec(*plan, in, out, CUFFT_INVERSE);
}
void myFFTR2C(Real* in, cuComplex* out){
  myCufftExecR2C(*planR2C, in, out);
}

void myFFTC2R(cuComplex* in, Real* out){
  myCufftExecC2R(*planR2C, in, out);
}

cuFuncTemplate(getWindow,(T* object, int shiftx, int shifty, int objrow, int objcol, T *window, bool replace, Real norm),(object,shiftx,shifty,objrow,objcol, window, replace, norm),{
    cudaIdx();
    T tmp = T();
    if(!(x+shiftx >= objrow || y+shifty >= objcol || x+shiftx < 0 || y+shifty < 0)) tmp =  multiply_dev(object[(x+shiftx)*objcol+y+shifty],norm);
    if(!replace) tmp = add(tmp,window[index]);
    window[index] = tmp;
    })

template void getWindow<Real>(Real*, int, int, int, int, Real*, bool, Real);
template<> void getWindow<complexFormat>(complexFormat* object, int shiftx, int shifty, int objrow, int objcol, complexFormat* window, bool replace, Real norm){
  getWindowWrap<<<numBlocks, threadsPerBlock>>>(addVar(object, shiftx, shifty, objrow, objcol, window, replace, norm));
}

cuFuncTemplate(cudaConvertFO, (T* data, T* out, Real norm),(data,out==0?data:out, norm),{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= (cuda_row*cuda_column)/2) return;
    int x = index/cuda_column;
    int y = index%cuda_column;
    //index = x*cuda_column+y;
    int indexp = (x + (cuda_row >> 1))*cuda_column + (y >= (cuda_column>>1)? y-(cuda_column>>1) : (y+(cuda_column>>1)));
    T tmp = data[index];
    out[index]=multiply_dev(data[indexp],norm);
    out[indexp]=multiply_dev(tmp,norm);
    })
template void cudaConvertFO<Real>(Real*, Real*, Real);
template<> void cudaConvertFO<complexFormat>(complexFormat* data, complexFormat* out, Real norm){
  cudaConvertFOWrap<<<numBlocks, threadsPerBlock>>>(addVar(data, (out==0?data:out), norm));
}
template<typename T1, typename T2>
__global__ void multiplyWrap(int cuda_row, int cuda_column, int cuda_height, T1* store, T1* src, T2* target){
  cuda1Idx3D();
  store[index] = multiply_dev(src[index], target[index]);
}
template <typename T1, typename T2>
void multiply(T1* store, T1* src, T2* target){
  multiplyWrap<<<numBlocks,threadsPerBlock>>>(cuda_imgsz.x, cuda_imgsz.y, cuda_imgsz.z, store, src, target);
}
template void multiply<Real, Real>(Real*, Real*, Real*);
template<> void multiply<complexFormat, complexFormat>(complexFormat* store, complexFormat* src, complexFormat* target){
  multiplyWrap<<<numBlocks,threadsPerBlock>>>(cuda_imgsz.x, cuda_imgsz.y, cuda_imgsz.z, store, src, target);
};
template<> void multiply<complexFormat, Real>(complexFormat* store, complexFormat* src, Real* target){
  multiplyWrap<<<numBlocks,threadsPerBlock>>>(cuda_imgsz.x, cuda_imgsz.y, cuda_imgsz.z, store, src, target);
};

void innerProd(Real* store, complexFormat* src, complexFormat* target){
  E2c(store, src, target, x.x*y.x + x.y*y.y);
}

cuFuncTemplate(rotate90, (T* data, T* out, bool clockwise),(data,out==0?data:out,clockwise),{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if((index+1)*4 > (cuda_row*cuda_column)) return;
    int rowp = cuda_row/2;
    int x = index/rowp;
    int y = index%rowp;
    int index0 = x*cuda_column+y;
    int indexp1 = y*cuda_column+cuda_column-x-1;
    int indexp2 = (cuda_column-x-1)*cuda_column+cuda_column-y-1;
    int indexp3 = (cuda_column-y-1)*cuda_column+x;
    T tmp = data[index0];
    if(clockwise){
    out[index0]=data[indexp1];
    out[indexp1]=data[indexp2];
    out[indexp2]=data[indexp3];
    out[indexp3]=tmp;
    }else{
    out[index0]=data[indexp3];
    out[indexp3]=data[indexp2];
    out[indexp2]=data[indexp1];
    out[indexp1]=tmp;
    }
})
template void rotate90<Real>(Real*, Real*, bool);
template<> void rotate90<complexFormat>(complexFormat* data, complexFormat* out, bool clockwise){
  rotate90Wrap<<<numBlocks, threadsPerBlock>>>(addVar(data, (out==0?data:out), clockwise));
}

cuFuncTemplate(rotate, (T* data, T* out, Real angle),(data,out,angle),{
    cudaIdx();
    Real xp, yp;
    Real c, s;
    sincosf(angle, &s, &c);
    xp = c*(x-cuda_row/2)-s*(y-cuda_column/2) + cuda_row/2;
    yp = c*(y-cuda_column/2)+s*(x-cuda_row/2) + cuda_column/2;
    int xpi = floor(xp);
    int ypi = floor(yp);
    if(xpi < 0 || ypi < 0 || xpi+1 >= cuda_row || ypi+1 >= cuda_column) {
    out[index] = 0;
    return;
    }
    xp -= xpi;
    yp -= ypi;
    int indexp = xpi*cuda_column + ypi;
    out[index] = data[indexp]*(1-xp)*(1-yp) + data[indexp+1]*(1-xp)*yp + data[indexp+cuda_column]*xp*(1-yp) + data[indexp+cuda_column+1]*xp*yp;
    })
template void rotate<Real>(Real*, Real*, Real);
template<> void rotate<complexFormat>(complexFormat* data, complexFormat* out, Real clockwise){
  rotate90Wrap<<<numBlocks, threadsPerBlock>>>(addVar(data, (out==0?data:out), clockwise));
}

cuFuncTemplate(transpose, (T* data, T* out),(data,out==0?data:out),{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= (cuda_row*cuda_column)/2) return;
    int indexp = cuda_row*cuda_column - index - 1;
    T tmp = data[index];
    out[index]=data[indexp];
    out[indexp]=tmp;
    })
template void transpose<Real>(Real*, Real*);
template<> void transpose<complexFormat>(complexFormat* data, complexFormat* out){
  transposeWrap<<<numBlocks, threadsPerBlock>>>(addVar(data, (out==0?data:out)));
}

cuFuncTemplate(flipx, (T* data, T* out),(data,out==0?data:out),{
    cudaIdx()
    if(index >= (cuda_row*cuda_column)/2) return;
    int indexp = (cuda_row-x-1)*cuda_column + y;
    T tmp = data[index];
    out[index]=data[indexp];
    out[indexp]=tmp;
    })
template void flipx<Real>(Real*, Real*);
template<> void flipx<complexFormat>(complexFormat* data, complexFormat* out){
  flipxWrap<<<numBlocks, threadsPerBlock>>>(addVar(data, (out==0?data:out)));
}

cuFuncTemplate(flipy, (T* data, T* out),(data,out==0?data:out),{
    cuda1Idx()
    if(index >= (cuda_row*cuda_column)/2) return;
    int y = index%(cuda_column>>1);
    int x = index/(cuda_column>>1);
    int indexp1 = x*cuda_column + y;
    int indexp2 = x*cuda_column + cuda_column-y-1;
    T tmp = data[indexp1];
    out[indexp1]=data[indexp2];
    out[indexp2]=tmp;
    })
template void flipy<Real>(Real*, Real*);
template<> void flipy<complexFormat>(complexFormat* data, complexFormat* out){
  flipyWrap<<<numBlocks, threadsPerBlock>>>(addVar(data, (out==0?data:out)));
}

template <typename T1, typename T2>
__global__ void assignValWrap(int cuda_row, int cuda_column, int cuda_height, T1* out, T2* input){
  cuda1Idx3D()
    out[index] = input[index];
}
template <typename T1, typename T2>
void assignVal(T1* out, T2* input){
  assignValWrap<<<numBlocks,threadsPerBlock>>>(cuda_imgsz.x, cuda_imgsz.y, cuda_imgsz.z, out, input);
}
template void assignVal<Real,Real>(Real*, Real*);
template void assignVal<Real,double>(Real*, double*);
template void assignVal<Real,int>(Real*, int*);
template void assignVal<Real,uint16_t>(Real*, uint16_t*);
template<> void assignVal<complexFormat, complexFormat>(complexFormat* out, complexFormat* input){
  assignValWrap<<<numBlocks,threadsPerBlock>>>(cuda_imgsz.x, cuda_imgsz.y, cuda_imgsz.z, out,input);
}

cuFuncTemplate(crop,(T* src, T* dest, int row, int col, Real midx, Real midy),(src,dest,row,col,midx,midy),{
    cudaIdx()
    int shiftx = int(row*midx);
    if(shiftx + cuda_row/2 > row/2) shiftx = (row-cuda_row)/2;
    else if(shiftx - cuda_row/2 < - row/2) shiftx = (cuda_row-row)/2;
    int shifty = int(col*midy);
    if(shifty + cuda_column/2 > col/2) shifty = (row-cuda_column)/2;
    else if(shifty - cuda_column/2 < - col/2) shifty = (cuda_column-col)/2;
    int targetindex = (x+(row-cuda_row)/2+shiftx)*col + y+(col-cuda_column)/2+shifty;
    dest[index] = src[targetindex];
    })
template void crop<Real>(Real*, Real*, int, int, Real, Real);
template<> void crop<complexFormat>(complexFormat* src, complexFormat* dest, int row, int col, Real midx, Real midy){
  cropWrap<<<numBlocks, threadsPerBlock>>>(addVar(src, dest, row, col, midx, midy));
}
template<> void setValue<Real>(Real* data, Real value){
  E1r(data, data, value);
}
template<> void setValue<complexFormat>(complexFormat* data, complexFormat value){
  cuComplex v = *&value;
  E1c(data, data, v);
}


cuFunc(multiplyShift,(complexFormat* object, Real shiftx, Real shifty),(object,shiftx,shifty),{
    cudaIdx();
    cuComplex tmp;
    sincosf(-2*M_PI*(shiftx*(x-cuda_row/2)/cuda_row+shifty*(y-cuda_column/2)/cuda_column), &tmp.y, &tmp.x);
    object[index] *= tmp;
    })

void regularize(complexFormat* object, Real sigma){
  EL1(object, object,
    [=]__device__(cuComplex x)->cuComplex{
      Real mod = cabs(x);
      if(mod <= sigma) return cuComplex();
      Real factor = 1 - sigma/mod;
      return make_float2(x.x*factor, x.y*factor);
    });
}

cuFunc(applyGaussMult,(complexFormat* input, complexFormat *output, Real sigma, bool isFreq),(input,output, sigma,isFreq),{
    cudaIdx()
    Real xrel, yrel;
    if(isFreq){
    if(x>=cuda_row/2) xrel=x-cuda_row;
    else xrel=x;
    if(y>=cuda_column/2) yrel=y-cuda_column;
    else yrel=y;
    }else{
    xrel = x - cuda_row/2;
    yrel = y - cuda_column/2;
    }
    Real factor = exp(-(xrel*xrel+yrel*yrel)/(2*sigma*sigma));
    output[index].x=factor*input[index].x;
    output[index].y=factor*input[index].y;
    })

void applyThreshold(complexFormat* input, complexFormat* output, Real threshold){
  EL1(output, input,
    [=]__device__(cuComplex x)->cuComplex{
      Real mod2 = x.x*x.x + x.y*x.y;
      if(mod2 <= threshold) return x;
      Real factor = sqrtf(threshold/mod2);
      return make_float2(x.x*factor, x.y*factor);
    });
}


cuFunc(applyGaussMult,(Real* input, Real *output, Real sigma, bool isFreq, Real x0 = 0, Real y0 = 0),(input,output,sigma,isFreq,x0,y0),{
    cudaIdx()
    Real xrel, yrel;
    if(isFreq){
    if(x>=cuda_row/2) xrel=x-cuda_row;
    else xrel=x;
    if(y>=cuda_column/2) yrel=y-cuda_column;
    else yrel=y;
    }else{
    xrel = x - cuda_row/2-x0;
    yrel = y - cuda_column/2-y0;
    }
    Real factor = exp(-(xrel*xrel+yrel*yrel)/(2*sigma*sigma));
    output[index]=factor*input[index];
    })

void applyNorm(complexFormat* data, complexFormat factor){
  E1c(data, data, x * factor);
}
void applyNorm(complexFormat* data, Real factor){
  E1c(data, data, x*factor);
}
void applyNorm(Real* data, Real factor){
  E1r(data, data, x*factor);
}

void invert(Real* data){
  E1r(data, data, 1-x);
}

void rotateToReal(complexFormat* data){
  E1c(data, data, make_float2(cabs(x), 0));
}

__global__ void createGaussKernel(Real* data, int sz, Real sigma){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= sz*sz) return;
  int dx = idx/sz-(sz>>1);
  int dy = (idx%sz)-(sz>>1);
  data[idx] = 1./(2*M_PI*sigma*sigma)*exp(-Real(dx*dx+dy*dy)/(2*sigma*sigma));
}

void createGauss(Real* data, int sz, Real sigma){
  createGaussKernel<<<(sz*sz-1)/threadsPerBlock.x+1,threadsPerBlock>>>(data, sz, sigma);
}
cuFunc(fillRedundantR2C,(complexFormat* data, complexFormat* dataout, Real factor),(data,dataout,factor),{
    cudaIdx()
    int targetIndex = x*(cuda_column/2+1)+y;
    if(y <= cuda_column/2) {
    dataout[index].x = data[targetIndex].x*factor;
    dataout[index].y = data[targetIndex].y*factor;
    return;
    }
    if(x==0) {
    targetIndex = cuda_column-y;
    }else{
    targetIndex = (cuda_row-x)*(cuda_column/2+1)+cuda_column-y;
    }
    dataout[index].x = data[targetIndex].x*factor;
    dataout[index].y = -data[targetIndex].y*factor;
    })

cuFuncShared(applyConvolution,(Real *input, Real *output, Real* kernel, int kernelwidth, int kernelheight),
    (input,output,kernel,kernelwidth,kernelheight),
    {
    extern __shared__ float tile[];
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int blockindex = threadIdx.x*blockDim.y+threadIdx.y;
    int tilewidth = kernelwidth*2+blockDim.x;
    int tilesize = tilewidth*tilewidth;
    int blocksize = blockDim.x*blockDim.y;
    for(int filltarget = blockindex; filltarget < tilesize; filltarget+=blocksize){
    int fillx = x - threadIdx.x - kernelwidth + filltarget/tilewidth;
    int filly = y - threadIdx.y - kernelwidth + filltarget%tilewidth;
    if(fillx < 0 || filly < 0 || filly >= cuda_column || fillx >= cuda_row) tile[filltarget] = 0;
    else {
    tile[filltarget] = input[fillx*cuda_column+filly];
    }
    }
    if(blockindex < (2*kernelwidth+1)*(2*kernelheight+1)) tile[tilesize+blockindex] = kernel[blockindex];
    if(x >= cuda_row || y >= cuda_column) return;
    int index = x*cuda_column+y;
    __syncthreads();
    int Idx = (threadIdx.x)*(tilewidth) + threadIdx.y;
    int IdxK = 0;
    Real n_output = 0;
    for(int x1 = -kernelwidth; x1 <= kernelwidth; x1++){
      for(int y1 = -kernelheight; y1 <= kernelheight; y1++){
        n_output+=tile[Idx++]*tile[tilesize+IdxK++];
      }
      Idx+=tilewidth-2*kernelheight-1;
    }
    output[index] = n_output;
    })

void interpolate(Real* out, Real* data0, Real* data1, Real dx){
  E2r(out, data0, data1, x*(1-dx) + y*dx);
}
void interpolate(complexFormat* out, complexFormat* data0, complexFormat* data1, Real dx){
  E2c(out, data0, data1,
    make_float2(x.x*(1-dx)+y.x*dx, x.y*(1-dx)+y.y*dx));
}
void adamUpdateV(Real* v, Real* grad, Real beta2){
  E2r(v, v, grad, y*y*(1-beta2) + beta2*x);
}
void adamUpdateV(Real* v, complexFormat* grad, Real beta2){
  EL2(v, v, grad,
    [=]__device__(Real x, cuComplex y)->Real{ return y.x*y.x*(1-beta2) + beta2*x; });
}
void adamUpdate(complexFormat* xn, complexFormat* m, Real* v, Real lr, Real eps){
  EL3(xn, xn, m, v,
    [=]__device__(cuComplex x, cuComplex y, Real z)->cuComplex{
      return make_float2(x.x + lr*y.x/(sqrtf(z)+eps), x.y);
    });
}
void ceiling(complexFormat* data, Real ceilval){
  EL1(data, data,
    [=]__device__(cuComplex x)->cuComplex{
      Real factor = ceilval/cabs(x);
      if(factor > 1) return x;
      return make_float2(x.x*factor, x.y*factor);
    });
}
void multiplyReal(Real* store, complexFormat* src, complexFormat* target){
  E2c(store, src, target, x.x*y.x);
}

cuFunc(divide_sqrt,(Real* store, complexFormat* src, Real* target, Real* bs),(store,src,target, bs),{
    cuda1Idx();
    store[index] = (bs && bs[index]>0.5)?0:(cabs(src[index])*rsqrt(target[index]+1e-6));
    })

cuFunc(sqrtdivide,(Real* store, Real* src, Real* target, Real* bs),(store,src,target, bs),{
    cuda1Idx();
    Real ret = 0;
    if((bs && bs[index]>0.5) || (target[index] > 0.99 && src[index] > 0.99)) ret = 1;
    else if(target[index] > 0) {
      ret = sqrtf(target[index]/max(src[index],1e-6));
    }
    store[index] = ret;
    })

void multiplyConj(complexFormat* store, complexFormat* src, complexFormat* target){
  E2c(store, src, target, x * conj(y));
}
void multiplyConj(complexFormat* store, complexFormat* src, complexFormat target){
  E1c(store, src, x * conj(target));
}

cuFunc(applyTransmission,(complexFormat* out,Real* sum_d,Real* sum_b,double* lambdas, Real lambda_ref),(out,sum_d,sum_b,lambdas, lambda_ref),{
  cudaIdx()
  Real k=2*M_PI/(lambdas[x]*lambda_ref);
  out[index]=make_polar(expf(-k*sum_b[index]), -k*sum_d[index]);
})

void computeTransmission(complexFormat* out,Real* maps,Real* deltas,Real* betas,double* lambdas,int ne,int nl, int npix, Real lambda_ref, Real thickness){
  myCuDMalloc(Real, sum_d,npix*nl);
  myCuDMalloc(Real, sum_b,npix*nl);
  cublasHandle_t handle;
  cublasCreate(&handle);
  const float beta = 0.0f;
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, npix, nl, ne, &thickness, maps, npix, deltas, nl, &beta, sum_d, npix);
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, npix, nl, ne, &thickness, maps, npix, betas, nl, &beta, sum_b, npix);
  cublasDestroy(handle);
  resize_cuda_image(nl, npix);
  applyTransmission(out,sum_d,sum_b,lambdas, lambda_ref);
  myCuFree(sum_d);
  myCuFree(sum_b);
}

void multiplyRegular(complexFormat* store, complexFormat* src, complexFormat* target){
  E2c(store, src, target, x * y);
}

void forcePositive(complexFormat* a){
  E1c(a, a, make_float2(max(x.x, 0.0f), 0.0f));
}

void forcePositive(Real* a){
  E1r(a, a, max(x, 0.0f));
}

template<typename T1, typename T2>
__forceinline__ __device__ auto sub(T1 a, T2 b) { return a - b; }
__forceinline__ __device__ float2 sub(float2 a, float2 b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

template<typename T1, typename T2>
__forceinline__ __device__ auto mul(T1 a, T2 b) { return a * b; }
__forceinline__ __device__ float2 mul(float2 a, float s) {
    return make_float2(a.x * s, a.y * s);
}

template<typename T>
__forceinline__ __device__ T bilinearInterpolate(const T* img, int w, int h, float x, float y) {
    x = min(max(x, 0.0f), static_cast<float>(w - 1));
    y = min(max(y, 0.0f), static_cast<float>(h - 1));

    int x1 = static_cast<int>(x);
    int y1 = static_cast<int>(y);
    int x2 = x1 + 1;
    int y2 = y1 + 1;

    x2 = min(x2, w - 1);
    y2 = min(y2, h - 1);

    float dx = x - x1;
    float dy = y - y1;

    T q11 = img[y1 * w + x1];
    T q12 = img[y2 * w + x1];
    T q21 = img[y1 * w + x2];
    T q22 = img[y2 * w + x2];

    T r1 = add(q11, mul(sub(q21, q11), dx)); // q11 + (q21 - q11)*dx
    T r2 = add(q12, mul(sub(q22, q12), dx)); // q12 + (q22 - q12)*dx

    return add(r1, mul(sub(r2, r1), dy));     // r1 + (r2 - r1)*dy
}

cuFuncTemplate(
    resize,
    (const T* input, T* output, int in_width, int in_height),
    (input, output, in_width, in_height),
{
    cudaIdx(); // Already defines x, y and exits early if out-of-bounds

    float scale_x = static_cast<float>(in_width) / cuda_column;
    float scale_y = static_cast<float>(in_height) / cuda_row;

    float px = (y + 0.5f) * scale_x - 0.5f; // mapped x in input
    float py = (x + 0.5f) * scale_y - 0.5f; // mapped y in input

    output[x * cuda_column + y] = bilinearInterpolate<T>(input, in_width, in_height, px, py);
})
template void resize<Real>(const Real*, Real*, int, int);
template<> void resize<complexFormat>(const complexFormat* input, complexFormat* output, int inw, int inh){
  resizeWrap<<<numBlocks, threadsPerBlock>>>(addVar(input, output, inw, inh));
};

cuFunc(multiplyPropagatePhase,(complexFormat* amp, Real a, Real b),(amp,a,b),{ // a=z/lambda, b = (lambda/s)^2, s is the image size
    cudaIdx();
    cuComplex phasefactor;
    Real square = 1-(sq(x-(cuda_row>>1))+sq(y-(cuda_column>>1)))*b;
    if(square < 0) {
      amp[index] = cuComplex(); // ignore evanescent wave
      return;
    }
    Real phase = a*sqrtf(square);
    sincosf(phase, &phasefactor.y, &phasefactor.x);
    amp[index] *= phasefactor;
    })

cuFunc(obliqueRotatePhase,(complexFormat* input, complexFormat* output, Real a, Real b, Real sinT, Real cosT, int sign),(input,output,a,b,sinT,cosT,sign),{ // a=2pi*D/lambda, b=(lambda/s)^2, sign=+1 fwd, -1 bwd
    // x = row (vertical, ky), y = column (horizontal, kx). Tilt around y-axis.
    // Surface normal: n_hat = (sinT, 0, cosT). Surface x-axis: (cosT, 0, sinT).
    // FWD: output in surface frame (k_parallel, ky), gather from pupil frame (kx, ky)
    // BWD: output in pupil frame (kx, ky), gather from surface frame (k_parallel, ky)
    cudaIdx();
    Real sqrtb = sqrtf(b);
    Real kx_out = (y - (cuda_column>>1)) * sqrtb;
    Real ky_out = (x - (cuda_row>>1)) * sqrtb;

    Real kx_src, kz_out;
    if(sign > 0){
      // FWD: inverse rotation (surface -> lab): kx = k_parallel*cosT - kz'*sinT
      Real disc = 1.0f - kx_out*kx_out - ky_out*ky_out;
      if(disc < 0){ output[index] = cuComplex(); return; }
      kz_out = sqrtf(disc);
      kx_src = kx_out * cosT - kz_out * sinT;
    } else {
      // BWD: forward rotation (lab -> surface): k_parallel = kx*cosT + kz*sinT
      Real kz_sq = 1.0f - kx_out*kx_out - ky_out*ky_out;
      if(kz_sq < 0){ output[index] = cuComplex(); return; }
      kz_out = sqrtf(kz_sq);
      kx_src = kx_out * cosT + kz_out * sinT;
    }
    Real ky_src = ky_out;

    // Bilinear interpolation in kx (y-direction only, ky unchanged)
    Real y_src = kx_src / sqrtb + (cuda_column>>1);
    int y0 = (int)floorf(y_src);
    int y1 = y0 + 1;
    if(y0 < 0 || y1 >= cuda_column){
      output[index] = cuComplex();
      return;
    }
    Real frac = y_src - y0;
    cuComplex c0 = input[x*cuda_column + y0];
    cuComplex c1 = input[x*cuda_column + y1];
    Real re = (1-frac)*c0.x + frac*c1.x;
    Real im = (1-frac)*c0.y + frac*c1.y;

    // Source kz (lab frame)
    Real kz_src_sq = 1.0f - kx_src*kx_src - ky_src*ky_src;
    if(kz_src_sq < 0){ output[index] = cuComplex(); return; }
    Real kz_src = sqrtf(kz_src_sq);

    // Propagation phase: D*kz
    Real phase = sign * a * kz_src;

    // Jacobian: |dkx/dk_parallel| = kz / (kz*cosT - kx*sinT)
    Real knorm = kz_src * cosT - kx_src * sinT; // k dot n_hat
    Real weight;
    if(sign > 0)
      weight = kz_src / (knorm + 1e-12f);
    else
      weight = (knorm + 1e-12f) / (kz_src + 1e-12f);

    cuComplex phasefactor;
    sincosf(phase, &phasefactor.y, &phasefactor.x);
    output[index] = cuComplex(re, im) * weight * phasefactor;
    })

cuFunc(stretchX,(complexFormat* input, complexFormat* output, Real cosT),(input,output,cosT),{
    // x = row (vertical), y = column (horizontal). Stretch in y (horizontal).
    cudaIdx();
    Real cosT_inv = 1.0f / cosT;
    Real y_src = (y - (cuda_column>>1)) * cosT + (cuda_column>>1);
    int y0 = (int)floorf(y_src);
    int y1 = y0 + 1;
    if(y0 < 0 || y1 >= cuda_column){
      output[index] = cuComplex();
    } else {
      Real frac = y_src - y0;
      cuComplex c0 = input[x*cuda_column + y0];
      cuComplex c1 = input[x*cuda_column + y1];
      Real re = ((1-frac)*c0.x + frac*c1.x) * cosT_inv;
      Real im = ((1-frac)*c0.y + frac*c1.y) * cosT_inv;
      output[index] = cuComplex(re, im);
    }
    })

void extendToComplex(Real* a, complexFormat* b){
  E1r(b, a, make_float2(x, 0));
}

cuFunc(createWaveFront,(Real* d_intensity, Real* d_phase, complexFormat* objectWave, int row, int col, int shiftx, int shifty, Real phaseFactor),(d_intensity,d_phase,objectWave,row,col,shiftx,shifty,phaseFactor),{
    cudaIdx()
    int marginx = (cuda_row-row)/2+shiftx;
    int marginy = (cuda_column-col)/2+shifty;
    if(x<marginx || x >= marginx+row || y < marginy || y >= marginy+col){
    objectWave[index].x = objectWave[index].y = 0;
    return;
    }
    int targetindex = (x-marginx)*col + y-marginy;
    Real phase = phaseFactor*sqSum(x-marginx-(row>>1),y-marginy-(col>>1));
    Real mod = d_intensity?sqrtf(max(0.,d_intensity[targetindex])):1;
    if(d_phase){
    phase += (d_phase[targetindex]-0.5)*2*M_PI;
    }
    if(phase){
    objectWave[index] = make_polar(mod, phase);
    }else{
    objectWave[index].x = mod;
    objectWave[index].y = 0;
    }
})

cuFunc(createWaveFront,(Real* d_intensity, Real* d_phase, complexFormat* objectWave, Real oversampling, Real shiftx, Real shifty, Real phaseFactor),(d_intensity,d_phase,objectWave,oversampling,shiftx,shifty,phaseFactor),{
    cudaIdx()
    Real marginratio = (1-1./oversampling)/2;
    int marginx = (marginratio+shiftx)*cuda_row;
    int marginy = (marginratio+shifty)*cuda_column;
    if(x<marginx || x >= cuda_row-marginx || y < marginy || y >= cuda_column-marginy){
    objectWave[index].x = objectWave[index].y = 0;
    return;
    }
    int row = ceil(cuda_row/oversampling);
    int col = ceil(cuda_column/oversampling);
    int targetindex = (x-marginx)*col + y-marginy;
    Real phase = phaseFactor*sqSum(x-marginx-(row>>1),y-marginy-(col>>1));
    Real mod = d_intensity?sqrtf(max(0.,d_intensity[targetindex])):1;
    if(d_phase) phase += (d_phase[targetindex]-0.5)*2*M_PI;
    //Real phase = d_phase? (d_phase[targetindex]-0.5) : 0;
    if(phase){
    objectWave[index] = make_polar(mod, phase);
    }else{
    objectWave[index].x = mod;
    objectWave[index].y = 0;
    }
})

cuFunc(initRand,(void* state, unsigned long long seed),(state,seed),{
    cuda1Idx()
    curand_init(seed,index,0,(curandStateMRG32k3a*)state+index);
    })

cuFunc(randMask,(char* mask, void* state, Real ratio),
    (mask, state, ratio),{
    cuda1Idx()
    mask[index]=curand_uniform((curandStateMRG32k3a*)state+index)>ratio;
    })

void getMod(Real* mod, complexFormat* amp){
  E1c(mod, amp, cabs(x));
}
void getMod(Real* mod, Real* amp){
  E1r(mod, amp, fabsf(x));
}
void getReal(Real* mod, complexFormat* amp, Real norm){
  E1c(mod, amp, x.x*norm);
}
void addReal(Real* mod, complexFormat* amp, Real norm){
  EL2(mod, mod, amp,
    [=]__device__(Real x, cuComplex y)->Real{ return x + y.x*norm; });
}
void getImag(Real* mod, complexFormat* amp){
  E1c(mod, amp, x.y);
}
void assignReal(Real* mod, complexFormat* amp){
  EL2(amp, amp, mod,
    []__device__(cuComplex x, Real y)->cuComplex{ return make_float2(y, x.y); });
}
void assignImag(Real* mod, complexFormat* amp){
  EL2(amp, amp, mod,
    []__device__(cuComplex x, Real y)->cuComplex{ return make_float2(x.x, y); });
}
void getMod2(Real* mod2, complexFormat* amp, Real norm){
  E1c(mod2, amp, (x.x*x.x + x.y*x.y)*norm);
}
void getArg(Real* mod2, complexFormat* amp){
  E1c(mod2, amp, atan2f(x.x, x.y));
}
void getMod2(complexFormat* mod2, complexFormat* amp, Real norm){
  E1c(mod2, amp, make_float2((x.x*x.x + x.y*x.y)*norm, 0));
}
void addMod2(Real* mod2, complexFormat* amp, Real norm){
  EL2(mod2, mod2, amp,
    [=]__device__(Real x, cuComplex y)->Real{ return x + (y.x*y.x + y.y*y.y)*norm; });
}
void addMod2(complexFormat* in, complexFormat* out, Real* mod2, Real norm){
  EL2(out, in, mod2,
    [=]__device__(cuComplex x, Real y)->cuComplex{
      Real omod2 = x.x*x.x + x.y*x.y;
      if(omod2 <= -y*norm) return cuComplex();
      Real factor = sqrtf(1 + y*norm/omod2);
      return make_float2(factor*x.x, factor*x.y);
    });
}
void getMod2(Real* mod2, Real* mod, Real norm){
  E1r(mod2, mod, x*x*norm);
}

void applyThreshold(Real* store, Real* input, Real threshold){
  E1r(store, input, x > threshold ? x : 0);
}

void linearConst(Real* store, Real* data, Real fact, Real shift){
  E1r(store, data, fact*x + shift);
}

void linearConst(complexFormat* store, complexFormat* data, complexFormat fact, complexFormat shift){
  E1c(store, data, fact*x + shift);
}

void addProbability(Real* data, Real* value, Real norm, Real sigma){
  EL2(data, data, value,
    [=]__device__(Real x, Real y)->Real{
      Real v = y*norm;
      if(fabsf(x+v) > fabsf(x)) return x + v*gaussian(x+v, 0, sigma);
      return x+v;
    });
}

cuFunc(applyModAbs,(complexFormat* source, Real* target, void* state),(source, target, state),{
    cuda1Idx();
    Real mod = hypot(source[index].x, source[index].y);
    Real rat = target[index];
    if(rat > 0) rat = sqrtf(rat);
    else rat = 0;
    if(mod==0) {
    Real randphase = state?curand_uniform((curandStateMRG32k3a*)state + index)*2*M_PI:0;
    source[index] = make_polar(rat, randphase);
    return;
    }
    rat /= mod;
    source[index].x *= rat;
    source[index].y *= rat;
    })
cuFunc(applyModAbsinner,(complexFormat* source, Real* target,  int row, int col, Real norm, void* state),(source,target,row,col,norm, state),{
    cudaIdx()
    int targetx = x >= cuda_row/2 ? x - (cuda_row - row) : x;
    int targety = y >= cuda_column/2 ? y - (cuda_column - col) : y;
    Real rat = target[index]*norm;
    index = targetx*col+targety;
    Real mod = hypot(source[index].x, source[index].y);
    if(rat > 0) rat = sqrtf(rat);
    else rat = 0;
    if(mod==0) {
    Real randphase = state?curand_uniform((curandStateMRG32k3a*)state+index)*2*M_PI:0;
    source[index] = make_polar(rat, randphase);
    return;
    }
    rat /= mod;
    source[index].x *= rat;
    source[index].y *= rat;
    })

cuFunc(convertFOPhase,(complexFormat* data, Real norm),(data, norm),{
    cudaIdx();
    if((x+y)%2==1) {
    norm = -norm;
    }else if(norm == 1){
    return;
    }
    data[index].x *= norm;
    data[index].y *= norm;
    })
void add(complexFormat* a, complexFormat* b, Real c){
  E2c(a, a, b,
    make_float2(x.x + y.x*c, x.y + y.y*c));
}
void addPhase(complexFormat* a, complexFormat* b, Real phi0){
  EL2(a, a, b,
    [=]__device__(cuComplex x, cuComplex y)->cuComplex{
      Real mod = cabs(y);
      if(mod < 1e-6) return x;
      Real r = y.x/mod, i = y.y/mod, s, c;
      sincosf(phi0, &s, &c);
      return make_float2(x.x + r*c - i*s, x.y + r*s + i*c);
    });
}
void add(complexFormat* store, complexFormat* a, complexFormat* b, Real c){
  E2c(store, a, b, make_float2(x.x + y.x*c, x.y + y.y*c));
}
void normAdd(complexFormat* store, complexFormat* a, complexFormat* b, Real c, Real d){
  E2c(store, a, b, make_float2(x.x*c + y.x*d, x.y*c + y.y*d));
}
void add(Real* a, Real* b, Real c){
  E2r(a, a, b, x + y*c);
}
void add(Real* a, Real* b, Real c, Real d){
  E2r(a, a, b, x*c + y*d);
}

void logorithm(Real* in, Real* out, Real k, Real b){
  E1r(out, in, k*logf(x) + b);
}

void power(Real* in, Real* out, Real p, Real k){
  E1r(out, in, k*powf(x, p));
}

void add(Real* store, Real* a, Real* b, Real c){
  E2r(store, a, b, x + y*c);
}
void addRemoveOE(Real* src, Real* sub, Real mult){
  E2r(src, src, sub, y < 0.99 ? x + y*mult : 0);
}
void addRemoveOE(Real* store, Real* src, Real* sub, Real mult){
  E2r(store, src, sub, y < 0.99 ? x + y*mult : 0);
}

cuFunc(drawCircle,(complexFormat* img, Real cx, Real cy, Real r, Real w, complexFormat data),(img,cx,cy,r,w, *&data),{
    cudaIdx()
    Real d2 = sq(cx-x) + sq(cy-y);
    if(d2 > r*r && d2 < (r+w)*(r+w)){
      img[index] = data;
    }
    })

cuFunc(drawCircle,(Real* img, Real cx, Real cy, Real r, Real w, Real data),(img,cx,cy,r,w, data),{
    cudaIdx()
    Real d2 = sq(cx-x) + sq(cy-y);
    if(d2 > r*r && d2 < (r+w)*(r+w)){
      img[index] = data;
    }
    })


cuFunc(cropinner,(Real* src, Real* dest, int row, int col, Real norm),(src,dest,row,col,norm),{
    cudaIdx()
    int targetx = x >= cuda_row/2 ? x + row - cuda_row : x;
    int targety = y >= cuda_column/2 ? y + col - cuda_column : y;
    int targetidx = targetx * col + targety;
    dest[index] = src[targetidx]*norm;
    })
cuFunc(mergePixel, (Real* output, Real* input, int col, int nmerge),(output, input, col, nmerge),{
    cudaIdx()
    int idx0 = x*nmerge*col+y*nmerge;
    Real out = 0;
    for(int dx = 0; dx < nmerge; dx ++){
    for(int dy = 0; dy < nmerge; dy ++){
    out += input[idx0];
    idx0++;
    }
    idx0+=col-nmerge;
    }
    output[index] = out/(nmerge*nmerge);
    })

cuFunc(cropinner,(complexFormat* src, complexFormat* dest, int row, int col, Real norm),(src,dest,row,col,norm),{
    cudaIdx()
    int targetx = x >= cuda_row/2 ? x + row - cuda_row : x;
    int targety = y >= cuda_column/2 ? y + col - cuda_column : y;
    int targetidx = targetx * col + targety;
    dest[index].x = src[targetidx].x*norm;
    dest[index].y = src[targetidx].y*norm;
    })
cuFunc(padinner,(Real* src, Real* dest, int row, int col, Real norm),(src,dest,row,col,norm),{
    cudaIdx()
    if((x >= row/2 && x < (cuda_row - row/2)) || (y >= col/2 && y < (cuda_column - col/2))){
    dest[index] = 0;
    return;
    }
    int targetx = x >= cuda_row/2 ? x - (cuda_row - row) : x;
    int targety = y >= cuda_column/2 ? y - (cuda_column - col) : y;
    dest[index] = src[targetx*col+targety]*norm;
    })

cuFunc(padinner,(complexFormat* src, complexFormat* dest, int row, int col, Real norm),(src, dest, row, col, norm),{
    cudaIdx()
    if((x >= row/2 && x < (cuda_row - row/2)) || (y >= col/2 && y < (cuda_column - col/2))){
    dest[index].x = dest[index].y = 0;
    return;
    }
    int targetx = x >= cuda_row/2 ? x - (cuda_row - row) : x;
    int targety = y >= cuda_column/2 ? y - (cuda_column - col) : y;
    int targetidx = targetx*col+targety;
    dest[index].x = src[targetidx].x*norm;
    dest[index].y = src[targetidx].y*norm;
    })

cuFunc(paste, (Real* out, Real* in, int colout, int posx, int posy, bool replace, Real norm),(out, in, colout, posx, posy, replace, norm),{
    cudaIdx();
    int tidx = (x+posx)*colout + y + posy;
    if(tidx < 0) return;
    Real data = in[index]*norm;
    if(!replace) data += out[tidx];
    out[tidx] = data>1?1:data;
    })
cuFunc(paste,(complexFormat* out, complexFormat* in, int colout, int posx, int posy, bool replace),(out, in, colout, posx, posy, replace),{
    cudaIdx();
    int tidx = (x+posx)*colout + y + posy;
    cuComplex data = in[index];
    if(!replace) {
    data.x += out[tidx].x;
    data.y += out[tidx].y;
    }
    out[tidx] = data;
    })
void getXYSlice (Real * slice, Real *data, int nx, int ny, int iz){
  myMemcpyD2D(slice, data+nx*ny*iz, nx*ny*sizeof(Real));
};
cuFunc(getXZSlice, (Real * slice, Real *data, int ny, int iy),
    (slice, data, ny, iy), {
    cudaIdx()
    slice[index] = data[y + cuda_column * iy + cuda_column * ny * x];
    });
cuFunc(getYZSlice, (Real * slice, Real *data, int nx, int ix),
    (slice, data, nx, ix), {
    cudaIdx()
    int idx = ix + nx * y + nx * cuda_column * x;
    slice[index] = data[idx];
    })
__forceinline__ __device__ float bilinear(const float* img, int width, int height, float x, float y) {
    x += width>>1;
    y += height>>1;
    int x0 = (int)floorf(x);
    int y0 = (int)floorf(y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    float fx = x - x0;
    float fy = y - y0;
    if (x0 < 0 || x1 >= width || y0 < 0 || y1 >= height)
        return 0.0f;
    int idx0 = x0*height + y0;
    float v00 = img[idx0];
    float v01 = img[idx0+1];
    float v10 = img[idx0+height];
    float v11 = img[idx0+height+1];
    return (1 - fx) * (1 - fy) * v00 +
           (1 - fx) * fy       * v01 +
           fx       * (1 - fy) * v10 +
           fx       * fy       * v11;
}

// Kernel: Cartesian → Polar (each thread = one (r,θ))
cuFunc(cart2polar_kernel,(Real* d_cart, Real* d_polar,int width, int height),(d_cart, d_polar, width, height), {
    cudaIdx()
    Real c,s;
    sincosf(x * (2.0f * M_PI / cuda_row), &s, &c);
    d_polar[index] = bilinear(d_cart, width, height, y*c, y*s);
})

cuFunc(edgeReduce,(Real* out, Real* in, int ny),(out, in, ny),{
    cuda1Idx();
    for(int iy = 0; iy < ny; iy++){
    out[index] += in[index + iy*cuda_row*cuda_column];
    }

    })
//-------colorbar-begin
cuFunc(createColorbar,(complexFormat* output),(output),{
    cudaIdx();
    Real xr = 2*Real(x)/cuda_row - 1;
    Real yr = 2*Real(y)/cuda_column - 1;
    Real mod2 = xr*xr + yr*yr;
    if(mod2 > 1) {
      xr = yr = 0;
    }
    output[index].x = xr;
    output[index].y = yr;
    })
//-------colorbar-end
//-------experimentConfig.cc-begin
// pixsize*pixsize*M_PI/(d*lambda) and 2*d*M_PI/lambda
cuFunc(multiplyPatternPhase_Device,(complexFormat* amp, Real r_d_lambda, Real d_r_lambda),(amp,r_d_lambda,d_r_lambda),{
    cudaIdx()
    Real phase = (sq(x-(cuda_row>>1))+sq(y-(cuda_column>>1)))*r_d_lambda+d_r_lambda;
    cuComplex p;
    sincosf(phase, &p.y, &p.x);
    amp[index] *= p;
    })

cuFunc(multiplyPatternPhaseOblique_Device,(complexFormat* amp, Real r_d_lambda, Real d_r_lambda, Real costheta),(amp,r_d_lambda,d_r_lambda,costheta),{ // pixsize*pixsize*M_PI/(d*lambda) and 2*d*M_PI/lambda and costheta = z/r
    cudaIdx()
    Real phase = (sq((x-(cuda_row>>1)*costheta))+sq(y-(cuda_column>>1)))*r_d_lambda+d_r_lambda;
    cuComplex p;
    sincosf(phase, &p.y, &p.x);
    amp[index] *= p;
    })

cuFunc(multiplyFresnelPhase_Device,(complexFormat* amp, Real phaseFactor),(amp,phaseFactor),{ // pixsize*pixsize*M_PI/(d*lambda) and 2*d*M_PI/lambda
    cudaIdx()
    Real phase = phaseFactor*(sq(x-(cuda_row>>1))+sq(y-(cuda_column>>1)));
    cuComplex p;
    sincosf(phase, &p.y, &p.x);
    if(cabs(amp[index])!=0) amp[index] *= p;
    })

cuFunc(multiplyFresnelPhaseOblique_Device,(complexFormat* amp, Real phaseFactor, Real costheta_r),(amp,phaseFactor,costheta_r),{ // costheta_r = 1./costheta = r/z
    cudaIdx()
    Real phase = phaseFactor*(sq((x-(cuda_row>>1))*costheta_r)+sq(y-(cuda_column>>1)));
    cuComplex p;
    sincosf(phase, &p.y, &p.x);
    if(cabs(amp[index])!=0) amp[index] *= p;
    })

//-------experimentConfig.cc-end


//-------FISTA.cc-------begin
cuFunc(partialx, (Real* b, Real* p), (b,p),{
    cuda1Idx()
    int x = index/cuda_column;
    Real target;
    if(x == cuda_row-1) target = b[index]-b[index%cuda_column];
    else target = b[index]-b[index+cuda_column];
    //if(fabs(target) > 3e-2) target = 0;
    p[index] = target;
    })
cuFunc(partialx,(complexFormat* b, complexFormat* p),(b,p),{
    cuda1Idx()
    int x = index/cuda_column;
    cuComplex target;
    if(x == cuda_row-1) target = b[index];//sub(b[index],b[index%cuda_column]);
    else target = sub(b[index],b[index+cuda_column]);
    //if(fabs(target) > 3e-2) target = 0;
    p[index] = target;
    })
cuFunc(partialy, (Real* b, Real* p), (b,p),{
    cuda1Idx()
    int y = index%cuda_column;
    Real target;
    if(y == cuda_column-1) target = b[index]-b[index-cuda_column+1];
    else target = b[index]-b[index+1];
    //if(fabs(target) > 3e-2) target = 0;
    p[index] = target;
    })
cuFunc(partialy,(complexFormat* b, complexFormat* p),(b,p),{
    cuda1Idx()
    int y = index%cuda_column;
    cuComplex target;
    if(y == cuda_column-1) target = b[index]; //sub(b[index],b[index-cuda_column+1]);
    else target = sub(b[index],b[index+1]);
    //if(fabs(target) > 3e-2) target = 0;
    p[index] = target;
    })
cuFunc(diffMax, (Real* p, Real* q), (p,q),{
    cuda1Idx()
    Real mod = hypot(p[index],q[index]);
    if(mod <= 1) return;
    p[index] /= mod;
    q[index] /= mod;
    })
cuFunc(diffMax,(complexFormat* p, complexFormat* q),(p,q),{
    cuda1Idx()
    Real mod = hypot(p[index].x,q[index].x);
    if(mod > 1) {
    p[index].x /= mod;
    q[index].x /= mod;
    }
    mod = hypot(p[index].y,q[index].y);
    if(mod > 1) {
    p[index].y /= mod;
    q[index].y /= mod;
    }
    })
cuFunc(calcLpq, (Real* out, Real* p, Real* q), (out,p,q),{
    cudaIdx()
    Real tmp = p[index]+q[index];
    if(x >= 1) tmp -= p[index-cuda_column];
    else tmp-=p[index+(cuda_row-1)*cuda_column];
    if(y >= 1) tmp -= q[index-1];
    else tmp-=q[index+cuda_column-1];
    out[index] = tmp;
    })
cuFunc(calcLpq,(complexFormat* out, complexFormat* p, complexFormat* q),(out,p,q),{
    cudaIdx()
    cuComplex tmp = add(p[index],q[index]);
    if(x >= 1) tmp = sub(tmp,p[index-cuda_column]);
    //else tmp=sub(tmp,p[index+(cuda_row-1)*cuda_column]);
    if(y >= 1) tmp = sub(tmp,q[index-1]);
    //else tmp=sub(tmp,q[index+cuda_column-1]);
    out[index] = tmp;
    })
//-------FISTA.cc-------end
//-------monoChromo.cc-------begin

cuFunc(assignRef_d,(complexFormat* wavefront, uint32_t* mmap, complexFormat* rf, int n),(wavefront, mmap, rf, n), {
    cuda1Idx()
    if(index >= n) return;
    rf[index] = wavefront[mmap[index]];
    })
cuFunc(expandRef,(complexFormat* rf, complexFormat* amp, uint32_t* mmap, int row, int col, int row0, int col0),(rf, amp, mmap, row, col, row0, col0),{
    cuda1Idx()
    int idx = mmap[index];
    int x = idx/col0 + (row-row0)/2;
    int y = idx%col0 + (col-col0)/2;
    amp[x*col+y] = rf[index];
    })
cuFunc(expandRef,(complexFormat* rf, complexFormat* amp, uint32_t* mmap, int row, int col, int row0, int col0, complexFormat a),(rf, amp, mmap, row, col, row0, col0, a),{
    cuda1Idx()
    int idx = mmap[index];
    int x = idx/col0 + (row-row0)/2;
    int y = idx%col0 + (col-col0)/2;
    amp[x*col+y] = rf[index] * a;
    })
cuFunc(saveRef,(complexFormat* rf, complexFormat* amp, uint32_t* mmap, int row, int col, int row0, int col0, Real norm),(rf, amp, mmap, row, col, row0, col0, norm),{
    cuda1Idx()
    int idx = mmap[index];
    int x = idx/col0 + (row-row0)/2;
    int y = idx%col0 + (col-col0)/2;
    rf[index].x = amp[x*col+y].x*norm;
    rf[index].y = amp[x*col+y].y*norm;
    })
cuFunc(saveRef_Real,(complexFormat* rf, complexFormat* amp, uint32_t* mmap, int row, int col, int col0, int n, Real norm),(rf, amp, mmap, row, col, col0, n, norm),{
    cuda1Idx()
    if(index >= n) return;
    int idx = mmap[index];
    int x = idx/col0 + (row-cuda_row)/2;
    int y = idx%col0 + (col-col0)/2;
    rf[index].x = amp[x*col+y].x*norm;
    rf[index].y = 0;
    })
//-------monoChromo.cc-------end
//-------holo.cc-------begin
cuFunc(dillate,(complexFormat* data, Real* support, int wid, int hit),(data,support,wid,hit),{
    cudaIdx();
    if(abs(data[index].x) < 0.5 && abs(data[index].y) < 0.5) return;
    int idxp = 0;
    for(int xp = 0; xp < cuda_row; xp++)
    for(int yp = 0; yp < cuda_column; yp++)
    {
    if(abs(xp - x) <= wid && abs(yp-y) <= hit) support[idxp] = 0;
    if(abs(x - xp) > cuda_row/2 || abs(y-yp)>cuda_column/2) support[idxp] = 0;
    idxp++;
    }
    })

void applyModCorr(complexFormat* obj, complexFormat* refer, Real* xcorrelation){
  EL3(obj, obj, refer, xcorrelation,
    []__device__(cuComplex x, cuComplex y, Real z)->cuComplex{
      if(y.x == 0 && y.y == 0) return x;
      Real fact = z/2 - y.x*x.x - y.y*x.y;
      fact /= y.x*y.x + y.y*y.y;
      return make_float2(x.x + fact*y.x, x.y + fact*y.y);
    });
}

void devideStar(complexFormat* obj, complexFormat* refer, complexFormat* xcorrelation){
  EL2(obj, refer, xcorrelation,
    []__device__(cuComplex x, cuComplex y)->cuComplex{
      Real fact = fmaxf(x.x*x.x + x.y*x.y, 1e-4);
      cuComplex prod = y * x;
      return make_float2(prod.x/fact, prod.y/fact);
    });
}

//-------holo.cc-------end
cuFuncTemplate(addMask, (Real* data, T* spt, bool isFrequency),(data,spt,isFrequency),{
    cudaIdx()
    if(isFrequency){
    if(x>=cuda_row/2) x-=cuda_row/2;
    else x+=cuda_row/2;
    if(y>=cuda_column/2) y-=cuda_column/2;
    else y+=cuda_column/2;
    }
    data[index]+=spt->isInside(x,y);
    })
template void addMask<rect>(Real*, rect*, bool isFrequency);
template void addMask<C_circle>(Real*, C_circle*, bool isFrequency);
template void addMask<diamond>(Real*, diamond*, bool isFrequency);
cuFuncTemplate(createMask, (Real* data, T* spt, bool isFrequency),(data,spt,isFrequency),{
    cudaIdx()
    if(isFrequency){
    if(x>=cuda_row/2) x-=cuda_row/2;
    else x+=cuda_row/2;
    if(y>=cuda_column/2) y-=cuda_column/2;
    else y+=cuda_column/2;
    }
    data[index]=spt->isInside(x,y);
    })
template void createMask<rect>(Real*, rect*, bool isFrequency);
template void createMask<C_circle>(Real*, C_circle*, bool isFrequency);
template void createMask<diamond>(Real*, diamond*, bool isFrequency);
cuFuncTemplate(createMaskBar, (Real* data, T* spt, bool isFrequency),(data,spt,isFrequency),{
    cudaIdx()
    if(isFrequency){
    if(x>=cuda_row/2) x-=cuda_row/2;
    else x+=cuda_row/2;
    if(y>=cuda_column/2) y-=cuda_column/2;
    else y+=cuda_column/2;
    }
    data[index]=!spt->isInside(x,y);
    })
template void createMaskBar<rect>(Real*, rect*, bool isFrequency);
template void createMaskBar<C_circle>(Real*, C_circle*, bool isFrequency);
template void createMaskBar<diamond>(Real*, diamond*, bool isFrequency);
void applyMask(Real* data, Real* mask, Real threshold){
  E2r(data, data, mask, y <= threshold ? 0 : x);
}
void applyMask(Real* data, char* mask){
  EL2(data, data, (const char*)mask,
    []__device__(Real x, char y)->Real{
      return y ? x : 0;
    });
}
void applyMask(complexFormat* data, Real* mask, Real threshold){
  EL2(data, data, mask,
    [=]__device__(cuComplex x, Real y)->cuComplex{
      return y <= threshold ? cuComplex() : x;
    });
}
void applyMaskBar(Real* data, Real* mask, Real threshold){
  E2r(data, data, mask, y > threshold ? 0 : x);
}
void applyMaskBar(Real* data, complexFormat* mask, Real threshold){
  EL2(data, data, mask,
    [=]__device__(Real x, cuComplex y)->Real{
      return y.x > threshold ? 0 : x;
    });
}
void applyMaskBar(complexFormat* data, Real* mask, Real threshold){
  EL2(data, data, mask,
    [=]__device__(cuComplex x, Real y)->cuComplex{
      return y > threshold ? cuComplex() : x;
    });
}
cuFunc(zeroEdge,(complexFormat* a, int n),(a,n),{
    cudaIdx()
    if(x<n || x>=cuda_row-n || y < n || y >= cuda_column-n)
    a[index] = cuComplex();
    })
cuFunc(zeroEdge,(Real* a, int n),(a,n),{
    cudaIdx()
    if(x<n || x>=cuda_row-n || y < n || y >= cuda_column-n)
    a[index] = 0;
    })
cuFunc(zeroEdgey,(complexFormat* a, int n),(a,n),{
    cuda1Idx()
    int y = index%cuda_column;
    if(y < n || y >= cuda_column-n)
    a[index] = cuComplex();
    })

cuFunc(ssimMap,(Real* mu1, Real* mu2, Real* sigma1sq, Real* sigma2sq, Real* sigma12, Real C1, Real C2),(mu1, mu2, sigma1sq, sigma2sq, sigma12, C1, C2),{
    cuda1Idx()
    Real mu12 = mu1[index]*mu2[index];
    mu1[index] = (2*mu12+C1)*(2*sigma12[index]+C2)/((sq(mu1[index])+sq(mu2[index])+C1)*(sigma1sq[index]+sigma2sq[index]+C2));
    })

cuFuncTemplate(randZero,(T* src, T* dest, void* state, Real ratio, char step),(src, dest, state, ratio, step),{
    cudaIdx()
    if(x % step != 0 || y % step != 0) return;
    bool isoff = curand_uniform((curandStateMRG32k3a*)state+index)>ratio;
    int tidx;
    for (int i = 0; i < step; i++) {
    for (int j = 0 ; j < step; j++) {
    if(x + i >= cuda_row || y + j >= cuda_column) continue;
    tidx = index + i * cuda_column + j;
    dest[tidx] = isoff ? T() : src[tidx];
    }
    }
    })
template void randZero<Real>(Real*, Real*, void*, Real, char);
template<> void randZero<complexFormat>(complexFormat* src, complexFormat* dest, void* state, Real ratio, char step){
  randZeroWrap<<<numBlocks, threadsPerBlock>>>(addVar(src, dest, state, ratio, step));
};

cuFuncTemplate(pad,(T* src, T* dest, int row, int col, int shiftx, int shifty),(src, dest, row, col, shiftx, shifty),{
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
template void pad<Real>(Real*, Real*, int, int, int, int);
template<> void pad<complexFormat>(complexFormat* src, complexFormat* dest, int row, int col, int shiftx, int shifty){
  padWrap<<<numBlocks, threadsPerBlock>>>(addVar(src, dest, row, col, shiftx, shifty));
};

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
template void refine<Real>(Real*, Real*, int);

cuFunc(multiplyx,(complexFormat* object, Real* out),(object,out),{
    cuda1Idx();
    Real x = index/cuda_column;
    out[index] = cabs(object[index]) * ((x+0.5)/cuda_row-0.5);
    })

cuFunc(multiplyy,(complexFormat* object, Real* out),(object,out),{
    cuda1Idx();
    Real y = index%cuda_column;
    out[index] = cabs(object[index]) * ((y+0.5)/cuda_column-0.5);
    })
cuFunc(multiplyx,(Real* object, Real* out),(object,out),{
    cuda1Idx();
    Real x = index/cuda_column;
    out[index] = object[index] * ((x+0.5)/cuda_row-0.5);
    })

cuFunc(multiplyy,(Real* object, Real* out),(object,out),{
    cuda1Idx()
    Real y = index%cuda_column;
    out[index] = object[index] * ((y+0.5)/cuda_column-0.5);
    })

cuFunc(peakGaussSum,(int* d_peaks, Real* d_energies, Real* d_total, Real sigma, int numPeaks),(d_peaks,d_energies,d_total,sigma,numPeaks),{
    cuda1Idx()
    Real x = index/cuda_column;
    Real y = index%cuda_column;
    d_total[index] = 0;
    for(int i = 0; i < numPeaks; i++){
      int idx = d_peaks[i];
      int py = idx%cuda_column;
      int px = idx/cuda_column;
      Real dx = x-px, dy = y-py;
      Real f2 = dx*dx + dy*dy;
      Real g = d_energies[i]/(f2 + sigma*sigma);
      d_total[index] += g;
    }
    })

cuFunc(peakGaussRatio,(int* d_peaks, Real* d_energies, Real* d_total, Real* d_ratio, Real sigma, int peakIdx),(d_peaks,d_energies,d_total,d_ratio,sigma,peakIdx),{
    cuda1Idx()
    Real x = index/cuda_column;
    Real y = index%cuda_column;
    int idx = d_peaks[peakIdx];
    int py = idx%cuda_column;
    int px = idx/cuda_column;
    Real dx = x-px, dy = y-py;
    Real f2 = dx*dx + dy*dy;
    Real g = d_energies[peakIdx]/(f2 + sigma*sigma);
    d_ratio[index] = (d_total[index] > 0) ? g/d_total[index] : 0;
    })

cuFunc(createPlaneWave,(complexFormat* out, Real peakX, Real peakY, Real amp),(out,peakX,peakY,amp),{
    cudaIdx()
    out[index] = make_polar(amp, 2.0f * M_PI * (peakX * x / cuda_column + peakY * y / cuda_row));
})

//-------GPU JPEG colormap LUT-------
__constant__ uint8_t d_turboLUT[768]; // 256 entries x 3 bytes (RGB)
static bool d_turboLUT_initialized = false;

const float TurboDataGPU[3][8][3] = {
  {{48.628736, 1.26493952, 0.0179480064}, {12.9012, 3.17599, -0.0432176},
   {550.736, -13.3905, 0.0850816}, {-377.035, 4.95565, -0.00546875},
   {-461.663, 6.91765, -0.0158418}, {-904.194, 12.7805, -0.0352175},
   {-397.815, 7.43316, -0.0210902}, {-513.745, 8.50106, -0.0235486}},
  {{18.3699, 2.98435, -0.00590619}, {25.5339, 2.60115, -0.000838337},
   {-65.9885, 5.48243, -0.023616}, {-90.9164, 5.82642, -0.024485},
   {-47.1592, 5.39126, -0.0237502}, {-65.915, 5.26759, -0.0222375},
   {1648.81, -12.6094, 0.0243797}, {1160.21, -8.10854, 0.014018}},
  {{59.4351, 7.57781, -0.0720408}, {29.4288, 9.18296, -0.0932682},
   {391.685, -2.3166, -0.0016649}, {770.071, -8.98098, 0.0266769},
   {606.475, -7.75046, 0.0270971}, {-638.138, 8.68843, -0.0270811},
   {1017.44, -8.76258, 0.0189386}, {625.18, -5.14916, 0.0106199}}
};

void initTurboLUT() {
  if (d_turboLUT_initialized) return;
  uint8_t h_lut[768];
  for (int i = 0; i < 256; i++) {
    Real x = (Real)i;
    int sec = i / 32;
    for (int c = 0; c < 3; c++) {
      int val = (int)(TurboDataGPU[c][sec][0] + TurboDataGPU[c][sec][1]*x + TurboDataGPU[c][sec][2]*x*x);
      if (val < 0) val = 0;
      if (val > 255) val = 255;
      h_lut[i*3+c] = (uint8_t)val;
    }
  }
  gpuErrchk(cudaMemcpyToSymbol(d_turboLUT, h_lut, 768));
  d_turboLUT_initialized = true;
}

cuFunc(applyColorMapGPU, (const pixeltype* input, uint8_t* rgb), (input, rgb), {
    cuda1Idx()
    int idx = (input[index] >> 8) & 0xFF;
    rgb[index*3]   = d_turboLUT[idx*3];
    rgb[index*3+1] = d_turboLUT[idx*3+1];
    rgb[index*3+2] = d_turboLUT[idx*3+2];
    })
