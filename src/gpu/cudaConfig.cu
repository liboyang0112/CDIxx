#include "cudaDefs_h.cu"
#include "cudaConfig.hpp"
#include "fmt/core.h"
#include <complex.h>
#include <curand_kernel.h>
#include <cub_wrap.hpp>
#include <cufft.h>
cudaVars* cudaVar = 0;
cudaVars* cudaVarLocal = 0;
dim3 numBlocks;
const dim3 threadsPerBlock = 256;
complexFormat *cudaData = 0;
static cufftHandle *plan = 0, *planR2C = 0;
int3 cuda_imgsz = {0,0,1};
void cuMemManager::c_malloc(void*& ptr, size_t sz) { gpuErrchk(cudaMalloc((void**)&ptr, sz)); }
void cuMemManager::c_memset(void*& ptr, size_t sz) { gpuErrchk(cudaMemset(ptr, 0, sz)); }
cuMemManager::cuMemManager():memManager(){}
cuMemManager memMngr;
void gpuAssert(int code, const char *file, int line)
{
  if (code != cudaSuccess)
  {
    fmt::println(stderr,"GPUassert: {} {} {}", cudaGetErrorString((cudaError_t)code), file, line);
    abort();
  }
}
int getCudaRows(){
  return cuda_imgsz.x;
}
int getCudaCols(){
  return cuda_imgsz.y;
}
void myMemcpyH2D(void* d, void* s, size_t sz){
  cudaMemcpy(d, s, sz, cudaMemcpyHostToDevice);
}
void myMemcpyD2D(void* d, void* s, size_t sz){
  cudaMemcpy(d, s, sz, cudaMemcpyDeviceToDevice);
}
void myMemcpyD2H(void* d, void* s, size_t sz){
  cudaMemcpy(d, s, sz, cudaMemcpyDeviceToHost);
}
void clearCuMem(void* ptr, size_t sz){
  cudaMemset(ptr, 0, sz);
}

void resize_cuda_image(int rows, int cols, int layers){
  cuda_imgsz.x = rows;
  cuda_imgsz.y = cols;
  cuda_imgsz.z = layers;
  numBlocks.x=(rows*cols*layers-1)/threadsPerBlock.x+1;
}
void init_cuda_image(int rcolor, Real scale){
  const int sz = sizeof(cudaVars);
  if(!cudaVar){
    cudaFree(0);
    cudaVar = (cudaVars*) memMngr.borrowCache(sz);
    myMalloc(cudaVars, cudaVarLocal, 1);
    cudaVarLocal->threshold = 0.5;
    cudaVarLocal->beta_HIO = 1;
    if(scale==scale) cudaVarLocal->scale = scale;
    if(rcolor!=0) cudaVarLocal->rcolor = rcolor;
    else cudaVarLocal->rcolor=65535;
    cudaMemcpy(cudaVar, cudaVarLocal, sz, cudaMemcpyHostToDevice);
    return;
  }
  if(rcolor!=0) cudaVarLocal->rcolor = rcolor;
  if(scale==scale) cudaVarLocal->scale = scale;
  cudaMemcpy(cudaVar, cudaVarLocal, sz, cudaMemcpyHostToDevice);
};
size_t getGPUFreeMem(){
  size_t freeBytes, totalBytes;
  cudaMemGetInfo(&freeBytes, &totalBytes);
  return freeBytes >> 20;
}
void setThreshold(Real val){
  myMemcpyH2D(&cudaVar->threshold, &val, sizeof(cudaVarLocal->threshold));
}
void* newRand(size_t sz){
  return memMngr.borrowCache(sz * sizeof(curandStateMRG32k3a));
}

void gpuerr(){
  gpuErrchk(cudaGetLastError());
}

static int rows_fft, cols_fft, batch_fft;
__device__ __host__ bool rect::isInside(int x, int y){
  if(x >= startx && x < endx && y >= starty && y < endy) return true;
  return false;
}
__device__ __host__ bool C_circle::isInside(int x, int y){
  Real dr = hypot(Real(x-x0),Real(y-y0));
  if(dr < r) return true;
  return false;
}
__device__ __host__ bool diamond::isInside(int x, int y){
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

__device__ __inline__ cuComplex multiply_dev(cuComplex val, cuComplex val2){
  return cuCmulf(val,val2);
}

template<typename T2>
__device__ __inline__ cuComplex multiply_dev(cuComplex val, T2 norm){
  val.x *= norm;
  val.y *= norm;
  return val;
}

template<typename T1, typename T2>
__device__ __inline__ T1 multiply_dev(T1 val, T2 norm){
  return val*norm;
}

__device__ __inline__ cuComplex add(cuComplex val1, cuComplex val2){
  val1.x += val2.x;
  val1.y += val2.x;
  return val1;
}

template<typename T1, typename T2>
__device__ __inline__ T1 add(T1 val1, T2 val2){
  return val1+val2;
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
void createPlan(int* handle, int row, int col){
  cufftPlan2d (handle, row, col, FFTformat);
}
void createPlan1d(int* handle, int n){
  cufftPlan1d(handle, n, FFTformat, 1);
}
void destroyPlan(int handle){
  cufftDestroy(handle);
}
void myFFTM(int handle, void* in, void* out){
  myCufftExec(handle, (cuComplex*)in, (cuComplex*)out, CUFFT_FORWARD);
}
void myIFFTM(int handle, void* in, void* out){
  myCufftExec(handle, (cuComplex*)in, (cuComplex*)out, CUFFT_INVERSE);
}
void myFFT(void* in, void* out){
  myCufftExec(*plan, (cuComplex*)in, (cuComplex*)out, CUFFT_FORWARD);
}
void myIFFT(void* in, void* out){
  myCufftExec(*plan, (cuComplex*)in, (cuComplex*)out, CUFFT_INVERSE);
}
void myFFTR2C(void* in, void* out){
  myCufftExecR2C(*planR2C, (Real*)in, (cuComplex*)out);
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
  getWindowWrap<<<numBlocks, threadsPerBlock>>>(addVar((cuComplex*)object, shiftx, shifty, objrow, objcol, (cuComplex*)window, replace, norm));
}

cuFuncTemplate(cudaConvertFO, (T* data, T* out),(data,out==0?data:out),{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= (cuda_row*cuda_column)/2) return;
    int x = index%cuda_row;
    int y = index/cuda_row;
    index = x*cuda_column+y;
    int indexp = (x >= (cuda_row>>1)? x - (cuda_row>>1) : (x + (cuda_row>>1)))*cuda_column + y + (cuda_column>>1);
    T tmp = data[index];
    out[index]=data[indexp];
    out[indexp]=tmp;
    })
template void cudaConvertFO<Real>(Real*, Real*);
template<> void cudaConvertFO<complexFormat>(complexFormat* data, complexFormat* out){
  cudaConvertFOWrap<<<numBlocks, threadsPerBlock>>>(addVar((cuComplex*)data, (cuComplex*)(out==0?data:out)));
}
template<typename T1, typename T2>
__global__ void multiplyWrap(int cuda_row, int cuda_column, int cuda_height, T1* store, T1* src, T2* target){
  cuda1Idx();
  store[index] = multiply_dev(src[index], target[index]);
}
template <typename T1, typename T2>
void multiply(T1* store, T1* src, T2* target){
  multiplyWrap<<<numBlocks,threadsPerBlock>>>(cuda_imgsz.x, cuda_imgsz.y, cuda_imgsz.z, store, src, target);
}
template void multiply(Real*, Real*, Real*);
template void multiply(complexFormat*, complexFormat*, complexFormat*);
template void multiply(complexFormat*, complexFormat*, Real*);

cuFuncc(innerProd, (Real* store, complexFormat* src, complexFormat* target), (Real* store, cuComplex* src, cuComplex* target), (store, (cuComplex*)src, (cuComplex*)target),{
  cuda1Idx();
  store[index] = src[index].x * target[index].x + target[index].y * target[index].y;
})

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
  rotate90Wrap<<<numBlocks, threadsPerBlock>>>(addVar((cuComplex*)data, (cuComplex*)(out==0?data:out), clockwise));
}

cuFuncTemplate(rotate, (T* data, T* out, Real angle),(data,out,angle),{
    cudaIdx();
    Real xp, yp;
    xp = cos(angle)*(x-cuda_row/2)-sin(angle)*(y-cuda_column/2) + cuda_row/2;
    yp = cos(angle)*(y-cuda_column/2)+sin(angle)*(x-cuda_row/2) + cuda_column/2;
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
  rotate90Wrap<<<numBlocks, threadsPerBlock>>>(addVar((cuComplex*)data, (cuComplex*)(out==0?data:out), clockwise));
}

cuFuncTemplate(transpose, (T* data, T* out),(data,out==0?data:out),{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= (cuda_row*cuda_column)/2) return;
    int indexp = cuda_row*cuda_column - index;
    T tmp = data[index];
    out[index]=data[indexp];
    out[indexp]=tmp;
    })
template void transpose<Real>(Real*, Real*);
template<> void transpose<complexFormat>(complexFormat* data, complexFormat* out){
  transposeWrap<<<numBlocks, threadsPerBlock>>>(addVar((cuComplex*)data, (cuComplex*)(out==0?data:out)));
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
  flipxWrap<<<numBlocks, threadsPerBlock>>>(addVar((cuComplex*)data, (cuComplex*)(out==0?data:out)));
}

template <typename T1, typename T2>
__global__ void assignValWrap(int cuda_row, int cuda_column, int cuda_height, T1* out, T2* input){
  cuda1Idx()
    out[index] = input[index];
}
template <typename T1, typename T2>
void assignVal(T1* out, T2* input){
  assignValWrap<<<numBlocks,threadsPerBlock>>>(cuda_imgsz.x, cuda_imgsz.y, cuda_imgsz.z, out, input);
}
template void assignVal<Real,Real>(Real*, Real*);
template void assignVal<Real,double>(Real*, double*);
template<> void assignVal<complexFormat, complexFormat>(complexFormat* out, complexFormat* input){
  assignValWrap<<<numBlocks,threadsPerBlock>>>(cuda_imgsz.x, cuda_imgsz.y, cuda_imgsz.z, (cuComplex*)out,(cuComplex*)input);
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
  cropWrap<<<numBlocks, threadsPerBlock>>>(addVar((cuComplex*)src, (cuComplex*)dest, row, col, midx, midy));
}
cuFuncTemplate(setValue,(T* data, T value),(data,value),{
    cuda1Idx()
    data[index] = value;
    })
template void setValue<Real>(Real*, Real);
template<> void setValue<complexFormat>(complexFormat* data, complexFormat value){
  setValueWrap<<<numBlocks, threadsPerBlock>>>(addVar((cuComplex*)data, *(cuComplex*)&value));
}


cuFuncc(multiplyShift,(complexFormat* object, Real shiftx, Real shifty),(cuComplex* object, Real shiftx, Real shifty),((cuComplex*)object,shiftx,shifty),{
    cudaIdx();
    Real phi = -2*M_PI*(shiftx*(x-cuda_row/2)/cuda_row+shifty*(y-cuda_column/2)/cuda_column);
    cuComplex tmp = {cos(phi),sin(phi)};
    object[index] = cuCmulf(object[index],tmp);
    })

cuFuncc(applyGaussMult,(complexFormat* input, complexFormat *output, Real sigma, bool isFreq),(cuComplex* input, cuComplex* output, Real sigma, bool isFreq),((cuComplex*)input,(cuComplex*)output, sigma,isFreq),{
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

cuFuncc(applyNorm,(complexFormat* data, Real factor),(cuComplex* data, Real factor),((cuComplex*)data,factor),{
    cuda1Idx()
    data[index].x*=factor;
    data[index].y*=factor;
    })
cuFunc(applyNorm,(Real* data, Real factor),(data,factor),{
    cuda1Idx()
    data[index]*=factor;
    })

cuFuncc(rotateToReal,(complexFormat* data),(cuComplex* data),((cuComplex*)data),{
    cuda1Idx();
    data[index].x = cuCabsf(data[index]);
    data[index].y = 0;
    })

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
cuFuncc(fillRedundantR2C,(complexFormat* data, complexFormat* dataout, Real factor),(cuComplex* data, cuComplex* dataout, Real factor),((cuComplex*)data,(cuComplex*)dataout,factor),{
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

cuFunc(interpolate,(Real* out, Real* data0, Real* data1, Real dx),(out, data0,data1,dx),{
    cuda1Idx()
    out[index] = data0[index]*(1-dx) + data1[index]*dx;
    })
cuFuncc(interpolate,(complexFormat* out, complexFormat* data0, complexFormat* data1, Real dx),(cuComplex* out, cuComplex* data0, cuComplex* data1, Real dx),((cuComplex*)out, (cuComplex*)data0,(cuComplex*)data1,dx),{
    cuda1Idx()
    out[index].x = data0[index].x*(1-dx) + data1[index].x*dx;
    out[index].y = data0[index].y*(1-dx) + data1[index].y*dx;
    })
cuFunc(adamUpdateV,(Real* v, Real* grad, Real beta2),(v,grad,beta2),{
    cuda1Idx()
    Real tmp = grad[index];
    v[index] = tmp*tmp*(1-beta2) + beta2*v[index];
    })
cuFuncc(adamUpdateV,(Real* v, complexFormat* grad, Real beta2),(Real* v, cuComplex* grad, Real beta2),(v,(cuComplex*)grad,beta2),{
    cuda1Idx()
    Real tmp = grad[index].x;
    v[index] = tmp*tmp*(1-beta2) + beta2*v[index];
    })
cuFuncc(adamUpdate,(complexFormat* xn, complexFormat* m, Real* v, Real lr, Real eps),(cuComplex* xn, cuComplex* m, Real* v, Real lr, Real eps),((cuComplex*)xn,(cuComplex*)m,v,lr,eps),{
    cuda1Idx()
    xn[index].x += lr*m[index].x/(sqrt(v[index])+eps);
    })
cuFuncc(ceiling,(complexFormat* data, Real ceilval),(cuComplex* data, Real ceilval),((cuComplex*)data,ceilval),{
    cuda1Idx()
    Real factor = ceilval/hypot(data[index].x, data[index].y);
    if(factor>1) return;
    data[index].x*=factor;
    data[index].y*=factor;
    })
cuFuncc(multiplyReal,(Real* store, complexFormat* src, complexFormat* target),(Real* store, cuComplex* src, cuComplex* target),(store,(cuComplex*)src,(cuComplex*)target),{
    cuda1Idx();
    store[index] = src[index].x*target[index].x;
    })

cuFuncc(multiplyConj,(complexFormat* store, complexFormat* src, complexFormat* target),(cuComplex* store, cuComplex* src, cuComplex* target),((cuComplex*)store,(cuComplex*)src,(cuComplex*)target),{
    cuda1Idx()
    store[index] = cuCmulf(src[index], cuConjf(target[index]));
    })

cuFuncc(multiplyRegular,(complexFormat* store, complexFormat* src, complexFormat* target, Real alpha),(cuComplex* store, cuComplex* src, cuComplex* target, Real alpha),((cuComplex*)store,(cuComplex*)src,(cuComplex*)target, alpha),{
    cuda1Idx()
    Real fact = 1;//target[index].x*target[index].x + target[index].y*target[index].y + alpha;
    cuComplex tmp = cuCmulf(src[index], target[index]);
    tmp.x /= fact;
    tmp.y /= fact;
    store[index] = tmp ; 
    })

cuFuncc(forcePositive,(complexFormat* a),(cuComplex* a),((cuComplex*)a),{
    cuda1Idx()
    if(a[index].x<0) a[index].x=0;
    a[index].y = 0;
    })

cuFunc(forcePositive,(Real* a),(a),{
    cuda1Idx()
    if(a[index]<0) a[index]=0;
    })

template<typename T1, typename T2>
__device__ auto sub(T1 a, T2 b) { return a - b; }

template<typename T1, typename T2>
__device__ auto mul(T1 a, T2 b) { return a * b; }
__device__ __inline__ float2 sub(float2 a, float2 b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

__device__ __inline__ float2 mul(float2 a, float s) {
    return make_float2(a.x * s, a.y * s);
}

template<typename T>
__device__ __inline__ T bilinearInterpolate(const T* img, int w, int h, float x, float y) {
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
template void resize<complexFormat>(const complexFormat*, complexFormat*, int, int);

cuFuncc(multiplyPropagatePhase,(complexFormat* amp, Real a, Real b),(cuComplex* amp, Real a, Real b),((cuComplex*)amp,a,b),{ // a=z/lambda, b = (lambda/s)^2, s is the image size
    cudaIdx();
    cuComplex phasefactor;
    Real phase = a*sqrt(1-(sq(x-(cuda_row>>1))+sq(y-(cuda_column>>1)))*b);
    phasefactor.x = cos(phase);
    phasefactor.y = sin(phase);
    amp[index] = cuCmulf(amp[index],phasefactor);
    })

cuFuncc(extendToComplex,(Real* a, complexFormat* b),(Real* a, cuComplex* b),(a,(cuComplex*)b),{
    cuda1Idx()
    b[index].x = a[index];
    b[index].y = 0;
    })

cuFuncc(createWaveFront,(Real* d_intensity, Real* d_phase, complexFormat* objectWave, int row, int col, int shiftx, int shifty, Real phaseFactor),(Real* d_intensity, Real* d_phase, cuComplex* objectWave, int row, int col, int shiftx, int shifty, Real phaseFactor),(d_intensity,d_phase,(cuComplex*)objectWave,row,col,shiftx,shifty,phaseFactor),{
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
    objectWave[index].x = mod*cos(phase);
    objectWave[index].y = mod*sin(phase);
    }else{
    objectWave[index].x = mod;
    objectWave[index].y = 0;
    }
})

cuFuncc(createWaveFront,(Real* d_intensity, Real* d_phase, complexFormat* objectWave, Real oversampling, Real shiftx, Real shifty, Real phaseFactor),(Real* d_intensity, Real* d_phase, cuComplex* objectWave, Real oversampling, Real shiftx, Real shifty, Real phaseFactor),(d_intensity,d_phase,(cuComplex*)objectWave,oversampling,shiftx,shifty,phaseFactor),{
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
    objectWave[index].x = mod*cos(phase);
    objectWave[index].y = mod*sin(phase);
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

cuFunc(ccdRecord, (uint16_t* data, Real* wave, int noiseLevel, void* state, Real exposure),
    (data,wave,noiseLevel,state,exposure),{
    cuda1Idx()
    int dataint = curand_poisson(&((curandStateMRG32k3a*)state)[index], noiseLevel + vars->rcolor*wave[index]*exposure);
    if(dataint >= vars->rcolor) dataint = vars->rcolor-1;
    data[index] = dataint-noiseLevel;
    });
cuFuncc(ccdRecord, (uint16_t* data, complexFormat* wave, int noiseLevel, void* state, Real exposure),(uint16_t* data, cuComplex* wave, int noiseLevel, void* state, Real exposure),
    (data,(cuComplex*)wave,noiseLevel,state,exposure),{
    cuda1Idx()
    cuComplex tmp = wave[index];
    int dataint = curand_poisson(&((curandStateMRG32k3a*)state)[index], noiseLevel + vars->rcolor*(tmp.x*tmp.x+tmp.y*tmp.y)*exposure);
    if(dataint >= vars->rcolor) dataint = vars->rcolor-1;
    data[index] = dataint-noiseLevel;
    });
cuFunc(ccdRecord, (Real* data, Real* wave, int noiseLevel, void* state, Real exposure, int rcolor),
    (data,wave,noiseLevel,state,exposure, rcolor),{
    cuda1Idx()
    if(rcolor == 0) rcolor = vars->rcolor;
    int dataint = curand_poisson(&((curandStateMRG32k3a*)state)[index], noiseLevel + rcolor*wave[index]*exposure);
    if(dataint >= rcolor) dataint = rcolor-1;
    data[index] = Real(dataint-noiseLevel)/rcolor;
    });
cuFuncc(ccdRecord, (Real* data, complexFormat* wave, int noiseLevel, void* state, Real exposure),(Real* data, cuComplex* wave, int noiseLevel, void* state, Real exposure),
    (data,(cuComplex*)wave,noiseLevel,state,exposure),{
    cuda1Idx()
    cuComplex tmp = wave[index];
    int dataint = curand_poisson(&((curandStateMRG32k3a*)state)[index], noiseLevel + vars->rcolor*(tmp.x*tmp.x+tmp.y*tmp.y)*exposure);
    if(dataint >= vars->rcolor) dataint = vars->rcolor-1;
    data[index] = Real(dataint-noiseLevel)/vars->rcolor;
    });
cuFuncc(ccdRecord, (complexFormat* data, complexFormat* wave, int noiseLevel, void* state, Real exposure),(cuComplex* data, cuComplex* wave, int noiseLevel, void* state, Real exposure),
    ((cuComplex*)data,(cuComplex*)wave,noiseLevel,state,exposure),{
    cuda1Idx()
    cuComplex tmp = wave[index];
    int dataint = curand_poisson(&((curandStateMRG32k3a*)state)[index], noiseLevel + vars->rcolor*(tmp.x*tmp.x+tmp.y*tmp.y)*exposure);
    if(dataint >= vars->rcolor) dataint = vars->rcolor-1;
    data[index].x = Real(dataint-noiseLevel)/vars->rcolor;
    data[index].y = 0;
    });
cuFunc(applyPoissonNoise,(Real* wave, Real noiseLevel, void* state, Real scale),
    (wave,noiseLevel,state,scale),{
    cuda1Idx()
    if(scale==0) scale = vars->scale;
    wave[index]+=scale*(curand_poisson(&((curandStateMRG32k3a*)state)[index], noiseLevel)-noiseLevel)/vars->rcolor;
    })
cuFunc(applyPoissonNoise_WO,(Real* wave, Real noiseLevel, void* state, Real scale),
    (wave,noiseLevel,state,scale),{
    cuda1Idx()
    if(scale==0) scale = vars->scale;
    wave[index]=scale*(int(wave[index]*vars->rcolor/scale) + curand_poisson(&((curandStateMRG32k3a*)state)[index], noiseLevel)-noiseLevel)/vars->rcolor;
    })

cuFuncc(getMod,(Real* mod, complexFormat* amp),(Real* mod, cuComplex* amp),(mod,(cuComplex*)amp),{
    cuda1Idx()
    mod[index] = cuCabsf(amp[index]);
    })
cuFuncc(getReal,(Real* mod, complexFormat* amp, Real norm),(Real* mod, cuComplex* amp, Real norm),(mod,(cuComplex*)amp, norm),{
    cuda1Idx()
    mod[index] = amp[index].x*norm;
    })
cuFuncc(addReal,(Real* mod, complexFormat* amp, Real norm),(Real* mod, cuComplex* amp, Real norm),(mod,(cuComplex*)amp, norm),{
    cuda1Idx()
    mod[index] += amp[index].x*norm;
    })
cuFuncc(getImag,(Real* mod, complexFormat* amp),(Real* mod, cuComplex* amp),(mod,(cuComplex*)amp),{
    cuda1Idx()
    mod[index] = amp[index].y;
    })
cuFuncc(assignReal,(Real* mod, complexFormat* amp),(Real* mod, cuComplex* amp),(mod,(cuComplex*)amp),{
    cuda1Idx()
    amp[index].x = mod[index];
    })
cuFuncc(assignImag,(Real* mod, complexFormat* amp),(Real* mod, cuComplex* amp),(mod,(cuComplex*)amp),{
    cuda1Idx()
    amp[index].y = mod[index];
    })
cuFuncc(getMod2,(Real* mod2, complexFormat* amp),(Real* mod2, cuComplex* amp),(mod2,(cuComplex*)amp),{
    cuda1Idx()
    cuComplex tmp = amp[index];
    mod2[index] = tmp.x*tmp.x + tmp.y*tmp.y;
    })
cuFuncc(getMod2,(complexFormat* mod2, complexFormat* amp),(cuComplex* mod2, cuComplex* amp),((cuComplex*)mod2,(cuComplex*)amp),{
    cuda1Idx()
    cuComplex tmp = amp[index];
    mod2[index].x = tmp.x*tmp.x + tmp.y*tmp.y;
    mod2[index].y = 0;
    })
cuFuncc(addMod2,(Real* mod2, complexFormat* amp, Real norm),(Real* mod2, cuComplex* amp, Real norm),(mod2,(cuComplex*)amp,norm),{
    cuda1Idx()
    cuComplex tmp = amp[index];
    mod2[index] += tmp.x*tmp.x*norm + tmp.y*tmp.y*norm;
    })
cuFunc(getMod2,(Real* mod2, Real* mod),(mod2,mod),{
    cuda1Idx()
    mod2[index] = sq(mod[index]);
    })

cuFuncc(bitMap,(Real* store, complexFormat* amp, Real threshold),(Real* store, cuComplex* amp, Real threshold),(store,(cuComplex*)amp, threshold),{
    cuda1Idx()
    if(threshold == 0) threshold = vars->threshold;
    cuComplex tmp = amp[index];
    store[index] = tmp.x*tmp.x+tmp.y*tmp.y > threshold*threshold;
    })

cuFunc(bitMap,(Real* store, Real* amp, Real threshold),(store,amp, threshold),{
    cuda1Idx()
    if(threshold == 0) threshold = vars->threshold;
    store[index] = amp[index] > threshold;
    })

cuFunc(applyThreshold,(Real* store, Real* input, Real threshold),(store,input,threshold),{
    cuda1Idx()
    store[index] = input[index] > threshold? input[index] : 0;
    })

cuFunc(linearConst,(Real* store, Real* data, Real fact, Real shift),(store, data, fact, shift),{
    cuda1Idx();
    store[index] = fact*data[index]+shift;
    })

cuFuncc(applyModAbs,(complexFormat* source, Real* target, void* state),(cuComplex* source, Real* target, void* state),((cuComplex*)source, target, state),{
    cuda1Idx();
    Real mod = hypot(source[index].x, source[index].y);
    Real rat = target[index];
    if(rat > 0) rat = sqrt(rat);
    else rat = 0;
    if(mod==0) {
    Real randphase = state?curand_uniform((curandStateMRG32k3a*)state + index)*2*M_PI:0;
    source[index].x = rat*cos(randphase);
    source[index].y = rat*sin(randphase);
    return;
    }
    rat /= mod;
    source[index].x *= rat;
    source[index].y *= rat;
    })
cuFuncc(applyModAbsinner,(complexFormat* source, Real* target,  int row, int col, Real norm, void* state),(cuComplex* source, Real* target,  int row, int col, Real norm, void* state),((cuComplex*)source,target,row,col,norm, state),{
    cudaIdx()
    int targetx = x >= cuda_row/2 ? x - (cuda_row - row) : x;
    int targety = y >= cuda_column/2 ? y - (cuda_column - col) : y;
    Real rat = target[index]*norm;
    index = targetx*col+targety;
    Real mod = hypot(source[index].x, source[index].y);
    if(rat > 0) rat = sqrt(rat);
    else rat = 0;
    if(mod==0) {
    Real randphase = state?curand_uniform((curandStateMRG32k3a*)state+index)*2*M_PI:0;
    source[index].x = rat*cos(randphase);
    source[index].x = rat*sin(randphase);
    return;
    }
    rat /= mod;
    source[index].x *= rat;
    source[index].y *= rat;
    })

cuFuncc(applyMod,(complexFormat* source, Real* target, Real *bs, bool loose, int iter, int noiseLevel),(cuComplex* source, Real* target, Real *bs, bool loose, int iter, int noiseLevel), ((cuComplex*)source, target, bs, loose, iter, noiseLevel),{
    cuda1Idx()
    Real maximum = vars->scale*0.95;
    Real mod2 = target[index];
    if(mod2<0) mod2=0;
    if(loose && bs && bs[index]>0.5) {
    //if(iter > 500) return;
    //else mod2 = maximum+1;
    return;
    }
   //Real tolerance = (sqrtf(mod2*vars->rcolor + noiseLevel))*vars->scale/vars->rcolor; // fluctuation caused by bit depth and noise
    Real tolerance = (1.+sqrtf(noiseLevel))*vars->scale/vars->rcolor; // fluctuation caused by bit depth and noise
    cuComplex sourcedata = source[index];
    Real srcmod2 = sourcedata.x*sourcedata.x + sourcedata.y*sourcedata.y;
    if(mod2>=maximum) {
    if(loose) mod2 = max(maximum,srcmod2);
    else tolerance*=1000;
    }
    if(srcmod2 == 0){
    source[index].x = sqrt(mod2);
    source[index].y = 0;
    return;
    }
    Real diff = mod2-srcmod2;
    Real val = mod2;
    if(diff>tolerance){
      val -= tolerance;
    }else if(diff < -tolerance ){
      val += tolerance;
    }
    val = sqrt(val/srcmod2);
    source[index].x = val*sourcedata.x;
    source[index].y = val*sourcedata.y;
})
cuFuncc(convertFOPhase, (complexFormat* data),(cuComplex* data),((cuComplex*)data),{
    cudaIdx()
    if((x+y)%2==1) {
    data[index].x = -data[index].x;
    data[index].y = -data[index].y;
    }
    })
cuFuncc(add,(complexFormat* a, complexFormat* b, Real c ),(cuComplex* a, cuComplex* b, Real c ),((cuComplex*)a,(cuComplex*)b,c),{
    cuda1Idx()
    a[index].x+=b[index].x*c;
    a[index].y+=b[index].y*c;
    })
cuFuncc(add,(complexFormat* store, complexFormat* a, complexFormat* b, Real c ),(cuComplex* store, cuComplex* a, cuComplex* b, Real c ),((cuComplex*)store,(cuComplex*)a,(cuComplex*)b,c),{
    cuda1Idx()
    store[index].x=a[index].x + b[index].x*c;
    store[index].y=a[index].y + b[index].y*c;
    })
cuFuncc(normAdd,(complexFormat* store, complexFormat* a, complexFormat* b, Real c, Real d),(cuComplex* store, cuComplex* a, cuComplex* b, Real c, Real d ),((cuComplex*)store,(cuComplex*)a,(cuComplex*)b,c,d),{
    cuda1Idx()
    store[index].x=a[index].x*c + b[index].x*d;
    store[index].y=a[index].y*c + b[index].y*d;
    })
cuFunc(add,(Real* a, Real* b, Real c),(a,b,c),{
    cuda1Idx()
    a[index]+=b[index]*c;
    })

cuFunc(add,(Real* store, Real* a, Real* b, Real c),(store, a,b,c),{
    cuda1Idx()
    store[index] = a[index]+b[index]*c;
    })
cuFunc(addRemoveOE, (Real* src, Real* sub, Real mult), (src, sub,mult), {
    cuda1Idx();
    if(sub[index] < 0.99){
    src[index]+=sub[index]*mult;
    }else{
    src[index] = 0;
    }
    });
cuFuncc(applyRandomPhase,(complexFormat* wave, Real* beamstop, void* state),(cuComplex* wave, Real* beamstop, void* state),
    ((cuComplex*)wave, beamstop, state),{
    cuda1Idx()
    cuComplex tmp = wave[index];
    if(beamstop && beamstop[index]>vars->threshold) {
    tmp.x = tmp.y = 0;
    }
    else{
    Real mod = cuCabsf(wave[index]);
    Real randphase = (curand_uniform((curandStateMRG32k3a*)state+index)>0.5)*M_PI;
    tmp.x = mod*cos(randphase);
    tmp.y = mod*sin(randphase);
    }
    wave[index] = tmp;
    })

__device__ cuComplex getFact(Real phase, int l){
  cuComplex nom;
  if(phase != 0){
    nom.x = cos(phase)-1;
    nom.y = sin(phase);
    Real mod2 = sqSum(nom.x, nom.y);
    nom.x = nom.x/mod2;
    nom.y = nom.y/mod2;  //omitted a - sign
    Real nomy = 2*sin(phase*l/2);
    nom.x = nom.y*nomy;
    nom.y = nom.x*nomy;
  }else{
    nom.x = l;
    nom.y = 0;
  }
  return nom;
}
cuFunc(cropinner,(Real* src, Real* dest, int row, int col, Real norm),(src,dest,row,col,norm),{
    cudaIdx()
    int targetx = x >= cuda_row/2 ? x + row - cuda_row : x;
    int targety = y >= cuda_column/2 ? y + col - cuda_column : y;
    int targetidx = targetx * col + targety;
    dest[index] = src[targetidx]*norm;
    })
cuFunc(mergePixel, (Real* output, Real* input, int row, int col, int nmerge),(output, input, row, col, nmerge),{
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

cuFuncc(cropinner,(complexFormat* src, complexFormat* dest, int row, int col, Real norm),(cuComplex* src, cuComplex* dest, int row, int col, Real norm),((cuComplex*)src,(cuComplex*)dest,row,col,norm),{
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

cuFuncc(padinner, (complexFormat* src, complexFormat* dest, int row, int col, Real norm),(cuComplex* src, cuComplex* dest, int row, int col, Real norm), ((cuComplex*)src, (cuComplex*)dest, row, col, norm),{
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
cuFuncc(paste, (complexFormat* out, complexFormat* in, int colout, int posx, int posy, bool replace),(cuComplex* out, cuComplex* in, int colout, int posx, int posy, bool replace),((cuComplex*)out, (cuComplex*)in, colout, posx, posy, replace),{
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
cuFunc(getXZSlice, (Real * slice, Real *data, int nx, int ny, int nz, int iy),
    (slice, data, nx, ny, nz, iy), {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nx * nz)
    return;
    int x = index % nx;
    int z = index / nx;
    slice[index] = data[x + nx * iy + nx * ny * z];
    });
cuFunc(getYZSlice, (Real * slice, Real *data, int nx, int ny, int nz, int ix),
    (slice, data, nx, ny, nz, ix), {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= ny * nz)
    return;
    int y = index % ny;
    int z = index / ny;
    int idx = ix + nx * y + nx * ny * z;
    slice[index] = data[idx];
    })
//-------colorbar-begin
cuFuncc(createColorbar, (complexFormat* output), (cuComplex *output), ((cuComplex*)output),{
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
cuFuncc(multiplyPatternPhase_Device,(complexFormat* amp, Real r_d_lambda, Real d_r_lambda),(cuComplex* amp, Real r_d_lambda, Real d_r_lambda),((cuComplex*)amp,r_d_lambda,d_r_lambda),{
    cudaIdx()
    Real phase = (sq(x-(cuda_row>>1))+sq(y-(cuda_column>>1)))*r_d_lambda+d_r_lambda;
    cuComplex p = {cos(phase),sin(phase)};
    amp[index] = cuCmulf(amp[index], p);
    })

cuFuncc(multiplyPatternPhaseOblique_Device,(complexFormat* amp, Real r_d_lambda, Real d_r_lambda, Real costheta),(cuComplex* amp, Real r_d_lambda, Real d_r_lambda, Real costheta),((cuComplex*)amp,r_d_lambda,d_r_lambda,costheta),{ // pixsize*pixsize*M_PI/(d*lambda) and 2*d*M_PI/lambda and costheta = z/r
    cudaIdx()
    Real phase = (sq((x-(cuda_row>>1)*costheta))+sq(y-(cuda_column>>1)))*r_d_lambda+d_r_lambda;
    cuComplex p = {cos(phase),sin(phase)};
    amp[index] = cuCmulf(amp[index], p);
    })

cuFuncc(multiplyFresnelPhase_Device,(complexFormat* amp, Real phaseFactor),(cuComplex* amp, Real phaseFactor),((cuComplex*)amp,phaseFactor),{ // pixsize*pixsize*M_PI/(d*lambda) and 2*d*M_PI/lambda
    cudaIdx()
    Real phase = phaseFactor*(sq(x-(cuda_row>>1))+sq(y-(cuda_column>>1)));
    cuComplex p = {cos(phase),sin(phase)};
    if(cuCabsf(amp[index])!=0) amp[index] = cuCmulf(amp[index], p);
    })

cuFuncc(multiplyFresnelPhaseOblique_Device,(complexFormat* amp, Real phaseFactor, Real costheta_r),(cuComplex* amp, Real phaseFactor, Real costheta_r),((cuComplex*)amp,phaseFactor,costheta_r),{ // costheta_r = 1./costheta = r/z
    cudaIdx()
    Real phase = phaseFactor*(sq((x-(cuda_row>>1))*costheta_r)+sq(y-(cuda_column>>1)));
    cuComplex p = {cos(phase),sin(phase)};
    if(cuCabsf(amp[index])!=0) amp[index] = cuCmulf(amp[index], p);
    })

//-------experimentConfig.cc-end

//-------cdi.cc-begin

cuFuncc(takeMod2Diff,(complexFormat* a, Real* b, Real *output, Real *bs),(cuComplex* a, Real* b, Real *output, Real *bs),((cuComplex*)a,b,output,bs),{
    cuda1Idx()
    Real mod2 = sq(a[index].x)+sq(a[index].y);
    Real tmp = b[index]-mod2;
    if(bs&&bs[index]>0.5) tmp=0;
    else if(b[index]>vars->scale) tmp = vars->scale-mod2;
    output[index] = tmp;
    })

cuFuncc(takeMod2Sum,(complexFormat* a, Real* b),(cuComplex* a, Real* b),((cuComplex*)a,b),{
    cuda1Idx()
    Real tmp = b[index]+sq(a[index].x)+sq(a[index].y);
    if(tmp<0) tmp=0;
    b[index] = tmp;
    })
__device__ void ApplyHIOSupport(bool insideS, cuComplex &rhonp1, cuComplex &rhoprime, Real beta){
  if(insideS){
    rhonp1.x = rhoprime.x;
    rhonp1.y = rhoprime.y;
  }else{
    rhonp1.x -= beta*rhoprime.x;
    rhonp1.y -= beta*rhoprime.y;
  }
}
__device__ void ApplyFHIOSupport(bool insideS, cuComplex &rhonp1, cuComplex &rhoprime){
  if(insideS){
    rhonp1.x += 1.9*(rhoprime.x-rhonp1.x);
    rhonp1.y += 1.9*(rhoprime.y-rhonp1.y);
  }else{
    rhonp1.x -= 1.2*rhoprime.x;
    rhonp1.y -= 1.2*rhoprime.y;
  }
}
__device__ void ApplyRAARSupport(bool insideS, cuComplex &rhonp1, cuComplex &rhoprime, Real beta){
  if(insideS){
    rhonp1.x = rhoprime.x;
    rhonp1.y = rhoprime.y;
  }else{

    rhonp1.x = beta*rhonp1.x+(1-2*beta)*rhoprime.x;
    rhonp1.y = beta*rhonp1.y+(1-2*beta)*rhoprime.y;
    //    rhonp1.x = beta*(rhonp1.x-rhoprime.x);
    //    rhonp1.y = beta*(rhonp1.y-rhoprime.y);
  }
}
__device__ void ApplyPOSERSupport(bool insideS, cuComplex &rhonp1, cuComplex &rhoprime){
  if(insideS && rhoprime.x > 0){
    rhonp1.x = rhoprime.x;
    rhonp1.y = rhoprime.y;
  }else{
    rhonp1.x = 0;
    rhonp1.y = 0;
  }
}
__device__ void ApplyERSupport(bool insideS, cuComplex &rhonp1, cuComplex &rhoprime){
  if(insideS){
    rhonp1.x += 1.9*(rhoprime.x-rhonp1.x);
    rhonp1.y += 1.9*(rhoprime.y-rhonp1.y);
  }else{
    rhonp1.x = 0;
    rhonp1.y = 0;
  }
}
__device__ void ApplyPOSHIOSupport(bool insideS, cuComplex &rhonp1, cuComplex &rhoprime, Real beta){
  if(rhoprime.x > 0 && insideS){
    rhonp1.x += 1.9*(rhoprime.x-rhonp1.x);
    //rhonp1.y = rhoprime.y;
  }else{
    rhonp1.x -= beta*rhoprime.x;
  }
  rhonp1.y -= beta*rhoprime.y;
}
__device__ void ApplyPOS0HIOSupport(bool insideS, cuComplex &rhonp1, cuComplex &rhoprime, Real beta){
  if(rhoprime.x > 0 && insideS){
    rhonp1.x = rhoprime.x;
    //rhonp1.y = rhoprime.y;
  }else{
    rhonp1.x -= beta*rhoprime.x;
  }
  rhonp1.y -= beta*rhoprime.y;
}
cuFuncc(applySupportOblique,(complexFormat *gkp1, complexFormat *gkprime, int algo, Real *spt, int iter, Real fresnelFactor, Real costheta_r),(cuComplex* gkp1, cuComplex* gkprime, int algo, Real *spt, int iter, Real fresnelFactor, Real costheta_r),((cuComplex*)gkp1,(cuComplex*)gkprime,algo,spt,iter,fresnelFactor,costheta_r),{
    cudaIdx()
    bool inside = spt[index] > vars->threshold;
    cuComplex &gkp1data = gkp1[index];
    cuComplex &gkprimedata = gkprime[index];
    if(algo==RAAR) ApplyRAARSupport(inside,gkp1data,gkprimedata,vars->beta_HIO);
    else if(algo==ER) ApplyERSupport(inside,gkp1data,gkprimedata);
    else if(algo==HIO) ApplyHIOSupport(inside,gkp1data,gkprimedata,vars->beta_HIO);
    if(fresnelFactor>1e-4 && iter < 400) {
    if(inside){
    Real phase = M_PI*fresnelFactor*(sq((x-(cuda_row>>1))*costheta_r)+sq(y-(cuda_column>>1)));
    //Real mod = cuCabs(gkp1data);
    Real mod = fabs(gkp1data.x*cos(phase)+gkp1data.y*sin(phase)); //use projection (Error reduction)
    gkp1data.x=mod*cos(phase);
    gkp1data.y=mod*sin(phase);
    }
    }
    })
cuFunc(applySupport,(void *gkp1, void *gkprime, int algo, Real *spt, int iter, Real fresnelFactor),(gkp1,gkprime,algo,spt,iter,fresnelFactor),{
    cudaIdx();
    bool inside = spt[index] > vars->threshold;
    cuComplex &gkp1data = ((cuComplex*)gkp1)[index];
    cuComplex &gkprimedata = ((cuComplex*)gkprime)[index];
    if(algo==RAAR) ApplyRAARSupport(inside,gkp1data,gkprimedata,vars->beta_HIO);
    else if(algo==ER) ApplyERSupport(inside,gkp1data,gkprimedata);
    else if(algo==POSER) ApplyPOSERSupport(inside,gkp1data,gkprimedata);
    else if(algo==POSHIO) ApplyPOSHIOSupport(inside,gkp1data,gkprimedata,vars->beta_HIO);
    else if(algo==HIO) ApplyHIOSupport(inside,gkp1data,gkprimedata,vars->beta_HIO);
    else if(algo==FHIO) ApplyFHIOSupport(inside,gkp1data,gkprimedata);
    if(fresnelFactor>1e-4 && iter < 400) {
    if(inside){
    Real phase = M_PI*fresnelFactor*(sq(x-(cuda_row>>1))+sq(y-(cuda_column>>1)));
    //Real mod = cuCabs(gkp1data);
    Real mod = fabs(gkp1data.x*cos(phase)+gkp1data.y*sin(phase)); //use projection (Error reduction)
    gkp1data.x=mod*cos(phase);
    gkp1data.y=mod*sin(phase);
    }
    }
    })
//-------cdi.cc-end

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
cuFuncc(partialx, (complexFormat* b, complexFormat* p), (cuComplex* b, cuComplex* p), ((cuComplex*)b,(cuComplex*)p),{
    cuda1Idx()
    int x = index/cuda_column;
    cuComplex target;
    if(x == cuda_row-1) target = sub(b[index],b[index%cuda_column]);
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
cuFuncc(partialy, (complexFormat* b, complexFormat* p), (cuComplex* b, cuComplex* p), ((cuComplex*)b,(cuComplex*)p),{
    cuda1Idx()
    int y = index%cuda_column;
    cuComplex target;
    if(y == cuda_column-1) target = sub(b[index],b[index-cuda_column+1]);
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
cuFuncc(diffMax, (complexFormat* p, complexFormat* q), (cuComplex* p, cuComplex* q), ((cuComplex*)p,(cuComplex*)q),{
    cuda1Idx()
    Real mod;
    mod = hypot(p[index].x,q[index].x);
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
cuFuncc(calcLpq, (complexFormat* out, complexFormat* p, complexFormat* q), (cuComplex* out, cuComplex* p, cuComplex* q), ((cuComplex*)out,(cuComplex*)p,(cuComplex*)q),{
    cudaIdx()
    cuComplex tmp = add(p[index],q[index]);
    if(x >= 1) tmp = sub(tmp,p[index-cuda_column]);
    else tmp=sub(tmp,p[index+(cuda_row-1)*cuda_column]);
    if(y >= 1) tmp = sub(tmp,q[index-1]);
    else tmp=sub(tmp,q[index+cuda_column-1]);
    out[index] = tmp;
    })
//-------FISTA.cc-------end
//-------monoChromo.cc-------begin
cuFuncc(updateMomentum,(complexFormat* force, complexFormat* mom, Real dx),(cuComplex* force, cuComplex* mom, Real dx),((cuComplex*)force, (cuComplex*)mom , dx),{
    cuda1Idx()
    Real m = mom[index].x;
    Real f = force[index].x;
    // interpolate with walls
    //if(m * f < 0) m = f*(1-dx);
    //else m = m*dx + f*(1-dx);
    //m = m*dx + f*(1-dx);
    if(m * f < 0) m = f*dx;
    else m = m + f*dx;
    mom[index].x = m;
    })

cuFuncc(overExposureZeroGrad, (complexFormat* deltab, complexFormat* b, int noiseLevel),(cuComplex* deltab, cuComplex* b, int noiseLevel),((cuComplex*)deltab, (cuComplex*)b, noiseLevel),{
    cuda1Idx();
    if(b[index].x >= vars->scale*0.99 && deltab[index].x < 0) deltab[index].x = 0;
    //if(fabs(deltab[index].x)*vars->rcolor < sqrtf(noiseLevel)) deltab[index].x = 0;
    deltab[index].y = 0;
    })

cuFuncc(assignRef_d, (complexFormat* wavefront, uint32_t* mmap, complexFormat* rf, int n), (cuComplex* wavefront, uint32_t* mmap, cuComplex* rf, int n),((cuComplex*)wavefront, mmap, (cuComplex*)rf, n), {
    cuda1Idx()
    if(index >= n) return;
    rf[index] = wavefront[mmap[index]];
    })
cuFuncc(expandRef, (complexFormat* rf, complexFormat* amp, uint32_t* mmap, int row, int col, int row0, int col0),(cuComplex* rf, cuComplex* amp, uint32_t* mmap, int row, int col, int row0, int col0),((cuComplex*)rf, (cuComplex*)amp, mmap, row, col, row0, col0),{
    cuda1Idx()
    int idx = mmap[index];
    int x = idx/col0 + (row-row0)/2;
    int y = idx%col0 + (col-col0)/2;
    amp[x*col+y] = rf[index];
    })
cuFuncc(expandRef, (complexFormat* rf, complexFormat* amp, uint32_t* mmap, int row, int col, int row0, int col0, complexFormat a),(cuComplex* rf, cuComplex* amp, uint32_t* mmap, int row, int col, int row0, int col0, cuComplex a),((cuComplex*)rf, (cuComplex*)amp, mmap, row, col, row0, col0, {crealf(a),cimagf(a)}),{
    cuda1Idx()
    int idx = mmap[index];
    int x = idx/col0 + (row-row0)/2;
    int y = idx%col0 + (col-col0)/2;
    amp[x*col+y] = cuCmulf(rf[index],a);
    })
cuFuncc(saveRef, (complexFormat* rf, complexFormat* amp, uint32_t* mmap, int row, int col, int row0, int col0, Real norm),(cuComplex* rf, cuComplex* amp, uint32_t* mmap, int row, int col, int row0, int col0, Real norm),((cuComplex*)rf, (cuComplex*)amp, mmap, row, col, row0, col0, norm),{
    cuda1Idx()
    int idx = mmap[index];
    int x = idx/col0 + (row-row0)/2;
    int y = idx%col0 + (col-col0)/2;
    rf[index].x = amp[x*col+y].x*norm;
    rf[index].y = amp[x*col+y].y*norm;
    })
cuFuncc(saveRef_Real, (complexFormat* rf, complexFormat* amp, uint32_t* mmap, int row, int col, int row0, int col0, int n, Real norm),(cuComplex* rf, cuComplex* amp, uint32_t* mmap, int row, int col, int row0, int col0, int n, Real norm),((cuComplex*)rf, (cuComplex*)amp, mmap, row, col, row0, col0, n, norm),{
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
cuFuncc(applySupportBarHalf,(complexFormat* img, Real* spt),(cuComplex* img, Real* spt),((cuComplex*)img,spt),{
    cudaIdx();
    int hr = cuda_row>>1;
    int hc = cuda_column>>1;
    if(x > hr) x -= hr;
    else x += hr;
    if(y > hc) y -= hc;
    else y += hc;
    if(spt[index] > vars->threshold || x + y > cuda_row)
    img[index].x = img[index].y = 0;
    })


cuFuncc(applySupportBar_Flip,(complexFormat* img, Real* spt),(cuComplex* img, Real* spt),((cuComplex*)img,spt),{
    cuda1Idx();
    if(spt[index] > vars->threshold){
    img[index].x *= -0.3;
    img[index].y *= -0.3;
    }
    })

cuFuncc(applySupport,(complexFormat* img, Real* spt),(cuComplex* img, Real* spt),((cuComplex*)img,spt),{
    cuda1Idx();
    if(spt[index] < vars->threshold)
    img[index].x = img[index].y = 0;
    })

cuFuncc(dillate, (complexFormat* data, Real* support, int wid, int hit), (cuComplex* data, Real* support, int wid, int hit), ((cuComplex*)data,support,wid,hit),{
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

cuFuncc(applyModCorr, (complexFormat* obj, complexFormat* refer, Real* xcorrelation),(cuComplex* obj ,cuComplex* refer, Real* xcorrelation),((cuComplex*)obj,(cuComplex*)refer,xcorrelation),{
    cuda1Idx();
    cuComplex objtmp = obj[index];
    cuComplex reftmp = refer[index];
    if(reftmp.x == 0 && reftmp.y == 0) return;
    Real fact = xcorrelation[index]/2 - reftmp.x*objtmp.x - reftmp.y*objtmp.y;
    fact /= reftmp.x*reftmp.x + reftmp.y*reftmp.y;
    obj[index].x = objtmp.x + fact*reftmp.x;
    obj[index].y = objtmp.y + fact*reftmp.y;
    })

cuFuncc(devideStar, (complexFormat* obj, complexFormat* refer, complexFormat* xcorrelation),(cuComplex* obj ,cuComplex* refer, cuComplex* xcorrelation),((cuComplex*)obj,(cuComplex*)refer,(cuComplex*)xcorrelation),{
    cuda1Idx();
    cuComplex xctmp = xcorrelation[index];
    cuComplex reftmp = refer[index];
    Real fact = max(sqSum(reftmp.x,reftmp.y),1e-4);
    xctmp = cuCmulf(xctmp, reftmp);
    obj[index].x = xctmp.x / fact;
    obj[index].y = xctmp.y / fact;
    })

//-------holo.cc-------end
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
cuFunc(applyMask, (Real* data, Real* mask, Real threshold),(data,mask,threshold),{
    cuda1Idx();
    if(mask[index]<=threshold) data[index] = 0;
    })

cuFunc(applyMask, (Real* data, char* mask),(data,mask),{
    cuda1Idx();
    if(!mask[index]) data[index] = 0;
    })

cuFuncc(applyMask, (complexFormat* data, Real* mask, Real threshold),(cuComplex* data, Real* mask, Real threshold),((cuComplex*)data,mask,threshold),{
    cuda1Idx();
    if(mask[index]<=threshold) data[index].x = data[index].y = 0;
    })
cuFunc(applyMaskBar, (Real* data, Real* mask, Real threshold),(data,mask,threshold),{
    cuda1Idx();
    if(mask[index]>threshold) data[index] = 0;
    })
cuFuncc(applyMaskBar, (Real* data, complexFormat* mask, Real threshold),(Real* data, cuComplex* mask, Real threshold),(data,(cuComplex*)mask,threshold),{
    cuda1Idx();
    if(mask[index].x>threshold) data[index] = 0;
    })
cuFuncc(applyMaskBar, (complexFormat* data, Real* mask, Real threshold),(cuComplex* data, Real* mask, Real threshold),((cuComplex*)data,mask,threshold),{
    cuda1Idx();
    if(mask[index]>threshold) data[index].x = data[index].y = 0;
    })
cuFuncc(zeroEdge,(complexFormat* a, int n),(cuComplex* a, int n),((cuComplex*)a,n),{
    cudaIdx()
    if(x<n || x>=cuda_row-n || y < n || y >= cuda_column-n)
    a[index] = cuComplex();
    })
cuFunc(zeroEdge,(Real* a, int n),(a,n),{
    cudaIdx()
    if(x<n || x>=cuda_row-n || y < n || y >= cuda_column-n)
    a[index] = 0;
    })
cuFuncc(zeroEdgey,(complexFormat* a, int n),(cuComplex* a, int n),((cuComplex*)a,n),{
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
  randZeroWrap<<<numBlocks, threadsPerBlock>>>(addVar((cuComplex*)src, (cuComplex*)dest, state, ratio, step));
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
  padWrap<<<numBlocks, threadsPerBlock>>>(addVar((cuComplex*)src, (cuComplex*)dest, row, col, shiftx, shifty));
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
    cuda1Idx()
    int y = index%cuda_column;
    out[index] = object[index] * ((y+0.5)/cuda_column-0.5);
    })
