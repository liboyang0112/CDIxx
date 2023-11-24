#include "cudaConfig.h"
#include "cudaDefs.h"
#include "imgio.h"
#include <curand_kernel.h>
#include <iostream>
cudaVars* cudaVar = 0;
cudaVars* cudaVarLocal = 0;
dim3 numBlocks;
const dim3 threadsPerBlock = 256;
complexFormat *cudaData = 0;
cufftHandle *plan, *planR2C;
int2 cuda_imgsz = {0,0};
void cuMemManager::c_malloc(void*& ptr, size_t sz) { gpuErrchk(cudaMalloc((void**)&ptr, sz)); }
void cuMemManager::c_memset(void*& ptr, size_t sz) { gpuErrchk(cudaMemset(ptr, 0, sz)); }
cuMemManager memMngr;
void myMemcpyH2D(void* d, void* s, size_t sz){
  cudaMemcpy(d, s, sz, cudaMemcpyHostToDevice);
}
void myMemcpyD2D(void* d, void* s, size_t sz){
  cudaMemcpy(d, s, sz, cudaMemcpyDeviceToDevice);
}
void myMemcpyD2H(void* d, void* s, size_t sz){
  cudaMemcpy(d, s, sz, cudaMemcpyDeviceToHost);
}
void resize_cuda_image(int rows, int cols){
  cuda_imgsz.x = rows;
  cuda_imgsz.y = cols;
  numBlocks.x=(rows*cols-1)/threadsPerBlock.x+1;
  //numBlocks.y=(cols-1)/threadsPerBlock.x+1;
}
void init_cuda_image(int rcolor, Real scale){
  const int sz = sizeof(cudaVars);
  if(!cudaVar){
    cudaVar = (cudaVars*) memMngr.borrowCache(sz);
    cudaVarLocal = (cudaVars*) malloc(sz);
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
void* newRand(size_t sz){
  return memMngr.borrowCache(sz * sizeof(curandStateMRG32k3a));
}
using namespace std;
static int rows_fft, cols_fft, batch_fft;
__device__ __host__ bool rect::isInside(int x, int y){
  if(x > startx && x <= endx && y > starty && y <= endy) return true;
  return false;
}
__device__ __host__ bool C_circle::isInside(int x, int y){
  Real dr = hypot(Real(x-x0),Real(y-y0));
  if(dr < r) return true;
  return false;
}
void init_fft(int rows, int cols, int batch){
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
  cudaConvertFOWrap<<<numBlocks, threadsPerBlock>>>(cudaVar, cuda_imgsz.x, cuda_imgsz.y, (cuComplex*)data, (cuComplex*)(out==0?data:out));
}

template <typename T1, typename T2>
__global__ void assignValWrap(int cuda_row, int cuda_column, T1* out, T2* input){
  cuda1Idx()
	out[index] = input[index];
}
template <typename T1, typename T2>
void assignVal(T1* out, T2* input){
  assignValWrap<<<numBlocks,threadsPerBlock>>>(cuda_imgsz.x, cuda_imgsz.y,out,input);
}
template void assignVal<Real,Real>(Real*, Real*);
template void assignVal<Real,double>(Real*, double*);
template<> void assignVal<complexFormat, complexFormat>(complexFormat* out, complexFormat* input){
  assignValWrap<<<numBlocks,threadsPerBlock>>>(cuda_imgsz.x, cuda_imgsz.y,(cuComplex*)out,(cuComplex*)input);
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
  cropWrap<<<numBlocks, threadsPerBlock>>>(cudaVar, cuda_imgsz.x, cuda_imgsz.y, (cuComplex*)src, (cuComplex*)dest, row, col, midx, midy);
}


cuFuncc(multiplyShift,(complexFormat* object, Real shiftx, Real shifty),(cuComplex* object, Real shiftx, Real shifty),((cuComplex*)object,shiftx,shifty),{
    cudaIdx();
    Real phi = -2*M_PI*(shiftx*(x-cuda_row/2)/cuda_row+shifty*(y-cuda_column/2)/cuda_column);
    cuComplex tmp = {cos(phi),sin(phi)};
    object[index] = cuCmulf(object[index],tmp);
    })


void shiftWave(complexFormat* wave, Real shiftx, Real shifty){
  myFFT(wave, wave);
  cudaConvertFO(wave);
  multiplyShift(wave, shiftx, shifty);
  cudaConvertFO(wave);
  applyNorm(wave, 1./(cuda_imgsz.x*cuda_imgsz.y));
  myIFFT(wave, wave);
}

cuFuncc(rotateToReal,(complexFormat* data),(cuComplex* data),((cuComplex*)data),{
    cuda1Idx();
    data[index].x = cuCabsf(data[index]);
    data[index].y = 0;
    })

cuFuncc(removeImag,(complexFormat* data),(cuComplex* data),((cuComplex*)data),{
    cuda1Idx();
    data[index].y = 0;
    })

void shiftMiddle(complexFormat* wave){
  cudaConvertFO(wave);
  myFFT(wave, wave);
  rotateToReal(wave);
  applyNorm(wave, 1./(cuda_imgsz.x*cuda_imgsz.y));
  myIFFT(wave, wave);
  cudaConvertFO(wave);
}

__global__ void createGaussKernel(Real* data, int sz, Real sigma){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= sz*sz) return;
  int dx = idx/sz-(sz>>1);
  int dy = (idx%sz)-(sz>>1);
  data[idx] = exp(-Real(dx*dx+dy*dy)/(sigma*sigma));
}

void createGauss(Real* data, int sz, Real sigma){
  createGaussKernel<<<(sz*sz-1)/threadsPerBlock.x+1,threadsPerBlock>>>(data, sz, sigma);
}
void applyGaussConv(Real* input, Real* output, Real* gaussMem, Real sigma){
  int size = floor(sigma*6); // r=3 sigma to ensure the contribution outside kernel is negligible (0.01 of the maximum)
  size = size>>1;
  int width = (size<<1)+1;
  createGauss(gaussMem, width, sigma);
  applyConvolution((sq(width-1+16)+(width*width))*sizeof(Real), input, output, gaussMem, size, size);
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
    else tile[filltarget] = input[fillx*cuda_column+filly];
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

cuFuncc(applyNorm,(complexFormat* data, Real factor),(cuComplex* data, Real factor),((cuComplex*)data,factor),{
    cuda1Idx()
    data[index].x*=factor;
    data[index].y*=factor;
    })
cuFunc(applyNorm,(Real* data, Real factor),(data,factor),{
    cuda1Idx()
    data[index]*=factor;
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

cuFuncc(multiply,(complexFormat* src, complexFormat* target),(cuComplex* src, cuComplex* target),((cuComplex*)src,(cuComplex*)target),{
    cuda1Idx()
    src[index] = cuCmulf(src[index], target[index]);
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

cuFuncc(multiply,(complexFormat* store, complexFormat* src, complexFormat* target),(cuComplex* store, cuComplex* src, cuComplex* target),((cuComplex*)store,(cuComplex*)src,(cuComplex*)target),{
    cuda1Idx()
    store[index] = cuCmulf(src[index], target[index]);
    })

cuFunc(multiply,(Real* store, Real* src, Real* target),(store,src,target),{
    cuda1Idx()
    store[index] = src[index]* target[index];
    })

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

cuFunc(add,(Real* a, Real* b, Real c),(a,b,c),{
    cuda1Idx()
    a[index]+=b[index]*c;
    })

cuFunc(add,(Real* store, Real* a, Real* b, Real c),(store, a,b,c),{
    cuda1Idx()
    store[index] = a[index]+b[index]*c;
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

void readComplexWaveFront(const char* intensityFile, const char* phaseFile, Real* &d_intensity, Real* &d_phase, int &objrow, int &objcol){
  size_t sz = 0;
  if(intensityFile) {
    Real* intensity = readImage(intensityFile, objrow, objcol);
    sz = objrow*objcol*sizeof(Real);
    d_intensity = (Real*)memMngr.borrowCache(sz); //use the memory allocated;
    cudaMemcpy(d_intensity, intensity, sz, cudaMemcpyHostToDevice);
    ccmemMngr.returnCache(intensity);
  }
  if(phaseFile) {
    int tmprow,tmpcol;
    Real* phase = readImage(phaseFile, tmprow,tmpcol);
    if(!intensityFile) {
      sz = tmprow*tmpcol*sizeof(Real);
      objrow = tmprow;
      objcol = tmpcol;
    }
    d_phase = (Real*)memMngr.borrowCache(sz);
    size_t tmpsz = tmprow*tmpcol*sizeof(Real);
    if(tmpsz!=sz){
      Real* d_phasetmp = (Real*)memMngr.borrowCache(tmpsz);
      gpuErrchk(cudaMemcpy(d_phasetmp,phase,tmpsz,cudaMemcpyHostToDevice));
      resize_cuda_image(objrow, objcol);
      if(tmpsz > sz){
        crop(d_phasetmp, d_phase, tmprow, tmpcol);
      }else{
        pad(d_phasetmp, d_phase, tmprow, tmpcol);
      }
      memMngr.returnCache(d_phasetmp);
    }
    else {
      gpuErrchk(cudaMemcpy(d_phase,phase,sz,cudaMemcpyHostToDevice));
    }

    ccmemMngr.returnCache(phase);
  }
  gpuErrchk(cudaGetLastError());
}

cuFunc(initRand,(void* state, unsigned long long seed),(state,seed),{
    cuda1Idx()
    curand_init(seed,index,0,&((curandStateMRG32k3a*)state)[index]);
    })

cuFunc(applyPoissonNoise_WO,(Real* wave, Real noiseLevel, void* state, Real scale),
    (wave,noiseLevel,state,scale),{
    cuda1Idx()
    if(scale==0) scale = vars->scale;
    wave[index]=scale*(int(wave[index]*vars->rcolor/scale) + curand_poisson(&((curandStateMRG32k3a*)state)[index], noiseLevel)-noiseLevel)/vars->rcolor;
    })
cuFunc(ccdRecord, (uint16_t* data, Real* wave, int noiseLevel, void* state, Real exposure),
    (data,wave,noiseLevel,state,exposure),{
    cuda1Idx()
    int dataint = curand_poisson(&((curandStateMRG32k3a*)state)[index], noiseLevel) + vars->rcolor*wave[index]*exposure;
    if(dataint >= vars->rcolor) dataint = vars->rcolor-1;
    data[index] = dataint-noiseLevel;
    });
cuFuncc(ccdRecord, (uint16_t* data, complexFormat* wave, int noiseLevel, void* state, Real exposure),(uint16_t* data, cuComplex* wave, int noiseLevel, void* state, Real exposure),
    (data,(cuComplex*)wave,noiseLevel,state,exposure),{
    cuda1Idx()
    cuComplex tmp = wave[index];
    int dataint = curand_poisson(&((curandStateMRG32k3a*)state)[index], noiseLevel) + vars->rcolor*(tmp.x*tmp.x+tmp.y*tmp.y)*exposure;
    if(dataint >= vars->rcolor) dataint = vars->rcolor-1;
    data[index] = dataint-noiseLevel;
    });
cuFunc(ccdRecord, (Real* data, Real* wave, int noiseLevel, void* state, Real exposure),
    (data,wave,noiseLevel,state,exposure),{
    cuda1Idx()
    int dataint = curand_poisson(&((curandStateMRG32k3a*)state)[index], noiseLevel) + vars->rcolor*wave[index]*exposure;
    if(dataint >= vars->rcolor) dataint = vars->rcolor-1;
    data[index] = Real(dataint-noiseLevel)/vars->rcolor;
    });
cuFuncc(ccdRecord, (Real* data, complexFormat* wave, int noiseLevel, void* state, Real exposure),(Real* data, cuComplex* wave, int noiseLevel, void* state, Real exposure),
    (data,(cuComplex*)wave,noiseLevel,state,exposure),{
    cuda1Idx()
    cuComplex tmp = wave[index];
    int dataint = curand_poisson(&((curandStateMRG32k3a*)state)[index], noiseLevel) + vars->rcolor*(tmp.x*tmp.x+tmp.y*tmp.y)*exposure;
    if(dataint >= vars->rcolor) dataint = vars->rcolor-1;
    data[index] = Real(dataint-noiseLevel)/vars->rcolor;
    });
cuFuncc(ccdRecord, (complexFormat* data, complexFormat* wave, int noiseLevel, void* state, Real exposure),(cuComplex* data, cuComplex* wave, int noiseLevel, void* state, Real exposure),
    ((cuComplex*)data,(cuComplex*)wave,noiseLevel,state,exposure),{
    cuda1Idx()
    cuComplex tmp = wave[index];
    int dataint = curand_poisson(&((curandStateMRG32k3a*)state)[index], noiseLevel) + vars->rcolor*(tmp.x*tmp.x+tmp.y*tmp.y)*exposure;
    if(dataint >= vars->rcolor) dataint = vars->rcolor-1;
    data[index].x = Real(dataint-noiseLevel)/vars->rcolor;
    data[index].y = 0;
    });
cuFunc(applyPoissonNoise,(Real* wave, Real noiseLevel, void* state, Real scale),
    (wave,noiseLevel,state,scale),{
    cuda1Idx()
    curand_init(1,index,0,&((curandStateMRG32k3a*)state)[index]);
    if(scale==0) scale = vars->scale;
    wave[index]+=scale*(curand_poisson(&((curandStateMRG32k3a*)state)[index], noiseLevel)-noiseLevel)/vars->rcolor;
    })

cuFuncc(getMod,(Real* mod, complexFormat* amp),(Real* mod, cuComplex* amp),(mod,(cuComplex*)amp),{
    cuda1Idx()
    mod[index] = cuCabsf(amp[index]);
    })
cuFuncc(getReal,(Real* mod, complexFormat* amp),(Real* mod, cuComplex* amp),(mod,(cuComplex*)amp),{
    cuda1Idx()
    mod[index] = amp[index].x;
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
    cuComplex tmp = amp[index];
    store[index] = tmp.x*tmp.x+tmp.y*tmp.y > threshold*threshold;
    })

cuFunc(bitMap,(Real* store, Real* amp, Real threshold),(store,amp, threshold),{
    cuda1Idx()
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
    Real randphase = state?curand_uniform(&((curandStateMRG32k3a*)state)[index])*2*M_PI:0;
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
    Real randphase = state?curand_uniform(&((curandStateMRG32k3a*)state)[index])*2*M_PI:0;
    source[index].x = rat*cos(randphase);
    source[index].x = rat*sin(randphase);
    return;
    }
    rat /= mod;
    source[index].x *= rat;
    source[index].y *= rat;
    })

cuFuncc(applyMod,(complexFormat* source, Real* target, Real *bs, bool loose, int iter, int noiseLevel),(cuComplex* source, Real* target, Real *bs, bool loose, int iter, int noiseLevel),
    ((cuComplex*)source, target, bs, loose, iter, noiseLevel),{
    cuda1Idx()
    Real maximum = vars->scale*0.95;
    Real mod2 = target[index];
    if(mod2<0) mod2=0;
    if(loose && bs && bs[index]>0.5) {
    //if(iter > 500) return;
    //else mod2 = maximum+1;
    return;
    }
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
    source[index].x = (0.+val)/1.*sourcedata.x;
    source[index].y = (0.+val)/1.*sourcedata.y;
    })
cuFuncc(add,(complexFormat* a, complexFormat* b, Real c ),(cuComplex* a, cuComplex* b, Real c ),((cuComplex*)a,(cuComplex*)b,c),{
    cuda1Idx()
    a[index].x+=b[index].x*c;
    a[index].y+=b[index].y*c;
    })
cuFuncc(convertFOPhase, (complexFormat* data),(cuComplex* data),((cuComplex*)data),{
    cudaIdx()
    if((x+y)%2==1) {
    data[index].x = -data[index].x;
    data[index].y = -data[index].y;
    }
    })
cuFuncc(add,(complexFormat* store, complexFormat* a, complexFormat* b, Real c ),(cuComplex* store, cuComplex* a, cuComplex* b, Real c ),((cuComplex*)store,(cuComplex*)a,(cuComplex*)b,c),{
    cuda1Idx()
    store[index].x=a[index].x + b[index].x*c;
    store[index].y=a[index].y + b[index].y*c;
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
    Real randphase = curand_uniform(&((curandStateMRG32k3a*)state)[index])*2*M_PI;
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
cuFunc(stretch,(Real* src, Real* dest, Real rat, int prec),(src,dest,rat,prec),{
    cudaIdx()
    int targetx = Real(x-cuda_row/2)/rat+cuda_row/2;
    int targety = Real(y-cuda_column/2)/rat+cuda_column/2;
    Real f = cuda_row*cuda_column*rat*rat;
    dest[index] = 0;
    Real sum = 0;
    Real sum1 = 0;
    for(int tx = targetx - prec; tx < targetx+prec; tx++){
      Real phase = 2*M_PI*(Real(x-cuda_row/2)/rat-tx+cuda_row/2)/cuda_row;
      cuComplex factor1 = getFact(phase, cuda_row);
      for(int ty = targety - prec; ty < targety+prec; ty++){
        phase = 2*M_PI*(Real(y-cuda_column/2)/rat-ty+cuda_column/2)/cuda_column;
        cuComplex factor2 = getFact(phase, cuda_row);
        factor2 = cuCmulf(factor1,factor2);
        if(x == 1 && y == 1) {
        sum += factor2.x/f;
        sum1 += factor2.y/f;
        printf("%d, %d, %f, %f\n", tx, ty, factor2.x/f, factor2.y/f);
        }
        dest[index] += src[tx*cuda_row+ty]*factor2.x /f;
      }
    }
    if(x == 1 && y == 1) printf("sum: %f, %f, %f\n", sum, sum1, sqSum(sum, sum1));
    })
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
cuFunc(applyMask, (Real* data, Real* mask, Real threshold),(data,mask,threshold),{
  cuda1Idx();
  if(mask[index]<=threshold) data[index] = 0;
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
  padWrap<<<numBlocks, threadsPerBlock>>>(cudaVar, cuda_imgsz.x, cuda_imgsz.y, (cuComplex*)src, (cuComplex*)dest, row, col, shiftx, shifty);
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
