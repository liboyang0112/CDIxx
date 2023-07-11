#include "cudaConfig.h"
#include "common.h"
#include <iostream>
using namespace std;
static int rows_fft, cols_fft;
void init_fft(int rows, int cols){
  if(rows!=rows_fft||cols!=cols_fft){
    if(!plan){
      plan = new cufftHandle();
      planR2C = new cufftHandle();
    }else{
      cufftDestroy(*plan);
      cufftDestroy(*planR2C);
    }
    cufftPlan2d ( plan, rows, cols, FFTformat);
    cufftPlan2d ( planR2C, rows, cols, FFTformatR2C);
    cols_fft = cols;
    rows_fft = rows;
  }
}

cuFunc(multiplyShift,(complexFormat* object, Real shiftx, Real shifty),(object,shiftx,shifty),{
  cudaIdx();
  Real phi = -2*M_PI*(shiftx*(x-cuda_row/2)/cuda_row+shifty*(y-cuda_column/2)/cuda_column);
  complexFormat tmp = {cos(phi),sin(phi)};
  object[index] = cuCmulf(object[index],tmp);
})


void shiftWave(complexFormat* wave, Real shiftx, Real shifty){
  myCufftExec( *plan, wave, wave, CUFFT_FORWARD);
  cudaConvertFO(wave);
  multiplyShift(wave, shiftx, shifty);
  cudaConvertFO(wave);
  applyNorm(wave, 1./(cuda_imgsz.x*cuda_imgsz.y));
  myCufftExec( *plan, wave, wave, CUFFT_INVERSE);
}

cuFunc(rotateToReal,(complexFormat* data),(data),{
  cudaIdx();
  data[index].x = cuCabsf(data[index]);
  data[index].y = 0;
})

cuFunc(removeImag,(complexFormat* data),(data),{
  cudaIdx();
  data[index].y = 0;
})

void shiftMiddle(complexFormat* wave){
  cudaConvertFO(wave);
  myCufftExec( *plan, wave, wave, CUFFT_FORWARD);
  rotateToReal(wave);
  applyNorm(wave, 1./(cuda_imgsz.x*cuda_imgsz.y));
  myCufftExec( *plan, wave, wave, CUFFT_INVERSE);
  cudaConvertFO(wave);
}

__global__ void createGaussKernel(Real* data, int sz, Real sigma){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x >= sz || y >= sz) return;
  int dx = x-(sz>>1);
  int dy = y-(sz>>1);
  data[x*sz+y] = exp(-Real(dx*dx+dy*dy)/(sigma*sigma));
}

void createGauss(Real* data, int sz, Real sigma){
  dim3 nblk;
  nblk.x = nblk.y =(sz-1)/threadsPerBlock.x+1;
  createGaussKernel<<<nblk,threadsPerBlock>>>(data, sz, sigma);
}
void applyGaussConv(Real* input, Real* output, Real* gaussMem, Real sigma){
  int size = floor(sigma*6); // r=3 sigma to ensure the contribution outside kernel is negligible (0.01 of the maximum)
  size = size>>1;
  int width = (size<<1)+1;
  createGauss(gaussMem, width, sigma);
  applyConvolution((pow(width-1+threadsPerBlock.x,2)+(width*width))*sizeof(Real), input, output, gaussMem, size, size);
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

cuFunc(applyNorm,(complexFormat* data, Real factor),(data,factor),{
  cudaIdx()
  data[index].x*=factor;
  data[index].y*=factor;
})
cuFunc(applyNorm,(Real* data, Real factor),(data,factor),{
  cudaIdx()
  data[index]*=factor;
})
cuFunc(interpolate,(Real* out, Real* data0, Real* data1, Real dx),(out, data0,data1,dx),{
  cudaIdx()
  out[index] = data0[index]*(1-dx) + data1[index]*dx;
})
cuFunc(interpolate,(complexFormat* out, complexFormat* data0, complexFormat* data1, Real dx),(out, data0,data1,dx),{
  cudaIdx()
  out[index].x = data0[index].x*(1-dx) + data1[index].x*dx;
  out[index].y = data0[index].y*(1-dx) + data1[index].y*dx;
})
cuFunc(adamUpdateV,(Real* v, Real* grad, Real beta2),(v,grad,beta2),{
  cudaIdx()
  Real tmp = grad[index];
  v[index] = tmp*tmp*(1-beta2) + beta2*v[index];
})
cuFunc(adamUpdateV,(Real* v, complexFormat* grad, Real beta2),(v,grad,beta2),{
  cudaIdx()
  Real tmp = grad[index].x;
  v[index] = tmp*tmp*(1-beta2) + beta2*v[index];
})
cuFunc(adamUpdate,(complexFormat* xn, complexFormat* m, Real* v, Real lr, Real eps),(xn,m,v,lr,eps),{
  cudaIdx()
  xn[index].x += lr*m[index].x/(sqrt(v[index])+eps);
})
cuFunc(ceiling,(complexFormat* data, Real ceilval),(data,ceilval),{
  cudaIdx()
  Real factor = ceilval/hypot(data[index].x, data[index].y);
  if(factor>1) return;
  data[index].x*=factor;
  data[index].y*=factor;
})
cuFunc(multiplyReal,(Real* store, complexFormat* src, complexFormat* target),(store,src,target),{
  cudaIdx();
  store[index] = src[index].x*target[index].x;
})

cuFunc(multiply,(complexFormat* src, complexFormat* target),(src,target),{
  cudaIdx()
  src[index] = cuCmulf(src[index], target[index]);
})
cuFunc(forcePositive,(complexFormat* a),(a),{
  cudaIdx()
  if(a[index].x<0) a[index].x=0;
  a[index].y = 0;
})

cuFunc(forcePositive,(Real* a),(a),{
  cudaIdx()
  if(a[index]<0) a[index]=0;
})

cuFunc(multiply,(complexFormat* store, complexFormat* src, complexFormat* target),(store,src,target),{
  cudaIdx()
  store[index] = cuCmulf(src[index], target[index]);
})

cuFunc(extendToComplex,(Real* a, complexFormat* b),(a,b),{
  cudaIdx()
  b[index].x = a[index];
  b[index].y = 0;
})

cuFunc(add,(Real* a, Real* b, Real c),(a,b,c),{
  cudaIdx()
  a[index]+=b[index]*c;
})

cuFunc(add,(Real* store, Real* a, Real* b, Real c),(store, a,b,c),{
  cudaIdx()
  store[index] = a[index]+b[index]*c;
})

cuFunc(createWaveFront,(Real* d_intensity, Real* d_phase, complexFormat* objectWave, int row, int col, int shiftx, int shifty),(d_intensity,d_phase,objectWave,row,col,shiftx,shifty),{
  cudaIdx()
  int marginx = (cuda_row-row)/2+shiftx;
  int marginy = (cuda_column-col)/2+shifty;
  if(x<marginx || x >= marginx+row || y < marginy || y >= marginy+col){
    objectWave[index].x = objectWave[index].y = 0;
    return;
  }
  int targetindex = (x-marginx)*col + y-marginy;
  Real mod = 1;
  if(d_intensity) mod = sqrtf(max(0.,d_intensity[targetindex]));
  //Real phase = d_phase? (d_phase[targetindex]-0.5) : 0;
  if(d_phase){
    Real phase = (d_phase[targetindex]-0.5)*2*M_PI;
    objectWave[index].x = mod*cos(phase);
    objectWave[index].y = mod*sin(phase);
  }else{
    objectWave[index].x = mod;
    objectWave[index].y = 0;
  }
})

cuFunc(createWaveFront,(Real* d_intensity, Real* d_phase, complexFormat* objectWave, Real oversampling, Real shiftx, Real shifty),(d_intensity,d_phase,objectWave,oversampling,shiftx,shifty),{
  cudaIdx()
  Real marginratio = (1-1./oversampling)/2;
  int marginx = (marginratio+shiftx)*cuda_row;
  int marginy = (marginratio+shifty)*cuda_column;
  if(x<marginx || x >= cuda_row-marginx || y < marginy || y >= cuda_column-marginy){
    objectWave[index].x = objectWave[index].y = 0;
    return;
  }
  int targetindex = (x-marginx)*ceil(cuda_column/oversampling) + y-marginy;
  Real mod = 1;
  if(d_intensity) mod = sqrtf(max(0.,d_intensity[targetindex]));
  Real phase = d_phase? (d_phase[targetindex]-0.5)*2*M_PI : 0;
  //Real phase = d_phase? (d_phase[targetindex]-0.5) : 0;
  objectWave[index].x = mod*cos(phase);
  objectWave[index].y = mod*sin(phase);
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

cuFunc(initRand,(curandStateMRG32k3a *state, unsigned long long seed),(state,seed),{
  cudaIdx()
  curand_init(seed,index,0,&state[index]);
})

cuFunc(applyPoissonNoise_WO,(Real* wave, Real noiseLevel, curandStateMRG32k3a *state, Real scale),
  (wave,noiseLevel,state,scale),{
  cudaIdx()
  if(scale==0) scale = vars->scale;
  wave[index]=scale*(int(wave[index]*vars->rcolor/scale) + curand_poisson(&state[index], noiseLevel)-noiseLevel)/vars->rcolor;
})
cuFunc(ccdRecord, (uint16_t* data, Real* wave, int noiseLevel, curandStateMRG32k3a *state, Real exposure),
  (data,wave,noiseLevel,state,exposure),{
  cudaIdx()
  int dataint = curand_poisson(&state[index], noiseLevel) + vars->rcolor*wave[index]*exposure;
  if(dataint >= vars->rcolor) dataint = vars->rcolor-1;
  data[index] = dataint-noiseLevel;
});
cuFunc(ccdRecord, (uint16_t* data, complexFormat* wave, int noiseLevel, curandStateMRG32k3a *state, Real exposure),
  (data,wave,noiseLevel,state,exposure),{
  cudaIdx()
  complexFormat tmp = wave[index];
  int dataint = curand_poisson(&state[index], noiseLevel) + vars->rcolor*(tmp.x*tmp.x+tmp.y*tmp.y)*exposure;
  if(dataint >= vars->rcolor) dataint = vars->rcolor-1;
  data[index] = dataint-noiseLevel;
});
cuFunc(ccdRecord, (Real* data, Real* wave, int noiseLevel, curandStateMRG32k3a *state, Real exposure),
  (data,wave,noiseLevel,state,exposure),{
  cudaIdx()
  int dataint = curand_poisson(&state[index], noiseLevel) + vars->rcolor*wave[index]*exposure;
  if(dataint >= vars->rcolor) dataint = vars->rcolor-1;
  data[index] = Real(dataint-noiseLevel)/vars->rcolor;
});
cuFunc(ccdRecord, (Real* data, complexFormat* wave, int noiseLevel, curandStateMRG32k3a *state, Real exposure),
  (data,wave,noiseLevel,state,exposure),{
  cudaIdx()
  complexFormat tmp = wave[index];
  int dataint = curand_poisson(&state[index], noiseLevel) + vars->rcolor*(tmp.x*tmp.x+tmp.y*tmp.y)*exposure;
  if(dataint >= vars->rcolor) dataint = vars->rcolor-1;
  data[index] = Real(dataint-noiseLevel)/vars->rcolor;
});
cuFunc(ccdRecord, (complexFormat* data, complexFormat* wave, int noiseLevel, curandStateMRG32k3a *state, Real exposure),
  (data,wave,noiseLevel,state,exposure),{
  cudaIdx()
  complexFormat tmp = wave[index];
  int dataint = curand_poisson(&state[index], noiseLevel) + vars->rcolor*(tmp.x*tmp.x+tmp.y*tmp.y)*exposure;
  if(dataint >= vars->rcolor) dataint = vars->rcolor-1;
  data[index].x = Real(dataint-noiseLevel)/vars->rcolor;
  data[index].y = 0;
});
cuFunc(applyPoissonNoise,(Real* wave, Real noiseLevel, curandStateMRG32k3a *state, Real scale),
  (wave,noiseLevel,state,scale),{
  cudaIdx()
  curand_init(1,index,0,&state[index]);
  if(scale==0) scale = vars->scale;
  wave[index]+=scale*(curand_poisson(&state[index], noiseLevel)-noiseLevel)/vars->rcolor;
})

cuFunc(getMod,(Real* mod, complexFormat* amp),(mod,amp),{
  cudaIdx()
  mod[index] = cuCabsf(amp[index]);
})
cuFunc(getReal,(Real* mod, complexFormat* amp),(mod,amp),{
  cudaIdx()
  mod[index] = amp[index].x;
})
cuFunc(getImag,(Real* mod, complexFormat* amp),(mod,amp),{
  cudaIdx()
  mod[index] = amp[index].y;
})
cuFunc(assignReal,(Real* mod, complexFormat* amp),(mod,amp),{
  cudaIdx()
  amp[index].x = mod[index];
})
cuFunc(assignImag,(Real* mod, complexFormat* amp),(mod,amp),{
  cudaIdx()
   amp[index].y = mod[index];
})
cuFunc(getMod2,(Real* mod2, complexFormat* amp),(mod2,amp),{
  cudaIdx()
  complexFormat tmp = amp[index];
  mod2[index] = tmp.x*tmp.x + tmp.y*tmp.y;
})

cuFunc(bitMap,(Real* store, complexFormat* amp, Real threshold),(store,amp, threshold),{
  cudaIdx()
  complexFormat tmp = amp[index];
  store[index] = tmp.x*tmp.x+tmp.y*tmp.y > threshold*threshold;
})

cuFunc(bitMap,(Real* store, Real* amp, Real threshold),(store,amp, threshold),{
  cudaIdx()
  store[index] = amp[index] > threshold;
})

cuFunc(applyThreshold,(Real* store, Real* input, Real threshold),(store,input,threshold),{
  cudaIdx()
  store[index] = input[index] > threshold? input[index] : 0;
})

cuFunc(linearConst,(Real* store, Real* data, Real fact, Real shift),(store, data, fact, shift),{
  cudaIdx();
  store[index] = fact*data[index]+shift;
})

cuFunc(applyModAbs,(complexFormat* source, Real* target),(source, target),{
  cudaIdx();
  Real mod = hypot(source[index].x, source[index].y);
  Real rat = sqrt(target[index]);
  if(mod==0) {
    source[index].x = rat;
    source[index].y = 0;
  }
  rat /= mod;
  source[index].x *= rat;
  source[index].y *= rat;
})
cuFunc(applyMod,(complexFormat* source, Real* target, Real *bs, bool loose, int iter, int noiseLevel),
  (source, target, bs, loose, iter, noiseLevel),{
  cudaIdx()
  Real maximum = vars->scale*0.95;
  Real mod2 = target[index];
  if(mod2<0) mod2=0;
  if(loose && bs && bs[index]>0.5) {
    //if(iter > 500) return;
    //else mod2 = maximum+1;
    return;
  }
  Real tolerance = (1.+sqrtf(noiseLevel))*vars->scale/vars->rcolor; // fluctuation caused by bit depth and noise
  complexFormat sourcedata = source[index];
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
cuFunc(add,(complexFormat* a, complexFormat* b, Real c ),(a,b,c),{
  cudaIdx()
  a[index].x+=b[index].x*c;
  a[index].y+=b[index].y*c;
})
cuFunc(convertFOPhase, (complexFormat* data),(data),{
  cudaIdx()
  if((x+y)%2==1) {
    data[index].x = -data[index].x;
    data[index].y = -data[index].y;
  }
})
cuFunc(add,(complexFormat* store, complexFormat* a, complexFormat* b, Real c ),(store,a,b,c),{
  cudaIdx()
  store[index].x=a[index].x + b[index].x*c;
  store[index].y=a[index].y + b[index].y*c;
})
cuFunc(applyRandomPhase,(complexFormat* wave, Real* beamstop, curandStateMRG32k3a *state),
 (wave, beamstop, state),{
  cudaIdx()
  complexFormat tmp = wave[index];
  if(beamstop && beamstop[index]>vars->threshold) {
    tmp.x = tmp.y = 0;
  }
  else{
    Real mod = cuCabsf(wave[index]);
    Real randphase = curand_uniform(&state[index])*2*M_PI;
    tmp.x = mod*cos(randphase);
    tmp.y = mod*sin(randphase);
  }
  wave[index] = tmp;
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

cuFunc(padinner, (complexFormat* src, complexFormat* dest, int row, int col, Real norm), (src, dest, row, col, norm),{
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

