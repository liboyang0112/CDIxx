#include "cudaConfig.h"
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
  int cuda_row = vars->rows;
  int cuda_column = vars->cols;
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

cuFunc(multiply,(complexFormat* store, complexFormat* src, complexFormat* target),(store,src,target),{
  cudaIdx()
  store[index] = cuCmulf(src[index], target[index]);
})

cuFunc(extendToComplex,(Real* a, complexFormat* b),(a,b),{
  cudaIdx()
  b[index].x = a[index];
  b[index].y = 0;
})

cuFunc(applyNorm,(Real* data, Real factor),(data,factor),{
  cudaIdx()
  data[index]*=factor;
})
cuFunc(add,(Real* a, Real* b, Real c),(a,b,c),{
  cudaIdx()
  a[index]+=b[index]*c;
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
  Real mod = sqrtf(max(0.,d_intensity[targetindex]));
  Real phase = d_phase? (d_phase[targetindex]-0.5)*2*M_PI : 0;
  //Real phase = d_phase? (d_phase[targetindex]-0.5) : 0;
  objectWave[index].x = mod*cos(phase);
  objectWave[index].y = mod*sin(phase);
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
  Real mod = sqrtf(max(0.,d_intensity[targetindex]));
  Real phase = d_phase? (d_phase[targetindex]-0.5)*2*M_PI : 0;
  //Real phase = d_phase? (d_phase[targetindex]-0.5) : 0;
  objectWave[index].x = mod*cos(phase);
  objectWave[index].y = mod*sin(phase);
})

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
cuFunc(getMod2,(Real* mod2, complexFormat* amp),(mod2,amp),{
  cudaIdx()
  complexFormat tmp = amp[index];
  mod2[index] = tmp.x*tmp.x + tmp.y*tmp.y;
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
  Real tolerance = 0;//(1.+sqrtf(noiseLevel))*vars->scale/vars->rcolor; // fluctuation caused by bit depth and noise
  complexFormat sourcedata = source[index];
  Real ratiox = 1;
  Real ratioy = 1;
  Real srcmod2 = sourcedata.x*sourcedata.x + sourcedata.y*sourcedata.y;
  if(mod2>=maximum) {
    if(loose) mod2 = max(maximum,srcmod2);
    else tolerance*=1000;
  }
  Real diff = mod2-srcmod2;
  if(diff>tolerance){
    ratioy=ratiox = sqrt((mod2-tolerance)/srcmod2);
  }else if(diff < -tolerance ){
    ratioy=ratiox = sqrt((mod2+tolerance)/srcmod2);
  }
  if(srcmod2 == 0){
    ratiox = sqrt(mod2);
    ratioy = 0;
  }
  source[index].x = ratiox*sourcedata.x;
  source[index].y = ratioy*sourcedata.y;
})
cuFunc(add,(complexFormat* a, complexFormat* b, Real c ),(a,b,c),{
  cudaIdx()
  a[index].x+=b[index].x*c;
  a[index].y+=b[index].y*c;
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
    Real randphase = curand_uniform(&state[index]);
    tmp.x = mod*cos(randphase);
    tmp.y = mod*sin(randphase);
  }
  wave[index] = tmp;
})
