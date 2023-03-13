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
__global__ void fillRedundantR2C(cudaVars* vars, complexFormat* data, complexFormat* dataout, Real factor){
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
}

__global__ void applyConvolution(cudaVars* vars, Real *input, Real *output, Real* kernel, int kernelwidth, int kernelheight)
{
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  int cuda_row = vars->rows;
  int cuda_column = vars->cols;
  int tilewidth = kernelwidth*2+blockDim.x;
  extern __shared__ float tile[];
  if(threadIdx.x<blockDim.x/2+kernelwidth && threadIdx.y<blockDim.y/2+kernelheight)
    tile[threadIdx.x*(tilewidth)+threadIdx.y]=(x>=kernelwidth && y>=kernelheight)?input[(x-kernelwidth)*cuda_column+y-kernelheight]:0;
  if(threadIdx.x>=blockDim.x/2-kernelwidth && threadIdx.y<blockDim.y/2+kernelheight)
    tile[(threadIdx.x+2*kernelwidth)*(tilewidth)+threadIdx.y]=(x<cuda_row-kernelwidth && y>=kernelheight)?input[(x+kernelwidth)*cuda_column+y-kernelheight]:0;
  if(threadIdx.x<blockDim.x/2+kernelwidth && threadIdx.y>=blockDim.y/2-kernelheight)
    tile[threadIdx.x*(tilewidth)+threadIdx.y+2*kernelheight]=(x>=kernelwidth && y<cuda_column-kernelheight)?input[(x-kernelwidth)*cuda_column+y+kernelheight]:0;
  if(threadIdx.x>=blockDim.x/2-kernelwidth && threadIdx.y>=blockDim.y/2-kernelheight)
    tile[(threadIdx.x+2*kernelwidth)*(tilewidth)+threadIdx.y+2*kernelheight]=(x<cuda_row-kernelwidth && y<cuda_column-kernelheight)?input[(x+kernelwidth)*cuda_column+y+kernelheight]:0;
  if(x >= cuda_row || y >= cuda_column) return;
  int index = x*cuda_column+y;
  __syncthreads();
  int Idx = (threadIdx.x)*(tilewidth) + threadIdx.y;
  int IdxK = 0;
  Real n_output = 0;
  for(int x = -kernelwidth; x <= kernelwidth; x++){
    for(int y = -kernelheight; y <= kernelheight; y++){
      n_output+=tile[Idx++]*kernel[IdxK++];
    }
    Idx+=tilewidth-2*kernelheight-1;
  }
  output[index] = n_output;
}

__global__ void applyNorm(cudaVars* vars, complexFormat* data, Real factor){
  cudaIdx()
  data[index].x*=factor;
  data[index].y*=factor;
}
__global__ void multiplyReal(cudaVars* vars, Real* store, complexFormat* src, complexFormat* target){
  cudaIdx();
  store[index] = src[index].x*target[index].x;
}

__global__ void multiply(cudaVars* vars, complexFormat* src, complexFormat* target){
  cudaIdx()
  src[index] = cuCmulf(src[index], target[index]);
}
__global__ void forcePositive(cudaVars* vars, complexFormat* a){
  cudaIdx()
  if(a[index].x<0) a[index].x=0;
  a[index].y = 0;
}

__global__ void multiply(cudaVars* vars, complexFormat* store, complexFormat* src, complexFormat* target){
  cudaIdx()
  store[index] = cuCmulf(src[index], target[index]);
}

__global__ void extendToComplex(cudaVars* vars, Real* a, complexFormat* b){
  cudaIdx()
  b[index].x = a[index];
  b[index].y = 0;
}

__global__ void applyNorm(cudaVars* vars, Real* data, Real factor){
  cudaIdx()
  data[index]*=factor;
}
__global__ void add(cudaVars* vars, Real* a, Real* b, Real c){
  cudaIdx()
  a[index]+=b[index]*c;
}

__global__ void createWaveFront(cudaVars* vars, Real* d_intensity, Real* d_phase, complexFormat* objectWave, int row, int col, int shiftx, int shifty){
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
  objectWave[index].x = mod*cos(phase);
  objectWave[index].y = mod*sin(phase);
}

__global__ void createWaveFront(cudaVars* vars, Real* d_intensity, Real* d_phase, complexFormat* objectWave, Real oversampling, Real shiftx, Real shifty){
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
  objectWave[index].x = mod*cos(phase);
  objectWave[index].y = mod*sin(phase);
}

__global__ void initRand(cudaVars* vars, curandStateMRG32k3a *state, unsigned long long seed){
  cudaIdx()
  curand_init(seed,index,0,&state[index]);
}

__global__ void applyPoissonNoise_WO(cudaVars* vars, Real* wave, Real noiseLevel, curandStateMRG32k3a *state, Real scale){
  cudaIdx()
  if(scale==0) scale = vars->scale;
  wave[index]=scale*(int(wave[index]*vars->rcolor/scale) + curand_poisson(&state[index], noiseLevel)-noiseLevel)/vars->rcolor;
}

__global__ void applyPoissonNoise(cudaVars* vars, Real* wave, Real noiseLevel, curandStateMRG32k3a *state, Real scale){
  cudaIdx()
  curand_init(1,index,0,&state[index]);
  if(scale==0) scale = vars->scale;
  wave[index]+=scale*(curand_poisson(&state[index], noiseLevel)-noiseLevel)/vars->rcolor;
}

__global__ void getMod(cudaVars* vars, Real* mod, complexFormat* amp){
  cudaIdx()
  mod[index] = cuCabsf(amp[index]);
}
__global__ void getReal(cudaVars* vars, Real* mod, complexFormat* amp){
  cudaIdx()
  mod[index] = amp[index].x;
}
__global__ void getMod2(cudaVars* vars, Real* mod2, complexFormat* amp){
  cudaIdx()
  mod2[index] = pow(amp[index].x,2)+pow(amp[index].y,2);
}

__global__ void applyMod(cudaVars* vars, complexFormat* source, Real* target, Real *bs, bool loose, int iter, int noiseLevel){
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
}
__global__ void add(cudaVars* vars, complexFormat* a, complexFormat* b, Real c ){
  cudaIdx()
  a[index].x+=b[index].x*c;
  a[index].y+=b[index].y*c;
}
__global__ void applyRandomPhase(cudaVars* vars, complexFormat* wave, Real* beamstop, curandStateMRG32k3a *state){
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
}
