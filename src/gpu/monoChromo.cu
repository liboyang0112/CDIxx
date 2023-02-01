#include "monoChromo.h"
#include "cudaConfig.h"
#include "common.h"
#include <cub/device/device_reduce.cuh>
int nearestEven(Real x){
  return round(x/2)*2;
}
struct CustomSum
{
  __device__ __forceinline__
    Real operator()(const Real &a, const Real &b) const {
      return a+b;
    }
};

static Real findSum(Real* d_in, int num_items)
{
  Real *d_out = NULL;
  d_out = (Real*)memMngr.borrowCache(sizeof(Real));

  void            *d_temp_storage = NULL;
  size_t          temp_storage_bytes = 0;
  CustomSum sum_op;
  Real tmp = 0;
  gpuErrchk(cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, sum_op, tmp));
  d_temp_storage = memMngr.borrowCache(temp_storage_bytes);

  // Run
  gpuErrchk(cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, sum_op, tmp));
  Real output;
  cudaMemcpy(&output, d_out, sizeof(Real), cudaMemcpyDeviceToHost);

  if (d_out) memMngr.returnCache(d_out);
  if (d_temp_storage) memMngr.returnCache(d_temp_storage);
  return output;
}

__global__ void multiplyReal(complexFormat* a, complexFormat* b, Real* c){
  cudaIdx();
  c[index] = a[index].x*b[index].x;
}

void monoChromo::init(int nrow, int ncol, int nlambda_, Real* lambdas_, Real* spectra_){
  nlambda = nlambda_;
  spectra = spectra_;
  row = nrow;
  column = ncol;
  rows = (int*)ccmemMngr.borrowCache(sizeof(int)*nlambda);
  cols = (int*)ccmemMngr.borrowCache(sizeof(int)*nlambda);
  plt.init(row,column);
  locplan = ccmemMngr.borrowCache(sizeof(cufftHandle)*nlambda);
  for(int i = 0; i < nlambda; i++){
    rows[i] = nearestEven(row*lambdas_[i]);
    cols[i] = nearestEven(column*lambdas_[i]);
    new(&(((cufftHandle*)locplan)[i]))cufftHandle();
    cufftPlan2d ( &(((cufftHandle*)locplan)[i]), rows[i], cols[i], FFTformat);
  }
}
void monoChromo::init(int nrow, int ncol, Real* lambdasi, Real* spectrumi, Real endlambda){
  row = nrow;
  column = ncol;
  Real currentLambda = 1;
  int currentPoint = 0;
  int jump = 1;
  Real stepsize = 2./row*jump;
  nlambda = (endlambda-1)/stepsize;
  spectra = (Real*) ccmemMngr.borrowCache(nlambda*sizeof(Real));
  int i = 0;
  Real tot = 0;
  while(currentLambda < endlambda){
    int count = 0;
    Real intensity = 0;
    while(lambdasi[currentPoint] < currentLambda+stepsize/2){
      count++;
      intensity += spectrumi[currentPoint];
      currentPoint++;
    }
    if(count >=2 ){ //use average
      spectra[i] = intensity/count;
    }else{ //use interpolation
      if(currentLambda == lambdasi[currentPoint-1]){
        spectra[i] = spectrumi[currentPoint-1];
      }
      else if(currentLambda > lambdasi[currentPoint-1]){
        Real dlambda = lambdasi[currentPoint]-lambdasi[currentPoint-1];
        Real dx = (currentLambda - lambdasi[currentPoint-1])/dlambda;
        spectra[i] = spectrumi[currentPoint-1]*(1-dx) + spectrumi[currentPoint]*(dx);
      }else{
        Real dlambda = lambdasi[currentPoint-1]-lambdasi[currentPoint-2];
        Real dx = (currentLambda - lambdasi[currentPoint-2])/dlambda;
        spectra[i] = spectrumi[currentPoint-2]*(1-dx) + spectrumi[currentPoint-1]*(dx);
      }
    }
    tot+=spectra[i];
    i++;
    currentLambda+=stepsize;
  }
  rows = (int*)ccmemMngr.borrowCache(sizeof(int)*nlambda);
  cols = (int*)ccmemMngr.borrowCache(sizeof(int)*nlambda);
  plt.init(row,column);
  locplan = ccmemMngr.borrowCache(sizeof(cufftHandle)*nlambda);
  for(int i = 0; i < nlambda; i++){
    rows[i] = row+i*2*jump;
    cols[i] = column+i*2*jump;
    printf("%d: (%d,%d)=%f\n",i, rows[i],cols[i],spectra[i]/=tot);
    new(&(((cufftHandle*)locplan)[i]))cufftHandle();
    cufftPlan2d ( &(((cufftHandle*)locplan)[i]), rows[i], cols[i], FFTformat);
  }
}
void monoChromo::generateMWL(void* d_input, void* d_patternSum, void* single, Real oversampling){
  Real *d_pattern = (Real*) memMngr.borrowCache(row*column*sizeof(Real));
  complexFormat *d_intensity = (complexFormat*)memMngr.borrowCache(rows[nlambda-1]*cols[nlambda-1]*sizeof(complexFormat));
  complexFormat* d_patternAmp = (complexFormat*)memMngr.borrowCache(row*column*sizeof(Real)*2);
  for(int i = 0; i < nlambda; i++){
    int thisrow = rows[i];
    int thiscol = cols[i];
    init_cuda_image(thisrow, thiscol, 65536, 1);
    cudaF(createWaveFront)((Real*)d_input, 0, (complexFormat*)d_intensity, row/oversampling, column/oversampling);
    myCufftExec( ((cufftHandle*)locplan)[i], d_intensity,d_intensity, CUFFT_FORWARD);
    cudaF(cudaConvertFO)(d_intensity);
    init_cuda_image(row, column, 65536, 1);
    cudaF(crop)(d_intensity,d_patternAmp,thisrow,thiscol);
    cudaF(applyNorm)(d_patternAmp, sqrt(spectra[i])/sqrt(thiscol*thisrow));
    if(i==0) {
      cudaF(getMod2)((Real*)d_patternSum, d_patternAmp);
      if(single!=0) {
        cudaF(extendToComplex)((Real*)d_patternSum, (complexFormat*)single);
        cudaF(applyNorm)((complexFormat*)single, 1./spectra[i]);
      }
    }else{
      cudaF(getMod2)(d_pattern, d_patternAmp);
      cudaF(add)((Real*)d_patternSum, (Real*)d_pattern, 1);
    }
  }
  memMngr.returnCache(d_pattern);
  memMngr.returnCache(d_intensity);
  memMngr.returnCache(d_patternAmp);
}
__global__ void printpix(Real* input, int x, int y){
  printf("%f", input[x*cuda_column+y]);
}
__global__ void printpixreal(complexFormat* input, int x, int y){
  printf("%f,", input[x*cuda_column+y].x);
}
void monoChromo::solveMWL(void* d_input, void* d_output, void* initial, int nIter, bool updateX, bool updateA)
{
  int sz = row*column*sizeof(complexFormat);
  float step_a = 0.00004;
  if(nlambda<0) printf("nlambda not initialized: %d\n",nlambda);
  Real dt = 2;
  Real friction = 0.1;
  cudaMemcpy(d_output, initial? initial : d_input, sz, cudaMemcpyDeviceToDevice);
  init_fft(row,column);
  complexFormat *deltab = (complexFormat*)memMngr.borrowCache(sz);
  complexFormat *deltabprev = (complexFormat*)memMngr.borrowCache(sz);
  Real *multiplied = (Real*)memMngr.borrowCache(sz/2);
  complexFormat *fbi = (complexFormat*)memMngr.borrowCache(sz);
  complexFormat *fftb = (complexFormat*)memMngr.borrowCache(sz);
  complexFormat *momentum = (complexFormat*)memMngr.borrowCache(sz);
  complexFormat *padded = (complexFormat*) memMngr.borrowCache(sizeof(complexFormat)*rows[nlambda-1]*cols[nlambda-1]);
  Real *momentum_a = (Real*) ccmemMngr.borrowCache(nlambda*sizeof(Real));
  Real *norm_a = (Real*) ccmemMngr.borrowCache(nlambda*sizeof(Real));
  Real mod2ref = 0;
  if(!updateX){
    cudaF(getMod2)(multiplied, (complexFormat*)initial);
    mod2ref = findSum(multiplied, row*column);
    if(mod2ref<=0 || mod2ref!=mod2ref) abort();
  }
  memset(momentum_a,0,nlambda*sizeof(Real));
  memset(norm_a,0,nlambda*sizeof(Real));
  norm_a[0] = 1;
  Real stepsize = 0.5;
  cudaMemset(momentum, 0, sz/sizeof(char));
  for(int i = 0; i < nIter; i++){
    Real maxstepsize = 0;
    bool updateAIter = (updateA&&(i%5==0) || updateX==0) && (i > 0);
    auto tmp_swap = deltabprev;
    deltabprev = deltab;
    deltab = tmp_swap;
    Real sumspect = 0;
    cudaMemcpy(deltab, d_input, sz, cudaMemcpyDeviceToDevice);
    myCufftExec( *plan, (complexFormat*)d_output, fftb, CUFFT_INVERSE);
    cudaF(cudaConvertFO)(fftb);
    cudaF(add)(deltab, (complexFormat*)d_output, -spectra[0]);
    if(updateAIter) {
      cudaF(multiplyReal)(deltabprev, (complexFormat*)d_output, multiplied);
      Real step = step_a*findSum(multiplied, row*column);
      momentum_a[0] += step*dt;
      momentum_a[0]*=(1-friction*dt);
      //spectra[0]+=step;//*spectra[0];
      spectra[0]+=momentum_a[0]*dt;//*spectra[0];
      //if(fabs(step) > maxstepsize) maxstepsize = fabs(step);
      if(spectra[0]<=0) spectra[0] = 1e-6;
      if(spectra[0]>0.03) spectra[0] = 0.03;
      sumspect+=spectra[0];
    }
    for(int j = 1; j < nlambda; j++){
      if(spectra[j]<=0) continue;
      size_t N = rows[j]*cols[j];
      init_cuda_image(rows[j], cols[j], 65536, 1);
      cudaF(pad)(fftb, padded, row, column);
      cudaF(cudaConvertFO)(padded);
      cudaF(applyNorm)(padded, 1./N);
      myCufftExec( ((cufftHandle*)locplan)[j], padded, padded, CUFFT_FORWARD);
      init_cuda_image(row, column, 65536, 1);
      cudaF(crop)(padded, fbi, rows[j], cols[j]);
      cudaF(add)(deltab, fbi, -spectra[j]);
      if(updateAIter) {
        if(!norm_a[j]){
          cudaF(getMod2)(multiplied, fbi);
          norm_a[j] = sqrt(mod2ref/findSum(multiplied, row*column));
          printf("norm[%d] = %f\n",j,norm_a[j]);
          if(norm_a[j]<=0 || norm_a[j]!=norm_a[j]) abort();
        }
        //cudaF(applyNorm)(fbi, 1+spectra[j]);
        //cudaF(add)(fbi, (complexFormat*)d_input, -1);
        //cudaF(add)(fbi, deltabprev, 1);
        cudaF(multiplyReal)(deltabprev, fbi, multiplied);
        //printf("middle[%d] = ",j);
        //printpixreal<<<{1,1},{1,1}>>>(deltabprev, row/2, column/2);
        //printpixreal<<<{1,1},{1,1}>>>(fbi, row/2, column/2);
        Real step = step_a*norm_a[j]*findSum(multiplied,row*column);//*(1-spectra[j]);
        //if(i > 100 && j == 86){
        //plt.plotComplex(deltabprev, REAL, 0, 1, "deltabprev", 1);
        //plt.plotComplex(fbi, REAL, 0, 1, "fbi", 1);
        //plt.plotFloat(multiplied, MOD, 0, 1, "multiplied", 1);
        //exit(0);
        //}
        if(fabs(step) > maxstepsize) maxstepsize = fabs(step);
        //if(i > 0){
        momentum_a[j] += step*dt;
        momentum_a[j]*=(1-friction*dt);
        //printf("\nmomentum[%d]=%f+%f=%f\n",j, momentum_a[j], step, momentum_a[j]+step);
        //printf("spectra[%d]=%f+%f=%f\n",j, spectra[j], step, spectra[j]+step);
        //if(j==nlambda-1 && i == 5) exit(0);
        if(fabs(step) > 1e2) exit(0);
        //}
        spectra[j]+=momentum_a[j]*dt;//*spectra[j];
        //spectra[j]+=step;//*spectra[j];
        if(spectra[j]<=0) spectra[j] = 1e-6;
        if(spectra[j]>0.03) spectra[j] = 0.03;
        sumspect+=spectra[j];
      }
    }
    if(updateAIter) { //adjust step size
      if(maxstepsize >0){
        step_a*=3e-5/maxstepsize;
      }
      //}else if(maxstepsize > 0.03)
        //step_a*=1e-4/maxstepsize;
      if(step_a > 0.004) step_a = 0.004;
    }
    if(updateAIter){
      //if(sumspect >= 1.1)
        for(int j = 0; j < nlambda; j++){
        spectra[j]-=(sumspect-1)/nlambda;
        if(spectra[j]<=0) spectra[j] = 1e-6;
        if(spectra[j]>0.03) spectra[j] = 0.03;
        }
      //    spectra[j]*=1.1/sumspect;
      //  }
      //if(sumspect <= 0.9)
      //  for(int j = 0; j < nlambda; j++){
      //    spectra[j]*=0.9/sumspect;
      //  }
    }
    if(updateX){
      cudaF(add)((complexFormat*)momentum, deltab, stepsize*spectra[0]);
      if(i==nIter-1) {
        plt.plotComplex(deltab, MOD, 0, 1, "residual", 1);
        cudaF(add)((complexFormat*)d_input, deltab, -1);
        break;
      }
      for(int j = 1; j < nlambda; j++){
        if(spectra[j]<=0) continue;
        init_cuda_image(rows[j], cols[j], 65536, 1);
        cudaF(pad)((complexFormat*)deltab, padded, row, column);
        myCufftExec( ((cufftHandle*)locplan)[j], padded, padded, CUFFT_INVERSE);
        cudaF(cudaConvertFO)(padded);
        init_cuda_image(row, column, 65536, 1);
        cudaF(crop)(padded, fbi, rows[j], cols[j]);
        cudaF(cudaConvertFO)(fbi);
        cudaF(applyNorm)(fbi, 1./(row*column));
        myCufftExec( *plan, fbi, fbi, CUFFT_FORWARD);
        cudaF(add)((complexFormat*)momentum, fbi, stepsize*spectra[j]);
      }
      cudaF(applyNorm)((complexFormat*)momentum, 1-friction*dt);
      cudaF(add)((complexFormat*)d_output, momentum, dt);
      cudaF(forcePositive)((complexFormat*)d_output);

    }
  }
  memMngr.returnCache(deltabprev);
  memMngr.returnCache(momentum);
  memMngr.returnCache(padded);
  memMngr.returnCache(fbi);
  memMngr.returnCache(fftb);
  memMngr.returnCache(deltab);
}
