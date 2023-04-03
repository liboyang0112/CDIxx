#include "monoChromo.h"
#include "orthFitter.h"
#include "cudaConfig.h"
#include "common.h"
#include "cub_wrap.h"
#include "cuPlotter.h"
#include "tvFilter.h"
#include <fstream>
#include <iostream>

cuFunc(multiplyPixelWeight, (complexFormat* img, Real* weights),(img, weights),{
  cudaIdx();
  int shift = max(abs(x+0.5-cuda_row/2), abs(y+0.5-cuda_column/2));
  img[index].x *= weights[shift];
})

int nearestEven(Real x){
  return round(x/2)*2;
}

void monoChromo::calcPixelWeights(){
  int sz = row/2*sizeof(Real);
  Real* loc_pixel_weight = (Real*) ccmemMngr.borrowCache(sz);
  memset(loc_pixel_weight, 0, sz);
  for(int lmd = 0; lmd < nlambda; lmd++){
    Real maxshift =  Real(row*row)/(2*(row+2*lmd*jump));
    Real spectWeight = spectra[lmd];
    for(int shift = 0; shift < maxshift; shift++){
      loc_pixel_weight[shift] += spectWeight;
    }
  }
  for(int shift = 0; shift < row/2; shift++){
    loc_pixel_weight[shift] = pow(loc_pixel_weight[shift], -0.6);
    printf("pixelweights: %f\n", loc_pixel_weight[shift]);
  }
  pixel_weight = (Real*) memMngr.borrowCache(sz);
  cudaMemcpy(pixel_weight, loc_pixel_weight, sz, cudaMemcpyHostToDevice);
}

cuFunc(multiplyReal,(complexFormat* a, complexFormat* b, Real* c),(a,b,c),{
  cudaIdx();
  c[index] = a[index].x*b[index].x;
})

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
    new((cufftHandle*)locplan+i)cufftHandle();
    cufftPlan2d ( (cufftHandle*)locplan+i, rows[i], cols[i], FFTformat);
  }
  calcPixelWeights();
}
Real monoChromo::init(int nrow, int ncol, Real* lambdasi, Real* spectrumi, Real endlambda){
  row = nrow;
  column = ncol;
  Real currentLambda = 1;
  int currentPoint = 0;
  Real stepsize = 2./row*jump;
  nlambda = (endlambda-1)/stepsize+1;
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
  nlambda = i;
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
  calcPixelWeights();
  return tot;
}
void monoChromo::resetSpectra(){
  for(int i = 0; i < nlambda; i++){
    spectra[i] = 1./nlambda;
  }
}
void monoChromo::writeSpectra(const char* filename){
  std::ofstream spectrafile;
  spectrafile.open(filename,ios::out);
  for(int i = 0; i < nlambda; i++){
    spectrafile<<Real(rows[i])/row<<" "<<spectra[i]<<std::endl;
  }
  spectrafile.close();
}
void monoChromo::generateMWL(void* d_input, void* d_patternSum, void* single, Real oversampling){
  Real *d_pattern = (Real*) memMngr.borrowCache(row*column*sizeof(Real));
  complexFormat *d_intensity = (complexFormat*)memMngr.borrowCache(rows[nlambda-1]*cols[nlambda-1]*sizeof(complexFormat));
  complexFormat* d_patternAmp = (complexFormat*)memMngr.borrowCache(row*column*sizeof(Real)*2);
  for(int i = 0; i < nlambda; i++){
    int thisrow = rows[i];
    int thiscol = cols[i];
    init_cuda_image(thisrow, thiscol, 65536, 1);
    cudaF(createWaveFront,(Real*)d_input, 0, (complexFormat*)d_intensity, row/oversampling, column/oversampling);
    myCufftExec( ((cufftHandle*)locplan)[i], d_intensity,d_intensity, CUFFT_FORWARD);
    cudaF(cudaConvertFO,d_intensity);
    init_cuda_image(row, column, 65536, 1);
    cudaF(crop,d_intensity,d_patternAmp,thisrow,thiscol);
    cudaF(applyNorm,d_patternAmp, sqrt(spectra[i])/sqrt(thiscol*thisrow));
    if(i==0) {
      cudaF(getMod2,(Real*)d_patternSum, d_patternAmp);
      if(single!=0) {
        cudaF(extendToComplex,(Real*)d_patternSum, (complexFormat*)single);
        cudaF(applyNorm,(complexFormat*)single, 1./spectra[i]);
      }
    }else{
      cudaF(getMod2,d_pattern, d_patternAmp);
      cudaF(add,(Real*)d_patternSum, (Real*)d_pattern, 1);
    }
  }
  memMngr.returnCache(d_pattern);
  memMngr.returnCache(d_intensity);
  memMngr.returnCache(d_patternAmp);
}
cuFunc(printpix,(Real* input, int x, int y),(input,x,y),{
  printf("%f", input[x*vars->cols+y]);
})
cuFunc(printpixreal,(complexFormat* input, int x, int y),(input,x,y),{
  printf("%f,", input[x*vars->cols+y].x);
})

void* createCache(void* b){
  size_t sz = memMngr.getSize(b);
  void* a = memMngr.borrowCache(sz);
  cudaMemcpy(a, b, sz, cudaMemcpyDeviceToDevice);
  return a;
}
void deleteCache(void* b){
  memMngr.returnCache(b);
}
void add(void* a, void* b, Real c){
  cudaF(add,(complexFormat*)a, (complexFormat*)b, c);
}
void mult(void* a, Real b){
  cudaF(applyNorm,(complexFormat*)a, b);
}
Real innerProd(void* a, void* b){
  Real* tmp = (Real*)memMngr.borrowCache(memMngr.getSize(a)/2);
  cudaF(multiplyReal,tmp, (complexFormat*)a, (complexFormat*)b);
  Real sum = findSum(tmp);
  memMngr.returnCache(tmp);
  return sum;
}

void applyC(Real* input, Real* output){
  cudaMemcpy(output, input, memMngr.getSize(input), cudaMemcpyDeviceToDevice);
  cudaF(forcePositive, output);
}

void monoChromo::solveMWL(void* d_input, void* d_output, bool restart, int nIter, bool updateX, bool updateA)
{
  useOrth = 1;
  bool writeResidual = 1;
  if(nlambda<0) printf("nlambda not initialized: %d\n",nlambda);
  size_t sz = row*column*sizeof(complexFormat);
  complexFormat *fftb = (complexFormat*)memMngr.borrowCache(sz);
  init_fft(row,column);
  init_cuda_image(row, column, 65536, 1);
  Real dt = 2;
  Real friction = .2;//0.1;
  if(restart) {
    cudaMemcpy(d_output, d_input, sz, cudaMemcpyDeviceToDevice);
    cudaF(zeroEdge, (complexFormat*)d_output, 80);
  }
  complexFormat *deltab = (complexFormat*)memMngr.borrowCache(sz);
  complexFormat *momentum = (complexFormat*)memMngr.borrowCache(sz);
  if(friction) cudaMemset(momentum, 0, sz);
  complexFormat *padded = (complexFormat*) memMngr.borrowCache(sizeof(complexFormat)*rows[nlambda-1]*cols[nlambda-1]);
  complexFormat *deltabprev = (complexFormat*)memMngr.borrowCache(sz);
  Real *multiplied = (Real*)memMngr.borrowCache(sz/2);
  Real *momentum_a = 0;
  float step_a = 0;
  ofstream fresidual;
  if(writeResidual) fresidual.open("residual.txt", ios::out);
  if(updateA){
    cudaF(getMod2,multiplied, (complexFormat*)d_input);
    Real mod2ref = findSum(multiplied);
    printf("normalization: %f\n",mod2ref);
    step_a = 1./(mod2ref*nlambda);
    if(step_a<=0 || step_a!=step_a) abort();
    momentum_a = (Real*) ccmemMngr.borrowCache(nlambda*sizeof(Real));
    memset(momentum_a,0,nlambda*sizeof(Real));
  }
  Real stepsize = 0.7;
  complexFormat *fbi;
  if(!updateX) {
    myCufftExec( *plan, (complexFormat*)d_output, fftb, CUFFT_INVERSE);
    cudaF(cudaConvertFO,fftb);
    gs = (void**)ccmemMngr.borrowCache(nlambda*sizeof(void*));
    for(int j = 0; j < nlambda; j++){
      gs[j] = memMngr.borrowCache(sz);
    }
  }
  if(!gs) fbi = (complexFormat*)memMngr.borrowCache(sz);
 // Real tk = 1;
  for(int i = 0; i < nIter; i++){
    bool updateAIter = (updateA&&(i%5==0) || updateX==0) && (i > 0);
    if(updateAIter){
      auto tmp_swap = deltabprev;
      deltabprev = deltab;
      deltab = tmp_swap;
    }
    Real sumspect = 0;
    if(updateX) {
      myCufftExec( *plan, (complexFormat*)d_output, fftb, CUFFT_INVERSE);
      cudaF(cudaConvertFO,fftb);
    }
    if(gs) 
      cudaMemcpy(gs[0], d_output, sz, cudaMemcpyDeviceToDevice);
    cudaMemcpy(deltab, d_input, sz, cudaMemcpyDeviceToDevice);
    cudaF(add,deltab, (complexFormat*)d_output, -spectra[0]);
    if(updateAIter) {
      cudaF(multiplyReal,deltabprev, (complexFormat*)d_output, multiplied);
      Real sum =findSum(multiplied, false);
      if(fabs(sum) > 1e3) {
        sum =findSum(multiplied, false);
        printf("WARING recalculated sum %f\n", sum);
        exit(0);
      }
      Real step = step_a*sum;
      momentum_a[0] += step*dt;
      momentum_a[0]*=(1-friction*dt);
      spectra[0]+=momentum_a[0]*dt;
      if(spectra[0]<=0) spectra[0] = 1e-6;
      if(spectra[0]>0.03) spectra[0] = 0.03;
      sumspect+=spectra[0];
    }
    for(int j = 1; j < nlambda; j++){
      if(spectra[j]<=0) continue;
      size_t N = rows[j]*cols[j];
      if(gs) fbi = (complexFormat*)gs[j];
      if(!gs || updateX || i==0){
        init_cuda_image(rows[j], cols[j], 65536, 1);
        cudaF(pad,fftb, padded, row, column);
        cudaF(cudaConvertFO,padded);
        cudaF(applyNorm,padded, 1./N);
        myCufftExec( ((cufftHandle*)locplan)[j], padded, padded, CUFFT_FORWARD);
        init_cuda_image(row, column, 65536, 1);
        cudaF(crop,padded, fbi, rows[j], cols[j]);
      }
      cudaF(add,deltab, fbi, -spectra[j]);
      if(updateAIter) {
        cudaF(multiplyReal,deltabprev, fbi, multiplied);
        Real sum = findSum(multiplied);
        if(fabs(sum) > 1e3) {
          sum = findSum(multiplied);
          printf("WARING recalculated sum %f\n", sum);
          exit(0);
        }
        Real step = step_a*Real(rows[j])/row*sum;
        momentum_a[j] += step*dt;
        momentum_a[j]*=(1-friction*dt);
        spectra[j]+=momentum_a[j]*dt;
        if(spectra[j]<=0) spectra[j] = 1e-6;
        if(spectra[j]>0.3) spectra[j] = 0.3;
        sumspect+=spectra[j];
      }
    }
    if(useOrth&&!updateX){
      Fit(spectra, nlambda, (void**)gs, d_input, innerProd, mult, add, createCache, deleteCache, 1);
      init_cuda_image(row, column, 65536, 1);
      break;
    }
    if(updateAIter){
      for(int j = 0; j < nlambda; j++){
        spectra[j]-=(sumspect-1)/nlambda;
        if(spectra[j]<=0) spectra[j] = 1e-6;
        if(spectra[j]>0.3) spectra[j] = 0.3;
      }
    }
    if(updateX){
      cudaMemset(deltabprev, 0, sz);
      cudaF(add, deltabprev, deltab, stepsize*spectra[0]);
      if(i==nIter-1) {
        plt.plotComplex(deltab, MOD, 0, 1, "residual", 1);
      }
      for(int j = 1; j < nlambda; j++){
        if(spectra[j]<=0) continue;
        init_cuda_image(rows[j], cols[j], 65536, 1);
        cudaF(pad,(complexFormat*)deltab, padded, row, column);
        myCufftExec( ((cufftHandle*)locplan)[j], padded, padded, CUFFT_INVERSE);
        cudaF(cudaConvertFO,padded);
        init_cuda_image(row, column, 65536, 1);
        cudaF(crop,padded, fbi, rows[j], cols[j]);
        cudaF(cudaConvertFO,fbi);
        cudaF(applyNorm,fbi, 1./(row*column));
        myCufftExec( *plan, fbi, fbi, CUFFT_FORWARD);
        cudaF(add,(complexFormat*)deltabprev, fbi, stepsize*spectra[j]);
      }
      //cudaF(multiplyPixelWeight, deltabprev, pixel_weight);
      if(friction){
        cudaF(add, (complexFormat*)momentum, deltabprev, 1);
        cudaF(applyNorm,(complexFormat*)momentum, 1-friction*dt);
        cudaF(add,(complexFormat*)d_output, momentum, dt);
      }else{
        cudaF(add, (complexFormat*)d_output, deltabprev, 1);
      }
      cudaF(forcePositive,(complexFormat*)d_output);
      /* //FISTA update, not quite effective
      cudaF(add, deltabprev, (complexFormat*)d_output, 1);
      cudaF(getReal, (Real*)fbi,deltabprev);
      FISTA((Real*)fbi, (Real*)deltabprev, 1e-6, 70, &applyC);
      cudaF(extendToComplex, (Real*)deltabprev, fbi);
      Real tmp = 0.5+sqrt(0.25+tk*tk);
      Real fact1 = (tk-1)/tmp;
      tk = tmp;
      cudaF(applyNorm, (complexFormat*)d_output, -fact1);
      cudaF(add, (complexFormat*)d_output, fbi, 1+fact1);
      */
    }
    if(writeResidual) {
      cudaF(getMod2,multiplied, deltab);
      fresidual<<i<<" "<<findSum(multiplied)<<endl;
    }
  }
  if(writeResidual) {
    fresidual.close();
  }
  if(updateA){
    ccmemMngr.returnCache(momentum_a);
  }
  memMngr.returnCache(deltabprev);
  memMngr.returnCache(multiplied);
  memMngr.returnCache(momentum);
  memMngr.returnCache(padded);
  if(!gs) memMngr.returnCache(fbi);
  memMngr.returnCache(fftb);
  memMngr.returnCache(deltab);

}
