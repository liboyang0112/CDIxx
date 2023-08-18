#include "monoChromo.h"
#include "orthFitter.h"
#include "cudaConfig.h"
#include "common.h"
#include "cub_wrap.h"
#include "cuPlotter.h"
#include "tvFilter.h"
#include <fstream>
#include <iostream>
#include <gsl/gsl_spline.h>


cuFunc(updateMomentum,(complexFormat* force, complexFormat* mom, Real dx),(force, mom , dx),{
  cudaIdx()
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

cuFunc(overExposureZeroGrad, (complexFormat* deltab, complexFormat* b, int noiseLevel),(deltab, b, noiseLevel),{
  cudaIdx();
  if(b[index].x >= vars->scale*0.99 && deltab[index].x < 0) deltab[index].x = 0;
  //if(fabs(deltab[index].x)*vars->rcolor < sqrtf(noiseLevel)) deltab[index].x = 0;
  deltab[index].y = 0;
})

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
  }
  pixel_weight = (Real*) memMngr.borrowCache(sz);
  cudaMemcpy(pixel_weight, loc_pixel_weight, sz, cudaMemcpyHostToDevice);
}

cuFunc(multiplyReal_inner,(complexFormat* a, complexFormat* b, Real* c, int d),(a,b,c,d),{
  cudaIdx();
  int removeCent = 50;
  if(x < d || x >= cuda_row - d || y < d || y > cuda_column - d 
    ||(abs(x-cuda_row/2) < removeCent && abs(y-cuda_column/2) < removeCent)
      )
    c[index] = 0;
  else c[index] = a[index].x*b[index].x;
})

void monoChromo::init(int nrow, int ncol, int nlambda_, double* lambdas_, double* spectra_){
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
void monoChromo::init(int nrow, int ncol, double minlambda, double maxlambda){
  row = nrow;
  column = ncol;
  Real stepsize = 2./row*jump;
  minlambda = 1-ceil((1-minlambda)/stepsize)*stepsize;
  maxlambda = 1+ceil((maxlambda-1)/stepsize)*stepsize;
  nlambda = (maxlambda-minlambda)/stepsize+1;
  rows = (int*)ccmemMngr.borrowCache(sizeof(int)*nlambda);
  cols = (int*)ccmemMngr.borrowCache(sizeof(int)*nlambda);
  spectra = (double*) ccmemMngr.borrowCache(nlambda*sizeof(double));
  memset(spectra, 0, nlambda*sizeof(double));
  locplan = ccmemMngr.borrowCache(sizeof(cufftHandle)*nlambda);
  for(int i = 0; i < nlambda; i++){
    rows[i] = row+(i+(minlambda-1)/stepsize)*2*jump;
    cols[i] = column+(i+(minlambda-1)/stepsize)*2*jump;
    printf("%d: (%d,%d)\n",i, rows[i],cols[i]);
    new(&(((cufftHandle*)locplan)[i]))cufftHandle();
    cufftPlan2d ( &(((cufftHandle*)locplan)[i]), rows[i], cols[i], FFTformat);
  }
}
Real monoChromo::init(int nrow, int ncol, double* lambdasi, double* spectrumi, int narray){
  row = nrow;
  column = ncol;
  Real stepsize = 2./row*jump;
  Real skiplambda = 2./row*skip;
  nlambda = (lambdasi[narray-1]-skiplambda-1)/stepsize+1;
  spectra = (double*) ccmemMngr.borrowCache(nlambda*sizeof(double));
  ofstream spectrumfile;
  int sz = narray*sizeof(double);
  double* spectrumraw = (double*)ccmemMngr.borrowCache(sz);
  memcpy(spectrumraw, spectrumi, sz);
  lambdasi[0] /= (1+skiplambda);
  for(int i = 1; i < narray; i++){
    lambdasi[i] /= (1+skiplambda);
    spectrumi[i] += spectrumi[i-1];
  }
  spectrumfile.open("spectra_raw.txt", std::ios::out);
  for(int i = 0; i < narray; i++){
    spectrumfile<<lambdasi[i]*(1+skiplambda)<<" "<<spectrumraw[i]/spectrumi[narray-1]<<std::endl;
  }
  ccmemMngr.returnCache(spectrumraw);
  spectrumfile.close();
  gsl_interp_accel *acc = gsl_interp_accel_alloc ();
  gsl_spline *spline = gsl_spline_alloc (gsl_interp_cspline, narray);
  gsl_spline_init (spline, lambdasi, spectrumi, narray);
  Real cumprev = 0;
  Real lambda = 1;
  for(int i = 0; i < nlambda-1; i++){
    double cumnow = gsl_spline_eval (spline, lambda+stepsize/2, acc);
    spectra[i] = cumnow-cumprev;
    cumprev = cumnow;
    lambda+=stepsize;
  }
  spectra[nlambda-1] = spectrumi[narray-1]-cumprev;
  gsl_spline_free (spline);
  gsl_interp_accel_free (acc);

  rows = (int*)ccmemMngr.borrowCache(sizeof(int)*nlambda);
  cols = (int*)ccmemMngr.borrowCache(sizeof(int)*nlambda);
  plt.init(row,column);
  locplan = ccmemMngr.borrowCache(sizeof(cufftHandle)*nlambda);
  for(int i = 0; i < nlambda; i++){
    rows[i] = row+i*2*jump;
    cols[i] = column+i*2*jump;
    printf("%d: (%d,%d)=%f\n",i, rows[i],cols[i],spectra[i]/=spectrumi[narray-1]);
    new(&(((cufftHandle*)locplan)[i]))cufftHandle();
    cufftPlan2d ( &(((cufftHandle*)locplan)[i]), rows[i], cols[i], FFTformat);
  }
  spectrumfile.open("spectra.txt", std::ios::out);
  for(int i = 0; i < nlambda; i++){
    spectrumfile<<1+stepsize*i+skiplambda<<" "<<spectra[i]/narray*nlambda<<std::endl;
  }
  spectrumfile.close();
  calcPixelWeights();
  return spectrumi[narray-1];
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
    spectrafile<<Real(rows[i])/rows[0]<<" "<<spectra[i]<<std::endl;
  }
  spectrafile.close();
}
void monoChromo::generateMWL(void* d_input, void* d_patternSum, void* single, Real oversampling){
  Real *d_pattern = (Real*) memMngr.borrowCache(row*column*sizeof(Real));
  complexFormat *d_intensity = (complexFormat*)memMngr.borrowCache(rows[nlambda-1]*cols[nlambda-1]*sizeof(complexFormat));
  complexFormat* d_patternAmp = (complexFormat*)memMngr.borrowCache(row*column*sizeof(Real)*2);
  complexFormat *d_inputWave = (complexFormat*)memMngr.borrowCache(int(row/oversampling)*int(column/oversampling)*sizeof(complexFormat));
  cudaMemset(d_inputWave, 0, int(row/oversampling)*int(column/oversampling)* sizeof(complexFormat));
  resize_cuda_image(int(row/oversampling), int(column/oversampling));
  assignReal( (Real*)d_input, d_inputWave);
  if(devstates) applyRandomPhase(d_inputWave,0,(curandStateMRG32k3a *)devstates);
  for(int i = 0; i < nlambda; i++){
    int thisrow = rows[i];
    int thiscol = cols[i];
    resize_cuda_image(thisrow, thiscol);
    pad( d_inputWave, d_intensity, int(row/oversampling), int(column/oversampling));
    myCufftExec( ((cufftHandle*)locplan)[i], d_intensity,d_intensity, CUFFT_FORWARD);
    cudaConvertFO(d_intensity);
    resize_cuda_image(row, column);
    crop(d_intensity,d_patternAmp,thisrow,thiscol);
    applyNorm(d_patternAmp, sqrt(spectra[i]/(thiscol*thisrow)));
    if(i==0) {
      getMod2((Real*)d_patternSum, d_patternAmp);
      if(single!=0) {
        extendToComplex((Real*)d_patternSum, (complexFormat*)single);
        applyNorm((complexFormat*)single, 1./spectra[i]);
      }
    }else{
      getMod2(d_pattern, d_patternAmp);
      add((Real*)d_patternSum, (Real*)d_pattern, 1);
      if(single!=0 && i == 1 ) {
        extendToComplex((Real*)d_pattern, (complexFormat*)single);
        applyNorm((complexFormat*)single, 1./spectra[i]);
      }
    }
  }
  memMngr.returnCache(d_pattern);
  memMngr.returnCache(d_intensity);
  memMngr.returnCache(d_patternAmp);
}
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
  add((complexFormat*)a, (complexFormat*)b, c);
}
void mult(void* a, Real b){
  applyNorm((complexFormat*)a, b);
}
Real innerProd(void* a, void* b, void* param){
  Real* tmp = (Real*)memMngr.borrowCache(memMngr.getSize(a)/2);
  multiplyReal(tmp,(complexFormat*)a, (complexFormat*)b);
  applyMask(tmp, (Real*)param);
  Real sum = findSum(tmp);
  memMngr.returnCache(tmp);
  return sum;
}

void applyC(Real* input, Real* output){
  cudaMemcpy(output, input, memMngr.getSize(input), cudaMemcpyDeviceToDevice);
  //forcePositive( output);
}

void monoChromo::solveMWL(void* d_input, void* d_output, int noiseLevel, bool restart, int nIter, bool updateX, bool updateA)
{
  useOrth = 1;
  bool writeResidual = 1;
  int monoidx = 0;
  for(int i = 0; i < nlambda; i++){
    if(row == rows[i]) {
      monoidx = i;
      break;
    }
  }
  int d = row/2-5;
  rect spt;
  spt.starty = spt.startx = d;
  spt.endx = spt.endy = row-d-1;
  rect *cuda_spt;
  cuda_spt = (rect*)memMngr.borrowCache(sizeof(rect));
  cudaMemcpy(cuda_spt, &spt, sizeof(spt), cudaMemcpyHostToDevice);
  Real *sptimg = (Real*)memMngr.borrowCache(row*column*sizeof(Real));
  createMaskBar(sptimg, cuda_spt, 0);
  memMngr.returnCache(cuda_spt);
  applyMaskBar(sptimg, (complexFormat*)d_input, 0.99);
  plt.plotFloat(sptimg, MOD, 0, 1, "innerprodspt", 0);
  if(nlambda<0) printf("nlambda not initialized: %d\n",nlambda);
  size_t sz = row*column*sizeof(complexFormat);
  complexFormat *fftb = (complexFormat*)memMngr.borrowCache(sz);
  init_fft(row,column);
  resize_cuda_image(row, column);
  Real lr = 1.;
  Real beta1 = 0.5;//0.1;
  Real beta2 = 0.;//5;//0.99;
  Real adamepsilon = 1e-4;
  if(restart) {
    //cudaMemcpy(d_output, d_input, sz, cudaMemcpyDeviceToDevice);
    //zeroEdge( (complexFormat*)d_output, 150);
    cudaMemset(d_output, 0, sz);
  }
  complexFormat *deltab = (complexFormat*)memMngr.borrowCache(sz);
  complexFormat *momentum = 0;
  Real *adamv = 0;
  if(beta1) {
    momentum = (complexFormat*)memMngr.borrowCache(sz);
    cudaMemset(momentum, 0, sz);
  }
  if(beta2) {
    adamv = (Real*)memMngr.borrowCache(sz/2);
    cudaMemset(adamv, 0, sz/2);
  }
  complexFormat *padded = (complexFormat*) memMngr.borrowCache(sizeof(complexFormat)*rows[nlambda-1]*cols[nlambda-1]);
  complexFormat *deltabprev = (complexFormat*)memMngr.borrowCache(sz);
  Real *multiplied = (Real*)memMngr.borrowCache(sz/2);
  Real *momentum_a = 0;
  float step_a = 0;
  ofstream fresidual;
  if(writeResidual) fresidual.open("residual.txt", ios::out);
  if(updateA){
    getMod2(multiplied, (complexFormat*)d_input);
    Real mod2ref = findSum(multiplied);
    printf("normalization: %f\n",mod2ref);
    step_a = 1./(mod2ref*nlambda);
    if(step_a<=0 || step_a!=step_a) abort();
    momentum_a = (Real*) ccmemMngr.borrowCache(nlambda*sizeof(Real));
    memset(momentum_a,0,nlambda*sizeof(Real));
  }
  complexFormat *fbi;
  if(!updateX) {
    myCufftExec( *plan, (complexFormat*)d_output, fftb, CUFFT_INVERSE);
    cudaConvertFO(fftb);
  }
  if(updateA) {
    gs = (void**)ccmemMngr.borrowCache(nlambda*sizeof(void*));
    for(int j = 0; j < nlambda; j++){
      gs[j] = memMngr.borrowCache(sz);
    }
  }
  if(!gs) fbi = (complexFormat*)memMngr.borrowCache(sz);
 // Real tk = 1;
  bool calcDeltab = 0;
  for(int i = 0; i < nIter; i++){
    bool updateAIter = !useOrth && (updateA&&(i%5==0) || updateX==0) && (i > 0);
    if(updateAIter){
      auto tmp_swap = deltabprev;
      deltabprev = deltab;
      deltab = tmp_swap;
    }
    Real sumspect = 0;
    if(updateX||i==0||!gs) {
      myCufftExec( *plan, (complexFormat*)d_output, fftb, CUFFT_INVERSE);
    }
    if(gs) 
      cudaMemcpy(gs[monoidx], d_output, sz, cudaMemcpyDeviceToDevice);
    if(updateX || updateAIter) cudaMemcpy(deltab, d_input, sz, cudaMemcpyDeviceToDevice);
    if(updateX || updateAIter) add(deltab, (complexFormat*)d_output, -spectra[monoidx]);
    if(updateAIter) {
      multiplyReal(multiplied, deltabprev, (complexFormat*)d_output);
      Real sum =findSum(multiplied, false);
      if(fabs(sum) > 1e3) {
        sum =findSum(multiplied, false);
        printf("WARING recalculated sum %f\n", sum);
        exit(0);
      }
      momentum_a[monoidx] = momentum_a[monoidx]*beta1 + (1-beta1)*sum*step_a;
      spectra[monoidx]+=momentum_a[monoidx]*lr;
      if(spectra[monoidx]<=0) spectra[monoidx] = 1e-6;
      if(spectra[monoidx]>0.03) spectra[monoidx] = 0.03;
      sumspect+=spectra[monoidx];
    }
    for(int j = 0; j < nlambda; j++){
      if(j == monoidx) continue;
      if(!updateA && spectra[j]<=0) continue;
      size_t N = rows[j]*cols[j];
      if(gs) fbi = (complexFormat*)gs[j];
      if(!gs || updateX || i==0){
        resize_cuda_image(rows[j], cols[j]);
        if(rows[j] > row) padinner(fftb, padded, row, column, 1./N);
        else cropinner(fftb, padded, row, column, 1./N);
        myCufftExec( ((cufftHandle*)locplan)[j], padded, padded, CUFFT_FORWARD);
        resize_cuda_image(row, column);
        if(rows[j] > row) crop(padded, fbi, rows[j], cols[j]);
        else pad(padded, fbi, rows[j], cols[j]);
        //plt.plotComplex(fbi, MOD, 0, 1, ("debug"+to_string(j)).c_str(), 1, 0, 1);
      }
      if((updateX || updateAIter)) add(deltab, fbi, -spectra[j]);
      if(updateAIter) {
        multiplyReal(multiplied, deltabprev, fbi);
        Real sum = findSum(multiplied);
        if(fabs(sum) > 1e3) {
          sum = findSum(multiplied);
          printf("WARING recalculated sum %f\n", sum);
          exit(0);
        }
        Real step = step_a*Real(rows[j])/row*sum;
        momentum_a[j] = momentum_a[j]*beta1 + (1-beta1)*step;
        spectra[j]+=momentum_a[j]*lr;
        if(spectra[j]<=0) spectra[j] = 1e-6;
        if(spectra[j]>0.3) spectra[j] = 0.3;
        sumspect+=spectra[j];
      }
    }
    if(writeResidual) {
      getMod2(multiplied, deltab);
      fresidual<<i<<" "<<findSum(multiplied)<<endl;
    }
    if(calcDeltab) break;
    if(useOrth&&updateA){
      Fit(spectra, nlambda, (void**)gs, d_input, innerProd, mult, add, createCache, deleteCache, 1, sptimg);
      //spectra[nlambda-2] = spectra[nlambda-1] = 0;
      //for(int il = 0; il < nlambda-2; il++){
      //  spectra[il] *= 1.1;
      //}
      printf("Fit spectrum done. %f\n", spectra[0]);
      if(!updateX) {
        i = nIter-2;
        updateA = 0;
        updateX = 1;
        calcDeltab = 1;
        continue;
      }
    }
    if(updateAIter){
      for(int j = 0; j < nlambda; j++){
        spectra[j]-=(sumspect-1)/nlambda;
        if(spectra[j]<=0) spectra[j] = 1e-6;
        if(spectra[j]>0.3) spectra[j] = 0.3;
      }
    }
    if(updateX){
      //overExposureZeroGrad( deltab, (complexFormat*)d_input, noiseLevel);
      if(i > 20 && noiseLevel){
        getReal( (Real*)fbi,deltab);
        FISTA((Real*)fbi, (Real*)fbi, 1e-5*sqrt(noiseLevel), 20, &applyC);
        extendToComplex( (Real*)fbi, deltab);
      }
      cudaMemset(deltabprev, 0, sz);
      add( deltabprev, deltab, spectra[monoidx]);
      for(int j = 1; j < nlambda; j++){
        if(j == monoidx) continue;
        if(spectra[j]<=0) continue;
        resize_cuda_image(rows[j], cols[j]);
        if(rows[j] > row) pad((complexFormat*)deltab, padded, row, column);
        else crop((complexFormat*)deltab, padded, row, column);
        myCufftExec( ((cufftHandle*)locplan)[j], padded, padded, CUFFT_INVERSE);
        resize_cuda_image(row, column);
        if(rows[j] > row) cropinner(padded, fbi, rows[j], cols[j], 1./(row*column));
        else padinner(padded, fbi, rows[j], cols[j], 1./(row*column));
        myCufftExec( *plan, fbi, fbi, CUFFT_FORWARD);
        add((complexFormat*)deltabprev, fbi, spectra[j]);
      }
      //multiplyPixelWeight( deltabprev, pixel_weight);
      if(beta1){
        updateMomentum( deltabprev, momentum, beta1);
        if(beta2) {
          adamUpdateV( adamv, deltabprev, beta2);
          adamUpdate( (complexFormat*)d_output, momentum, adamv, lr, adamepsilon);
        }else add((complexFormat*)d_output, momentum, lr);
      }else{
        add( (complexFormat*)d_output, deltabprev, lr);
      }
      forcePositive((complexFormat*)d_output);
      /* //FISTA update, not quite effective
      add( deltabprev, (complexFormat*)d_output, 1);
      getReal( (Real*)fbi,deltabprev);
      FISTA((Real*)fbi, (Real*)deltabprev, 1e-6, 70, &applyC);
      extendToComplex( (Real*)deltabprev, fbi);
      Real tmp = 0.5+sqrt(0.25+tk*tk);
      Real fact1 = (tk-1)/tmp;
      tk = tmp;
      applyNorm( (complexFormat*)d_output, -fact1);
      add( (complexFormat*)d_output, fbi, 1+fact1);
      */
    }
  }
  if(writeResidual) {
    plt.plotComplex(deltab, REAL, 0, 1, "residual_pulseGen", 1, 0, 1);
    add(deltab,(complexFormat*)d_input, -1);
    plt.plotComplex(deltab, MOD, 0, 1, "broad_recon", 0, 0, 0);
    fresidual.close();
  }
  if(updateA){
    ccmemMngr.returnCache(momentum_a);
  }
  memMngr.returnCache(deltabprev);
  memMngr.returnCache(multiplied);
  if(momentum) memMngr.returnCache(momentum);
  if(adamv) memMngr.returnCache(adamv);
  memMngr.returnCache(padded);
  if(!gs) memMngr.returnCache(fbi);
  memMngr.returnCache(fftb);
  memMngr.returnCache(deltab);

}
