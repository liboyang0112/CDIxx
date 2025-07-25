#include "broadBand.hpp"
#include "cudaConfig.hpp"
#include "cuPlotter.hpp"
#include "fmt/core.h"
#include "fmt/os.h"
#include <gsl/gsl_spline.h>
#include <math.h>

using namespace std;


int inline nearestEven(Real x){
  return round(x/2)*2;
}

void broadBand_base::plotAutoCorr(const char* filename, Real* pattern, Real weight){
  complexFormat* cache = padding_cache;
  extendToComplex(pattern, cache);
  cudaConvertFO(cache);
  init_fft(row, column);
  myIFFT(cache, cache);
  resize_cuda_image(row, column);
  plt.plotComplexColor(cache, 1, weight, filename, 0, 0);
}

void broadBand::init(int nrow, int ncol, int nlambda_, double* lambdas_, double* spectra_){
  lambdas = lambdas_;
  nlambda = nlambda_;
  spectra = spectra_;
  row = nrow;
  column = ncol;
  rows = (int*)ccmemMngr.borrowCache(sizeof(int)*nlambda);
  cols = (int*)ccmemMngr.borrowCache(sizeof(int)*nlambda);
  plt.init(row,column);
  locplan = (int*)ccmemMngr.borrowCache(sizeof(int)*nlambda);
  fmt::ostream file = fmt::output_file("spectrum_raw.txt");
  for(int i = 0; i < nlambda; i++){
    rows[i] = nearestEven(row*lambdas_[i]);
    cols[i] = nearestEven(column*lambdas_[i]);
    createPlan(locplan+i, rows[i], cols[i]);
    //file << lambdas_[i] << " " << 0 << endl;
    file.print("{} {}\n", lambdas_[i] , spectra_[i]*nlambda);
    //file << lambdas_[i] << " " << 0 << endl;
  }
  file.close();
  padding_cache = (complexFormat*) memMngr.borrowCache(sizeof(complexFormat)*rows[nlambda-1]*cols[nlambda-1]);
}
void broadBand::init(int nrow, int ncol, double minlambda, double maxlambda){
  row = nrow;
  column = ncol;
  Real stepsize = 2./row*jump;
  minlambda = 1-ceil((1-minlambda)/stepsize)*stepsize;
  maxlambda = 1+ceil((maxlambda-1)/stepsize)*stepsize;
  nlambda = (maxlambda-minlambda)/stepsize+1;
  rows = (int*)ccmemMngr.borrowCache(sizeof(int)*nlambda);
  cols = (int*)ccmemMngr.borrowCache(sizeof(int)*nlambda);
  lambdas = (double*)ccmemMngr.borrowCache(sizeof(double)*nlambda);
  spectra = (double*) ccmemMngr.borrowCleanCache(nlambda*sizeof(double));
  locplan = (int*)ccmemMngr.borrowCache(sizeof(int)*nlambda);
  for(int i = 0; i < nlambda; i++){
    rows[i] = row+(i+(minlambda-1)/stepsize)*2*jump;
    cols[i] = column+(i+(minlambda-1)/stepsize)*2*jump;
    lambdas[i] = double(rows[i])/row;
    fmt::println("{}: ({},{})",i, rows[i],cols[i]);
    createPlan(locplan+i, rows[i], cols[i]);
  }
  padding_cache = (complexFormat*) memMngr.borrowCache(sizeof(complexFormat)*rows[nlambda-1]*cols[nlambda-1]);
}
void broadBand_constRatio::init(int nrow, int ncol, double minlambda, double maxlambda){
  row = nrow;
  column = ncol;
  thisrow = row+2*jump;
  thiscol = column+2*jump;
  thisrowp = row-2*jump;
  thiscolp = column-2*jump;
  Real factor = Real(thisrowp)/row;
  nmiddle = 0;
  if(maxlambda < 1) {
    fmt::println("Max lambda is shorter than 1, please check your middle lambda and lambda range");
    abort();
  }
  myMalloc(int, locplan, 2);
  if(minlambda > 1) {
    fmt::println(stderr, "ERROR: minimum lambda > 1 detected, please reset it to 1");
  }else if(minlambda < factor){
    nmiddle = log(minlambda)/log(factor);
    minlambda = pow(factor, nmiddle);
    createPlan(locplan+1, thisrowp, thiscolp);
  }
  Real factor1 = Real(thisrow)/row;
  nlambda = log(maxlambda)/log(factor1)+nmiddle;
  myMalloc(double, lambdas, nlambda);
  lambdas[nmiddle] = 1;
  for(int i = nmiddle-1; i >= 0; i--){
    lambdas[i] = lambdas[i+1]*factor;
  }
  for(int i = nmiddle+1; i < nlambda; i++){
    lambdas[i] = lambdas[i-1]*factor1;
  }
  spectra = (double*) ccmemMngr.borrowCleanCache(nlambda*sizeof(double));
  createPlan(locplan, thisrow, thiscol);
  myCuMalloc(complexFormat, cache, thisrow*thiscol);
}

void broadBand_base::applyAT(complexFormat* image, complexFormat* output, int trow, int tcol, int plan, char freq){
  complexFormat* cache = padding_cache;
  if(trow == row && tcol == column) {
    myMemcpyD2D(output, image, row*column*sizeof(complexFormat));
    return;
  }
  resize_cuda_image(trow, tcol);
  if(trow > row) {
    if((freq & 1) == 0)
      pad(image, cache, row, column);
    else
      padinner(image, cache, row, column);
  }
  else {
    if((freq & 1) == 0)
      crop(image, cache, row, column);
    else
      cropinner(image, cache, row, column);
  }
  myIFFTM(plan, cache, cache);
  resize_cuda_image(row, column);
  if(trow > row) {
    if((freq & 2) == 0)
      crop(cache, output, trow, tcol);
    else
      cropinner(cache, output, trow, tcol);
  }
  else {
    if((freq & 2) == 0)
      padinner(cache, output, trow, tcol);
    else
      pad(cache, output, trow, tcol);
  }
  applyNorm(output, 1./(row*column));
  myFFT(output, output);
}

Real broadBand::init(int nrow, int ncol, double* lambdasi, double* spectrumi, int narray){
  row = nrow;
  column = ncol;
  Real stepsize = 2./row*jump;
  Real skiplambda = 2./row*skip;
  nlambda = (lambdasi[narray-1]-skiplambda-1)/stepsize+1;
  spectra = (double*) ccmemMngr.borrowCache(nlambda*sizeof(double));
  fmt::ostream spectrumfile = fmt::output_file("spectra_raw.txt");
  for(int i = 0; i < narray; i++){
    spectrumfile.print("{} {}\n",lambdasi[i],spectrumi[i]/spectrumi[narray-1]);
  }
  lambdasi[0] /= (1+skiplambda);
  stepsize /= (1+skiplambda);
  for(int i = 1; i < narray; i++){
    lambdasi[i] /= (1+skiplambda);
    spectrumi[i] += spectrumi[i-1];
  }
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
  myMalloc(double, lambdas, nlambda);
  plt.init(row,column);
  locplan = (int*)ccmemMngr.borrowCache(sizeof(int)*nlambda);
  for(int i = 0; i < nlambda; i++){
    rows[i] = row+i*2*jump;
    cols[i] = column+i*2*jump;
    lambdas[i] = double(rows[i])/row;
    fmt::println("{}: ({},{})={:f}",i, rows[i],cols[i],spectra[i]/=spectrumi[narray-1]);
    createPlan( locplan+i, rows[i], cols[i]);
  }
  fmt::ostream spectrafile = fmt::output_file("spectra.txt");
  for(int i = 0; i < nlambda; i++){
    spectrafile.print("{} {}\n",1+stepsize*i+skiplambda,spectra[i]/narray*nlambda);
  }
  spectrafile.close();
  return spectrumi[narray-1];
}
Real broadBand_constRatio::init(int nrow, int ncol, double* lambdasi, double* spectrumi, int narray){
  nmiddle = 0;
  row = nrow;
  column = ncol;
  thisrow = row+2*jump;
  thiscol = column+2*jump;
  Real skiplambda = 2./row*skip;
  Real factor = Real(thisrow)/row;
  nlambda = log(lambdasi[narray-1]/(skiplambda+1))/log(factor)+1;
  myMalloc(double, spectra, nlambda);
  myMalloc(double, lambdas, nlambda);
  fmt::ostream spectrumfile = fmt::output_file("spectra_raw.txt");
  for(int i = 0; i < narray; i++){
    spectrumfile.print("{} {}\n",lambdasi[i],spectrumi[i]/spectrumi[narray-1]);
  }
  spectrumfile.close();
  lambdasi[0] /= (1+skiplambda);
  for(int i = 1; i < narray; i++){
    lambdasi[i] /= (1+skiplambda);
    spectrumi[i] += spectrumi[i-1];
  }
  gsl_interp_accel *acc = gsl_interp_accel_alloc ();
  gsl_spline *spline = gsl_spline_alloc (gsl_interp_cspline, narray);
  gsl_spline_init (spline, lambdasi, spectrumi, narray);
  Real cumprev = 0;
  lambdas[0] = 1;
  for(int i = 0; i < nlambda-1; i++){
    lambdas[i+1] = lambdas[i]*factor;
    double cumnow = gsl_spline_eval (spline, lambdas[i]*(1+factor)/2, acc);
    spectra[i] = cumnow-cumprev;
    cumprev = cumnow;
  }
  spectra[nlambda-1] = spectrumi[narray-1]-cumprev;
  gsl_spline_free (spline);
  gsl_interp_accel_free (acc);

  myMalloc(int, locplan, 1);
  createPlan(locplan, thisrow, thiscol);
  fmt::ostream spectrafile = fmt::output_file("spectra.txt");
  for(int i = 0; i < nlambda; i++){
    spectra[i]/=spectrumi[narray-1];
    spectrafile.print("{} {}\n", lambdas[i]+skiplambda,spectra[i]/narray*nlambda/lambdas[i]);
  }
  spectrafile.close();
  myCuMalloc(complexFormat, cache, thisrow*thiscol);
  return spectrumi[narray-1];
}
void broadBand_base::resetSpectra(){
  for(int i = 0; i < nlambda; i++){
    spectra[i] = 1./nlambda;
  }
}
void broadBand_base::writeSpectra(const char* filename, Real factor){
  fmt::ostream spectrafile = fmt::output_file(filename);
  for(int i = 0; i < nlambda; i++){
    spectrafile.print("{} {}\n", lambdas[i]*factor,spectra[i]);
  }
  spectrafile.close();
}
void broadBand_constRatio::writeSpectra(const char* filename, Real factor){
  fmt::ostream spectrafile = fmt::output_file(filename);
  for(int i = 0; i < nlambda; i++){
    spectrafile.print("{} {}\n", lambdas[i]*factor,spectra[i]/lambdas[i]);
  }
  spectrafile.close();
}
void broadBand::generateMWL(void* d_input, void* d_patternSum, void* single){
  myCuDMalloc(Real, d_pattern, row*column);
  myCuDMalloc(complexFormat, d_intensity, rows[nlambda-1]*cols[nlambda-1]);
  myCuDMalloc(complexFormat, d_patternAmp, row*column);
  myCuDMalloc(complexFormat, d_inputWave, row*column);
  resize_cuda_image(row, column);
  extendToComplex( (Real*)d_input, d_inputWave);
  //applyRandomPhase(d_inputWave, 0, devstates);
  for(int i = 0; i < nlambda; i++){
    int thisrow = rows[i];
    int thiscol = cols[i];
    resize_cuda_image(thisrow, thiscol);
    pad( d_inputWave, d_intensity, row, column);
    if(0 && thisrow == rows[0]*2) {
      plt.init(thisrow, thiscol);
      plt.plotComplex(d_intensity, MOD, 0, 1, "padded_demo", 0, 0, 0);
    }
    myFFTM( locplan[i], d_intensity,d_intensity);
    cudaConvertFO(d_intensity);
    if(0&&thisrow == rows[0]*2) {
      plt.plotComplex(d_intensity, MOD2, 0, 1./(thisrow*thiscol), "padded_fft_demo", 1, 0, 1);
    }
    resize_cuda_image(row, column);
    crop(d_intensity,d_patternAmp,thisrow,thiscol);
    applyNorm(d_patternAmp, sqrt(spectra[i]/(thiscol*thisrow)));
    if(0&&thisrow == rows[0]*2) {
      plt.init(row, column);
      plt.plotComplex(d_patternAmp, MOD2, 0, 1./spectra[i], "pattern1_demo", 1, 0, 1);
    }
    if(i==0) {
      getMod2((Real*)d_patternSum, d_patternAmp);
      if(single!=0) {
        extendToComplex((Real*)d_patternSum, (complexFormat*)single);
        applyNorm((complexFormat*)single, 1./spectra[i]);
      }
    }else{
      getMod2(d_pattern, d_patternAmp);
      add((Real*)d_patternSum, (Real*)d_pattern, 1);
      if(single!=0 && i == 0 ) {
        extendToComplex((Real*)d_pattern, (complexFormat*)single);
        applyNorm((complexFormat*)single, 1./spectra[i]);
      }
    }
  }
  myCuFree(d_pattern);
  myCuFree(d_intensity);
  myCuFree(d_patternAmp);
}
void broadBand_constRatio::applyAT(complexFormat* src, complexFormat* dest, char zoom){
  if(zoom == 0) {
    resize_cuda_image(thisrow, thiscol);
    pad(src, cache, row, column);
    myFFTM(*locplan,cache,cache);
    resize_cuda_image(row, column);
    cropinner(cache, dest, thisrow, thiscol, 1./(thisrow*thiscol));
  }else{
    resize_cuda_image(thisrowp, thiscolp);
    crop(src, cache, thisrow, thiscol);
    myFFTM(locplan[1],cache,cache);
    resize_cuda_image(row, column);
    cropinner(cache, dest, thisrowp, thiscolp, 1./(thisrowp*thiscolp));
  }
  myIFFT(dest, dest);
}
void broadBand_constRatio::applyA(complexFormat* src, complexFormat* dest, char zoom){
  myFFT(src, dest);
  if(zoom == 0) {
    resize_cuda_image(thisrow, thiscol);
    padinner(dest, cache, row, column, 1./(thisrow*thiscol));
    myIFFTM(*locplan,cache,cache);
    resize_cuda_image(row, column);
    crop(cache, dest, thisrow, thiscol);
  }else{
    resize_cuda_image(thisrowp, thiscolp);
    cropinner(dest, cache, row, column, 1./(thisrowp*thiscolp));
    myIFFTM(locplan[1],cache,cache);
    resize_cuda_image(row, column);
    pad(cache, dest, thisrowp, thiscolp);
  }
}
char broadBand_constRatio::nextPattern(complexFormat* currentp, complexFormat* nextp, complexFormat* origin, char transpose){
  if(patternptr >= nlambda-1) return 0;
  if(patternptr == 0){
    patternptr = nmiddle+1;
    if(transpose)
      applyAT(origin, nextp, 0);
    else
      applyA(origin, nextp, 0);
    return patternptr;
  }
  if(patternptr > nmiddle){
    if(transpose)
      applyAT(currentp, nextp, 0);
    else
      applyA(currentp, nextp, 0);
    return ++patternptr;
  }
  if(patternptr == nmiddle){
    if(transpose)
      applyAT(origin, nextp, 1);
    else
      applyA(origin, nextp, 1);
    return --patternptr;
  }
  if(transpose)
    applyAT(currentp, nextp, 1);
  else
    applyA(currentp, nextp, 1);
  return --patternptr;
}
char broadBand_constRatio::skipPattern(){
  if(patternptr >= nlambda-1) return 0;
  if(patternptr >= nmiddle){
    patternptr++;
  }else{
    if(patternptr == 0){
      patternptr = nmiddle+1;
    }else{
      patternptr--;
    }
  }
  return 1;
}
void broadBand_constRatio::generateMWL(void* d_input, void* d_patternSum, void* single){
  myCuDMalloc(complexFormat, d_inputWave, row*column);
  resize_cuda_image(row, column);
  extendToComplex( (Real*)d_input, d_inputWave);
  myFFT(d_inputWave, d_inputWave);
  applyNorm(d_inputWave, 1./sqrt(row*column));
  cudaConvertFO(d_inputWave);
  getMod2(d_inputWave, d_inputWave);
  myMemcpyD2D(single, d_inputWave, row*column*sizeof(complexFormat));
  getReal((Real*)d_patternSum, d_inputWave, spectra[0]);

  for(int i = 1; i < nlambda; i++){
    applyA(d_inputWave, d_inputWave);
    addReal((Real*)d_patternSum, d_inputWave, spectra[i]);
  }
  myCuFree(d_inputWave);
}
