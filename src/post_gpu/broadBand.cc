#include "broadBand.h"
#include "cudaConfig.h"
#include "cuPlotter.h"
#include <fstream>
#include <gsl/gsl_spline.h>
#include <math.h>
#include <string.h>

using namespace std;


int inline nearestEven(Real x){
  return round(x/2)*2;
}

void broadBand::calcPixelWeights(){
  int sz = row/2*sizeof(Real);
  Real* loc_pixel_weight = (Real*) ccmemMngr.borrowCleanCache(sz);
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
  myMemcpyH2D(pixel_weight, loc_pixel_weight, sz);
}


void broadBand::init(int nrow, int ncol, int nlambda_, double* lambdas_, double* spectra_){
  nlambda = nlambda_;
  spectra = spectra_;
  row = nrow;
  column = ncol;
  rows = (int*)ccmemMngr.borrowCache(sizeof(int)*nlambda);
  cols = (int*)ccmemMngr.borrowCache(sizeof(int)*nlambda);
  plt.init(row,column);
  locplan = (int*)ccmemMngr.borrowCache(sizeof(int)*nlambda);
  for(int i = 0; i < nlambda; i++){
    rows[i] = nearestEven(row*lambdas_[i]);
    cols[i] = nearestEven(column*lambdas_[i]);
    createPlan(locplan+i, rows[i], cols[i]);
  }
  calcPixelWeights();
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
  spectra = (double*) ccmemMngr.borrowCleanCache(nlambda*sizeof(double));
  locplan = (int*)ccmemMngr.borrowCache(sizeof(int)*nlambda);
  for(int i = 0; i < nlambda; i++){
    rows[i] = row+(i+(minlambda-1)/stepsize)*2*jump;
    cols[i] = column+(i+(minlambda-1)/stepsize)*2*jump;
    printf("%d: (%d,%d)\n",i, rows[i],cols[i]);
    createPlan(locplan+i, rows[i], cols[i]);
  }
}
Real broadBand::init(int nrow, int ncol, double* lambdasi, double* spectrumi, int narray){
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
  myFree(spectrumraw);
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
  locplan = (int*)ccmemMngr.borrowCache(sizeof(int)*nlambda);
  for(int i = 0; i < nlambda; i++){
    rows[i] = row+i*2*jump;
    cols[i] = column+i*2*jump;
    printf("%d: (%d,%d)=%f\n",i, rows[i],cols[i],spectra[i]/=spectrumi[narray-1]);
    createPlan( locplan+i, rows[i], cols[i]);
  }
  spectrumfile.open("spectra.txt", std::ios::out);
  for(int i = 0; i < nlambda; i++){
    spectrumfile<<1+stepsize*i+skiplambda<<" "<<spectra[i]/narray*nlambda<<std::endl;
  }
  spectrumfile.close();
  calcPixelWeights();
  return spectrumi[narray-1];
}
void broadBand::resetSpectra(){
  for(int i = 0; i < nlambda; i++){
    spectra[i] = 1./nlambda;
  }
}
void broadBand::writeSpectra(const char* filename){
  std::ofstream spectrafile;
  spectrafile.open(filename,ios::out);
  for(int i = 0; i < nlambda; i++){
    spectrafile<<Real(rows[i])/rows[0]<<" "<<spectra[i]<<std::endl;
  }
  spectrafile.close();
}
void broadBand::generateMWL(void* d_input, void* d_patternSum, void* single){
  Real *d_pattern = (Real*) memMngr.borrowCache(row*column*sizeof(Real));
  complexFormat *d_intensity = (complexFormat*)memMngr.borrowCache(rows[nlambda-1]*cols[nlambda-1]*sizeof(complexFormat));
  complexFormat* d_patternAmp = (complexFormat*)memMngr.borrowCache(row*column*sizeof(Real)*2);
  complexFormat *d_inputWave = (complexFormat*)memMngr.borrowCleanCache(row*column*sizeof(complexFormat));
  resize_cuda_image(row, column);
  assignReal( (Real*)d_input, d_inputWave);
  for(int i = 0; i < nlambda; i++){
    int thisrow = rows[i];
    int thiscol = cols[i];
    resize_cuda_image(thisrow, thiscol);
    pad( d_inputWave, d_intensity, row, column);
    myFFTM( locplan[i], d_intensity,d_intensity);
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
  myCuFree(d_pattern);
  myCuFree(d_intensity);
  myCuFree(d_patternAmp);
}
