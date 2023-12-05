#include "monoChromo.h"
#include "orthFitter.h"
#include "cudaConfig.h"
#include "imgio.h"
#include "cub_wrap.h"
#include "cuPlotter.h"
#include "tvFilter.h"
#include <fstream>
#include <iostream>
#include <gsl/gsl_spline.h>
#include <math.h>
#include <string.h>

using namespace std;


int inline nearestEven(Real x){
  return round(x/2)*2;
}

void monoChromo::assignRef(void* wavefront, int i){
  assignRef_d((complexFormat*)wavefront, (uint32_t*)d_maskMap, (complexFormat*)(refs[i]), pixCount);
}
void monoChromo::assignRef(void* wavefront){
  assignRef(wavefront, 0);
  for(int i = 1; i < nlambda; i++) 
    myMemcpyD2D(refs[i], refs[0], pixCount*sizeof(complexFormat));
}

void monoChromo::calcPixelWeights(){
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


void monoChromo::init(int nrow, int ncol, int nlambda_, double* lambdas_, double* spectra_){
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
void monoChromo::init(int nrow, int ncol, double minlambda, double maxlambda){
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
void monoChromo::initRefs(const char* maskFile){  //mask file, full image size, 
  //==create reference mask and it's map: d_maskMap, allocate refs==
  int mrow, mcol;
  Real* refMask = readImage(maskFile, mrow, mcol);
  pixCount = 0;
  for(int idx = 0; idx < mrow*mcol ; idx++){
    if(refMask[idx] > 0.5) pixCount++;
  }
  uint32_t* maskMap = (uint32_t*)ccmemMngr.borrowCache(pixCount*sizeof(uint32_t));
  int idx = 0, ic = 0;
  for(int x = 0; x < mrow ; x++){
  for(int y = 0; y < mcol ; y++){
    if(refMask[idx] > 0.5) {
      maskMap[ic] = (x+(row-mrow)/2)*row + y+(column-mcol)/2;
      ic++;
    }
    idx++;
  }
  }
  d_maskMap = (uint32_t*)memMngr.borrowCache(pixCount*sizeof(uint32_t));
  myMemcpyH2D(d_maskMap, maskMap, pixCount*sizeof(uint32_t));
  myFree(maskMap);
  myFree(refMask);
  refs = (void**)ccmemMngr.borrowCache(nlambda*sizeof(void*));
  for(int i = 0; i < nlambda; i++){
    refs[i] = memMngr.borrowCache(pixCount*sizeof(complexFormat));
  }
  printf("mask has %d pixels\n", pixCount);
}
void monoChromo::generateMWLRefPattern(void* d_patternSum, bool debug){
  complexFormat *amp = (complexFormat*)memMngr.borrowCleanCache(rows[nlambda-1]*cols[nlambda-1]*sizeof(complexFormat));
  complexFormat *camp = (complexFormat*)memMngr.borrowCache(row*column*sizeof(complexFormat));
  Real *d_pattern= (Real*)memMngr.borrowCleanCache(row*column*sizeof(Real));
  resize_cuda_image(row, column);
  for(int i = 0; i < nlambda; i++){
    int thisrow = rows[i];
    int thiscol = cols[i];
    expandRef((complexFormat*)(refs[i]), (complexFormat*)amp, (uint32_t*)d_maskMap, thisrow, thiscol, pixCount);
    myFFTM(locplan[i], amp, amp);
    if(debug && i == 1){
      resize_cuda_image(thisrow, thiscol);
      plt.init(thisrow, thiscol);
      plt.plotComplex(amp, MOD2, 1, spectra[i]/(thiscol*thisrow), "debug", 1, 0, 1);
      exit(0);
    }
    cropinner(amp,camp,thisrow,thiscol, sqrt(spectra[i]/(thiscol*thisrow)));
    cudaConvertFO(camp);
    if(i==0) {
      getMod2((Real*)d_patternSum, camp);
    }else{
      getMod2(d_pattern, camp);
      add((Real*)d_patternSum, (Real*)d_pattern, 1);
    }
    clearCuMem(amp,  thisrow*thiscol*sizeof(complexFormat));
  }
  myCuFree(amp);
  myCuFree(camp);
  myCuFree(d_pattern);
}
void monoChromo::clearRefs(){
  for(int i = 0; i < nlambda; i++) clearCuMem(refs[i],  pixCount*sizeof(complexFormat));
  //for(int i = 0; i < 1; i++) clearCuMem(refs[i],  pixCount*sizeof(complexFormat));
}
void monoChromo::reconRefs(void* d_patternSum){
  void* devstates = newRand(rows[nlambda-1]*cols[nlambda-1]);
  resize_cuda_image(rows[nlambda-1],cols[nlambda-1]);
  initRand(devstates, time(NULL));
  resize_cuda_image(row, column);
  clearRefs();
  uint32_t sz = memMngr.getSize(d_patternSum);
  Real* patternRecon = (Real*)memMngr.borrowCache(sz);
  Real* residual = (Real*)memMngr.borrowCache(sz);
  complexFormat *amp = (complexFormat*)memMngr.borrowCache(rows[nlambda-1]*cols[nlambda-1]*sizeof(complexFormat));
  complexFormat *camp = (complexFormat*)memMngr.borrowCache(row*column*sizeof(complexFormat));
  Real *campm = (Real*)memMngr.borrowCache(row*column*sizeof(Real));
  Real *deltao = (Real*)memMngr.borrowCache(row*column*sizeof(Real));
  Real *deltao2 = (Real*)memMngr.borrowCache(row*column*sizeof(Real));
  complexFormat* reftmp = (complexFormat*)memMngr.borrowCache(sizeof(complexFormat)*pixCount);
  Real kfactor = 0;
  Real stepsize = 1;
  plt.plotFloat(d_patternSum, MOD, 0, 1, "target_pattern", 1, 0, 1);
  for(int niter = 0; niter < 200; niter++){
    generateMWLRefPattern(patternRecon);
    myMemcpyD2D(residual, d_patternSum, sz);
    addRemoveOE(residual, (Real*)patternRecon, -1);
    cudaConvertFO(residual);
    if(niter%200 == 0){
      getMod2(deltao2, residual);
      printf("residuale=%f\n", findSum(deltao2));
    }

    for(int i = 0; i < nlambda; i++){
      int thisrow = rows[i];
      int thiscol = cols[i];
      Real normi = 1./(thiscol*thisrow);
      for(int it = 0; it < 5; it++){
        clearCuMem(amp,  thisrow*thiscol*sizeof(complexFormat));
        expandRef(it == 0? (complexFormat*)(refs[i]) : reftmp, (complexFormat*)amp, (uint32_t*)d_maskMap, thisrow, thiscol, pixCount);
        myFFTM(locplan[i], amp, amp);
        if(it == 0) {
          cropinner(amp,camp,thisrow,thiscol,sqrt(normi));
          getMod2(campm, camp);
          //add(campm, residual, stepsize*spectra[i]);
          //add(campm, residual, stepsize);
          add(campm, residual, stepsize/spectra[i]);
        }
        applyModAbsinner(amp,campm, thisrow,thiscol, 1./normi, devstates);
        //if(i == 1){
        //  plt.init(thisrow, thiscol);
        //  plt.plotComplex(amp, MOD2, 1, 1./normi, "debug", 1, 0, 1);
        //  exit(0);
        //}
        myIFFTM(locplan[i], amp, amp);
        saveRef(reftmp, (complexFormat*)amp, (uint32_t*)d_maskMap, thisrow, thiscol, pixCount, normi);
      }
      //add(campm, residual, -stepsize*spectra[i]);  //previous intensity
      //add(campm, residual, -stepsize);  //previous intensity
      add(campm, residual, -stepsize/spectra[i]);  //previous intensity
      clearCuMem(amp,  thisrow*thiscol*sizeof(complexFormat));
      expandRef(reftmp, (complexFormat*)amp, (uint32_t*)d_maskMap, thisrow, thiscol, pixCount);
      myFFTM(locplan[i], amp, amp);
      cropinner(amp,camp,thisrow,thiscol,sqrt(normi));
      getMod2(deltao, camp);  //delta intensity
      add(deltao, campm, -1);
      multiply(deltao2, residual, deltao);
      kfactor = -findSum(deltao2);
      getMod2(deltao2, deltao);
      kfactor /= findSum(deltao2);
      //if(kfactor > 0) continue;
      if(fabs(kfactor) > 1) kfactor *= 1./fabs(kfactor);
      add(campm, deltao, -kfactor);
      //add(residual, deltao, kfactor*spectra[i]);
      applyModAbsinner(amp,campm, thisrow,thiscol, 1./normi, devstates);
      myIFFTM(locplan[i], amp, amp);
      saveRef(reftmp, (complexFormat*)amp, (uint32_t*)d_maskMap, thisrow, thiscol, pixCount, normi);
      getMod2((Real*)camp, residual);
      //printf("residual=%f\n", findSum((Real*)camp));
      for(int it = 0; it < 5; it++){
        clearCuMem(amp,  thisrow*thiscol*sizeof(complexFormat));
        expandRef((complexFormat*)(refs[i]), (complexFormat*)amp, (uint32_t*)d_maskMap, thisrow, thiscol, pixCount);
        myFFTM(locplan[i], amp, amp);
        Real normi = 1./(thiscol*thisrow);
        applyModAbsinner(amp,campm, thisrow,thiscol, 1./normi, devstates);
        myIFFTM( locplan[i], amp, amp);
        saveRef((complexFormat*)(refs[i]), (complexFormat*)amp, (uint32_t*)d_maskMap, thisrow, thiscol, pixCount, normi);
      }
    }
    /*
    resize_cuda_image(pixCount, 1);
    for(int i = 1; i < nlambda; i++) {
      add((complexFormat*)(refs[0]), (complexFormat*)(refs[i]));
    }
    applyNorm((complexFormat*)(refs[0]),1./nlambda);
    for(int i = 1; i < nlambda; i++) {
      myMemcpyD2D((complexFormat*)(refs[i]),(complexFormat*)(refs[0]),pixCount*sizeof(complexFormat));
    }
    resize_cuda_image(row, column);
    */
  }
  resize_cuda_image(row, column);
  for(int i = 0; i < nlambda; i++) {
    clearCuMem(amp,  rows[i]*cols[i]*sizeof(complexFormat));
    expandRef((complexFormat*)(refs[i]), (complexFormat*)amp, (uint32_t*)d_maskMap, rows[i], cols[i], pixCount);
    crop(amp,camp,rows[i],cols[i]);
    plt.plotComplex(camp, MOD2, 0, 1, (string("recon")+char(i+'0')).c_str(), 0, 0, 0);
    plt.plotComplex(camp, PHASE, 0, 1, (string("reconphase")+char(i+'0')).c_str(), 0, 0, 0);
  }

  generateMWLRefPattern(patternRecon);
  plt.plotFloat(patternRecon, MOD, 0, 1, "recon_pattern", 1, 0, 1);
  myMemcpyD2D(residual, d_patternSum, sz);
  addRemoveOE(residual, (Real*)patternRecon, -1);
  plt.plotFloat(residual, MOD, 0, 1, "residual", 1, 0, 1);

  myCuFree(patternRecon);
  myCuFree(residual);
  myCuFree(amp);
  myCuFree(camp);
}
void monoChromo::generateMWL(void* d_input, void* d_patternSum, void* single){
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
Real innerProd(void* a, void* b, void* param){
  Real* tmp = (Real*)memMngr.borrowCache(memMngr.getSize(a)/2);
  multiplyReal(tmp,(complexFormat*)a, (complexFormat*)b);
  applyMask(tmp, (Real*)param);
  Real sum = findSum(tmp);
  myCuFree(tmp);
  return sum;
}

void applyC(Real* input, Real* output){
  myMemcpyD2D(output, input, memMngr.getSize(input));
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
  int d = row/2-20;
  rect spt;
  spt.starty = spt.startx = d;
  spt.endx = spt.endy = row-d-1;
  rect *cuda_spt;
  cuda_spt = (rect*)memMngr.borrowCache(sizeof(rect));
  myMemcpyH2D(cuda_spt, &spt, sizeof(spt));
  Real *sptimg = 0;
  if(updateA){
    sptimg = (Real*)memMngr.borrowCache(row*column*sizeof(Real));
    createMaskBar(sptimg, cuda_spt, 0);
    myCuFree(cuda_spt);
    applyMaskBar(sptimg, (complexFormat*)d_input, 0.99);
    plt.plotFloat(sptimg, MOD, 0, 1, "innerprodspt", 0);
  }
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
    //myMemcpyD2D(d_output, d_input, sz);
    //zeroEdge( (complexFormat*)d_output, 150);
    clearCuMem(d_output,  sz);
  }
  complexFormat *deltab = (complexFormat*)memMngr.borrowCache(sz);
  complexFormat *momentum = 0;
  Real *adamv = 0;
  if(beta1) {
    momentum = (complexFormat*)memMngr.borrowCleanCache(sz);
  }
  if(beta2) {
    adamv = (Real*)memMngr.borrowCleanCache(sz/2);
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
    momentum_a = (Real*) ccmemMngr.borrowCleanCache(nlambda*sizeof(Real));
  }
  complexFormat *fbi;
  if(!updateX) {
    myIFFT((complexFormat*)d_output, fftb);
    cudaConvertFO(fftb);
  }
  int nmem = nlambda/8;
  if(updateA) {
    gs = (void**)ccmemMngr.borrowCache((nmem+1)*sizeof(void*));
    for(int j = 0; j < nmem+1; j++){
      gs[j] = memMngr.borrowCache(sz);
    }
  }else fbi = (complexFormat*)memMngr.borrowCache(sz);
 // Real tk = 1;
  bool calcDeltab = 0;
  double* matrix = (double*) ccmemMngr.borrowCache(nlambda*nlambda*sizeof(double));
  double* right = (double*) ccmemMngr.borrowCache(nlambda*sizeof(double));
  for(int i = 0; i < nIter; i++){
    if(updateX||i==0||!gs) {
      myIFFT((complexFormat*)d_output, fftb);
    }
    if(gs && monoidx < nmem) 
      myMemcpyD2D(gs[monoidx], d_output, sz);
    if(updateX){
      myMemcpyD2D(deltab, d_input, sz);
      add(deltab, (complexFormat*)d_output, -spectra[monoidx]);
    }
    for(int j = 0; j < nlambda; j++){
      if(j != monoidx){
        if(!updateA && spectra[j]<=0) continue;
        size_t N = rows[j]*cols[j];
        if(gs){
          if(j <= nmem) fbi = (complexFormat*)gs[j];
        }
        if(!gs || updateX || i==0){
          resize_cuda_image(rows[j], cols[j]);
          if(rows[j] > row) padinner(fftb, padded, row, column, 1./N);
          else cropinner(fftb, padded, row, column, 1./N);
          myFFTM( ((int*)locplan)[j], padded, padded);
          resize_cuda_image(row, column);
          if(rows[j] > row) crop(padded, fbi, rows[j], cols[j]);
          else pad(padded, fbi, rows[j], cols[j]);
        }
      }
      if(gs){
        for(int i = 0; i < min(j+1,nmem); i++){
          matrix[i+j*nlambda] = innerProd(j==monoidx?d_output:fbi, gs[i], sptimg);
        }
        right[j] = innerProd(fbi, d_input, sptimg);
      }
      if(updateX) add(deltab, fbi, -spectra[j]);
    }
    if(gs){
      int nblk = (nlambda-1)/nmem+1;
      for(int iblk = 1; iblk < nblk; iblk++){
        if(monoidx >= iblk*nmem && monoidx < iblk*nmem+nmem)
          myMemcpyD2D(gs[monoidx-iblk*nmem], d_output, sz);
        for(int j = iblk*nmem; j < nlambda; j++){
          if(j <= iblk*nmem+nmem) fbi = (complexFormat*)gs[j-iblk*nmem];
          if(j != monoidx){
            size_t N = rows[j]*cols[j];
            resize_cuda_image(rows[j], cols[j]);
            if(rows[j] > row) padinner(fftb, padded, row, column, 1./N);
            else cropinner(fftb, padded, row, column, 1./N);
            myFFTM( locplan[j], padded, padded);
            resize_cuda_image(row, column);
            if(rows[j] > row) crop(padded, fbi, rows[j], cols[j]);
            else pad(padded, fbi, rows[j], cols[j]);
          }
          int nprod = iblk == nblk-1? j-iblk*nmem:nmem-1;
          for(int i = 0; i <= nprod; i++){
            matrix[i+iblk*nmem+j*nlambda] = innerProd(j == monoidx? d_output:fbi, gs[i],sptimg);
          }
        }
      }
      for(int i = 0; i < nlambda; i++){
        for(int j = i+1; j < nlambda; j++){
          matrix[j+i*nlambda] = matrix[i+j*nlambda];
        }
      }
    }
    if(writeResidual) {
      getMod2(multiplied, deltab);
      fresidual<<i<<" "<<findSum(multiplied)<<endl;
    }
    if(calcDeltab) break;
    if(useOrth&&updateA){
      Fit_fast_matrix(spectra, nlambda, matrix, right);
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
    if(updateX){
      //overExposureZeroGrad( deltab, (complexFormat*)d_input, noiseLevel);
      if(i > 20 && noiseLevel){
        getReal( (Real*)fbi,deltab);
        FISTA((Real*)fbi, (Real*)fbi, 1e-5*sqrt(noiseLevel), 20, &applyC);
        extendToComplex( (Real*)fbi, deltab);
      }
      clearCuMem(deltabprev,  sz);
      add( deltabprev, deltab, spectra[monoidx]);
      for(int j = 1; j < nlambda; j++){
        if(j == monoidx) continue;
        if(spectra[j]<=0) continue;
        resize_cuda_image(rows[j], cols[j]);
        if(rows[j] > row) pad((complexFormat*)deltab, padded, row, column);
        else crop((complexFormat*)deltab, padded, row, column);
        myIFFTM(locplan[j], padded, padded);
        resize_cuda_image(row, column);
        if(rows[j] > row) cropinner(padded, fbi, rows[j], cols[j], 1./(row*column));
        else padinner(padded, fbi, rows[j], cols[j], 1./(row*column));
        myFFT(fbi, fbi);
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
    plt.plotComplex(deltab, MOD, 0, 1, "broad_recon_log", 1, 0, 1);
    fresidual.close();
  }
  myFree(momentum_a);
  myFree(matrix);
  myFree(right);
  myCuFree(deltabprev);
  myCuFree(multiplied);
  myCuFree(momentum);
  myCuFree(adamv);
  myCuFree(padded);
  myCuFree(fbi);
  myCuFree(fftb);
  myCuFree(deltab);

}
