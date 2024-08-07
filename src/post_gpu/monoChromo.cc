#include "orthFitter.hpp"
#include "cudaConfig.hpp"
#include "cub_wrap.hpp"
#include "cuPlotter.hpp"
#include "tvFilter.hpp"
#include <fstream>
#include <iostream>
#include <math.h>
#include <string.h>
#include "monoChromo.hpp"

using namespace std;

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
  Real lr = 1.4;
  Real k = 0.8;
  Real beta1 = 1;//0.1;
  //Real beta1 = 0;//0.1;
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
  complexFormat *patternstep = (complexFormat*)memMngr.borrowCache(sz);
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
          myFFTM( locplan[j], padded, padded);
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
    if(updateA){
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
      clearCuMem(patternstep,  sz);
      add( patternstep, deltab, spectra[monoidx]);
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
        add((complexFormat*)patternstep, fbi, spectra[j]);
      }
      //multiplyPixelWeight( patternstep, pixel_weight);
      if(beta1){
        //updateMomentum( patternstep, momentum, 2*beta1);
        add( momentum, patternstep, 2*beta1);
        applyNorm(momentum, k);
        if(beta2) {
          adamUpdateV( adamv, patternstep, beta2);
          adamUpdate( (complexFormat*)d_output, momentum, adamv, lr, adamepsilon);
        }else add((complexFormat*)d_output, momentum, 1);
      }else{
        add( (complexFormat*)d_output, patternstep, lr);
      }
      forcePositive((complexFormat*)d_output);
      /* //FISTA update, not quite effective
         add( patternstep, (complexFormat*)d_output, 1);
         getReal( (Real*)fbi,patternstep);
         FISTA((Real*)fbi, (Real*)patternstep, 1e-6, 70, &applyC);
         extendToComplex( (Real*)patternstep, fbi);
         Real tmp = 0.5+sqrt(0.25+tk*tk);
         Real fact1 = (tk-1)/tmp;
         tk = tmp;
         applyNorm( (complexFormat*)d_output, -fact1);
         add( (complexFormat*)d_output, fbi, 1+fact1);
         */
    }
    //myFFT(d_output, d_output);
    //cudaConvertFO((complexFormat*)d_output);
    //zeroEdge( (complexFormat*)d_output, 30);
    //cudaConvertFO((complexFormat*)d_output);
    //applyNorm((complexFormat*)d_output, 1./(row*column));
    //myIFFT(d_output, d_output);
  }
  if(writeResidual) {
    plt.plotComplex(deltab, REAL, 0, 1, "residual_pulseGen", 1, 0, 1);
    add(deltab,(complexFormat*)d_input, -1);
    fresidual.close();
  }
  myFree(momentum_a);
  myFree(matrix);
  myFree(right);
  myCuFree(patternstep);
  myCuFree(multiplied);
  myCuFree(momentum);
  myCuFree(adamv);
  myCuFree(padded);
  myCuFree(fbi);
  myCuFree(fftb);
  myCuFree(deltab);

}
void monoChromo_constRatio::solveMWL(void* d_input, void* d_output, int noiseLevel, bool restart, int nIter, bool updateX, bool updateA)
{
  bool writeResidual = 1;
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
  size_t sz = row*column*sizeof(complexFormat);
  init_fft(row,column);
  resize_cuda_image(row, column);
  Real lr = 1.4;
  Real k = 0.9;
  Real beta1 = 1;//0.1;
  Real beta2 = 0.;//5;//0.99;
  Real adamepsilon = 1e-4;
  if(!restart)
    clearCuMem(d_output,  sz);
  complexFormat *deltab = (complexFormat*)memMngr.borrowCache(sz);
  complexFormat *momentum = 0;
  Real *adamv = 0;
  if(beta1) {
    momentum = (complexFormat*)memMngr.borrowCleanCache(sz);
  }
  if(beta2) {
    adamv = (Real*)memMngr.borrowCleanCache(sz/2);
  }
  complexFormat *patternstep = (complexFormat*)memMngr.borrowCache(sz);
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
  complexFormat *fbj;

  int nmem = 0;
  if(updateA) {
    gs = (void**)ccmemMngr.borrowCache((nlambda)*sizeof(void*));
    for(int j = 0; j < nlambda; j++){
      gs[j] = memMngr.borrowCache(sz);
      nmem++;
      if(getGPUFreeMem()<1000) break;
    }
  }else {
    fbj = fbi = (complexFormat*)memMngr.borrowCache(sz);
  }
  // Real tk = 1;
  bool calcDeltab = 0;
  double* matrix = (double*) ccmemMngr.borrowCleanCache(nlambda*nlambda*sizeof(double));
  double* right = (double*) ccmemMngr.borrowCache(nlambda*sizeof(double));
  for(int iter = 0; iter < nIter; iter++){
    if(updateX){
      add(deltab, (complexFormat*)d_input, (complexFormat*)d_output, -spectra[0]);
    }
    if(updateA){
      matrix[0] = innerProd(d_output, d_output, sptimg);
      right[0] = innerProd(d_input, d_output, sptimg);
    }
    initptr();
    for(int j = 1; j < nlambda; j++){
      if(gs && j <= nmem) {
        fbi = j>1?(complexFormat*)gs[j-2]:(complexFormat*)d_output;
        fbj = (complexFormat*)gs[j-1];
      }else fbi = fbj;
      nextPattern(fbi, fbj, (complexFormat*)d_output);
      if(updateA){
        for(int i = 1; i < min(j+1,nmem); i++){
          matrix[i+j*nlambda] = innerProd(fbj, gs[i-1], sptimg);
        }
        matrix[j*nlambda] = innerProd(fbj, d_output, sptimg);
        right[j] = innerProd(fbj, d_input, sptimg);
      }
      if(updateX) add(deltab, fbj, -spectra[j]);
    }
    if(writeResidual) {
      getMod2(multiplied, deltab);
      fresidual<<iter<<" "<<findSum(multiplied)<<endl;
    }
    if(calcDeltab) break;
    if(gs){
      int nblk = (nlambda-2)/(nmem-1)+1;
      for(int iblk = 1; iblk < nblk; iblk++){
        int offset = 1+iblk*(nmem-1);
        patternptr = (offset-1<=nmiddle)?nmiddle+1-offset:(offset-1);
        nextPattern((complexFormat*)gs[nmem-2], (complexFormat*)gs[0],(complexFormat*) d_output);
        matrix[offset*(1+nlambda)] = innerProd(gs[0], gs[0],sptimg);
        for(int j = 1+offset; j < nlambda; j++){
          int subidx = j - offset;
          if(subidx < nmem) {
            fbi = (complexFormat*)gs[subidx-1];
            fbj = (complexFormat*)gs[subidx];
          }else fbi = fbj;
          nextPattern(fbi, fbj, (complexFormat*)d_output);
          int nprod = min(subidx+1,nmem);
          for(int i = 0; i < nprod; i++){
            matrix[i+offset+j*nlambda] = innerProd(fbj, gs[i],sptimg);
          }
        }
      }
      for(int i = 0; i < nlambda; i++){
        for(int j = i+1; j < nlambda; j++){
          matrix[j+i*nlambda] = matrix[i+j*nlambda];
        }
      }
      ofstream matfile;
      matfile.open("matrix.txt", ios::out);
      for(int j = 0; j < nlambda; j++){
        for(int i = 0; i < nlambda; i++)
          matfile << matrix[i+j*nlambda] << " ";
        matfile << endl;
      }
      Fit_fast_matrix(spectra, nlambda, matrix, right);
      //spectra[nlambda-2] = spectra[nlambda-1] = 0;
      //for(int il = 0; il < nlambda-2; il++){
      //  spectra[il] *= 1.1;
      //}
      printf("Fit spectrum done. %f\n", spectra[0]);
      if(!updateX) {
        iter = nIter-2;
        updateA = 0;
        updateX = 1;
        calcDeltab = 1;
        continue;
      }
    }
    if(updateX){
      //overExposureZeroGrad( deltab, (complexFormat*)d_input, noiseLevel);
      if(iter > 20 && noiseLevel){
        getReal( (Real*)fbi,deltab);
        FISTA((Real*)fbi, (Real*)fbi, 1e-5*sqrt(noiseLevel), 20, &applyC);
        extendToComplex( (Real*)fbi, deltab);
      }
      clearCuMem(patternstep,  sz);
      add( patternstep, deltab, spectra[0]);
      initptr();
      for(int j = 1; j < nlambda; j++){
        nextPattern(fbi, fbi, deltab, 1);
        add((complexFormat*)patternstep, fbi, spectra[j]);
      }
      if(beta1){
        updateMomentum( patternstep, momentum, 2*beta1);
        applyNorm(momentum, k);
        if(beta2) {
          adamUpdateV( adamv, patternstep, beta2);
          adamUpdate( (complexFormat*)d_output, momentum, adamv, lr, adamepsilon);
        }else add((complexFormat*)d_output, momentum, 1);
      }else{
        add( (complexFormat*)d_output, patternstep, lr);
      }
      forcePositive((complexFormat*)d_output);
    }
  }
  //myFFT(d_output, d_output);
  //cudaConvertFO((complexFormat*)d_output);
  //zeroEdge( (complexFormat*)d_output, 30);
  //cudaConvertFO((complexFormat*)d_output);
  //applyNorm((complexFormat*)d_output, 1./(row*column));
  //myIFFT(d_output, d_output);
  if(writeResidual) {
    plt.plotComplex(deltab, REAL, 0, 1, "residual_pulseGen", 1, 0, 1);
    add(deltab,(complexFormat*)d_input, -1);
    fresidual.close();
  }
  myFree(momentum_a);
  myFree(matrix);
  myFree(right);
  myCuFree(patternstep);
  myCuFree(multiplied);
  myCuFree(momentum);
  myCuFree(adamv);
  if(!updateA) myCuFree(fbi);
  myCuFree(deltab);

}
