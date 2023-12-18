#include "spectImaging.hpp"
#include "cudaConfig.hpp"
#include <iostream>
#include "cub_wrap.hpp"
#include <math.h>
#include "cuPlotter.hpp"
#include <string>
#include "imgio.hpp"
using namespace std;
void spectImaging::assignRef(void* wavefront, int i){
  assignRef_d((complexFormat*)wavefront, (uint32_t*)d_maskMap, (complexFormat*)(refs[i]), pixCount);
}
void spectImaging::assignRef(void* wavefront){
  assignRef(wavefront, 0);
  for(int i = 1; i < nlambda; i++) 
    myMemcpyD2D(refs[i], refs[0], pixCount*sizeof(complexFormat));
}

void spectImaging::initRefs(const char* maskFile){  //mask file, full image size, 
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
void spectImaging::generateMWLRefPattern(void* d_patternSum, bool debug){
  complexFormat *amp = (complexFormat*)memMngr.borrowCleanCache(rows[nlambda-1]*cols[nlambda-1]*sizeof(complexFormat));
  complexFormat *camp = (complexFormat*)memMngr.borrowCache(row*column*sizeof(complexFormat));
  Real *d_pattern= (Real*)memMngr.borrowCleanCache(row*column*sizeof(Real));
  resize_cuda_image(row, column);
  for(int i = 0; i < nlambda; i++){
    int thisrow = rows[i];
    int thiscol = cols[i];
    resize_cuda_image(pixCount, 1);
    expandRef((complexFormat*)(refs[i]), (complexFormat*)amp, (uint32_t*)d_maskMap, thisrow, thiscol, row, column);
    resize_cuda_image(row, column);
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
void spectImaging::clearRefs(){
  for(int i = 0; i < nlambda; i++) clearCuMem(refs[i],  pixCount*sizeof(complexFormat));
  //for(int i = 0; i < 1; i++) clearCuMem(refs[i],  pixCount*sizeof(complexFormat));
}
void spectImaging::reconRefs(void* d_patternSum){
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

    resize_cuda_image(pixCount, 1);
    for(int i = 0; i < nlambda; i++){
      int thisrow = rows[i];
      int thiscol = cols[i];
      Real normi = 1./(thiscol*thisrow);
      for(int it = 0; it < 5; it++){
        clearCuMem(amp,  thisrow*thiscol*sizeof(complexFormat));
        expandRef(it == 0? (complexFormat*)(refs[i]) : reftmp, (complexFormat*)amp, (uint32_t*)d_maskMap, thisrow, thiscol, row, column);
        resize_cuda_image(row,column);
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
        resize_cuda_image(pixCount, 1);
        saveRef(reftmp, (complexFormat*)amp, (uint32_t*)d_maskMap, thisrow, thiscol, row, column, normi);
      }
      //add(campm, residual, -stepsize*spectra[i]);  //previous intensity
      //add(campm, residual, -stepsize);  //previous intensity
      resize_cuda_image(row,column);
      add(campm, residual, -stepsize/spectra[i]);  //previous intensity
      clearCuMem(amp,  thisrow*thiscol*sizeof(complexFormat));
      resize_cuda_image(pixCount, 1);
      expandRef(reftmp, (complexFormat*)amp, (uint32_t*)d_maskMap, thisrow, thiscol, row, column);
      resize_cuda_image(row,column);
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
      resize_cuda_image(pixCount, 1);
      saveRef(reftmp, (complexFormat*)amp, (uint32_t*)d_maskMap, thisrow, thiscol, row, column, normi);
      resize_cuda_image(row,column);
      getMod2((Real*)camp, residual);
      resize_cuda_image(pixCount, 1);
      //printf("residual=%f\n", findSum((Real*)camp));
      for(int it = 0; it < 5; it++){
        clearCuMem(amp,  thisrow*thiscol*sizeof(complexFormat));
        expandRef((complexFormat*)(refs[i]), (complexFormat*)amp, (uint32_t*)d_maskMap, thisrow, thiscol, row, column);
        resize_cuda_image(row,column);
        myFFTM(locplan[i], amp, amp);
        Real normi = 1./(thiscol*thisrow);
        applyModAbsinner(amp,campm, thisrow,thiscol, 1./normi, devstates);
        myIFFTM( locplan[i], amp, amp);
        resize_cuda_image(pixCount, 1);
        saveRef((complexFormat*)(refs[i]), (complexFormat*)amp, (uint32_t*)d_maskMap, thisrow, thiscol, row, column, normi);
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
    resize_cuda_image(pixCount, 1);
    expandRef((complexFormat*)(refs[i]), (complexFormat*)amp, (uint32_t*)d_maskMap, rows[i], cols[i], row, column);
    resize_cuda_image(row,column);
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
