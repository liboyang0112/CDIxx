#include "spectImaging.hpp"
#include "cudaConfig.hpp"
#include "cudaDefs_h.cu"
#include "tvFilter.hpp"
#include "memManager.hpp"
#include "misc.hpp"
#include <cstdint>
#include <time.h>
#include "cub_wrap.hpp"
#include <math.h>
#include "cuPlotter.hpp"
#include <string>
using namespace std;
void spectImaging::assignRef(void* wavefront, int i){ //wavefront has to be of the size row, column
  assignRef_d((complexFormat*)wavefront, d_maskMap, (complexFormat*)(refs[i]), pixCount);
}
void spectImaging::assignRef(void* wavefront){
  assignRef(wavefront, 0);
  for(int i = 1; i < nlambda; i++)
    myMemcpyD2D(refs[i], refs[0], pixCount*sizeof(complexFormat));
}

void spectImaging::saveHSI(const char* name, Real* support){
  //int vid = plt.initVideo((name + string("_mod.mp4")).c_str(), 8);
  //plt.toVideo = vid;
  resize_cuda_image(imrow, imcol);
  plt.init(imrow,imcol);
  for (int i = 0; i < nlambda; i++) {
    if(support) multiply((complexFormat*)padding_cache, (complexFormat*)spectImages[i], support);
    plt.plotComplexColor(padding_cache, 0, 1, (name + std::to_string(i)).c_str());  
  }
  //plt.saveVideo(vid);
}

void spectImaging::initHSI(int _row, int _col){  //mask file, this is not to init refs, this is for init image coding masks
  imrow = _row;
  imcol = _col;
  myMalloc(void*, spectImages, nlambda);
  for (int i = 0; i < nlambda; i++) {
    myCuMalloc(complexFormat, spectImages[i], _row*_col);
  }
}

void spectImaging::initRefs(Real* refMask_dev, int mrow, int mcol, int shiftx, int shifty){  //mask file, this is to init refs
  myDMalloc(Real, refMask, mrow*mcol);
  myMemcpyH2D(refMask, refMask_dev, mrow*mcol*sizeof(Real));
  uint32_t* maskMap = createMaskMap(refMask, pixCount, row, column, mrow, mcol, shiftx, shifty);
  d_maskMap = (uint32_t*)memMngr.borrowCache(pixCount*sizeof(uint32_t));
  myMemcpyH2D(d_maskMap, maskMap, pixCount*sizeof(uint32_t));
  myFree(maskMap);
  myFree(refMask);
  refs = (void**)ccmemMngr.borrowCache(nlambda*sizeof(void*));

  complexFormat val = 1;
  for(int i = 0; i < nlambda; i++){
    refs[i] = memMngr.borrowCache(pixCount*sizeof(complexFormat));
    setValue((complexFormat*)refs[i], val);
  }
  printf("mask has %d pixels\n", pixCount);
}
void spectImaging::pointRefs(int npoints, int *xs, int *ys){  //mask file, this is to init refs
  pixCount = npoints;
  myDMalloc(uint32_t, maskMap, pixCount);
  for (int i = 0; i < npoints; i++) {
    maskMap[i] = xs[i]*column + ys[i];
  }
  printf("%d, %d\n", maskMap[0],maskMap[1]);
  myCuMalloc(uint32_t, d_maskMap, pixCount);
  myMemcpyH2D(d_maskMap, maskMap, pixCount*sizeof(uint32_t));
  refs = (void**)ccmemMngr.borrowCache(nlambda*sizeof(void*));
  for(int i = 0; i < nlambda; i++){
    refs[i] = memMngr.borrowCache(pixCount*sizeof(complexFormat));
  }
  myDMalloc(complexFormat, d_ref, pixCount);
  for (int i = 0; i < pixCount; i++) {
    d_ref[i] = 100;
  }
  for (int i = 0; i < nlambda; i++) {
    myMemcpyH2D(refs[i], d_ref, pixCount*sizeof(complexFormat));
  }
  printf("mask has %d pixels\n", pixCount);
}
void spectImaging::generateMWLPattern(void* d_patternSum, bool debug, Real* mask){
  complexFormat *amp = (complexFormat*)memMngr.borrowCleanCache(rows[nlambda-1]*cols[nlambda-1]*sizeof(complexFormat));
  complexFormat *camp = (complexFormat*)memMngr.borrowCache(row*column*sizeof(complexFormat));
  Real *d_pattern= (Real*)memMngr.borrowCleanCache(row*column*sizeof(Real));
  complexFormat* masked = 0;
  if(mask) myCuMalloc(complexFormat, masked, imrow*imcol);
  for(int i = 0; i < nlambda; i++){
    int thisrow = rows[i];
    int thiscol = cols[i];
    if(mask) {
      resize_cuda_image(imrow,imcol);
      multiply(masked, (complexFormat*)spectImages[i], mask);
      resize_cuda_image(thisrow, thiscol);
      pad(masked, amp, imrow, imcol);
    }else{
      resize_cuda_image(thisrow, thiscol);
      pad((complexFormat*)spectImages[i], amp, imrow, imcol);
    }
    resize_cuda_image(pixCount, 1);
    expandRef((complexFormat*)(refs[i]), (complexFormat*)amp, d_maskMap, thisrow, thiscol, row, column);
    resize_cuda_image(row, column);
    if(debug && i == 0){
      resize_cuda_image(thisrow, thiscol);
      plt.init(thisrow, thiscol);
      plt.plotComplexColor(amp, 0, 5, "object_mono", 1, 0);
      myFFTM(locplan[i], amp, amp);
      plt.plotComplex(amp, MOD2, 1, spectra[i]/(thiscol*thisrow), "pattern_mono", 1, 0, 1);
    }else
      myFFTM(locplan[i], amp, amp);
    cropinner(amp,camp,thisrow,thiscol, sqrt(spectra[i]/(thiscol*thisrow)));
    cudaConvertFO(camp);
    if(i==0) {
      getMod2((Real*)d_patternSum, camp);
      if(debug){
        extendToComplex((Real*)d_patternSum, camp);
        cudaConvertFO(camp);
        myIFFTM(locplan[i],camp, camp);
        plt.plotComplexColor(camp, 1, 1./sqrt(pixCount), "hologram_mono", 1, 0);
      }
    }else{
      getMod2(d_pattern, camp);
      add((Real*)d_patternSum, (Real*)d_pattern, 1);
    }
    clearCuMem(amp,  thisrow*thiscol*sizeof(complexFormat));
  }
  resize_cuda_image(row,column);
  plt.init(row,column);
  myCuFree(amp);
  myCuFree(camp);
  myCuFree(d_pattern);
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
    expandRef((complexFormat*)(refs[i]), (complexFormat*)amp, d_maskMap, thisrow, thiscol, row, column);
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
void spectImaging::clearHSI(){
  for (int i = 0; i < nlambda; i++) {
    clearCuMem(spectImages[i], imrow*imcol*sizeof(complexFormat));
  }
}

void createACmask_diamond(Real* acmask, int row, int imrow, int column, int imcol){
  diamond acdmasks;
  acdmasks.startx = (row - (imrow*2))/2;
  acdmasks.starty = (column - (imcol*2))/2;
  acdmasks.width = imrow*2;
  acdmasks.height = imcol*2;
  rect acrmasks;
  acrmasks.startx = 0;
  acrmasks.endx = row;
  acrmasks.starty =  (column >> 1) - 30;
  acrmasks.endy = (column >> 1) + 30;
  myCuDMalloc(Real, acrmask, row*column);
  resize_cuda_image(row, column);
  plt.init(row,column);
  createMaskBar(acmask, &acdmasks, 1);
  createMaskBar(acrmask, &acrmasks, 1);
  multiply(acmask, acmask, acrmask);
  myCuFree(acrmask);
}
void createACmask(Real* acmask, int row, int imrow, int column, int imcol){
  rect acdmasks;
  acdmasks.startx = row/2 - imrow;
  acdmasks.starty = column/2 - imcol;
  acdmasks.endx = row/2 + imrow;
  acdmasks.endy = column/2 + imcol;
  myCuDMalloc(rect, cuda_spt, 1);
  myMemcpyH2D(cuda_spt, &acdmasks, sizeof(rect));
  resize_cuda_image(row, column);
  createMaskBar(acmask, cuda_spt, 1);
}
void spectImaging::reconstructHSI(void* d_patternSum, Real* mask){
  Real stepsize = 0.3/pixCount;
  Real normf = 1./(row*column);
  //--create mask to block quadratic term--
  myCuDMalloc(Real, acmask, row*column);
  createACmask(acmask, row, imrow, column, imcol);
  plt.plotFloat(acmask, MOD, 1, 1, "acmask", 0, 0, 0);

  myCuDMalloc(Real, residual, row*column);
  myCuDMalloc(complexFormat, autocorr, row*column);
  myCuDMalloc(complexFormat, step, row*column);
  myCuDMalloc(complexFormat, Ritilder, row*column);

  init_fft(row, column);
  myDMalloc(complexFormat*, spectImages_prev, nlambda);
  myCuDMalloc(complexFormat, twist_cache, imrow*imcol);
  for (int i = 0 ; i < nlambda; i++) {
    myCuMalloc(complexFormat, spectImages_prev[i], imrow*imcol);
    clearCuMem(spectImages_prev[i], imrow*imcol*sizeof(complexFormat));
  }
  Real alpha = 1.92, beta = 3.96, norm = 0.15/(nlambda*2)/pixCount;
  bool runAP = 1;
  complexFormat** tmp;
  int niter = 2000;
  for(int iter = 0; iter < niter; iter ++){
    generateMWLPattern(residual, 0, mask);
    add(residual, (Real*)d_patternSum, residual,  -1.);
    if(iter %30==0) printf("iter = %d, residual = %f\n", iter, findRootSumSq(residual));
    extendToComplex(residual, autocorr);
    cudaConvertFO(autocorr);
    myIFFT(autocorr,autocorr);
    if(iter == niter-1){
    plt.plotComplexColor(autocorr, 1, 1./sqrt(imrow*imcol/2), "residual_autocorr", 1);
    }
    multiply(autocorr, autocorr, acmask);
    applyNorm(autocorr, 1./sqrt(row*column));
    if(!runAP){
      tmp = spectImages_prev;
      spectImages_prev = (complexFormat**)spectImages;
      spectImages = (void**)tmp;
    }
    myFFT(autocorr,autocorr);
    for (int i = 0; i < nlambda; i++) {
      applyAT(autocorr, step, rows[i], cols[i], locplan[i], 0b11);
      resize_cuda_image(pixCount, 1);
      clearCuMem(Ritilder, row*column*sizeof(complexFormat));
      expandRef((complexFormat*)refs[i], Ritilder, d_maskMap, row, column, row, column);
      resize_cuda_image(row, column);
      myFFT(Ritilder, Ritilder);
      multiplyRegular(Ritilder, step, Ritilder, 1);
      //multiply(Ritilder, step, Ritilder);
      applyNorm(Ritilder, normf*stepsize);
      myIFFT(Ritilder, Ritilder);
      if(iter == niter-1) plt.plotComplexColor(Ritilder, 0, 1, "step");
      resize_cuda_image(imrow, imcol);
      crop(Ritilder, step, row, column);
      applyMask(step, mask);
      if(runAP){
        //--alternating projection--
        add((complexFormat*)spectImages[i], step);
        FISTA((complexFormat*)spectImages[i], (complexFormat*)spectImages[i], 2e-4, 1, 0);
      }else{
      //--TwIST--
      //recon_imgs[i] = (1-alpha)*recon_imgs[i] + (alpha-beta)*recon_imgs_prev[i] +
      //beta*FISTA(recon_imgs_prev[i] + norm*step,1e-5,1,list_FISTA)
        add(twist_cache, spectImages_prev[i], step, norm);
        FISTA(twist_cache, twist_cache, 1e-4, 1, 0);
        normAdd((complexFormat*)spectImages[i],(complexFormat*)spectImages[i], twist_cache, 1-alpha, beta);
        add((complexFormat*)spectImages[i], spectImages_prev[i], alpha-beta);
      }
    }
    //if(iter == 100){
    //  runAP = 0;
    //  for (int i = 0 ; i < nlambda ; i++) {
    //    myMemcpyD2D(spectImages_prev[i], spectImages[i], imrow*imcol*sizeof(complexFormat));
    //  }
    //  //break;
    //}
  }
  resize_cuda_image(row, column);
  plt.init(row,column);
  plt.plotFloat(residual, MOD, 0, 1, "residual", 1, 0, 1);
  myCuFree(residual);
  myCuFree(autocorr);
  myCuFree(step);
  myCuFree(Ritilder);
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
        expandRef(it == 0? (complexFormat*)(refs[i]) : reftmp, (complexFormat*)amp, d_maskMap, thisrow, thiscol, row, column);
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
        saveRef(reftmp, (complexFormat*)amp, d_maskMap, thisrow, thiscol, row, column, normi);
      }
      //add(campm, residual, -stepsize*spectra[i]);  //previous intensity
      //add(campm, residual, -stepsize);  //previous intensity
      resize_cuda_image(row,column);
      add(campm, residual, -stepsize/spectra[i]);  //previous intensity
      clearCuMem(amp,  thisrow*thiscol*sizeof(complexFormat));
      resize_cuda_image(pixCount, 1);
      expandRef(reftmp, (complexFormat*)amp, d_maskMap, thisrow, thiscol, row, column);
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
      saveRef(reftmp, (complexFormat*)amp, d_maskMap, thisrow, thiscol, row, column, normi);
      resize_cuda_image(row,column);
      getMod2((Real*)camp, residual);
      resize_cuda_image(pixCount, 1);
      //printf("residual=%f\n", findSum((Real*)camp));
      for(int it = 0; it < 5; it++){
        clearCuMem(amp,  thisrow*thiscol*sizeof(complexFormat));
        expandRef((complexFormat*)(refs[i]), (complexFormat*)amp, d_maskMap, thisrow, thiscol, row, column);
        resize_cuda_image(row,column);
        myFFTM(locplan[i], amp, amp);
        Real normi = 1./(thiscol*thisrow);
        applyModAbsinner(amp,campm, thisrow,thiscol, 1./normi, devstates);
        myIFFTM( locplan[i], amp, amp);
        resize_cuda_image(pixCount, 1);
        saveRef((complexFormat*)(refs[i]), (complexFormat*)amp, d_maskMap, thisrow, thiscol, row, column, normi);
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
    expandRef((complexFormat*)(refs[i]), (complexFormat*)amp, d_maskMap, rows[i], cols[i], row, column);
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
