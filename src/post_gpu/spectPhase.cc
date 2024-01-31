#include "cudaConfig.hpp"
#include "spectPhase.hpp"
#include "cub_wrap.hpp"
#include <complex.h>
#include "material.hpp"
#include "cuPlotter.hpp"
#include <fstream>
void spectPhase::initRefSupport(complexFormat* refer, complexFormat* d_supportinput){  //mask file, full image size,
  myMalloc(complexFormat, cspectrum, nlambda);
  myDMalloc(complexFormat, support, row*column);
  myMemcpyD2H(support, d_supportinput, row*column*sizeof(complexFormat));
  for(int i = 0; i < nlambda; i++) cspectrum[i] = 0;
  pixCount = 0;
  for(int idx = 0; idx < row*column ; idx++){
    if(creal(support[idx]) > 0.1) pixCount++;
  }
  if(pixCount ==0) {
    fprintf(stderr, "ERROR: support is completely black\n");
    abort();
  }
  myDMalloc(uint32_t, maskMap, pixCount);
  int idx = 0, ic = 0;
  for(int x = 0; x < row ; x++){
    for(int y = 0; y < column ; y++){
      if(creal(support[idx]) > 0.1) {
        maskMap[ic] = idx;
        ic++;
      }
      idx++;
    }
  }
  myCuMalloc(uint32_t, d_supportMap, pixCount);
  myMemcpyH2D(d_supportMap, maskMap, pixCount*sizeof(uint32_t));
  myCuMalloc(complexFormat, d_support, pixCount);
  resize_cuda_image(pixCount, 1);
  init_fft(row, column);
  saveRef(d_support, d_supportinput, (uint32_t*)d_supportMap, row, column, row, column, 1);
  d_ref = refer;
  myFree(maskMap);
  ccmemMngr.returnCache(support);
}
void spectPhase::solvecSpectrum(Real* pattern, int niter){
  myCuDMalloc(Real, d_pattern, row*column);
  myCuDMalloc(Real, single_pattern, row*column);
  myCuDMalloc(complexFormat, img0, row*column);
  myCuDMalloc(complexFormat, d_amp, rows[nlambda-1]*cols[nlambda-1]);
  myCuDMalloc(complexFormat, d_obj, row*column);
  Real step_size = 0.5;
  myDMalloc(complexFormat, cspectrum_step, nlambda);
  for(int i = 0; i < niter; i++){
    myMemcpyD2D(d_pattern, pattern, row*column*sizeof(Real));
    for(int j = 0; j < nlambda; j++){
      int thisrow = rows[j];
      int thiscol = cols[j];
      resize_cuda_image(pixCount, 1);
      expandRef(d_support, d_ref, (uint32_t*)d_supportMap, row, column, row, column, cspectrum[j]);
      resize_cuda_image(thisrow, thiscol);
      pad(d_ref, d_amp, row, column);
      myFFTM(locplan[j], d_amp, d_amp);
      resize_cuda_image(row, column);
      cropinner(d_amp, img0, thisrow, thiscol, 1./sqrt(thisrow*thiscol));
      getMod2(single_pattern, img0);
      add(d_pattern, single_pattern, -spectra[j]);
    }
    int N = sqrt(row*column);
    for(int j = 0; j < nlambda; j++){
      //if(spectra[j] < 3e-2) continue;
      int thisrow = rows[j];
      int thiscol = cols[j];
      //int M = sqrt(thisrow*thiscol);
      clearCuMem(d_obj, row*column*sizeof(complexFormat));
      resize_cuda_image(pixCount, 1);

      expandRef(d_support, d_obj, (uint32_t*)d_supportMap, row, column, row, column);
      expandRef(d_support, d_ref, (uint32_t*)d_supportMap, row, column, row, column, cspectrum[j]);

      resize_cuda_image(thisrow, thiscol);
      pad(d_ref, d_amp, row, column);
      myFFTM(locplan[j], d_amp, d_amp);
      resize_cuda_image(row, column);
      cropinner(d_amp, img0, thisrow, thiscol, 1./N);

      resize_cuda_image(thisrow, thiscol);
      pad(d_obj, d_amp, row, column);
      myFFTM(locplan[j], d_amp, d_amp);
      resize_cuda_image(row, column);
      cropinner(d_amp, d_obj, thisrow, thiscol, 1./N);

      multiplyConj(img0, d_obj);
      multiply(img0, d_pattern);
      complexFormat step = findSum(img0);
      cspectrum_step[j] = step;
    }
    multiply(d_pattern, d_pattern, d_pattern);
    Real residual = findSum(d_pattern, row*column);
    Real k = 0;
    for(int j = 0; j < nlambda; j++){
      k += (sq(creal(cspectrum_step[j])) + sq(cimag(cspectrum_step[j])))*spectra[j];
    }
    printf("residual = %f, sum = %f, ", residual, k);
    k = residual/(k+1e-3);
    printf("k = %f\n", k);
    for(int j = 0; j < nlambda; j++){
      cspectrum[j] += k*step_size*cspectrum_step[j];
    }
  }
  std::ofstream file1("spectrum.txt", std::ios::out);
  for(int i = 0; i < nlambda; i++){
    file1<<i << " " << Real(rows[i]) / row<< " "<< creal(cspectrum[i]) << " " << cimag(cspectrum[i]) << " " << cabs(cspectrum[i]) << " " << carg(cspectrum[i])<<std::endl;
  }
  file1.close();
  myCuFree(d_pattern);
  myCuFree(single_pattern);
  myCuFree(img0);
  myCuFree(d_amp);
  myCuFree(d_obj);
}
void spectPhase::generateMWL(void* d_pattern, void* mat, Real thickness){
  BaseMaterial* matp = (BaseMaterial*) mat;
  myCuDMalloc(Real, single_pattern, row*column);
  myCuDMalloc(complexFormat, img0, row*column);
  myCuDMalloc(complexFormat, d_amp, rows[nlambda-1]*cols[nlambda-1]);
  clearCuMem(d_pattern, row*column*sizeof(Real));
  std::ofstream file1("spectrum_sim.txt", std::ios::out);
  for(int j = 0; j < nlambda; j++){
    int thisrow = rows[j];
    int thiscol = cols[j];
    Real rat = Real(thisrow)/row;
    complexFormat amp = cexp(-thickness/matp->getExtinctionLength(rat)+1.0i*(matp->getRefractiveIndex(rat)-1)*thickness*2*M_PI/rat);
    file1<<j << " " << rat << " " << creal(amp) << " " << cimag(amp) << " " << cabs(amp) << " " << carg(amp)<<std::endl;
    resize_cuda_image(pixCount, 1);
    expandRef(d_support, d_ref, (uint32_t*)d_supportMap, row, column, row, column, amp);
    resize_cuda_image(thisrow, thiscol);
    pad(d_ref, d_amp, row, column);
    myFFTM(locplan[j], d_amp, d_amp);
    resize_cuda_image(row, column);
    cropinner(d_amp, img0, thisrow, thiscol, 1./sqrt(thisrow*thiscol));
    getMod2(single_pattern, img0);
    add((Real*)d_pattern, single_pattern, spectra[j]);
  }
  Real m = findMax((Real*)d_pattern, row*column);
  printf("max = %f\n",m);
  applyNorm((Real*)d_pattern, 1./m);
  for(int i = 0; i < nlambda; i++) spectra[i] *= 1./m;
  file1.close();
  myCuFree(single_pattern);
  myCuFree(img0);
  myCuFree(d_amp);
}
