#include "cudaConfig.hpp"
#include "fmt/core.h"
#include "imgio.hpp"
#include "material.hpp"
#include "cub_wrap.hpp"
#include "readConfig.hpp"
#include "spectImaging.hpp"
#include "cuPlotter.hpp"
#include <stdio.h>

int main(int argc, char* argv[]){
  if(argc < 2){
    fmt::println("Usage: simLineSpectrumImaging_run xxx.cfg");
    return 0;
  }
  ToyMaterial mat;
  readConfig cfg(argv[1]);
  //split reference and object support into two images.
  const int nlambda = 5;
  double lambdas[nlambda] = {1, 11./9, 11./7, 11./5, 11./3};
  double spectra[nlambda] = {0.1,0.2,0.3,0.3,0.1};
  spectImaging mwl;
  int objrow = 256, objcol=256;
  int row = 512, col = 512;
  Real* d_intensity = 0, *d_phase = 0;
  readComplexWaveFront(cfg.pupil.Intensity, cfg.phaseModulation? cfg.pupil.Phase:0, d_intensity, d_phase, objrow, objcol);
  myCuDMalloc(complexFormat, objectWave, row*col);
  myCuDMalloc(Real, pattern, row*col);
  myCuDMalloc(complexFormat, cpattern, row*col);
  init_cuda_image();
  resize_cuda_image(row, col);
  createWaveFront( d_intensity, d_phase, (complexFormat*)objectWave, objrow, objcol);
  mwl.init(row, col, nlambda, lambdas, spectra);
  int mrow, mcol;
  Real* refMask = readImage("mask.png", mrow, mcol);
  mwl.initRefs(refMask, mrow, mcol);
  plt.plotComplex(objectWave, PHASE, 0, 1, "objp", 0, 0, 0);
  plt.plotComplex(objectWave, MOD2, 0, 1, "obj", 0, 0, 0);
  init_fft(row, col);
  if(cfg.runSim) {
    mwl.assignRef(objectWave);
    mwl.generateMWLRefPattern(pattern);
    extendToComplex(pattern,cpattern);
    myFFT(cpattern, cpattern);
    plt.plotComplex(cpattern, MOD, 1, 30, "mergedac", 0, 0, 0);
    applyNorm(pattern, cfg.exposure);
    mwl.reconRefs(pattern);
    plt.plotFloat(pattern, MOD, 0, 1, "pattern", 1, 0, 1);
  }
}
