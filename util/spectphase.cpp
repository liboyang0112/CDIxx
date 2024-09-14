#include "cudaConfig.hpp"
#include "material.hpp"
#include "cub_wrap.hpp"
#include "readConfig.hpp"
#include "spectPhase.hpp"
#include "cuPlotter.hpp"
#include <stdio.h>
#include <math.h>
#include <time.h>

int main(int argc, char* argv[]){
  if(argc < 2){
    printf("Usage: spectphase_run xxx.cfg");
    exit(0);
  }
  init_cuda_image();
  ToyMaterial mat;
  readConfig cfg(argv[1]);
  //split reference and object support into two images.
  double* lambdas, *spectra;
  int row = 512, col = 512;
#if 1
  Real lambdarange = 2;
  int nlambda = row*(lambdarange-1)/2;
  lambdas = (double*)ccmemMngr.borrowCache(sizeof(double)*nlambda);
  spectra = (double*)ccmemMngr.borrowCache(sizeof(double)*nlambda);
  for(int i = 0; i < nlambda; i++){
    lambdas[i] = 1 + 2.*i/row;
    spectra[i] = 1; //gaussian, -1,1 with sigma=1
  }
#else
  const int nlambda = 5;
  double lambdas[nlambda] = {1, 11./9, 11./7, 11./5, 11./3};
  double spectra[nlambda] = {0.1,0.2,0.3,0.3,0.1};
#endif
  spectPhase mwl;
  mwl.jump = cfg.spectrumSamplingStep;
  mwl.skip = mwl.jump/2;
  Real* d_intensity = 0, *d_phase = 0;
  myCuDMalloc(complexFormat, refer, row*col);
  myCuDMalloc(complexFormat, d_support, row*col);
  myCuDMalloc(Real, d_pattern, row*col);
  myCuDMalloc(complexFormat, d_wave, row*col);
  readComplexWaveFront(cfg.pupil.Intensity, cfg.phaseModulation? cfg.pupil.Phase:0, d_intensity, d_phase, row, col);
  resize_cuda_image(row, col);
  createWaveFront( d_intensity, d_phase, (complexFormat*)refer, row, col);
  readComplexWaveFront(cfg.common.Intensity, cfg.phaseModulation? cfg.common.Phase:0, d_intensity, d_phase, row, col);
  resize_cuda_image(row, col);
  createWaveFront( d_intensity, d_phase, (complexFormat*)d_support, row, col);
  mwl.init(row, col, nlambda, lambdas, spectra);
  mwl.initRefSupport(refer, d_support);  //mask file, full image size,
  void* randstate = newRand(row*col);
  if(cfg.runSim){
    mwl.generateMWL(d_pattern, &mat, 100);
    initRand(randstate, time(NULL));
    ccdRecord(d_pattern, d_pattern, cfg.noiseLevel, randstate, cfg.exposure);
    plt.plotFloat(d_pattern, MOD, 1, 1, "mergedlog", 1, 0, 1);
    plt.plotFloat(d_pattern, MOD, 1, 1, "merged", 0, 0, 0);
    extendToComplex(d_pattern,d_wave);
    myFFT(d_wave,d_wave);
    plt.plotComplex(d_wave, MOD2, 1, 1./32, "autocorrelation_merged", 0, 0, 1);
    mwl.solvecSpectrum((Real*)d_pattern, 800);
  }
  return 0;
}
