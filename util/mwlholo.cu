#include <iostream>
#include <chrono>
#include <fstream>
#include <vector>
#include "cudaConfig.h"
#include "cuPlotter.h"
#include "mnistData.h"
#include "common.h"
#include "monoChromo.h"
#include "cdi.h"

int main(int argc, char** argv){
  if(argc==1) { printf("Tell me which one is the mnist data folder\n"); }
  monoChromo mwl;
  CDI cdi(argv[1]);
  cuMnist *mnist_dat = 0;
  int objrow;
  int objcol;
  Real* d_input;
  Real* intensity;
  if(cdi.runSim){
    if(cdi.domnist) {
      objrow = 128;
      objcol = 128;
      mnist_dat = new cuMnist(cdi.mnistData,1, 3, objrow, objcol);
      d_input = (Real*) memMngr.borrowCache(objrow*objcol*sizeof(Real));
      objrow *= cdi.oversampling;
      objcol *= cdi.oversampling;
    }
    else {
      intensity = readImage(cdi.common.Intensity, objrow, objcol);
      d_input = (Real*) memMngr.borrowCache(objrow*objcol*sizeof(Real));
      cudaMemcpy(d_input, intensity, objrow*objcol*sizeof(Real), cudaMemcpyHostToDevice);
      ccmemMngr.returnCache(intensity);
      objrow *= cdi.oversampling;
      objcol *= cdi.oversampling;
    }
  }else{
    intensity = readImage(cdi.common.Pattern, objrow, objcol);
    ccmemMngr.returnCache(intensity);
  }
#if 1
  int lambdarange = 4;
  int nlambda = objrow*(lambdarange-1)/2;
  double *lambdas;
  double *spectra;
  lambdas = (double*)ccmemMngr.borrowCache(sizeof(double)*nlambda);
  spectra = (double*)ccmemMngr.borrowCache(sizeof(double)*nlambda);
  for(int i = 0; i < nlambda; i++){
    lambdas[i] = 1 + 2.*i/objrow;
    spectra[i] = exp(-pow(i*2./nlambda-1,2))/nlambda; //gaussian, -1,1 with sigma=1
  }
#elif 0
  const int nlambda = 5;
  Real lambdas[nlambda] = {1, 11./9, 11./7, 11./5, 11./3};
  Real spectra[nlambda] = {0.1,0.2,0.3,0.3,0.1};
#else
  int nlambda;
  Real* lambdas, *spectra;
  Real startlambda = 700;
  Real endlambda = 1000;
  Real rate = endlambda/startlambda;
  getNormSpectrum(argv[2],argv[3],startlambda,nlambda,lambdas,spectra); //this may change startlambda
  printf("lambda range = (%f, %f), ratio=%f", startlambda, endlambda, rate);
  rate = 1.15;
#endif
  std::ofstream spectrafile;
  spectrafile.open("spectra_raw.txt",ios::out);
  for(int i = 0; i < nlambda; i++){
    spectrafile<<lambdas[i]<<" "<<spectra[i]<<endl;
  }
  spectrafile.close();
  mwl.init(objrow, objcol, nlambda, lambdas, spectra);
  //mwl.init(objrow, objcol, lambdas, spectra, rate);
  int sz = mwl.row*mwl.column*sizeof(Real);
  Real *d_patternSum = (Real*)memMngr.borrowCache(sz);
  complexFormat *single = (complexFormat*)memMngr.borrowCache(sz*2);
  complexFormat *d_CpatternSum = (complexFormat*)memMngr.borrowCache(sz*2);
  init_cuda_image(mwl.row, mwl.column);
  plt.init(mwl.row, mwl.column);
  curandStateMRG32k3a *devstates = (curandStateMRG32k3a *)memMngr.borrowCache(mwl.column * mwl.row * sizeof(curandStateMRG32k3a));
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  initRand(devstates, seed);
  mwl.writeSpectra("spectra.txt");
  for(int j = 0; j < 1; j++){
    if(cdi.runSim && cdi.domnist) {
        mnist_dat->cuRead(d_input);
        init_cuda_image(objrow/cdi.oversampling, objcol/cdi.oversampling);
        plt.init(objrow/cdi.oversampling, objcol/cdi.oversampling);
        plt.plotFloat(d_input, MOD, 0, 1, ("input"+to_string(j)).c_str(), 0);
        init_cuda_image(objrow, objcol);
        plt.init(objrow, objcol);
    }
    if(cdi.runSim){
      mwl.generateMWL(d_input, d_patternSum, single, cdi.oversampling);
      applyPoissonNoise_WO(d_patternSum, cdi.noiseLevel, devstates);
      plt.plotFloat(d_patternSum, MOD, 0, cdi.exposure, ("merged"+to_string(j)).c_str(), 0);
      plt.plotFloat(d_patternSum, MOD, 0, cdi.exposure, ("mergedlog"+to_string(j)).c_str(), 1);
      plt.plotComplex(single, MOD, 0, cdi.exposure, ("single"+to_string(j)).c_str(), 1);
      intensity = readImage(("merged"+to_string(j)+".png").c_str(), objrow, objcol);
    }else{
      intensity = readImage(cdi.common.Pattern, objrow, objcol);
    }
    extendToComplex(d_patternSum, d_CpatternSum);
    init_fft(objrow,objcol);
    myCufftExec(*plan, d_CpatternSum, d_CpatternSum, CUFFT_FORWARD);
    applyNorm(d_CpatternSum, 1./sqrt(objrow*objcol));
    plt.plotComplex(d_CpatternSum, MOD, 1, 2./mwl.row, ("autocsolved"+to_string(j)).c_str(), 1);
  }
  mwl.writeSpectra("spectra_new.txt");

  return 0;
}

