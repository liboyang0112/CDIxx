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
      mnist_dat = new cuMnist(cdi.mnistData, 3, objrow, objcol);
      cudaMalloc((void**)&d_input, objrow*objcol*sizeof(Real));
      objrow *= cdi.oversampling;
      objcol *= cdi.oversampling;
    }
    else {
      intensity = readImage(cdi.common.Intensity, objrow, objcol);
      cudaMalloc((void**)&d_input, objrow*objcol*sizeof(Real));
      cudaMemcpy(d_input, intensity, objrow*objcol*sizeof(Real), cudaMemcpyHostToDevice);
      ccmemMngr.returnCache(intensity);
      objrow *= cdi.oversampling;
      objcol *= cdi.oversampling;
    }
  }else{
    intensity = readImage(cdi.common.Pattern, objrow, objcol);
    ccmemMngr.returnCache(intensity);
  }
#if 0
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
  double lambdas[nlambda] = {1, 11./9, 11./7, 11./5, 11./3};
  double spectra[nlambda] = {0.1,0.2,0.3,0.3,0.1};
#else
  int nlambda;
  double* lambdas, *spectra;
  Real startlambda = 400;
  Real endlambda = 990;
  getNormSpectrum(argv[2],argv[3],startlambda,endlambda,nlambda,lambdas,spectra); //this may change startlambda
  printf("lambda range = (%f, %f), ratio=%f\n", startlambda, endlambda*startlambda, endlambda);
#endif
  //mwl.init(objrow, objcol, nlambda, lambdas, spectra);
  Real tot = mwl.init(objrow, objcol, lambdas, spectra, nlambda);
  std::ofstream spectrafile;
  spectrafile.open("spectra_raw.txt",ios::out);
  for(int i = 0; i < nlambda; i++){
    //spectrafile<<lambdas[i]<<" "<<spectra[i]/tot<<endl;
    spectrafile<<lambdas[i]<<" "<<spectra[i]<<endl;
  }
  spectrafile.close();
  int sz = mwl.row*mwl.column*sizeof(Real);
  Real *d_patternSum = (Real*)memMngr.borrowCache(sz);
  complexFormat *single = (complexFormat*)memMngr.borrowCache(sz*2);
  complexFormat *d_CpatternSum = (complexFormat*)memMngr.borrowCache(sz*2);
  complexFormat *d_solved = (complexFormat*)memMngr.borrowCache(sz*2);
  init_cuda_image(mwl.row, mwl.column,65536,1);
  plt.init(mwl.row, mwl.column);
  curandStateMRG32k3a *devstates = (curandStateMRG32k3a *)memMngr.borrowCache(mwl.column * mwl.row * sizeof(curandStateMRG32k3a));
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  cudaF(initRand,devstates,seed);
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
      cudaF(applyNorm,d_patternSum,cdi.exposure);
      plt.saveFloat(d_patternSum, "merged");
      if(cdi.simCCDbit) cudaF(applyPoissonNoise_WO,d_patternSum, cdi.noiseLevel, devstates,1);
      plt.plotFloat(d_patternSum, MOD, 0, 1, ("merged"+to_string(j)).c_str(), 0);
      plt.plotComplex(single, MOD, 0, cdi.exposure, ("single"+to_string(j)).c_str(), 1);
      if(cdi.simCCDbit){
        intensity = readImage(("merged"+to_string(j)+".png").c_str(), objrow, objcol);
        cudaMemcpy(d_patternSum, intensity, objrow*objcol*sizeof(Real), cudaMemcpyHostToDevice);
        ccmemMngr.returnCache(intensity);
      }
    }else{
      intensity = readImage(cdi.common.Pattern, objrow, objcol);
      cudaMemcpy(d_patternSum, intensity, objrow*objcol*sizeof(Real), cudaMemcpyHostToDevice);
      ccmemMngr.returnCache(intensity);
    }
    cudaF(extendToComplex,d_patternSum, d_CpatternSum);
    plt.plotFloat(d_patternSum, MOD, 0, 1, ("mergedlog"+to_string(j)).c_str(), 1);
    mwl.solveMWL(d_CpatternSum, d_solved, 1, cdi.nIter, 1, 0);
    plt.plotComplex(d_solved, MOD, 0, 1, ("solved"+to_string(j)).c_str(), 0);
    plt.plotComplex(d_solved, MOD, 0, 1, ("solvedlog"+to_string(j)).c_str(), 1);
    cudaF(getMod,d_patternSum, d_solved);
    plt.saveFloat(d_patternSum, "pattern");
    //run Phase retrievial;
    cdi.row = objrow;
    cdi.column = objcol;
    cdi.init();
    cudaF(applyNorm,d_patternSum, 1./cdi.exposure);
    cdi.setPattern(d_patternSum);
    init_cuda_image(objrow, objcol, 65535, 1./cdi.exposure);
    cdi.phaseRetrieve();

    for(int i = 0; i < 0; i++){
      cudaF(getMod2,cdi.patternData, cdi.patternWave);
      cudaF(applyNorm,cdi.patternData, cdi.exposure);
      cudaF(extendToComplex,cdi.patternData, d_solved);
      cudaF(cudaConvertFO,d_solved);
      mwl.solveMWL(d_CpatternSum, d_solved, 0, 20); // starting point
      plt.plotComplex(d_solved, MOD, 0, 1, ("solved"+to_string(i)).c_str(), 0);
      plt.plotComplex(d_solved, MOD, 0, 1, ("solvedlog"+to_string(i)).c_str(), 1);
      cudaF(getMod,d_patternSum, d_solved);
      cudaF(applyNorm,d_patternSum, 1./cdi.exposure);
      cdi.setPattern(d_patternSum);
      init_cuda_image(objrow, objcol, 65535, 1./cdi.exposure);
      cdi.phaseRetrieve();
    }
    //if(cdi.runSim){
    //  mwl.resetSpectra();
    //  mwl.solveMWL(d_CpatternSum, single, 0, 2000, 0, 1);
    //  mwl.writeSpectra("spectra_new.txt");
    //}

    myCufftExec(*plan, d_solved, d_CpatternSum, CUFFT_FORWARD);
    plt.plotComplex(d_CpatternSum, MOD, 1, 2./mwl.row, ("autocsolved"+to_string(j)).c_str(), 1);

  }

  return 0;
}

