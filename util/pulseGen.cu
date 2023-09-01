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
#include "cdilmdb.h"
#include "cub_wrap.h"
using namespace std;

int main(int argc, char** argv){
  cudaFree(0); // to speed up the cuda malloc; https://forums.developer.nvidia.com/t/cudamalloc-slow/40238
  if(argc==1) { printf("Tell me which one is the mnist data folder\n"); }
  int handle;
  bool training = 1;
  int ntraining = 1000;
  int testingstart = ntraining;
  monoChromo mwl;
  CDI cdi(argv[1]);
  int datamerge[] = {2,2,2,2,3,3,3,4,4,4,4};
  int datarefine[] ={2,3,4,1,2,3,1,1,2,3,4};
  const int nconfig = 11;
  int ntesting = nconfig;
  cuMnist *mnist_dat[nconfig];
  int objrow;
  int objcol;
  Real* d_input;
  Real* intensity;
  if(cdi.runSim){
    if(cdi.domnist) {
      initLMDB(&handle, training?"traindb":"testdb");
      //setCompress(&handle);
      objrow = 256;
      objcol = 256;
      for(int iconfig  = 0; iconfig < nconfig; iconfig++)
        mnist_dat[iconfig] = new cuMnist(cdi.mnistData, datamerge[iconfig], datarefine[iconfig], objrow, objcol);
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
  double* lambdas, *spectra;
  mwl.jump = cdi.spectrumSamplingStep;
  mwl.skip = mwl.jump/2;
  Real monoLambda = cdi.lambda;
#if 0
  int lambdarange = 4;
  int nlambda = objrow*(lambdarange-1)/2;
  lambdas = (double*)ccmemMngr.borrowCache(sizeof(double)*nlambda);
  spectra = (double*)ccmemMngr.borrowCache(sizeof(double)*nlambda);
  for(int i = 0; i < nlambda; i++){
    lambdas[i] = 1 + 2.*i/objrow;
    spectra[i] = exp(-pow(i*2./nlambda-1,2))/nlambda; //gaussian, -1,1 with sigma=1
  }
  mwl.init(objrow, objcol, nlambda, lambdas, spectra);
#elif 0
  const int nlambda = 5;
  double lambdas[nlambda] = {1, 11./9, 11./7, 11./5, 11./3};
  double spectra[nlambda] = {0.1,0.2,0.3,0.3,0.1};
  mwl.init(objrow, objcol, nlambda, lambdas, spectra);
#elif 1
  Real startlambda = 500;
  Real endlambda = 1000;
  int nlambda;
  if(cdi.solveSpectrum) {
    Real minlambda = startlambda/monoLambda;
    Real maxlambda = endlambda/monoLambda;
    mwl.init(objrow, objcol, minlambda, maxlambda);
    getNormSpectrum(cdi.spectrum,cdi.ccd_response,startlambda,endlambda,nlambda,lambdas,spectra); //this may change startlambda
    //mwl.init(objrow, objcol, 1, 2);
  }else{
    getNormSpectrum(cdi.spectrum,cdi.ccd_response,startlambda,endlambda,nlambda,lambdas,spectra); //this may change startlambda
    printf("lambda range = (%f, %f), ratio=%f, first bin: %f\n", startlambda, endlambda*startlambda, endlambda, startlambda*(1 + mwl.skip*2./objrow));
    mwl.init(objrow, objcol, lambdas, spectra, nlambda);
  }
#endif
  int sz = mwl.row*mwl.column*sizeof(Real);
  Real *d_patternSum = (Real*)memMngr.borrowCache(sz);
  Real *realcache = (Real*)memMngr.borrowCache(sz);
  Real *single = (Real*)ccmemMngr.borrowCache(sz);
  Real *merged = (Real*)ccmemMngr.borrowCache(sz);
  complexFormat *d_CpatternSum = (complexFormat*)memMngr.borrowCache(sz*2);
  complexFormat *d_solved = (complexFormat*)memMngr.borrowCache(sz*2);
  init_cuda_image(65536,1);
  resize_cuda_image(mwl.row, mwl.column);
  plt.init(mwl.row, mwl.column);
  curandStateMRG32k3a *devstates = (curandStateMRG32k3a *)memMngr.borrowCache(mwl.column * mwl.row * sizeof(curandStateMRG32k3a));
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  initRand(devstates,seed);
  mwl.devstates = 0;//devstates;
  Real maxmerged = 0;
  //mwl.writeSpectra("spectra.txt");
  for(int j = 0; j < (cdi.domnist? (training? ntraining:ntesting):1); j++){
    if(cdi.runSim && cdi.domnist) {
      mnist_dat[j%nconfig]->setIndex((training?0:testingstart)+j);
      mnist_dat[j%nconfig]->cuRead(d_input);
      init_cuda_image(objrow/cdi.oversampling, objcol/cdi.oversampling);
      plt.init(objrow/cdi.oversampling, objcol/cdi.oversampling);
      //plt.plotFloat(d_input, MOD, 0, 1, ("input"+to_string(j)).c_str(), 0);
      init_cuda_image(objrow, objcol);
      plt.init(objrow, objcol);
    }
    if(cdi.runSim){
      mwl.generateMWL(d_input, d_patternSum, d_solved, cdi.oversampling);
      if(maxmerged==0) maxmerged = findMax(d_patternSum);
      if(cdi.simCCDbit) ccdRecord(d_patternSum, d_patternSum, cdi.noiseLevel_pupil, devstates, cdi.exposure/maxmerged);
      else applyNorm( d_patternSum, cdi.exposure/maxmerged);
      plt.saveFloat(d_patternSum, "floatimage");
      //plt.plotFloat(d_patternSum, MOD, 0, 1, ("merged"+to_string(j)).c_str(), 0);
      //plt.plotFloat(d_patternSum, MOD, 0, 1, ("mergedlog"+to_string(j)).c_str(), 1);
      plt.plotComplex(d_solved, MOD, 0, cdi.exposure/maxmerged, ("singlelog"+to_string(j)).c_str(), 1);
      cudaMemcpy(merged, d_patternSum, sz, cudaMemcpyDeviceToHost);
      getMod( realcache, d_solved);
      applyNorm( realcache, 1./findMax(realcache));
      cudaMemcpy(single, realcache, sz, cudaMemcpyDeviceToHost);
      if(cdi.domnist) {
        int key = j+1000;
        void* ptrs[] = {merged, single};
        size_t sizes[] = {objrow*objcol*sizeof(float),objrow*objcol*sizeof(float)};
        fillLMDB(handle, &key, 2, ptrs, sizes);
      }
    }else{
      intensity = readImage(cdi.common.Pattern, objrow, objcol);
      cudaMemcpy(d_patternSum, intensity, objrow*objcol*sizeof(Real), cudaMemcpyHostToDevice);
      ccmemMngr.returnCache(intensity);
      if(cdi.solveSpectrum){
        int tmprow, tmpcol;
        intensity = readImage(cdi.pupil.Pattern, tmprow, tmpcol);
        cudaMemcpy(d_solved, intensity, tmprow*tmpcol*sizeof(Real), cudaMemcpyHostToDevice);
        if(tmprow > objrow) crop((Real*)d_solved, realcache, tmprow, tmpcol);
        else pad((Real*)d_solved, realcache, tmprow,tmpcol);
        extendToComplex(realcache, d_solved);
        ccmemMngr.returnCache(intensity);
      }
    }
    if(cdi.doIteration){
      extendToComplex(d_patternSum, d_CpatternSum);
      plt.plotFloat(d_patternSum, MOD, 0, 1, ("mergedlog"+to_string(j)).c_str(), 1, 0, 1);
      plt.plotFloat(d_patternSum, MOD, 0, 1, ("merged"+to_string(j)).c_str(), 0);
      if(!cdi.solveSpectrum) {
        mwl.solveMWL(d_CpatternSum, d_solved, cdi.noiseLevel, 1, cdi.nIter, 1, 0);
        //applyNorm( d_CpatternSum, 20);
        //applyNorm( d_solved, 20);
        getMod(d_patternSum, d_solved);
        plt.saveFloat(d_patternSum, "pattern");
      }
      plt.plotComplex(d_solved, MOD, 0, 1, ("solved"+to_string(j)).c_str(), 0);
      plt.plotComplex(d_solved, MOD, 0, 1, ("solvedlog"+to_string(j)).c_str(), 1, 0, 1);
      if(cdi.solveSpectrum) {
        mwl.solveMWL(d_CpatternSum, d_solved, 0, 0, 1, 0, 1);
        mwl.writeSpectra("spectra.txt");
        lambdas = (double*)ccmemMngr.borrowCache(mwl.nlambda*sizeof(double));
        spectra = (double*)ccmemMngr.borrowCache(mwl.nlambda*sizeof(double));
        memcpy(spectra, mwl.spectra, mwl.nlambda*sizeof(double));
        for(int i = 0; i < mwl.nlambda ; i++) {
          lambdas[i] = monoLambda*mwl.rows[i]/mwl.row;
        }
        getRealSpectrum(cdi.ccd_response, mwl.nlambda, lambdas, spectra);
        ofstream fout("spectrum_solved.txt", ios::out);
        for(int i = 0; i < mwl.nlambda ; i++) {
          //fout << lambdas[i] << " " << (spectra[i]>0?spectra[i]:0) << endl;
          fout << lambdas[i] << " " << spectra[i] << endl;
        }
        break;
      }
      //run Phase retrievial;
      /*
         cdi.row = objrow;
         cdi.column = objcol;
         cdi.init();
         applyNorm(d_patternSum, 1./cdi.exposure);
         cdi.setPattern(d_patternSum);
         init_cuda_image(objrow, objcol, 65535, 1./cdi.exposure);
         cdi.phaseRetrieve();
       */

      for(int i = 0; i < 0; i++){
        getMod2(cdi.patternData, cdi.patternWave);
        applyNorm(cdi.patternData, cdi.exposure);
        extendToComplex(cdi.patternData, d_solved);
        cudaConvertFO(d_solved);
        mwl.solveMWL(d_CpatternSum, d_solved, 0, 20); // starting point
        plt.plotComplex(d_solved, MOD, 0, 1, ("solved"+to_string(i)).c_str(), 0);
        plt.plotComplex(d_solved, MOD, 0, 1, ("solvedlog"+to_string(i)).c_str(), 1);
        getMod(d_patternSum, d_solved);
        applyNorm(d_patternSum, 1./cdi.exposure);
        cdi.setPattern(d_patternSum);
        init_cuda_image(65535, 1./cdi.exposure);
        resize_cuda_image(objrow, objcol);
        cdi.phaseRetrieve();
      }
      //if(cdi.runSim){
      //  mwl.resetSpectra();
      //  mwl.solveMWL(d_CpatternSum, d_solved, 0, 2000, 0, 1);
      //  mwl.writeSpectra("spectra_new.txt");
      //}

      myCufftExec(*plan, d_solved, d_CpatternSum, CUFFT_FORWARD);
      plt.plotComplex(d_CpatternSum, MOD, 1, 2./mwl.row, ("autocsolved"+to_string(j)).c_str(), 1);
    }

  }
  if(cdi.domnist) saveLMDB(handle);

  return 0;
}

