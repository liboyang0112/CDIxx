#include <iostream>
#include <chrono>
#include <fstream>
#include <vector>
#include "cudaConfig.hpp"
#include "cuPlotter.hpp"
#include <string.h>
#include "mnistData.hpp"
#include "imgio.hpp"
#include "monoChromo.hpp"
#include "cdi.hpp"
#include "cdilmdb.hpp"
#include "cub_wrap.hpp"
#include <gsl/gsl_spline.h>
#include <math.h>
using namespace std;

void getNormSpectrum(const char* fspectrum, const char* ccd_response, Real &startLambda, Real &endLambda, int &nlambda, double *& outlambda, double *& outspectrum){
  std::vector<double> spectrum_lambda;
  std::vector<double> spectrum;
  std::vector<double> ccd_lambda;
  std::vector<double> ccd_rate;
  std::ifstream file_spectrum, file_ccd_response;
  std::ofstream file_out("spectTccd.txt");
  double threshold = 1e-3;
  file_spectrum.open(fspectrum);
  file_ccd_response.open(ccd_response);
  double lambda, val, maxval;
  maxval = 0;
  while(file_spectrum){
    file_spectrum >> lambda >> val;
    spectrum_lambda.push_back(lambda);
    spectrum.push_back(val);
    if(val > maxval) maxval = val;
  }
  while(file_ccd_response>>lambda>>val){
    ccd_lambda.push_back(lambda);
    ccd_rate.push_back(val);
  }
  endLambda = std::min(Real(spectrum_lambda.back()),endLambda);
  bool isShortest = 1;
  nlambda = 0;
  gsl_interp_accel *acc = gsl_interp_accel_alloc ();
  gsl_spline *spline = gsl_spline_alloc (gsl_interp_cspline, ccd_lambda.size());
  gsl_spline_init (spline, &ccd_lambda[0], &ccd_rate[0], ccd_lambda.size());
  double ccdmax = ccd_lambda.back();
  double ccd_rate_max = ccd_rate.back();
  for(unsigned int i = 0; i < spectrum.size(); i++){
    lambda = spectrum_lambda[i];
    if(lambda<startLambda) continue;
    if(lambda>=endLambda) break;
    if(isShortest && spectrum[i] < threshold*maxval) continue;
    if(isShortest) startLambda = lambda;
    isShortest = 0;
    double ccd_rate_i = ccd_rate[0];
    if(lambda >= ccdmax) ccd_rate_i = ccd_rate_max;
    else if(lambda > ccd_lambda[0]) ccd_rate_i = gsl_spline_eval (spline, lambda, acc);
    spectrum_lambda[nlambda] = lambda/startLambda;
    //if(lambda >= 940) ccd_rate_i*=2;
    //if(lambda < 800) ccd_rate_i *= 0.9;
    spectrum[nlambda] = spectrum[i]/maxval*ccd_rate_i;
    nlambda++;
  }
  endLambda /= startLambda;
  outlambda = (double*) ccmemMngr.borrowCache(sizeof(double)*nlambda);
  outspectrum = (double*) ccmemMngr.borrowCache(sizeof(double)*nlambda);
  for(int i = 0; i < nlambda; i++){
    outlambda[i] = spectrum_lambda[i];
    outspectrum[i] = spectrum[i];
    file_out << spectrum_lambda[i]*startLambda<<" "<<spectrum[i]<<std::endl;
  }
  file_out.close();
  gsl_spline_free (spline);
  gsl_interp_accel_free (acc);
}
void getRealSpectrum(const char* ccd_response, int nlambda, double* lambdas, double* spectrum){
  std::vector<double> ccd_lambda;
  std::vector<double> ccd_rate;
  std::ifstream file_ccd_response;
  file_ccd_response.open(ccd_response);
  double lambda, val;
  while(file_ccd_response>>lambda>>val){
    ccd_lambda.push_back(lambda);
    ccd_rate.push_back(val);
  }
  gsl_interp_accel *acc = gsl_interp_accel_alloc ();
  gsl_spline *spline = gsl_spline_alloc (gsl_interp_cspline, ccd_lambda.size());
  gsl_spline_init (spline, &ccd_lambda[0], &ccd_rate[0], ccd_lambda.size());
  if(0)
    for(int i = 0; i < nlambda; i++){
      if(lambdas[i] < ccd_lambda[0]){
        printf("lambda smaller than ccd curve min %f < %f\n", lambdas[i], ccd_lambda[0]);
        spectrum[i] /= ccd_rate[0];
      }else if(lambdas[i] > ccd_lambda.back()){
        printf("lambda larger than ccd curve max %f > %f\n", lambdas[i], ccd_lambda.back());
        spectrum[i] /= ccd_rate.back();
      }else
        spectrum[i] /= gsl_spline_eval (spline, lambdas[i], acc);
    }
  gsl_spline_free (spline);
  gsl_interp_accel_free (acc);
}

int main(int argc, char** argv){
  if(argc < 2) {
    printf("Usage: mono_run xxx.cfg");
    exit(0);
  }
  int handle;
  char training = 1; //0: no DB generation, 1: trainging DB, 2: testing DB
  int ntraining = 1000;
  int ntesting = 100;
  int testingstart = ntraining+100;
  CDI cdi(argv[1]);
  int datamerge[] = {1,2,2,2,2,3,3,3};
  int datarefine[] ={3,2,3,4,1,2,3,1};
  //int datamerge[] = {1};
  //int datarefine[] ={3};
  const int nconfig = sizeof(datamerge)/sizeof(int);
  cuMnist *mnist_dat[nconfig];
  int objrow;
  int objcol;
  Real* d_obj, *d_input;
  Real* intensity;
  init_cuda_image(65535,1);
  if(cdi.runSim){
    if(cdi.domnist) {
      if(training) initLMDB(&handle, training==1?"traindb":"testdb");
      //setCompress(&handle);
      objrow = 128;
      objcol = 128;
      for(int iconfig  = 0; iconfig < nconfig; iconfig++)
        mnist_dat[iconfig] = new cuMnist(cdi.mnistData, datamerge[iconfig], datarefine[iconfig], objrow, objcol);
      d_input = (Real*) memMngr.borrowCache(objrow*objcol*sizeof(Real));
      objrow *= cdi.oversampling;
      objcol *= cdi.oversampling;
      d_obj = (Real*)memMngr.borrowCache(objrow*objcol*sizeof(Real));
    }
    else {
      intensity = readImage(cdi.common.Intensity, objrow, objcol);
      d_input = (Real*) memMngr.borrowCache(objrow*objcol*sizeof(Real));
      myMemcpyH2D(d_input, intensity, objrow*objcol*sizeof(Real));
      ccmemMngr.returnCache(intensity);
      objrow *= cdi.oversampling;
      objcol *= cdi.oversampling;
      d_obj = (Real*)memMngr.borrowCache(objrow*objcol*sizeof(Real));
      resize_cuda_image(objrow, objcol);
      pad(d_input, d_obj, objrow/cdi.oversampling, objcol/cdi.oversampling);
      plt.init(objrow, objcol);
      plt.plotFloat(d_obj, MOD, 0, 1, "object", 0, 0, 0);
      memMngr.returnCache(d_input);
    }
  }else{
    training = 0;
    intensity = readImage(cdi.common.Pattern, objrow, objcol);
  }
  resize_cuda_image(objrow, objcol);
  double* lambdas, *spectra;
  monoChromo_constRatio mwl;
  //monoChromo mwl;
  mwl.jump = cdi.spectrumSamplingStep;
  mwl.skip = 0;
  Real monoLambda = cdi.lambda;
  if(string(cdi.spectrum) == "gaussian"){
    int lambdarange = 6;
    int nlambda = objrow*(lambdarange-1)/2;
    lambdas = (double*)ccmemMngr.borrowCache(sizeof(double)*nlambda);
    spectra = (double*)ccmemMngr.borrowCache(sizeof(double)*nlambda);
    for(int i = 0; i < nlambda; i++){
      lambdas[i] = 1 + 2.*i/objrow;
      spectra[i] = exp(-pow(2*(i*2./nlambda-1),2))/nlambda; //gaussian, -1,1 with sigma=1
    }
    mwl.init(objrow, objcol, lambdas, spectra, nlambda);
  }
  //else if(string(cdi.spectrum) == "comb"){
  //  int maxh = 37;
  //  int maxl = 6;
  //  int minh = ((maxh / maxl)>>1<<1) - 1 ;
  //  int nlambda = (maxh - minh)/2;
  //  myMalloc(double, lambdas, nlambda);
  //  myMalloc(double, spectra, nlambda);
  //  float spectsum = 0;
  //  for(int i = 0; i < nlambda; i++){
  //    lambdas[i] = float(maxh)/(maxh-2*i);
  //      //spectra[i] = exp(-pow(2*((lambdas[i]-1)*2./(maxl-1)-1),2))*5; //gaussian, -1,1 with sigma=1
  //    if(i < 4) spectra[i] = 0.03+i*0.03;
  //    else if(i >=nlambda-3) spectra[i] = 0.1+(nlambda-i)*0.3;
  //    else if(i == 5) spectra[i] = 0.6;
  //    else if(i == 4) spectra[i] = 0.3;
  //    else spectra[i] = 0.3+0.08*i;
  //    spectsum += spectra[i];
  //  }
  //  for(int i = 0; i < nlambda; i++){
  //     spectra[i]/=spectsum;
  //  }
  //  mwl.init(objrow, objcol, nlambda, lambdas, spectra);
  //}
  else{
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
      mwl.writeSpectra("spectra.txt", startlambda);
    }
  }
  init_fft(objrow, objcol);
  int sz = mwl.row*mwl.column*sizeof(Real);
  Real *d_patternSum = (Real*)memMngr.borrowCache(sz);
  Real *realcache = (Real*)memMngr.borrowCache(sz);
  Real *single = (Real*)ccmemMngr.borrowCache(sz);
  Real *merged = (Real*)ccmemMngr.borrowCache(sz);
  complexFormat *d_CpatternSum = (complexFormat*)memMngr.borrowCache(sz*2);
  complexFormat *d_solved = (complexFormat*)memMngr.borrowCache(sz*2);
  resize_cuda_image(mwl.row, mwl.column);
  plt.init(mwl.row, mwl.column);
  void *devstates = newRand(mwl.column * mwl.row);
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  initRand(devstates,seed);
  mwl.devstates = 0;//devstates;
  Real maxmerged = 0;
  for(int j = 0; j < (training? (training==1? ntraining:ntesting):1); j++){
    if(cdi.runSim){
      if(cdi.domnist) {
        mnist_dat[j%nconfig]->setIndex((training==2?testingstart:0)+j);
        mnist_dat[j%nconfig]->cuRead(d_input);
        resize_cuda_image(mwl.row, mwl.column);
        plt.init(mwl.row, mwl.column);
        pad(d_input, d_obj, objrow/cdi.oversampling, objcol/cdi.oversampling);
        if(j<nconfig){
          plt.plotFloat(d_obj, MOD, 0, 1, ("input"+to_string(j)).c_str(), 0);
        }
      }
      mwl.generateMWL(d_obj, d_patternSum, d_solved);
      if(maxmerged==0){
       maxmerged = findMax(d_patternSum);
       Real rat = findSum(d_patternSum)/maxmerged;
       printf("ratio = %f\n", rat);
      }
      if(cdi.simCCDbit) ccdRecord(d_patternSum, d_patternSum, cdi.noiseLevel_pupil, devstates, cdi.exposure/maxmerged, 6553500);
      else applyNorm( d_patternSum, cdi.exposure/maxmerged);
      plt.saveFloat(d_patternSum, "broad_pattern");
      myMemcpyD2H(merged, d_patternSum, sz);
      getMod( realcache, d_solved);
      plt.saveFloat(realcache, "hene_pattern");
      applyNorm( realcache, 1./findMax(realcache));
      myMemcpyD2H(single, realcache, sz);
      if(training) {
        int key = j;
        void* ptrs[] = {merged, single};
        size_t sizes[] = {objrow*objcol*sizeof(float),objrow*objcol*sizeof(float)};
        fillLMDB(handle, &key, 2, ptrs, sizes);
      }
      if(j<nconfig){
        plt.plotFloat(d_patternSum, MOD, 0, 1, ("merged"+to_string(j)).c_str(), 0);
        plt.plotFloat(d_patternSum, MOD, 0, 1, ("mergedlog"+to_string(j)).c_str(), 1, 0, 1);
        plt.plotComplex(d_solved, MOD, 0, 1, ("singlelog"+to_string(j)).c_str(), 1, 0, 1);
      }
      clearCuMem(d_solved,sz*2);
    }else{
      myMemcpyH2D(d_patternSum, intensity, objrow*objcol*sizeof(Real));
      ccmemMngr.returnCache(intensity);
      if(cdi.solveSpectrum){
        mwl.resetSpectra();
        int tmprow, tmpcol;
        intensity = readImage(cdi.pupil.Pattern, tmprow, tmpcol);
        myMemcpyH2D(d_solved, intensity, tmprow*tmpcol*sizeof(Real));
        if(tmprow > objrow) crop((Real*)d_solved, realcache, tmprow, tmpcol);
        else pad((Real*)d_solved, realcache, tmprow,tmpcol);
        extendToComplex(realcache, d_solved);
        ccmemMngr.returnCache(intensity);
      }
      plt.plotFloat(d_patternSum, MOD, 0, 1, ("mergedlog"+to_string(j)).c_str(), 1, 0, 1);
      plt.plotFloat(d_patternSum, MOD, 0, 1, ("merged"+to_string(j)).c_str(), 0);
    }
    if(cdi.doIteration){
      extendToComplex(d_patternSum, d_CpatternSum);
      if(!cdi.solveSpectrum) {
        mwl.solveMWL(d_CpatternSum, d_solved, cdi.noiseLevel, 0, cdi.nIter, 1, 0);
        getMod(d_patternSum, d_solved);
        plt.saveFloat(d_patternSum, "pattern");
        plt.plotComplex(d_solved, MOD, 0, 1, ("solved"+to_string(j)).c_str(), 0);
        plt.plotComplex(d_solved, MOD, 0, 1, ("solvedlog"+to_string(j)).c_str(), 1, 0, 1);
        myFFT(d_solved, d_CpatternSum);
        plt.plotComplex(d_CpatternSum, MOD, 1, 2./mwl.row, ("autocsolved"+to_string(j)).c_str(), 1);
      }
      if(cdi.solveSpectrum) {
        mwl.resetSpectra();
        mwl.solveMWL(d_CpatternSum, d_solved, 0, 1, 1, 0, 1);
        mwl.writeSpectra("spectrum_solved.txt", monoLambda);
        spectra = (double*)ccmemMngr.borrowCache(mwl.nlambda*sizeof(double));
        memcpy(spectra, mwl.spectra, mwl.nlambda*sizeof(double));
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
        getMod2(cdi.patternData, (complexFormat*)cdi.patternWave);
        applyNorm(cdi.patternData, cdi.exposure);
        extendToComplex(cdi.patternData, d_solved);
        cudaConvertFO(d_solved);
        //mwl.solveMWL(d_CpatternSum, d_solved, 0, 20); // starting point
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

    }

  }
  if(training) saveLMDB(handle);

  return 0;
}

