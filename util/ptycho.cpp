#include <complex.h>
#include <fmt/base.h>
#include <stdio.h>
#include <random>
#include <chrono>
#include "cudaDefs_h.cu"
#include "fmt/core.h"
#include "imgio.hpp"
#include "cudaConfig.hpp"
#include "experimentConfig.hpp"
#include "cuPlotter.hpp"
#include "cub_wrap.hpp"
#include "misc.hpp"
#include "ptycho.hpp"
#include "tvFilter.hpp"
#include "beamDecomposition.hpp"


using namespace std;

//#define Bits 16

class ptycho : public experimentConfig{
  public:
    int row_O = 512;  //in ptychography this is different from row (the size of probe).
    int column_O = 512;
    int sz = 0;
    int doPhaseModulationPupil = 0;
    int scanx = 0;
    int scany = 0;
    Real *shiftx = 0;
    Real *shifty = 0;
    Real *shifts = 0;
    Real *step_shift = 0;
    Real *d_shift = 0;
    Real *d_shifts = 0;
    Real **patterns; //patterns[i*scany+j] points to the address on device to store pattern;
    complexFormat *esw;
    complexFormat* objectWave_t = 0;
    complexFormat* pupilpatternWave_t = 0;
    void *devstates = 0;

    ptycho(const char* configfile):experimentConfig(configfile){}
    Real computeErrorSim();
    void allocateMem(){
      if(devstates) return;
      devstates = newRand(column_O * row_O);
      fmt::println("allocating memory");
      scanx = (row_O-row)/stepSize+1;
      scany = (column_O-column)/(stepSize*sqrtf(3)/2)+1;
      size_t scansz = scanx*scany*sizeof(Real);
      fmt::println("scanning {} x {} steps", scanx, scany);
      objectWave = (complexFormat*)memMngr.borrowCache(row_O*column_O*sizeof(Real)*2);
      objectWave_t = (complexFormat*)memMngr.borrowCache(row_O*column_O*sizeof(Real)*2);
      pupilpatternWave = (complexFormat*)memMngr.borrowCache(sz*2);
      pupilpatternWave_t = (complexFormat*)memMngr.borrowCache(sz*2);
      esw = (complexFormat*) memMngr.borrowCache(sz*2);
      patterns = (Real**)ccmemMngr.borrowCleanCache(scanx*scany*sizeof(Real*));
      fmt::println("initializing cuda image");
      resize_cuda_image(row_O,column_O);
      init_cuda_image(rcolor, 1./exposure);
      unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
      initRand(devstates,seed);
      shifts = (Real*)ccmemMngr.borrowCleanCache(scansz*2);
      d_shifts = (Real*)memMngr.borrowCleanCache(scansz*2);
      shiftx = shifts;
      shifty = shifts+scanx*scany;
      d_shift = (Real*)memMngr.borrowCache(scansz*2);
      if(runSim && positionUncertainty > 1e-4){
        initPosition();
      }
    }
    void readPupilAndObject(){
      Real* d_object_intensity = 0;
      Real* d_object_phase = 0;
      readComplexWaveFront(common.Intensity, common.Phase, d_object_intensity, d_object_phase, row_O, column_O);
      Real* pupil_intensity = readImage(pupil.Intensity, row, column);
      sz = row*column*sizeof(Real);
      int row_tmp=row*oversampling;
      int column_tmp=column*oversampling;
      allocateMem();
      createWaveFront(d_object_intensity, d_object_phase, objectWave_t, 1);
      memMngr.returnCache(d_object_intensity);
      memMngr.returnCache(d_object_phase);
      verbose(2,
          plt.init(row_O,column_O, outputDir);
          plt.plotComplexColor(objectWave_t, 0, 1, "inputObject");
          //plt.plotPhase(objectWave_t, PHASERAD, 0, 1, "inputPhase");
          )
        Real* d_intensity = (Real*) memMngr.borrowCache(sz); //use the memory allocated;
      myMemcpyH2D(d_intensity, pupil_intensity, sz);
      ccmemMngr.returnCache(pupil_intensity);
      Real* d_phase = 0;
      if(doPhaseModulationPupil){
        d_phase = (Real*) memMngr.borrowCache(sz);
        int tmp;
        Real* pupil_phase = readImage(pupil.Phase, tmp,tmp);
        myMemcpyH2D(d_phase, pupil_phase, sz);
        ccmemMngr.returnCache(pupil_phase);
      }
      pupilobjectWave = (complexFormat*)memMngr.borrowCache(row_tmp*column_tmp*sizeof(complexFormat));
      resize_cuda_image(row_tmp,column_tmp);
      createWaveFront(d_intensity, d_phase, pupilobjectWave, oversampling);
      memMngr.returnCache(d_intensity);
      if(d_phase) memMngr.returnCache(d_phase);
      plt.init(row_tmp,column_tmp, outputDir);
      plt.plotComplexColor(pupilobjectWave, 0, 1, "pupilWave", 0);
      init_fft(row_tmp,column_tmp);
      //opticalPropagate(pupilobjectWave, lambda, dpupil, beamspotsize*oversampling, row_tmp*column_tmp); //granularity changes
      angularSpectrumPropagate(pupilobjectWave, pupilobjectWave, beamspotsize*oversampling/lambda, dpupil/lambda, row_tmp*column_tmp); //granularity is the same
      plt.plotComplex(pupilobjectWave, MOD2, 0, 1, "pupilPattern", 0);
      resize_cuda_image(row,column);
      init_fft(row,column);
      crop(pupilobjectWave, pupilpatternWave_t, row_tmp, column_tmp);
      plt.init(row,column, outputDir);
      plt.plotComplexColor(pupilpatternWave_t, 0, 1, "probeWave", 0);
      calculateParameters();
      //multiplyFresnelPhase(pupilpatternWave_t, d);
    }
    void initPosition(){
      unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
      std::default_random_engine generator(seed);
      std::normal_distribution<double> distribution(0.0, 1.);
      for(int i = 0 ; i < scanx*scany; i++){
        shiftx[i]+= distribution(generator)*positionUncertainty;
        shifty[i]+= distribution(generator)*positionUncertainty;
        if(verbose >=3 ) fmt::println("shifts ({}, {}): ({:f}, {:f})", i/scany, i%scany, shiftx[i],shifty[i]);
      }
    }
    void resetPosition(){
      for(int i = 0 ; i < scanx*scany; i++){
        shiftx[i] = shifty[i] = 0;
      }
    }
    void createPattern(){
      int idx = 0;
      if(useBS) {
        createBeamStop();
        plt.plotFloat(beamstop, MOD, 1, 1,"beamstop", 0);
      }
      complexFormat* window = (complexFormat*)memMngr.borrowCache(sz*2);
      for(int i = 0; i < scanx; i++){
        for(int j = 0; j < scany; j++){
          Real posx = i*stepSize+(j%2 == 1? 0:(Real(stepSize)/2)) + shiftx[idx];
          Real posy = j*stepSize*sqrtf(3)/2 + shifty[idx];
          int shiftxpix = posx-round(posx);
          int shiftypix = posy-round(posy);
          getWindow(objectWave_t, +round(posx), round(posy), row_O, column_O, window);
          if(fabs(shiftxpix)>1e-3||fabs(shiftypix)>1e-3){
            shiftWave(window, shiftxpix, shiftypix);
          }
          multiply(esw, pupilpatternWave_t, window);
          verbose(5, plt.plotComplex(esw, MOD2, 0, 1, ("ptycho_esw"+to_string(i)+"_"+to_string(j)).c_str()));
          myFFT(esw,esw);
          if(!patterns[idx]) patterns[idx] = (Real*)memMngr.borrowCache(sz);
          getMod2(patterns[idx], esw, 1./(row*column));
          if(useBS) applySupport(patterns[idx], beamstop);
          if(simCCDbit) {
            //applyPoissonNoise_WO(patterns[idx], noiseLevel, devstates, 1./exposure);
            ccdRecord(patterns[idx], patterns[idx], noiseLevel, devstates, exposure);
          }else{
            applyNorm(patterns[idx], exposure);
          }
          verbose(2, plt.plotFloat(patterns[idx], MOD, 1, 1, (common.Pattern+to_string(i)+"_"+to_string(j)).c_str()));
          verbose(4, plt.plotFloat(patterns[idx], MOD, 1, 1, (common.Pattern+to_string(i)+"_"+to_string(j)+"log").c_str(),1));
          applyNorm(patterns[idx], 1./exposure);
          idx++;
        }
      }
      memMngr.returnCache(window);
    }
    void initObject(){
      resize_cuda_image(row_O,column_O);
      random(objectWave, devstates);
      resize_cuda_image(row,column);
      pupilFunc(pupilpatternWave);
    }
    void updatePosition(int scanidx, complexFormat* obj, complexFormat* probe, Real* pattern, complexFormat* Fn){
      size_t siz = memMngr.getSize(obj);
      Real norm = 1./(row*column);
      complexFormat *cachex = (complexFormat*)memMngr.borrowCache(siz);
      complexFormat *cachey = (complexFormat*)memMngr.borrowCache(siz);
      Real *cache = (Real*)memMngr.borrowCache(siz>>1);
      myFFT(obj, cachex);
      cudaConvertFO(cachex, cachex, sqrt(norm));
      multiplyy(cachex, cachey);
      multiplyx(cachex, cachex);
      cudaConvertFO(cachex, cachex, norm);
      cudaConvertFO(cachey, cachey, norm);
      myIFFT(cachex, cachex);
      myIFFT(cachey, cachey);
      multiply(cachex, cachex, probe);
      multiply(cachey, cachey, probe);
      myFFT(cachex, cachex);
      myFFT(cachey, cachey);
      calcPartial(cache, cachex, Fn, pattern, beamstop);
      findSum(cache, 0, d_shift+scanidx);
      calcPartial(cache, cachey, Fn, pattern, beamstop);
      findSum(cache, 0, d_shift+scanidx+scanx*scany);
      memMngr.returnCache(cachex);
      memMngr.returnCache(cachey);
      memMngr.returnCache(cache);
    }
    void iterate(){
      resetPosition();
      resize_cuda_image(row,column);
      Real objMax;
      Real probeMax;
      complexFormat *Fn = (complexFormat*)memMngr.borrowCache(sz*2);
      complexFormat *probeStep = (complexFormat*)memMngr.borrowCache(sz*2);
      complexFormat *objCache = (complexFormat*)memMngr.borrowCache(sz*2);
      Real *maxCache = (Real*)memMngr.borrowCache(max(row_O*column_O/4, row*column)*sizeof(Real));
      myCuDMalloc(complexFormat, cropObj, row_O*column_O/4);
      myCuDMalloc(Real, tmp, row*column);
      Real norm = 1./sqrt(row*column);
      int update_probe_iter = 4;
      int positionUpdateIter = 50;
      int objFFT;
      createPlan(&objFFT, row_O, column_O); 
      int pupildiameter = 150;
      myCuDMalloc(complexFormat, zernikeCrop, pupildiameter*pupildiameter);
      void* zernike = zernike_init(pupildiameter, pupildiameter, 20, 0); //40 is already high enough for modelling complex beams
      myCuDMalloc(Real, pupilSupport, pupildiameter*pupildiameter);
      resize_cuda_image(pupildiameter, pupildiameter);
      createCircleMask(pupilSupport, Real(pupildiameter+1)/2, Real(pupildiameter+1)/2, Real(pupildiameter)/2);
      resize_cuda_image(row,column);
      bool mPIE = 0;
      Real probeStepSize = 4;
      Real zernikeStepSize = 6;
      myCuDMalloc(Real, d_norm, 2);
      findSum(tmp, row*column, d_norm);
      myDMalloc(Real, h_norm, 2);
      int vidhandle_pupil = 0;
      int vidhandle_probe = 0;
      int vidhandle_O = 0;

      cuPlotter plt_O;
      plt_O.init(row_O,column_O, outputDir);
      if(saveVideoEveryIter){
        vidhandle_O = plt_O.initVideo("recon_object.mp4",24);
        plt_O.showVid = -1;
        vidhandle_pupil = plt.initVideo("recon_pupil.mp4",24);
        vidhandle_probe = plt.initVideo("recon_probe.mp4",24);
        plt.showVid = -1;
      }
      for(int iter = 0; iter < nIter; iter++){
        int idx = 0;
        getMod2(maxCache, pupilpatternWave);
        findMax(maxCache, row*column ,d_norm);
        if(iter >= update_probe_iter) {
          clearCuMem(probeStep, sz*2);
          resize_cuda_image(row_O>>2,column_O>>2);
          crop(objectWave, cropObj, row_O, column_O);
          getMod2(maxCache, cropObj);
          findMax(maxCache, row_O*column_O/16, d_norm+1);
          myMemcpyD2H(h_norm, d_norm, 2*sizeof(Real));
          objMax = h_norm[1];
          resize_cuda_image(row_O,column_O);
          applyThreshold(objectWave, objectWave, objMax);
          resize_cuda_image(row,column);
        }else{
          myMemcpyD2H(h_norm, d_norm, sizeof(Real));
        }
        probeMax = h_norm[0];
        if(iter >= update_probe_iter){
          resize_cuda_image(row_O,column_O);
          Real sf = pow(probeMax/objMax, 0.25);
          applyNorm(objectWave, sf);
          resize_cuda_image(row,column);
          applyNorm(pupilpatternWave, 1./sf);
          if(mPIE){
            applyNorm(probeStep, 1./sf);
          }
          objMax = probeMax = sqrt(objMax*probeMax);
        }
        //complexFormat* coeff = NULL, *projection = NULL;
        bool doUpdatePosition = iter % 20 == 0 && iter >= positionUpdateIter;
        for(int i = 0; i < scanx; i++){
          for(int j = 0; j < scany; j++){
            Real posx = i*stepSize+(j%2 == 1? 0:(Real(stepSize)/2)) + shiftx[idx];
            Real posy = j*stepSize*sqrtf(3)/2 + shifty[idx];
            int shiftxpix = posx-round(posx);
            int shiftypix = posy-round(posy);
            getWindow(objectWave, posx, posy, row_O, column_O, objCache);
            bool shiftpix = fabs(shiftxpix)>1e-3||fabs(shiftypix)>1e-3;
            if(shiftpix){
              shiftWave(objCache, shiftxpix, shiftypix);
            }
            multiply(esw, pupilpatternWave, objCache);
            myFFT(esw,Fn);
            if(doUpdatePosition) {
              applyNorm(Fn, norm);
              updatePosition(idx, objCache, pupilpatternWave, patterns[idx], Fn);
              applyModAccurate(Fn, patterns[idx],beamstop, 1);
            }else applyModAccurate(Fn, patterns[idx],beamstop, norm);
            myIFFT(Fn,Fn);
            add(esw, Fn, -norm);
            if(iter < update_probe_iter) updateObject(objCache, pupilpatternWave, esw, probeMax);
            else {
              updateObjectAndProbeStep(objCache, pupilpatternWave, probeStep, esw,probeMax, objMax);
            //updateObjectAndProbe(objCache, pupilpatternWave, esw,probeMax, objMax);
              if(mPIE){
                applyNorm(probeStep, 0.6);
                add(pupilpatternWave, probeStep);
              }
            }
            if(shiftpix){
              shiftWave(objCache, -shiftxpix, -shiftypix);
            }
            updateWindow(objectWave, posx, posy, row_O, column_O, objCache);
            idx++;
          }
        }

        resize_cuda_image(row_O, column_O);
        //FISTA((complexFormat*) objectWave, (complexFormat*) objectWave, 3e-4, 1, 0);
        resize_cuda_image(row, column);
        if(!mPIE && iter >= update_probe_iter) add(pupilpatternWave, probeStep,(iter > zernikeIter?probeStepSize:zernikeStepSize)/(scanx*scany));
        if(doUpdatePosition){
          resize_cuda_image(scanx*scany*2, 1);
          addProbability(d_shifts, d_shift, 0.8, positionUncertainty);
          resize_cuda_image(row, column);
        }
        if(iter >= update_probe_iter){
          if(saveVideoEveryIter && iter%saveVideoEveryIter == 0){
            plt.toVideo = vidhandle_probe;
            plt.plotComplexColor(pupilpatternWave, 0, 1, ("recon_probe"+to_string(iter)).c_str(), 0, isFlip);
            plt.toVideo = -1;
          }
          const int N = scanx * scany;
          findSum(d_shifts, N, d_norm);
          findSum(d_shifts+N, N, d_norm+1);
          myMemcpyD2H(h_norm, d_norm, 2*sizeof(Real));
          resize_cuda_image(scanx*scany, 1);
          h_norm[0] /= N;
          h_norm[1] /= N;
          linearConst(d_shifts, d_shifts, 1, -h_norm[0]);
          linearConst(d_shifts+N, d_shifts+N, 1, -h_norm[1]);
          resize_cuda_image(scanx*scany*2, 1);
          resize_cuda_image(row_O, column_O);
          shiftWave(objFFT,objectWave, -h_norm[0], -h_norm[1]);
          if(saveVideoEveryIter && iter%saveVideoEveryIter == 0){
            plt_O.toVideo = vidhandle_O;
            plt_O.plotComplexColor(objectWave, 0, 1, ("recon_object"+to_string(iter)).c_str(), 0, isFlip);
            plt_O.toVideo = -1;
          }
          resize_cuda_image(row, column);
          angularSpectrumPropagate(pupilpatternWave, pupilpatternWave, beamspotsize*oversampling/lambda, -dpupil/lambda, row*column); //granularity is the same
          if(saveVideoEveryIter && iter%saveVideoEveryIter == 0){
            plt.toVideo = vidhandle_pupil;
            plt.plotComplexColor(pupilpatternWave, 0, 1, ("recon_pupil"+to_string(iter)).c_str(), 0, isFlip);
            plt.toVideo = -1;
          }
          if(iter == zernikeIter - 1) {
            plt.plotComplexColor(pupilpatternWave, 0, 1, "recon_pupil_b4_proj");
          }
          resize_cuda_image(pupildiameter, pupildiameter);
          crop(pupilpatternWave, zernikeCrop, row, column);
          if(iter < zernikeIter){
            zernike_compute(zernike, zernikeCrop, Real(pupildiameter-1)/2, Real(pupildiameter-1)/2, Real(pupildiameter)/2);
            zernike_reconstruct(zernike, zernikeCrop, Real(pupildiameter)/2);
          }else{
            applyMask(zernikeCrop, pupilSupport);
          }
          //zernike_compute(zernike, zernikeCrop, pupildiameter>>1, pupildiameter>>1, 30);
          //zernike_reconstruct(zernike, zernikeCrop, 30);
          resize_cuda_image(row, column);
          pad(zernikeCrop, pupilpatternWave, pupildiameter, pupildiameter);
          if(iter == zernikeIter - 1) {
            plt.plotComplexColor(pupilpatternWave, 0, 1, "recon_pupil_proj");
          }
          angularSpectrumPropagate(pupilpatternWave, pupilpatternWave, beamspotsize*oversampling/lambda, dpupil/lambda, row*column); //granularity is the same
        }
        if(doUpdatePosition || iter >= update_probe_iter){
          myMemcpyD2H(shifts, d_shifts, scanx*scany*sizeof(Real)*2);
        }
        if(iter == positionUpdateIter){
          resize_cuda_image(row_O,column_O);
          plt_O.plotComplex(objectWave, MOD2, 0, 1, "ptycho_b4position", 0);
          plt_O.plotComplex(objectWave, PHASE, 0, 1, "ptycho_b4positionphase", 0);
          resize_cuda_image(row,column);
        }
      }
      angularSpectrumPropagate(pupilpatternWave, pupilpatternWave, beamspotsize*oversampling/lambda, -dpupil/lambda, row*column); //granularity is the same
      plt.plotComplexColor(pupilpatternWave, 0, 1, "recon_pupil");
      if(verbose>=3){
        for(int i = 0 ; i < scanx*scany; i++){
          fmt::println("recon shifts ({}, {}): ({:f}, {:f})", i/scany, i%scany, shiftx[i],shifty[i]);
        }
      }
      myCuFree(d_norm);
      myFree(h_norm);
      memMngr.returnCache(Fn);
      memMngr.returnCache(objCache);
      plt.plotComplexColor(pupilpatternWave, 0, 0.12, "ptycho_probe_afterIter", 0);
      Real crop_ratio = 0.65;
      int rowc = row_O*crop_ratio;
      int colc = column_O*crop_ratio;
      resize_cuda_image(rowc, colc);
      plt.init(rowc, colc, outputDir);
      complexFormat* cropped = (complexFormat*)memMngr.borrowCache(rowc*colc*sizeof(complexFormat));
      myCuDMalloc(Real, angle, rowc*colc);
      crop(objectWave, cropped, row_O, column_O);
      getArg(angle, cropped);
      applyNorm(angle, 1./(2*M_PI));
      plt.plotComplex(cropped, MOD2, 0, 0.7, "ptycho_afterIter");
      //plt.plotPhase(cropped, PHASERAD, 0, 1, "ptycho_afterIterphase");
      //phaseUnwrapping(angle, angle, rowc, colc);
      plt.plotFloat(angle, REAL, 0, 1, "ptycho_afterIterphase");
      plt.plotComplexColor(cropped, 0, 1, "ptycho_afterIterwave");
    }
    void readPattern(){
      Real* pattern = readImage((outputDir + string(common.Pattern)+"0_0.png").c_str(), row, column);
      plt.init(row,column, outputDir);
      init_fft(row,column);
      sz = row*column*sizeof(Real);
      allocateMem();
      resize_cuda_image(row,column);
      if(useBS) {
        createBeamStop();
      }
      int idx = 0;
      for(int i = 0; i < scanx; i++){
        for(int j = 0; j < scany; j++){
          if(idx!=0) pattern = readImage((string(outputDir) + common.Pattern+to_string(i)+"_"+to_string(j)+".png").c_str(), row, column);
          if(!patterns[idx]) patterns[idx] = (Real*)memMngr.borrowCache(sz);
          myMemcpyH2D(patterns[idx], pattern, sz);
          ccmemMngr.returnCache(pattern);
          cudaConvertFO(patterns[idx]);
          applyNorm(patterns[idx], 1./exposure);
          verbose(3, plt.plotFloat(patterns[idx], MOD, 1, exposure, ("input"+string(common.Pattern)+to_string(i)+"_"+to_string(j)).c_str()));
          idx++;
        }
      }
      fmt::println("Created pattern data");
      calculateParameters();
    }
    void calculateParameters(){
      resolution = lambda*dpupil/beamspotsize/oversampling;
      if(runSim){
        d = resolution*pixelsize*row/lambda;
        fmt::println("Desired d = {}", d);
      }
      experimentConfig::calculateParameters();
    }
};
Real ptycho::computeErrorSim(){
  myCuDMalloc(complexFormat, convoluted, row*column);
  myCuDMalloc(complexFormat, cache, row*column);
  convolute(convoluted, pupilpatternWave_t, pupilpatternWave, cache);
  return 0;
}
int main(int argc, char** argv )
{
  ptycho setups(argv[1]);
  if(argc < 2){
    fmt::println("please feed the object intensity and phase image");
  }
  if(setups.runSim){
    setups.readPupilAndObject();
    setups.createPattern();
  }else{
    setups.readPattern();
  }
  fmt::println("Imaging distance = {:4.2f}cm", setups.d*1e-4);
  fmt::println("fresnel factor = {:f}", setups.fresnelFactor);
  fmt::println("Resolution = {:4.2f}nm", setups.resolution*1e3);

  fmt::println("pupil Imaging distance = {:4.2f}cm", setups.dpupil*1e-4);
  fmt::println("pupil fresnel factor = {:f}", setups.fresnelFactorpupil);
  fmt::println("pupil enhancement = {:f}", setups.enhancementpupil);
  setups.initObject();
  setups.iterate();

  return 0;
}

