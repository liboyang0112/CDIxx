#include <complex.h>
#include <complex>
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
#include "beamDecomposition.hpp"


using namespace std;

//#define Bits 16

class ptycho : public experimentConfig{
  public:
    int row_O = 512;  //in ptychography this is different from row (the size of probe).
    int column_O = 512;
    int sz = 0;
    int stepSize = 16;
    int doPhaseModulationPupil = 0;
    int scanx = 0;
    int scany = 0;
    Real *shiftx = 0;
    Real *shifty = 0;
    Real *step_shiftx = 0;
    Real *step_shifty = 0;
    Real *d_shiftx = 0;
    Real *d_shifty = 0;
    Real **patterns; //patterns[i*scany+j] points to the address on device to store pattern;
    complexFormat *esw;
    void *devstates = 0;

    ptycho(const char* configfile):experimentConfig(configfile){}
    void allocateMem(){
      if(devstates) return;
      devstates = newRand(column_O * row_O);
      fmt::println("allocating memory");
      scanx = (row_O-row)/stepSize+1;
      scany = (column_O-column)/stepSize+1;
      size_t scansz = scanx*scany*sizeof(Real);
      fmt::println("scanning {} x {} steps", scanx, scany);
      objectWave = (complexFormat*)memMngr.borrowCache(row_O*column_O*sizeof(Real)*2);
      pupilpatternWave = (complexFormat*)memMngr.borrowCache(sz*2);
      esw = (complexFormat*) memMngr.borrowCache(sz*2);
      patterns = (Real**)ccmemMngr.borrowCleanCache(scanx*scany*sizeof(Real*));
      fmt::println("initializing cuda image");
      resize_cuda_image(row_O,column_O);
      init_cuda_image(rcolor, 1./exposure);
      unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
      initRand(devstates,seed);
      shiftx = (Real*)ccmemMngr.borrowCleanCache(scansz);
      shifty = (Real*)ccmemMngr.borrowCleanCache(scansz);
      d_shiftx = (Real*)memMngr.borrowCleanCache(scansz);
      d_shifty = (Real*)memMngr.borrowCleanCache(scansz);
      step_shiftx = (Real*)ccmemMngr.borrowCache(scansz);
      step_shifty = (Real*)ccmemMngr.borrowCache(scansz);
      if(positionUncertainty > 1e-4){
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
      createWaveFront(d_object_intensity, d_object_phase, (complexFormat*)objectWave, 1);
      memMngr.returnCache(d_object_intensity);
      memMngr.returnCache(d_object_phase);
      verbose(2,
          plt.init(row_O,column_O);
          plt.plotComplex(objectWave, MOD2, 0, 1, "inputObject");
          plt.plotComplex(objectWave, PHASE, 0, 1, "inputPhase");
          //plt.plotPhase(objectWave, PHASERAD, 0, 1, "inputPhase");
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
      createWaveFront(d_intensity, d_phase, (complexFormat*)pupilobjectWave, oversampling);
      memMngr.returnCache(d_intensity);
      if(d_phase) memMngr.returnCache(d_phase);
      plt.init(row_tmp,column_tmp);
      plt.plotComplex(pupilobjectWave, MOD2, 0, 1, "pupilIntensity", 0);
      init_fft(row_tmp,column_tmp);
      //opticalPropagate((complexFormat*)pupilobjectWave, lambda, dpupil, beamspotsize*oversampling, row_tmp*column_tmp); //granularity changes
      angularSpectrumPropagate(pupilobjectWave, pupilobjectWave, beamspotsize*oversampling/lambda, dpupil/lambda, row_tmp*column_tmp); //granularity is the same
      plt.plotComplex(pupilobjectWave, MOD2, 0, 1, "pupilPattern", 0);
      resize_cuda_image(row,column);
      init_fft(row,column);
      crop((complexFormat*)pupilobjectWave, (complexFormat*)pupilpatternWave, row_tmp, column_tmp);
      plt.init(row,column);
      plt.plotComplexColor(pupilpatternWave, 0, 1, "probeWave", 0);
      calculateParameters();
      multiplyFresnelPhase(pupilpatternWave, d);
    }
    void initPosition(){
      if(runSim && positionUncertainty>1e-4){
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine generator(seed);
        std::normal_distribution<double> distribution(0.0, 1.);
        for(int i = 0 ; i < scanx*scany; i++){
          shiftx[i]+= distribution(generator)*positionUncertainty;
          shifty[i]+= distribution(generator)*positionUncertainty;
          if(verbose >=3 ) fmt::println("shifts ({}, {}): ({:f}, {:f})", i/scany, i%scany, shiftx[i],shifty[i]);
        }
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
          int shiftxpix = shiftx[idx]-round(shiftx[idx]);
          int shiftypix = shifty[idx]-round(shifty[idx]);
          getWindow((complexFormat*)objectWave, i*stepSize+round(shiftx[idx]), j*stepSize+round(shifty[idx]), row_O, column_O, window);
          if(fabs(shiftxpix)>1e-3||fabs(shiftypix)>1e-3){
            shiftWave((complexFormat*)window, shiftxpix, shiftypix);
          }
          multiply(esw, (complexFormat*)pupilpatternWave, window);
          verbose(5, plt.plotComplex(esw, MOD2, 0, 1, ("ptycho_esw"+to_string(i)+"_"+to_string(j)).c_str()));
          myFFT(esw,esw);
          if(!patterns[idx]) patterns[idx] = (Real*)memMngr.borrowCache(sz);
          getMod2(patterns[idx], esw, 1./(row*column));
          if(useBS) applySupport(patterns[idx], beamstop);
          if(simCCDbit) applyPoissonNoise_WO(patterns[idx], noiseLevel, devstates, 1./exposure);
          verbose(2, plt.plotFloat(patterns[idx], MOD, 1, exposure, (common.Pattern+to_string(i)+"_"+to_string(j)).c_str()));
          verbose(4, plt.plotFloat(patterns[idx], MOD, 1, exposure, (common.Pattern+to_string(i)+"_"+to_string(j)+"log").c_str(),1));
          idx++;
        }
      }
      memMngr.returnCache(window);
    }
    void initObject(){
      resize_cuda_image(row_O,column_O);
      random((complexFormat*)objectWave, devstates);
      resize_cuda_image(row,column);
      pupilFunc((complexFormat*)pupilpatternWave);
    }
    void updatePosition(int scanidx, complexFormat* obj, complexFormat* probe, Real* pattern, complexFormat* Fn){
      size_t siz = memMngr.getSize(obj);
      Real norm = 1./(row*column);
      complexFormat *cachex = (complexFormat*)memMngr.borrowCache(siz);
      complexFormat *cachey = (complexFormat*)memMngr.borrowCache(siz);
      Real *cache = (Real*)memMngr.borrowCache(siz>>1);
      myFFT(obj, cachex);
      cudaConvertFO(cachex, cachex, sqrt(norm));
      myMemcpyD2D(cachey, cachex, siz);
      multiplyx(cachex);
      multiplyy(cachey);
      cudaConvertFO(cachex, cachex, norm);
      cudaConvertFO(cachey, cachey, norm);
      myIFFT(cachex, cachex);
      myIFFT(cachey, cachey);
      multiply(cachex, cachex, probe);
      multiply(cachey, cachey, probe);
      myFFT(cachex, cachex);
      myFFT(cachey, cachey);
      calcPartial(cache, cachex, Fn, pattern, beamstop);
      findSum(cache, 0, d_shiftx+scanidx);
      calcPartial(cache, cachey, Fn, pattern, beamstop);
      findSum(cache, 0, d_shifty+scanidx);
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
      complexFormat *objCache = (complexFormat*)memMngr.borrowCache(sz*2);
      myCuDMalloc(Real, tmp, row*column);
      Real norm = 1./sqrt(row*column);
      int update_probe_iter = 4;
      int objFFT;
      createPlan(&objFFT, row_O, column_O); 
      int pupildiameter = 128;
      myCuDMalloc(complexFormat, zernikeCrop, pupildiameter*pupildiameter);
      void* zernike = zernike_init(pupildiameter, pupildiameter, 40, 0);
      for(int iter = 0; iter < nIter; iter++){
        int idx = 0;
        if(iter >= update_probe_iter) objMax = findMod2Max((complexFormat*)objectWave);
        probeMax = findMod2Max((complexFormat*)pupilpatternWave);
        if(iter >= update_probe_iter && iter%50==0){
          resize_cuda_image(row_O,column_O);
          applyNorm((complexFormat*)objectWave, pow(probeMax/objMax, 0.25));
          resize_cuda_image(row,column);
          applyNorm((complexFormat*)pupilpatternWave, pow(objMax/probeMax,0.25));
          objMax = probeMax = sqrt(objMax*probeMax);
        }
        //complexFormat* coeff = NULL, *projection = NULL;
        int positionUpdateIter = 50;
        bool doUpdatePosition = iter % 20 == 0 && iter >= positionUpdateIter;
        for(int i = 0; i < scanx; i++){
          for(int j = 0; j < scany; j++){
            int shiftxpix = shiftx[idx]-round(shiftx[idx]);
            int shiftypix = shifty[idx]-round(shifty[idx]);
            int shiftxn = i*stepSize+round(shiftx[idx]);
            int shiftyn = j*stepSize+round(shifty[idx]);
            getWindow((complexFormat*)objectWave, shiftxn, shiftyn, row_O, column_O, objCache);
            bool shiftpix = fabs(shiftxpix)>1e-3||fabs(shiftypix)>1e-3;
            if(shiftpix){
              shiftWave((complexFormat*)objCache, shiftxpix, shiftypix);
            }
            multiply(esw, (complexFormat*)pupilpatternWave, objCache);
            myFFT(esw,Fn);
            if(doUpdatePosition) {
              applyNorm(Fn, norm);
              updatePosition(idx, objCache, (complexFormat*)pupilpatternWave, patterns[idx], Fn);
              applyMod(Fn, patterns[idx],beamstop,noiseLevel, 1);
            }else applyMod(Fn, patterns[idx],beamstop,noiseLevel, norm);
            myIFFT(Fn,Fn);
            add(esw, Fn, -norm);
            if(iter < update_probe_iter) updateObject(objCache, (complexFormat*)pupilpatternWave, esw,//1,1);
                probeMax);
            else updateObjectAndProbe(objCache, (complexFormat*)pupilpatternWave, esw,//1,1);
                probeMax, objMax);
            if(shiftpix){
              shiftWave(objCache, -shiftxpix, -shiftypix);
            }
            updateWindow((complexFormat*)objectWave, shiftxn, shiftyn, row_O, column_O, objCache);
            idx++;
          }
        }
        if(doUpdatePosition){
          size_t scansz = scanx*scany*sizeof(Real);
          myMemcpyD2H(step_shiftx, d_shiftx, scansz);
          myMemcpyD2H(step_shifty, d_shifty, scansz);
          clearCuMem(d_shifty, scansz);
          clearCuMem(d_shiftx, scansz);
          const int N = scanx * scany;
          for (int i = 0; i < N; ++i) {
            shiftx[i] += 0.8*step_shiftx[i];
            shifty[i] += 0.8*step_shifty[i];
          }
        }
        if(iter >= update_probe_iter){
          getMod2(tmp, (complexFormat*)pupilpatternWave);
          complexFormat middle = findMiddle(tmp);
          Real biasx = crealf(middle), biasy = cimagf(middle);
          Real offsetx = 0, offsety = 0;
          const int N = scanx * scany;
          for (int i = 0; i < N; ++i) {
            offsetx += shiftx[i];
            offsety += shifty[i];
          }
          offsetx = offsetx / N;
          offsety = offsety / N;
          for (int i = 0; i < N; ++i) {
            shiftx[i] -= offsetx;
            shifty[i] -= offsety;
          }
          angularSpectrumPropagate(pupilpatternWave, pupilpatternWave, beamspotsize*oversampling/lambda, -dpupil/lambda, row*column); //granularity is the same
          getMod2(tmp, (complexFormat*)pupilpatternWave);
          myCuDMalloc(Real, d_norm, 2);
          myDMalloc(Real, h_norm, 2);
          findSum(tmp, row*column, d_norm);
          if(iter == nIter -1) {
            plt.plotComplexColor(pupilpatternWave, 0, 1, "recon_pupil");
            plt.saveComplex(pupilpatternWave, "pupilwave");
          }
          resize_cuda_image(row_O, column_O);
          shiftWave(objFFT,(complexFormat*)objectWave, -offsetx-biasx*row, -offsety-biasy*column);
          resize_cuda_image(pupildiameter, pupildiameter);
          crop((complexFormat*)pupilpatternWave, zernikeCrop, row, column, biasx, biasy);
          zernike_compute(zernike, (complexFormat*)zernikeCrop, pupildiameter>>1, pupildiameter>>1);
          zernike_reconstruct(zernike, (complexFormat*)zernikeCrop);
          resize_cuda_image(row, column);
          pad(zernikeCrop, (complexFormat*)pupilpatternWave, pupildiameter, pupildiameter);
          /*
             complexFormat** result = zernikeDecomposition((complexFormat*)pupilpatternWave, 5, 48, coeff, projection);
             if(result){
             coeff = result[0];
             projection = (complexFormat*)pupilpatternWave;
             pupilpatternWave = result[1];
             myFree(result);
             }else{
             complexFormat* tmp = projection;
             projection = (complexFormat*)pupilpatternWave;
             pupilpatternWave = tmp;
             }
             if(iter == nIter -1) {
             for (int ic=0 ; ic < 21; ic++) {
             fmt::println("x[{}]=({:.2g},{:.2g})",ic, crealf(coeff[ic]), cimagf(coeff[ic]));
             }
             }
             */
          getMod2(tmp, (complexFormat*)pupilpatternWave);
          findSum(tmp, row*column, d_norm+1);
          myMemcpyD2H(h_norm, d_norm, 2*sizeof(Real));
          Real norm = h_norm[0]/h_norm[1];
          myCuFree(d_norm);
          myFree(h_norm);
          if(norm > 2 || norm < 0.5) applyNorm((complexFormat*)pupilpatternWave, sqrt(norm));
          if(iter == nIter -1) plt.plotComplexColor(pupilpatternWave, 0, 1, "recon_pupil_proj");
          angularSpectrumPropagate(pupilpatternWave, pupilpatternWave, beamspotsize*oversampling/lambda, dpupil/lambda, row*column); //granularity is the same
        }
        if(iter == positionUpdateIter){
          resize_cuda_image(row_O,column_O);
          plt.init(row_O, column_O);
          plt.plotComplex(objectWave, MOD2, 0, 1, "ptycho_b4position", 0);
          plt.plotComplex(objectWave, PHASE, 0, 1, "ptycho_b4positionphase", 0);
          resize_cuda_image(row,column);
          plt.init(row, column);
        }
      }
      if(verbose>=3){
        for(int i = 0 ; i < scanx*scany; i++){
          fmt::println("recon shifts ({}, {}): ({:f}, {:f})", i/scany, i%scany, shiftx[i],shifty[i]);
        }
      }
      memMngr.returnCache(Fn);
      memMngr.returnCache(objCache);
      plt.plotComplexColor(pupilpatternWave, 0, 0.12, "ptycho_probe_afterIter", 0);
      Real crop_ratio = 0.65;
      int rowc = row_O*crop_ratio;
      int colc = column_O*crop_ratio;
      resize_cuda_image(rowc, colc);
      plt.init(rowc, colc);
      complexFormat* cropped = (complexFormat*)memMngr.borrowCache(rowc*colc*sizeof(complexFormat));
      myCuDMalloc(Real, angle, rowc*colc);
      crop((complexFormat*)objectWave, cropped, row_O, column_O);
      getArg(angle, cropped);
      applyNorm(angle, 1./(2*M_PI));
      plt.plotComplex(cropped, MOD2, 0, 0.7, "ptycho_afterIter");
      //plt.plotPhase(cropped, PHASERAD, 0, 1, "ptycho_afterIterphase");
      //phaseUnwrapping(angle, angle, rowc, colc);
      plt.plotFloat(angle, REAL, 0, 1, "ptycho_afterIterphase");
      plt.plotComplexColor(cropped, 0, 1, "ptycho_afterIterwave");
    }
    void readPattern(){
      Real* pattern = readImage((string(common.Pattern)+"0_0.png").c_str(), row, column);
      plt.init(row,column);
      init_fft(row,column);
      sz = row*column*sizeof(Real);
      allocateMem();
      resize_cuda_image(row,column);
      createBeamStop();
      int idx = 0;
      for(int i = 0; i < scanx; i++){
        for(int j = 0; j < scany; j++){
          if(idx!=0) pattern = readImage((common.Pattern+to_string(i)+"_"+to_string(j)+".png").c_str(), row, column);
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
      if(runSim) d = resolution*pixelsize*row/lambda;
      experimentConfig::calculateParameters();
    }
};
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

