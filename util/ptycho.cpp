#include <stdio.h>
#include <time.h>
#include <random>
#include <chrono>
#include <fstream>
#include "imgio.h"
#include "cudaConfig.h"
#include "experimentConfig.h"
#include "cuPlotter.h"
#include "cub_wrap.h"
#include "ptycho.h"


using namespace std;

//#define Bits 16

class ptycho : public experimentConfig{
  public:
    int row_O = 512;  //in ptychography this is different from row (the size of probe).
    int column_O = 512;
    int sz = 0;
    int stepSize = 32;
    int doPhaseModulationPupil = 0;
    int scanx = 0;
    int scany = 0;
    Real *shiftx = 0;
    Real *shifty = 0;
    Real **patterns; //patterns[i*scany+j] points to the address on device to store pattern;
    complexFormat *esw;
    void *devstates = 0;

    ptycho(const char* configfile):experimentConfig(configfile){}
    void allocateMem(){
      if(devstates) return;
      devstates = newRand(column_O * row_O);
      printf("allocating memory\n");
      scanx = (row_O-row)/stepSize+1;
      scany = (column_O-column)/stepSize+1;
      printf("scanning %d x %d steps\n", scanx, scany);
      objectWave = (complexFormat*)memMngr.borrowCache(row_O*column_O*sizeof(Real)*2);
      pupilpatternWave = (complexFormat*)memMngr.borrowCache(sz*2);
      esw = (complexFormat*) memMngr.borrowCache(sz*2);
      patterns = (Real**)ccmemMngr.borrowCleanCache(scanx*scany*sizeof(Real*));
      printf("initializing cuda image\n");
      resize_cuda_image(row_O,column_O);
      init_cuda_image(rcolor, 1./exposure);
      unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
      initRand(devstates,seed);
      shiftx = (Real*)ccmemMngr.borrowCleanCache(scanx*scany*sizeof(Real));
      shifty = (Real*)ccmemMngr.borrowCleanCache(scanx*scany*sizeof(Real));
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
      opticalPropagate((complexFormat*)pupilobjectWave, lambda, dpupil, beamspotsize*oversampling, row_tmp*column_tmp); //granularity changes
      //angularSpectrumPropagate(pupilobjectWave, pupilobjectWave, beamspotsize*oversampling/lambda, dpupil/lambda, row_tmp*column_tmp); //granularity is the same
      plt.plotComplex(pupilobjectWave, MOD2, 0, 1, "pupilPattern", 0);
      resize_cuda_image(row,column);
      init_fft(row,column);
      crop((complexFormat*)pupilobjectWave, (complexFormat*)pupilpatternWave, row_tmp, column_tmp);
      plt.init(row,column);
      plt.plotComplex(pupilpatternWave, MOD2, 0, 0.08, "probeIntensity", 0);
      plt.plotComplex(pupilpatternWave, PHASE, 0, 1, "probePhase", 0);
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
          if(verbose >=3 ) printf("shifts (%d, %d): (%f, %f)\n", i/scany, i%scany, shiftx[i],shifty[i]);
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
          int shiftypix = shiftx[idx]-round(shifty[idx]);
          getWindow((complexFormat*)objectWave, i*stepSize+round(shiftx[idx]), j*stepSize+round(shifty[idx]), row_O, column_O, window);
          if(fabs(shiftxpix)>1e-3||fabs(shiftypix)>1e-3){
            shiftWave((complexFormat*)window, shiftxpix, shiftypix);
          }
          multiply(esw, (complexFormat*)pupilpatternWave, window);
          verbose(5, plt.plotComplex(esw, MOD2, 0, 1, ("ptycho_esw"+to_string(i)+"_"+to_string(j)).c_str()));
          propagate(esw,esw,1);
          if(!patterns[idx]) patterns[idx] = (Real*)memMngr.borrowCache(sz);
          getMod2(patterns[idx], esw);
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
    void updatePosition(Real &shiftx, Real &shifty, complexFormat* obj, complexFormat* probe, Real* pattern, complexFormat* Fn){
      Real siz = memMngr.getSize(obj);
      complexFormat *cachex = (complexFormat*)memMngr.borrowCache(siz);
      complexFormat *cachey = (complexFormat*)memMngr.borrowCache(siz);
      propagate(obj, cachex, 1);
      cudaConvertFO(cachex);
      myMemcpyD2D(cachey, cachex, siz);
      multiplyx(cachex);
      multiplyy(cachey);
      cudaConvertFO(cachex);
      cudaConvertFO(cachey);
      propagate(cachex, cachex, 0);
      propagate(cachey, cachey, 0);
      multiply(cachex, probe);
      multiply(cachey, probe);
      propagate(cachex, cachex, 1);
      propagate(cachey, cachey, 1);
      calcPartial(cachex, Fn, pattern, beamstop);
      calcPartial(cachey, Fn, pattern, beamstop);
      shiftx += 0.3*findSumReal(cachex);
      shifty += 0.3*findSumReal(cachey);
      memMngr.returnCache(cachex);
      memMngr.returnCache(cachey);
      if(shiftx!=shiftx || shifty!=shifty) exit(0);
    }
    void iterate(){
      resetPosition();
      resize_cuda_image(row,column);
      Real objMax;
      Real probeMax;
      complexFormat *Fn = (complexFormat*)memMngr.borrowCache(sz*2);
      complexFormat *objCache = (complexFormat*)memMngr.borrowCache(sz*2);
      int update_probe_iter = 4;
      for(int iter = 0; iter < nIter; iter++){
        int idx = 0;
        if(iter >= update_probe_iter) objMax = findMod2Max((complexFormat*)objectWave);
        probeMax = findMod2Max((complexFormat*)pupilpatternWave);
        if(iter >= update_probe_iter && iter%200==0){
          resize_cuda_image(row_O,column_O);
          applyNorm((complexFormat*)objectWave, pow(probeMax/objMax, 0.25));
          resize_cuda_image(row,column);
          applyNorm((complexFormat*)pupilpatternWave, pow(objMax/probeMax,0.25));
          objMax = probeMax = sqrt(objMax*probeMax);
        }
        for(int i = 0; i < scanx; i++){
          for(int j = 0; j < scany; j++){
            int shiftxpix = shiftx[idx]-round(shiftx[idx]);
            int shiftypix = shiftx[idx]-round(shifty[idx]);
            int shiftxn = i*stepSize+round(shiftx[idx]);
            int shiftyn = j*stepSize+round(shifty[idx]);
            getWindow((complexFormat*)objectWave, shiftxn, shiftyn, row_O, column_O, objCache);
            bool shiftpix = fabs(shiftxpix)>1e-3||fabs(shiftypix)>1e-3;
            if(shiftpix){
              shiftWave((complexFormat*)objCache, shiftxpix, shiftypix);
            }
            multiply(esw, (complexFormat*)pupilpatternWave, objCache);
            propagate(esw,Fn,1);
            if(iter % 20 == 0 && iter >= 20) {
              updatePosition(shiftx[idx], shifty[idx], objCache, (complexFormat*)pupilpatternWave, patterns[idx], Fn);
            }
            applyMod(Fn, patterns[idx],beamstop,1);
            propagate(Fn,Fn,0);
            add(esw, Fn, -1);
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
        if(iter == 100){
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
          printf("recon shifts (%d, %d): (%f, %f)\n", i/scany, i%scany, shiftx[i],shifty[i]);
        }
      }
      memMngr.returnCache(Fn);
      memMngr.returnCache(objCache);
      plt.plotComplex(pupilpatternWave, MOD2, 0, 0.12, "ptycho_probe_afterIter", 0);
      plt.plotComplex(pupilpatternWave, PHASE, 0, 1, "ptycho_probe_afterIterphase", 0);
      Real crop_ratio = 0.65;
      int rowc = row_O*crop_ratio;
      int colc = column_O*crop_ratio;
      resize_cuda_image(rowc, colc);
      plt.init(rowc, colc);
      complexFormat* cropped = (complexFormat*)memMngr.borrowCache(rowc*colc*sizeof(complexFormat));
      crop((complexFormat*)objectWave, cropped, row_O, column_O);
      plt.plotComplex(cropped, MOD2, 0, 0.7, "ptycho_afterIter");
      //plt.plotPhase(cropped, PHASERAD, 0, 1, "ptycho_afterIterphase");
      plt.plotComplex(objectWave, PHASE, 0, 1, "ptycho_afterIterphase");
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
      printf("Created pattern data\n");
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
    printf("please feed the object intensity and phase image\n");
  }
  if(setups.runSim){
    setups.readPupilAndObject();
    setups.createPattern();
  }else{
    setups.readPattern();
  }
  printf("Imaging distance = %4.2fcm\n", setups.d*1e-4);
  printf("fresnel factor = %f\n", setups.fresnelFactor);
  printf("Resolution = %4.2fnm\n", setups.resolution*1e3);

  printf("pupil Imaging distance = %4.2fcm\n", setups.dpupil*1e-4);
  printf("pupil fresnel factor = %f\n", setups.fresnelFactorpupil);
  printf("pupil enhancement = %f\n", setups.enhancementpupil);
  setups.initObject();
  setups.iterate();

  return 0;
}

