#include <complex.h>
#include <fmt/base.h>
#include <fmt/os.h>
#include <fstream>
#include <stdio.h>
#include <random>
#include <chrono>
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
    int row_O = 725;  //in ptychography this is different from row (the size of probe).
    int column_O = 725;
    int sz = 0;
    int doPhaseModulationPupil = 0;
    complexFormat* registerCache = 0;
    int registerFFTHandle = 0;
    int nscan = 0;
    Real *scanpos = 0;
    Real *scanposx = 0;
    Real *scanposy = 0;
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
    void readScan(){
      std::ifstream file(distFile);
      if (!file.is_open()) {
        throw std::runtime_error(fmt::format("Could not open file '{}' for reading.", distFile));
      }
      std::vector<std::vector<float>> distances;
      std::string line;
      std::getline(file, line);
      std::istringstream iss(line);
      iss >> stepSize;
      while (std::getline(file, line)) {
        int i; 
        float x, y;
        std::istringstream iss(line);
        if (iss >> i >> x >> y) {
          distances.push_back({x, y});
        } else {
          throw std::runtime_error(fmt::format("Failed to parse line: {}", line));
        }
      }
      nscan = distances.size();
      size_t scansz = nscan*sizeof(Real);
      patterns = (Real**)ccmemMngr.borrowCleanCache(nscan*sizeof(Real*));
      shifts = (Real*)ccmemMngr.borrowCleanCache(scansz*2);
      scanpos = (Real*)ccmemMngr.borrowCleanCache(scansz*2);
      scanposx = scanpos;
      scanposy = scanpos + nscan;
      d_shifts = (Real*)memMngr.borrowCleanCache(scansz*2);
      shiftx = shifts;
      shifty = shifts+nscan;
      d_shift = (Real*)memMngr.borrowCache(scansz*2);
      for(int i = 0; i < nscan; i++){
        scanposx[i] = distances[i][0]*stepSize;
        scanposy[i] = distances[i][1]*stepSize;
        fmt::println("scan position: ({}, {}) ", scanposx[i], scanposy[i]);
      }
    }
    void initScan(){
      int scanx = 3;//(row_O-row)/stepSize+1;
      int scany = 3;//(column_O-column)/(stepSize*sqrtf(3)/2)+1;
      nscan = scanx*scany;
      size_t scansz = nscan*sizeof(Real);
      patterns = (Real**)ccmemMngr.borrowCleanCache(nscan*sizeof(Real*));
      fmt::println("scanning {} x {} steps", scanx, scany);
      scanpos = (Real*)ccmemMngr.borrowCleanCache(scansz*2);
      scanposx = scanpos;
      scanposy = scanpos + nscan;
      shifts = (Real*)ccmemMngr.borrowCleanCache(scansz*2);
      d_shifts = (Real*)memMngr.borrowCleanCache(scansz*2);
      shiftx = shifts;
      shifty = shifts+nscan;
      d_shift = (Real*)memMngr.borrowCache(scansz*2);

      fmt::ostream scanFile(fmt::output_file("scan.txt"));
      scanFile.print("{}\n", stepSize);
      int iscan = 0;
      for(int i = 0; i < scanx; i++){
        for(int j = 0; j < scany; j++){
          Real xstep = (i+(j%2 == 0? 0:0.5));
          Real ystep = j*sqrtf(3)/2;
          scanFile.print("{} {} {}\n", iscan ,xstep,ystep);
          scanposx[iscan] = xstep*stepSize;
          scanposy[iscan] = ystep*stepSize;
          fmt::println("scan position: ({}, {}) ", scanposx[iscan], scanposy[iscan]);
          iscan++;
        }
      }
    }
    void allocateMem(){
      if(devstates) return;
      devstates = newRand(column_O * row_O);
      fmt::println("allocating memory");
      objectWave = (complexFormat*)memMngr.borrowCache(row_O*column_O*sizeof(Real)*2);
      objectWave_t = (complexFormat*)memMngr.borrowCache(row_O*column_O*sizeof(Real)*2);
      pupilpatternWave = (complexFormat*)memMngr.borrowCache(sz*2);
      pupilpatternWave_t = (complexFormat*)memMngr.borrowCache(sz*2);
      esw = (complexFormat*) memMngr.borrowCache(sz*2);
      fmt::println("initializing cuda image");
      resize_cuda_image(row_O,column_O);
      init_cuda_image(rcolor, 1./exposure);
      unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
      initRand(devstates,seed);
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
      initScan();
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
    }
    void initPosition(){
      unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
      std::default_random_engine generator(seed);
      std::normal_distribution<double> distribution(0.0, 1.);
      for(int i = 0 ; i < nscan; i++){
        shiftx[i]+= distribution(generator)*positionUncertainty;
        shifty[i]+= distribution(generator)*positionUncertainty;
        if(verbose >=3 ) fmt::println("shifts {}: ({:f}, {:f})", i, shiftx[i],shifty[i]);
      }
    }
    void resetPosition(){
      for(int i = 0 ; i < nscan; i++){
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
      for(int i = 0; i < nscan; i++){
          Real posx = scanposx[i] + shiftx[idx];
          Real posy = scanposy[i] + shifty[idx];
          int shiftxpix = posx-round(posx);
          int shiftypix = posy-round(posy);
          getWindow(objectWave_t, +round(posx), round(posy), row_O, column_O, window);
          if(fabs(shiftxpix)>1e-3||fabs(shiftypix)>1e-3){
            shiftWave(window, shiftxpix, shiftypix);
          }
          multiply(esw, pupilpatternWave_t, window);
          verbose(5, plt.plotComplex(esw, MOD2, 0, 1, ("ptycho_esw"+to_string(i)).c_str()));
          multiplyFresnelPhase(esw, d);
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
          cudaConvertFO(patterns[idx]);
          verbose(2, plt.plotFloat(patterns[idx], MOD, 0, 1, (common.Pattern+to_string(i)).c_str()));
          verbose(4, plt.plotFloat(patterns[idx], MOD, 0, 1, (common.Pattern+to_string(i)+"log").c_str(),1, 0, 1));
          plt.saveFloat(patterns[idx], (common.Pattern+to_string(i)).c_str());
          if(idx == 0) {
            extendToComplex(patterns[0],esw);
            myFFT(esw, esw);
            plt.plotComplex(esw, MOD2, 1, 1./exposure/row/column, "autocorrelation", 1, 0, 1);
          }
          cudaConvertFO(patterns[idx]);
          applyNorm(patterns[idx], 1./exposure);
          idx++;
        }
      memMngr.returnCache(window);
    }
    void initObject(){
      resize_cuda_image(row_O,column_O);
      random(objectWave, devstates);
      linearConst(objectWave, objectWave, 0.5, 0.0);
      resize_cuda_image(row,column);
      pupilFunc(pupilpatternWave);
      angularSpectrumPropagate(pupilpatternWave, pupilpatternWave, beamspotsize*oversampling/lambda, dpupil/lambda, row*column); //granularity is the same
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
      findSum(cache, 0, d_shift+scanidx+nscan);
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
      int pupildiameter = 145;
      myCuDMalloc(complexFormat, zernikeCrop, pupildiameter*pupildiameter);
      void* zernike = zernike_init(pupildiameter, pupildiameter, 20, 0); //40 is already high enough for modelling complex beams
      myCuDMalloc(Real, pupilSupport, pupildiameter*pupildiameter);
      resize_cuda_image(pupildiameter, pupildiameter);
      createCircleMask(pupilSupport, Real(pupildiameter+1)/2, Real(pupildiameter+1)/2, Real(pupildiameter)/2);
      resize_cuda_image(row,column);
      bool mPIE = 0;
      Real probeStepSize = 4;
      Real zernikeStepSize = 4;
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
          resize_cuda_image(row_O>>4,column_O>>4);
          crop(objectWave, cropObj, row_O, column_O);
          getMod2(maxCache, cropObj);
          findMax(maxCache, row_O*column_O/256, d_norm+1);
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
        for(int i = 0; i < nscan; i++){
          Real posx = scanposx[i] + shiftx[idx];
          Real posy = scanposy[i] + shifty[idx];
            int shiftxpix = posx-round(posx);
            int shiftypix = posy-round(posy);
            getWindow(objectWave, posx, posy, row_O, column_O, objCache);
            bool shiftpix = fabs(shiftxpix)>1e-3||fabs(shiftypix)>1e-3;
            if(shiftpix){
              shiftWave(objCache, shiftxpix, shiftypix);
            }
            multiply(esw, pupilpatternWave, objCache);
            multiplyFresnelPhase(esw, d);
            myFFT(esw,Fn);
            if(iter == nIter - 1){
              verbose(4,plt.plotComplex(Fn, MOD2, 1, 1./(row*column), ("ptycho_recon_pattern" + to_string(i)).c_str(), 1, 0, 1));
            }
            if(doUpdatePosition) {
              applyNorm(Fn, norm);
              updatePosition(idx, objCache, pupilpatternWave, patterns[idx], Fn);
              applyModAccurate(Fn, patterns[idx],beamstop, 1);
            }else applyModAccurate(Fn, patterns[idx],beamstop, norm);
            myIFFT(Fn,Fn);
            add(esw, Fn, -norm);
            multiplyFresnelPhase(esw, -d);
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
        resize_cuda_image(row_O, column_O);
        //FISTA((complexFormat*) objectWave, (complexFormat*) objectWave, 3e-4, 1, 0);
        resize_cuda_image(row, column);
        if(!mPIE && iter >= update_probe_iter) add(pupilpatternWave, probeStep,(iter > zernikeIter?probeStepSize:zernikeStepSize)/(nscan));
        if(doUpdatePosition){
          resize_cuda_image(nscan*2, 1);
          addProbability(d_shifts, d_shift, 0.8, positionUncertainty);
          resize_cuda_image(row, column);
        }
        if(iter >= update_probe_iter){
          getMod2(tmp, (complexFormat*)pupilpatternWave);
          complexFormat middle = findMiddle(tmp);
          if(saveVideoEveryIter && iter%saveVideoEveryIter == 0){
            plt.toVideo = vidhandle_probe;
            plt.plotComplexColor(pupilpatternWave, 0, 1, ("recon_probe"+to_string(iter)).c_str(), 0, isFlip);
            plt.toVideo = -1;
          }
          findSum(d_shifts, nscan, d_norm);
          findSum(d_shifts+nscan, nscan, d_norm+1);
          myMemcpyD2H(h_norm, d_norm, 2*sizeof(Real));
          resize_cuda_image(nscan, 1);
          h_norm[0] /= nscan;
          h_norm[1] /= nscan;
          linearConst(d_shifts, d_shifts, 1, -h_norm[0]);
          linearConst(d_shifts+nscan, d_shifts+nscan, 1, -h_norm[1]);
          resize_cuda_image(nscan*2, 1);
          resize_cuda_image(row_O, column_O);
          Real biasx = crealf(middle);
          Real biasy = cimagf(middle);
          shiftWave(objFFT,(complexFormat*)objectWave, -h_norm[0]-biasx*row, -h_norm[1]-biasy*column);
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
          crop((complexFormat*)pupilpatternWave, zernikeCrop, row, column, biasx, biasy);
          if(iter < zernikeIter){
            zernike_compute(zernike, zernikeCrop, Real(pupildiameter-1)/2, Real(pupildiameter-1)/2, Real(pupildiameter)/2);
            zernike_reconstruct(zernike, zernikeCrop, Real(pupildiameter)/2);
          }else{
            applyMask(zernikeCrop, pupilSupport);
          }
          resize_cuda_image(row, column);
          pad(zernikeCrop, pupilpatternWave, pupildiameter, pupildiameter);
          if(iter == zernikeIter - 1) {
            plt.plotComplexColor(pupilpatternWave, 0, 1, "recon_pupil_proj");
          }
          angularSpectrumPropagate(pupilpatternWave, pupilpatternWave, beamspotsize*oversampling/lambda, dpupil/lambda, row*column); //granularity is the same
            myFFT(pupilpatternWave, esw); //just reuse esw, instread of allocating new memory
            cudaConvertFO(esw);
            getMod2(tmp, esw);
            middle = findMiddle(tmp);
            multiplyShift(pupilpatternWave, crealf(middle)*row, cimagf(middle)*column);
            if(iter == nIter-1) plt.plotComplexColor(pupilpatternWave, 0, 1, "debug1");
            resize_cuda_image(row_O, column_O);
            //plt.init(row_O, column_O, outputDir);
            //plt.plotComplexColor(objectWave, 0, 1, "debug2");
            multiplyShift(objectWave, -crealf(middle)*row_O, -cimagf(middle)*column_O);
            //plt.plotComplexColor(objectWave, 0, 1, "debug3");
            resize_cuda_image(row, column);
            //plt.init(row, column, outputDir);
        }
        if(doUpdatePosition || iter >= update_probe_iter){
          myMemcpyD2H(shifts, d_shifts, nscan*sizeof(Real)*2);
        }
        if(iter == positionUpdateIter){
          resize_cuda_image(row_O,column_O);
          plt_O.plotComplex(objectWave, MOD2, 0, 1, "ptycho_b4position", 0);
          plt_O.plotComplex(objectWave, PHASE, 0, 1, "ptycho_b4positionphase", 0);
          resize_cuda_image(row,column);
        }
      }
      plt.plotComplexColor(pupilpatternWave, 0, 1, "ptycho_probe_afterIter", 1);
      angularSpectrumPropagate(pupilpatternWave, pupilpatternWave, beamspotsize*oversampling/lambda, -dpupil/lambda, row*column); //granularity is the same
      plt.plotComplexColor(pupilpatternWave, 0, 1, "recon_pupil");
      if(verbose>=3){
        for(int i = 0 ; i < nscan; i++){
          fmt::println("recon shifts {}: ({:f}, {:f})", i, shiftx[i],shifty[i]);
        }
      }
      myCuFree(d_norm);
      myFree(h_norm);
      memMngr.returnCache(Fn);
      memMngr.returnCache(objCache);
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
      Real* pattern = readImage((outputDir + string(common.Pattern)+"0.bin").c_str(), row, column);
      plt.init(row,column, outputDir);
      init_fft(row,column);
      sz = row*column*sizeof(Real);
      readScan();
      allocateMem();
      resize_cuda_image(row,column);
      if(useBS) {
        createBeamStop();
      }
      int idx = 0;
      for(int i = 0; i < nscan; i++){
        if(idx!=0) pattern = readImage((string(outputDir) + common.Pattern+to_string(i)+".bin").c_str(), row, column);
        if(!patterns[idx]) patterns[idx] = (Real*)memMngr.borrowCache(sz);
        myMemcpyH2D(patterns[idx], pattern, sz);
        ccmemMngr.returnCache(pattern);
        cudaConvertFO(patterns[idx]);
        applyNorm(patterns[idx], 1./exposure);
        verbose(3, plt.plotFloat(patterns[idx], MOD, 1, exposure, ("input"+string(common.Pattern)+to_string(i)).c_str()));
        idx++;
      }
      fmt::println("Created pattern data");
      calculateParameters();
    }
    void calculateParameters(){
      resolution = lambda*dpupil/beamspotsize/oversampling;
      if(runSim){
        //d = resolution*pixelsize*row/lambda;
        //fmt::println("Desired d = {}", d);
      }
      experimentConfig::calculateParameters();
    }
};
Real ptycho::computeErrorSim(){
  int upsampling = 3;
  if(!registerFFTHandle){
    createPlan(&registerFFTHandle, row*upsampling, column*upsampling);
    myCuMalloc(complexFormat, registerCache, upsampling*upsampling*row*column);
  } 
  myCuDMalloc(complexFormat, convoluted, row*column);
  convolute(convoluted, pupilpatternWave_t, pupilpatternWave, registerCache, upsampling, registerFFTHandle);
  getMod2((Real*)registerCache, convoluted);
  int index = findMaxIdx((Real*)registerCache, row*column);
  Real x = index/column;
  Real y = index%column;
  x = (x-row/2)/upsampling+row/2;
  y = (y-column/2)/upsampling+column/2;
  shiftWave(pupilpatternWave, -x, -y);
  shiftWave(objectWave, -x, -y);
  //getPhaseDiff(registerCache, pupilpatternWave_t, pupilpatternWave);
  //myFFT
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

