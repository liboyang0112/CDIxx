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
#include "memManager.hpp"
#include "misc.hpp"
#include "ptycho.hpp"
#include "tvFilter.hpp"
#include "beamDecomposition.hpp"


using namespace std;

void shuffle_array(int *array, int n) {
    if (array == NULL || n <= 1) return;

    for (int i = n - 1; i > 0; --i) {
        int j = rand() % (i + 1);  // random index in [0, i]
        // Swap array[i] and array[j]
        int temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

#define N 5                // number of vertices
#define EDGE_LENGTH 1.0

class ptycho : public experimentConfig{
  public:
    int row_O = 1280;  //in ptychography this is different from row (the size of probe).
    int column_O = 1280;
    int sz = 0;
    int doPhaseModulationPupil = 0;
    complexFormat* registerCache = 0;
    void* registerFFTHandle = 0;
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
      stepSize /= resolution;
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
      int offsetx = row/10;
      int offsety = column/8;
      for(int i = 0; i < nscan; i++){
        scanposx[i] = distances[i][0]*stepSize-offsetx;
        scanposy[i] = distances[i][1]*stepSize-offsety;
        fmt::println("scan position: ({}, {}) ", scanposx[i], scanposy[i]);
      }
    }
    void allocScan(){
      size_t scansz = nscan*sizeof(Real);
      patterns = (Real**)ccmemMngr.borrowCleanCache(nscan*sizeof(Real*));
      scanpos = (Real*)ccmemMngr.borrowCleanCache(scansz*2);
      scanposx = scanpos;
      scanposy = scanpos + nscan;
      shifts = (Real*)ccmemMngr.borrowCleanCache(scansz*2);
      d_shifts = (Real*)memMngr.borrowCleanCache(scansz*2);
      shiftx = shifts;
      shifty = shifts+nscan;
      d_shift = (Real*)memMngr.borrowCache(scansz*2);
    }
    void initScan_triangle(){
      int scanx = 3;//(row_O-row)/stepSize+1;
      int scany = 3;//(column_O-column)/(stepSize*sqrtf(3)/2)+1;
      nscan = scanx*scany-1;
      int offsetx = -row*0.27;
      int offsety = -column*0.23;
      fmt::println("scanning {} x {} steps", scanx, scany);
      allocScan();
      fmt::ostream scanFile(fmt::output_file("scan.txt"));
      scanFile.print("{}\n", stepSize*resolution);
      int iscan = 0;
      for(int i = 0; i < scanx; i++){
        for(int j = 0; j < scany; j++){
          if(i == 2 && j == 1) continue;
          Real xstep = (i+(j%2 == 0? 0:0.5));
          Real ystep = j*sqrtf(3)/2;
          scanFile.print("{} {} {}\n", iscan ,ystep,xstep);
          scanposy[iscan] = xstep*stepSize + offsetx;
          scanposx[iscan] = ystep*stepSize + offsety;
          fmt::println("scan position: ({}, {}) ", scanposx[iscan], scanposy[iscan]);
          iscan++;
        }
      }
    }
    void initScan(){
      Real R = EDGE_LENGTH / (2.0 * sin(M_PI / N));
      nscan = N;
      int offsetx = row_O/2 - row/2;
      int offsety = column_O/2 - column/2;
      allocScan();
      fmt::println("Vertices of a regular pentagon with edge length {:.1f}:", EDGE_LENGTH);
      fmt::ostream scanFile(fmt::output_file("scan.txt"));
      scanFile.print("{}\n", stepSize*resolution);
      for (int i = 0; i < N; ++i) {
        Real angle = 2.0 * M_PI * i / N; // start from top (rotate -90°)
        Real xstep = R * cos(angle);
        Real ystep = R * sin(angle);
        scanFile.print("{} {} {}\n", i ,xstep,ystep);
        scanposx[i] = xstep * stepSize + offsetx;
        scanposy[i] = ystep * stepSize + offsety;
        fmt::println("scan position: ({}, {}) ", scanposx[i], scanposy[i]);
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
      calculateParameters();
      initScan_triangle();
      allocateMem();
      createWaveFront(d_object_intensity, d_object_phase, objectWave_t, 1);
      //createWaveFront(d_object_intensity, 0, objectWave_t, 1);
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
      if(useBS) {
        createBeamStop();
        plt.plotFloat(beamstop, MOD, 1, 1,"beamstop", 0);
      }
      complexFormat* window = (complexFormat*)memMngr.borrowCache(sz*2);
      for(int i = 0; i < nscan; i++){
        Real posx = scanposx[i] + shiftx[i];
        Real posy = scanposy[i] + shifty[i];
        int shiftxpix = posx-round(posx);
        int shiftypix = posy-round(posy);
        getWindow(objectWave_t, +round(posx), round(posy), row_O, column_O, window);
        if(fabs(shiftxpix)>1e-3||fabs(shiftypix)>1e-3){
          shiftWave(window, shiftxpix, shiftypix);
        }
        multiply(esw, pupilpatternWave_t, window);
        verbose(5, plt.plotComplex(esw, MOD2, 0, 1, ("ptycho_esw"+to_string(i)).c_str()));
        if(isFresnel) multiplyFresnelPhase(esw, d);
        myFFT(esw,esw);
        if(!patterns[i]) patterns[i] = (Real*)memMngr.borrowCache(sz);
        getMod2(patterns[i], esw, 1./(row*column));
        if(useBS) applySupport(patterns[i], beamstop);
        if(simCCDbit) {
          //applyPoissonNoise_WO(patterns[i], noiseLevel, devstates, 1./exposure);
          ccdRecord(patterns[i], patterns[i], noiseLevel, devstates, exposure);
        }else{
          applyNorm(patterns[i], exposure);
        }
        cudaConvertFO(patterns[i]);
        verbose(2, plt.plotFloat(patterns[i], MOD, 0, 1, (common.Pattern+to_string(i)).c_str()));
        verbose(4, plt.plotFloat(patterns[i], MOD, 0, 1, (common.Pattern+to_string(i)+"log").c_str(),1, 0, 1));
        plt.saveFloat(patterns[i], (common.Pattern+to_string(i)).c_str());
        if(i == 0) {
          extendToComplex(patterns[0],esw);
          myFFT(esw, esw);
          plt.plotComplex(esw, MOD2, 1, 1./exposure/row/column, "autocorrelation", 1, 0, 1);
        }
        cudaConvertFO(patterns[i]);
        applyNorm(patterns[i], 1./exposure);
      }
      memMngr.returnCache(window);
    }
    void initObject(){
      resize_cuda_image(row_O,column_O);
      random(objectWave, devstates);
      resize_cuda_image(row,column);
      pupilFunc(pupilpatternWave);
      angularSpectrumPropagate(pupilpatternWave, pupilpatternWave, beamspotsize*oversampling/lambda, dpupil/lambda, row*column); //granularity is the same
    }
    void updatePosition(int scanidx, complexFormat* obj, complexFormat* probe, Real* pattern, complexFormat* Fn){
      size_t n = row*column;
      Real norm = 1./(n);
      myCuDMalloc(complexFormat, cachex, n);
      myCuDMalloc(complexFormat, cachey, n);
      myCuDMalloc(Real, cache, n);
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
      findSum(cache, n, d_shift+scanidx);
      calcPartial(cache, cachey, Fn, pattern, beamstop);
      findSum(cache, n, d_shift+scanidx+nscan);
      memMngr.returnCache(cachex);
      memMngr.returnCache(cachey);
      memMngr.returnCache(cache);
    }
    void updatePosition_conv(complexFormat* obj, complexFormat* probe, complexFormat* Fn){
      size_t n = row*column;
      Real norm = 1./sqrt(n);
      myCuDMalloc(complexFormat, O2, n);
      myCuDMalloc(complexFormat, P2, n);
      getMod2(O2, obj, norm);
      getMod2(P2, probe, norm);
      myFFT(O2, O2);
      myFFT(P2, P2);
      complexFormat* A = P2;
      multiplyConj(A, P2, O2);
      myIFFT(A, A);
      myCuDMalloc(complexFormat, G, n);
      multiplyConj(G, Fn, obj);
      myFFT(G, G); 
      complexFormat* Phat = O2;
      myFFT(probe, Phat); 
      complexFormat* B = O2;
      multiplyConj(B, Phat, G);
      applyNorm(B, 2*norm*norm);
      myIFFT(B, B);
      add(B, A, -1);
      Real* S = (Real*)A;
      getReal(S,B,norm);
      int pos = findMaxIdx(S, n);
      if(pos != 0) {
        fmt::println("best position: {}", pos);
        plt.plotFloat(S, REAL, 1, 1./62, "S", 0, 0, 1);
      }
      myCuFree(O2);
      myCuFree(P2);
      myCuFree(G);
    }
    void iterate(){
      resetPosition();
      resize_cuda_image(row,column);
      Real objMax;
      Real probeMax;
      complexFormat *Fn = (complexFormat*)memMngr.borrowCache(sz*2);
      myCuDMallocClean(complexFormat, probeStep, row*column);
      complexFormat *objCache = (complexFormat*)memMngr.borrowCache(sz*2);
      myCuDMallocClean(complexFormat, objStep, row_O*column_O);
      Real *maxCache = (Real*)memMngr.borrowCache(max(row_O*column_O/4, row*column)*sizeof(Real));
      myCuDMalloc(complexFormat, cropObj, row_O*column_O/4);
      myCuDMalloc(Real, tmp, row*column);
      Real norm = 1./sqrt(row*column);
      int update_probe_iter = 4;
      int positionUpdateIter = 500;
      void* objFFT;
      createPlan(&objFFT, row_O, column_O); 
      int pupildiameter = pupilSize;
      myCuDMalloc(complexFormat, zernikeCrop, pupildiameter*pupildiameter);
      void* zernike = zernike_init(pupildiameter, pupildiameter, 25, 0); //40 is already high enough for modelling complex beams
      myCuDMalloc(Real, pupilSupport, pupildiameter*pupildiameter);
      resize_cuda_image(pupildiameter, pupildiameter);
      createCircleMask(pupilSupport, Real(pupildiameter+1)/2, Real(pupildiameter+1)/2, Real(pupildiameter)/2);
      resize_cuda_image(row,column);
      Real probeStepSize = 0.20;
      Real objStepSize = 0.10;
      myCuDMalloc(Real, d_norm, 2);
      findSum(tmp, row*column, d_norm);
      myDMalloc(Real, h_norm, 2);
      int vidhandle_pupil = 0;
      int vidhandle_probe = 0;
      int vidhandle_O = 0;

      cuPlotter plt_O;
      plt_O.init(row_O,column_O, outputDir);
      if(saveVideoEveryIter){
        vidhandle_O = plt_O.initVideo("recon_object.mp4",24, 1, online);
        plt_O.toVideo = -1;
        vidhandle_pupil = plt.initVideo("recon_pupil.mp4",24);
        vidhandle_probe = plt.initVideo("recon_probe.mp4",24);
        plt.toVideo = -1;
      }
      myDMalloc(int, iterOrder, nscan);
      for (int i = 0 ; i < nscan ; i++) {
        iterOrder[i] = i;
      }
      for(int iter = 0; iter < nIter; iter++){
        getMod2(maxCache, pupilpatternWave);
        findMax(maxCache, row*column ,d_norm);
        if(iter >= update_probe_iter) {
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
          applyNorm(objStep, sf);
          resize_cuda_image(row,column);
          applyNorm(pupilpatternWave, 1./sf);
          applyNorm(probeStep, 1./sf);
          objMax = probeMax = sqrt(objMax*probeMax);
        }
        //complexFormat* coeff = NULL, *projection = NULL;
        bool doUpdatePosition = iter % 20 == 0 && iter >= positionUpdateIter;

        shuffle_array(iterOrder, nscan);
        for(int ic = 0; ic < nscan; ic++){
          int i = iterOrder[ic];
          Real posx = scanposx[i] + shiftx[i];
          Real posy = scanposy[i] + shifty[i];
          int shiftxpix = posx-round(posx);
          int shiftypix = posy-round(posy);
          getWindow(objectWave, posx, posy, row_O, column_O, objCache);
          bool shiftpix = fabs(shiftxpix)>1e-3||fabs(shiftypix)>1e-3;
          if(shiftpix){
            shiftWave(objCache, shiftxpix, shiftypix);
          }
          multiply(esw, pupilpatternWave, objCache);
          if(isFresnel) multiplyFresnelPhase(esw, d);
          myFFT(esw,Fn);
          if(iter == nIter - 1){
            verbose(4,plt.plotComplex(Fn, MOD2, 1, exposure/(row*column), ("ptycho_recon_pattern" + to_string(i)).c_str(), 1, 0, 1));
          }
          if(doUpdatePosition) {
            applyNorm(Fn, norm);
            updatePosition(i, objCache, pupilpatternWave, patterns[i], Fn);
            applyModAccurate(Fn, patterns[i],beamstop, 1);
          }else applyModAccurate(Fn, patterns[i],beamstop, norm);
          myIFFT(Fn,Fn);
          applyNorm(Fn, norm);
          //if(doUpdatePosition) updatePosition(i, objCache, pupilpatternWave, Fn);
          add(esw, Fn, -1);
          //add(esw, Fn, -norm);
          if(isFresnel) multiplyFresnelPhase(esw, -d);
          if(iter < update_probe_iter) updateObject(objCache, pupilpatternWave, esw, probeMax);
          else {
            //updateObjectAndProbe(objCache, pupilpatternWave, esw,probeMax, objMax);
            if(mPIE){
              updateObjectStepAndProbeStep(objCache, pupilpatternWave, probeStep, esw,probeMax, objMax, probeStepSize);
            }else{
              updateObjectAndProbeStep(objCache, pupilpatternWave, probeStep, esw,probeMax, objMax, probeStepSize);
            }
          }
          if(shiftpix){
            shiftWave(objCache, -shiftxpix, -shiftypix);
          }
          if(mPIE && iter >= update_probe_iter){
            addWindow(objStep, posx, posy, row_O, column_O, objCache, objStepSize);
          }else
            updateWindow(objectWave, posx, posy, row_O, column_O, objCache);
        }
        if(mPIE && iter >= update_probe_iter){
          resize_cuda_image(row_O, column_O);
          applyNorm(objStep, 0.98);
          add(objectWave, objStep, 1);
          //FISTA(objStep, objStep, 3e-3, 1, NULL);
          //FISTA(objectWave, objectWave, 3e-3, 1, NULL);
        }
        resize_cuda_image(row, column);
        if(iter >= update_probe_iter) {
          add(pupilpatternWave, probeStep);
          applyNorm(probeStep, 0.);
        }
        if(doUpdatePosition){
          resize_cuda_image(nscan*2, 1);
          addProbability(d_shifts, d_shift, 0.3, positionUncertainty);
          findSum(d_shifts, nscan, d_norm);
          findSum(d_shifts+nscan, nscan, d_norm+1);
          myMemcpyD2H(h_norm, d_norm, 2*sizeof(Real));
          resize_cuda_image(nscan, 1);
          h_norm[0] /= nscan;
          h_norm[1] /= nscan;
          linearConst(d_shifts, d_shifts, 1, -h_norm[0]);
          linearConst(d_shifts+nscan, d_shifts+nscan, 1, -h_norm[1]);
          resize_cuda_image(row_O, column_O);
          shiftWave(objFFT,objectWave, -h_norm[0], -h_norm[1]);
          shiftWave(objFFT,objStep, -h_norm[0], -h_norm[1]);
          myMemcpyD2H(shifts, d_shifts, nscan*sizeof(Real)*2);
          //for (int iscan = 0; iscan < nscan; iscan++) {
          //  fmt::println("{},{}", shiftx[iscan], shifty[iscan]);
          //}
          //exit(0);
          resize_cuda_image(row, column);
        }
        if(saveVideoEveryIter && iter%saveVideoEveryIter == 0){
          resize_cuda_image(row_O, column_O);
          plt_O.toVideo = vidhandle_O;
          plt_O.plotComplexColor(objectWave, 0, 1, ("recon_object"+to_string(iter)).c_str(), 0, isFlip);
          plt_O.toVideo = -1;
          resize_cuda_image(row, column);
          plt.toVideo = vidhandle_probe;
          plt.plotComplexColor(pupilpatternWave, 0, 1, ("recon_probe"+to_string(iter)).c_str(), 0, isFlip);
          plt.toVideo = -1;
        }
        if(iter >= update_probe_iter){
          complexFormat middle = 0;
          if(iter < 10000){
            getMod2(tmp, (complexFormat*)pupilpatternWave);
            middle = findMiddle(tmp);
          }
          //Real biasx = crealf(middle);
          //Real biasy = cimagf(middle);
          //shiftWave(pupilpatternWave, -biasx*row, -biasy*column);
          //shiftWave(probeStep, -biasx*row, -biasy*column);
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
          crop((complexFormat*)pupilpatternWave, zernikeCrop, row, column);
          if(iter < zernikeIter){
            zernike_compute(zernike, zernikeCrop, Real(pupildiameter-1)/2, Real(pupildiameter-1)/2, Real(pupildiameter)/2);
            zernike_reconstruct(zernike, zernikeCrop, Real(pupildiameter)/2);
          }else{
            //FISTA(zernikeCrop, zernikeCrop, 1e-3, 1, NULL);
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
          if(iter == nIter-1) plt.plotComplexColor(esw, 0, 1./(row), "debug0");
          multiplyShift(pupilpatternWave, crealf(middle)*row-0.5, cimagf(middle)*column-0.5); //freq middle is n/2+1, not n/2+0.5
          if(iter == nIter-1) plt.plotComplexColor(pupilpatternWave, 0, 1, "debug1");
          resize_cuda_image(row_O, column_O);
          //plt.init(row_O, column_O, outputDir);
          //plt.plotComplexColor(objectWave, 0, 1, "debug2");
          //multiplyShift(objectWave, -crealf(middle)*row_O, -cimagf(middle)*column_O);
          //plt.plotComplexColor(objectWave, 0, 1, "debug3");
          resize_cuda_image(row, column);
          //plt.init(row, column, outputDir);
        }
        if(iter == positionUpdateIter && verbose > 5){
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
      if(saveVideoEveryIter){
        plt_O.saveVideo(vidhandle_O);
        plt.saveVideo(vidhandle_probe);
        plt.saveVideo(vidhandle_pupil);
      }
      myCuFree(d_norm);
      myFree(h_norm);
      memMngr.returnCache(Fn);
      memMngr.returnCache(objCache);
      Real crop_ratio = 1;
      int rowc = row_O*crop_ratio;
      int colc = column_O*crop_ratio;
      resize_cuda_image(rowc, colc);
      plt.init(rowc, colc, outputDir);
      complexFormat* cropped = (complexFormat*)memMngr.borrowCache(rowc*colc*sizeof(complexFormat));
      myCuDMalloc(Real, angle, rowc*colc);
      for (int i = 0; i < nscan ; i++) {
        drawCircle(objectWave, scanposx[i]+row/2, scanposy[i]+column/2, pupilSize/2-1, 3, 0);
      }
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
      calculateParameters();
      readScan();
      row_O = column_O = int(stepSize * (sqrt(nscan)*0.8-1.25) + row)/4*4;
      allocateMem();
      resize_cuda_image(row,column);
      if(useBS) {
        createBeamStop();
      }
      for(int i = 0; i < nscan; i++){
        if(i!=0) pattern = readImage((string(outputDir) + common.Pattern+to_string(i)+".bin").c_str(), row, column);
        if(!patterns[i]) patterns[i] = (Real*)memMngr.borrowCache(sz);
        myMemcpyH2D(patterns[i], pattern, sz);
        ccmemMngr.returnCache(pattern);
        cudaConvertFO(patterns[i]);
        applyNorm(patterns[i], 1./exposure);
        verbose(3, plt.plotFloat(patterns[i], MOD, 1, exposure, ("input"+string(common.Pattern)+to_string(i)).c_str()));
      }
      fmt::println("Created pattern data");
    }
    void calculateParameters(){
      resolution = lambda*d/pixelsize/row;
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
  fmt::println("Resolution = {:4.2f}um", setups.resolution);

  fmt::println("pupil Imaging distance = {:4.2f}cm", setups.dpupil*1e-4);
  fmt::println("pupil fresnel factor = {:f}", setups.fresnelFactorpupil);
  setups.initObject();
  setups.iterate();

  return 0;
}

