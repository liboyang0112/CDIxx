#include <cmath>
#include <complex.h>
#include <fmt/base.h>
#include <fmt/os.h>
#include <fstream>
#include <stdio.h>
#include <random>
#include <chrono>
#include <vector>
#include <cstdlib>
#include "imgio.hpp"
#include "cudaConfig.hpp"
#include "readConfig.hpp"
#include "propagator.hpp"
#include "cuPlotter.hpp"
#include "cub_wrap.hpp"
#include "memManager.hpp"
#include "misc.hpp"
#include "ptycho.hpp"
#include "beamDecomposition.hpp"
#include "broadBand.hpp"
#include "material.hpp"

#define verbose(i,a) if(verbose>=i){a;}
#define m_verbose(m,i,a) if(m.verbose>=i){a;}

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

class multi_ptycho : public readConfig, public broadBand_constRatio{
  public:
    int row_O = 1280;  //in ptychography this is different from row (the size of probe).
    int column_O = 1280;
    int sz = 0;
    int doPhaseModulationPupil = 0;
    complexFormat* registerCache = 0;
    void* registerFFTHandle = 0;
    int nscan = 0;
    propagator propagate_pupil;
    propagator propagate_esw;
    material mat;
    Real resolution = 0;
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
    Real *beamstop = 0;
    complexFormat *padded = 0; //used for zooming
    int* widths = 0;
    complexFormat *esw;
    complexFormat** esws;
    complexFormat* objectWave = 0; // stores reconstructed, higher resolution for short wave, C, H, W
    complexFormat** objectWaves = 0; // stores reconstructed, higher resolution for short wave, C, H, W
    complexFormat* objectWave_t = 0; //stores truth, same resolution for all wavelength, C, H, W
    complexFormat* pupilpatternWave_t = 0;
    complexFormat* pupilpatternWave = 0;
    complexFormat** pupilpatternWaves = 0;

    multi_ptycho(const char* configfile):readConfig(configfile), broadBand_constRatio(){
      propagate_esw.lambda = propagate_pupil.lambda = lambda;
      propagate_esw.distance = d;
      propagate_esw.pixelsize = pixelsize;
      propagate_pupil.distance = dpupil;
      init_cuda_image(rcolor, 1./exposure);
    }
    Real computeErrorSim();
    void readScan(){
      std::ifstream file(std::string(outputDir) + distFile);
      if (!file.is_open()) {
        throw std::runtime_error(fmt::format("Could not open file '{}' for reading.", std::string(outputDir) + distFile));
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
      allocScan();
      int offsetx = row/10;
      int offsety = column/8;
      loop(i, nscan){
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
      nscan = scanx*scany;
      int offsetx = -row*0.27;
      int offsety = -column*0.23;
      fmt::println("scanning {} x {} steps", scanx, scany);
      allocScan();
      fmt::ostream scanFile(fmt::output_file(std::string(outputDir) + "scan.txt"));
      scanFile.print("{}\n", stepSize*resolution);
      int iscan = 0;
      loop(i, scanx){
        loop(j, scany){
          //if(i == 2 && j == 1) continue;
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
      fmt::ostream scanFile(fmt::output_file(std::string(outputDir) + "scan.txt"));
      scanFile.print("{}\n", stepSize*resolution);
      loop(i, N){
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
      myCuMalloc(complexFormat, objectWave, row_O*column_O*nlambda);
      myCuMalloc(complexFormat, objectWave_t, row_O*column_O*nlambda);
      esw = (complexFormat*) memMngr.borrowCache(sz*nlambda*2);
      myMalloc(complexFormat*, objectWaves, nlambda);
      myMalloc(complexFormat*, pupilpatternWaves, nlambda);
      myMalloc(complexFormat*, esws, nlambda);
      myMalloc(int, widths, nlambda);
      pupilpatternWave = (complexFormat*)memMngr.borrowCache(sz*nlambda*2);
      loop(i, nlambda){
        widths[i] = pupilSize / lambdas[i]; 
        objectWaves[i] = objectWave + row_O*column_O*i;
        pupilpatternWaves[i] = pupilpatternWave + row*column*i;
        esws[i] = esw + i*row*column;
      }
      pupilpatternWave_t = (complexFormat*)memMngr.borrowCache(sz*nlambda*2);
      propagate_esw.row = propagate_pupil.row = row;
      propagate_esw.column = propagate_pupil.column = column;
      propagate_pupil.pixelsize = resolution;
      fmt::println("initializing cuda image");
      resize_cuda_image(row_O,column_O);
      unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
      initRand(devstates,seed);
      if(runSim && positionUncertainty > 1e-4){
        initPosition();
      }
    }
    void readPupilAndObject(){
      Image3 rgbimg = readImage3(common.Intensity, nullptr);
      row_O = rgbimg.rows;
      column_O = rgbimg.cols;
      myCuDMalloc(Real, rgbimg_d, row_O*column_O*3);
      myMemcpyH2D(rgbimg_d, rgbimg.r, row_O*column_O*sizeof(Real)*3);
      Real* pupil_intensity = readImage(pupil.Intensity, row, column);
      skip = 0;
      jump = 22;
      fmt::println("imgsize = {}x{}", row, column);

      int lambdarange = 2;
      int nlambdai = row*(lambdarange-1)/2;
      double* lambdasi = (double*)ccmemMngr.borrowCache(sizeof(double)*nlambdai);
      double* spectrai = (double*)ccmemMngr.borrowCache(sizeof(double)*nlambdai);
      for(int i = 0; i < nlambdai; i++){
        lambdasi[i] = 1 + 2.*i/row;
        spectrai[i] = exp(-pow(2*(i*2./nlambdai-1),2))/nlambdai; //gaussian, -1,1 with sigma=1
      }
      broadBand_constRatio::init(row, column, lambdasi, spectrai, nlambdai, true);
      writeSpectra("spectrum.txt");
      //spectra[0] = spectra[1] = spectra[2] = spectra[3] = 0.24;
      loop(i, nlambda){
        //spectra[i] = 1./(nlambda*lambdas[i]);
        spectra[i] = 1./(nlambda);
      }
      //Real sum = 0;
      //loop(i, nlambda){
      //  sum += spectra[i];
      //}
      //loop(i, nlambda){
      //  spectra[i] = spectra[i]/sum;
      //}

      //broadBand_constRatio::init(row, column, 1, 2);
      fmt::println("nlambda = {}", nlambda);
      std::vector<std::string> fnames;
      fnames.push_back(std::getenv("CDI_DIR") + std::string("/data/H2O.dat"));
      fnames.push_back(std::getenv("CDI_DIR") + std::string("/data/C.dat"));
      fnames.push_back(std::getenv("CDI_DIR") + std::string("/data/N.dat"));
      mat.init(fnames , lambdas, nlambda, lambda);
      sz = row*column*sizeof(Real);
      resolution = lambda*d/pixelsize/row;
      initScan_triangle();
      allocateMem();
      resize_cuda_image(row_O, column_O);
      mat.Transmission(objectWave_t, rgbimg_d, row_O*column_O, 2);
      verbose(2,
          resize_cuda_image(row_O,column_O);
          plt.init(row_O,column_O, outputDir);
          loop(i, nlambda)
          plt.plotComplexColor(objectWave_t + i*row_O*column_O, 0, 1, ("inputObject" + to_string(i)).c_str());
          //plt.plotPhase(objectWave_t, PHASERAD, 0, 1, "inputPhase");
          );
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
      pupilpatternWave_t = (complexFormat*)memMngr.borrowCache(row*column*sizeof(complexFormat)*nlambda);
      resize_cuda_image(row,column);
      createWaveFront(d_intensity, d_phase, pupilpatternWave_t, 1);
      memMngr.returnCache(d_intensity);
      if(d_phase) memMngr.returnCache(d_phase);
      plt.init(row,column, outputDir);
      plt.plotComplexColor(pupilpatternWave_t, 0, 1, "pupilWave", 0);
      init_fft(row,column);
      fmt::println("Wavelengths:");
      fmt::println("{} {} = {}", 0, lambda, spectra[0]);
      for(int i = 1; i < nlambda; i++){ // assume all pupil functions are the same
        complexFormat* cur_pat = pupilpatternWave_t+i*row*column;
        myMemcpyD2D(cur_pat, pupilpatternWave_t, row*column*sizeof(complexFormat));
        applyNorm(cur_pat, spectra[i]);
        propagate_pupil.lambda = lambdas[i]*lambda;
        propagate_pupil.angularSpectrumPropagate(cur_pat, cur_pat);
        fmt::println("{} {} = {}", i, lambdas[i]*lambda, spectra[i]);
      }
      applyNorm(pupilpatternWave_t, spectra[0]);
      propagate_pupil.lambda = lambdas[0]*lambda;
      propagate_pupil.angularSpectrumPropagate(pupilpatternWave_t, pupilpatternWave_t);
      plt.plotComplexColor(pupilpatternWave_t, 0, 1, "probeWave", 0);
    }
    void initPosition(){
      unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
      std::default_random_engine generator(seed);
      std::normal_distribution<double> distribution(0.0, 1.);
      loop(i, nscan){
        shiftx[i]+= distribution(generator)*positionUncertainty;
        shifty[i]+= distribution(generator)*positionUncertainty;
        if(verbose >=3 ) fmt::println("shifts {}: ({:f}, {:f})", i, shiftx[i],shifty[i]);
      }
    }
    void resetPosition(){
      loop(i, nscan){
        shiftx[i] = shifty[i] = 0;
      }
    }
    void createPattern(){
      if(useBS) {
        beamstop = createBeamStop(row,column,beamStopSize);
        plt.plotFloat(beamstop, MOD, 1, 1,"beamstop", 0);
      }
      complexFormat* window = (complexFormat*)memMngr.borrowCache(sz*2);
      myCuMalloc(complexFormat, padded, row*column*lambdas[nlambda-1]*lambdas[nlambda-1]);
      loop(i, nscan){
        Real posx = scanposx[i] + shiftx[i];
        Real posy = scanposy[i] + shifty[i];
        int shiftxpix = posx-round(posx);
        int shiftypix = posy-round(posy);
        if(!patterns[i]) patterns[i] = (Real*)memMngr.borrowCache(sz);
        loop(il, nlambda){
          getWindow(objectWave_t + row_O*column_O*il, round(posx), round(posy), row_O, column_O, window);
          if(fabs(shiftxpix)>1e-3||fabs(shiftypix)>1e-3){
            shiftWave(window, shiftxpix, shiftypix);
          }
          multiply(esw, pupilpatternWave_t + row*column*il, window);

          verbose(5, plt.plotComplexColor(esw, 0, 1./spectra[i], ("ptycho_esw"+to_string(i) + "_" + to_string(il)).c_str()));
          if(isFresnel) {
            propagate_esw.lambda = lambdas[il]*lambda;
            propagate_esw.multiplyFresnelPhase(esw);
          }
          int thisrow = row*lambdas[il];
          int thiscol = column*lambdas[il];
          resize_cuda_image(thisrow, thiscol);
          pad(esw, padded, row, column);
          myFFTM(locplan[il], padded, padded);
          resize_cuda_image(row, column);
          cropinner(padded, esw, thisrow, thiscol);
          if(il == 0) getMod2(patterns[i], esw, 1./(row*column));
          else addMod2(patterns[i], esw, 1./(row*column));
        }

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
      loop(i, nlambda){
        random(objectWaves[i], devstates);
      }
      resize_cuda_image(row,column);
      loop(i, nlambda){
        pupilFunc(pupilpatternWaves[i], Real(row>>2)/lambdas[i]);
        propagate_pupil.pixelsize = resolution*lambdas[i];
        propagate_pupil.lambda = lambdas[i]*lambda;
        propagate_pupil.angularSpectrumPropagate(pupilpatternWaves[i], pupilpatternWaves[i]);
      }
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
      myDMalloc(Real, objMax, nlambda);
      myDMalloc(Real, probeMax, nlambda);
      myCuDMallocClean(complexFormat, probeStep, row*column*nlambda);
      myCuDMallocClean(complexFormat, objStep, row_O*column_O*nlambda);
      myCuDMallocClean(complexFormat, objectWave_prev, row_O*column_O*nlambda);
      complexFormat *Fn = (complexFormat*)memMngr.borrowCache(sz*2*nlambda);
      myDMalloc(complexFormat*, Fns, nlambda);
      myDMalloc(complexFormat*, objSteps, nlambda);
      myDMalloc(complexFormat*, objectWave_prevs, nlambda);
      myDMalloc(complexFormat*, probeSteps, nlambda);
      complexFormat *objCache = (complexFormat*)memMngr.borrowCache(sz*2*nlambda);
      myDMalloc(complexFormat*, objCaches, nlambda);
      Real *maxCache = (Real*)memMngr.borrowCache(max(row_O*column_O/4, row*column)*sizeof(Real));
      myCuDMalloc(complexFormat, cropObj, row_O*column_O/4);
      myCuDMalloc(Real, tmp, row*column);
      Real norm = 1./sqrt(row*column);
      int update_probe_iter = 4;
      int positionUpdateIter = 50000;
      void* objFFT;
      createPlan(&objFFT, row_O, column_O); 
      myCuDMalloc(complexFormat, zernikeCrop, pupilSize*pupilSize);
      void** zernike = zernike_init_group(widths, 25, nlambda);
      myCuDMalloc(Real, pupilSupport, row*column*nlambda);
      myCuDMalloc(Real, patternSum, row*column);
      loop(i, nlambda){
        objCaches[i] = objCache + i*row_O*column_O;
        objSteps[i] = objStep + i*row_O*column_O;
        objectWave_prevs[i] = objectWave_prev + i*row_O*column_O;
        probeSteps[i] = probeStep + i*row*column;
        Fns[i] = Fn + i*row*column;
        createCircleMask(pupilSupport + row*column*i, Real(row+1)/2, Real(column+1)/2, Real(pupilSize)/(2*lambdas[i]));
      }
      resize_cuda_image(row,column);
      Real probeStepSize = 0.20;
      Real objStepSize = 0.40;
      bool doprl = 0;
      void **prlplan;
      if(doprl){
        myMallocClean(void*, prlplan, nlambda);
      }
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
      loop(i, nscan){
        iterOrder[i] = i;
      }
      createCircleMask(tmp, row>>1, column>>1 , pupilSize/2);
      Real* masksum = (Real*)objStep;
      myCuDMalloc(Real, probesum, row_O*column_O);
      loop(i, nscan) {
        addWindow(probesum, scanposx[i], scanposy[i], row_O, column_O, tmp, 1);
      }
      Real maxOverlap = findMax(probesum, row_O*column_O);
      resize_cuda_image(row_O,column_O);
      plt_O.plotFloat(probesum, MOD, 0, 1./maxOverlap, "masksum", 0, 0, 1);
      bitMap(masksum, probesum, 0.5);
      Real redundancy = findSum(masksum, row_O*column_O);
      redundancy = (nscan*M_PI*pupilSize*pupilSize)/(4*redundancy);
      fmt::println("Max overlap = {}, Redundancy = {}", maxOverlap, redundancy - 1);
      clearCuMem(masksum, row_O*column_O);
      Real tk = 0.5+sqrt(1.25);
      Real tkp1;
      Real residual = 0, residual_prev = 0;
      int momentum_tolerance = 0;
      loop(iter, nIter){
        loop(il, nlambda){
          resize_cuda_image(row, column);
          getMod2(maxCache, pupilpatternWaves[il]);
          findMax(maxCache, row*column ,d_norm);
          if(iter >= update_probe_iter) {
            resize_cuda_image(row_O>>4,column_O>>4);
            crop(objectWaves[il], cropObj, row_O, column_O);
            getMod2(maxCache, cropObj);
            findMax(maxCache, row_O*column_O/256, d_norm+1);
            myMemcpyD2H(h_norm, d_norm, 2*sizeof(Real));
            objMax[il] = h_norm[1];
            resize_cuda_image(row_O,column_O);
            applyThreshold(objectWaves[il], objectWaves[il], objMax[il]);
            resize_cuda_image(row,column);
          }else{
            myMemcpyD2H(h_norm, d_norm, sizeof(Real));
          }
          probeMax[il] = h_norm[0];
          if(iter >= update_probe_iter){
            resize_cuda_image(row_O,column_O);
            Real sf = pow(probeMax[il]/objMax[il], 0.25);
            applyNorm(objectWaves[il], sf);
            applyNorm(objectWave_prevs[il], sf);
            resize_cuda_image(row,column);
            applyNorm(pupilpatternWaves[il], 1./sf);
            applyNorm(probeSteps[il], 1./sf);
            objMax[il] = probeMax[il] = sqrt(objMax[il]*probeMax[il]);
          }
        }
        //complexFormat* coeff = NULL, *projection = NULL;
        bool doUpdatePosition = iter % 20 == 0 && iter >= positionUpdateIter;

        shuffle_array(iterOrder, nscan);
        residual_prev = residual;
        residual = 0;
        loop(ic, nscan){
          int i = iterOrder[ic];
          Real posx0 = scanposx[i] + shiftx[i];
          Real posy0 = scanposy[i] + shifty[i];
          loop(il , nlambda){ // compute all diffraction patterns
            Real posx = posx0 / lambdas[il];
            Real posy = posy0 / lambdas[il];
            int shiftxpix = posx-round(posx);
            int shiftypix = posy-round(posy);
            bool shiftpix = fabs(shiftxpix)>1e-3||fabs(shiftypix)>1e-3;
            getWindow(objectWaves[il], round(posx), round(posy), row_O, column_O, objCaches[il]);
            if(shiftpix){
              shiftWave(objCaches[il], shiftxpix, shiftypix);
            }
            multiply(esws[il], pupilpatternWaves[il], objCaches[il]);
            if(isFresnel) {
              propagate_esw.lambda = lambdas[il]*lambda;
              propagate_esw.multiplyFresnelPhase(esws[il]);
            }
            myFFT(esws[il],Fns[il]);
          }
          // ------------- sum patterns and compare to detector data ----------------
          getMod2(patternSum, Fns[0]);
          for(int il = 1; il < nlambda; il++){
            addMod2(patternSum, Fns[il]);
          }
          if(iter == nIter - 1){
            verbose(4,plt.plotFloat(patternSum, MOD, 1, exposure/(row*column), ("ptycho_recon_pattern" + to_string(i)).c_str(), 1, 0, 1));
          }
          applyNorm(patternSum, norm*norm);
          addRemoveOE( tmp, patternSum, patterns[i], -1);
          getMod2(tmp, tmp);
          residual += findSum(tmp) / (row*column);
          //sqrtdivide(patternSum, patternSum, patterns[i]);

          loop(il , nlambda){
            Real posx = posx0 / lambdas[il];
            Real posy = posy0 / lambdas[il];
            int shiftxpix = posx-round(posx);
            int shiftypix = posy-round(posy);
            bool shiftpix = fabs(shiftxpix)>1e-3||fabs(shiftypix)>1e-3;
            //multiply(Fns[il], Fns[il], patternSum);
            //applyNorm(Fns[il], norm*norm);
            //applyModAccurate(Fns[il], patternSum, patterns[i],beamstop, norm);
            applyMod(Fns[il], patternSum, patterns[i],beamstop, noiseLevel, norm);
            applyNorm(Fns[il], norm);
            myIFFT(Fns[il],Fns[il]);
            //if(doUpdatePosition) updatePosition(i, objCache, pupilpatternWave, Fn);
            add(esws[il], Fns[il], -1);
            if(isFresnel) {
              propagate_esw.lambda = lambdas[il]*lambda;
              propagate_esw.removeFresnelPhase(esws[il]);
            }
            if(iter < update_probe_iter){
              if(mPIE){
                updateObjectStep(objCaches[il], pupilpatternWaves[il], esws[il], probeMax[il]);
              }else{
                updateObject(objCaches[il], pupilpatternWaves[il], esws[il], probeMax[il]);
              }
            } 
            else {
              if(mPIE){
                updateObjectStepAndProbeStep(objCaches[il], pupilpatternWaves[il], probeSteps[il], esws[il],probeMax[il], objMax[il], probeStepSize);
              }else{
                updateObjectAndProbeStep(objCaches[il], pupilpatternWaves[il], probeSteps[il], esws[il],probeMax[il], objMax[il], probeStepSize);
              }
            }
            if(shiftpix){
              shiftWave(objCaches[il], -shiftxpix, -shiftypix);
            }
            if(mPIE){
              addWindow(objSteps[il], round(posx), round(posy), row_O, column_O, objCaches[il], objStepSize);
            }else{
              updateWindow(objectWaves[il], round(posx), round(posy), row_O, column_O, objCaches[il]);
            }
          }
        }
        if(residual>residual_prev) {
          momentum_tolerance++;
          //if(momentum_tolerance>0){
            if(mPIE) myMemcpyD2D(objectWave_prev, objectWave, nlambda*row_O*column_O*sizeof(complexFormat));
            if(momentum_tolerance > 4 && nIter > iter+2){
              nIter = iter+2;
            }
          //}
        }else{
          momentum_tolerance = 0;
        }
        //if(iter%20 == 0)
          fmt::println("residual = {}", residual/nscan);
        if(mPIE){
          resize_cuda_image(row_O*column_O*nlambda, 1);
          add(objectWave, objStep, 2./maxOverlap); //x_k
          add(objectWave_prev, objectWave, objectWave_prev, -1);
          tkp1 = 0.5+sqrt(0.25+tk*tk);
          add(objectWave_prev, objectWave, objectWave_prev, 0.5*(tk-1)/tkp1);
          tk = tkp1;
          complexFormat* tmp;
          tmp = objectWave; objectWave = objectWave_prev; objectWave_prev = tmp;
          applyNorm(objStep, 0);
        }
        if(iter >= update_probe_iter) {
          resize_cuda_image(row*column*nlambda, 1);
          add(pupilpatternWave, probeStep, 1./nscan);
          //clearCuMem(probeStep, row*column*nlambda*sizeof(complexFormat));
          applyNorm(probeStep, 0);//(iter-update_probe_iter)/(iter-update_probe_iter+3));
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
          loop(il, nlambda){
            shiftWave(objFFT,objectWaves[il], -h_norm[0], -h_norm[1]);
            shiftWave(objFFT,objectWave_prevs[il], -h_norm[0], -h_norm[1]);
          }
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
          plt_O.plotComplexColor(objectWaves[nlambda>>1], 0, sqrt(nlambda), ("recon_object"+to_string(iter)).c_str(), 0, isFlip);
          plt_O.toVideo = -1;
          resize_cuda_image(row, column);
          plt.toVideo = vidhandle_probe;
          plt.plotComplexColor(pupilpatternWaves[nlambda>>1], 0, sqrt(nlambda), ("recon_probe"+to_string(iter)).c_str(), 0, isFlip);
          plt.toVideo = -1;
        }
        if(iter >= update_probe_iter){
          complexFormat middle = 0;
          resize_cuda_image(row, column);
          if(doprl){
            int midlambda = nlambda>>1;
            propagate_pupil.pixelsize = resolution*lambdas[midlambda];
            propagate_pupil.lambda = lambdas[midlambda]*lambda_ref;
            propagate_pupil.angularSpectrumPropagateReverse(pupilpatternWaves[midlambda], pupilpatternWaves[midlambda]);
            loop(il, nlambda){
              if(il == midlambda) continue;
              thisrow = row*lambdas[il] / midlambda;
              thiscol = column*lambdas[il] / midlambda;
              if(!prlplan[il]) createPlan(prlplan + il, thisrow, thiscol);
              resize_cuda_image(thisrow, thiscol);
              if(il > midlambda){
                pad(pupilpatternWaves[midlambda], padded, row, column);
                myFFTM(prlplan[il], padded, padded);
                resize_cuda_image(row, column);
                cropinner(padded, pupilpatternWaves[il], thisrow, thiscol, 1./(row * column));
              }else{
                crop(pupilpatternWaves[midlambda], padded, row, column);
                myFFTM(prlplan[il], padded, padded);
                resize_cuda_image(row, column);
                padinner(padded, pupilpatternWaves[il], thisrow, thiscol, 1./(row * column));
              }
              myIFFT(pupilpatternWaves[il], pupilpatternWaves[il]);
              propagate_pupil.pixelsize = resolution*lambdas[il];
              propagate_pupil.lambda = lambdas[il]*lambda_ref;
              propagate_pupil.angularSpectrumPropagate(pupilpatternWaves[il], pupilpatternWaves[il]);
            }

          }else
          loop(il, nlambda){
            propagate_pupil.pixelsize = resolution*lambdas[il];
            propagate_pupil.lambda = lambdas[il]*lambda;
            propagate_pupil.angularSpectrumPropagateReverse(pupilpatternWaves[il], pupilpatternWaves[il]);
            if(saveVideoEveryIter && iter%saveVideoEveryIter == 0 && il == nlambda>>1){
              plt.toVideo = vidhandle_pupil;
              plt.plotComplexColor(pupilpatternWaves[il], 0, sqrt(nlambda), ("recon_pupil"+to_string(iter)+"_"+to_string(il)).c_str(), 0, isFlip);
              plt.toVideo = -1;
            }
            if(iter == zernikeIter - 1 || (iter <zernikeIter && iter == nIter-1)) {
              plt.plotComplexColor(pupilpatternWaves[il], 0, sqrt(nlambda), ("recon_pupil_b4_proj"+to_string(il)).c_str());
            }
            if(iter < zernikeIter){
              resize_cuda_image(widths[il], widths[il]);
              crop((complexFormat*)pupilpatternWaves[il], zernikeCrop, row, column);
              zernike_compute(zernike[il], zernikeCrop, Real(widths[il]-1)/2, Real(widths[il]-1)/2, Real(widths[il])/2);
              zernike_reconstruct(zernike[il], zernikeCrop, Real(widths[il])/2);
              resize_cuda_image(row, column);
              pad(zernikeCrop, pupilpatternWaves[il], widths[il], widths[il]);
            }else{
              //FISTA(zernikeCrop, zernikeCrop, 1e-3, 1, NULL);
              applyMask(pupilpatternWaves[il], pupilSupport + il*row*column);
            }
            if(iter == zernikeIter - 1 || (iter <zernikeIter && iter == nIter-1)) {
              plt.plotComplexColor(pupilpatternWaves[il], 0, sqrt(nlambda), ("recon_pupil_proj"+to_string(il)).c_str());
            }
            propagate_pupil.pixelsize = resolution*lambdas[il];
            propagate_pupil.lambda = lambdas[il]*lambda;
            propagate_pupil.angularSpectrumPropagate(pupilpatternWaves[il], pupilpatternWaves[il]);
            myFFT(pupilpatternWaves[il], esws[il]); //just reuse esw, instread of allocating new memory
            cudaConvertFO(esws[il]);
            getMod2(tmp, esws[il]);
            middle = findMiddle(tmp);
            multiplyShift(pupilpatternWaves[il], crealf(middle)*row-0.5, cimagf(middle)*column-0.5); //freq middle is n/2+1, not n/2+0.5
          }
        }
        if(iter == positionUpdateIter && verbose > 5){
          resize_cuda_image(row_O,column_O);
          plt_O.plotComplex(objectWave, MOD2, 0, 1, "ptycho_b4position", 0);
          plt_O.plotComplex(objectWave, PHASE, 0, 1, "ptycho_b4positionphase", 0);
          resize_cuda_image(row,column);
        }
      }
      loop(il, nlambda){
        plt.plotComplexColor(pupilpatternWaves[il], 0, sqrt(nlambda), "ptycho_probe_afterIter", 1);
        propagate_pupil.pixelsize = resolution*lambdas[il];
        propagate_pupil.lambda = lambdas[il]*lambda;
        propagate_pupil.angularSpectrumPropagateReverse(pupilpatternWaves[il], pupilpatternWaves[il]);
        plt.plotComplexColor(pupilpatternWaves[il], 0, sqrt(nlambda), "recon_pupil");
      }
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
      loop(il, nlambda){
      complexFormat* cropped = (complexFormat*)memMngr.borrowCache(rowc*colc*sizeof(complexFormat));
      myCuDMalloc(Real, angle, rowc*colc);
      //for (int i = 0; i < nscan ; i++) {
      //  drawCircle(objectWaves[il], scanposx[i] / lambdas[il]+row/2, scanposy[i]/ lambdas[il]+column/2, (pupilSize/ lambdas[il])/2-1, 3, 0);
      //}
      crop(objectWaves[il], cropped, row_O, column_O);
      //getArg(angle, cropped);
      applyNorm(angle, 1./(2*M_PI));
      //plt.plotComplex(cropped, MOD2, 0, 0.7, ("ptycho_afterIter" + to_string(il)).c_str());
      //plt.plotPhase(cropped, PHASERAD, 0, 1, "ptycho_afterIterphase");
      //phaseUnwrapping(angle, angle, rowc, colc);
      //plt.plotFloat(angle, REAL, 0, 1, ("ptycho_afterIterphase" + to_string(il)).c_str());
      complexFormat sum = findSum(cropped);
      sum /= hypot(crealf(sum), cimagf(sum));
      multiplyConj(cropped, cropped, sum);
      plt.plotComplexColor(cropped, 0, sqrt(nlambda), ("ptycho_afterIterwave" + to_string(il)).c_str());
      }
    }
    void readPattern(){
      Real* pattern = readImage((outputDir + string(common.Pattern)+"0.bin").c_str(), row, column);
      plt.init(row,column, outputDir);
      init_fft(row,column);
      sz = row*column*sizeof(Real);
      resolution = lambda*d/pixelsize/row;
      readScan();
      row_O = column_O = int(stepSize * (sqrt(nscan)*0.8-1.25) + row)/32*32;
      fmt::println("object size: {} x {}.", row_O, column_O);
      allocateMem();
      resize_cuda_image(row,column);
      if(useBS) {
        beamstop = createBeamStop(row,column,beamStopSize);
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
      broadBand_constRatio::init_flatspectrum(row, column, 2, true);
    }
};
Real multi_ptycho::computeErrorSim(){
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
  multi_ptycho setups(argv[1]);
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
  fmt::println("Resolution = {:4.2f}um", setups.resolution);

  fmt::println("pupil Imaging distance = {:4.2f}cm", setups.dpupil*1e-4);
  setups.initObject();
  setups.iterate();

  return 0;
}

