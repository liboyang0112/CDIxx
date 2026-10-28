#include <cmath>
#include <complex.h>
#include <chrono>
#include <cstring>
#include <fmt/base.h>
#include <fmt/os.h>
#include "fmt/core.h"
#include "imgio.hpp"
#include "cudaConfig.hpp"
#include "readConfig.hpp"
#include "propagator.hpp"
#include "cuPlotter.hpp"
#include "cub_wrap.hpp"
#include "memManager.hpp"
#include "misc.hpp"
#include "ptycho.hpp"
#include "tvFilter.hpp"
#include "beamDecomposition.hpp"

#define verbose(i,a) if(verbose>=i){a;}
#define m_verbose(m,i,a) if(m.verbose>=i){a;}

using namespace std;

class ssptycho : public readConfig{
  public:
    int row = 2048;
    int column = 2048;
    int sz = 0;
    int split = 3;
    Real *masksum = 0;
    Real *beamstop = 0;
    Real I2 = 0;

    propagator propagate_pupil;
    propagator propagate_esw;
    propagator propagate_complete;

    Real resolution = 0;
    complexFormat* objectWave = 0;
    complexFormat* objectWave_t = 0;
    complexFormat* pupilpatternWave_t = 0;
    complexFormat* pupilpatternWave = 0; // Structured probe at sample plane

    // Grating-based probe members
    complexFormat* grating = 0;          // Grating pattern in pupil plane
    complexFormat* probeWave = 0;        // Probe wave at sample plane (plane wave * grating)
    int nPeaks = 0;                      // Number of detected peaks
    Real* peakEnergies = 0;              // Peak energies (converted to amplitude)
    std::vector<PeakPos> peaks;          // Refined peak positions (normalized)
    std::vector<Real> energies;          // Peak energies
    std::vector<Real*> ratios;

    Real* pattern = 0; // Measured single-shot pattern
    void* devstates = 0;


    ssptycho(const char* configfile):readConfig(configfile){
      propagate_pupil.distance = dpupil;
      propagate_complete.lambda = propagate_esw.lambda = propagate_pupil.lambda = lambda;
      propagate_complete.distance = d+dpupil;
      propagate_esw.distance = d;
      propagate_esw.pixelsize = pixelsize;
      // pixelsize set in allocateMem() after resolution is computed
    }

    void allocateMem(){
      fmt::println("resolution={}", resolution);
      if(devstates) return;
      sz = row*column*sizeof(Real);
      objectWave = (complexFormat*)memMngr.borrowCache(row*column*sizeof(Real)*2);
      objectWave_t = (complexFormat*)memMngr.borrowCache(row*column*sizeof(Real)*2);
      pupilpatternWave_t = (complexFormat*)memMngr.borrowCache(row*column*sizeof(complexFormat));
      pupilpatternWave = (complexFormat*)memMngr.borrowCache(row*column*sizeof(complexFormat));
      pattern = (Real*)memMngr.borrowCache(sz);
      propagate_complete.row = propagate_esw.row = propagate_pupil.row = row;
      propagate_complete.column = propagate_esw.column = propagate_pupil.column = column;
      propagate_pupil.pixelsize = resolution;
      propagate_complete.pixelsize = lambda*(d+dpupil)/(row*resolution);
      resize_cuda_image(row, column);
      init_cuda_image(rcolor, 1./exposure);
      unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
      devstates = newRand(column * row);
      initRand(devstates, seed);
    }

    // Recover the grating from peak positions and energies via direct plane wave summation
    void recoverGrating(){
      nPeaks = peaks.size();
      if(nPeaks == 0) return;

      // Allocate grating and peak parameter arrays
      resize_cuda_image(row, column);
      grating = (complexFormat*)memMngr.borrowCleanCache(row*column*sizeof(complexFormat));
      peakEnergies = (Real*)ccmemMngr.borrowCache(nPeaks*sizeof(Real));

      // Copy energies and compute amplitudes (sqrt of normalized energy)
      Real totalE = 0;
      for(int i = 0; i < nPeaks; i++) totalE += energies[i];
      for(int i = 0; i < nPeaks; i++){
        peakEnergies[i] = sqrt(energies[i]/totalE);
      }

      // Zero-init, then sum plane waves for each peak
      complexFormat* pw = (complexFormat*)memMngr.borrowCache(row*column*sizeof(complexFormat));
      for(int i = 0; i < nPeaks; i++){
        createPlaneWave(pw, peaks[i].x*row, peaks[i].y*column, peakEnergies[i]);
        add(grating, pw);
      }
      memMngr.returnCache(pw);

      // Plot the recovered grating
      plt.init(row, column, outputDir);
      plt.plotComplexColor(grating, 0, 1, "recon_grating");
    }

    void initIteration(){
      I2 = findSum(pattern)/(row*column);
      if(nPeaks > 0 && grating){
        resize_cuda_image(row, column);
        probeWave = (complexFormat*)memMngr.borrowCleanCache(row*column*sizeof(complexFormat));
        myCuDMalloc(Real, tmp, row*column);
        createCircleMask(tmp, row>>1, column>>1, pupilSize/2);
        extendToComplex(tmp, probeWave);
        myCuFree(tmp);
        multiply(pupilpatternWave, probeWave, grating);
      }else{
        pupilFuncSS(pupilpatternWave, row>>2, pupilSize, 3);
      }
      if(cancelFresnel) {
        propagate_complete.removeFresnelPhase(pupilpatternWave);
      }
      propagate_pupil.angularSpectrumPropagate(pupilpatternWave, pupilpatternWave);

      myCuMalloc(Real, masksum, row*column);
      getMod2(masksum, pupilpatternWave);
      bitMap(masksum, masksum, findMax(masksum)/1000);
      plt.plotFloat(masksum, MOD, 0, 1, "masksum");

      plt.init(row, column, outputDir);
      plt.plotComplexColor(probeWave, 0, 1, "init_probe_wave");
      plt.plotComplexColor(pupilpatternWave, 0, 1, "init_probe_pattern");
      random(objectWave, devstates);
    }

    void initPupilAndObject(){
      Real* d_object_intensity = 0;
      Real* d_object_phase = 0;
      int objrow, objcol;
      readComplexWaveFront(common.Intensity, common.Phase, d_object_intensity, d_object_phase, objrow, objcol);

      row = objrow*oversampling;
      column = objcol*oversampling;

      resize_cuda_image(row, column);
      resolution = lambda*d/pixelsize/row;
      allocateMem();
      createWaveFront(d_object_intensity, d_object_phase, objectWave_t, oversampling);
      // Plot object
      resize_cuda_image(row, column);
      plt.init(row, column, outputDir);
      plt.plotComplexColor(objectWave_t, 0, 1, "input_object_wave");
      int r_img, c_img;
      complexFormat* h_pupil = (complexFormat*)readImage(pupil.Intensity, r_img, c_img);
      if(r_img != row || c_img != column){
        fmt::println("Warning: image.bin size ({}x{}) does not match expected size ({}x{})", r_img, c_img, row, column);
        myCuDMalloc(complexFormat, d_pupil, r_img*c_img);
        myMemcpyH2D(d_pupil, h_pupil, sz*2);
        if(r_img > row){
          crop(d_pupil, pupilpatternWave_t, r_img, c_img);
        }else{
          pad(d_pupil, pupilpatternWave_t, r_img, c_img);
        }
        myCuFree(d_pupil);
      }else{
        myMemcpyH2D(pupilpatternWave_t, h_pupil, sz*2);
      }
      ccmemMngr.returnCache(h_pupil);
      plt.plotComplexColor(pupilpatternWave_t, 0, 1, "input_pupil_wave");
      init_fft(row, column);
      if(cancelFresnel){
        propagate_complete.removeFresnelPhase(pupilpatternWave_t);
      }
      propagate_pupil.angularSpectrumPropagate(pupilpatternWave_t, pupilpatternWave_t);
      plt.plotComplexColor(pupilpatternWave_t, 0, 1, "input_probe_wave");
    }

    void readPattern(){
      // Read measured diffraction pattern from experiment
      Real* h_pat = (Real*)readImage((outputDir + string(common.Pattern) + ".bin").c_str(), row, column);
      resolution = lambda*d/pixelsize/row;
      allocateMem();
      sz = row * column * sizeof(Real);
      myMemcpyH2D(pattern, h_pat, sz);
      ccmemMngr.returnCache(h_pat);
      resize_cuda_image(row, column);
      init_fft(row, column);
      plt.init(row, column, outputDir);
      plt.plotFloat(pattern, MOD, 0, 1, "input_pattern", 1, 0, 1);
      cudaConvertFO(pattern, pattern, 1./exposure);
    }

    void createPattern(){
      if(useBS) {
        beamstop = createBeamStop(row,column,beamStopSize);
        plt.plotFloat(beamstop, MOD, 1, 1,"beamstop", 0);
      }
      complexFormat *Fn = (complexFormat*)memMngr.borrowCache(sz*2);
      multiply(Fn, pupilpatternWave_t, objectWave_t);
      if(isFresnel || cancelFresnel) {
        propagate_esw.multiplyFresnelPhase(Fn);
      }
      plt.plotComplexColor(Fn, 0, 1, "simulated_esw");
      myFFT(Fn, Fn);
      getMod2(pattern, Fn, exposure/(row*column));
      cudaConvertFO(pattern);
      if(simCCDbit) ccdRecord(pattern, pattern, noiseLevel, devstates);
      plt.saveFloat(pattern, common.Pattern);
      plt.plotFloat(pattern, MOD, 0, 1, "simulated_pattern", 1, 0, 1);
      cudaConvertFO(pattern,pattern,1./exposure);
    }

    void gratingCalib(){
      Real *spots = readImage("gratings.png", row, column);
      myCuDMalloc(Real, pattern, row*column);
      resize_cuda_image(row, column);
      myMemcpyH2D(pattern, spots, row*column*sizeof(Real));
      auto rawpeaks = findPeaks(pattern, 100);
      energies = getPeakEnergies(pattern, rawpeaks, 100); //energies should be class member;
      peaks = refinePeaks(pattern, rawpeaks, 100.0); //peaks should be class member;
      Real epsilon = 1.0/row;
      ratios = createPeakGauss(rawpeaks, energies, epsilon);
      if(verbose > 5)
      for(uint i = 0; i < ratios.size(); i++){
        plt.plotFloat(ratios[i], MOD, 0, 1, ("peak_lorentzian_"+to_string(i)).c_str(), 0, 0, 1);
        memMngr.returnCache(ratios[i]);
      }
      // Recover grating from refined peaks
      myCuFree(pattern);
      recoverGrating();
    }

    // After reconstruction: multiply probe by each peak's plane wave, propagate, plot each beam
    void plotEachBeam(){
      if(nPeaks == 0) return;
      complexFormat *Fn = (complexFormat*)memMngr.borrowCache(sz*2);
      // Allocate per-peak plane wave buffer
      complexFormat* planeWave = (complexFormat*)memMngr.borrowCache(row*column*sizeof(complexFormat));
      for(int i = 0; i < nPeaks; i++){
        Real px = peaks[i].x * row;
        Real py = peaks[i].y * column;
        createPlaneWave(planeWave, px, py, peakEnergies[i]);
        multiply(Fn, probeWave, planeWave);
        propagate_pupil.angularSpectrumPropagate(Fn, Fn);
        multiply(Fn, Fn, objectWave);
        myFFT(Fn, Fn);
        plt.plotComplex(Fn, MOD2, 1, exposure/(3*row*column), ("beam_diffraction_"+to_string(i)).c_str(), 1, 0, 1);
      }
      memMngr.returnCache(Fn);
      memMngr.returnCache(planeWave);
    }

    void plotResults(){
      plt.plotComplexColor(objectWave, 0, 1, "recon_object_wave");
      plt.plotComplexColor(objectWave, 0, 1, "recon_phase", 0, 0, 1);
      plt.plotComplex(objectWave, MOD2, 0, 1, "recon_intensity",0,0,1);
      myFFT(objectWave, objectWave);
      //plt.plotComplex(objectWave, MOD, 1, sqrt(exposure/(30*row*column)), "obj_freq",1,0,1);
      plt.plotComplex(objectWave, MOD2, 1, exposure/(9*row*column), "obj_freq",1,0,1);
      plt.plotComplexColor(probeWave, 0, 1, "recon_probe_wave");
      cudaConvertFO(objectWave);
      myCuDMalloc(complexFormat, lowres, (row/3)*(column/3));
      resize_cuda_image(row/3, column/3);
      crop(objectWave, lowres, row, column);
      cudaConvertFO(lowres, lowres, 1./(row*column));
      plt.init(row/3, column/3);
      init_fft(row/3, column/3);
      myIFFT(lowres, lowres);
      plt.plotComplexColor(lowres, 0, 1, "recon_lowres");
    }

    void iterate(){
      Real norm = 1./sqrt(row*column);
      // Allocate block buffers
      complexFormat *Fn = (complexFormat*)memMngr.borrowCache(sz*2);
      complexFormat *esw = (complexFormat*)memMngr.borrowCache(sz*2);
      myCuDMallocClean(complexFormat, objectWave_prev, row*column);
      myCuDMallocClean(complexFormat, probe_prev, row*column);
      myCuDMalloc(complexFormat, probestep, row*column);
      myCuDMalloc(Real, split_mask_tmp, (row/split)*(column/split));
      myCuDMalloc(complexFormat, objeff, row*column);
      resize_cuda_image(row/split, column/split);
      linearConst(split_mask_tmp, split_mask_tmp, 0, 1);
      myCuDMalloc(Real, split_mask, row*column);
      resize_cuda_image(row, column);
      pad(split_mask_tmp, split_mask, row/split, column/split);
      myCuFree(split_mask_tmp);
      // Pre-normalize pattern blocks by exposure (same as ptycho.cpp createPattern)
      int update_probe_iter = 4000;
      Real objMax, probeMax;
      Real *maxCache = (Real*)memMngr.borrowCache(row*column*sizeof(Real));
      myCuDMalloc(Real, d_norm, 2);
      myDMalloc(Real, h_norm, 2);

      Real probeStepSize = 1.;
      Real tk = 0.5+sqrt(1.25);
      Real tkp1;
      Real tkp1probe;
      Real tkprobe = 0.5+sqrt(1.25);
      int vidhandle_object;
      if(saveVideoEveryIter){
        vidhandle_object = plt.initVideo("recon_object.mp4",24, 1, online);
        plt.toVideo = -1;
      }
      Real stepsize = 0.05;
      cudaConvertFO(ratios[4]);
      //linearConst(ratios[4], ratios[4], stepsize, 1-stepsize);
      power(ratios[4], ratios[4], stepsize);
      //logorithm(ratios[4], ratios[4], stepsize, 0);
      rect spt;
      spt.startx = spt.starty = row/3;
      spt.endx = spt.endy = row*2/3;
      myCuDMalloc(rect, d_spt, 1);
      myMemcpyH2D(d_spt, &spt, sizeof(rect));
      createMask(ratios[4], d_spt, 1);
      myCuFree(d_spt);

      plt.plotFloat(ratios[4], REAL, 1, 1, "reg_term", 0, 0, 1);

      getMod2(maxCache, grating);
      Real gratingMax = findMax(maxCache);
      fmt::println("grating Max: {}", gratingMax);
      myCuDMalloc(complexFormat, zernikeCrop, pupilSize*pupilSize);
      void* zernike = zernike_init(pupilSize, 40, 0); //40 is already high enough for modelling complex beams
      myCuDMalloc(Real, pupilSupport, pupilSize*pupilSize);
      resize_cuda_image(pupilSize, pupilSize);
      createCircleMask(pupilSupport, Real(pupilSize+1)/2, Real(pupilSize+1)/2, Real(pupilSize)/2);
      resize_cuda_image(row, column);
      bool dozernike;
      if(isFresnel || cancelFresnel) {
        propagate_esw.multiplyFresnelPhase(pupilpatternWave);
      }
      for(int iter = 0; iter < nIter; iter++){
        // Forward: compute exit wave and FFT
        dozernike = iter < zernikeIter && iter%1==0;
        getMod2(maxCache, pupilpatternWave);
        findMax(maxCache, row*column ,d_norm);
        getMod2(maxCache, objectWave);
        multiply(maxCache, maxCache, masksum);
        findMax(maxCache, row*column, d_norm+1);
        myMemcpyD2H(h_norm, d_norm, 2*sizeof(Real));
        probeMax = h_norm[0];
        objMax = h_norm[1];
        applyThreshold(objectWave, objectWave, sqrt(objMax));
        resize_cuda_image(row,column);
        myMemcpyD2H(h_norm, d_norm, sizeof(Real));
        Real sf = pow(probeMax/objMax, 0.25);
        applyNorm(objectWave, sf);
        applyNorm(objectWave_prev, sf);
        resize_cuda_image(row,column);
        applyNorm(pupilpatternWave, 1./sf);
        applyNorm(probe_prev, 1./sf);
        applyNorm(probeWave, 1./sf);
        objMax = probeMax = sqrt(objMax*probeMax);
        multiply(esw, pupilpatternWave, objectWave);
        if(iter == nIter-1) plt.plotComplexColor(esw, 0, 1, "recon_esw");
        myFFT(esw, Fn);
        if(iter == nIter-1) plt.plotComplex(Fn, MOD2, 1, exposure/(row*column), "recon_pattern", 1, 0, 1);
        applyModAccurate(Fn, pattern, beamstop, norm);
        myIFFT(Fn, Fn);
        applyNorm(Fn, norm);
        add(esw, Fn, -1);
        if(iter < update_probe_iter) {
          updateObject(objectWave, pupilpatternWave, esw, probeMax);
        }else {
          //update probe
          clearCuMem(probestep, row*column*sizeof(complexFormat));
          updateObjectAndProbeStep(objectWave, pupilpatternWave, probestep, esw, probeMax, objMax, probeStepSize, 0.5);
          if(isFresnel || cancelFresnel) propagate_esw.removeFresnelPhase(probestep);
          propagate_pupil.angularSpectrumPropagateReverse(probestep, probestep);
          if(cancelFresnel) {
            propagate_complete.multiplyFresnelPhase(probestep);
          }
          applyNorm(probestep, -1);
          updateObject(probeWave, grating, probestep, gratingMax);
          //regularize probe
          resize_cuda_image(pupilSize, pupilSize);
          crop(probeWave, zernikeCrop, row, column);
          if(dozernike){
            zernike_compute(zernike, zernikeCrop, Real(pupilSize-1)/2, Real(pupilSize-1)/2, Real(pupilSize)/2, 5e-4*sqrt(I2));
            zernike_reconstruct(zernike, zernikeCrop, Real(pupilSize)/2);
          }else{
            applyMask(zernikeCrop, pupilSupport);
          }
          resize_cuda_image(row, column);
          pad(zernikeCrop, probeWave, pupilSize, pupilSize);
          if(iter == zernikeIter - 1) {
            plt.plotComplexColor(probeWave, 0, 0.5, "probe_after_zernike");
          }
          multiply(pupilpatternWave, probeWave, grating);
          if(cancelFresnel) {
            propagate_complete.removeFresnelPhase(pupilpatternWave);
          }
          propagate_pupil.angularSpectrumPropagate(pupilpatternWave, pupilpatternWave);
          if(isFresnel||cancelFresnel) propagate_esw.multiplyFresnelPhase(pupilpatternWave);
        }

        //Regularize object
        if(0){
          myFFT(objectWave,objectWave);
          applyNorm(objectWave, 1./(row*column));
          multiply(objectWave , objectWave, ratios[4]);
          //regularize(objectWave, sqrt(objMax/(row*column))*1e-4);
          myIFFT(objectWave,objectWave);
        }
        if(1){
          if(iter < 500){
            FISTA(objectWave, objectWave, 0.2*sqrt(I2), 1, NULL);
          }else
            FISTA(objectWave, objectWave, 0.05*sqrt(I2), 1, NULL);
        //FISTA_step(objectWave, objectWave, 0.02, NULL);
        //L2_step(objectWave, objectWave, 0.1, NULL);
        }

        if(mPIE > iter){ //momentum update
          multiply(objeff, objectWave, masksum);
          complexFormat sum = findSum(objeff);
          sum /= cabs(sum);
          multiplyConj(objectWave, objectWave, sum);
          multiplyConj(objectWave_prev, objectWave_prev, sum);
          add(objectWave_prev, objectWave, objectWave_prev, -1);
          tkp1 = 0.5+sqrt(0.25+tk*tk);
          add(objectWave_prev, objectWave, objectWave_prev, (tk-1)/tkp1);
          tk = tkp1;
          complexFormat* tmp = objectWave; objectWave = objectWave_prev; objectWave_prev = tmp;
          resize_cuda_image(row, column);
          if(dozernike){
            tkp1probe = 0.5+sqrt(0.25+tkprobe*tkprobe);
            add(probe_prev, pupilpatternWave, probe_prev, -1);
            add(probe_prev, pupilpatternWave, probe_prev, (tkprobe-1)/tkp1probe);
            tmp = pupilpatternWave; pupilpatternWave = probe_prev; probe_prev = tmp;
            tkprobe = tkp1probe;
          }
        }
        if(saveVideoEveryIter && iter % saveVideoEveryIter == 0) {
          plt.toVideo = vidhandle_object;
          plt.plotComplexColor(objectWave, 0, 1, ("recon_object_"+to_string(iter)).c_str());
          //plt.plotComplexColor(pupilpatternWave, 0, 1, ("recon_probe_"+to_string(iter)).c_str());
          plt.toVideo = -1;
        }

      }
      if(runSim){
        resize_cuda_image(row, column);
        multiplyConj(objeff, objeff, objectWave_t);
        plt.plotComplexColor(objeff, 0, 1, "phase_shift");
        complexFormat sum = findSum(objeff);
        sum /= cabs(sum);
        applyNorm(objectWave_t, sum);
        plt.plotComplexColor(objectWave_t, 0, 1, "input_object_wave");
      }
      fmt::println("Iterations complete.");
      plt.saveVideo(vidhandle_object);
      memMngr.returnCache(Fn);
      memMngr.returnCache(esw);
    }
};

int main(int argc, char** argv) {
  if(argc < 2){
    fmt::println("please feed the config file");
  }

  ssptycho setups(argv[1]);

  if(setups.runSim){
    setups.initPupilAndObject();
    setups.createPattern();
  } else {
    setups.readPattern();
  }
  setups.gratingCalib();

  if(setups.doIteration) {
    setups.initIteration();
    setups.iterate();
    setups.plotEachBeam();
    setups.plotResults();
  }

  return 0;
}
