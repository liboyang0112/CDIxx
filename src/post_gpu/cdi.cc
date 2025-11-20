#include <cassert>
#include <fmt/os.h>
#include <stdio.h>
#include <chrono>
#include <stdio.h>
#include "cudaDefs_h.cu"
#include "fmt/core.h"
#include "imgio.hpp"
#include "imageFile.hpp"
#include <ctime>
#include "cudaConfig.hpp"
#include "experimentConfig.hpp"
#include "memManager.hpp"
#include "mnistData.hpp"
#include "tvFilter.hpp"
#include "cuPlotter.hpp"
#include "cub_wrap.hpp"
#include "misc.hpp"
#include "cdi.hpp"
#include <complex.h>
using namespace std;
//#define Bits 16

void applySupport
    (void *gkp1, void *gkprime, int algo, Real *spt)
{
    switch (algo) {
      case RAAR:    ApplyRAARSupport  (gkp1, gkprime, spt);break;
      case ER:      ApplyERSupport    (gkp1, gkprime, spt);break;
      case POSER:   ApplyPOSERSupport (gkp1, gkprime, spt);break;
      case POSHIO:  ApplyPOSHIOSupport(gkp1, gkprime, spt);break;
      case HIO:     ApplyHIOSupport   (gkp1, gkprime, spt);break;
      default:      ApplyFHIOSupport  (gkp1, gkprime, spt);break;
    }
}
CDI::CDI(const char* configfile):experimentConfig(configfile){
  verbose(4, print())
    if(runSim) d = oversampling_spt*pixelsize*beamspotsize/lambda; //distance to guarentee setups.oversampling
}
void CDI::multiplyPatternPhaseMid(complexFormat* amp, Real distance){
  multiplyPatternPhase_factor(amp, resolution*resolution*M_PI/(distance*lambda), 2*distance*M_PI/lambda);
}
void CDI::multiplyFresnelPhaseMid(complexFormat* amp, Real distance){
  Real fresfactor = M_PI*lambda*distance/(sq(resolution*row));
  multiplyFresnelPhase_factor(amp, fresfactor);
}
void CDI::allocateMem(){
  if(objectWave) return;
  fmt::println("allocating memory");
  int sz = row*column;
  myCuMalloc(complexFormat, objectWave, sz);
  myCuMalloc(complexFormat, patternWave, sz);
  myCuMalloc(complexFormat, autoCorrelation, sz);
  myCuMalloc(Real, patternData, sz);
  fmt::println("initializing cuda image");
  resize_cuda_image(row,column);
  init_cuda_image(rcolor,1./exposure);
  init_fft(row,column);
  fmt::println("initializing cuda plotter");
  plt.init(row,column);
}
void CDI::readObjectWave(){
  if(domnist){
    row = column = 256;
    mnist_dat = new cuMnist(mnistData,1, 3, row, column);
    allocateMem();
    return;
  }
  int objrow,objcol;
  Real* d_intensity = 0;
  Real* d_phase = 0;
  readComplexWaveFront(intensityModulation?common.Intensity:0, phaseModulation?common.Phase:0, d_intensity, d_phase, objrow,objcol);
  row = objrow*oversampling;
  column = objcol*oversampling;
  allocateMem();
  if(phaseModulation){
    applyNorm(d_phase, phaseModulation);
  }
  createWaveFront( d_intensity, d_phase, objectWave, objrow, objcol);
  if(d_phase) memMngr.returnCache(d_phase);
  if(d_intensity) memMngr.returnCache(d_intensity);
}
void CDI::readPattern(){
  Real* pattern = readImage(common.Pattern, row, column);
  if(cropPattern) {
    Real* tmp = (Real*)memMngr.borrowCache(row*column*sizeof(Real));
    myMemcpyH2D(tmp, pattern, row*column*sizeof(Real));
    int rowtmp = row;
    int coltmp = column;
    row = column = cropPattern;
    allocateMem();
    crop( tmp, patternData, rowtmp, coltmp);
    memMngr.returnCache(tmp);
  }else{
    allocateMem();
    myMemcpyH2D(patternData, pattern, row*column*sizeof(Real));
    Real maxs = findMax(patternData,row*column);
    if(maxs < 1e-7 || maxs!=maxs){
      fmt::println("max = {:f}", maxs);
      exit(0);
    }
  }
  ccmemMngr.returnCache(pattern);
  //multiplySqy(patternData,patternData);
  cudaConvertFO(patternData);
  applyNorm(patternData, 1./exposure);
  fmt::println("Created pattern data");
}
void CDI::calculateParameters(){
  experimentConfig::calculateParameters();
  if(dopupil) {
    Real k = row*sq(pixelsize)/(lambda*d);
    dpupil = d*k/(k+1);
    resolution = lambda*dpupil/(row*pixelsize);
    fmt::println("Resolution={:4.2f}um", resolution);
    enhancementpupil = sq(pixelsize)*sqrt(row*column)/(lambda*dpupil); // this guarentee energy conservation
    fresnelFactorpupil = lambda*dpupil/sq(pixelsize)/row/column;
    enhancementMid = sq(resolution)*sqrt(row*column)/(lambda*(d-dpupil)); // this guarentee energy conservation
    fresnelFactorMid = lambda*(d-dpupil)/sq(resolution)/row/column;
  }
}
void CDI::readFiles(){
  if(runSim) {
    fmt::println("running simulation, reading input images");
    readObjectWave();
  }else{
    fmt::println("running reconstruction, reading input pattern");
    readPattern();
  }
}
void CDI::setPattern_c(complexFormat* pattern){
  cudaConvertFO(pattern,patternWave);
  getMod2(patternData, patternWave);
  applyRandomPhase(patternWave, useBS?beamstop:NULL, devstates);
}

void CDI::setPattern(Real* pattern){
  cudaConvertFO((Real*)pattern,patternData);
  createWaveFront(patternData, 0, patternWave, 1);
  applyRandomPhase(patternWave, useBS?beamstop:NULL, devstates);
}
void CDI::init(){
  allocateMem();
  if(useBS){
    if(BSimage[0] != '\0'){
      int r,c;
      Real* bs = readImage(BSimage,r,c);
      if(r!=row || c!=column) {
        fmt::println(stderr, "Beam stop mask ({}, {}) is not of the same size as diffraction pattern ({}, {})!", r,c,row,column);
        abort();
      }
      size_t sz = r*c*sizeof(Real);
      beamstop = (Real*)memMngr.borrowCache(sz);
      myMemcpyH2D(beamstop, bs, sz);
      cudaConvertFO(beamstop);
      invert(beamstop);
      plt.plotFloat(beamstop, MOD, 1, 1, "beamstop");
    }
    else
      createBeamStop();
  }
  calculateParameters();
  //inittvFilter(row,column);
  createSupport();
  devstates = newRand(column * row);
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  initRand(devstates, seed);
}
void CDI::prepareIter(){
  if(runSim) {
    if(domnist){
      myCuDMalloc(Real, intensity, row*column);
      Real* phase = 0;
      ((cuMnist*)mnist_dat)->cuRead(intensity);
      if(phaseModulation) {
        myCuMalloc(Real, phase, row*column);
        ((cuMnist*)mnist_dat)->cuRead(phase);
      }
      createWaveFront((Real*)intensity, (Real*)phase, objectWave, 1);
      memMngr.returnCache(intensity);
      if(phaseModulation) memMngr.returnCache(phase);
    }
    if(isFresnel) multiplyFresnelPhase(objectWave, d);
    //applyRandomPhase(objectWave, 0, devstates);
    verbose(2,plt.plotComplex(objectWave, MOD2, 0, 1, "inputIntensity", 0));
    verbose(2,plt.plotComplex(objectWave, PHASE, 0, 1, "inputPhase", 0));
    verbose(4,fmt::println("Generating diffraction pattern"));
    myFFT(objectWave,patternWave);
    convertFOPhase( patternWave, 1./sqrt(row*column));
    plt.plotComplex(patternWave, PHASE, 1, 1, "init_pattern_phase", 0);
    getMod2(patternData, patternWave);
    if(useBS) {
      myCuDMalloc(Real, patterncache, row*column);
      linearConst(patterncache, beamstop, -1, 1);
      plt.plotFloat(patterncache, MOD, 1, 1, "beamstop");
      multiply(patternData, patternData, patterncache);
      myCuFree(patterncache);
    }
    plt.plotFloat(patternData, MOD, 1, exposure, "theory_pattern", 0);
    plt.plotFloat(patternData, MOD, 1, exposure, "theory_pattern_log", 1);
    cudaConvertFO(patternData);
    if(simCCDbit){
      applyPoissonNoise_WO(patternData, noiseLevel, devstates);
    }
    applyNorm(patternData, exposure);
    plt.saveFloat(patternData, "sim_pattern");
    applyNorm(patternData, 1./exposure);
    cudaConvertFO(patternData);
  }
  if(restart){
    imageFile fdata;
    FILE* frestart = fopen(common.restart, "r");
    if(frestart)
      if(!fread(&fdata, sizeof(fdata), 1, frestart)){
        fmt::println("WARNING: file {} is empty!", common.restart);
      }
    if(fdata.rows == row && fdata.cols == column){
      size_t sz = row*column*sizeof(complexFormat);
      complexFormat *wf = (complexFormat*) ccmemMngr.borrowCache(sz);
      if(!fread(wf, sz, 1, frestart)){
        fmt::println("WARNING: file {} is empty!", common.restart);
      }
      myMemcpyH2D(patternWave, wf, sz);
      ccmemMngr.returnCache(wf);
      verbose(2,plt.plotComplex(patternWave, MOD2, 1, exposure, "restart_pattern", 1));
    }else{
      fmt::println("Restart file size mismatch: {}!={} || {}!={}", fdata.rows, row, fdata.cols, column);
      restart = 0;
    }
  }else{
    createWaveFront(patternData, 0, patternWave, 1);
    applyRandomPhase(patternWave, useBS?beamstop:NULL, devstates);
  }
  propagate(patternWave, objectWave, 0);
  initSupport();
  verbose(1,plt.plotFloat(patternData, MOD, 1, exposure, "init_logpattern", 1, 0, 1));
  verbose(1,plt.plotFloat(patternData, MOD, 1, exposure, ("init_pattern"+save_suffix).c_str(), 0));
}
void CDI::checkAutoCorrelation(){
  size_t sz = row*column*sizeof(Real);
  auto tmp = (complexFormat*)memMngr.useOnsite(sz);
  myFFTR2C(patternData, tmp);
  fillRedundantR2C(tmp,  autoCorrelation, 1./sqrt(row*column));
  plt.plotComplex(autoCorrelation, IMAG, 1, 1, "autocorrelation_imag", 1);
  plt.plotComplex(autoCorrelation, REAL, 1, exposure, "autocorrelation_real", 1);
  plt.plotComplex(autoCorrelation, MOD, 1, exposure, "autocorrelation", 1);
}
void CDI::createSupport(){
  rect re;
  re.startx = (oversampling_spt-1)/2*row/oversampling_spt-1;
  re.starty = (oversampling_spt-1)/2*column/oversampling_spt-1;
  re.endx = row-re.startx;
  re.endy = column-re.starty;
  cuda_spt = memMngr.borrowCache(sizeof(rect));
  myMemcpyH2D(cuda_spt, &re, sizeof(rect));
  support = (Real*)memMngr.borrowCache(row*column*sizeof(Real));
  createMask(support, (rect*)cuda_spt,0);
  memMngr.returnCache(cuda_spt);
}
void CDI::initSupport(){
  createMask(support, (rect*)cuda_spt,0);
}
void CDI::saveState(){
  myDMalloc(complexFormat, outputData, row*column);
  myMemcpyD2H(outputData, patternWave, row*column*sizeof(complexFormat));
  writeComplexImage(common.restart, outputData, row, column);//save the step
  ccmemMngr.returnCache(outputData);

  complexFormat *cuda_gkp1 = objectWave;
  verbose(2,plt.plotComplex(patternWave, MOD2, 1, exposure, "recon_pattern", 1, 0, 1))
  plt.plotComplex(cuda_gkp1, MOD2, 0, 1, ("recon_intensity"+save_suffix).c_str(), 0, isFlip);
  plt.plotComplex(cuda_gkp1, PHASE, 0, 1, ("recon_phase"+save_suffix).c_str(), 0, isFlip);
  bitMap( support, support);
  plt.plotFloat(support, MOD, 0, 1, "support", 0);
  complexFormat mid = findMiddle(support, row*column);
  int row_spt = row/oversampling_spt;
  int col_spt = column/oversampling_spt;
  size_t sptsz = row_spt*col_spt*sizeof(Real);
  resize_cuda_image(row_spt,col_spt);
  complexFormat* tmp = (complexFormat*)memMngr.borrowCache(sptsz<<1);
  Real* tmp1 = (Real*)memMngr.borrowCache(sptsz);
  fmt::println("mid= {:f},{:f}",crealf(mid), cimagf(mid));
  crop(cuda_gkp1, tmp, row, column,crealf(mid), cimagf(mid));
  getMod2(tmp1, tmp);
  Real maxnorm = 1./min(1.f,findMax(tmp1,row_spt*col_spt));
  plt.init(row_spt,col_spt);
  plt.plotFloat(tmp1, MOD, 0, maxnorm, ("recon_intensity_cropped"+save_suffix).c_str(), 0, isFlip);
  plt.plotComplex(tmp, PHASE, 0, 1, ("recon_phase_cropped"+save_suffix).c_str(), 0, isFlip);
  plt.plotComplexColor(tmp, 0, maxnorm, ("recon_wave"+save_suffix).c_str(), 0, isFlip);
  resize_cuda_image(row,column);
  plt.init(row,column);
  memMngr.returnCache(tmp);
  if(isFresnel) {
    multiplyFresnelPhase(cuda_gkp1, -d);
    plt.plotComplex(cuda_gkp1, PHASE, 0, 1, ("recon_phase_fresnelRemoved"+save_suffix).c_str(), 0, isFlip);
  }
  Real* tmp2 = (Real*)memMngr.useOnsite(row*column*sizeof(Real));
  getMod2( tmp2, patternWave);
  extendToComplex(tmp2, autoCorrelation);
  myFFT(autoCorrelation,autoCorrelation);
  plt.plotComplex(autoCorrelation, MOD2, 1, exposure/(row*column), "autocorrelation_recon", 1);
}


complexFormat* CDI::phaseRetrieve(){
  int vidhandle = 0;
  if(saveVideoEveryIter){
    vidhandle = plt.initVideo("recon_intensity.mp4",24);
    plt.showVid = -1;
  }

  complexFormat *cuda_gkp1 = objectWave;

  size_t sz = row*column;
  myCuDMalloc(complexFormat, cuda_gkprime, sz);
  myCuDMallocClean(complexFormat, prtf_map, sz);
  myCuDMalloc(complexFormat, gamma0, sz);
  myCuDMalloc(complexFormat, cache, sz);
  myCuDMalloc(Real, cuda_diff, sz);

  AlgoParser algo(algorithm);
  applyNorm(cuda_gkp1, 1./sqrt(row*column));
  Real gaussianSigma = 3;
  Real avg = 0, phi0;
  int iter = 0;
  myCuDMalloc(Real, cuda_objMod, sz);
  myCuDMalloc(Real, d_polar, row*column);
  myCuDMalloc(Real, d_prtf, row>>1);
  myDMalloc(Real, prtf, row>>1);
  myDMalloc(Real, weights, row>>1);
  for(; ; iter++){
    int ialgo = algo.next();
    if(ialgo<0) break;
    //start iteration
    if(ialgo == shrinkWrap){
      applyGaussMult(patternWave, cuda_gkprime, Real(row>>2)/gaussianSigma, 1); //multiply a gaussian in frequency domain, equivalent to convolution in spatial domain.
      myIFFT(cuda_gkprime, cuda_gkprime);
      getMod2(support,cuda_gkprime);
      Real maxs = findMax(support,row*column);
      if(avg == 0) avg = findSum(support,row*column)*4/row/column;
      if(avg*5 < maxs) maxs = avg*5;
      setThreshold(maxs*shrinkThreshold);
      if(fabs(maxs) < 1e-7 || maxs!=maxs) {
        fmt::println("max is {:f}", maxs);
        plt.plotFloat(support, MOD, 1, 1./row/column, "debug", 1, 0, 1);
      }
      if(gaussianSigma>2) {
        gaussianSigma*=0.99;
      }
      continue;
    }else if(ialgo == TV){
      getMod2(cuda_diff, cuda_gkp1);
      FISTA(cuda_diff, cuda_diff, 0.1, 1, 0);
      applyModAbs(cuda_gkp1,cuda_diff);
      myFFT( cuda_gkp1, patternWave);
      continue;
    }
    if(simCCDbit) applyMod(patternWave,patternData, useBS? beamstop:NULL, noiseLevel);
    else applyModAbs(patternWave,patternData);

    myIFFT( patternWave, cuda_gkprime);
    applyNorm(cuda_gkprime, 1./(row*column));
    applySupport(cuda_gkp1, cuda_gkprime, (Algorithm)ialgo, support);
    if(iter == prtfIter){
      applySupport(gamma0, cuda_gkprime, ER, support);
      multiply(cache, gamma0, gamma0);
      complexFormat sum = findSum(cache);
      phi0 = -atan2(cimagf(sum), crealf(sum))/2;
      addPhase(prtf_map, patternWave, phi0);

      getMod(cuda_objMod, patternWave);
      bitMap(cuda_objMod, cuda_objMod, 1e-6);
      cudaConvertFO(cuda_objMod);
      plt.plotFloat(cuda_objMod, MOD, 0, 1, "prtf_bitmap");
      resize_cuda_image(column<<1, row>>1);
      plt.init(column<<1,row>>1);
      cart2polar_kernel(cuda_objMod, d_polar, row, column);
      plt.plotFloat(d_polar, MOD, 0, 1, "prtf_bitmap_polar");
      resize_cuda_image(row>>1,1);
      edgeReduce(d_prtf, d_polar, column<<1);
      myMemcpyD2H(weights, d_prtf, (row>>1)*sizeof(Real));
      resize_cuda_image(row, column);
    }else if(iter > prtfIter) {
      applySupport(cache, cuda_gkprime, ER, support);
      multiplyConj(cache, cache, gamma0);
      complexFormat sum = findSum(cache);
      Real phi = atan2(cimagf(sum), crealf(sum));
      addPhase(prtf_map, patternWave, phi0+phi);
    }
    myFFT( cuda_gkp1, patternWave);
    if(saveVideoEveryIter && iter%saveVideoEveryIter == 0){
      plt.toVideo = vidhandle;
      plt.plotComplex(cuda_gkp1, MOD2, 0, row*column, ("recon_intensity"+to_string(iter)).c_str(), 0, isFlip, 1);
      plt.toVideo = -1;
    }
    if(saveIter){
      if(iter%100==0) {
        std::string iterstr = to_string(iter);
        plt.plotComplex(cuda_gkp1, MOD2, 0, row*column, ("recon_intensity"+iterstr).c_str(), 0, isFlip);
        plt.plotComplex(cuda_gkp1, PHASE, 0, row*column, ("recon_phase"+iterstr).c_str(), 0, isFlip);
        plt.plotComplex(patternWave, MOD2, 1, exposure, ("recon_pattern"+iterstr).c_str(), 0);
      }
      if(0 && iter > 1){  //Do Total variation denoising during the reconstruction, disabled because not quite effective.
        takeMod2Diff(patternWave,patternData, cuda_diff, useBS? beamstop:NULL);
        cudaConvertFO(cuda_diff);
        FISTA(cuda_diff, cuda_diff, 0.01, 80, 0);
        //plt.plotFloat(cuda_diff, MOD, 0, exposure, ("smootheddiff"+iterstr).c_str(), 1);
        cudaConvertFO(cuda_diff);
        //plt.plotFloat(cuda_diff, MOD, 1, exposure, ("diff"+iterstr).c_str(), 1);
        takeMod2Sum(patternWave, cuda_diff);
        //plt.plotFloat(cuda_diff, MOD, 1, exposure, ("smoothed"+iterstr).c_str(), 1);
      }
    }
  }
  applyNorm(cuda_gkp1, sqrt(row*column));
  if(saveVideoEveryIter) plt.saveVideo(vidhandle);
  if(verbose >= 4){
    cudaConvertFO(cuda_gkp1, cuda_gkprime, 1./sqrt(row*column));
    myFFT(cuda_gkprime, cuda_gkprime);
    plt.plotComplex(cuda_gkprime, PHASE, 1, 1, "recon_pattern_phase", 0, 0);
  }
  getMod2(cuda_objMod, patternWave);
  addRemoveOE( cuda_objMod, patternData, -1);  //ignore overexposed pixels when getting difference
  if(useBS){
    invert(beamstop);
    multiply(cuda_objMod, cuda_objMod, beamstop);
  }
  plt.plotFloat(cuda_objMod, MOD, 1, 1, "residual", 0, 0, 1);
  getMod2(cuda_objMod, cuda_objMod);
  residual = findSum(cuda_objMod);
  fmt::println("residual= {:f}",residual);
  
  getMod(cuda_objMod,prtf_map);
  cudaConvertFO(cuda_objMod);
  applyNorm(cuda_objMod, 1./(iter - prtfIter));
  plt.plotFloat(cuda_objMod, MOD, 0, 1, "prtf");
  resize_cuda_image(column<<1, row>>1);
  cart2polar_kernel(cuda_objMod, d_polar, row, column);
  resize_cuda_image(row>>1,1);
  clearCuMem(d_prtf, (row>>1)*sizeof(Real));
  edgeReduce(d_prtf, d_polar, column<<1);
  myMemcpyD2H(prtf, d_prtf, (row>>1)*sizeof(Real));


  fmt::ostream prtffile = fmt::output_file("prtf.txt");
  for(int i = 0; i < row>>1; i++){
    prtffile.print("{}, {:f}\n", i, prtf[i]/weights[i]);
  }
  prtffile.close();

  plt.init(row, column);
  resize_cuda_image(row, column);

  memMngr.returnCache(cuda_gkprime);
  memMngr.returnCache(cuda_objMod);

  return patternWave;
}

