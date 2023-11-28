#include <cassert>
#include <stdio.h>
#include <time.h>
#include <random>
#include <chrono>
#include <stdio.h>
#include "imgio.h"
#include "imageFile.h"
#include <ctime>
#include "cudaConfig.h"
#include <curand_kernel.h>
#include "cudaDefs.h"
#include "experimentConfig.h"
#include "mnistData.h"
#include "tvFilter.h"
#include "cuPlotter.h"
#include "cub_wrap.h"
#include "cdi.h"
#include <complex.h>
using namespace std;

//#define Bits 16

__device__ __host__ Real gaussian(Real x, Real y, Real sigma){
  Real r2 = sq(x) + sq(y);
  return exp(-r2/2/sq(sigma));
}

Real gaussian_norm(Real x, Real y, Real sigma){
  return 1./(2*M_PI*sigma*sigma)*gaussian(x,y,sigma);
}

cuFuncc(takeMod2Diff,(complexFormat* a, Real* b, Real *output, Real *bs),(cuComplex* a, Real* b, Real *output, Real *bs),((cuComplex*)a,b,output,bs),{
  cuda1Idx()
    Real mod2 = sq(a[index].x)+sq(a[index].y);
  Real tmp = b[index]-mod2;
  if(bs&&bs[index]>0.5) tmp=0;
  else if(b[index]>vars->scale) tmp = vars->scale-mod2;
  output[index] = tmp;
})

cuFuncc(takeMod2Sum,(complexFormat* a, Real* b),(cuComplex* a, Real* b),((cuComplex*)a,b),{
  cuda1Idx()
    Real tmp = b[index]+sq(a[index].x)+sq(a[index].y);
  if(tmp<0) tmp=0;
  b[index] = tmp;
})

cuFunc(multiplySqy,(Real* object, Real* out),(object,out),{
  cuda1Idx();
  int y = index%cuda_column;
  out[index] = object[index] * sq(Real(y)/cuda_column-0.5);
})

cuFunc(multiplySqx,(Real* object, Real* out),(object,out),{
  cuda1Idx();
  int x = index/cuda_column;
  out[index] = object[index] * sq(Real(x)/cuda_row-0.5);
})

__device__ void ApplyHIOSupport(bool insideS, cuComplex &rhonp1, cuComplex &rhoprime, Real beta){
  if(insideS){
    rhonp1.x = rhoprime.x;
    rhonp1.y = rhoprime.y;
  }else{
    rhonp1.x -= beta*rhoprime.x;
    rhonp1.y -= beta*rhoprime.y;
  }
}

__device__ void ApplyFHIOSupport(bool insideS, cuComplex &rhonp1, cuComplex &rhoprime){
  if(insideS){
    rhonp1.x += 1.9*(rhoprime.x-rhonp1.x);
    rhonp1.y += 1.9*(rhoprime.y-rhonp1.y);
  }else{
    //rhonp1.x += -0.9*rhoprime.x;
    //rhonp1.y += -0.9*rhoprime.y;
    rhonp1.x -= 0.9*rhoprime.x;
    rhonp1.y -= 0.9*rhoprime.y;
  }
}


__device__ void ApplyRAARSupport(bool insideS, cuComplex &rhonp1, cuComplex &rhoprime, Real beta){
  if(insideS){
    rhonp1.x = rhoprime.x;
    rhonp1.y = rhoprime.y;
  }else{
    rhonp1.x = beta*rhonp1.x+(1-2*beta)*rhoprime.x;
    rhonp1.y = beta*rhonp1.y+(1-2*beta)*rhoprime.y;
  }
}

__device__ void ApplyPOSERSupport(bool insideS, cuComplex &rhonp1, cuComplex &rhoprime){
  if(insideS && rhoprime.x > 0){
    rhonp1.x = rhoprime.x;
    rhonp1.y = rhoprime.y;
  }else{
    rhonp1.x = 0;
    rhonp1.y = 0;
  }
}
__device__ void ApplyERSupport(bool insideS, cuComplex &rhonp1, cuComplex &rhoprime){
  if(insideS){
    rhonp1.x += 1.9*(rhoprime.x-rhonp1.x);
    rhonp1.y += 1.9*(rhoprime.y-rhonp1.y);
  }else{
    rhonp1.x = 0;
    rhonp1.y = 0;
  }
}

__device__ void ApplyPOSHIOSupport(bool insideS, cuComplex &rhonp1, cuComplex &rhoprime, Real beta){
  if(rhoprime.x > 0 && insideS){
    rhonp1.x += 1.9*(rhoprime.x-rhonp1.x);
    //rhonp1.y = rhoprime.y;
  }else{
    rhonp1.x -= beta*rhoprime.x;
  }
  rhonp1.y -= beta*rhoprime.y;
}

__device__ void ApplyPOS0HIOSupport(bool insideS, cuComplex &rhonp1, cuComplex &rhoprime, Real beta){
  if(rhoprime.x > 0 && insideS){
    rhonp1.x = rhoprime.x;
    //rhonp1.y = rhoprime.y;
  }else{
    rhonp1.x -= beta*rhoprime.x;
  }
  rhonp1.y -= beta*rhoprime.y;
}
CDI::CDI(const char* configfile):experimentConfig(configfile){
  verbose(4, print())
    if(runSim) d = oversampling_spt*pixelsize*beamspotsize/lambda; //distance to guarentee setups.oversampling
}
void CDI::multiplyPatternPhaseMid(void* amp, Real distance){
  multiplyPatternPhase_factor(amp, resolution*resolution*M_PI/(distance*lambda), 2*distance*M_PI/lambda);
}
void CDI::multiplyFresnelPhaseMid(void* amp, Real distance){
  Real fresfactor = M_PI*lambda*distance/(sq(resolution*row));
  multiplyFresnelPhase_factor(amp, fresfactor);
}
void CDI::allocateMem(){
  if(objectWave) return;
  printf("allocating memory\n");
  int sz = row*column*sizeof(Real);
  objectWave = memMngr.borrowCache(sz*2);
  patternWave = memMngr.borrowCache(sz*2);
  autoCorrelation = memMngr.borrowCache(sz*2);
  patternData = (Real*)memMngr.borrowCache(sz);
  printf("initializing cuda image\n");
  resize_cuda_image(row,column);
  init_cuda_image(rcolor,1./exposure);
  init_fft(row,column);
  printf("initializing cuda plotter\n");
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
  createWaveFront( d_intensity, d_phase, (complexFormat*)objectWave, objrow, objcol);
  if(d_phase) memMngr.returnCache(d_phase);
  if(d_intensity) memMngr.returnCache(d_intensity);
}
void CDI::readPattern(){
  Real* pattern = readImage(common.Pattern, row, column);
  if(cropPattern) {
    Real* tmp = (Real*)memMngr.borrowCache(row*column*sizeof(Real));
    cudaMemcpy(tmp, pattern, row*column*sizeof(Real), cudaMemcpyHostToDevice);
    int rowtmp = row;
    int coltmp = column;
    row = column = cropPattern;
    allocateMem();
    crop( tmp, patternData, rowtmp, coltmp);
    memMngr.returnCache(tmp);
  }else{
    allocateMem();
    cudaMemcpy(patternData, pattern, row*column*sizeof(Real), cudaMemcpyHostToDevice);
  }
  ccmemMngr.returnCache(pattern);
  //multiplySqy(patternData,patternData);
  cudaConvertFO(patternData);
  applyNorm(patternData, 1./exposure);
  printf("Created pattern data\n");
}
void CDI::calculateParameters(){
  experimentConfig::calculateParameters();
  if(dopupil) {
    Real k = row*sq(pixelsize)/(lambda*d);
    dpupil = d*k/(k+1);
    resolution = lambda*dpupil/(row*pixelsize);
    printf("Resolution=%4.2fum\n", resolution);
    enhancementpupil = sq(pixelsize)*sqrt(row*column)/(lambda*dpupil); // this guarentee energy conservation
    fresnelFactorpupil = lambda*dpupil/sq(pixelsize)/row/column;
    enhancementMid = sq(resolution)*sqrt(row*column)/(lambda*(d-dpupil)); // this guarentee energy conservation
    fresnelFactorMid = lambda*(d-dpupil)/sq(resolution)/row/column;
  }
}
void CDI::readFiles(){
  if(runSim) {
    printf("running simulation, reading input images\n");
    readObjectWave();
  }else{
    printf("running reconstruction, reading input pattern\n");
    readPattern();
  }
}
void CDI::setPattern_c(void* pattern){
  cudaConvertFO((complexFormat*)pattern,(complexFormat*)patternWave);
  getMod2(patternData, (complexFormat*)patternWave);
  applyRandomPhase((complexFormat*)patternWave, useBS?beamstop:0, (curandStateMRG32k3a*)devstates);
}

void CDI::setPattern(void* pattern){
  cudaConvertFO((Real*)pattern,patternData);
  createWaveFront(patternData, 0, (complexFormat*)patternWave, 1);
  applyRandomPhase((complexFormat*)patternWave, useBS?beamstop:0, (curandStateMRG32k3a*)devstates);
}
void CDI::init(){
  allocateMem();
  if(useBS) createBeamStop();
  calculateParameters();
  //inittvFilter(row,column);
  createSupport();
  devstates = memMngr.borrowCache(column * row * sizeof(curandStateMRG32k3a));
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  initRand((curandStateMRG32k3a*)devstates, seed);
}
void CDI::prepareIter(){
  if(runSim) {
    if(domnist){
      void* intensity = memMngr.borrowCache(row*column*sizeof(Real));
      void* phase = 0;
      ((cuMnist*)mnist_dat)->cuRead(intensity);
      if(phaseModulation) {
        phase = memMngr.borrowCache(row*column*sizeof(Real));
        ((cuMnist*)mnist_dat)->cuRead(phase);
      }
      createWaveFront((Real*)intensity, (Real*)phase, (complexFormat*)objectWave, 1);
      memMngr.returnCache(intensity);
      if(phaseModulation) memMngr.returnCache(phase);
    }
    if(isFresnel) multiplyFresnelPhase(objectWave, d);
    verbose(2,plt.plotComplex(objectWave, MOD2, 0, 1, "inputIntensity", 0));
    verbose(2,plt.plotComplex(objectWave, PHASE, 0, 1, "inputPhase", 0));
    verbose(4,printf("Generating diffraction pattern\n"));
    propagate(objectWave,(complexFormat*)patternWave, 1);
    convertFOPhase( (complexFormat*)patternWave);
    plt.plotComplex(patternWave, PHASE, 1, 1, "init_pattern_phase", 0);
    getMod2(patternData, (complexFormat*)patternWave);
    if(simCCDbit){
      verbose(4,printf("Applying Poisson noise\n"));
      plt.plotFloat(patternData, MOD, 1, exposure, "theory_pattern", 0);
      plt.plotFloat(patternData, MOD, 1, exposure, "theory_pattern_log", 1);
      auto img = readImage("theory_pattern.png", row, column);
      cudaMemcpy(patternData, img, row*column*sizeof(Real),cudaMemcpyHostToDevice);
      ccmemMngr.returnCache(img);
      applyPoissonNoise_WO(patternData, noiseLevel, (curandStateMRG32k3a*)devstates,1);
      applyNorm(patternData, 1./exposure);
      cudaConvertFO(patternData);
    }
    cudaConvertFO(patternData);
    //multiplySqy(patternData,patternData);
    applyNorm(patternData, exposure);
    plt.saveFloat(patternData, "sim_pattern");
    applyNorm(patternData, 1./exposure);
    cudaConvertFO(patternData);
  }
  if(restart){
    imageFile fdata;
    FILE* frestart = fopen(common.restart, "r");
    if(frestart) fread(&fdata, sizeof(fdata), 1, frestart);
    if(fdata.rows == row && fdata.cols == column){
      size_t sz = row*column*sizeof(complexFormat);
      complexFormat *wf = (complexFormat*) ccmemMngr.borrowCache(sz);
      fread(wf, sz, 1, frestart);
      cudaMemcpy(patternWave, wf, sz, cudaMemcpyHostToDevice);
      ccmemMngr.returnCache(wf);
      verbose(2,plt.plotComplex(patternWave, MOD2, 1, exposure, "restart_pattern", 1));
    }else{
      printf("Restart file size mismatch: %d!=%d || %d!=%d\n", fdata.rows, row, fdata.cols, column);
      restart = 0;
    }
  }
  if(!restart){
    createWaveFront(patternData, 0, (complexFormat*)patternWave, 1);
    applyRandomPhase((complexFormat*)patternWave, useBS?beamstop:0, (curandStateMRG32k3a*)devstates);
  }
  propagate((complexFormat*)patternWave, objectWave, 0);
  initSupport();
  verbose(1,plt.plotFloat(patternData, MOD, 1, exposure, "init_logpattern", 1, 0, 1));
  verbose(1,plt.plotFloat(patternData, MOD, 1, exposure, ("init_pattern"+save_suffix).c_str(), 0));
}
void CDI::checkAutoCorrelation(){
  size_t sz = row*column*sizeof(Real);
  auto tmp = (complexFormat*)memMngr.useOnsite(sz);
  myFFTR2C(patternData, tmp);
  fillRedundantR2C(tmp,  (complexFormat*)autoCorrelation, 1./sqrt(row*column));
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
  cudaMemcpy(cuda_spt, &re, sizeof(rect), cudaMemcpyHostToDevice);
  support = (Real*)memMngr.borrowCache(row*column*sizeof(Real));
  createMask(support, (rect*)cuda_spt,0);
  memMngr.returnCache(cuda_spt);
}
void CDI::initSupport(){
  createMask(support, (rect*)cuda_spt,0);
}
void CDI::saveState(){
  size_t sz = row*column*sizeof(complexFormat);
  void* outputData = ccmemMngr.borrowCache(sz);
  cudaMemcpy(outputData, patternWave, sz, cudaMemcpyDeviceToHost);
  writeComplexImage(common.restart, outputData, row, column);//save the step
  ccmemMngr.returnCache(outputData);

  complexFormat *cuda_gkp1 = (complexFormat*)objectWave;
  verbose(2,plt.plotComplex(patternWave, MOD2, 1, exposure, "recon_pattern", 1, 0, 1))
  plt.plotComplex(cuda_gkp1, MOD2, 0, 1, ("recon_intensity"+save_suffix).c_str(), 0, isFlip);
  plt.plotComplex(cuda_gkp1, PHASE, 0, 1, ("recon_phase"+save_suffix).c_str(), 0, isFlip);
  bitMap( support, support, cudaVarLocal->threshold);
  plt.plotFloat(support, MOD, 0, 1, "support", 0);
  complex<float> mid = findMiddle(support, row*column);
  resize_cuda_image(row/oversampling_spt,column/oversampling_spt);
  complexFormat* tmp = (complexFormat*)memMngr.borrowCache(sizeof(complexFormat*)*(row/oversampling_spt)*(column/oversampling_spt));
  Real* tmp1 = (Real*)memMngr.borrowCache(sizeof(Real*)*(row/oversampling_spt)*(column/oversampling_spt));
  printf("mid= %f,%f\n",mid.real(), mid.imag());
  crop(cuda_gkp1, tmp, row, column,mid.real(), mid.imag());
  getMod2(tmp1, tmp);
  Real max = min(1.,findMax(tmp1,row/oversampling_spt*(column/oversampling_spt)));
  plt.init(row/oversampling_spt,column/oversampling_spt);
  plt.plotFloat(tmp1, MOD, 0, 1./max, ("recon_intensity_cropped"+save_suffix).c_str(), 0, isFlip);
  plt.plotComplex(tmp, PHASE, 0, 1, ("recon_phase_cropped"+save_suffix).c_str(), 0, isFlip);
  resize_cuda_image(row,column);
  plt.init(row,column);
  memMngr.returnCache(tmp);

  if(isFresnel) multiplyFresnelPhase(cuda_gkp1, -d);
  plt.plotComplex(cuda_gkp1, PHASE, 0, 1, ("recon_phase_fresnelRemoved"+save_suffix).c_str(), 0, isFlip);
  Real* tmp2 = (Real*)memMngr.useOnsite(sz/2);
  getMod2( tmp2, (complexFormat*)patternWave);
  myFFTR2C(tmp2, (complexFormat*)tmp2);
  fillRedundantR2C((complexFormat*)tmp2,  (complexFormat*)autoCorrelation, 1./sqrt(row*column));
  plt.plotComplex(autoCorrelation, MOD, 1, exposure, "autocorrelation_recon", 1);
}

cuFuncc(applySupportOblique,(complexFormat *gkp1, complexFormat *gkprime, Algorithm algo, Real *spt, int iter = 0, Real fresnelFactor = 0, Real costheta_r = 1),(cuComplex* gkp1, cuComplex* gkprime, Algorithm algo, Real *spt, int iter, Real fresnelFactor, Real costheta_r),((cuComplex*)gkp1,(cuComplex*)gkprime,algo,spt,iter,fresnelFactor,costheta_r),{
  cudaIdx()
    bool inside = spt[index] > vars->threshold;
  cuComplex &gkp1data = gkp1[index];
  cuComplex &gkprimedata = gkprime[index];
  if(algo==RAAR) ApplyRAARSupport(inside,gkp1data,gkprimedata,vars->beta_HIO);
  else if(algo==ER) ApplyERSupport(inside,gkp1data,gkprimedata);
  else if(algo==HIO) ApplyHIOSupport(inside,gkp1data,gkprimedata,vars->beta_HIO);
  if(fresnelFactor>1e-4 && iter < 400) {
    if(inside){
      Real phase = M_PI*fresnelFactor*(sq((x-(cuda_row>>1))*costheta_r)+sq(y-(cuda_column>>1)));
      //Real mod = cuCabs(gkp1data);
      Real mod = fabs(gkp1data.x*cos(phase)+gkp1data.y*sin(phase)); //use projection (Error reduction)
      gkp1data.x=mod*cos(phase);
      gkp1data.y=mod*sin(phase);
    }
  }
})


cuFunc(applySupport,(void *gkp1, void *gkprime, Algorithm algo, Real *spt, int iter, Real fresnelFactor),(gkp1,gkprime,algo,spt,iter,fresnelFactor),{

  cudaIdx();
  bool inside = spt[index] > vars->threshold;
  cuComplex &gkp1data = ((cuComplex*)gkp1)[index];
  cuComplex &gkprimedata = ((cuComplex*)gkprime)[index];
  if(algo==RAAR) ApplyRAARSupport(inside,gkp1data,gkprimedata,vars->beta_HIO);
  else if(algo==ER) ApplyERSupport(inside,gkp1data,gkprimedata);
  else if(algo==POSER) ApplyPOSERSupport(inside,gkp1data,gkprimedata);
  else if(algo==POSHIO) ApplyPOSHIOSupport(inside,gkp1data,gkprimedata,vars->beta_HIO);
  else if(algo==HIO) ApplyHIOSupport(inside,gkp1data,gkprimedata,vars->beta_HIO);
  else if(algo==FHIO) ApplyFHIOSupport(inside,gkp1data,gkprimedata);
  if(fresnelFactor>1e-4 && iter < 400) {
    if(inside){
      Real phase = M_PI*fresnelFactor*(sq(x-(cuda_row>>1))+sq(y-(cuda_column>>1)));
      //Real mod = cuCabs(gkp1data);
      Real mod = fabs(gkp1data.x*cos(phase)+gkp1data.y*sin(phase)); //use projection (Error reduction)
      gkp1data.x=mod*cos(phase);
      gkp1data.y=mod*sin(phase);
    }
  }
})
cuFunc(applyHIOpack,(void *gkp1, void *gkprime, int iter,Real* spt),(gkp1,gkprime,iter,spt),{
  cuda1Idx();
  if(spt[index] > vars->threshold){
    ((cuComplex*)gkp1)[index].x = iter*((cuComplex*)gkp1)[index].x - (iter-1)*((cuComplex*)gkprime)[index].x;
    ((cuComplex*)gkp1)[index].y = iter*((cuComplex*)gkp1)[index].y - (iter-1)*((cuComplex*)gkprime)[index].y;
  }
})
cuFuncc(addPsbar,(complexFormat* lambda, complexFormat* cuda_gkp1, Real beta1, Real* support),(cuComplex* lambda, cuComplex* cuda_gkp1, Real beta1, Real* support),((cuComplex*)lambda, (cuComplex*)cuda_gkp1, beta1, support),{
  cuda1Idx();
  if(support[index] < vars->threshold){
    lambda[index].x += beta1*cuda_gkp1[index].x;
    lambda[index].y += beta1*cuda_gkp1[index].y;
  }
});
void* CDI::phaseRetrieve(){
  int vidhandle = 0;
  if(saveVideoEveryIter){
    vidhandle = plt.initVideo("recon_intensity.mp4",24);
    plt.showVid = -1;
  }
  //Real beta = -1;
  //Real gammas = -1./beta;
  //Real gammam = 1./beta;
  Real gaussianSigma = 2.5;

  size_t sz = row*column*sizeof(complexFormat);
  complexFormat *cuda_gkp1 = (complexFormat*)objectWave;

  complexFormat *cuda_gkprime;
  Real *cuda_diff;
  Real *cuda_objMod;
  complexFormat *lambda = (complexFormat*)memMngr.borrowCleanCache(sz);
  cuda_diff = (Real*) memMngr.borrowCache(sz/2);
  cuda_gkprime = (complexFormat*)memMngr.borrowCache(sz);
  cuda_objMod = (Real*)memMngr.borrowCache(sz/2);
  cudaMemcpy(cuda_diff, patternData, sz/2, cudaMemcpyDeviceToDevice);

  //cudaConvertFO( cuda_diff);
  //FISTA(cuda_diff, cuda_diff, 0.01, 80, 0);
  //cudaConvertFO( cuda_diff);
  //plt.plotFloat(cuda_diff, MOD, 1, exposure, "smoothed_pattern",1);
  AlgoParser algo(algorithm);
  int size = floor(gaussianSigma*6);
  size = ((size>>1)<<1)+1;
  Real*  d_gaussianKernel = (Real*) memMngr.borrowCache(size*size*sizeof(Real));
  applyNorm(cuda_gkp1, 1./sqrt(row*column));
  for(int iter = 0; ; iter++){
    int ialgo = algo.next();
    if(ialgo<0) break;
    //start iteration
    if(ialgo == shrinkWrap){
      getMod2(cuda_objMod,cuda_gkp1);
      applyGaussConv(cuda_objMod, support, d_gaussianKernel, gaussianSigma);
      cudaVarLocal->threshold = findMax(support,row*column)*shrinkThreshold;
      cudaMemcpy(cudaVar, cudaVarLocal, sizeof(cudaVars),cudaMemcpyHostToDevice);
      if(gaussianSigma>1) {
        gaussianSigma*=0.99;
      }
      continue;
    }
    if(simCCDbit) applyMod((complexFormat*)patternWave,cuda_diff, useBS? beamstop:0, !reconAC || iter > 1000,iter, noiseLevel);
    else applyModAbs((complexFormat*)patternWave,cuda_diff);
    myIFFT( (complexFormat*)patternWave, cuda_gkprime);
    applyNorm(cuda_gkprime, 1./(row*column));
    if(costheta == 1) applySupport(cuda_gkp1, cuda_gkprime, (Algorithm)ialgo, support, iter, isFresnel? fresnelFactor:0);
    else applySupportOblique(cuda_gkp1, cuda_gkprime, (Algorithm)ialgo, support, iter, isFresnel? fresnelFactor:0, 1./costheta);
    myFFT( cuda_gkp1, (complexFormat*)patternWave);
    if(saveVideoEveryIter && iter%saveVideoEveryIter == 0){
      plt.toVideo = vidhandle;
      plt.plotComplex(cuda_gkp1, MOD2, 0, row*column, ("recon_intensity"+to_string(iter)).c_str(), 0, isFlip, 1);
      plt.toVideo = -1;
    }
    if(iter%100==0) {
      std::string iterstr = to_string(iter);
      if(saveIter){
        plt.plotComplex(cuda_gkp1, MOD2, 0, 1, ("recon_intensity"+iterstr).c_str(), 0, isFlip);
        plt.plotComplex(cuda_gkp1, PHASE, 0, 1, ("recon_phase"+iterstr).c_str(), 0, isFlip);
        plt.plotComplex(patternWave, MOD2, 1, exposure, ("recon_pattern"+iterstr).c_str(), 0);
      }
      if(0&&iter > 1){  //Do Total variation denoising during the reconstruction, disabled because not quite effective.
        takeMod2Diff((complexFormat*)patternWave,patternData, cuda_diff, useBS? beamstop:0);
        cudaConvertFO(cuda_diff);
        FISTA(cuda_diff, cuda_diff, 0.01, 80, 0);
        //plt.plotFloat(cuda_diff, MOD, 0, exposure, ("smootheddiff"+iterstr).c_str(), 1);
        cudaConvertFO(cuda_diff);
        plt.plotFloat(cuda_diff, MOD, 1, exposure, ("diff"+iterstr).c_str(), 1);
        takeMod2Sum((complexFormat*)patternWave, cuda_diff);
        //plt.plotFloat(cuda_diff, MOD, 1, exposure, ("smoothed"+iterstr).c_str(), 1);
      }
    }
  }
  applyNorm(cuda_gkp1, sqrt(row*column));
  if(saveVideoEveryIter) plt.saveVideo(vidhandle);
  if(d_gaussianKernel) memMngr.returnCache(d_gaussianKernel);
  if(verbose >= 4){
    cudaConvertFO(cuda_gkp1, cuda_gkprime);
    propagate(cuda_gkprime, cuda_gkprime, 1);
    plt.plotComplex(cuda_gkprime, PHASE, 1, 1, "recon_pattern_phase", 0, 0);
  }
  getMod2(cuda_objMod, (complexFormat*)patternWave);
  addRemoveOE( cuda_objMod, patternData, -1);  //ignore overexposed pixels when getting difference
  getMod2(cuda_objMod, cuda_objMod);
  plt.plotFloat(cuda_objMod, MOD, 1, 1, "residual", 0, 0, 1);
  residual = findSum(cuda_objMod);
  printf("residual= %f\n",residual);

  memMngr.returnCache(cuda_gkprime);
  memMngr.returnCache(cuda_objMod);
  memMngr.returnCache(cuda_diff);

  return patternWave;
}

