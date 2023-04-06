#include <complex>
#include <cassert>
#include <stdio.h>
#include <time.h>
#include <random>
#include <chrono>

#include <stdio.h>
#include "common.h"
#include <ctime>
#include "cudaConfig.h"
#include "experimentConfig.h"
#include "mnistData.h"
#include "tvFilter.h"
#include "cuPlotter.h"
#include "cub_wrap.h"
#include "cdi.h"


//#define Bits 16

__device__ __host__ Real gaussian(Real x, Real y, Real sigma){
  Real r2 = pow(x,2) + pow(y,2);
  return exp(-r2/2/pow(sigma,2));
}

Real gaussian_norm(Real x, Real y, Real sigma){
  return 1./(2*M_PI*sigma*sigma)*gaussian(x,y,sigma);
}

cuFunc(takeMod2Diff,(complexFormat* a, Real* b, Real *output, Real *bs),(a,b,output,bs),{
  cudaIdx()
    Real mod2 = pow(a[index].x,2)+pow(a[index].y,2);
  Real tmp = b[index]-mod2;
  if(bs&&bs[index]>0.5) tmp=0;
  else if(b[index]>vars->scale) tmp = vars->scale-mod2;
  output[index] = tmp;
})

cuFunc(takeMod2Sum,(complexFormat* a, Real* b),(a,b),{
  cudaIdx()
    Real tmp = b[index]+pow(a[index].x,2)+pow(a[index].y,2);
  if(tmp<0) tmp=0;
  b[index] = tmp;
})


__device__ void ApplyHIOSupport(bool insideS, complexFormat &rhonp1, complexFormat &rhoprime, Real beta){
  if(insideS){
    rhonp1.x = rhoprime.x;
    rhonp1.y = rhoprime.y;
  }else{
    rhonp1.x -= beta*rhoprime.x;
    rhonp1.y -= beta*rhoprime.y;
  }
}

__device__ void ApplyRAARSupport(bool insideS, complexFormat &rhonp1, complexFormat &rhoprime, Real beta){
  if(insideS){
    rhonp1.x = rhoprime.x;
    rhonp1.y = rhoprime.y;
  }else{
    rhonp1.x = beta*rhonp1.x+(1-2*beta)*rhoprime.x;
    rhonp1.y = beta*rhonp1.y+(1-2*beta)*rhoprime.y;
  }
}

__device__ void ApplyERSupport(bool insideS, complexFormat &rhonp1, complexFormat &rhoprime){
  if(insideS){
    rhonp1.x = rhoprime.x;
    rhonp1.y = rhoprime.y;
  }else{
    rhonp1.x = 0;
    rhonp1.y = 0;
  }
}


__device__ void ApplyPOSHIOSupport(bool insideS, complexFormat &rhonp1, complexFormat &rhoprime, Real beta){
  if(rhoprime.x > 0 && (insideS/* || rhoprime[0]<30./rcolor*/)){
    rhonp1.x = rhoprime.x;
  }else{
    rhonp1.x -= beta*rhoprime.x;
  }
  rhonp1.y -= beta*rhoprime.y;
}
CDI::CDI(const char* configfile):experimentConfig(configfile){
  verbose(2, print())
    if(runSim) d = oversampling_spt*pixelsize*beamspotsize/lambda; //distance to guarentee setups.oversampling
}
void CDI::propagatepupil(complexFormat* datain, complexFormat* dataout, bool isforward){
  myCufftExec( *plan, datain, dataout, isforward? CUFFT_FORWARD: CUFFT_INVERSE);
  cudaF(applyNorm,dataout, isforward? forwardFactorpupil: inverseFactorpupil);
}
void CDI::propagateMid(complexFormat* datain, complexFormat* dataout, bool isforward){
  myCufftExec( *plan, datain, dataout, isforward? CUFFT_FORWARD: CUFFT_INVERSE);
  cudaF(applyNorm,dataout, isforward? forwardFactorMid: inverseFactorMid);
}
void CDI::multiplyPatternPhaseMid(complexFormat* amp, Real distance){
  multiplyPatternPhase_factor(amp, resolution*resolution*M_PI/(distance*lambda), 2*distance*M_PI/lambda);
}
void CDI::multiplyFresnelPhaseMid(complexFormat* amp, Real distance){
  Real fresfactor = M_PI*lambda*distance/(pow(resolution*row,2));
  multiplyFresnelPhase_factor(amp, fresfactor);
}
void CDI::allocateMem(){
  if(objectWave) return;
  printf("allocating memory\n");
  int sz = row*column*sizeof(Real);
  objectWave = (complexFormat*)memMngr.borrowCache(sz*2);
  patternWave = (complexFormat*)memMngr.borrowCache(sz*2);
  autoCorrelation = (complexFormat*)memMngr.borrowCache(sz*2);
  patternData = (Real*)memMngr.borrowCache(sz);
  printf("initializing cuda image\n");
  init_cuda_image(row,column,rcolor,1./exposure);
  init_fft(row,column);
  printf("initializing cuda plotter\n");
  plt.init(row,column);
}
void CDI::readObjectWave(){
  if(domnist){
    row = column = 256;
    mnist_dat = new cuMnist(mnistData.c_str(), 3, row, column);
    allocateMem();
    return;
  }
  int objrow,objcol;
  Real* d_intensity = 0;
  Real* d_phase = 0;
  readComplexWaveFront(intensityModulation?common.Intensity.c_str():0, phaseModulation?common.Phase.c_str():0, d_intensity, d_phase, objrow,objcol);
  row = objrow*oversampling;
  column = objcol*oversampling;
  allocateMem();
  cudaF(createWaveFront, d_intensity, d_phase, (complexFormat*)objectWave, objrow, objcol);
  if(d_phase) memMngr.returnCache(d_phase);
  if(d_intensity) memMngr.returnCache(d_intensity);
}
void CDI::readPattern(){
  Real* pattern = readImage(common.Pattern.c_str(), row, column);
  allocateMem();
  cudaMemcpy(patternData, pattern, row*column*sizeof(Real), cudaMemcpyHostToDevice);
  ccmemMngr.returnCache(pattern);
  cudaF(cudaConvertFO,patternData);
  cudaF(applyNorm,patternData, 1./exposure);
  printf("Created pattern data\n");
}
void CDI::calculateParameters(){
  experimentConfig::calculateParameters();
  if(dopupil) {
    Real k = row*pow(pixelsize,2)/(lambda*d);
    dpupil = d*k/(k+1);
    resolution = lambda*dpupil/(row*pixelsize);
    printf("Resolution=%4.2fum\n", resolution);
    enhancementpupil = pow(pixelsize,2)*sqrt(row*column)/(lambda*dpupil); // this guarentee energy conservation
    fresnelFactorpupil = lambda*dpupil/pow(pixelsize,2)/row/column;
    forwardFactorpupil = fresnelFactorpupil*enhancementpupil;
    inverseFactorpupil = 1./row/column/forwardFactorpupil;
    enhancementMid = pow(resolution,2)*sqrt(row*column)/(lambda*(d-dpupil)); // this guarentee energy conservation
    fresnelFactorMid = lambda*(d-dpupil)/pow(resolution,2)/row/column;
    forwardFactorMid = fresnelFactorMid*enhancementMid;
    inverseFactorMid = 1./row/column/forwardFactorMid;
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
  cudaF(cudaConvertFO,(complexFormat*)pattern,patternWave);
  cudaF(getMod2,patternData, patternWave);
  cudaF(applyRandomPhase,patternWave, useBS?beamstop:0, devstates);
}

void CDI::setPattern(void* pattern){
  cudaF(cudaConvertFO,(Real*)pattern,patternData);
  cudaF(createWaveFront,patternData, 0, patternWave, 1);
  cudaF(applyRandomPhase,patternWave, useBS?beamstop:0, devstates);
}
void CDI::init(){
  allocateMem();
  if(useBS) createBeamStop();
  calculateParameters();
  //inittvFilter(row,column);
  createSupport();
  devstates = (curandStateMRG32k3a *)memMngr.borrowCache(column * row * sizeof(curandStateMRG32k3a));
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  cudaF(initRand,devstates, seed);
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
      cudaF(createWaveFront,(Real*)intensity, (Real*)phase, (complexFormat*)objectWave, 1);
      memMngr.returnCache(intensity);
      if(phaseModulation) memMngr.returnCache(phase);
      initSupport();
    }
    if(isFresnel) multiplyFresnelPhase(objectWave, d);
    verbose(2,plt.plotComplex(objectWave, MOD2, 0, 1, "inputIntensity", 0));
    verbose(2,plt.plotComplex(objectWave, PHASE, 0, 1, "inputPhase", 0));
    verbose(4,printf("Generating diffraction pattern\n"));
    propagate(objectWave,patternWave, 1);
    cudaF(getMod2,patternData, patternWave);
    if(simCCDbit){
      verbose(4,printf("Applying Poisson noise\n"));
      verbose(1,plt.plotFloat(patternData, MOD, 1, exposure, "theory_pattern", 0));
      auto img = readImage("theory_pattern.png", row, column);
      cudaMemcpy(patternData, img, row*column*sizeof(Real),cudaMemcpyHostToDevice);
      ccmemMngr.returnCache(img);
      cudaF(applyPoissonNoise_WO,patternData, noiseLevel, devstates,1);
      cudaF(applyNorm,patternData, 1./exposure);
      cudaF(cudaConvertFO,patternData);
    }
  }
  if(restart){
    complexFormat *wf = (complexFormat*) readComplexImage(common.restart.c_str());
    cudaMemcpy(patternWave, wf, row*column*sizeof(complexFormat), cudaMemcpyHostToDevice);
    verbose(2,plt.plotComplex(patternWave, MOD2, 1, exposure, "restart_pattern", 1));
    ccmemMngr.returnCache(wf);
  }else {
    cudaF(createWaveFront,patternData, 0, patternWave, 1);
    cudaF(applyRandomPhase,patternWave, useBS?beamstop:0, devstates);
  }
  verbose(1,plt.plotFloat(patternData, MOD, 1, exposure, "init_logpattern", 1));
  plt.plotFloat(patternData, MOD, 1, exposure, ("init_pattern"+save_suffix).c_str(), 0);
}
void CDI::checkAutoCorrelation(){
  size_t sz = row*column*sizeof(Real);
  auto tmp = (complexFormat*)memMngr.useOnsite(sz);
  myCufftExecR2C( *planR2C, patternData, tmp);
  cudaF(fillRedundantR2C,tmp, autoCorrelation, 1./sqrt(row*column));
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
  cuda_spt = (rect*)memMngr.borrowCache(sizeof(rect));
  cudaMemcpy(cuda_spt, &re, sizeof(rect), cudaMemcpyHostToDevice);
  support = (Real*)memMngr.borrowCache(row*column*sizeof(Real));
  cudaF(createMask,support, cuda_spt,0);
  memMngr.returnCache(cuda_spt);
}
void CDI::initSupport(){
  cudaF(createMask,support, cuda_spt,0);
}
void CDI::saveState(){
  size_t sz = row*column*sizeof(complexFormat);
  void* outputData = ccmemMngr.borrowCache(sz);
  cudaMemcpy(outputData, patternWave, sz, cudaMemcpyDeviceToHost);
  writeComplexImage(common.restart.c_str(), outputData, row, column);//save the step
  ccmemMngr.returnCache(outputData);
}

cuFunc(applySupportOblique,(complexFormat *gkp1, complexFormat *gkprime, Algorithm algo, Real *spt, int iter = 0, Real fresnelFactor = 0, Real costheta_r = 1),(gkp1,gkprime,algo,spt,iter,fresnelFactor,costheta_r),{
  cudaIdx()
    bool inside = spt[index] > vars->threshold;
  complexFormat &gkp1data = gkp1[index];
  complexFormat &gkprimedata = gkprime[index];
  if(algo==RAAR) ApplyRAARSupport(inside,gkp1data,gkprimedata,vars->beta_HIO);
  else if(algo==ER) ApplyERSupport(inside,gkp1data,gkprimedata);
  else if(algo==HIO) ApplyHIOSupport(inside,gkp1data,gkprimedata,vars->beta_HIO);
  if(fresnelFactor>1e-4 && iter < 400) {
    if(inside){
      Real phase = M_PI*fresnelFactor*(pow((x-(cuda_row>>1))*costheta_r,2)+pow(y-(cuda_column>>1),2));
      //Real mod = cuCabs(gkp1data);
      Real mod = fabs(gkp1data.x*cos(phase)+gkp1data.y*sin(phase)); //use projection (Error reduction)
      gkp1data.x=mod*cos(phase);
      gkp1data.y=mod*sin(phase);
    }
  }
})


cuFunc(applySupport,(complexFormat *gkp1, complexFormat *gkprime, Algorithm algo, Real *spt, int iter = 0, Real fresnelFactor = 0),(gkp1,gkprime,algo,spt,iter,fresnelFactor),{

  cudaIdx()
    bool inside = spt[index] > vars->threshold;
  complexFormat &gkp1data = gkp1[index];
  complexFormat &gkprimedata = gkprime[index];
  if(algo==RAAR) ApplyRAARSupport(inside,gkp1data,gkprimedata,vars->beta_HIO);
  else if(algo==ER) ApplyERSupport(inside,gkp1data,gkprimedata);
  else if(algo==POSHIO) ApplyPOSHIOSupport(inside,gkp1data,gkprimedata,vars->beta_HIO);
  else if(algo==HIO) ApplyHIOSupport(inside,gkp1data,gkprimedata,vars->beta_HIO);
  if(fresnelFactor>1e-4 && iter < 400) {
    if(inside){
      Real phase = M_PI*fresnelFactor*(pow(x-(cuda_row>>1),2)+pow(y-(cuda_column>>1),2));
      //Real mod = cuCabs(gkp1data);
      Real mod = fabs(gkp1data.x*cos(phase)+gkp1data.y*sin(phase)); //use projection (Error reduction)
      gkp1data.x=mod*cos(phase);
      gkp1data.y=mod*sin(phase);
    }
  }
})

complexFormat* CDI::phaseRetrieve(){
  Real beta = -1;
  Real gammas = -1./beta;
  Real gammam = 1./beta;
  Real gaussianSigma = 2.5;

  size_t sz = row*column*sizeof(complexFormat);
  complexFormat *cuda_gkp1 = (complexFormat*)objectWave;

  complexFormat *cuda_gkprime;
  Real *cuda_diff;
  Real *cuda_objMod;
  cuda_diff = (Real*) memMngr.borrowCache(sz/2);
  cuda_gkprime = (complexFormat*)memMngr.borrowCache(sz);
  cuda_objMod = (Real*)memMngr.borrowCache(sz/2);
  cudaMemcpy(cuda_diff, patternData, sz/2, cudaMemcpyDeviceToDevice);

  AlgoParser algo(algorithm);
  Real* d_gaussianKernel = 0;
  Real* gaussianKernel = 0;
  for(int iter = 0; ; iter++){
    int ialgo = algo.next();
    if(ialgo<0) break;
    //start iteration
    if(!simCCDbit) cudaF(applyMod,patternWave,cuda_diff, useBS? beamstop:0, !reconAC || iter > 1000,iter, noiseLevel);
    else cudaF(applyModAbs,patternWave,cuda_diff);
    propagate(patternWave, cuda_gkprime, 0);
    if(costheta == 1) cudaF(applySupport,cuda_gkp1, cuda_gkprime, (Algorithm)ialgo, support, iter, isFresnel? fresnelFactor:0);
    else cudaF(applySupportOblique,cuda_gkp1, cuda_gkprime, (Algorithm)ialgo, support, iter, isFresnel? fresnelFactor:0, 1./costheta);
    //update mask
    if(iter%20==0){
      cudaF(getMod2,cuda_objMod,cuda_gkp1);
      if(iter > 0 && useShrinkMap){
        int size = floor(gaussianSigma*6); // r=3 sigma to ensure the contribution outside kernel is negligible (0.01 of the maximum)
        size = size/2;
        int width = size*2+1;
        int kernelsz = width*width*sizeof(Real);
        if(!d_gaussianKernel){
          d_gaussianKernel = (Real*) memMngr.borrowCache(kernelsz);
          gaussianKernel =  (Real*) ccmemMngr.borrowCache(kernelsz);
        }
        Real total = 0;
        for(int i = 0; i < width*width; i++) {
          gaussianKernel[i] = gaussian((i/width-size),i%width-size, gaussianSigma);
          total+= gaussianKernel[i];
        }
        for(int i = 0; i < width*width; i++){
          gaussianKernel[i] /= total;
        }
        cudaMemcpy(d_gaussianKernel, gaussianKernel, kernelsz, cudaMemcpyHostToDevice);
        cudaFShared(applyConvolution,(pow(width-1+threadsPerBlock.x,2)+(width*width))*sizeof(Real),cuda_objMod, support, d_gaussianKernel, size, size);

        cudaVarLocal->threshold = findMax(support,row*column)*shrinkThreshold;
        cudaMemcpy(cudaVar, cudaVarLocal, sizeof(cudaVars),cudaMemcpyHostToDevice);

        if(gaussianSigma>1.5) {
          gaussianSigma*=0.99;
        }
      }
      /*
      cudaF(getReal,cuda_objMod,cuda_gkp1);
      FISTA(cuda_objMod, cuda_objMod, 2e-2, 1, 0);
      cudaF(assignReal, cuda_objMod,cuda_gkp1);
      cudaF(getImag,cuda_objMod,cuda_gkp1);
      FISTA(cuda_objMod, cuda_objMod, 2e-2, 1, 0);
      cudaF(assignImag, cuda_objMod,cuda_gkp1);
      */
    }
    propagate( cuda_gkp1, patternWave, 1);
    if(iter%100==0) {
      std::string iterstr = to_string(iter);
      if(saveIter){
        plt.plotComplex(cuda_gkp1, MOD2, 0, 1, ("recon_intensity"+iterstr).c_str(), 0, isFlip);
        plt.plotComplex(cuda_gkp1, PHASE, 0, 1, ("recon_phase"+iterstr).c_str(), 0, isFlip);
        plt.plotComplex(patternWave, MOD2, 1, exposure, ("recon_pattern"+iterstr).c_str(), 0);
      }
      if(0&&iter > 1){  //Do Total variation denoising during the reconstruction, disabled because not quite effective.
        cudaF(takeMod2Diff,patternWave,patternData, cuda_diff, useBS? beamstop:0);
        //plt.plotFloat(cuda_diff, MOD, 1, exposure, ("diff"+iterstr).c_str(), 1);
        cudaF(cudaConvertFO,cuda_diff);
        FISTA(cuda_diff, cuda_diff, 0.001, 80, 0);
        cudaF(cudaConvertFO,cuda_diff);
        //plt.plotFloat(cuda_diff, MOD, 1, exposure, ("smootheddiff"+iterstr).c_str(), 1);
        cudaF(takeMod2Sum,patternWave, cuda_diff);
        //plt.plotFloat(cuda_diff, MOD, 1, exposure, ("smoothed"+iterstr).c_str(), 1);
      }
    }
  }
  if(gaussianKernel) ccmemMngr.returnCache(gaussianKernel);
  if(d_gaussianKernel) memMngr.returnCache(d_gaussianKernel);
  verbose(2,plt.plotComplex(patternWave, MOD2, 1, exposure, "recon_pattern", 1, 0))
    if(verbose >= 4){
      cudaF(cudaConvertFO,(complexFormat*)cuda_gkp1, cuda_gkprime);
      propagate(cuda_gkprime, cuda_gkprime, 1);
      plt.plotComplex(cuda_gkprime, PHASE, 1, 1, "recon_pattern_phase", 0, 0);
    }
  plt.plotFloat(support, MOD, 0, 1, "support", 0);
  plt.plotComplex(cuda_gkp1, MOD2, 0, 1, ("recon_intensity"+save_suffix).c_str(), 0, isFlip);
  plt.plotComplex(cuda_gkp1, PHASE, 0, 1, ("recon_phase"+save_suffix).c_str(), 0, isFlip);
  auto mid = findMiddle(cuda_gkp1, row*column);
  init_cuda_image(row/oversampling_spt,column/oversampling_spt);
  complexFormat* tmp = (complexFormat*)memMngr.borrowCache(sizeof(complexFormat*)*(row/oversampling_spt)*(column/oversampling_spt));
  printf("mid= %f,%f\n",mid.x,mid.y);
  cudaF(crop,cuda_gkp1, tmp, row, column,mid.x,mid.y);
  plt.init(row/oversampling_spt,column/oversampling_spt);
  plt.plotComplex(tmp, MOD2, 0, 1, ("recon_intensity_cropped"+save_suffix).c_str(), 0, isFlip);
  plt.plotComplex(tmp, PHASE, 0, 1, ("recon_phase_cropped"+save_suffix).c_str(), 0, isFlip);
  init_cuda_image(row,column);
  plt.init(row,column);
  memMngr.returnCache(tmp);

  if(isFresnel) multiplyFresnelPhase(cuda_gkp1, -d);
  plt.plotComplex(cuda_gkp1, PHASE, 0, 1, ("recon_phase_fresnelRemoved"+save_suffix).c_str(), 0, isFlip);
  cudaF(applyMod,patternWave,patternData,useBS?beamstop:0,1,nIter, noiseLevel);
  cudaF(getMod2, cuda_objMod, patternWave);
  cudaF(add, patternData, cuda_objMod, -1);
  cudaF(add, cuda_objMod, patternData, 1);
  plt.plotFloat(patternData, MOD, 1, exposure, "residual",1);
  myCufftExecR2C( *planR2C, cuda_objMod, (complexFormat*)cuda_objMod);
  cudaF(fillRedundantR2C,(complexFormat*)cuda_objMod, cuda_gkprime, 1./sqrt(row*column));
  plt.plotComplex(cuda_gkprime, MOD, 1, exposure, "autocorrelation_recon", 1);
  memMngr.returnCache(cuda_gkprime);
  memMngr.returnCache(cuda_objMod);
  memMngr.returnCache(cuda_diff);

  return patternWave;
}
