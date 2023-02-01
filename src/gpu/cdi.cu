#include <complex>
#include <cassert>
#include <stdio.h>
#include <time.h>
#include <random>

#include <stdio.h>
#include <libconfig.h++>
#include "cufft.h"
#include "common.h"
#include <ctime>
#include "cudaConfig.h"
#include "experimentConfig.h"
#include "tvFilter.h"
#include "cuPlotter.h"
#include "mnistData.h"
#include "cdi.h"

#include <cub/device/device_reduce.cuh>

struct CustomMax
{
  __device__ __forceinline__
    Real operator()(const Real &a, const Real &b) const {
      return (b > a) ? b : a;
    }
};

Real findMax(Real* d_in, int num_items)
{
  Real *d_out = (Real*)memMngr.borrowCache(sizeof(Real));

  void            *d_temp_storage = NULL;
  size_t          temp_storage_bytes = 0;
  CustomMax max_op;
  gpuErrchk(cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, max_op, 0));
  d_temp_storage = (Real*)memMngr.borrowCache(temp_storage_bytes);

  // Run
  gpuErrchk(cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, max_op, 0));
  Real output;
  cudaMemcpy(&output, d_out, sizeof(Real), cudaMemcpyDeviceToHost);

  memMngr.returnCache(d_out);
  memMngr.returnCache(d_temp_storage);
  return output;
}

//#define Bits 16

Real gaussian(Real x, Real y, Real sigma){
  Real r2 = pow(x,2) + pow(y,2);
  return exp(-r2/2/pow(sigma,2));
}

Real gaussian_norm(Real x, Real y, Real sigma){
  return 1./(2*M_PI*sigma*sigma)*gaussian(x,y,sigma);
}





__global__ void takeMod2Diff(complexFormat* a, Real* b, Real *output, Real *bs){
  cudaIdx()
    Real mod2 = pow(a[index].x,2)+pow(a[index].y,2);
  Real tmp = b[index]-mod2;
  if(bs&&bs[index]>0.5) tmp=0;
  else if(b[index]>0.99) tmp = 0.99-mod2;
  output[index] = tmp;
}

__global__ void takeMod2Sum(complexFormat* a, Real* b){
  cudaIdx()
    Real tmp = b[index]+pow(a[index].x,2)+pow(a[index].y,2);
  if(tmp<0) tmp=0;
  b[index] = tmp;
}


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
  applyNorm<<<numBlocks,threadsPerBlock>>>(dataout, isforward? forwardFactorpupil: inverseFactorpupil);
}
void CDI::propagateMid(complexFormat* datain, complexFormat* dataout, bool isforward){
  myCufftExec( *plan, datain, dataout, isforward? CUFFT_FORWARD: CUFFT_INVERSE);
  applyNorm<<<numBlocks,threadsPerBlock>>>(dataout, isforward? forwardFactorMid: inverseFactorMid);
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
  Real* intensity = readImage(common.Intensity.c_str(), row, column);
  size_t sz = row*column*sizeof(Real);
  row*=oversampling;
  column*=oversampling;
  allocateMem();
  Real* d_intensity = (Real*)memMngr.borrowCache(sz); //use the memory allocated;
  cudaMemcpy(d_intensity, intensity, sz, cudaMemcpyHostToDevice);
  ccmemMngr.returnCache(intensity);
  Real* d_phase = 0;
  if(phaseModulation) {
    int tmp;
    Real* phase = readImage(common.Phase.c_str(), tmp,tmp);
    d_phase = support;
    gpuErrchk(cudaMemcpy(d_phase, phase, sz, cudaMemcpyHostToDevice));
    ccmemMngr.returnCache(phase);
  }
  createWaveFront<<<numBlocks,threadsPerBlock>>>(d_intensity, d_phase, (complexFormat*)objectWave, oversampling);
  memMngr.returnCache(d_intensity);
  // if(setups.useRectHERALDO){
  //   pixeltype *rowp;
  //   for(int i = 0; i < row ; i++){
  //     rowp = intensity.ptr<pixeltype>(i);
  //     for(int j = 0; j < column ; j++){
  //       if(i > row/2 || j > column/2) rowp[j] = rcolor-1;
  //     }
  //   }
  // }
  //if(setups.useGaussionLumination){
  //  //setups.spt = &re;
  //  //if(!setups.useShrinkMap) setups.spt = &cir3;
  //  //diffraction image, either from simulation or from experiments.
  //  auto f = [&](int x, int y, fftw_format &data){
  //    auto tmp = (complex<Real>*)&data;
  //    bool inside = cir3.isInside(x,y);
  //    if(!inside) *tmp = 0.;
  //    *tmp *= gaussian(x-cir2.x0,y-cir2.y0,cir3.r);
  //    //if(cir2.isInside(x,y))printf("%f, ",gaussian(x-cir2.x0,y-cir2.y0,cir2.r/2));
  //  };
  //  imageLoop<decltype(f)>(gkp1,&f,0);
  //}
  //if(setups.useGaussionHERALDO){
  //  auto f = [&](int x, int y, fftw_format &data){
  //    auto tmp = (complex<Real>*)&data;
  //    if(cir2.isInside(x,y)) 
  //      *tmp *= gaussian(x-cir2.x0,y-cir2.y0,cir2.r*4);
  //    else *tmp = gaussian(x-cir2.x0,y-cir2.y0,cir2.r*4);
  //    if(x < row*1/3 && y < row*1/3) *tmp = 0;
  //    //if(cir2.isInside(x,y))printf("%f, ",gaussian(x-cir2.x0,y-cir2.y0,cir2.r/2));
  //  };
  //  imageLoop<decltype(f)>(gkp1,&f,0);
  //}
  //if(useRectHERALDO)
  //  setRectHERALDO<<<numBlocks,threadsPerBlock>>>(objectWave, oversampling);
}
void CDI::readPattern(){
  Real* pattern = readImage(common.Pattern.c_str(), row, column);
  allocateMem();
  cudaMemcpy(patternData, pattern, row*column*sizeof(Real), cudaMemcpyHostToDevice);
  ccmemMngr.returnCache(pattern);
  cudaConvertFO<<<numBlocks,threadsPerBlock>>>(patternData);
  applyNorm<<<numBlocks,threadsPerBlock>>>(patternData, 1./exposure);
  printf("Created pattern data\n");
}
void CDI::calculateParameters(){
  experimentConfig::calculateParameters();
  Real beta_HIO = 0.9;
  cudaMemcpyToSymbol(cuda_beta_HIO,&beta_HIO,sizeof(beta_HIO));
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
  cudaF(cudaConvertFO)((complexFormat*)pattern,patternWave);
  cudaF(getMod2)(patternData, patternWave);
  applyRandomPhase<<<numBlocks,threadsPerBlock>>>(patternWave, useBS?beamstop:0, devstates);
}

void CDI::setPattern(void* pattern){
  cudaF(cudaConvertFO)((Real*)pattern,patternData);
  cudaF(createWaveFront)(patternData, 0, patternWave, 1);
  applyRandomPhase<<<numBlocks,threadsPerBlock>>>(patternWave, useBS?beamstop:0, devstates);
}
void CDI::init(){
  allocateMem();
  if(useBS) createBeamStop();
  calculateParameters();
  inittvFilter(row,column);
  createSupport();
  devstates = (curandStateMRG32k3a *)memMngr.borrowCache(column * row * sizeof(curandStateMRG32k3a));
  cudaF(initRand)(devstates);
}
void CDI::prepareIter(){
  if(runSim) {
    if(domnist){
      void* intensity = memMngr.borrowCache(row*column*sizeof(Real));
      void* phase = 0;
      mnist_dat->cuRead(intensity);
      if(phaseModulation) {
        phase = memMngr.borrowCache(row*column*sizeof(Real));
        mnist_dat->cuRead(phase);
      }
      cudaF(createWaveFront)((Real*)intensity, (Real*)phase, (complexFormat*)objectWave, 1);
      memMngr.returnCache(intensity);
      if(phaseModulation) memMngr.returnCache(phase);
      initSupport();
    }
    if(isFresnel) multiplyFresnelPhase(objectWave, d);
    verbose(2,plt.plotComplex(objectWave, MOD2, 0, 1, "inputIntensity", 0));
    verbose(2,plt.plotComplex(objectWave, PHASE, 0, 1, "inputPhase", 0));
    verbose(4,printf("Generating diffraction pattern\n"));
    propagate(objectWave, patternWave, 1);
    getMod2<<<numBlocks,threadsPerBlock>>>(patternData, patternWave);
    if(simCCDbit){
      verbose(4,printf("Applying Poisson noise\n"))
        verbose(1,plt.plotFloat(patternData, MOD, 1, exposure, "theory_pattern", 1));
      cudaF(applyPoissonNoise_WO)(patternData, noiseLevel, devstates);
    }
  }
  if(restart){
    complexFormat *wf = (complexFormat*) readComplexImage(common.restart.c_str());
    cudaMemcpy(patternWave, wf, row*column*sizeof(complexFormat), cudaMemcpyHostToDevice);
    verbose(2,plt.plotComplex(patternWave, MOD2, 1, exposure, "restart_pattern", 1))
      ccmemMngr.returnCache(wf);
  }else {
    createWaveFront<<<numBlocks,threadsPerBlock>>>(patternData, 0, patternWave, 1);
    verbose(1,plt.plotFloat(patternData, MOD, 1, exposure, "init_logpattern", 1))
      plt.plotFloat(patternData, MOD, 1, exposure, ("init_pattern"+save_suffix).c_str(), 0);
    applyRandomPhase<<<numBlocks,threadsPerBlock>>>(patternWave, useBS?beamstop:0, devstates);
  }
}
void CDI::checkAutoCorrelation(){
  size_t sz = row*column*sizeof(Real);
  auto tmp = (complexFormat*)memMngr.useOnsite(sz*2);
  myCufftExecR2C( *planR2C, patternData, (complexFormat*)tmp);// re-use the memory allocated for pupil
  cudaF(fillRedundantR2C)((complexFormat*)tmp, autoCorrelation, 1./sqrt(row*column));
  plt.plotComplex(autoCorrelation, IMAG, 1, 1, "autocorrelation_imag", 1); // only positive values are shown
  plt.plotComplex(autoCorrelation, REAL, 1, 1, "autocorrelation_real", 1); // only positive values are shown
  plt.plotComplex(autoCorrelation, MOD, 1, 1, "autocorrelation", 1);
}
void CDI::createSupport(){
  rect re;
  re.startx = (oversampling_spt-1)/2*row/oversampling_spt-1;
  re.starty = (oversampling_spt-1)/2*column/oversampling_spt-1;
  re.endx = row-re.startx-2;
  re.endy = column-re.starty-2;
  cuda_spt = (rect*)memMngr.borrowCache(sizeof(rect));
  cudaMemcpy(cuda_spt, &re, sizeof(rect), cudaMemcpyHostToDevice);
  support = (Real*)memMngr.borrowCache(row*column*sizeof(Real));
  createMask<<<numBlocks,threadsPerBlock>>>(support, cuda_spt,0);
}
void CDI::initSupport(){
  createMask<<<numBlocks,threadsPerBlock>>>(support, cuda_spt,0);
}
void CDI::saveState(){
  size_t sz = row*column*sizeof(complexFormat);
  void* outputData = ccmemMngr.borrowCache(sz);
  cudaMemcpy(outputData, patternWave, sz, cudaMemcpyDeviceToHost);
  writeComplexImage(common.restart.c_str(), outputData, row, column);//save the step
  ccmemMngr.returnCache(outputData);
}

__global__ void applySupportOblique(complexFormat *gkp1, complexFormat *gkprime, Algorithm algo, Real *spt, int iter = 0, Real fresnelFactor = 0, Real costheta_r = 1){
  cudaIdx()
    bool inside = spt[index] > cuda_threshold;
  complexFormat &gkp1data = gkp1[index];
  complexFormat &gkprimedata = gkprime[index];
  if(algo==RAAR) ApplyRAARSupport(inside,gkp1data,gkprimedata,cuda_beta_HIO);
  else if(algo==ER) ApplyERSupport(inside,gkp1data,gkprimedata);
  else if(algo==HIO) ApplyHIOSupport(inside,gkp1data,gkprimedata,cuda_beta_HIO);
  if(fresnelFactor>1e-4 && iter < 400) {
    if(inside){
      Real phase = M_PI*fresnelFactor*(pow((x-(cuda_row>>1))*costheta_r,2)+pow(y-(cuda_column>>1),2));
      //Real mod = cuCabs(gkp1data);
      Real mod = fabs(gkp1data.x*cos(phase)+gkp1data.y*sin(phase)); //use projection (Error reduction)
      gkp1data.x=mod*cos(phase);
      gkp1data.y=mod*sin(phase);
    }
  }
}


__global__ void applySupport(complexFormat *gkp1, complexFormat *gkprime, Algorithm algo, Real *spt, int iter = 0, Real fresnelFactor = 0){

  cudaIdx()
    bool inside = spt[index] > cuda_threshold;
  complexFormat &gkp1data = gkp1[index];
  complexFormat &gkprimedata = gkprime[index];
  if(algo==RAAR) ApplyRAARSupport(inside,gkp1data,gkprimedata,cuda_beta_HIO);
  else if(algo==ER) ApplyERSupport(inside,gkp1data,gkprimedata);
  //else if(algo==HIO) ApplyPOSHIOSupport(inside,gkp1data,gkprimedata,cuda_beta_HIO);
  else if(algo==HIO) ApplyHIOSupport(inside,gkp1data,gkprimedata,cuda_beta_HIO);
  if(fresnelFactor>1e-4 && iter < 400) {
    if(inside){
      Real phase = M_PI*fresnelFactor*(pow(x-(cuda_row>>1),2)+pow(y-(cuda_column>>1),2));
      //Real mod = cuCabs(gkp1data);
      Real mod = fabs(gkp1data.x*cos(phase)+gkp1data.y*sin(phase)); //use projection (Error reduction)
      gkp1data.x=mod*cos(phase);
      gkp1data.y=mod*sin(phase);
    }
  }
}

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
    cudaF(applyMod)(patternWave,cuda_diff, useBS? beamstop:0, !reconAC || iter > 1000,iter, noiseLevel);
    propagate(patternWave, cuda_gkprime, 0);
    if(costheta == 1) cudaF(applySupport)(cuda_gkp1, cuda_gkprime, (Algorithm)ialgo, support, iter, isFresnel? fresnelFactor:0);
    else cudaF(applySupportOblique)(cuda_gkp1, cuda_gkprime, (Algorithm)ialgo, support, iter, isFresnel? fresnelFactor:0, 1./costheta);
    //update mask
    if(iter%20==0){
      cudaF(getMod)(cuda_objMod,cuda_gkp1);
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
        Real weight;
        for(int i = 0; i < width*width; i++) {
          weight = gaussian((i/width-size),i%width-size, gaussianSigma);
          total+= weight;
          gaussianKernel[i] = weight;
        }
        for(int i = 0; i < width*width; i++)
          gaussianKernel[i] /= total;
        cudaMemcpy(d_gaussianKernel, gaussianKernel, kernelsz, cudaMemcpyHostToDevice);
        applyConvolution<<<numBlocks,threadsPerBlock, pow(size*2+threadsPerBlock.x,2)*sizeof(Real)>>>(cuda_objMod, support, d_gaussianKernel, size, size);

        Real threshold = findMax(support, row*column);
        threshold*=shrinkThreshold;
        cudaMemcpyToSymbol(cuda_threshold, &threshold, sizeof(threshold));

        if(gaussianSigma>1.5) {
          gaussianSigma*=0.99;
        }
      }
      //if(iter%100==0){
      //tvFilterWrap(cuda_objMod, 2e8, 20);
      //applyMod<<<numBlocks,threadsPerBlock>>>(cuda_gkp1, cuda_objMod);
      //}
    }
    propagate( cuda_gkp1, patternWave, 1);
    if(iter%100==0) {
      std::string iterstr = to_string(iter);
      if(saveIter){
        plt.plotComplex(cuda_gkp1, MOD2, 0, 1, ("recon_intensity"+iterstr).c_str(), 0);
        plt.plotComplex(cuda_gkp1, PHASE, 0, 1, ("recon_phase"+iterstr).c_str(), 0);
        plt.plotComplex(patternWave, MOD2, 1, exposure, ("recon_pattern"+iterstr).c_str(), 1);
      }
      if(0){  //Do Total variation denoising during the reconstruction, disabled because not quite effective.
        takeMod2Diff<<<numBlocks,threadsPerBlock>>>(patternWave,patternData, cuda_diff, useBS? beamstop:0);
        cudaConvertFO<<<numBlocks,threadsPerBlock>>>(cuda_diff);
        tvFilterWrap(cuda_diff, noiseLevel, 200);
        cudaConvertFO<<<numBlocks,threadsPerBlock>>>(cuda_diff);
        plt.plotFloat(cuda_diff, MOD, 1, 1, ("smootheddiff"+iterstr).c_str(), 1);
        takeMod2Sum<<<numBlocks,threadsPerBlock>>>(patternWave, cuda_diff);
        plt.plotFloat(cuda_diff, MOD, 1, 1, ("smoothed"+iterstr).c_str(), 1);
      }
    }
  }
  if(gaussianKernel) ccmemMngr.returnCache(gaussianKernel);
  if(d_gaussianKernel) memMngr.returnCache(d_gaussianKernel);
  verbose(2,plt.plotComplex(patternWave, MOD2, 1, exposure, "recon_pattern", 1))
    if(verbose >= 4){
      cudaF(cudaConvertFO)((complexFormat*)cuda_gkp1, cuda_gkprime);
      propagate(cuda_gkprime, cuda_gkprime, 1);
      plt.plotComplex(cuda_gkprime, PHASE, 1, 1, "recon_pattern_phase", 0);
    }
  applyMod<<<numBlocks,threadsPerBlock>>>(patternWave,patternData,useBS?beamstop:0,1,nIter, noiseLevel);
  plt.plotComplex(cuda_gkp1, MOD2, 0, 1, ("recon_intensity"+save_suffix).c_str(), 0);
  plt.plotComplex(cuda_gkp1, PHASE, 0, 1, ("recon_phase"+save_suffix).c_str(), 0);
  if(isFresnel) multiplyFresnelPhase(cuda_gkp1, -d);
  plt.plotComplex(cuda_gkp1, PHASE, 0, 1, ("recon_phase_fresnelRemoved"+save_suffix).c_str(), 0);
  memMngr.returnCache(cuda_gkprime);
  memMngr.returnCache(cuda_objMod);
  memMngr.returnCache(cuda_diff);

  return patternWave;
}
