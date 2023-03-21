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
#include "cuPlotter.h"
#include "mnistData.h"

#include "cdi.h"

cuFunc(applyESWSupport,(complexFormat* ESW, complexFormat* ISW, complexFormat* ESWP, Real* length),(ESW,ISW,ESWP,length),{
  cudaIdx()
    auto tmp = ISW[index];
  auto tmp2 = ESWP[index];
  auto sum = cuCaddf(tmp,ESWP[index]);
  //these are for amplitude modulation only
  Real prod = tmp.x*tmp2.x+tmp.y*tmp2.y;
  if(prod>0) prod=0;
  if(prod<-2) prod = -2;
  auto rmod2 = 1./(tmp.x*tmp.x+tmp.y*tmp.y);
  ESW[index].x = prod*tmp.x*rmod2;
  ESW[index].y = prod*tmp.y*rmod2;
  /*
     if(cuCabsf(tmp) > cuCabsf(sum)) {
     ESW[index] = ESWP[index];
     length[index] = 0;
     return;
     }
     Real factor = cuCabsf(tmp)/cuCabsf(sum);
     if(x<vars->rows/3||x>vars->rows*2/3||y<cuda_column||y>2*cuda_column/3) factor = 0;
     ESW[index].x = factor*sum.x-tmp.x;
     ESW[index].y = factor*sum.y-tmp.y;

     ESW[index].x -= vars->beta_HIO*(1-factor)*sum.x;
     ESW[index].y -= vars->beta_HIO*(1-factor)*sum.y;
     length[index] = 1;
   */
})
cuFunc(initESW,(complexFormat* ESW, Real* mod, complexFormat* amp),(ESW,mod,amp),{
  cudaIdx()
    auto tmp = amp[index];
  if(cuCabsf(tmp)<=1e-10) {
    ESW[index] = tmp;
    return;
  }
  if(mod[index]<=0) {
    ESW[index].x = -tmp.x;
    ESW[index].y = -tmp.y;
    return;
  }
  Real factor = sqrtf(mod[index])/cuCabsf(tmp)-1;
  ESW[index].x = factor*tmp.x;
  ESW[index].y = factor*tmp.y;
})
cuFunc(applyESWMod,(complexFormat* ESW, Real* mod, complexFormat* amp, int noiseLevel),(ESW,mod,amp,noiseLevel),{
  cudaIdx()
    Real tolerance = 0;//1./vars->rcolor*vars->scale+1.5*sqrtf(noiseLevel)/vars->rcolor; // fluctuation caused by bit depth and noise
  auto tmp = amp[index];
  auto sum = cuCaddf(ESW[index],tmp);
  Real mod2 = mod[index];
  if(mod2<=0){
    ESW[index].x = -tmp.x;
    ESW[index].y = -tmp.y;
    return;
  }
  Real factor = 0;
  if(cuCabsf(sum)>1e-10){
    //factor = mod[index]/cuCabsf(sum);
    Real mod2s = sum.x*sum.x+sum.y*sum.y;
    if(mod2+tolerance < mod2s) factor = sqrt((mod2+tolerance)/mod2s);
    else if(mod2-tolerance > mod2s) factor = sqrt((mod2-tolerance)/mod2s);
    else factor=1;
  }
  //if(mod[index] >= 0.99) factor = max(0.99/cuCabsf(sum), 1.);
  //printf("factor=%f, mod=%f, sum=%f\n", factor, mod[index], cuCabsf(sum));
  ESW[index].x = factor*sum.x-tmp.x;
  ESW[index].y = factor*sum.y-tmp.y;
})

cuFunc(calcESW,(complexFormat* sample, complexFormat* ISW),(sample,ISW),{
  cudaIdx()
    complexFormat tmp = sample[index];
  tmp.x = -tmp.x;  // Here we reverse the image, use tmp.x = tmp.x - 1 otherwise;
                   //Real ttmp = tmp.y;
                   //tmp.y=tmp.x;   // We are ignoring the factor (-i) each time we do fresnel propagation, which causes this transform in the ISW. ISW=iA ->  ESW=(O-1)A=(i-iO)ISW
                   //tmp.x=ttmp;
  sample[index]=cuCmulf(tmp,ISW[index]);
})

cuFunc(calcO,(complexFormat* ESW, complexFormat* ISW),(ESW,ISW),{
  cudaIdx()
    if(cuCabsf(ISW[index])<1e-4) {
      ESW[index].x = 0;
      ESW[index].y = 0;
      return;
    }
  complexFormat tmp = cuCdivf(ESW[index],ISW[index]);
  /*
     Real ttmp = tmp.y;
     tmp.y=tmp.x;   
     tmp.x=1-ttmp;
   */
  ESW[index].x=1+tmp.x;
  ESW[index].y=tmp.y;
})

cuFunc(applyAutoCorrelationMod,(complexFormat* source,complexFormat* target, Real *bs = 0),(source,target,bs),{
  cudaIdx()
  Real targetdata = target[index].x;
  Real retval = targetdata;
  source[index].y = 0;
  Real maximum = pow(mergeDepth,2)*vars->scale*0.99;
  Real sourcedata = source[index].x;
  Real tolerance = 0.5/vars->rcolor*vars->scale;
  Real diff = sourcedata-targetdata;
  if(bs && bs[index]>0.5) {
    if(targetdata<0) target[index].x = 0;
    return;
  }
  if(diff>tolerance){
    retval = targetdata+tolerance;
  }else if(diff < -tolerance ){
    retval = targetdata-tolerance;
  }else{
    retval = targetdata;
  }
  if(targetdata>=maximum) {
    retval = max(sourcedata,maximum);
  }
  source[index].x = retval;
})

int main(int argc, char** argv )
{
  CDI setups(argv[1]);
  cudaFree(0); // to speed up the cudaMalloc; https://forums.developer.nvidia.com/t/cudamalloc-slow/40238
  if(argc < 2){
    printf("please feed the object intensity and phase image\n");
  }
  setups.readFiles();
  setups.init();

  //-----------------------configure experiment setups-----------------------------
  printf("Imaging distance = %4.2fcm\n", setups.d*1e-4);
  printf("forward norm = %f\n", setups.forwardFactor);
  printf("backward norm = %f\n", setups.inverseFactor);
  printf("fresnel factor = %f\n", setups.fresnelFactor);
  printf("enhancement = %f\n", setups.enhancement);

  printf("pupil Imaging distance = %4.2fcm\n", setups.dpupil*1e-4);
  printf("pupil forward norm = %f\n", setups.forwardFactorpupil);
  printf("pupil backward norm = %f\n", setups.inverseFactorpupil);
  printf("pupil fresnel factor = %f\n", setups.fresnelFactorpupil);
  printf("pupil enhancement = %f\n", setups.enhancementpupil);

  Real fresnelNumber = M_PI*pow(setups.beamspotsize,2)/(setups.d*setups.lambda);
  printf("Fresnel Number = %f\n",fresnelNumber);

  int sz = setups.row*setups.column*sizeof(complexFormat);
  complexFormat* cuda_pupilAmp, *cuda_ESW, *cuda_ESWP, *cuda_ESWPattern, *cuda_pupilAmp_SIM;
  if(setups.dopupil){
    cuda_pupilAmp = (complexFormat*)memMngr.borrowCache(sz);
    if(setups.runSim) cudaMemcpy(cuda_pupilAmp, setups.objectWave, sz, cudaMemcpyDeviceToDevice);
  }
  if(setups.doIteration) {
    if(setups.runSim && setups.domnist){
      for(int i = 0; i < setups.mnistN; i++){
        setups.save_suffix = to_string(i);
        setups.prepareIter();
        setups.phaseRetrieve(); 
      }
    }else{
      setups.prepareIter();
      setups.phaseRetrieve(); 
    }
    setups.saveState();
  }
  setups.checkAutoCorrelation();

  //Now let's do pupil
  if(setups.dopupil){ 
    Real* cuda_pupilmod;
    cuda_pupilmod = (Real*)memMngr.borrowCache(sz/2);
    cuda_pupilAmp_SIM = (complexFormat*)memMngr.borrowCache(sz);
    cuda_ESW = (complexFormat*)memMngr.borrowCache(sz);
    cuda_ESWP = (complexFormat*)memMngr.borrowCache(sz);
    cuda_ESWPattern = (complexFormat*)memMngr.borrowCache(sz);
    //cuda_debug = (complexFormat*)memMngr.borrowCache(sz);
    if(setups.runSim){
      cudaMemcpy(cuda_pupilAmp_SIM, cuda_pupilAmp, sz, cudaMemcpyDeviceToDevice);
      setups.multiplyFresnelPhase(cuda_pupilAmp, -setups.d);
      setups.multiplyFresnelPhaseMid(cuda_pupilAmp, setups.d-setups.dpupil);
      setups.propagateMid(cuda_pupilAmp, cuda_pupilAmp, 1);
      cudaF(cudaConvertFO,cuda_pupilAmp);
      setups.multiplyPatternPhaseMid(cuda_pupilAmp, setups.d-setups.dpupil);

      m_verbose(setups,2,plt.plotComplex(cuda_pupilAmp, MOD2, 0, 1, "ISW"));
      int row, column;
      Real* pupilInput = readImage(setups.pupil.Intensity.c_str(), row, column);
      cudaMemcpy(cuda_pupilmod, pupilInput, sz/2, cudaMemcpyHostToDevice);
      ccmemMngr.returnCache(pupilInput);
      cudaF(createWaveFront,cuda_pupilmod, 0, cuda_ESW, row, column);
      m_verbose(setups,1,plt.plotComplex(cuda_ESW, MOD2, 0, 1, "pupilsample", 0));
      cudaF(calcESW,cuda_ESW, cuda_pupilAmp);
      m_verbose(setups,2,plt.plotComplex(cuda_ESW, MOD2, 0, 1, "ESW"));

      setups.multiplyFresnelPhase(cuda_ESW, setups.dpupil);
      setups.propagatepupil(cuda_ESW, cuda_ESW, 1);
      setups.multiplyPatternPhase(cuda_ESW, setups.dpupil); //the same effect as setups.multiplyPatternPhase(cuda_pupilAmp, -setups.dpupil);
      m_verbose(setups,2,plt.plotComplex(cuda_ESW, MOD2, 0, setups.exposure, "ESWPattern",1));

      setups.propagate(cuda_pupilAmp_SIM, cuda_pupilAmp_SIM, 1); // equivalent to fftresult
      cudaF(cudaConvertFO,cuda_pupilAmp_SIM);
      setups.multiplyPatternPhase(cuda_pupilAmp_SIM, setups.d);

      plt.plotComplex(cuda_pupilAmp_SIM, MOD2, 0, setups.exposure, "srcPattern",0);

      cudaF(add,cuda_pupilAmp_SIM, cuda_ESW);
      cudaF(getMod2,cuda_pupilmod, cuda_pupilAmp_SIM);
      cudaF(applyPoissonNoise_WO,cuda_pupilmod, setups.noiseLevel_pupil, setups.devstates, 1./setups.exposurepupil);
      plt.plotFloat(cuda_pupilmod, MOD, 0, setups.exposurepupil, "pupil_logintensity", 1);
      plt.plotComplex(cuda_pupilAmp, PHASE, 0, 1, "pupil_phase", 0);
      plt.plotFloat(cuda_pupilmod, MOD, 0, setups.exposurepupil, setups.pupil.Pattern.c_str(), 0);
    }else{
      int row, column;
      Real* pattern = readImage(setups.pupil.Pattern.c_str(),row,column); //reconstruction is better after integerization
      cudaMemcpy(cuda_pupilmod, pattern, sz/2, cudaMemcpyHostToDevice);
      cudaF(applyNorm,cuda_pupilmod, 1./setups.exposurepupil);
      ccmemMngr.returnCache(pattern);
    }
    //pupil reconstruction needs:
    //  1. fftresult from previous phaseRetrieve
    //  2. pupil pattern.
    //Storage:
    //  1. Amp_i
    //  2. ESW
    //  3. ISW
    //  4. sqrt(pupilmod2)
    //
    //pupil reconstruction procedure:
    //      Amp_i = PatternPhase_d/PatternPhase_pupil*fftresult
    //  1. ISW = IFFT(Amp_i)
    //  2. ESW = IFFT{(sqrt(pupilmod2)/mod(Amp_i)-1)*(Amp_i)}
    //  3. validate: |FFT(ISW+ESW)| = sqrt(pupilmod2)
    //  4. if(|ESW+ISW| > |ISW|) ESW' = |ISW|/|ESW+ISW|*(ESW+ISW)-ISW
    //     else ESW'=ESW
    //     ESW'->ESW
    //  5. ESWfft = FFT(ESW)
    //  6. ESWfft' = sqrt(pupilmod2)/|Amp_i+ESWfft|*(Amp_i+ESWfft)-Amp_i
    //  7. ESW = IFFT(ESWfft')
    //  repeat from step 4

    complexFormat* cuda_ISW;
    cuda_ISW = (complexFormat*)memMngr.borrowCache(sz);
    cudaMemcpy(cuda_pupilAmp, setups.patternWave, sz, cudaMemcpyDeviceToDevice);
    //cudaMemcpy(cuda_pupilAmp, cuda_pupilAmp_SIM, sz, cudaMemcpyHostToDevice);

    cudaF(cudaConvertFO,cuda_pupilAmp);
    setups.multiplyPatternPhase(cuda_pupilAmp, setups.d);
    setups.multiplyPatternPhase_reverse(cuda_pupilAmp, setups.dpupil);
    plt.plotComplex(cuda_pupilAmp, MOD2, 0, setups.exposure, "amp",0);
    setups.propagatepupil(cuda_pupilAmp, cuda_ISW, 0);

    plt.plotComplex(cuda_ISW, MOD2, 0, 1, "ISW_debug",0);

    cudaF(initESW,cuda_ESW, cuda_pupilmod, cuda_pupilAmp);
    setups.propagatepupil(cuda_ESW, cuda_ESW, 0);
    cudaMemcpy(cuda_ESWP, cuda_ESW, sz, cudaMemcpyDeviceToDevice);
    Real *cuda_steplength, *steplength;//, lengthsum;
    steplength = (Real*)malloc(sz/2);
    cuda_steplength = (Real*)memMngr.borrowCache(sz/2);
    for(int iter = 0; iter < setups.nIterpupil ;iter++){
      cudaF(applyESWSupport,cuda_ESW, cuda_ISW, cuda_ESWP, cuda_steplength);
      cudaMemcpy(steplength, cuda_steplength, sz/2, cudaMemcpyDeviceToHost);
      /*
         lengthsum = 0;
         for(int i = 0; i < row*column; i++) lengthsum+=steplength[i];
         if(iter%500==0) printf("step: %d, steplength=%f\n", iter, lengthsum);
         if(lengthsum<1e-6) break;
       */
      setups.propagatepupil(cuda_ESW, cuda_ESWPattern, 1);
      cudaF(applyESWMod,cuda_ESWPattern, cuda_pupilmod, cuda_pupilAmp, 0);//setups.noiseLevel);
      setups.propagatepupil(cuda_ESWPattern, cuda_ESWP, 0);
    }

    //convert from ESW to object
    setups.propagatepupil(cuda_ESW, cuda_ESWPattern, 1);
    cudaF(add,cuda_ESWPattern,cuda_pupilAmp);
    plt.plotComplex(cuda_ESWPattern, MOD2, 0, setups.exposurepupil, "ESW_pattern_recon", 1);

    plt.plotComplex(cuda_ESWP, MOD2, 0, 1, "ESW_recon");

    //cudaF(//applyESWSupport,cuda_ESW, cuda_ISW, cuda_ESWP,cuda_steplength);
    cudaF(calcO,cuda_ESWP, cuda_ISW);
    plt.plotComplex(cuda_ESWP, MOD2, 0, 1, "object");
  }
  return 0;
}

