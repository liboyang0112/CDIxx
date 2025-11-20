#include <unistd.h>
#include "fmt/core.h"
#include "imgio.hpp"
#include "cudaConfig.hpp"
#include "cuPlotter.hpp"
#include "cdi.hpp"
#include <bits/stdc++.h>
//#include <tracy/Tracy.hpp>
using namespace std;
int main(int argc, char** argv )
{
  //ZoneScoped;
  CDI setups(argv[1]);
  if(argc < 2){
    fmt::println("please feed the object intensity and phase image");
  }
  setups.readFiles();
  setups.init();

  //-----------------------configure experiment setups-----------------------------
  fmt::println("Imaging distance = {:4.2f}cm", setups.d*1e-4);
  fmt::println("fresnel factor = {:f}", setups.fresnelFactor);
  fmt::println("enhancement = {:f}", setups.enhancement);

  fmt::println("pupil Imaging distance = {:4.2f}cm", setups.dpupil*1e-4);
  fmt::println("pupil fresnel factor = {:f}", setups.fresnelFactorpupil);
  fmt::println("pupil enhancement = {:f}", setups.enhancementpupil);

  Real fresnelNumber = M_PI*sq(setups.beamspotsize)/(setups.d*setups.lambda);
  fmt::println("Fresnel Number = {:f}",fresnelNumber);

  int sz = setups.row*setups.column*sizeof(complexFormat);
  complexFormat* cuda_pupilAmp = 0, *cuda_ESW = 0, *cuda_ESWP = 0, *cuda_ESWPattern = 0, *cuda_pupilAmp_SIM = 0;
  if(setups.dopupil){
    cuda_pupilAmp = (complexFormat*)memMngr.borrowCache(sz);
    if(setups.runSim) myMemcpyD2D(cuda_pupilAmp, setups.objectWave, sz);
  }
    if(setups.runSim && setups.domnist){
      for(int i = 0; i < setups.mnistN; i++){
        setups.save_suffix = to_string(i);
        setups.prepareIter();
        if(setups.doIteration) setups.phaseRetrieve();
      }
    }else{
      setups.prepareIter();
      if(setups.doIteration){
        setups.phaseRetrieve();
        setups.saveState();
      }
      double smallresidual = setups.residual;
      for(int i = 0; i < setups.nIter; i++){
        setups.prepareIter();
        if(setups.doIteration){
          setups.phaseRetrieve();
          if(smallresidual > setups.residual){
            smallresidual = setups.residual;
            setups.saveState();
          }
        }
      }
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
      myMemcpyD2D(cuda_pupilAmp_SIM, cuda_pupilAmp, sz);
      setups.multiplyFresnelPhase(cuda_pupilAmp, -setups.d);
      setups.multiplyFresnelPhaseMid(cuda_pupilAmp, setups.d-setups.dpupil);
      setups.propagate(cuda_pupilAmp, cuda_pupilAmp, 1);
      cudaConvertFO(cuda_pupilAmp);
      setups.multiplyPatternPhaseMid(cuda_pupilAmp, setups.d-setups.dpupil);

      m_verbose(setups,2,plt.plotComplex(cuda_pupilAmp, MOD2, 0, 1, "ISW"));
      int row, column;
      Real* pupilInput = readImage(setups.pupil.Intensity, row, column);
      myMemcpyH2D(cuda_pupilmod, pupilInput, sz/2);
      ccmemMngr.returnCache(pupilInput);
      createWaveFront(cuda_pupilmod, 0, cuda_ESW, row, column);
      m_verbose(setups,1,plt.plotComplex(cuda_ESW, MOD2, 0, 1, "pupilsample", 0));
      calcESW(cuda_ESW, cuda_pupilAmp);
      m_verbose(setups,2,plt.plotComplex(cuda_ESW, MOD2, 0, 1, "ESW"));

      setups.multiplyFresnelPhase(cuda_ESW, setups.dpupil);
      setups.propagate(cuda_ESW, cuda_ESW, 1);
      setups.multiplyPatternPhase(cuda_ESW, setups.dpupil); //the same effect as setups.multiplyPatternPhase(cuda_pupilAmp, -setups.dpupil);
      m_verbose(setups,2,plt.plotComplex(cuda_ESW, MOD2, 0, setups.exposure, "ESWPattern",1));

      setups.propagate(cuda_pupilAmp_SIM, cuda_pupilAmp_SIM, 1); // equivalent to fftresult
      cudaConvertFO(cuda_pupilAmp_SIM);
      setups.multiplyPatternPhase(cuda_pupilAmp_SIM, setups.d);

      plt.plotComplex(cuda_pupilAmp_SIM, MOD2, 0, setups.exposure, "srcPattern",0);

      add(cuda_pupilAmp_SIM, cuda_ESW);
      getMod2(cuda_pupilmod, cuda_pupilAmp_SIM);
      applyPoissonNoise_WO(cuda_pupilmod, setups.noiseLevel_pupil, setups.devstates, 1./setups.exposurepupil);
      plt.plotFloat(cuda_pupilmod, MOD, 0, setups.exposurepupil, "pupil_logintensity", 1);
      plt.plotComplex(cuda_pupilAmp, PHASE, 0, 1, "pupil_phase", 0);
      plt.plotFloat(cuda_pupilmod, MOD, 0, setups.exposurepupil, setups.pupil.Pattern, 0);
    }else{
      int row, column;
      Real* pattern = readImage(setups.pupil.Pattern,row,column); //reconstruction is better after integerization
      myMemcpyH2D(cuda_pupilmod, pattern, sz/2);
      applyNorm(cuda_pupilmod, 1./setups.exposurepupil);
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
    myMemcpyD2D(cuda_pupilAmp, setups.patternWave, sz);
    //myMemcpyH2D(cuda_pupilAmp, cuda_pupilAmp_SIM, sz);

    cudaConvertFO(cuda_pupilAmp);
    setups.multiplyPatternPhase(cuda_pupilAmp, setups.d);
    setups.multiplyPatternPhase_reverse(cuda_pupilAmp, setups.dpupil);
    plt.plotComplex(cuda_pupilAmp, MOD2, 0, setups.exposure, "amp",0);
    setups.propagate(cuda_pupilAmp, cuda_ISW, 0);

    plt.plotComplex(cuda_ISW, MOD2, 0, 1, "ISW_debug",0);

    initESW(cuda_ESW, cuda_pupilmod, cuda_pupilAmp);
    setups.propagate(cuda_ESW, cuda_ESW, 0);
    myMemcpyD2D(cuda_ESWP, cuda_ESW, sz);
    Real *cuda_steplength, *steplength;//, lengthsum;
    steplength = (Real*)malloc(sz/2);
    cuda_steplength = (Real*)memMngr.borrowCache(sz/2);
    for(int iter = 0; iter < setups.nIterpupil ;iter++){
      applyESWSupport(cuda_ESW, cuda_ISW, cuda_ESWP);
      myMemcpyD2H(steplength, cuda_steplength, sz/2);
      /*
         lengthsum = 0;
         for(int i = 0; i < row*column; i++) lengthsum+=steplength[i];
         if(iter%500==0) printf("step: %d, steplength=%f\n", iter, lengthsum);
         if(lengthsum<1e-6) break;
       */
      setups.propagate(cuda_ESW, cuda_ESWPattern, 1);
      applyESWMod(cuda_ESWPattern, cuda_pupilmod, cuda_pupilAmp);//setups.noiseLevel);
      setups.propagate(cuda_ESWPattern, cuda_ESWP, 0);
    }

    //convert from ESW to object
    setups.propagate(cuda_ESW, cuda_ESWPattern, 1);
    add(cuda_ESWPattern,cuda_pupilAmp);
    plt.plotComplex(cuda_ESWPattern, MOD2, 0, setups.exposurepupil, "ESW_pattern_recon", 1);

    plt.plotComplex(cuda_ESWP, MOD2, 0, 1, "ESW_recon");

    ////applyESWSupport(cuda_ESW, cuda_ISW, cuda_ESWP,cuda_steplength);
    calcO(cuda_ESWP, cuda_ISW);
    plt.plotComplex(cuda_ESWP, MOD2, 0, 1, "object");
  }
  return 0;
}

