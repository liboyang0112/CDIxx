#include "experimentConfig.hpp"
#include "cudaConfig.hpp"
#include "fmt/core.h"
#include "misc.hpp"
#include "fmt/core.h"
#include <math.h>

void opticalPropagate(complexFormat* field, Real lambda, Real d, Real imagesize, int n){
  multiplyFresnelPhase_Device(field, M_PI/lambda/d*(imagesize*imagesize/n));
  cudaConvertFO(field);
  myFFT(field, field);
  applyNorm(field, 1./sqrt(n));
  cudaConvertFO(field);
  multiplyPatternPhase_Device(field, M_PI*lambda*d/(imagesize*imagesize), 2*d*M_PI/lambda - M_PI/2);
}

void angularSpectrumPropagate(complexFormat* input, complexFormat*field, Real imagesize_over_lambda, Real z_over_lambda, int n){
  myFFT(input, field);
  applyNorm(field, 1./n);
  cudaConvertFO(field);
  multiplyPropagatePhase(field, 2*M_PI*z_over_lambda, 1./(imagesize_over_lambda*imagesize_over_lambda));
  cudaConvertFO(field);
  myIFFT(field, field);
}

void experimentConfig::createBeamStop(){
  fmt::println("Creating default circular beamstop!");
  beamstop = (Real*)memMngr.borrowCache(row*column*sizeof(Real));
  createCircleMask(beamstop, row>>1, column>>1, beamStopSize, 1);
}
void experimentConfig::angularPropagate(complexFormat* datain, complexFormat* dataout, bool isforward){
  angularSpectrumPropagate(datain, dataout, beamspotsize*oversampling/lambda, (isforward?d:-d)/lambda, row*column);
}
void experimentConfig::propagate(complexFormat* datain, complexFormat* dataout, bool isforward){
  if(isforward) myFFT(datain, dataout);
  else myIFFT(datain, dataout);
  applyNorm(dataout, forwardFactor);
}
void experimentConfig::multiplyPatternPhase(complexFormat* amp, Real distance){
  if(costheta == 1){
    multiplyPatternPhase_Device(amp,
         pixelsize*pixelsize*M_PI/(distance*lambda),  2*distance*M_PI/lambda-M_PI/2);
  }else{
    multiplyPatternPhaseOblique_Device(amp,
         pixelsize*pixelsize*M_PI/(distance*lambda),  2*distance*M_PI/lambda-M_PI/2, costheta);
  }
}
void experimentConfig::multiplyPatternPhase_reverse(complexFormat* amp, Real distance){
  if(costheta == 1){
    multiplyPatternPhase_Device(amp,
        -pixelsize*pixelsize*M_PI/(distance*lambda), -2*distance*M_PI/lambda+M_PI/2);
  }else{
    multiplyPatternPhaseOblique_Device(amp,
        -pixelsize*pixelsize*M_PI/(distance*lambda), -2*distance*M_PI/lambda+M_PI/2, costheta);
  }
}
void experimentConfig::multiplyPatternPhase_factor(complexFormat* amp, Real factor1, Real factor2){
  if(costheta == 1){
    multiplyPatternPhase_Device(amp, factor1, factor2-M_PI/2);
  }else{
    multiplyPatternPhaseOblique_Device(amp, factor1, factor2-M_PI/2, costheta);
  }
}
void experimentConfig::multiplyFresnelPhase(complexFormat* amp, Real distance){
  Real fresfactor = M_PI*lambda*distance/(sq(pixelsize*row));
  multiplyFresnelPhase_factor(amp, fresfactor);
}
void experimentConfig::multiplyFresnelPhase_factor(complexFormat* amp, Real factor){
  if(costheta == 1){
    multiplyFresnelPhase_Device(amp, factor);
  }else{
    multiplyFresnelPhaseOblique_Device(amp, factor, 1./costheta);
  }
}
void experimentConfig::calculateParameters(){
  fresnelFactor = lambda*d/sq(pixelsize)/row/column;
  forwardFactor = 1./sqrt(row*column);
}
