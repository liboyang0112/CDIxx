#include "experimentConfig.hpp"
#include "cudaConfig.hpp"
#include <math.h>

void opticalPropagate(void* field, Real lambda, Real d, Real imagesize, int n){
  multiplyFresnelPhase_Device((complexFormat*)field, M_PI/lambda/d*(imagesize*imagesize/n));
  cudaConvertFO((complexFormat*)field);
  myFFT((complexFormat*)field, (complexFormat*)field);
  applyNorm((complexFormat*)field, 1./sqrt(n));
  cudaConvertFO((complexFormat*)field);
  multiplyPatternPhase_Device((complexFormat*)field, M_PI*lambda*d/(imagesize*imagesize), 2*d*M_PI/lambda - M_PI/2);
}

void angularSpectrumPropagate(void* input, void*field, Real imagesize_over_lambda, Real z_over_lambda, int n){
  myFFT((complexFormat*)input, (complexFormat*)field);
  applyNorm((complexFormat*)field, 1./n);
  cudaConvertFO((complexFormat*)field);
  multiplyPropagatePhase((complexFormat*)field, 2*M_PI*z_over_lambda, 1./(imagesize_over_lambda*imagesize_over_lambda));
  cudaConvertFO((complexFormat*)field);
  myIFFT((complexFormat*)field, (complexFormat*)field);
}

void experimentConfig::createBeamStop(){
  C_circle cir;
  cir.x0=row/2;
  cir.y0=column/2;
  cir.r=beamStopSize;
  decltype(cir) *cuda_spt;
  cuda_spt = (decltype(cir)*)memMngr.borrowCache(sizeof(cir));
  myMemcpyH2D(cuda_spt, &cir, sizeof(cir));
  beamstop = (Real*)memMngr.borrowCache(row*column*sizeof(Real));
  createMask(beamstop, cuda_spt,1);
  memMngr.returnCache(cuda_spt);
}
void experimentConfig::angularPropagate(void* datain, void* dataout, bool isforward){
  angularSpectrumPropagate(datain, dataout, beamspotsize*oversampling/lambda, (isforward?d:-d)/lambda, row*column);
}
void experimentConfig::propagate(void* datain, void* dataout, bool isforward){
  if(isforward) myFFT((complexFormat*)datain, (complexFormat*)dataout);
  else myIFFT((complexFormat*)datain, (complexFormat*)dataout);
  applyNorm((complexFormat*)dataout, forwardFactor);
}
void experimentConfig::multiplyPatternPhase(void* amp, Real distance){
  if(costheta == 1){
    multiplyPatternPhase_Device((complexFormat*)amp,
         pixelsize*pixelsize*M_PI/(distance*lambda),  2*distance*M_PI/lambda-M_PI/2);
  }else{
    multiplyPatternPhaseOblique_Device((complexFormat*)amp,
         pixelsize*pixelsize*M_PI/(distance*lambda),  2*distance*M_PI/lambda-M_PI/2, costheta);
  }
}
void experimentConfig::multiplyPatternPhase_reverse(void* amp, Real distance){
  if(costheta == 1){
    multiplyPatternPhase_Device((complexFormat*)amp,
        -pixelsize*pixelsize*M_PI/(distance*lambda), -2*distance*M_PI/lambda+M_PI/2);
  }else{
    multiplyPatternPhaseOblique_Device((complexFormat*)amp,
        -pixelsize*pixelsize*M_PI/(distance*lambda), -2*distance*M_PI/lambda+M_PI/2, costheta);
  }
}
void experimentConfig::multiplyPatternPhase_factor(void* amp, Real factor1, Real factor2){
  if(costheta == 1){
    multiplyPatternPhase_Device((complexFormat*)amp, factor1, factor2-M_PI/2);
  }else{
    multiplyPatternPhaseOblique_Device((complexFormat*)amp, factor1, factor2-M_PI/2, costheta);
  }
}
void experimentConfig::multiplyFresnelPhase(void* amp, Real distance){
  Real fresfactor = M_PI*lambda*distance/(sq(pixelsize*row));
  if(costheta == 1){
    multiplyFresnelPhase_Device((complexFormat*)amp, fresfactor);
  }else{
    multiplyFresnelPhaseOblique_Device((complexFormat*)amp, fresfactor, 1./costheta);
  }
}
void experimentConfig::multiplyFresnelPhase_factor(void* amp, Real factor){
  if(costheta == 1){
    multiplyFresnelPhase_Device((complexFormat*)amp, factor);
  }else{
    multiplyFresnelPhaseOblique_Device((complexFormat*)amp, factor, 1./costheta);
  }
}
void experimentConfig::calculateParameters(){
  enhancement = sq(pixelsize)*sqrt(row*column)/(lambda*d); // this guarentee energy conservation
  fresnelFactor = lambda*d/sq(pixelsize)/row/column;
  forwardFactor = 1./sqrt(row*column);
}
