#include "experimentConfig.h"
#include "cudaConfig.h"

// pixsize*pixsize*M_PI/(d*lambda) and 2*d*M_PI/lambda
cuFunc(multiplyPatternPhase_Device,(complexFormat* amp, Real r_d_lambda, Real d_r_lambda),(amp,r_d_lambda,d_r_lambda),{
  cudaIdx()
  Real phase = (sq(x-(cuda_row>>1))+sq(y-(cuda_column>>1)))*r_d_lambda+d_r_lambda;
  complexFormat p = {cos(phase),sin(phase)};
  amp[index] = cuCmulf(amp[index], p);
})

cuFunc(multiplyPatternPhaseOblique_Device,(complexFormat* amp, Real r_d_lambda, Real d_r_lambda, Real costheta),(amp,r_d_lambda,d_r_lambda,costheta),{ // pixsize*pixsize*M_PI/(d*lambda) and 2*d*M_PI/lambda and costheta = z/r
  cudaIdx()
  Real phase = (sq((x-(cuda_row>>1)*costheta))+sq(y-(cuda_column>>1)))*r_d_lambda+d_r_lambda;
  complexFormat p = {cos(phase),sin(phase)};
  amp[index] = cuCmulf(amp[index], p);
})

cuFunc(multiplyFresnelPhase_Device,(complexFormat* amp, Real phaseFactor),(amp,phaseFactor),{ // pixsize*pixsize*M_PI/(d*lambda) and 2*d*M_PI/lambda
  cudaIdx()
  Real phase = phaseFactor*(sq(x-(cuda_row>>1))+sq(y-(cuda_column>>1)));
  complexFormat p = {cos(phase),sin(phase)};
  if(cuCabsf(amp[index])!=0) amp[index] = cuCmulf(amp[index], p);
})

cuFunc(multiplyFresnelPhaseOblique_Device,(complexFormat* amp, Real phaseFactor, Real costheta_r),(amp,phaseFactor,costheta_r),{ // costheta_r = 1./costheta = r/z
  cudaIdx()
  Real phase = phaseFactor*(sq((x-(cuda_row>>1))*costheta_r)+sq(y-(cuda_column>>1)));
  complexFormat p = {cos(phase),sin(phase)};
  if(cuCabsf(amp[index])!=0) amp[index] = cuCmulf(amp[index], p);
})

void opticalPropagate(void* field, Real lambda, Real d, Real imagesize){
  multiplyFresnelPhase_Device((complexFormat*)field, M_PI/lambda/d*(imagesize*imagesize/(cuda_imgsz.x*cuda_imgsz.y)));
  cudaConvertFO((complexFormat*)field);
  myCufftExec(*plan, (complexFormat*)field, (complexFormat*)field, CUFFT_FORWARD);
  applyNorm((complexFormat*)field, 1./sqrt(cuda_imgsz.x*cuda_imgsz.y));
  cudaConvertFO((complexFormat*)field);
  multiplyPatternPhase_Device((complexFormat*)field, M_PI*lambda*d/(imagesize*imagesize), 2*d*M_PI/lambda - M_PI/2);
}

cuFunc(multiplyPropagatePhase,(complexFormat* amp, Real a, Real b),(amp,a,b),{
  cudaIdx();
  complexFormat phasefactor;
  Real phase = a*sqrt(1-(sq(x-(cuda_row>>1))+sq(y-(cuda_column>>1)))*b);
  phasefactor.x = cos(phase);
  phasefactor.y = sin(phase);
  amp[index] = cuCmulf(amp[index],phasefactor);
})
void angularSpectrumPropagate(void* input, void*field, Real imagesize_over_lambda, Real z_over_lambda){
  myCufftExec(*plan, (complexFormat*)input, (complexFormat*)field, CUFFT_FORWARD);
  applyNorm((complexFormat*)field, 1./(cuda_imgsz.x*cuda_imgsz.y));
  cudaConvertFO((complexFormat*)field);
  multiplyPropagatePhase((complexFormat*)field, 2*M_PI*z_over_lambda, 1./(imagesize_over_lambda*imagesize_over_lambda));
  cudaConvertFO((complexFormat*)field);
  myCufftExec(*plan, (complexFormat*)field, (complexFormat*)field, CUFFT_INVERSE);
}


void experimentConfig::createBeamStop(){
  C_circle cir;
  cir.x0=row/2;
  cir.y0=column/2;
  cir.r=beamStopSize;
  decltype(cir) *cuda_spt;
  cuda_spt = (decltype(cir)*)memMngr.borrowCache(sizeof(cir));
  cudaMemcpy(cuda_spt, &cir, sizeof(cir), cudaMemcpyHostToDevice);
  beamstop = (Real*)memMngr.borrowCache(row*column*sizeof(Real));
  createMask(beamstop, cuda_spt,1);
  memMngr.returnCache(cuda_spt);
}
void experimentConfig::angularPropagate(void* datain, void* dataout, bool isforward){
  angularSpectrumPropagate(datain, dataout, beamspotsize*oversampling/lambda, (isforward?d:-d)/lambda);
}
void experimentConfig::propagate(void* datain, void* dataout, bool isforward){
  myCufftExec( *plan, (complexFormat*)datain, (complexFormat*)dataout, isforward? CUFFT_FORWARD: CUFFT_INVERSE);
  applyNorm((complexFormat*)dataout, isforward? forwardFactor: inverseFactor);
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
  forwardFactor = fresnelFactor*enhancement;
  inverseFactor = 1./row/column/forwardFactor;
}
