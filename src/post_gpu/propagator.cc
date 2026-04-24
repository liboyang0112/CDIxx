#include "propagator.hpp"
#include "cudaConfig.hpp"
#include "fmt/core.h"
#include "misc.hpp"
#include "fmt/core.h"
#include <math.h>

void multiplyPatternPhase_Device(complexFormat* amp, Real r_d_lambda, Real d_r_lambda); //cuda functions
void multiplyFresnelPhase_Device(complexFormat* amp, Real phaseFactor); //cuda functions


void propagator::multiplyPatternPhase(complexFormat* amp){
    multiplyPatternPhase_Device(amp, pixelsize*pixelsize*M_PI/(distance*lambda),  2*distance*M_PI/lambda-M_PI/2);
}
void propagator::multiplyFresnelPhase(complexFormat* amp){
  multiplyFresnelPhase_Device(amp, M_PI*lambda*distance/(sq(pixelsize*row)));
}
void propagator::removeFresnelPhase(complexFormat* amp){
  multiplyFresnelPhase_Device(amp, -M_PI*lambda*distance/(sq(pixelsize*row)));
}

void propagator::fresnelPropagate(complexFormat* field){
  multiplyFresnelPhase(field);
  cudaConvertFO(field);
  myFFT(field, field);
  applyNorm(field, 1./sqrt(row*column));
  cudaConvertFO(field);
  multiplyPatternPhase(field);
}

void propagator::angularSpectrumPropagate(complexFormat* input, complexFormat* output){
  myFFT(input, output);
  applyNorm(output, 1./(row*column));
  cudaConvertFO(output);
  Real lambda_over_imagesize = lambda/(pixelsize*row);
  multiplyPropagatePhase(output, 2*M_PI*distance/lambda, lambda_over_imagesize*lambda_over_imagesize);
  cudaConvertFO(output);
  myIFFT(output, output);
}

void propagator::angularSpectrumPropagateReverse(complexFormat* input, complexFormat* output){
  myFFT(input, output);
  applyNorm(output, 1./(row*column));
  cudaConvertFO(output);
  Real imagesize_over_lambda = pixelsize*row/lambda;
  multiplyPropagatePhase(output, -2*M_PI*distance/lambda, 1./(imagesize_over_lambda*imagesize_over_lambda));
  cudaConvertFO(output);
  myIFFT(output, output);
}

