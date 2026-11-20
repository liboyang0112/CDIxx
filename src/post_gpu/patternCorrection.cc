#include "patternCorrection.hpp"
#include "cudaConfig.hpp"
#include <math.h>

extern void reflCorrectKernel(Real* input, Real* output, Real* beamstop, float sin_t, float cos_t, float a);
extern void reflSimulateKernel(Real* input, Real* output, float sin_t, float cos_t, float a);
extern void reflCorrectSineKernel(Real* input, Real* output, Real* beamstop, float sin_t, float cos_t, float a);
extern void reflSimulateSineKernel(Real* input, Real* output, float sin_t, float cos_t, float a);
extern void ewaldCorrectKernel(Real* input, Real* output, Real* beamstop, float a, float centerPx, float centerPy);
extern void ewaldSimulateKernel(Real* input, Real* output, float a, float centerPx, float centerPy);

void reflCorrectToRegular(Real* tilted, Real* regular, Real** beamstop, Real tiltDeg, Real distance, Real pixelSize, int row, int column, int useSineLens){
  Real tiltRad = tiltDeg * M_PI / 180.0;
  resize_cuda_image(row, column);
  if(!(*beamstop)){
    myCuMalloc(Real, *beamstop, row*column);
    myDMalloc(Real, h_ones, row*column);
    for(int i = 0; i < row*column; i++) h_ones[i] = 0.0f;
    myMemcpyH2D(*beamstop, h_ones, row*column*sizeof(Real));
    myFree(h_ones);
  }
  if(useSineLens){
    reflCorrectSineKernel(tilted, regular, *beamstop, sinf(tiltRad), cosf(tiltRad), (float)(pixelSize/distance));
  } else {
    reflCorrectKernel(tilted, regular, *beamstop, sinf(tiltRad), cosf(tiltRad),
                      pixelSize/distance);
  }
}

void reflSimulateTilted(Real* regular, Real* tilted, Real tiltDeg, Real distance, Real pixelSize, int row, int column, int useSineLens){
  Real tiltRad = tiltDeg * M_PI / 180.0;
  resize_cuda_image(row, column);
  if(useSineLens){
    reflSimulateSineKernel(regular, tilted, sinf(tiltRad), cosf(tiltRad), (float)(pixelSize/distance));
  } else {
    reflSimulateKernel(regular, tilted, sinf(tiltRad), cosf(tiltRad),
                       pixelSize/distance);
  }
}

void ewaldCorrectToRegular(Real* detector, Real* regular, Real** beamstop, Real a, int row, int column, Real centerPx, Real centerPy){
  resize_cuda_image(row, column);
  if(!(*beamstop)){
    myCuMalloc(Real, *beamstop, row*column);
    myDMalloc(Real, h_ones, row*column);
    for(int i = 0; i < row*column; i++) h_ones[i] = 0.0f;
    myMemcpyH2D(*beamstop, h_ones, row*column*sizeof(Real));
    myFree(h_ones);
  }
  ewaldCorrectKernel(detector, regular, *beamstop, (float)a, (float)centerPx, (float)centerPy);
}

void ewaldSimulateTilted(Real* regular, Real* detector, Real a, int row, int column, Real centerPx, Real centerPy){
  resize_cuda_image(row, column);
  ewaldSimulateKernel(regular, detector, (float)a, (float)centerPx, (float)centerPy);
}
