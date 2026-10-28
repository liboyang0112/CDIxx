#include "imgio.hpp"
#include <math.h>
#include <stdint.h>
#include "beamDecomposition.hpp"
#include "cuPlotter.hpp"
#include "cudaConfig.hpp"
#include "misc.hpp"
#include "ptycho.hpp"
//#include <cstring>
int main() {
  int rows = 2048, cols = 2048;
  int pupilsize = 1024;
  init_cuda_image();
  resize_cuda_image(rows, cols);
  init_fft(rows, cols);
  myCuDMalloc(complexFormat, cimage, rows*cols);
  myCuDMalloc(complexFormat, pupil, pupilsize*pupilsize);
  /*
  myCuDMalloc(Real, image, rows*cols);
  myDMalloc(Real, detector, rows*cols);
  memset(detector, 0, rows*cols*sizeof(Real));
  int narray = 3;
  int interval = rows / narray;
  for(int x = 0; x < narray; x++)
    for(int y = 0; y < narray; y++){
      detector[x*interval*cols + y*interval + (cols*interval/2 + interval/2)] = 9;
    }
  myMemcpyH2D(image, detector, rows*cols*sizeof(Real));
  extendToComplex(image, cimage);
  cudaConvertFO(cimage);
  myFFT(cimage, cimage);
  cudaConvertFO(cimage,cimage,1./9);
  createCircleMask(image, (rows>>1) , (rows>>1) , 255);
  //applyGaussMult(image, image, 180, 0, 0, 0);
  multiply(cimage, cimage, image);
  */
  /*
  void* devstate = newRand(rows*cols);
  initRand(devstate, 0);
  random(cimage, devstate);
  myCuDMalloc(Real, image, rows*cols);
  createCircleMask(image, (rows>>1) , (rows>>1) , pupilsize/2);
  multiply(cimage, cimage, image);
  */

  pupilFuncSS(cimage, 50, pupilsize, 3);
  applyNorm(cimage, 2);
  //applyGaussMult(cimage, cimage, 384, 0);
  resize_cuda_image(pupilsize, pupilsize);
  crop(cimage, pupil, rows, cols);
  resize_cuda_image(rows, cols);

  plt.init(rows, cols);
  plt.saveComplex(cimage, "image");
  myFFT(cimage, cimage);
  plt.plotComplex(cimage, MOD2, 1, 1./(rows*cols), "diffraction", 1, 0, 1);
  plt.plotComplex(cimage, MOD2, 1, 1./(rows*cols), "gratings");
  resize_cuda_image(pupilsize, pupilsize);
  plt.init(pupilsize, pupilsize);
  plt.plotComplexColor(pupil, 0, 1, "image");
}
