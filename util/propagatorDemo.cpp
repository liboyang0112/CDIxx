#include "fmt/core.h"
#include "imgio.hpp"  //readImage
#include "cudaConfig.hpp" //cuda related
#include "cuPlotter.hpp" //plt
#include "cub_wrap.hpp"
#include "misc.hpp"
#include <complex>
using namespace std;


int main(int argc, char** argv )
{
  if(argc < 2){
    fmt::print("Usage: propagatorDemo_run complex_image.bin");
    exit(0);
  }
  init_cuda_image();  //always needed

  Real s_over_lambda2 = pow((633/(2.37*1024*1000)),2);
  Real dz_over_lambda = 2*M_PI*0.1e6/633;

  int row, col;
  complexFormat* wave = (complexFormat*)readImage(argv[1], row, col);  //read the image to memory
  myCuDMalloc(complexFormat, d_wave, row*col); //allocate GPU memory
  myCuDMalloc(Real, d_mod2, row*col); //allocate GPU memory
  myMemcpyH2D(d_wave, wave,row*col*sizeof(complexFormat));
  ccmemMngr.returnCache(wave); //the image on memory is not needed later, therefore we recycled it here.
  myCuDMalloc(complexFormat, d_propagatedwave, row*col) //allocate the memory on GPU memory for complex amplitude, oversampled by oversampling x oversampling

  resize_cuda_image(row, col);  //tell cuda to process the image of this size
  plt.init(row, col); //tell cuPlotter to plot the image of this size
  init_fft(row, col); //tell cufft to process the image of this size
  getMod2(d_mod2, d_wave);
  getMod2(d_mod2, d_mod2);
  std::complex<Real> mid(findMiddle(d_mod2, row*col));
  plt.plotComplex(d_wave, MOD2, 0, 0.1, "bftest", 0, 0, 1);  //save the mod square to a png file
  plt.plotComplex(d_wave, PHASE, 0, 1, "bftest_phase", 0, 0, 0);  //save the mod square to a png file
  myFFT(d_wave, d_wave);  //execute FFT
  cudaConvertFO(d_wave);
  applyNorm(d_wave, 1./(row*col));
  int rowi = row;
  int coli = col;
  myCuDMalloc(complexFormat, d_crop, rowi*coli); //allocate GPU memory
  plt.init(rowi, coli);
  int handle = plt.initVideo("propagator.mp4", 4, 0, 0);
  multiplyPropagatePhase(d_wave, -100*dz_over_lambda, s_over_lambda2); // a=z/lambda, b = (s/lambda)^2, s is the image size
  for(int i = 0; i < 100; i++){
    resize_cuda_image(row, col);  //tell cuda to process the image of this size
    multiplyPropagatePhase(d_wave, dz_over_lambda, s_over_lambda2); // a=z/lambda, b = (s/lambda)^2, s is the image size
    cudaConvertFO(d_wave, d_propagatedwave);
    myIFFT(d_propagatedwave, d_propagatedwave);  //execute FFT
    resize_cuda_image(rowi, coli);  //tell cuda to process the image of this size
    crop(d_propagatedwave, d_crop, row, col,mid.real(),mid.imag());
    //crop(d_propagatedwave, d_crop, row, col);
    plt.plotComplexColor(d_crop, 0, 0.3, ("test_" + to_string(i)).c_str(), 0, 0);
  }
  plt.saveVideo(handle);
}

