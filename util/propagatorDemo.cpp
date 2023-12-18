#include "imgio.hpp"  //readImage
#include "cudaConfig.hpp" //cuda related
#include "cuPlotter.hpp" //plt
#include "cub_wrap.hpp"
#include <complex>
using namespace std;


int main(int argc, char** argv )
{
  init_cuda_image();  //always needed

  Real s_over_lambda2 = (1./5.42)*1e-6;
  Real dz_over_lambda = 10;

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
  applyNorm(d_wave, 1./(row*col));
  int rowi = row/5;
  int coli = col/5;
  myCuDMalloc(complexFormat, d_crop, rowi*coli); //allocate GPU memory
  plt.init(rowi, coli);
  int handle = plt.initVideo("propagator.hevc", 10);
  multiplyPropagatePhase(d_wave, -100*dz_over_lambda, s_over_lambda2); // a=z/lambda, b = (s/lambda)^2, s is the image size
  for(int i = 0; i < 150; i++){
    resize_cuda_image(row, col);  //tell cuda to process the image of this size
    multiplyPropagatePhase(d_wave, dz_over_lambda, s_over_lambda2); // a=z/lambda, b = (s/lambda)^2, s is the image size
    myIFFT(d_wave, d_propagatedwave);  //execute FFT
    resize_cuda_image(rowi, coli);  //tell cuda to process the image of this size
    crop(d_propagatedwave, d_crop, row, col,mid.real(),mid.imag());
    //crop(d_propagatedwave, d_crop, row, col);
    plt.plotComplex(d_crop, MOD2, 0, 0.3, "test", 0, 0, 1, ("i="+to_string(i)).c_str());
  }
  plt.saveVideo(handle);
}

