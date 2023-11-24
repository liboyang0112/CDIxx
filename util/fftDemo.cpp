#include "imgio.h"  //readImage
#include "cudaConfig.h" //cuda related
#include "cuPlotter.h" //plt
using namespace std;


int main(int argc, char** argv )
{
  init_cuda_image();  //always needed
  int oversampling = 3;

  int row, col;
  Real* img = readImage(argv[1], row, col);  //read the image to memory
  myCuDMalloc(Real, d_intensity, row*col); //allocate GPU memory
  myMemcpyH2D(d_intensity, img, row*col*sizeof(Real)); //copy the image from memory to GPU memory
  ccmemMngr.returnCache(img); //the image on memory is not needed later, therefore we recycled it here.
  myCuDMalloc(complexFormat, d_amp, row*col*sq(oversampling)) //allocate the memory on GPU memory for complex amplitude, oversampled by oversamplingxoversampling

  resize_cuda_image(row*oversampling, col*oversampling);  //tell cuda to process the image of this size
  createWaveFront(d_intensity, 0, d_amp, row, col);  //create complex wave front according to the intensity and 0 phase
  memMngr.returnCache(d_intensity); //the intensity is not needed later, therefore we recycled it here. use memMngr for GPU memory instead of ccmemMngr

  init_fft(row*oversampling, col*oversampling); //tell cufft to process the image of this size
  myFFT(d_amp, d_amp);  //execute FFT

  plt.init(row*oversampling, col*oversampling); //tell cuPlotter to plot the image of this size
  plt.plotComplex(d_amp, MOD2, 1, 1./(row*col*sq(oversampling)), "test", 1, 0, 1);  //save the mod square to a png file
}

