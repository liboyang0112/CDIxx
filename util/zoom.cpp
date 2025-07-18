#include "fmt/core.h"
#include "imgio.hpp"  //readImage
#include "cudaConfig.hpp" //cuda related
#include "cuPlotter.hpp" //plt
#include <string>

using namespace std;

int main(int argc, char** argv )
{
  if(argc < 2){
    fmt::print("Usage: zoom_run image.png");
    exit(0);
  }
  init_cuda_image();  //always needed
  int row, col;
  Real factor = 1.02;
  Real* img = readImage(argv[1], row, col);  //read the image to memory
  myCuDMalloc(Real, d_intensity, row*col); //allocate GPU memory
  myMemcpyH2D(d_intensity, img, row*col*sizeof(Real)); //copy the image from memory to GPU memory
  ccmemMngr.returnCache(img); //the image on memory is not needed later, therefore we recycled it here.
  myCuDMalloc(complexFormat, d_amp, row*col); //allocate the memory on GPU for complex amplitude, oversampled by oversamplingxoversampling

  init_cuda_image();
  resize_cuda_image(row, col);
  extendToComplex(d_intensity, d_amp);  //create complex wave front according to the intensity and 0 phase

  int newrow = int(row*factor)/2*2;
  int newcol = int(col*factor)/2*2;
  for(int i = 0; i < 100; i++){
    init_fft(row, col); //tell cufft to process the image of this size
    myFFT(d_amp, d_amp);  //execute FFT
    resize_cuda_image(newrow, newcol);
    myCuDMalloc(complexFormat, padded, newrow*newcol);
    padinner(d_amp, padded, row, col, 1./(row*col));
    init_fft(newrow, newcol); //tell cufft to process the image of this size
    myIFFT(padded, padded);  //execute FFT
    resize_cuda_image(row, col);
    crop(padded, d_amp, newrow, newcol);
    plt.init(row, col); //tell cuPlotter to plot the image of this size
    plt.plotComplex(d_amp, MOD, 0, 1, ("zoom" + to_string(i)).c_str(), 1, 0, 1);  //save the mod square to a png file
  }
}

