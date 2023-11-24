#include <stdio.h>
#include "format.h"
#include <random>
#include "cuPlotter.h"
#include "cudaConfig.h"
#include "cudaDefs.h"
#include "memManager.h"
using namespace std;
cuFunc(setHoloRef,(Real* image), (image),{
  cudaIdx()
  if(x <= 4 || y <= 4) image[index] = 1;
  else image[index] = 0;
  if(y > 2 && (y%30 < 15)) image[index] = 0;
});
cuFunc(setStripMask,(Real* image), (image),{
  cudaIdx()
  //if((y > 120 && x > 120)) image[index] = 0;
  //if(x%6 < 4 || y%6 < 4 || (y > 120 && x > 120)) image[index] = 0;
  if(x< 1 || (x == 140 && y==75)) image[index] = 1;
  //if((x > 50 && x < 100) && (y > 50 && y < 100)) image[index] = 0;
  else image[index] = 0;
});
int main(int argc, char** argv )
{
  int rows = 150;
  int cols = 150;
  Real* image = (Real*)memMngr.borrowCache(rows*cols*sizeof(Real));
  resize_cuda_image(rows,cols);
  plt.init(rows,cols);
  init_cuda_image();
  setStripMask(image);
  plt.plotFloat(image, MOD, 0, 1, "image");
}
