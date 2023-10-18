#include <stdio.h>
#include "format.h"
#include <random>
#include "cuPlotter.h"
#include "cudaConfig.h"
#include "memManager.h"
using namespace std;
cuFunc(setHoloRef,(Real* image), (image),{
  cudaIdx()
  if(x <= 4 || y <= 4) image[index] = 1;
  else image[index] = 0;
  if(y > 2 && (y%30 < 15)) image[index] = 0;
});
cuFunc(setStripMask,(Real* image), (image),{
  cuda1Idx()
  int y = index%cuda_column;
  if(y%10 < 5) image[index] = 0;
  else image[index] = 1;
});
int main(int argc, char** argv )
{
  int rows = 256;
  int cols = 256;
  Real* image = (Real*)memMngr.borrowCache(rows*cols*sizeof(Real));
  resize_cuda_image(rows,cols);
  plt.init(rows,cols);
  init_cuda_image();
  setStripMask(image);
  plt.plotFloat(image, MOD, 0, 1, "image");
}
