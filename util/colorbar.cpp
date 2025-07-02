#include "cudaConfig.hpp"
#include "cuPlotter.hpp"

int main(){
  int row=512, col=512;
  init_cuda_image();
  resize_cuda_image(row, col);
  myCuDMalloc(complexFormat, image, row*col);
  createColorbar(image);
  plt.init(row, col);
  plt.plotComplexColor(image, 0, 1, "colorBar", 0, 0);
}
