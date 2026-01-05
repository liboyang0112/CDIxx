#include "imgio.hpp"
#include "time.h"
#include <math.h>
#include <random>
#include <stdint.h>
#include <string.h>
#include "beamDecomposition.hpp"
#include "cuPlotter.hpp"
#include "cudaConfig.hpp"
const uint16_t maxpix = 0xffff;
uint16_t inline setHoloRef(int x, int y) {
  if (x <= 4 || y <= 4)
    return maxpix;
  else
    return 0;
  if (y > 2 && (y % 30 < 15))
    return 0;
}
uint16_t inline setStripMask(int x, int y) {
  // if((y > 120 && x > 120)) image[index] = 0;
  // if(x%6 < 4 || y%6 < 4 || (y > 120 && x > 120)) image[index] = 0;
  if (x < 1 || (x == 140 && y == 75))
    return maxpix;
  // if((x > 50 && x < 100) && (y > 50 && y < 100)) image[index] = 0;
  else
    return 0;
};
uint16_t inline setHole(int x, int y) {
  Real r = sqrt(sqSum(x - 128, y - 128));
  if (r < 3)
    return maxpix;
  return 0;
};
void setHoleArray(uint16_t* arr, int rows, int cols, int randstep) {
  auto seed = time(NULL);
  std::default_random_engine generator(seed);
  std::normal_distribution<double> distribution(0.0, 1.);
  int offset = 1;
  int r = 2, interval = 12;
  int r2 = r * r;
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      int centx = offset + i * interval + distribution(generator)*randstep;
      int centy = offset + j * interval + distribution(generator)*randstep;
      for(int x = centx - r; x < centx+r; x++){
      for(int y = centy - r; y < centy+r; y++){
        if(x < 0 || x >= rows || y < 0 || y >= cols) continue;
         Real dist = sqSum(centx - x, centy - y);
         if (dist < r2)
           arr[x*rows+y] = maxpix;
      }}
    }
  }
};
void setrandomFTHArray(uint16_t* arr, int cols) {
  auto seed = time(NULL);
  std::default_random_engine generator(seed);
  std::normal_distribution<double> distribution(0.0, rcolor);
  for(int i = 128; i < 192; i++) for(int j = 128; j < 192; j++){
    //arr[i*cols + j] = distribution(generator);
    //arr[(rows-i-1)*cols + cols-j-1] = distribution(generator);
    arr[(i)*cols + cols-j-1] = distribution(generator);
  }
};

void setbars(uint16_t* arr, int cols) {
  for(int i = 0 ; i < 52; i++){
    for(int j = 0 ; j < 52; j++){
      //if((j/10)%2 == 0) arr[i*cols+j] = maxpix;
      if((j/10)%2 == 0) arr[(i)*cols+j] = maxpix;
      if((i/5)%2 == 0 && i < 26 && j < 26) arr[(i+60)*cols+j] = maxpix;
      if((j/3)%2 == 0 && i < 15 && j < 15) arr[(i+60)*cols+j+36] = maxpix;
      //if((j/4)%2 == 0) arr[(i+60)*cols+j+60] = maxpix;
    }
  }
};

//void setrandom(uint16_t* image){
//  for()
//
//}

//int main(int argc, char **argv) {
int main() {
  /* // for DMD
  int rows = 1920, cols = 1080;
  myDMalloc(uint16_t, image, rows * cols);
  memset(image, 0, sizeof(uint16_t)*rows*cols);
  setbars(image, cols);
  //setHoleArray(image, rows, cols, 0);
  writePng("mask.png", image, rows, cols, 16, 0);
  */ 
  int rows = 1000, cols = 1000;
  init_cuda_image();
  resize_cuda_image(rows, cols);
  myCuDMalloc(Real, image, rows*cols);
  myCuDMalloc(Real, image1, rows*cols);
  C_circle spt;
  spt.r = 234;
  //spt.r = 13;
  spt.x0=spt.y0 = (rows>>1)+10;
  myCuDMalloc(C_circle, d_spt, 1);
  myMemcpyH2D(d_spt, &spt, sizeof(C_circle));
  createMask(image, d_spt);
  applyGaussMult(image, image, 130, 0);
  //multiplyHermit(image, image, 100, 3,3);
  //rotate(image, image1, M_PI/4);
  //applyNorm(image1,4);
  //getMod2(image, image1);
  plt.init(rows, cols);
  plt.plotFloat(image, MOD, 0, 1, "image");
  plt.saveFloat(image, "image");
}
