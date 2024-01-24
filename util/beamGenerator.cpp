#include <stdint.h>
#include <math.h>
#include "memManager.hpp"
#include "imgio.hpp"
const uint16_t maxpix = 0xffff;
uint16_t inline setHoloRef(int x, int y){
  if(x <= 4 || y <= 4) return maxpix;
  else return 0;
  if(y > 2 && (y%30 < 15)) return 0;
}
uint16_t inline setStripMask(int x, int y){
  //if((y > 120 && x > 120)) image[index] = 0;
  //if(x%6 < 4 || y%6 < 4 || (y > 120 && x > 120)) image[index] = 0;
  if(x< 1 || (x == 140 && y==75)) return maxpix;
  //if((x > 50 && x < 100) && (y > 50 && y < 100)) image[index] = 0;
  else return 0;
};
uint16_t inline setHole(int x, int y){
  Real r = sqrt(sqSum(x-128, y-128));
  if(r < 3) return maxpix;
  return 0;
};
uint16_t inline setHoleArray(int x, int y){
  int offset = 15;
  int r = 8, interval = 25;
  int r2 = r*r;
  for(int i = 0; i < 10; i++){
  for(int j = 0; j < 10; j++){
    int centx = offset + i*interval;
    int centy = offset + j*interval;
    Real dist = sqSum(centx-x, centy-y);
    if(dist < r2) return maxpix;
  }}
  return 0;
};
int main(int argc, char** argv )
{
  int rows = 256, cols = 256, idx = 0;
  myDMalloc(uint16_t, image, rows*cols);
  for(int i = 0; i < rows; i++){
    for(int j = 0; j < cols; j++){
      image[idx] = setHole(i,j);
      image[idx] = setHoleArray(i,j);
      idx++;
    }
  }
  writePng("mask.png", image, rows, cols, 16, 0);
}
