#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "memManager.h"
#include "imgio.h"
const uint16_t maxpix = 65535;
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
  Real r = sqrt(sqSum(x-128, y-256));
  if(r < 10) return maxpix;
  return 0;
};
int main(int argc, char** argv )
{
  const char* filename = "mask.png";
  int rows = 512, cols = 512, idx = 0;
  myDMalloc(uint16_t, image, rows*cols);
  for(int i = 0; i < rows; i++){
    for(int j = 0; j < cols; j++){
      image[idx] = setHole(i,j);
      idx++;
    }
  }
  writePng(filename, image, rows, cols, 16, 0);
}
