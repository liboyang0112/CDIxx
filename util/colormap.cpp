#include <stdio.h>
#include <opencv2/imgcodecs.hpp>
#include "format.h"
#include <random>
using namespace cv;
using namespace std;
Real gaussian(Real x, Real y, Real sigma){
  Real r2 = pow(x,2) + pow(y,2);
  return exp(-r2/pow(sigma,2));
}
int main(int argc, char** argv )
{
  int row = 20;
  int column = 511;
  //Mat image (row, column, float_cv_format(1), Scalar::all(0));
  Mat image (row, column, CV_8UC3, Scalar::all(0));
  unsigned char* data = (unsigned char*)image.data;
  size_t index = 0;
  for(int x = 0; x < row ; x++){
    for(int y = 0; y<column; y++){
      uchar r,g,b;
      b = 255-y;
      if(y < 255) b = 255-y;
      else b = 0;
      g = 255-abs(y-255);
      if(y >= 255) r = y-255;
      else r = 0;
      data[index] = b;
      data[index+1] = g;
      data[index+2] = r;
      index+=3;
    }
  }
  imwrite("image.png", image);
  return 0;
}
