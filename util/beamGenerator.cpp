#include <stdio.h>
#include <opencv2/imgcodecs.hpp>
#include "format.h"
#include <random>
using namespace cv;
using namespace std;
//auto format_cv = CV_16UC(1);
auto format_cv = CV_64FC(1);
//auto format_cv = CV_8UC(1);
Real gaussian(Real x, Real y, Real sigma){
  Real r2 = pow(x,2) + pow(y,2);
  return exp(-r2/pow(sigma,2));
}
int main(int argc, char** argv )
{
  int row = 320;
  int column = 320;
  //Mat image (row, column, float_cv_format(1), Scalar::all(0));
  Mat image (row, column, CV_8UC1, Scalar::all(0));
  int rad = row/12;
  int radcut = row/12;
  int spotr = 4;
  int spotsx[] = {row/2, 3};
  int spotsy[] = {3, column/2};
  int spotblock[] = {row/2+5, column/2+5};
  auto seed = (unsigned)time(NULL);
  srand(seed);
  //Real* rowo;
  char* rowo;
  Real tot = 0;
  Real totx = 0;
  Real toty = 0;
  Real sumx = 0;
  Real sumy = 0;
  Real max = 0;
  for(int x = 0; x < row ; x++){
    //rowo =   image.ptr<Real>(x);
    rowo = image.ptr<char>(x);
    for(int y = 0; y<column; y++){
#if 0
      Real r = hypot(x-row/2,y-row/2);
      for(int i = 0 ; i < 2; i++){
        Real r = hypot(x-spotsx[i],y-spotsy[i]);
        if(r < spotr) rowo[y] = 1;
      }
      //if(abs(x-row/2)<=1) continue;
      if(r<radcut) {
        Real randm = static_cast<Real>(rand())/Real(RAND_MAX);
        //if(randm < 0.3) continue;
        rowo[y] = gaussian(x-row/2,y-row/2,rad);
      }
      r = hypot(x-spotblock[0],y-spotblock[1]);
      if(r<spotr) rowo[y] = 0;
      //if(r<60) rowo[y] *= 0.4;
#endif
      if(x <= 7 || y <= 7) rowo[y] = 255;
      else rowo[y] = 0;
      if(y > 7 && (y%30 < 7)) rowo[y] = 0; 
    }
  }

  imwrite("image.tiff", image);
  return 0;
}
