#include <stdio.h>
#include <opencv2/imgcodecs.hpp>
#include "format.h"
#include <random>
using namespace cv;
using namespace std;
int main(int argc, char** argv )
{
  int row = 320;
  int column = 320;
  Mat image (row, column, CV_8UC1, Scalar::all(0));
  for(int x = 0; x < row ; x++){
    char* rowo = image.ptr<char>(x);
    for(int y = 0; y<column; y++){
      if(x <= 4 || y <= 4) rowo[y] = -1;
      else rowo[y] = 0;
      if(y > 2 && (y%30 < 15)) rowo[y] = 0; 
    }
  }
  imwrite("image.png", image);
  return 0;
}
