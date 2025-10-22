#include <fstream>
#include <iostream>
#include "fmt/core.h"
#include "format.hpp"
#include "imgio.hpp"
#include "memManager.hpp"
#include <png.h>
struct pngdata{
  png_structp png_ptr;
  png_infop info_ptr;
};
using namespace std;

bool onCurve(unsigned char *pixColor, unsigned char *curveColor, int tolerance){
  bool retval = 1;
  for ( int i = 0; i < 3; i++){
    if(abs(pixColor[i]-curveColor[i]) > tolerance) {
      retval = 0;
      break;
    }
  }
  return retval;
}

int main(int argc, char** argv )
{
  if(argc < 2){
    fmt::print("Usage: readCCDResponse response.png");
    return 0;
  }
  int start[] = {0,0};
  int end[2];
  Real startlambda = 50;
  Real endlambda = 400;
  int tolerance = 50;
  float dfloor = 0.00631313;
  int row, column;
  unsigned char curveColor[3] = {0, 0, 255};

  struct imageFile fdata;
  void* pngfile = readpng(argv[1], &fdata);
  row = fdata.rows;
  column = fdata.cols;
  unsigned char* rowp = (unsigned char*)allocpngrow(pngfile);
  end[0] = row-1;
  end[1] = column-1;

  fmt::println("image {} x {}", row, column);
  int nlambda = end[0]-start[0];
  Real *lambdas = (Real*) ccmemMngr.borrowCache(nlambda*sizeof(Real));
  Real *rate = (Real*) ccmemMngr.borrowCleanCache(nlambda*sizeof(Real));
  int *count = (int*) ccmemMngr.borrowCleanCache(nlambda*sizeof(int));

  for(int x = end[1]; x>start[1]; x--){
    png_read_row(((struct pngdata*)pngfile)->png_ptr, rowp, NULL);
    for(int y = start[0]; y < end[0] ; y++){
      if(x == end[1]) lambdas[y-start[0]] = startlambda + (endlambda-startlambda)/(end[0]-start[0])*(y-start[0]);
      unsigned char* p = rowp+3*y;
      if(onCurve(p, curveColor, tolerance)) {
        Real tmp = Real(start[1]-x)/(start[1]-end[1]);
        fmt::println("find curve at ({}, {}) = [{}, {}, {}], rate= {:f}", x, y, static_cast<signed char>(*p), static_cast<signed char>(p[1]), static_cast<signed char>(p[2]), tmp);
        p[2] = 255;
        p[0] = p[1] = 0;
        rate[y-start[0]] += tmp;
        count[y-start[0]] += 1;
      }
    }
  }
  ofstream outfile;
  outfile.open("ccd_response.txt", ios::out);
  for(int x = 0; x < nlambda; x++){
    if(count[x]) rate[x]/=count[x];
    rate[x] -= dfloor;
    if(rate[x] < 0) rate[x] = 0;
    outfile << lambdas[x] << " "<< rate[x] << endl;
  }
  outfile.close();
  return 0;
}
