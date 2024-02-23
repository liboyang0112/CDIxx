#include <stdio.h>
#include <vector>
#include <fstream>
#include <iostream>
#include "format.hpp"
#include "imgio.hpp"
#include "memManager.hpp"
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
  int start[] = {0,0};
  int end[2];
  Real startlambda = 50;
  Real endlambda = 400;
  int tolerance = 50;
  int row, column;
  unsigned char curveColor[3] = {0, 0, 255};

  struct imageFile fdata;
  void* pngfile = readpng(argv[1], &fdata);
  row = fdata.rows;
  column = fdata.cols;
  unsigned char* rowp = (unsigned char*)allocpngrow(pngfile);
  end[0] = row-1;
  end[1] = column-1;

  printf("image %d x %d\n", row, column);
  int nlambda = end[0]-start[0];
  Real *lambdas = (Real*) ccmemMngr.borrowCache(nlambda*sizeof(Real));
  Real *rate = (Real*) ccmemMngr.borrowCleanCache(nlambda*sizeof(Real));
  int *count = (int*) ccmemMngr.borrowCleanCache(nlambda*sizeof(int));

  for(int x = start[1]; x>end[1]; x--){
    readpngrow(pngfile, rowp);
    for(int y = start[0]; y < end[0] ; y++){
      if(x == start[1]) lambdas[y-start[0]] = startlambda + (endlambda-startlambda)/(end[0]-start[0])*(y-start[0]);
      unsigned char* p = rowp+3*y;
      if(onCurve(p, curveColor, tolerance)) {
        Real tmp = Real(start[1]-x)/(start[1]-end[1]);
        printf("find curve at (%d, %d) = [%d, %d, %d], rate= %f\n", x, y, *p, p[1], p[2], tmp);
        p[2] = 255;
        p[0] = p[1] = 0;
        rate[y-start[0]] += tmp;
        count[y-start[0]] += 1;
      }
      //else printf("find non-curve at (%d, %d)=[%d, %d, %d]\n", x, y, rowp[y][0], rowp[y][1], rowp[y][2]);
    }
  }
  ofstream outfile;
  outfile.open("ccd_response.txt", ios::out);
  for(int x = 0; x < nlambda; x++){
    if(count[x]) rate[x]/=count[x];
    outfile << lambdas[x] << " "<< rate[x] << endl;
  }
  outfile.close();
  return 0;
}
