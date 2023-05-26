#ifndef __MNISTDATA_H__
#define __MNISTDATA_H__
#include "format.h"

class mnistData{
  void* dataset;
  int idat;
  protected:
  int rowraw;
  int colraw;
  Real* output;
	public:
  mnistData(const char* dir);
  Real* read();
  void setIndex(int idx) {idat = idx;}
  ~mnistData();
};

class cuMnist : public mnistData{
  void *cuOut, *cuRaw, *cuRefine;
  int rowrf;
  int colrf;
  int row;
  int col;
  int refinement;
  int nmerge;
  public:
  cuMnist(const char* dir, int nm, int re, int r, int c);
  void cuRead(void*);
};
#endif
