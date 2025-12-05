#ifndef __MNISTDATA_H__
#define __MNISTDATA_H__
#include "format.hpp"

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
  void *cuRaw, *cuRefine;
  int rowrf;
  int colrf;
  int refinement;
  int row;
  int col;
  int nmerge;
  void* handleraw;
  void* handle;
  complexFormat* cache, *cacheraw;
  public:
  cuMnist(const char* dir, int nm, int re, int r, int c);
  void cuRead(void*);
};
#endif
