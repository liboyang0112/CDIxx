#ifndef __CUPLOTTER_H__
#define __CUPLOTTER_H__

#include "format.hpp"
enum mode {MOD2,MOD, REAL, IMAG, PHASE, PHASERAD};

class cuPlotter
{
  int rows;
  int cols;
  pixeltype *cuCache_data = 0; //cv format
  void *cv_data = 0; //cv_data = cv_cache->data
  void *cv_cache = 0;
  void *cv_complex_data = 0; //cv_data = cv_cache->data
                               //
  void *cv_float_data = 0; //cv_data = cv_cache->data
  Real *cuCache_float_data = 0; //cv format

  
  void *videoWriterVec[100];
  int nvid = 0;
  public:
  int toVideo = -1;
  int showVid = 0;
  cuPlotter():cuCache_data(0),cv_data(0),cv_cache(0),cv_complex_data(0){};
  void init(int rows_, int cols_);
  void* getCache(){return cv_cache;}
  int initVideo(const char* filename, int fps = 24);
  void saveVideo(int handle = 0);
  void freeCuda();
  void* cvtTurbo(void* cache = 0);
  void saveFloat(void* cudaData, const char* label= "default");  //call saveData
  void saveComplex(void* cudaData, const char* label= "default");  //call saveData
  void saveFloatData(void* cudaData);
  void saveComplexData(void* cudaData);
  void plotComplex(void* cudaData, const mode m=MOD, bool isFrequency=0, Real decay=1, const char* label= "default",bool islog = 0, bool isFlip = 0, bool isColor = 0, const char* caption = 0);  //call processData
  void plotFloat(void* cudaData, const mode m=MOD, bool isFrequency=0, Real decay=1, const char* label= "default",bool islog = 0, bool isFlip = 0, bool isColor = 0, const char* caption = 0);
  void* processFloatData(void* cudaData, const mode m=MOD, bool isFrequency=0, Real decay = 1, bool islog = 0, bool isFlip = 0); //calculate using cuCache_data and copy data to cv_data
  void* processComplexData(void* cudaData, const mode m=MOD, bool isFrequency=0, Real decay = 1, bool islog = 0, bool isFlip = 0); //calculate using cuCache_data and copy data to cv_data
  void plot(const char* label, bool islog = 0, const char* caption = 0);
  void* cvt8bit(void* cache = 0);
  ~cuPlotter();
};
extern cuPlotter plt;
#endif
