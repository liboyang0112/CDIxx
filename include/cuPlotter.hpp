#ifndef __CUPLOTTER_H__
#define __CUPLOTTER_H__

#include "format.hpp"
#include <cstdint>
enum mode {MOD2,MOD, REAL, IMAG, PHASE, PHASERAD};

class cuPlotter
{
  int rows;
  int cols;
  void *cuCache_data = 0; //cv format
  void *cuRGB_data = 0; //GPU RGB buffer for JPEG
  void *cv_data = 0; //cv_data = cv_cache->data
  void *cv_complex_data = 0; //cv_data = cv_cache->data
                               //
  void *cv_float_data = 0; //cv_data = cv_cache->data

  
  const char* prefix = 0;
  int nvid = 0;
  unsigned long int vid_thread = 0;
  public:
  uint8_t *cv_cache = 0;
  void *videoWriterVec[100];
  void* videoPreview[100];
  bool isOnline[100];
  int toVideo = -1;
  int writeVideo = -1;
  void* preview_dPtr = 0; // GPU pointer to preview shared buffer (OPAQUE_FD)
  cuPlotter():cuCache_data(0),cv_data(0),cv_complex_data(0),cv_cache(0){
    for(int i = 0; i < 100; i++){
      videoWriterVec[i] = 0;
      videoPreview[i] = 0;
      isOnline[i] = false;
    }
  }
  void init(int rows_, int cols_, const char* prefix_ = "./");
  void* getCache(){return cv_cache;}
  int initVideo(const char* filename, int fps = 24, bool preview = 0, bool online = 0);
  void videoFrame(Real* cudaData);
  void saveVideo(int handle = 0);
  void freeMem();
  void* cvtTurbo(void* cache = 0);
  void saveFloat(Real* cudaData, const char* label= "default");  //call saveData
  void saveComplex(complexFormat* cudaData, const char* label= "default");  //call saveData
  void plotComplex(complexFormat* cudaData, const mode m=MOD, bool isFrequency=0, Real decay=1, const char* label= "default",bool islog = 0, bool isFlip = 0, bool isColor = 0, const char* caption = 0);  //call processData
  void plotComplexColor(complexFormat* cudaData, bool isFrequency=0, Real decay=1, const char* label= "default",bool islog = 0, bool isFlip = 0, bool phaseOnly = 0);  //call processData
  void plotFloat(Real* cudaData, const mode m=MOD, bool isFrequency=0, Real decay=1, const char* label= "default",bool islog = 0, bool isFlip = 0, bool isColor = 0, const char* caption = 0);
  void* processFloatData(Real* cudaData, const mode m=MOD, bool isFrequency=0, Real decay = 1, bool islog = 0, bool isFlip = 0); //calculate using cuCache_data and copy data to cv_data
  void* processComplexData(complexFormat* cudaData, const mode m=MOD, bool isFrequency=0, Real decay = 1, bool islog = 0, bool isFlip = 0); //calculate using cuCache_data and copy data to cv_data
  void* processComplexColor(complexFormat* cudaData, bool isFrequency=0, Real decay = 1, bool islog = 0, bool isFlip = 0, bool phaseOnly = 0); //calculate using cuCache_data and copy data to cv_data
  void plot(const char* label, bool islog = 0, const char* caption = 0);
  void* cvt8bit(void* cache = 0);
  ~cuPlotter();
};
extern cuPlotter plt;
void processFloat(Real* cudaData, pixeltype* cache, mode m, bool isFrequency=0, Real decay = 1, bool islog = 0, bool isFlip = 0);
void processComplex(complexFormat* cudaData, pixeltype* cache, mode m, bool isFrequency=0, Real decay = 1, bool islog = 0, bool isFlip = 0);
void process_rgb(complexFormat* cudaData, col_rgb* cache, bool isFrequency=0, Real decay = 1, bool islog = 0, bool isFlip = 0, bool phaseOnly = 0);
#endif
