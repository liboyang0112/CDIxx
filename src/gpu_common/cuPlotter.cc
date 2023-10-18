#include "cuPlotter.h"
#include "memManager.h"
#include "videoWriter.h"
#include "imageFile.h"
#include "writePng.h"
#include <map>
#include <string>
#include <png.h>
#include <stdio.h>
const float TurboData[3][8][3] = {  //color, sector, fit parameterx : a+b*x+c*x*x
  {
    {48.628736, 1.26493952, 0.0179480064},
    {12.9012, 3.17599, -0.0432176},
    {550.736, -13.3905, 0.0850816},
    {-377.035, 4.95565, -0.00546875},
    {-461.663, 6.91765, -0.0158418},
    {-904.194, 12.7805, -0.0352175},
    {-397.815, 7.43316, -0.0210902},
    {-513.745, 8.50106, -0.0235486}
  },{
     {18.3699, 2.98435, -0.00590619},
     {25.5339, 2.60115, -0.000838337},
     {-65.9885, 5.48243, -0.023616},
     {-90.9164, 5.82642, -0.024485},
     {-47.1592, 5.39126, -0.0237502},
     {-65.915, 5.26759, -0.0222375},
     {1648.81, -12.6094, 0.0243797},
     {1160.21, -8.10854, 0.014018}
  },{
     {59.4351, 7.57781, -0.0720408},
     {29.4288, 9.18296, -0.0932682},
     {391.685, -2.3166, -0.0016649},
     {770.071, -8.98098, 0.0266769},
     {606.475, -7.75046, 0.0270971},
     {-638.138, 8.68843, -0.0270811},
     {1017.44, -8.76258, 0.0189386},
     {625.18, -5.14916, 0.0106199}
  }
};

void getTurboColor(Real x, int bit_depth, char* store){
  //convert to 8 bit;
  x = x / (1<<(bit_depth-8));
  if(x > 255) x = 255;
  int sec = x/32; // deside sector;
  for(int i = 0 ; i < 3; i++){
    int val = TurboData[i][sec][0] + TurboData[i][sec][1]*x + TurboData[i][sec][2]*x*x;
    if(val < 0) val = 0;
    if(val > 255) val = 255;
    store[i] = val;
  }
}

int cuPlotter::initVideo(const char* filename, const char* codec, int fps){
  int handle = nvid;
  if(handle == 100){
    for(int i = 0; i < 100; i++){
      if(videoWriterVec[handle] == 0){
        handle = i;
      }
    }
  }else
    nvid++;
  if(handle==100) {
    printf("You created too many videos (100), please release some before create new ones");
    exit(0);
  }
  videoWriterVec[handle] = createVideo(filename, rows, cols, fps);
  toVideo = handle;
  return handle;
}
void cuPlotter::saveVideo(int handle){
  ::saveVideo(videoWriterVec[handle]);
}
void cuPlotter::init(int rows_, int cols_){
  if(rows==rows_ && cols==cols_) return;
  if(cv_cache) {
    ccmemMngr.returnCache(cv_cache);
    ccmemMngr.returnCache(cv_data);
    cv_cache = 0;
    cv_data = 0;
  }
  if(cv_float_data) {
    ccmemMngr.returnCache(cv_float_data);
    cv_float_data = 0;
  }
  if(cv_complex_data){
    ccmemMngr.returnCache(cv_complex_data);
    cv_complex_data = 0;
  }
  printf("init plot %d, %d\n",rows_,cols_);
  rows=rows_;
  cols=cols_;
  cv_cache = ccmemMngr.borrowCache(rows*cols*3);
  cv_data = ccmemMngr.borrowCache(rows*cols*2);
  freeCuda();
}
void cuPlotter::plotComplex(void* cudaData, mode m, bool isFrequency, Real decay, const char* label,bool islog, bool isFlip, bool isColor){
  cuPlotter::processComplexData(cudaData,m,isFrequency,decay,islog,isFlip);
  plot(label, isColor);
}
void cuPlotter::plotFloat(void* cudaData, mode m, bool isFrequency, Real decay, const char* label,bool islog, bool isFlip, bool isColor){
  cuPlotter::processFloatData(cudaData,m,isFrequency,decay,islog,isFlip);
  plot(label, isColor);
}
void cuPlotter::saveComplex(void* cudaData, const char* label){
  FILE* fout = fopen((std::string(label)+".bin").c_str(),"w");
  if(!cv_complex_data){
    cv_complex_data = ccmemMngr.borrowCache(rows*cols*sizeof(Real)*2);
  }
  imageFile ff;
  ff.type = COMPLEXIDX;
  ff.rows = rows;
  ff.cols = cols;
  saveComplexData(cudaData);
  fwrite(&ff, sizeof(ff), 1, fout);
  fwrite(cv_float_data, rows*cols*sizeof(Real)*2, 1, fout);
  fclose(fout);
}
void cuPlotter::saveFloat(void* cudaData, const char* label){
  FILE* fout = fopen((std::string(label)+".bin").c_str(),"w");
  if(!cv_float_data){
    cv_float_data = ccmemMngr.borrowCache(rows*cols*sizeof(Real));
  }
  imageFile ff;
  ff.type = REALIDX;
  ff.rows = rows;
  ff.cols = cols;
  saveFloatData(cudaData);
  fwrite(&ff, sizeof(ff), 1, fout);
  fwrite(cv_float_data, rows*cols*sizeof(Real), 1, fout);
  fclose(fout);
}
/*
#include "opencv2/phase_unwrapping/histogramphaseunwrapping.hpp"
using namespace cv;
void cuPlotter::plotPhase(void* cudaData, mode m, bool isFrequency, Real decay, const char* label,bool islog, bool isFlip){
  if(!cv_float_cache){
    Mat *tmpfloat = new Mat(rows, cols, CV_32FC1, Scalar(0));
    cv_float_cache = tmpfloat;
    cv_float_data = tmpfloat->data;
  }
  cuPlotter::processPhaseData(cudaData,m,isFrequency,decay, isFlip);
  cv::phase_unwrapping::HistogramPhaseUnwrapping::Params pars;
  pars.height = cols;
  pars.width = rows;
  auto uwrap = phase_unwrapping::HistogramPhaseUnwrapping::create(pars);
  uwrap->unwrapPhaseMap(*(Mat*)cv_float_cache, *(Mat*)cv_float_cache);
  for(int i = 0; i < rows*cols; i++)
    ((pixeltype*)cv_data)[i] = std::min(int((((Real*)cv_float_data)[i]+Real(M_PI))*rcolor/phaseMax),rcolor-1);
  plot(label, islog);
}
*/
void cuPlotter::plot(const char* label, bool iscolor){
  std::string fname = label;
  if(fname.find(".")==std::string::npos) fname+=".png";
  if(iscolor){
    for(int i = 0 ; i < rows*cols; i++){
      getTurboColor(((pixeltype*)cv_data)[i], Bits, (char*)cv_cache+i*3);
    }
    if(toVideo>=0) {
      //char color[3] = {0,0,-1};
      //put_formula(label, 0,0,cols, cv_cache, 1, color);
      flushVideo(videoWriterVec[toVideo], cv_cache);
      return;
    }
    else
      writePng(fname.c_str(), cv_cache, rows, cols, 8, 1);
  }else
    writePng(fname.c_str(), cv_data, rows, cols, 16, 0);
  printf("written to file %s\n", fname.c_str());
}
cuPlotter::~cuPlotter(){
  freeCuda();
}

