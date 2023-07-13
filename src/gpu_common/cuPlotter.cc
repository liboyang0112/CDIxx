#include "cuPlotter.h"
#include "memManager.h"
#include "opencv2/opencv.hpp"
#include "opencv2/phase_unwrapping/histogramphaseunwrapping.hpp"
#include <map>
#include <string>
#define videoWriters(x) ((VideoWriter*)videoWriterVec[x])
using namespace cv;
int cuPlotter::initVideo(const char* filename){
  int handle = nvid;
  if(handle == 100){
    for(int i = 0; i < 100; i++){
      if(videoWriters(i) == 0){
        handle = i;
      }
    }
  }else
    nvid++;
  if(handle==100) {
    printf("You created too many videos (100), please release some before create new ones");
    exit(0);
  }
  videoWriterVec[handle] = ccmemMngr.borrowCache(sizeof(VideoWriter));
  new(videoWriters(handle))VideoWriter(filename, 0x7634706d, 24, Size(rows,cols), true);
  printf("create movie %s\n", filename);
  toVideo = handle;
  return handle;
}
void cuPlotter::saveVideo(int handle){
  videoWriters(handle)->release();
  free(videoWriters(handle));
  videoWriterVec[handle] = 0;
}
void cuPlotter::init(int rows_, int cols_){
  if(rows==rows_ && cols==cols_) return;
  if(cv_cache) delete (Mat*)cv_cache;
  if(cv_float_cache) delete (Mat*)cv_float_cache;
  if(cv_complex_cache) delete (Mat*)cv_complex_cache;
  printf("init plot %d, %d\n",rows_,cols_);
  rows=rows_;
  cols=cols_;
  Mat *tmp = new Mat(rows_, cols_, CV_16UC1, Scalar(0));
  cv_cache = tmp;
  cv_data = tmp->data;
  freeCuda();
}
void cuPlotter::plotComplex(void* cudaData, mode m, bool isFrequency, Real decay, const char* label,bool islog, bool isFlip){
  cuPlotter::processComplexData(cudaData,m,isFrequency,decay,islog,isFlip);
  plot(label, islog);
}
void cuPlotter::plotFloat(void* cudaData, mode m, bool isFrequency, Real decay, const char* label,bool islog, bool isFlip, bool isColor){
  cuPlotter::processFloatData(cudaData,m,isFrequency,decay,islog,isFlip);
  plot(label, islog || isColor);
}
void cuPlotter::saveComplex(void* cudaData, const char* label){
  if(!cv_complex_cache){
    Mat *tmpcomplex = new Mat(rows, cols, CV_32FC2, Scalar(0));
    cv_complex_cache = tmpcomplex;
    cv_complex_data = tmpcomplex->data;
  }
  saveComplexData(cudaData);
  FileStorage fs(label,FileStorage::WRITE);
  fs<<"data"<<*((Mat*)cv_complex_cache);
  fs.release();
}
void cuPlotter::saveFloat(void* cudaData, const char* label){
  if(!cv_float_cache){
    Mat *tmpfloat = new Mat(rows, cols, CV_32FC1, Scalar(0));
    cv_float_cache = tmpfloat;
    cv_float_data = tmpfloat->data;
  }
  saveFloatData(cudaData);
  imwrite(std::string(label)+".tiff",*((Mat*)cv_float_cache));
}
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
void cuPlotter::plot(const char* label, bool iscolor){
  std::string fname = label;
  if(fname.find(".")==std::string::npos) fname+=".png";
  if(iscolor){
    Mat* tmp = (Mat*)cv_cache;
	  Mat dst8 = Mat::zeros(tmp->size(), CV_8U);
	  //normalize(*tmp, *tmp, 0, 255, NORM_MINMAX);
    *tmp *= 1./256;
	  convertScaleAbs(*tmp, dst8);
	  applyColorMap(dst8, dst8, COLORMAP_TURBO);
    if(toVideo>=0) {
      ((VideoWriter*)videoWriters(toVideo))->write(dst8);
      return;
    }
    else imwrite(fname,dst8);
  }else
    imwrite(fname, *(Mat*)cv_cache);
  printf("written to file %s\n", fname.c_str());
}
cuPlotter::~cuPlotter(){
  freeCuda();
  delete (Mat*)cv_cache;
}

