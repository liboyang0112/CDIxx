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
